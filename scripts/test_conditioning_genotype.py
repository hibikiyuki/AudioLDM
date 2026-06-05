#!/usr/bin/env python3
"""
条件付けベクトル（CLAP text embedding）を遺伝子とするIECの検証スクリプト

仮説: CLAP text embedding 空間での SLERP 補間は、
      音響的に「2つのプロンプトの中間」にある音を生成できる。

検証実験:
  Exp 1 — 補間の連続性
    slerp(c_A, c_B, α) で生成した音声が α に応じて単調に変化するか。
    CLAPスコア sim(出力, text_A) が α=0→1 で単調減少、
             sim(出力, text_B) が α=0→1 で単調増加するか。

  Exp 2 — 突然変異の効果
    c_A に N(0, σ²) を加えたとき、σ が小さいほど元の音に近いか。
    CLAPスコアと σ の間に負の相関があるか。

  Exp 3 — 多様性の相関
    conditioning 空間でのペアワイズ距離が、
    音声 CLAP 埋め込み空間での距離と相関するか。

実行方法:
  cd /workspaces/AudioLDM
  python scripts/test_conditioning_genotype.py

出力:
  scripts/outputs/conditioning_test/
    exp1_interp_alpha{α:.2f}.wav   — 補間音声
    exp2_mutate_sigma{σ:.3f}.wav   — 突然変異音声
    exp3_diverse_{i}.wav           — 多様性テスト個体
    results.txt                    — 数値結果サマリ
"""

import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audioldm.pipeline import build_model, duration_to_latent_t_size
from audioldm.latent_diffusion.ddim import DDIMSampler


# ──────────────────────────────────────────────────────────────
# 遺伝子型: conditioning vector
# ──────────────────────────────────────────────────────────────

@dataclass
class ConditioningGenotype:
    """CLAP text embedding ベクトルを遺伝子として保持する個体。

    embedding: shape (1, 1, 512) — sample_log の cond に直接渡せる形状。
    """
    embedding: torch.Tensor          # (1, 1, 512)
    source_prompt: str = ""          # デバッグ用: 元のプロンプト
    metadata: dict = field(default_factory=dict)

    def clone(self) -> "ConditioningGenotype":
        return ConditioningGenotype(
            embedding=self.embedding.clone(),
            source_prompt=self.source_prompt,
            metadata=self.metadata.copy(),
        )


def slerp_conditioning(
    c0: torch.Tensor, c1: torch.Tensor, t: float, threshold: float = 0.9995
) -> torch.Tensor:
    """2つの conditioning vector を球面線形補間する。

    c0, c1: (1, 1, 512)
    Returns: (1, 1, 512)
    """
    shape = c0.shape
    v0 = c0.reshape(-1).float()
    v1 = c1.reshape(-1).float()

    v0_n = v0 / (torch.norm(v0) + 1e-8)
    v1_n = v1 / (torch.norm(v1) + 1e-8)
    dot = torch.clamp(torch.dot(v0_n, v1_n), -1.0, 1.0)

    if abs(dot.item()) > threshold:
        result = (1.0 - t) * v0 + t * v1
    else:
        theta = torch.acos(dot)
        sin_theta = torch.sin(theta)
        s0 = torch.sin((1.0 - t) * theta) / sin_theta
        s1 = torch.sin(t * theta) / sin_theta
        result = s0 * v0 + s1 * v1

    return result.reshape(shape).to(c0.dtype)


def mutate_conditioning(
    genotype: ConditioningGenotype, sigma: float
) -> ConditioningGenotype:
    """conditioning vector にガウシアンノイズを加えて突然変異させる。"""
    mutant = genotype.clone()
    mutant.embedding = mutant.embedding + torch.randn_like(mutant.embedding) * sigma
    mutant.metadata["operation"] = "mutate_gaussian"
    mutant.metadata["sigma"] = sigma
    return mutant


# ──────────────────────────────────────────────────────────────
# 音声生成 (conditioning vector → waveform)
# ──────────────────────────────────────────────────────────────

class ConditioningGenotypeRunner:
    """条件付けベクトル遺伝子を使った音声生成・評価のヘルパー。"""

    def __init__(
        self,
        model_name: str = "audioldm-m-full",
        ckpt_path: Optional[str] = None,
        duration: float = 5.0,
        ddim_steps: int = 200,         # 検証用に短縮
        guidance_scale: float = 2.5,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.ddim_steps = ddim_steps
        self.guidance_scale = guidance_scale

        print(f"モデルをロード中: {model_name} (device={self.device})")
        self.model = build_model(ckpt_path=ckpt_path, model_name=model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.model.latent_t_size = duration_to_latent_t_size(duration)
        self.model.cond_stage_model.embed_mode = "text"

        self.latent_shape = (
            self.model.channels,
            self.model.latent_t_size,
            self.model.latent_f_size,
        )
        print(f"潜在空間形状: {self.latent_shape}")

    # ── エンコード ───────────────────────────────────────────

    def encode_text(self, prompt: str) -> torch.Tensor:
        """テキストを CLAP text embedding に変換する。

        Returns: (1, 1, 512)
        """
        cond = self.model.cond_stage_model
        orig_mode = cond.embed_mode
        orig_prob = cond.unconditional_prob
        cond.embed_mode = "text"
        cond.unconditional_prob = 0.0
        try:
            with torch.no_grad():
                emb = cond([prompt, prompt])  # (2, 1, 512)
                return emb[0:1].clone()       # (1, 1, 512)
        finally:
            cond.embed_mode = orig_mode
            cond.unconditional_prob = orig_prob

    def encode_audio_embedding(self, waveform: np.ndarray) -> torch.Tensor:
        """音声波形を CLAP audio embedding に変換する。

        waveform: numpy array — vocoder 出力の (1, 1, T) または (1, T)、16kHz
        Returns: (1, 512)
        """
        cond = self.model.cond_stage_model
        orig_mode = cond.embed_mode
        orig_prob = cond.unconditional_prob
        cond.embed_mode = "audio"
        cond.unconditional_prob = 0.0
        try:
            with torch.no_grad():
                wf_tensor = torch.tensor(waveform, dtype=torch.float32).to(self.device)
                # vocoder は (bs, 1, t) を返すので (bs, t) に次元を合わせる
                if wf_tensor.dim() == 3:
                    wf_tensor = wf_tensor.squeeze(1)  # (1, t)
                # CLAP 内部の mel 計算に合わせて正規化
                wf_tensor = wf_tensor / (wf_tensor.abs().max() + 1e-8) * 0.5
                emb = cond(wf_tensor).squeeze(1)  # (1, 512)
                return emb.clone()
        finally:
            cond.embed_mode = orig_mode
            cond.unconditional_prob = orig_prob

    # ── 固定ノイズ管理 ────────────────────────────────────────

    def make_fixed_noise(self, seed: int = 123) -> torch.Tensor:
        """実験全体で共有する固定 x_T を生成する (conditioning 効果を分離するため)。

        Returns: (1, C, T, F)
        """
        gen = torch.Generator(device=self.device).manual_seed(seed)
        return torch.randn((1,) + self.latent_shape, device=self.device, generator=gen)

    # ── 音声生成 ─────────────────────────────────────────────

    def generate(
        self,
        conditioning: torch.Tensor,    # (1, 1, 512) or (n, 1, 512)
        x_T: torch.Tensor,             # (1, C, T, F) — 全個体で共有
    ) -> List[np.ndarray]:
        """conditioning vector と固定 x_T から音声を生成する。

        conditioning が (1,1,512) なら 1 サンプル、(n,1,512) なら n サンプルを返す。
        """
        n = conditioning.shape[0]
        x_T_batch = x_T.expand(n, -1, -1, -1).clone()

        uc = (
            self.model.cond_stage_model.get_unconditional_condition(n)
            if self.guidance_scale != 1.0 else None
        )

        with self.model.ema_scope("ConditioningGenotype Generation"):
            with torch.no_grad():
                samples, _ = self.model.sample_log(
                    cond=conditioning,
                    batch_size=n,
                    ddim=True,
                    ddim_steps=self.ddim_steps,
                    eta=0.0,
                    unconditional_guidance_scale=self.guidance_scale,
                    unconditional_conditioning=uc,
                    x_T=x_T_batch,
                )
                if torch.max(torch.abs(samples)) > 1e2:
                    samples = torch.clip(samples, min=-10, max=10)
                mel = self.model.decode_first_stage(samples)
                wf_batch = self.model.mel_spectrogram_to_waveform(mel)

        return [wf_batch[i : i + 1] for i in range(n)]

    # ── CLAP スコア計算 ─────────────────────────────────────

    def clap_similarity_text(
        self, audio_emb: torch.Tensor, text: str
    ) -> float:
        """音声 CLAP embedding と テキスト CLAP embedding のコサイン類似度。

        audio_emb: (1, 512)
        Returns: float in [-1, 1]
        """
        text_emb = self.encode_text(text).squeeze(1)  # (1, 512)
        sim = F.cosine_similarity(audio_emb, text_emb, dim=1).item()
        return sim

    def clap_similarity_audio(
        self, emb_a: torch.Tensor, emb_b: torch.Tensor
    ) -> float:
        """2つの音声 CLAP embedding 間のコサイン類似度。"""
        return F.cosine_similarity(emb_a, emb_b, dim=1).item()


# ──────────────────────────────────────────────────────────────
# 検証実験
# ──────────────────────────────────────────────────────────────

def save_wav(waveform: np.ndarray, path: str, sr: int = 16000):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    wav = waveform.squeeze()
    sf.write(path, wav, sr)


def pairwise_cosine_mean(embeddings: List[torch.Tensor]) -> float:
    """埋め込みリストのペアワイズ cosine 距離（1 - similarity）の平均。"""
    n = len(embeddings)
    if n < 2:
        return 0.0
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = F.cosine_similarity(embeddings[i], embeddings[j], dim=1).item()
            dists.append(1.0 - sim)
    return float(np.mean(dists))


def run_exp1_interpolation(runner: ConditioningGenotypeRunner, out_dir: str, lines: List[str]):
    """Exp 1: SLERP 補間の連続性テスト。"""
    print("\n" + "=" * 60)
    print("Exp 1: 補間の連続性テスト")
    print("=" * 60)

    prompt_a = "calm acoustic piano music"
    prompt_b = "energetic electronic dance music with heavy bass"
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]

    c_a = runner.encode_text(prompt_a)
    c_b = runner.encode_text(prompt_b)
    x_T = runner.make_fixed_noise(seed=42)

    print(f"  Prompt A: {prompt_a}")
    print(f"  Prompt B: {prompt_b}")
    print(f"  α: {alphas}")

    lines.append("\n=== Exp 1: 補間の連続性 ===")
    lines.append(f"Prompt A: {prompt_a}")
    lines.append(f"Prompt B: {prompt_b}")
    lines.append(f"{'α':>5}  {'sim→A':>8}  {'sim→B':>8}  {'quality':>10}")
    lines.append("-" * 40)

    for alpha in alphas:
        c_interp = slerp_conditioning(c_a, c_b, alpha)
        waveforms = runner.generate(c_interp, x_T)
        wf = waveforms[0]

        has_nan = bool(np.isnan(wf).any())
        max_amp = float(np.max(np.abs(wf)))
        quality = "OK" if not has_nan and max_amp > 1e-6 else "DEGRADED"

        audio_emb = runner.encode_audio_embedding(wf)
        sim_a = runner.clap_similarity_text(audio_emb, prompt_a)
        sim_b = runner.clap_similarity_text(audio_emb, prompt_b)

        wav_path = os.path.join(out_dir, f"exp1_interp_alpha{alpha:.2f}.wav")
        save_wav(wf, wav_path)

        line = f"  α={alpha:.2f}  sim→A={sim_a:+.4f}  sim→B={sim_b:+.4f}  [{quality}]  → {wav_path}"
        print(line)
        lines.append(f"{alpha:>5.2f}  {sim_a:>+8.4f}  {sim_b:>+8.4f}  {quality:>10}")

    # 単調性チェック
    # 理想: sim_a は α が増えるにつれ減少、sim_b は増加


def run_exp2_mutation(runner: ConditioningGenotypeRunner, out_dir: str, lines: List[str]):
    """Exp 2: 突然変異の効果テスト。"""
    print("\n" + "=" * 60)
    print("Exp 2: 突然変異の効果テスト")
    print("=" * 60)

    prompt = "calm acoustic piano music"
    sigmas = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5]
    x_T = runner.make_fixed_noise(seed=42)

    c_orig = runner.encode_text(prompt)
    genotype = ConditioningGenotype(embedding=c_orig, source_prompt=prompt)

    print(f"  Base prompt: {prompt}")
    print(f"  σ values: {sigmas}")

    lines.append("\n=== Exp 2: 突然変異の効果 ===")
    lines.append(f"Base prompt: {prompt}")
    lines.append(f"{'σ':>6}  {'sim→base':>10}  {'cond_dist':>10}  {'quality':>10}")
    lines.append("-" * 44)

    # ベース音声の CLAP 埋め込み
    wf_base = runner.generate(c_orig, x_T)[0]
    base_audio_emb = runner.encode_audio_embedding(wf_base)

    for sigma in sigmas:
        if sigma == 0.0:
            mutant = genotype.clone()
        else:
            mutant = mutate_conditioning(genotype, sigma=sigma)

        wf = runner.generate(mutant.embedding, x_T)[0]
        has_nan = bool(np.isnan(wf).any())
        max_amp = float(np.max(np.abs(wf)))
        quality = "OK" if not has_nan and max_amp > 1e-6 else "DEGRADED"

        audio_emb = runner.encode_audio_embedding(wf)
        sim_to_base = runner.clap_similarity_audio(audio_emb, base_audio_emb)

        # conditioning 空間での距離
        c_dist = 1.0 - F.cosine_similarity(
            c_orig.reshape(1, -1), mutant.embedding.reshape(1, -1), dim=1
        ).item()

        wav_path = os.path.join(out_dir, f"exp2_mutate_sigma{sigma:.3f}.wav")
        save_wav(wf, wav_path)

        line = f"  σ={sigma:.3f}  sim_audio={sim_to_base:+.4f}  cond_dist={c_dist:.4f}  [{quality}]"
        print(line)
        lines.append(f"{sigma:>6.3f}  {sim_to_base:>+10.4f}  {c_dist:>10.4f}  {quality:>10}")


def run_exp3_diversity(runner: ConditioningGenotypeRunner, out_dir: str, lines: List[str]):
    """Exp 3: conditioning 空間の多様性と音響多様性の相関。"""
    print("\n" + "=" * 60)
    print("Exp 3: 多様性の相関テスト")
    print("=" * 60)

    prompt = "calm acoustic piano music"
    n_individuals = 6
    mutation_sigma = 0.1
    x_T = runner.make_fixed_noise(seed=42)

    c_orig = runner.encode_text(prompt)
    print(f"  Base prompt: {prompt}")
    print(f"  個体数: {n_individuals}, σ={mutation_sigma}")

    # 個体群を突然変異で生成
    genotypes: List[ConditioningGenotype] = []
    g0 = ConditioningGenotype(embedding=c_orig, source_prompt=prompt)
    genotypes.append(g0)
    for _ in range(n_individuals - 1):
        genotypes.append(mutate_conditioning(g0, sigma=mutation_sigma))

    # 音声生成と CLAP 埋め込み
    cond_batch = torch.cat([g.embedding for g in genotypes], dim=0)  # (n, 1, 512)
    waveforms = runner.generate(cond_batch, x_T)

    cond_embs: List[torch.Tensor] = [g.embedding.reshape(1, -1) for g in genotypes]  # (1, 512) each
    audio_embs: List[torch.Tensor] = []
    for i, wf in enumerate(waveforms):
        wav_path = os.path.join(out_dir, f"exp3_diverse_{i}.wav")
        save_wav(wf, wav_path)
        audio_embs.append(runner.encode_audio_embedding(wf))

    cond_diversity = pairwise_cosine_mean(cond_embs)
    audio_diversity = pairwise_cosine_mean(audio_embs)

    lines.append("\n=== Exp 3: 多様性の相関 ===")
    lines.append(f"Base prompt: {prompt}")
    lines.append(f"σ = {mutation_sigma}, n = {n_individuals}")
    lines.append(f"conditioning 空間の平均ペアワイズ cosine 距離: {cond_diversity:.4f}")
    lines.append(f"音声 CLAP 空間の平均ペアワイズ cosine 距離: {audio_diversity:.4f}")

    print(f"  conditioning 空間の多様性 (cosine dist): {cond_diversity:.4f}")
    print(f"  音声 CLAP 空間の多様性    (cosine dist): {audio_diversity:.4f}")

    # 仮説: cond_diversity と audio_diversity の比が 1 に近いほど、
    #       conditioning 操作が音響多様性をうまく制御できている
    ratio = audio_diversity / (cond_diversity + 1e-8)
    msg = f"  多様性伝達率 (audio/cond): {ratio:.4f}"
    print(msg)
    lines.append(f"多様性伝達率 (audio/cond): {ratio:.4f}")
    lines.append("  ※ 1.0 に近いほど conditioning 操作が音響多様性を正確に制御できている")


def run_exp4_sanity(runner: ConditioningGenotypeRunner, out_dir: str, lines: List[str]):
    """Exp 4: 埋め込み空間の基本的な sanity check。"""
    print("\n" + "=" * 60)
    print("Exp 4: Sanity check — text embedding 間の距離")
    print("=" * 60)

    prompt_pairs = [
        ("calm piano music", "calm piano music"),           # 同一: cos sim ≈ 1.0
        ("calm piano music", "slow relaxing melody"),      # 近い概念
        ("calm piano music", "energetic metal guitar"),    # 遠い概念
        ("calm piano music", "dog barking loudly"),        # 無関係
    ]

    lines.append("\n=== Exp 4: Sanity check ===")
    lines.append(f"{'Prompt A':35}  {'Prompt B':35}  {'cos_sim':>8}")
    lines.append("-" * 82)

    for pa, pb in prompt_pairs:
        ca = runner.encode_text(pa).reshape(1, -1)
        cb = runner.encode_text(pb).reshape(1, -1)
        sim = F.cosine_similarity(ca, cb, dim=1).item()
        line = f"  [{pa}] vs [{pb}] → cos_sim = {sim:.4f}"
        print(line)
        lines.append(f"{pa:35}  {pb:35}  {sim:>+8.4f}")


# ──────────────────────────────────────────────────────────────
# エントリポイント
# ──────────────────────────────────────────────────────────────

def main():
    out_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "outputs", "conditioning_test"
    )
    os.makedirs(out_dir, exist_ok=True)

    runner = ConditioningGenotypeRunner(
        model_name="audioldm-m-full",
        duration=5.0,
        ddim_steps=200,          # 検証用: 本番は 200 推奨
        guidance_scale=2.5,
    )

    lines: List[str] = [
        "条件付けベクトル（CLAP text embedding）遺伝子型 — 検証結果",
        f"実行日時: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"デバイス: {runner.device}",
        f"ddim_steps: {runner.ddim_steps}",
        "",
    ]

    run_exp4_sanity(runner, out_dir, lines)
    run_exp1_interpolation(runner, out_dir, lines)
    run_exp2_mutation(runner, out_dir, lines)
    run_exp3_diversity(runner, out_dir, lines)

    # 結果の保存
    result_path = os.path.join(out_dir, "results.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n結果を保存しました: {result_path}")
    print(f"音声ファイル: {out_dir}/")

    # ──────────────────────────────────
    # 判定サマリ
    # ──────────────────────────────────
    print("\n" + "=" * 60)
    print("判定のポイント（結果を見て確認してください）")
    print("=" * 60)
    print("""
  Exp 4 (Sanity check):
    同一プロンプトのcos_simが≈1.0  → CLAPエンコーダが正常に動作している
    意味的に近い/遠いペアで差が出る → CLAP空間に意味構造がある

  Exp 1 (補間の連続性):
    sim→A が α=0→1 で単調減少      → SLERP補間が意味的に機能している
    sim→B が α=0→1 で単調増加      → 補間が「中間の音」を生成できている
    途中で [DEGRADED] が出ない       → 分布外れが起きていない

  Exp 2 (突然変異の効果):
    σ小 → sim_audio が高い          → 微小変異は音響的に安定している
    σ大 → sim_audio が低い          → 大きな変異は音響的に遠ざかる
    緩やかな単調減少 (急崩壊しない)  → 突然変異が制御可能

  Exp 3 (多様性の相関):
    多様性伝達率 > 0.3              → conditioning操作が音響多様性に伝わっている
    多様性伝達率 ≈ 0.0              → 非線形性が強く、SLERP/変異は機能しない
    """)


if __name__ == "__main__":
    main()

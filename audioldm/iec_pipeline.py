"""
AudioLDMとIECを統合したパイプライン
"""

import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import torchaudio
from typing import List, Optional, Tuple, Dict
import os

from audioldm.pipeline import build_model, duration_to_latent_t_size
from audioldm.latent_diffusion.ddim import DDIMSampler
from audioldm.audio import wav_to_fbank, TacotronSTFT
from audioldm.utils import default_audioldm_config
from audioldm.iec import (
    AudioGenotype, IECPopulation,
    TransformGenotype, crossover_matrix_uniform, mutate_transform_gaussian,
    LatentZ0Genotype, crossover_z0_slerp, mutate_z0_gaussian, slerp,
    StyleTransferGenotype, crossover_style_transfer, mutate_style_transfer,
    ConditioningGenotype, slerp_conditioning, mutate_conditioning_gaussian,
    mutate_conditioning_micro_slerp, crossover_conditioning_slerp,
    STYLE_WORD_BANK,
)
from audioldm.prompt_pool import PROMPT_POOL, sample_prompts


class AudioLDM_IEC:
    """
    AudioLDMを用いた対話型進化的音声生成システム
    """
    
    def __init__(
        self,
        model_name: str = "audioldm-s-full-v2",
        ckpt_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        population_size: int = 6,
        duration: float = 5.0,
        guidance_scale: float = 2.5, # ガイダンススケールのデフォルト値を2.5に設定
        ddim_steps: int = 200,
        n_candidate_gen_per_text: int = 3,
        ga_mode: str = "latent",
    ):
        """
        Args:
            model_name: AudioLDMのモデル名
            ckpt_path: チェックポイントのパス
            device: 使用デバイス
            population_size: 1世代あたりの個体数
            duration: 生成する音声の長さ (秒)
            guidance_scale: ガイダンススケール
            ddim_steps: DDIMサンプリングのステップ数
            n_candidate_gen_per_text: テキストごとの候補生成数
            ga_mode: "latent" (潜在ノイズGA) または "transform" (変換行列GA)
        """
        self.device = device
        self.population_size = population_size
        self.duration = duration
        self.guidance_scale = guidance_scale
        self.ddim_steps = ddim_steps
        self.n_candidate_gen_per_text = n_candidate_gen_per_text
        self.ga_mode = ga_mode
        self._shared_base_noise: Optional[torch.Tensor] = None
        self._fn_stft: Optional[TacotronSTFT] = None
        self._style_transfer_sampler: Optional[DDIMSampler] = None
        
        # AudioLDMモデルのロード
        print(f"AudioLDMモデルをロード中: {model_name}")
        self.latent_diffusion = build_model(
            ckpt_path=ckpt_path,
            model_name=model_name
        )
        self.latent_diffusion = self.latent_diffusion.to(device)
        self.latent_diffusion.eval()
        
        # 潜在空間のサイズを設定
        self.latent_diffusion.latent_t_size = duration_to_latent_t_size(duration)
        self.latent_diffusion.cond_stage_model.embed_mode = "text"
        
        # 潜在空間の形状
        self.latent_shape = (
            self.latent_diffusion.channels,
            self.latent_diffusion.latent_t_size,
            self.latent_diffusion.latent_f_size
        )
        
        # IEC個体群の初期化
        self.population = IECPopulation(population_size=population_size)
        
        print(f"初期化完了: 個体数={population_size}, 音声長={duration}秒, デバイス={device}, モード={ga_mode}")
        print(f"潜在空間形状: {self.latent_shape}")

    # ------------------------------------------------------------------
    # 共通ヘルパー
    # ------------------------------------------------------------------

    def _get_or_create_base_noise(self, seed: Optional[int] = None) -> torch.Tensor:
        """変換行列モード用の共有ベースノイズを取得・生成する。"""
        if self._shared_base_noise is None:
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
                self._shared_base_noise = torch.randn(
                    (1,) + self.latent_shape, device=self.device, generator=generator
                )
            else:
                self._shared_base_noise = torch.randn(
                    (1,) + self.latent_shape, device=self.device
                )
            print(f"ベースノイズを生成しました (seed={seed})")
        return self._shared_base_noise

    def _sample_audio(
        self,
        x_T: torch.Tensor,
        text: str = "",
        unconditional_guidance_scale: Optional[float] = None,
        ddim_eta: float = 0.0,
    ) -> np.ndarray:
        """x_T とテキストから音声波形を生成する共通ルーチン。"""
        return self._sample_audio_batch([x_T], text=text,
                                        unconditional_guidance_scale=unconditional_guidance_scale,
                                        ddim_eta=ddim_eta)[0]

    def _sample_audio_batch(
        self,
        x_T_list: List[torch.Tensor],
        text: str = "",
        unconditional_guidance_scale: Optional[float] = None,
        ddim_eta: float = 0.0,
    ) -> List[np.ndarray]:
        """複数のx_Tをバッチ処理して音声波形リストを返す。テキスト条件・DDIMは1回だけ実行。"""
        if unconditional_guidance_scale is None:
            unconditional_guidance_scale = self.guidance_scale

        n = len(x_T_list)

        with self.latent_diffusion.ema_scope("IEC Generation"):
            with torch.no_grad():
                if text:
                    c_single = self.latent_diffusion.cond_stage_model([text, text])
                    c_single = c_single[0:1]
                else:
                    c_single = self.latent_diffusion.cond_stage_model([" ", " "])
                    c_single = c_single[0:1]

                c = torch.cat([c_single] * n, dim=0)

                uc = (
                    self.latent_diffusion.cond_stage_model.get_unconditional_condition(n)
                    if unconditional_guidance_scale != 1.0
                    else None
                )

                x_T_batch = torch.cat(x_T_list, dim=0)

                samples, _ = self.latent_diffusion.sample_log(
                    cond=c,
                    batch_size=n,
                    ddim=True,
                    ddim_steps=self.ddim_steps,
                    eta=ddim_eta,
                    unconditional_guidance_scale=unconditional_guidance_scale,
                    unconditional_conditioning=uc,
                    x_T=x_T_batch,
                )

                if torch.max(torch.abs(samples)) > 1e2:
                    samples = torch.clip(samples, min=-10, max=10)

                mel = self.latent_diffusion.decode_first_stage(samples)
                waveform_batch = self.latent_diffusion.mel_spectrogram_to_waveform(mel)

        return [waveform_batch[i:i+1] for i in range(n)]

    def _generate_audio_from_any_genotype(
        self,
        genotype,
        text: str = "",
        unconditional_guidance_scale: Optional[float] = None,
        ddim_eta: float = 0.0,
    ) -> np.ndarray:
        """AudioGenotype / TransformGenotype / LatentZ0Genotype / ConditioningGenotype を問わず音声を生成するディスパッチャ。"""
        if isinstance(genotype, ConditioningGenotype):
            return self._generate_audio_from_conditioning_genotype(genotype)
        if isinstance(genotype, LatentZ0Genotype):
            return self._generate_audio_from_z0_genotype(genotype)
        if isinstance(genotype, TransformGenotype):
            return self._generate_audio_from_transform_genotype(
                genotype, text=text,
                unconditional_guidance_scale=unconditional_guidance_scale,
                ddim_eta=ddim_eta,
            )
        return self._generate_audio_from_genotype(
            genotype, text=text,
            unconditional_guidance_scale=unconditional_guidance_scale,
            ddim_eta=ddim_eta,
        )

    # ------------------------------------------------------------------
    # 潜在ノイズ GA（既存モード）
    # ------------------------------------------------------------------

    def _generate_audio_from_genotype(
        self,
        genotype: AudioGenotype,
        text: str = "",
        unconditional_guidance_scale: Optional[float] = None,
        ddim_eta: float = 0.0
    ) -> np.ndarray:
        """
        遺伝子型から音声を生成
        
        Args:
            genotype: 音声遺伝子型（潜在ベクトルを含む）
            text: プロンプトテキスト
            unconditional_guidance_scale: ガイダンススケール
            ddim_eta: DDIMサンプリングのeta（確率性を制御）
        
        Returns:
            生成された音声波形 (numpy array)
        """
        if genotype.seed is not None:
            torch.manual_seed(genotype.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(genotype.seed)
            np.random.seed(genotype.seed % (2**32))

        x_T = genotype.latent_noise.to(self.device)
        return self._sample_audio(x_T, text=text,
                                  unconditional_guidance_scale=unconditional_guidance_scale,
                                  ddim_eta=ddim_eta)

    # ------------------------------------------------------------------
    # 変換行列 GA（新モード）
    # ------------------------------------------------------------------

    def _generate_audio_from_transform_genotype(
        self,
        genotype: TransformGenotype,
        text: str = "",
        unconditional_guidance_scale: Optional[float] = None,
        ddim_eta: float = 0.0,
    ) -> np.ndarray:
        """変換行列遺伝子型から音声を生成する。"""
        x_T = genotype.get_transformed_noise().to(self.device)
        return self._sample_audio(x_T, text=text,
                                  unconditional_guidance_scale=unconditional_guidance_scale,
                                  ddim_eta=ddim_eta)

    def initialize_population_transform(
        self,
        prompt: Optional[str] = None,
        base_noise_seed: Optional[int] = None,
    ) -> List[Tuple[TransformGenotype, np.ndarray]]:
        """
        変換行列モードで初期個体群を生成する。

        各個体の変換行列は N(0, I) からサンプルした C×C 行列で初期化する（論文 Sec. C.1 に準拠）。
        ベースノイズ z_T はセッション全体で共有される。
        """
        print(f"[変換行列モード] 第{self.population.generation_number}世代を生成中...")
        if prompt is None:
            prompt = ""

        base_noise = self._get_or_create_base_noise(seed=base_noise_seed)
        C = self.latent_shape[0]

        genotypes: List[TransformGenotype] = []
        for _ in range(self.population_size):
            seed = np.random.randint(0, 2**32 - 1)
            torch.manual_seed(seed)
            M = torch.randn(C, C, device=self.device)  # N(0,I) → QR → ランダム直交行列 (論文準拠)

            g = TransformGenotype(
                transform_matrix=M,
                base_noise=base_noise,
                seed=seed,
                metadata={
                    "ga_mode": "transform",
                    "initialization": "random_gaussian",
                    "prompt": prompt,
                    "base_noise_seed": base_noise_seed,
                }
            )
            g.generation = self.population.generation_number
            genotypes.append(g)

        self.population.current_generation = genotypes
        self.population.history.append([g.clone() for g in genotypes])

        x_T_list = [g.get_transformed_noise().to(self.device) for g in genotypes]
        waveforms = self._sample_audio_batch(x_T_list, text=prompt)
        return list(zip(genotypes, waveforms))

    def evolve_population_transform(
        self,
        selected_indices: List[int],
        mutation_strength: float = 0.1,
        elite_count: int = 1,
    ) -> List[Tuple[TransformGenotype, np.ndarray]]:
        """
        変換行列モードで次世代を生成する。

        交叉: uniform crossover（論文 Sec. B.3, B.5 に準拠）
        突然変異: 行列にガウスノイズを加算（σ = mutation_strength）
        """
        print(f"\n[変換行列モード] 第{self.population.generation_number + 1}世代を生成中...")
        if not selected_indices:
            raise ValueError("少なくとも1つの個体を選択してください")

        selected = [self.population.current_generation[i] for i in selected_indices]
        prompt = selected[0].metadata.get("prompt", "")

        next_gen: List[TransformGenotype] = []

        for i in range(min(elite_count, len(selected))):
            elite = selected[i].clone()
            elite.generation = self.population.generation_number + 1
            elite.metadata["elite"] = True
            elite.metadata["elite_parent_pop_index"] = selected_indices[i]
            next_gen.append(elite)

        while len(next_gen) < self.population_size:
            if len(selected) == 1:
                child = mutate_transform_gaussian(selected[0], mutation_strength)
                child.metadata["parent_pop_index"] = selected_indices[0]
            else:
                pair_idx = np.random.choice(len(selected), size=2, replace=False)
                p1, p2 = selected[pair_idx[0]], selected[pair_idx[1]]
                child = crossover_matrix_uniform(p1, p2)  # uniform crossover (論文 Sec. B.3, B.5)
                crossover_p = child.metadata.get("crossover_p", 0.5)
                child = mutate_transform_gaussian(child, mutation_strength)
                child.metadata.update({
                    "operation": "crossover_then_mutate",
                    "crossover_type": "uniform",
                    "crossover_parent1_pop_index": selected_indices[pair_idx[0]],
                    "crossover_parent2_pop_index": selected_indices[pair_idx[1]],
                    "crossover_parent1_seed": p1.seed,
                    "crossover_parent2_seed": p2.seed,
                    "crossover_p": crossover_p,
                })

            child.generation = self.population.generation_number + 1
            child.metadata["prompt"] = prompt
            next_gen.append(child)

        self.population.generation_number += 1
        self.population.current_generation = next_gen[:self.population_size]
        self.population.history.append([g.clone() for g in self.population.current_generation])

        genotypes = self.population.current_generation
        x_T_list = [g.get_transformed_noise().to(self.device) for g in genotypes]
        waveforms = self._sample_audio_batch(x_T_list, text=prompt)
        return list(zip(genotypes, waveforms))

    # ------------------------------------------------------------------
    # z_0 GA（新モード: DDIM逆拡散で得られた潜在表現を個体とする）
    # ------------------------------------------------------------------

    def _sample_z0_batch(
        self,
        x_T_list: List[torch.Tensor],
        text: str = "",
        unconditional_guidance_scale: Optional[float] = None,
        ddim_eta: float = 0.0,
    ) -> List[torch.Tensor]:
        """複数の z_T から DDIM で z_0 をバッチ生成し、z_0 テンソルのリストを返す。"""
        if unconditional_guidance_scale is None:
            unconditional_guidance_scale = self.guidance_scale
        n = len(x_T_list)
        with self.latent_diffusion.ema_scope("Z0 DDIM Sampling"):
            with torch.no_grad():
                if text:
                    c_single = self.latent_diffusion.cond_stage_model([text, text])[0:1]
                else:
                    c_single = self.latent_diffusion.cond_stage_model([" ", " "])[0:1]
                c = torch.cat([c_single] * n, dim=0)
                uc = (
                    self.latent_diffusion.cond_stage_model.get_unconditional_condition(n)
                    if unconditional_guidance_scale != 1.0 else None
                )
                x_T_batch = torch.cat(x_T_list, dim=0)
                samples, _ = self.latent_diffusion.sample_log(
                    cond=c, batch_size=n, ddim=True,
                    ddim_steps=self.ddim_steps, eta=ddim_eta,
                    unconditional_guidance_scale=unconditional_guidance_scale,
                    unconditional_conditioning=uc, x_T=x_T_batch,
                )
        return [samples[i:i+1] for i in range(n)]

    def _decode_z0_to_audio(self, z0: torch.Tensor) -> np.ndarray:
        """z_0 を VAE デコードして音声波形を返す（DDIM 不要）。"""
        with self.latent_diffusion.ema_scope("Z0 Decode"):
            with torch.no_grad():
                if torch.max(torch.abs(z0)) > 1e2:
                    z0 = torch.clip(z0, min=-10, max=10)
                mel = self.latent_diffusion.decode_first_stage(z0)
                waveform = self.latent_diffusion.mel_spectrogram_to_waveform(mel)
        return waveform[0:1]

    def _decode_z0_batch(self, z0_list: List[torch.Tensor]) -> List[np.ndarray]:
        """複数の z_0 をバッチ VAE デコードして音声波形リストを返す。"""
        with self.latent_diffusion.ema_scope("Z0 Batch Decode"):
            with torch.no_grad():
                z0_batch = torch.cat(z0_list, dim=0)
                if torch.max(torch.abs(z0_batch)) > 1e2:
                    z0_batch = torch.clip(z0_batch, min=-10, max=10)
                mel = self.latent_diffusion.decode_first_stage(z0_batch)
                waveform_batch = self.latent_diffusion.mel_spectrogram_to_waveform(mel)
        n = len(z0_list)
        return [waveform_batch[i:i+1] for i in range(n)]

    def _generate_audio_from_z0_genotype(self, genotype: LatentZ0Genotype) -> np.ndarray:
        """z_0 を直接 VAE デコードして音声生成（DDIM 不要）。"""
        return self._decode_z0_to_audio(genotype.z0.to(self.device))

    def _sdedit_refine_z0_batch(
        self,
        z0_list: List[torch.Tensor],
        text: str = "",
        noise_strength: float = 0.2,
        unconditional_guidance_scale: Optional[float] = None,
    ) -> List[torch.Tensor]:
        """方針A: z0 → z_t（順拡散）→ z0_refined（DDIM逆拡散）。

        交叉で得られたz0を拡散モデルの多様体へ投影して音質を改善する。
        noise_strength が大きいほど多様体への投影が強いが親の特徴が薄れる。
        """
        if unconditional_guidance_scale is None:
            unconditional_guidance_scale = self.guidance_scale
        n = len(z0_list)
        noise_steps = max(1, int(noise_strength * self.ddim_steps))

        with self.latent_diffusion.ema_scope("SDEdit Refine"):
            with torch.no_grad():
                sampler = DDIMSampler(self.latent_diffusion)
                sampler.make_schedule(ddim_num_steps=self.ddim_steps, ddim_eta=0.0, verbose=False)

                if text:
                    c_single = self.latent_diffusion.cond_stage_model([text, text])[0:1]
                else:
                    c_single = self.latent_diffusion.cond_stage_model([" ", " "])[0:1]
                c = torch.cat([c_single] * n, dim=0)
                uc = (
                    self.latent_diffusion.cond_stage_model.get_unconditional_condition(n)
                    if unconditional_guidance_scale != 1.0 else None
                )

                z0_batch = torch.cat(z0_list, dim=0)
                t = torch.full((n,), noise_steps - 1, dtype=torch.long, device=self.device)

                z_t_batch = sampler.stochastic_encode(z0_batch, t)
                z0_refined_batch = sampler.decode(
                    z_t_batch, c, t_start=noise_steps,
                    unconditional_guidance_scale=unconditional_guidance_scale,
                    unconditional_conditioning=uc,
                )
        return [z0_refined_batch[i:i+1] for i in range(n)]

    # ------------------------------------------------------------------
    # スタイル転送ヘルパー
    # ------------------------------------------------------------------

    def _get_stft_instance(self) -> TacotronSTFT:
        """TacotronSTFT インスタンスを遅延生成してキャッシュする。"""
        if self._fn_stft is None:
            config = default_audioldm_config()
            pp = config["preprocessing"]
            self._fn_stft = TacotronSTFT(
                pp["stft"]["filter_length"],
                pp["stft"]["hop_length"],
                pp["stft"]["win_length"],
                pp["mel"]["n_mel_channels"],
                pp["audio"]["sampling_rate"],
                pp["mel"]["mel_fmin"],
                pp["mel"]["mel_fmax"],
            )
        return self._fn_stft

    def _get_style_transfer_sampler(self) -> DDIMSampler:
        """SDEdit 用 DDIMSampler をキャッシュして返す。"""
        if self._style_transfer_sampler is None:
            sampler = DDIMSampler(self.latent_diffusion)
            sampler.make_schedule(
                ddim_num_steps=self.ddim_steps, ddim_eta=0.0, verbose=False)
            self._style_transfer_sampler = sampler
        return self._style_transfer_sampler

    def _load_waveform_for_clap(self, audio_path: str) -> torch.Tensor:
        """WAV ファイルを 16kHz モノラルに読み込み (bs=1, t_samples) テンソルを返す。

        CLAP forward() は (bs, t) を期待する (_random_mute コメント参照)。
        batch_to_list 後に (t,) の 1D テンソルになり get_audio_features に渡る。
        """
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)
        waveform = waveform / (waveform.abs().max() + 1e-8) * 0.5
        return waveform.float()  # (1, t)

    def encode_audio_to_z0(self, audio_path: str) -> torch.Tensor:
        """WAV ファイルを mel 変換 → VAE エンコードして z0 潜在表現を返す。

        pipeline.style_transfer() の lines 204-213 と同じパターン。
        Returns:
            z0: shape (1, C, T, F) on self.device
        """
        fn_stft = self._get_stft_instance()
        mel, _, _ = wav_to_fbank(
            audio_path,
            target_length=int(self.duration * 102.4),
            fn_STFT=fn_stft,
        )
        mel = mel.unsqueeze(0).unsqueeze(0).to(self.device)
        with self.latent_diffusion.ema_scope("Encode Audio"):
            with torch.no_grad():
                z0 = self.latent_diffusion.get_first_stage_encoding(
                    self.latent_diffusion.encode_first_stage(mel)
                )
        if torch.max(torch.abs(z0)) > 1e2:
            z0 = torch.clip(z0, -10, 10)
        return z0.detach()

    def rank_style_words(
        self,
        audio_path: str,
        style_words: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """音声Bに対する CLAP コサイン類似度でスタイル語を降順にランキングする。

        Returns:
            [(word, score), ...] 長さ top_k のリスト（降順ソート済み）
        """
        words = style_words if style_words is not None else STYLE_WORD_BANK
        waveform = self._load_waveform_for_clap(audio_path).to(self.device)

        cond = self.latent_diffusion.cond_stage_model
        orig_prob = cond.unconditional_prob
        cond.unconditional_prob = 0.0
        try:
            with torch.no_grad():
                cond.embed_mode = "audio"
                audio_emb = cond(waveform).squeeze(1)  # (1, 512)
                cond.embed_mode = "text"
                text_emb = cond(words).squeeze(1)      # (N, 512)
                sims = F.cosine_similarity(
                    audio_emb.expand(len(words), -1), text_emb, dim=1
                )
        finally:
            cond.embed_mode = "text"
            cond.unconditional_prob = orig_prob

        ranked = sorted(zip(words, sims.cpu().tolist()), key=lambda x: -x[1])
        return ranked[:top_k]

    def rank_differential_style_words(
        self,
        audio_b_path: str,
        audio_a_path: str,
        style_words: Optional[List[str]] = None,
        top_k: int = 5,
        pool_size: int = 10,
    ) -> List[Tuple[str, float]]:
        """B の Top-pool_size スタイル語から A の Top-pool_size スタイル語を除いた B 固有語。

        集合差分: B のトップリストに含まれ A のトップリストに含まれない語を返す。
        両親で共通して高い語（mysterious, ambient 等）は除外され B 固有のシグナルだけが残る。

        Returns:
            [(word, score_b), ...] top_k 件（B スコア降順）
        """
        words = style_words if style_words is not None else STYLE_WORD_BANK
        waveform_a = self._load_waveform_for_clap(audio_a_path).to(self.device)
        waveform_b = self._load_waveform_for_clap(audio_b_path).to(self.device)

        cond = self.latent_diffusion.cond_stage_model
        orig_prob = cond.unconditional_prob
        cond.unconditional_prob = 0.0
        try:
            with torch.no_grad():
                cond.embed_mode = "audio"
                audio_emb_a = cond(waveform_a).squeeze(1)
                audio_emb_b = cond(waveform_b).squeeze(1)
                cond.embed_mode = "text"
                text_emb = cond(words).squeeze(1)
                sims_a = F.cosine_similarity(
                    audio_emb_a.expand(len(words), -1), text_emb, dim=1).cpu().tolist()
                sims_b = F.cosine_similarity(
                    audio_emb_b.expand(len(words), -1), text_emb, dim=1).cpu().tolist()
        finally:
            cond.embed_mode = "text"
            cond.unconditional_prob = orig_prob

        top_b = sorted(zip(words, sims_b), key=lambda x: -x[1])[:pool_size]
        top_a_words = {w for w, _ in sorted(zip(words, sims_a), key=lambda x: -x[1])[:pool_size]}
        exclusive_b = [(w, s) for w, s in top_b if w not in top_a_words]
        return exclusive_b[:top_k]

    def build_style_prompt(
        self,
        base_prompt: str,
        ranked_words: List[Tuple[str, float]],
    ) -> str:
        """ベースプロンプト + スタイル語を結合して合成プロンプトを生成する。"""
        words_str = ", ".join(w for w, _ in ranked_words)
        if base_prompt.strip():
            return f"{base_prompt.strip()}, {words_str}"
        return words_str

    def _sdedit_single(
        self,
        sampler: DDIMSampler,
        z0: torch.Tensor,
        style_prompt: str,
        noise_steps: int,
        guidance_scale: float,
    ) -> torch.Tensor:
        """z0 に noise_steps 分の順拡散を加え、style_prompt で逆拡散する (SDEdit)。"""
        t = torch.full((1,), noise_steps - 1, dtype=torch.long, device=self.device)
        c_single = self.latent_diffusion.cond_stage_model([style_prompt, style_prompt])[0:1]
        uc = (
            self.latent_diffusion.cond_stage_model.get_unconditional_condition(1)
            if guidance_scale != 1.0 else None
        )
        z_t = sampler.stochastic_encode(z0, t)
        z0_out = sampler.decode(
            z_t, c_single, t_start=noise_steps,
            unconditional_guidance_scale=guidance_scale,
            unconditional_conditioning=uc,
        )
        return z0_out

    def generate_style_transfer_audio_batch(
        self,
        genotypes: List[StyleTransferGenotype],
    ) -> List[np.ndarray]:
        """StyleTransferGenotype のリストから SDEdit で音声を生成する。

        各個体の noise_strength / guidance_scale / seed / mask_start / mask_end を尊重して
        per-individual に SDEdit を実行する。
        """
        sampler = self._get_style_transfer_sampler()
        waveforms = []

        with self.latent_diffusion.ema_scope("StyleTransfer Generation"):
            with torch.no_grad():
                for g in genotypes:
                    torch.manual_seed(g.seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed(g.seed)

                    noise_steps = max(1, int(g.noise_strength * self.ddim_steps))
                    z0 = g.z0_content.to(self.device)
                    T = z0.shape[2]

                    if g.mask_start > 0.0 or g.mask_end < 1.0:
                        ts = int(g.mask_start * T)
                        te = max(ts + 1, int(g.mask_end * T))
                        z0_slice = z0[:, :, ts:te, :]
                        slice_T = z0_slice.shape[2]
                        pad_factor = 16
                        pad_to = ((slice_T + pad_factor - 1) // pad_factor) * pad_factor
                        if pad_to != slice_T:
                            z0_slice = F.pad(z0_slice, (0, 0, 0, pad_to - slice_T))
                        z0_slice_refined = self._sdedit_single(
                            sampler, z0_slice, g.style_prompt, noise_steps, g.guidance_scale)
                        z0_slice_refined = z0_slice_refined[:, :, :slice_T, :]
                        z0_out = z0.clone()
                        z0_out[:, :, ts:te, :] = z0_slice_refined
                    else:
                        z0_out = self._sdedit_single(
                            sampler, z0, g.style_prompt, noise_steps, g.guidance_scale)

                    waveforms.append(self._decode_z0_to_audio(z0_out))

        return waveforms

    def _crossover_z0_via_zt_batch(
        self,
        parent_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
        alphas: List[float],
        text: str = "",
        noise_strength: float = 0.2,
        unconditional_guidance_scale: Optional[float] = None,
    ) -> List[torch.Tensor]:
        """方針B: 両親を同一ノイズでz_t空間へ変換 → SLERP → DDIM逆拡散。

        同一ノイズを使うことで z_t = sqrt(α)*z0 + sqrt(1-α)*ε の z0 部分だけが
        異なる状態を作り、z_t 空間での SLERP が有意義になる。
        """
        if unconditional_guidance_scale is None:
            unconditional_guidance_scale = self.guidance_scale
        n = len(parent_pairs)
        noise_steps = max(1, int(noise_strength * self.ddim_steps))

        with self.latent_diffusion.ema_scope("ZT Crossover"):
            with torch.no_grad():
                sampler = DDIMSampler(self.latent_diffusion)
                sampler.make_schedule(ddim_num_steps=self.ddim_steps, ddim_eta=0.0, verbose=False)

                if text:
                    c_single = self.latent_diffusion.cond_stage_model([text, text])[0:1]
                else:
                    c_single = self.latent_diffusion.cond_stage_model([" ", " "])[0:1]
                c = torch.cat([c_single] * n, dim=0)
                uc = (
                    self.latent_diffusion.cond_stage_model.get_unconditional_condition(n)
                    if unconditional_guidance_scale != 1.0 else None
                )

                z_t_children = []
                t = torch.full((1,), noise_steps - 1, dtype=torch.long, device=self.device)
                for (z0_p1, z0_p2), alpha in zip(parent_pairs, alphas):
                    noise = torch.randn_like(z0_p1)
                    z_t_p1 = sampler.stochastic_encode(z0_p1, t, noise=noise)
                    z_t_p2 = sampler.stochastic_encode(z0_p2, t, noise=noise)
                    z_t_child = slerp(z_t_p1, z_t_p2, alpha)
                    z_t_children.append(z_t_child)

                z_t_batch = torch.cat(z_t_children, dim=0)
                z0_batch = sampler.decode(
                    z_t_batch, c, t_start=noise_steps,
                    unconditional_guidance_scale=unconditional_guidance_scale,
                    unconditional_conditioning=uc,
                )
        return [z0_batch[i:i+1] for i in range(n)]

    # ------------------------------------------------------------------
    # Conditioning Vector GA
    # ------------------------------------------------------------------

    # B-2 SLERP 用のデフォルトプロンプトプール
    def _encode_text_single(self, prompt: str) -> torch.Tensor:
        """テキストを CLAP text embedding (1, 1, 512) に変換する。"""
        cond = self.latent_diffusion.cond_stage_model
        orig_mode = cond.embed_mode
        orig_prob = cond.unconditional_prob
        cond.embed_mode = "text"
        cond.unconditional_prob = 0.0
        try:
            with torch.no_grad():
                emb = cond([prompt, prompt])   # (2, 1, 512)
                return emb[0:1].clone()        # (1, 1, 512)
        finally:
            cond.embed_mode = orig_mode
            cond.unconditional_prob = orig_prob

    def _generate_audio_from_conditioning_genotype(
        self, genotype: ConditioningGenotype
    ) -> np.ndarray:
        """ConditioningGenotype (CLAP embedding + 固定 x_T) から音声を生成する。"""
        cond = genotype.embedding.to(self.device)   # (1, 1, 512)
        x_T = genotype.x_T.to(self.device)          # (1, C, T, F)
        with self.latent_diffusion.ema_scope("Conditioning Genotype"):
            with torch.no_grad():
                uc = self.latent_diffusion.cond_stage_model.get_unconditional_condition(1)
                samples, _ = self.latent_diffusion.sample_log(
                    cond=cond,
                    batch_size=1,
                    ddim=True,
                    ddim_steps=self.ddim_steps,
                    eta=0.0,
                    unconditional_guidance_scale=self.guidance_scale,
                    unconditional_conditioning=uc,
                    x_T=x_T,
                )
                if torch.max(torch.abs(samples)) > 1e2:
                    samples = torch.clip(samples, min=-10, max=10)
                mel = self.latent_diffusion.decode_first_stage(samples)
                wf = self.latent_diffusion.mel_spectrogram_to_waveform(mel)
        return wf[0:1]

    def _compute_centroid_stability(
        self,
        embeddings_prev: List[torch.Tensor],
        embeddings_curr: List[torch.Tensor],
    ) -> float:
        """選択個体の重心の世代間コサイン距離を返す。

        値が小さいほど重心が安定（収束）している。閾値 ε=0.01 以下で収束警告。
        """
        centroid_prev = torch.stack([e.float().flatten() for e in embeddings_prev]).mean(0)
        centroid_curr = torch.stack([e.float().flatten() for e in embeddings_curr]).mean(0)
        sim = F.cosine_similarity(centroid_prev.unsqueeze(0), centroid_curr.unsqueeze(0)).item()
        return float(1.0 - sim)

    def _compute_population_diversity(
        self,
        embeddings: List[torch.Tensor],
    ) -> float:
        """集団内の平均ペアワイズ余弦距離を返す。

        値が小さいほど集団が均質（多様性消失）。閾値 δ=0.05 以下で多様性消失警告。
        """
        vecs = [e.float().flatten() for e in embeddings]
        dists = []
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                sim = F.cosine_similarity(vecs[i].unsqueeze(0), vecs[j].unsqueeze(0)).item()
                dists.append(1.0 - sim)
        return float(np.mean(dists)) if dists else 0.0

    def _compute_conditioning_convergence(
        self,
        selected: List[ConditioningGenotype],
        centroid_eps: float = 0.01,
        diversity_delta: float = 0.05,
    ) -> Dict:
        """選択個体の収束状態を計算し、警告フラグと指標を辞書で返す。

        convergence_history に記録し、2世代連続で重心安定なら centroid_converged=True。
        """
        curr_embeddings = [g.embedding for g in self.population.current_generation]
        diversity = self._compute_population_diversity(curr_embeddings)

        centroid_dist = None
        if hasattr(self, "_prev_selected_embeddings") and self._prev_selected_embeddings:
            centroid_dist = self._compute_centroid_stability(
                self._prev_selected_embeddings,
                [g.embedding for g in selected],
            )
        self._prev_selected_embeddings = [g.embedding.clone() for g in selected]

        centroid_stable = centroid_dist is not None and centroid_dist < centroid_eps
        diversity_low = diversity < diversity_delta

        # 連続収束カウント
        if not hasattr(self, "_centroid_stable_streak"):
            self._centroid_stable_streak = 0
        if centroid_stable:
            self._centroid_stable_streak += 1
        else:
            self._centroid_stable_streak = 0

        info = {
            "generation": self.population.generation_number,
            "centroid_dist": centroid_dist,
            "diversity": diversity,
            "centroid_stable": centroid_stable,
            "centroid_stable_streak": self._centroid_stable_streak,
            "diversity_low": diversity_low,
            "centroid_converged": self._centroid_stable_streak >= 2,
        }

        if not hasattr(self.population, "convergence_history"):
            self.population.convergence_history = []
        self.population.convergence_history.append(info)

        return info

    def _generate_audio_batch_conditioning(
        self, genotypes: List[ConditioningGenotype]
    ) -> List[np.ndarray]:
        """ConditioningGenotype のバッチを一括処理する（x_T は全個体で同一を期待）。"""
        n = len(genotypes)
        cond_batch = torch.cat([g.embedding.to(self.device) for g in genotypes], dim=0)  # (N,1,512)
        x_T_batch = genotypes[0].x_T.to(self.device).expand(n, -1, -1, -1).clone()

        with self.latent_diffusion.ema_scope("Conditioning Batch"):
            with torch.no_grad():
                uc = self.latent_diffusion.cond_stage_model.get_unconditional_condition(n)
                samples, _ = self.latent_diffusion.sample_log(
                    cond=cond_batch,
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
                mel = self.latent_diffusion.decode_first_stage(samples)
                wf_batch = self.latent_diffusion.mel_spectrogram_to_waveform(mel)
        return [wf_batch[i:i+1] for i in range(n)]

    def initialize_population_conditioning(
        self,
        prompt: str,
        slerp_alpha: float = 0.2,
        x_T_seed: Optional[int] = None,
        prompt_pool: Optional[List[str]] = None,
    ) -> List[Tuple[ConditioningGenotype, np.ndarray]]:
        """Conditioning Vector GA の初期個体群を生成する。

        初期多様性の注入: B-2 SLERP（ベースプロンプトのCLAP埋め込みと
        ランダムプロンプトのCLAP埋め込みの間を alpha だけ補間）。
        x_T はセッション全体で固定・共有する。
        """
        print(f"[Conditioning GA] 第{self.population.generation_number}世代を生成中...")
        print(f"  プロンプト: '{prompt}'  alpha={slerp_alpha}")

        pool = prompt_pool or PROMPT_POOL

        # 固定 x_T を生成・保存
        if x_T_seed is not None:
            gen = torch.Generator(device=self.device).manual_seed(x_T_seed)
            x_T = torch.randn((1,) + self.latent_shape, device=self.device, generator=gen)
        else:
            x_T = torch.randn((1,) + self.latent_shape, device=self.device)
        self._conditioning_x_T = x_T

        # ベースプロンプトをエンコード
        c_base = self._encode_text_single(prompt)           # (1, 1, 512)
        base_norm = float(torch.norm(c_base.float()).item())

        # 重複なしでサンプリング。プール数が個体数より少ない場合のみ replace=True にフォールバック
        if len(pool) >= self.population_size:
            sampled_prompts = np.random.choice(pool, size=self.population_size, replace=False).tolist()
        else:
            # プールを使い切った後、残りは再度重複なしで補完する
            sampled_prompts = list(np.random.permutation(pool))
            while len(sampled_prompts) < self.population_size:
                extra = np.random.choice(
                    pool,
                    size=min(len(pool), self.population_size - len(sampled_prompts)),
                    replace=False,
                ).tolist()
                sampled_prompts.extend(extra)
            sampled_prompts = sampled_prompts[:self.population_size]

        genotypes: List[ConditioningGenotype] = []
        for i, rand_prompt in enumerate(sampled_prompts):
            seed = int(np.random.randint(0, 2**32 - 1))
            if slerp_alpha > 0.0:
                c_rand = self._encode_text_single(rand_prompt)
                embedding = slerp_conditioning(c_base, c_rand, slerp_alpha)
            else:
                embedding = c_base.clone()
            g = ConditioningGenotype(
                embedding=embedding,
                x_T=x_T,
                source_prompt=rand_prompt if slerp_alpha > 0.0 else prompt,
                seed=seed,
                metadata={
                    "ga_mode": "conditioning",
                    "initialization": "slerp_prompt",
                    "base_prompt": prompt,
                    "rand_prompt": rand_prompt,
                    "slerp_alpha": slerp_alpha,
                    "prompt": prompt,
                },
            )
            g.generation = self.population.generation_number
            genotypes.append(g)

        self.population.current_generation = genotypes
        self.population.history.append([g.clone() for g in genotypes])

        waveforms = self._generate_audio_batch_conditioning(genotypes)
        print(f"  [Conditioning GA] {len(genotypes)}個体の音声生成完了")
        return list(zip(genotypes, waveforms))

    def evolve_population_conditioning(
        self,
        selected_indices: List[int],
        mutation_mu_range: Tuple[float, float] = (0.05, 0.15),
        p_mut: float = 0.4,
        elite_count: int = 2,
        random_sample_count: int = 1,
        prompt_pool: Optional[List[str]] = None,
    ) -> Tuple[List[Tuple[ConditioningGenotype, np.ndarray]], Dict]:
        """Conditioning Vector GA の次世代を生成する。

        世代構成（6体標準）:
          - エリート: min(elite_count, len(selected)) 体
          - ランダムサンプル: random_sample_count 体（末尾、SLERP B-2）
          - 交叉スロット: 残り体数（SLERP 交叉 + 確率 p_mut で Micro-SLERP 変異）

        交叉: conditioning vector 間の SLERP (alpha ~ Uniform(0.3, 0.7))
        変異: Micro-SLERP（プール embedding 方向へ mu ~ Uniform(*mutation_mu_range) だけ移動）
        ランダムサンプル: プールから新規 CLAP embedding → SLERP B-2 (alpha ~ Uniform(0.3, 0.6))

        Returns:
            (個体, 波形) のリストと収束情報辞書
        """
        print(f"\n[Conditioning GA] 第{self.population.generation_number + 1}世代を生成中...")
        if not selected_indices:
            raise ValueError("少なくとも1つの個体を選択してください")

        selected = [self.population.current_generation[i] for i in selected_indices]
        prompt = selected[0].metadata.get("base_prompt", "")
        x_T = selected[0].x_T   # 全個体で共有

        pool = prompt_pool or PROMPT_POOL
        # ベースプロンプトの CLAP embedding（Micro-SLERP 変異の起点として使用）
        c_base = self._encode_text_single(prompt) if prompt else None

        next_gen: List[ConditioningGenotype] = []

        # 1. エリート保存
        actual_elite = min(elite_count, len(selected))
        for i in range(actual_elite):
            elite = selected[i].clone()
            elite.generation = self.population.generation_number + 1
            elite.metadata["elite"] = True
            elite.metadata["elite_parent_pop_index"] = selected_indices[i]
            next_gen.append(elite)

        # 2. 末尾にランダムサンプルスロットを予約（後で埋める）
        actual_random = min(random_sample_count, self.population_size - actual_elite)
        crossover_slots = self.population_size - actual_elite - actual_random

        # 3. 交叉スロットを埋める
        while len(next_gen) < actual_elite + crossover_slots:
            if len(selected) == 1:
                # 選択個体が1体の場合は Micro-SLERP のみ
                pool_prompt = sample_prompts(1, exclude=[prompt])[0]
                c_pool = self._encode_text_single(pool_prompt)
                child = mutate_conditioning_micro_slerp(
                    selected[0],
                    c_pool,
                    mu=float(np.random.uniform(*mutation_mu_range)),
                )
                child.metadata["parent_pop_index"] = selected_indices[0]
            else:
                pair_idx = np.random.choice(len(selected), size=2, replace=False)
                p1, p2 = selected[pair_idx[0]], selected[pair_idx[1]]
                alpha = float(np.random.uniform(0.3, 0.7))
                child = crossover_conditioning_slerp(p1, p2, alpha)
                child.metadata["crossover_parent1_pop_index"] = selected_indices[pair_idx[0]]
                child.metadata["crossover_parent2_pop_index"] = selected_indices[pair_idx[1]]
                # 確率 p_mut で Micro-SLERP 変異を適用
                if np.random.random() < p_mut:
                    pool_prompt = sample_prompts(1, exclude=[prompt])[0]
                    c_pool = self._encode_text_single(pool_prompt)
                    mu = float(np.random.uniform(*mutation_mu_range))
                    child = mutate_conditioning_micro_slerp(child, c_pool, mu=mu)
                    child.metadata.update({
                        "operation": "crossover_then_mutate",
                        "crossover_type": "slerp",
                        "crossover_parent1_pop_index": selected_indices[pair_idx[0]],
                        "crossover_parent2_pop_index": selected_indices[pair_idx[1]],
                        "crossover_alpha": alpha,
                        "mu": mu,
                        "pool_prompt": pool_prompt,
                    })
            child.x_T = x_T
            child.generation = self.population.generation_number + 1
            child.metadata["base_prompt"] = prompt
            child.metadata["prompt"] = prompt
            next_gen.append(child)

        # 4. ランダムサンプルスロット: プールから新規 CLAP embedding → SLERP B-2
        rs_prompts = sample_prompts(actual_random, exclude=[prompt])
        for rs_prompt in rs_prompts:
            c_rand = self._encode_text_single(rs_prompt)
            alpha_rand = float(np.random.uniform(0.3, 0.6))
            if c_base is not None:
                embedding = slerp_conditioning(c_base, c_rand, alpha_rand)
            else:
                embedding = c_rand
            child = ConditioningGenotype(
                embedding=embedding,
                x_T=x_T,
                source_prompt=rs_prompt,
                seed=int(np.random.randint(0, 2**32 - 1)),
                metadata={
                    "ga_mode": "conditioning",
                    "operation": "random_sample",
                    "base_prompt": prompt,
                    "prompt": prompt,
                    "rand_prompt": rs_prompt,
                    "slerp_alpha": alpha_rand,
                },
            )
            child.generation = self.population.generation_number + 1
            next_gen.append(child)

        self.population.generation_number += 1
        self.population.current_generation = next_gen[:self.population_size]
        self.population.history.append([g.clone() for g in self.population.current_generation])

        # 5. 収束情報を計算
        convergence_info = self._compute_conditioning_convergence(selected)

        waveforms = self._generate_audio_batch_conditioning(self.population.current_generation)
        return list(zip(self.population.current_generation, waveforms)), convergence_info

    def initialize_population_z0(
        self,
        prompt: Optional[str] = None,
    ) -> List[Tuple[LatentZ0Genotype, np.ndarray]]:
        """
        z_0 モードの初期個体群を生成する。

        各個体: ランダム z_T → DDIM サンプリング → z_0 を個体として保持。
        以降の子個体の評価は VAE decode のみで済む（DDIM 不要）。
        """
        print(f"[z0モード] 第{self.population.generation_number}世代を生成中...")
        if prompt is None:
            prompt = ""

        x_T_list = []
        seeds = []
        for _ in range(self.population_size):
            seed = np.random.randint(0, 2**32 - 1)
            seeds.append(seed)
            generator = torch.Generator(device=self.device).manual_seed(seed)
            x_T = torch.randn((1,) + self.latent_shape, device=self.device, generator=generator)
            x_T_list.append(x_T)

        print(f"  DDIM サンプリングで z_0 を生成中 (個体数={self.population_size})...")
        z0_list = self._sample_z0_batch(x_T_list, text=prompt)

        genotypes = []
        for z0, seed in zip(z0_list, seeds):
            g = LatentZ0Genotype(
                z0=z0,
                seed=seed,
                metadata={"initialization": "ddim_sampling", "prompt": prompt, "ga_mode": "z0"}
            )
            g.generation = self.population.generation_number
            genotypes.append(g)

        self.population.current_generation = genotypes
        self.population.history.append([g.clone() for g in genotypes])

        print(f"  VAE デコードで音声を生成中...")
        waveforms = self._decode_z0_batch([g.z0.to(self.device) for g in genotypes])
        return list(zip(genotypes, waveforms))

    def evolve_population_z0(
        self,
        selected_indices: List[int],
        mutation_strength: float = 0.15,
        elite_count: int = 1,
        fresh_count: int = 1,
        crossover_mode: str = "z0",
        sdedit_strength: float = 0.0,
    ) -> List[Tuple[LatentZ0Genotype, np.ndarray]]:
        """
        z_0 空間での交叉・変異 → VAE デコードで音声生成。

        Args:
            crossover_mode: "z0" = z0空間でSLERP後にオプションでSDEdit（方針A）、
                            "zt" = z_t空間でSLERP→逆拡散（方針B）。
            sdedit_strength: 0.0 = 無効。>0 のとき:
                             方針A: 交叉後z0にSDEditを適用するノイズ量 (ddim_stepsの割合)。
                             方針B: z_t空間交叉のノイズ量。
            fresh_count: 毎世代 DDIM で完全新規生成する個体数。
        """
        print(f"\n[z0モード] 第{self.population.generation_number + 1}世代を生成中...")
        if not selected_indices:
            raise ValueError("少なくとも1つの個体を選択してください")

        selected = [self.population.current_generation[i] for i in selected_indices]
        prompt = selected[0].metadata.get("prompt", "")

        next_gen: List[LatentZ0Genotype] = []

        # エリート保存
        for i in range(min(elite_count, len(selected))):
            elite = selected[i].clone()
            elite.generation = self.population.generation_number + 1
            elite.metadata["elite"] = True
            elite.metadata["elite_parent_pop_index"] = selected_indices[i]
            next_gen.append(elite)

        # DDIM 新鮮注入: slerp だけだと全子が親の「間」に収束するため、
        # 完全ランダムな z_T から DDIM で生成した個体を混入して多様性を維持する。
        actual_fresh = min(fresh_count, self.population_size - len(next_gen))
        if actual_fresh > 0:
            print(f"  DDIM 新鮮注入: {actual_fresh} 体を DDIM で新規生成中...")
            fresh_x_T_list = []
            fresh_seeds = []
            for _ in range(actual_fresh):
                seed = np.random.randint(0, 2**32 - 1)
                fresh_seeds.append(seed)
                generator = torch.Generator(device=self.device).manual_seed(seed)
                x_T = torch.randn((1,) + self.latent_shape, device=self.device, generator=generator)
                fresh_x_T_list.append(x_T)
            fresh_z0_list = self._sample_z0_batch(fresh_x_T_list, text=prompt)
            for z0, seed in zip(fresh_z0_list, fresh_seeds):
                g = LatentZ0Genotype(
                    z0=z0, seed=seed,
                    metadata={"operation": "ddim_fresh_injection", "prompt": prompt,
                               "ga_mode": "z0", "generation": self.population.generation_number + 1}
                )
                g.generation = self.population.generation_number + 1
                next_gen.append(g)

        # 交叉 + 突然変異で残りを埋める
        if crossover_mode == "zt" and len(selected) >= 2 and sdedit_strength > 0:
            # 方針B: z_t空間での交叉（ペアを先にまとめてバッチ逆拡散）
            print(f"  方針B: z_t空間交叉 (noise_strength={sdedit_strength})...")
            pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
            alphas_list: List[float] = []
            pair_meta: List[dict] = []
            while len(next_gen) + len(pairs) < self.population_size:
                pair_idx = np.random.choice(len(selected), size=2, replace=False)
                p1, p2 = selected[pair_idx[0]], selected[pair_idx[1]]
                alpha = np.random.uniform(0.1, 0.9)
                pairs.append((p1.z0.to(self.device), p2.z0.to(self.device)))
                alphas_list.append(alpha)
                pair_meta.append({
                    "operation": "crossover_zt_slerp",
                    "crossover_mode": "zt",
                    "crossover_type": "zt_slerp",
                    "crossover_parent1_pop_index": selected_indices[pair_idx[0]],
                    "crossover_parent2_pop_index": selected_indices[pair_idx[1]],
                    "crossover_parent1_seed": p1.seed,
                    "crossover_parent2_seed": p2.seed,
                    "crossover_alpha": alpha,
                    "sdedit_strength": sdedit_strength,
                    "prompt": prompt,
                })
            z0_children = self._crossover_z0_via_zt_batch(
                pairs, alphas_list, text=prompt, noise_strength=sdedit_strength
            )
            for z0_child, meta in zip(z0_children, pair_meta):
                child = LatentZ0Genotype(
                    z0=z0_child,
                    seed=np.random.randint(0, 2**32 - 1),
                    metadata=meta,
                )
                child.generation = self.population.generation_number + 1
                next_gen.append(child)
        else:
            # 方針A: z0空間でSLERP交叉（既存ロジック）
            while len(next_gen) < self.population_size:
                if len(selected) == 1:
                    child = mutate_z0_gaussian(selected[0], mutation_strength)
                    child.metadata["parent_pop_index"] = selected_indices[0]
                else:
                    pair_idx = np.random.choice(len(selected), size=2, replace=False)
                    p1, p2 = selected[pair_idx[0]], selected[pair_idx[1]]
                    alpha = np.random.uniform(0.1, 0.9)
                    child = crossover_z0_slerp(p1, p2, alpha)
                    child = mutate_z0_gaussian(child, mutation_strength)
                    child.metadata.update({
                        "operation": "crossover_then_mutate",
                        "crossover_type": "slerp",
                        "crossover_parent1_pop_index": selected_indices[pair_idx[0]],
                        "crossover_parent2_pop_index": selected_indices[pair_idx[1]],
                        "crossover_parent1_seed": p1.seed,
                        "crossover_parent2_seed": p2.seed,
                        "crossover_alpha": alpha,
                    })
                child.generation = self.population.generation_number + 1
                child.metadata["prompt"] = prompt
                next_gen.append(child)

            # 方針A: SDEdit後処理（sdedit_strength > 0 のとき交叉子を精製）
            if sdedit_strength > 0:
                crossover_children = [
                    g for g in next_gen
                    if g.metadata.get("operation") == "crossover_then_mutate"
                ]
                if crossover_children:
                    print(f"  方針A: SDEdit精製中 (strength={sdedit_strength}, {len(crossover_children)}体)...")
                    refined_list = self._sdedit_refine_z0_batch(
                        [g.z0.to(self.device) for g in crossover_children],
                        text=prompt, noise_strength=sdedit_strength,
                    )
                    for g, z0_refined in zip(crossover_children, refined_list):
                        g.z0 = z0_refined
                        g.metadata["sdedit_strength"] = sdedit_strength

        self.population.generation_number += 1
        self.population.current_generation = next_gen[:self.population_size]
        self.population.history.append([g.clone() for g in self.population.current_generation])

        print(f"  VAE デコードで音声を生成中...")
        waveforms = self._decode_z0_batch(
            [g.z0.to(self.device) for g in self.population.current_generation]
        )
        return list(zip(self.population.current_generation, waveforms))

    # ------------------------------------------------------------------
    # 公開 API（モードディスパッチ）
    # ------------------------------------------------------------------

    def initialize_population(
        self,
        prompt: Optional[str] = None,
        variation_strength: float = 0.3,
        ga_mode: Optional[str] = None,
        base_noise_seed: Optional[int] = None,
    ) -> List[Tuple]:
        """
        初期個体群を生成する。ga_mode を渡すと self.ga_mode を上書きできる。

        Args:
            prompt: 初期プロンプト (None で無条件生成)
            variation_strength: 未使用（互換性のため残存）
            ga_mode: "latent" または "transform"（省略時は self.ga_mode を使用）
            base_noise_seed: 変換行列モード用ベースノイズの seed
        """
        if ga_mode is not None:
            self.ga_mode = ga_mode
            # モード切替時はベースノイズをリセット
            self._shared_base_noise = None

        if self.ga_mode == "conditioning":
            return self.initialize_population_conditioning(prompt=prompt or "")

        if self.ga_mode == "transform":
            return self.initialize_population_transform(
                prompt=prompt, base_noise_seed=base_noise_seed
            )

        if self.ga_mode == "z0":
            return self.initialize_population_z0(prompt=prompt)

        print(f"[潜在ノイズモード] 第{self.population.generation_number}世代の個体群を生成中...")
        
        if prompt is None:
            prompt = ""
            print("ランダム（無条件）生成")
        else:
            print(f"プロンプト: '{prompt}'")
        
        # プロンプトベースの初期個体群を生成
        if prompt:
            # テキスト条件付けベクトルを取得（全個体で共通）
            # 通常のAudioLDMと同じ方法：単一テキストは2つ複製してから1つ取る
            text_embedding = self.latent_diffusion.cond_stage_model([prompt, prompt])
            text_embedding = text_embedding[0:1]
            
            genotypes = []
            for _ in range(self.population_size):
                # 各個体に完全に独立したseedを生成
                seed = np.random.randint(0, 2**32 - 1)

                # seedを使用してランダムノイズを生成
                generator = torch.Generator(device=self.device).manual_seed(seed)
                latent_noise = torch.randn((1,) + self.latent_shape, device=self.device, generator=generator)

                genotype = AudioGenotype(
                    latent_noise=latent_noise,
                    conditioning=text_embedding,  # 条件付けは共通（ノイズを加えない）
                    seed=seed,
                    metadata={"initialization": "from_prompt", "prompt": prompt}
                )
                genotype.generation = self.population.generation_number
                genotypes.append(genotype)
        else:
            # ランダム初期化（各個体に完全に独立したランダムノイズ）
            genotypes = []
            for _ in range(self.population_size):
                # 各個体に完全に独立したseedを生成
                seed = np.random.randint(0, 2**32 - 1)
                
                # seedを使用してランダムノイズを生成
                generator = torch.Generator(device=self.device).manual_seed(seed)
                latent_noise = torch.randn((1,) + self.latent_shape, device=self.device, generator=generator)
                
                genotype = AudioGenotype(
                    latent_noise=latent_noise,
                    conditioning=None,
                    seed=seed,
                    metadata={"initialization": "random", "prompt": prompt}
                )
                genotype.generation = self.population.generation_number
                genotypes.append(genotype)
        
        self.population.current_generation = genotypes
        self.population.history.append([g.clone() for g in genotypes])

        x_T_list = [g.latent_noise.to(self.device) for g in genotypes]
        waveforms = self._sample_audio_batch(x_T_list, text=prompt)
        return list(zip(genotypes, waveforms))

    def evolve_population(
        self,
        selected_indices: List[int],
        mutation_rate: float = 0.3,
        mutation_strength: float = 0.15,
        elite_count: int = 1,
        fresh_count: int = 1,
        crossover_mode: str = "z0",
        sdedit_strength: float = 0.0,
    ) -> List[Tuple]:
        """選択された個体から次世代を生成する（モードに応じてディスパッチ）。"""
        if self.ga_mode == "conditioning":
            results, _convergence_info = self.evolve_population_conditioning(
                selected_indices,
                elite_count=elite_count,
            )
            return results

        if self.ga_mode == "transform":
            return self.evolve_population_transform(
                selected_indices,
                mutation_strength=mutation_strength,
                elite_count=elite_count,
            )

        if self.ga_mode == "z0":
            return self.evolve_population_z0(
                selected_indices,
                mutation_strength=mutation_strength,
                elite_count=elite_count,
                fresh_count=fresh_count,
                crossover_mode=crossover_mode,
                sdedit_strength=sdedit_strength,
            )

        print(f"\n[潜在ノイズモード] 第{self.population.generation_number + 1}世代を生成中...")
        print(f"選択された個体: {selected_indices}")

        if not selected_indices:
            raise ValueError("少なくとも1つの個体を選択してください")

        selected = [self.population.current_generation[i] for i in selected_indices]
        prompt = selected[0].metadata.get("prompt", "")

        from audioldm.iec import crossover_slerp, mutate_gaussian

        next_generation = []

        for i in range(min(elite_count, len(selected))):
            elite = selected[i].clone()
            elite.generation = self.population.generation_number + 1
            elite.metadata["elite"] = True
            elite.metadata["elite_parent_pop_index"] = selected_indices[i]
            next_generation.append(elite)

        while len(next_generation) < self.population_size:
            if len(selected) == 1:
                child = mutate_gaussian(selected[0], mutation_rate, mutation_strength)
                child.metadata["parent_pop_index"] = selected_indices[0]
            else:
                pair_idx = np.random.choice(len(selected), size=2, replace=False)
                parent1, parent2 = selected[pair_idx[0]], selected[pair_idx[1]]
                alpha = np.random.uniform(0.3, 0.7)
                child = crossover_slerp(parent1, parent2, alpha)
                child.metadata["crossover_parent1_pop_index"] = selected_indices[pair_idx[0]]
                child.metadata["crossover_parent2_pop_index"] = selected_indices[pair_idx[1]]
                if np.random.random() < mutation_rate:
                    child = mutate_gaussian(child, 1.0, mutation_strength)
                    child.metadata.update({
                        "operation": "crossover_then_mutate",
                        "crossover_type": "slerp",
                        "crossover_parent1_pop_index": selected_indices[pair_idx[0]],
                        "crossover_parent2_pop_index": selected_indices[pair_idx[1]],
                        "crossover_parent1_seed": parent1.seed,
                        "crossover_parent2_seed": parent2.seed,
                        "crossover_alpha": alpha,
                    })

            child.generation = self.population.generation_number + 1
            child.metadata["prompt"] = prompt
            next_generation.append(child)

        self.population.generation_number += 1
        self.population.current_generation = next_generation[:self.population_size]
        self.population.history.append([g.clone() for g in self.population.current_generation])

        genotypes = self.population.current_generation
        x_T_list = [g.latent_noise.to(self.device) for g in genotypes]
        waveforms = self._sample_audio_batch(x_T_list, text=prompt)
        return list(zip(genotypes, waveforms))
    
    def save_generation_audio(
        self,
        generation_results: List[Tuple[AudioGenotype, np.ndarray]],
        output_dir: str,
        prefix: str = "gen"
    ) -> List[str]:
        """
        世代の音声を保存
        
        Args:
            generation_results: (遺伝子型, 音声波形) のリスト
            output_dir: 出力ディレクトリ
            prefix: ファイル名のプレフィックス
        
        Returns:
            保存されたファイルパスのリスト
        """
        os.makedirs(output_dir, exist_ok=True)
        
        saved_paths = []
        gen_num = self.population.generation_number
        
        for i, (_, waveform) in enumerate(generation_results):
            filename = f"{prefix}_gen{gen_num:03d}_ind{i:02d}.wav"
            filepath = os.path.join(output_dir, filename)
            
            # waveformの形状: (batch, samples) または (batch, 1, samples)
            # モノラル音声として保存
            if len(waveform.shape) == 3:
                # (batch, 1, samples) -> (samples,)
                audio_data = waveform[0, 0, :]
            elif len(waveform.shape) == 2:
                # (batch, samples) -> (samples,)
                audio_data = waveform[0, :]
            else:
                # すでに1次元の場合
                audio_data = waveform
            
            sf.write(filepath, audio_data, samplerate=16000)
            saved_paths.append(filepath)
        
        print(f"音声を保存しました: {output_dir}")
        return saved_paths
    
    # ------------------------------------------------------------------
    # スタイル転送 公開 API
    # ------------------------------------------------------------------

    def initialize_style_transfer_population(
        self,
        audio_a_path: str,
        audio_b_path: str,
        base_prompt: str = "",
        top_k_styles: int = 5,
        population_size: Optional[int] = None,
        noise_strength_range: Tuple[float, float] = (0.1, 0.4),
        guidance_scale_range: Tuple[float, float] = (3.0, 10.0),
        z0_content: Optional[torch.Tensor] = None,
        style_words_override: Optional[List[str]] = None,
    ) -> Tuple[List[Tuple[StyleTransferGenotype, np.ndarray]], List[Tuple[str, float]]]:
        """スタイル転送の初期個体群を生成する。

        1. 音声Aを VAE エンコードして z0_content を取得
        2. 音声Bの CLAP スコアでスタイル語をランキング
        3. 合成プロンプトを構築
        4. noise_strength / guidance_scale を等間隔サンプリングして個体群を初期化
        5. SDEdit で全個体の音声を生成

        Returns:
            (results, ranked_words):
                results: List[(genotype, waveform)]
                ranked_words: [(word, score), ...] — UI 表示用
        """
        pop_size = population_size or self.population_size

        print("音声Aをエンコード中...")
        if z0_content is None:
            z0_content = self.encode_audio_to_z0(audio_a_path)

        print("音声Bのスタイル語をCLAPでランキング中...")
        ranked_words = self.rank_style_words(
            audio_b_path, style_words=style_words_override, top_k=top_k_styles)
        style_prompt = self.build_style_prompt(base_prompt, ranked_words)
        print(f"スタイルプロンプト: {style_prompt}")

        ns_vals = np.linspace(noise_strength_range[0], noise_strength_range[1], pop_size)
        gs_vals = np.linspace(guidance_scale_range[0], guidance_scale_range[1], pop_size)
        np.random.shuffle(gs_vals)

        genotypes = []
        for i in range(pop_size):
            g = StyleTransferGenotype(
                z0_content=z0_content,
                style_prompt=style_prompt,
                noise_strength=float(ns_vals[i]),
                guidance_scale=float(gs_vals[i]),
                seed=int(np.random.randint(0, 2**32 - 1)),
                mask_start=0.0,
                mask_end=1.0,
                metadata={"prompt": style_prompt, "initialization": "style_transfer"},
            )
            genotypes.append(g)

        self.population.current_generation = genotypes
        self.population.generation_number = 0
        self.population.history = [list(genotypes)]

        print(f"スタイル転送音声を生成中 (個体数={pop_size})...")
        waveforms = self.generate_style_transfer_audio_batch(genotypes)
        results = list(zip(genotypes, waveforms))
        return results, ranked_words

    def evolve_style_transfer_population(
        self,
        selected_indices: List[int],
        mutation_noise_sigma: float = 0.05,
        mutation_gs_sigma: float = 1.0,
        mutation_mask_sigma: float = 0.05,
        elite_count: int = 1,
    ) -> List[Tuple[StyleTransferGenotype, np.ndarray]]:
        """スタイル転送個体群を進化させる。

        選択個体から crossover_style_transfer + mutate_style_transfer で次世代を生成。
        """
        current = self.population.current_generation
        if not selected_indices:
            selected_indices = list(range(min(2, len(current))))

        parents = [current[i] for i in selected_indices if i < len(current)]
        if not parents:
            parents = [current[0]]

        next_gen: List[StyleTransferGenotype] = []

        for i in range(min(elite_count, len(parents))):
            elite = parents[i].clone()
            elite.metadata["elite"] = True
            next_gen.append(elite)

        while len(next_gen) < self.population_size:
            if len(parents) == 1:
                child = mutate_style_transfer(
                    parents[0], noise_sigma=mutation_noise_sigma,
                    gs_sigma=mutation_gs_sigma, mask_sigma=mutation_mask_sigma)
            else:
                p1, p2 = parents[np.random.randint(len(parents))], \
                          parents[np.random.randint(len(parents))]
                child = crossover_style_transfer(p1, p2)
                child = mutate_style_transfer(
                    child, noise_sigma=mutation_noise_sigma,
                    gs_sigma=mutation_gs_sigma, mask_sigma=mutation_mask_sigma)
            next_gen.append(child)

        next_gen = next_gen[:self.population_size]
        self.population.generation_number += 1
        self.population.current_generation = next_gen
        self.population.history.append(list(next_gen))

        waveforms = self.generate_style_transfer_audio_batch(next_gen)
        return list(zip(next_gen, waveforms))

    def get_generation_info(self) -> Dict:
        """
        現在の世代情報を取得
        
        Returns:
            世代情報の辞書
        """
        return {
            "generation_number": self.population.generation_number,
            "population_size": self.population_size,
            "history_length": len(self.population.history),
            "best_count": len(self.population.best_individuals)
        }


def run_iec_session(
    prompt: Optional[str] = None,
    model_name: str = "audioldm-s-full-v2",
    population_size: int = 6,
    duration: float = 5.0,
    output_dir: str = "./output/iec_session",
    max_generations: int = 10
):
    """
    IECセッションを実行 (CLIベース)
    
    Args:
        prompt: 初期プロンプト
        model_name: モデル名
        population_size: 個体数
        duration: 音声長
        output_dir: 出力ディレクトリ
        max_generations: 最大世代数
    """
    # システムの初期化
    iec_system = AudioLDM_IEC(
        model_name=model_name,
        population_size=population_size,
        duration=duration
    )
    
    # 初期個体群を生成
    results = iec_system.initialize_population(prompt=prompt)
    
    # 音声を保存
    saved_paths = iec_system.save_generation_audio(
        results,
        output_dir=output_dir,
        prefix="initial"
    )
    
    print(f"\n初期個体群を生成しました: {len(saved_paths)}個")
    print("音声を聴いて、気に入った個体の番号を選択してください (スペース区切り)")
    
    for _ in range(max_generations):
        print(f"\n--- 第{iec_system.population.generation_number}世代 ---")
        for i, path in enumerate(saved_paths):
            print(f"  [{i}] {os.path.basename(path)}")
        
        # ユーザー入力
        user_input = input("\n選択する個体番号 (例: 0 2 3): ").strip()
        
        if user_input.lower() in ['q', 'quit', 'exit']:
            print("セッションを終了します。")
            break
        
        try:
            selected_indices = [int(x) for x in user_input.split()]
            
            if len(selected_indices) == 0:
                print("少なくとも1つの個体を選択してください。")
                continue
            
            # 次世代を生成
            results = iec_system.evolve_population(selected_indices)
            
            # 音声を保存
            saved_paths = iec_system.save_generation_audio(
                results,
                output_dir=output_dir,
                prefix=f"gen{iec_system.population.generation_number}"
            )
            
        except ValueError as e:
            print(f"エラー: {e}")
            continue
    
    # 履歴を保存
    history_path = os.path.join(output_dir, "iec_history.json")
    iec_system.population.save_history(history_path)
    
    print(f"\n=== IECセッション完了 ===")
    print(f"総世代数: {iec_system.population.generation_number}")
    print(f"出力ディレクトリ: {output_dir}")


if __name__ == "__main__":
    # テスト実行
    run_iec_session(
        prompt="爆発音",
        population_size=4,
        duration=3.0,
        max_generations=5
    )

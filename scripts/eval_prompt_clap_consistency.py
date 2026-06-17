#!/usr/bin/env python3
"""
プロンプトプール CLAP自己整合スコア評価

各プロンプトについて AudioLDM で音声を生成し、生成音声の CLAP audio embedding と
プロンプトの CLAP text embedding のコサイン類似度（自己整合スコア）を計測する。

スコアが高いプロンプト = AudioLDM がそのプロンプトを「理解して」生成できている
スコアが低いプロンプト = AudioLDM が苦手なプロンプト（プールから除外/重み低減の候補）

オプションで CLAP空間上のプロンプト分布を2次元PCAでマップ化し、スコアで色付けする
（--map）。c_base からどの方向が「得意エリア」かを把握する手がかりになる。

Usage:
  NUMBA_CACHE_DIR=/tmp/numba_cache HF_HOME=/tmp/huggingface_cache \\
    python scripts/eval_prompt_clap_consistency.py [--model audioldm-s-full-v2] \\
        [--n_samples 1] [--ddim_steps 200] [--limit 10] [--map]

出力:
  scripts/outputs/prompt_clap_consistency/
    results.csv
    results.json
    clap_map.png        (--map 指定時)
    audio/{category}__{slug}__s{i}.wav  (--no_save_audio 指定時は省略)
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from typing import List

import numpy as np

sys.path.insert(0, ".")

import importlib.util
_pp_path = os.path.join(os.path.dirname(__file__), "..", "audioldm", "prompt_pool.py")
_spec = importlib.util.spec_from_file_location("prompt_pool_raw", _pp_path)
_pp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pp)

CATEGORIES = {
    "EMOTION_MOOD": _pp.EMOTION_MOOD,
    "INSTRUMENTS": _pp.INSTRUMENTS,
    "GENRES": _pp.GENRES,
    "ENVIRONMENTS": _pp.ENVIRONMENTS,
    "TEXTURES": _pp.TEXTURES,
    "COMPOUND_PHRASES": _pp.COMPOUND_PHRASES,
}


@dataclass
class PromptResult:
    category: str
    prompt: str
    scores: List[float] = field(default_factory=list)
    audio_paths: List[str] = field(default_factory=list)

    @property
    def mean_score(self) -> float:
        return float(np.mean(self.scores))

    @property
    def std_score(self) -> float:
        return float(np.std(self.scores))

    @property
    def min_score(self) -> float:
        return float(np.min(self.scores))

    @property
    def max_score(self) -> float:
        return float(np.max(self.scores))


def slugify(prompt: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")
    return s[:40]


def main():
    parser = argparse.ArgumentParser(description="プロンプトプール CLAP自己整合スコア評価")
    parser.add_argument("--model", default="audioldm-s-full-v2", help="AudioLDM モデル名")
    parser.add_argument("--duration", type=float, default=5.0, help="生成音声の長さ(秒)")
    parser.add_argument("--ddim_steps", type=int, default=200, help="DDIMステップ数")
    parser.add_argument("--n_samples", type=int, default=1, help="プロンプトごとの生成サンプル数")
    parser.add_argument("--seed", type=int, default=42, help="x_T生成用のベースseed")
    parser.add_argument("--limit", type=int, default=0, help="評価するプロンプト数の上限（0=全件）")
    parser.add_argument(
        "--categories", default="", help="評価するカテゴリをカンマ区切りで指定（空欄=全カテゴリ）"
    )
    parser.add_argument(
        "--out_dir", default="scripts/outputs/prompt_clap_consistency", help="出力ディレクトリ"
    )
    parser.add_argument("--no_save_audio", action="store_true", help="生成音声を保存しない")
    parser.add_argument("--map", action="store_true", help="CLAP空間PCAマップを生成する")
    parser.add_argument("--top_k", type=int, default=10, help="上位/下位サマリの表示件数")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    audio_dir = os.path.join(args.out_dir, "audio")
    if not args.no_save_audio:
        os.makedirs(audio_dir, exist_ok=True)

    categories = CATEGORIES
    if args.categories:
        wanted = {c.strip() for c in args.categories.split(",")}
        categories = {k: v for k, v in CATEGORIES.items() if k in wanted}
        if not categories:
            raise ValueError(f"未知のカテゴリ指定: {args.categories} (選択肢: {list(CATEGORIES)})")

    prompts = [(cat, p) for cat, plist in categories.items() for p in plist]
    if args.limit > 0:
        prompts = prompts[: args.limit]

    print(f"{'='*60}")
    print("プロンプトプール CLAP自己整合スコア評価")
    print(f"  model={args.model}  n_prompts={len(prompts)}  n_samples={args.n_samples}"
          f"  ddim_steps={args.ddim_steps}")
    print(f"{'='*60}")

    from audioldm.iec import ConditioningGenotype
    from audioldm.iec_pipeline import AudioLDM_IEC

    pipeline = AudioLDM_IEC(
        model_name=args.model,
        ga_mode="conditioning",
        duration=args.duration,
        ddim_steps=args.ddim_steps,
        population_size=1,
    )
    pipeline._rng = np.random.default_rng(args.seed)

    results: List[PromptResult] = []
    for idx, (category, prompt) in enumerate(prompts):
        # cond_stage_model([text, text]) は unconditional_prob による条件ドロップアウトが
        # かかるため、_encode_text_single で得た embedding を ConditioningGenotype 経由で
        # 渡し、各サンプルが必ずプロンプト条件で生成されるようにする。
        c_base = pipeline._encode_text_single(prompt)
        genotypes = []
        for _ in range(args.n_samples):
            x_T, seed = pipeline._make_x_T()
            genotypes.append(ConditioningGenotype(
                embedding=c_base.clone(), x_T=x_T, source_prompt=prompt, seed=seed))
        waveforms = pipeline._generate_audio_batch_conditioning(genotypes)

        r = PromptResult(category=category, prompt=prompt)
        for i, wf in enumerate(waveforms):
            score = pipeline.compute_clap_audio_text_similarity(wf, prompt)
            r.scores.append(score)
            if not args.no_save_audio:
                path = os.path.join(audio_dir, f"{category}__{slugify(prompt)}__s{i}.wav")
                audio_data = wf[0, 0, :] if wf.ndim == 3 else wf[0, :]
                import soundfile as sf
                sf.write(path, audio_data, samplerate=16000)
                r.audio_paths.append(path)
        results.append(r)
        print(f"  [{idx+1}/{len(prompts)}] ({category}) '{prompt}': "
              f"score={r.mean_score:.4f} (±{r.std_score:.4f})")

    # ------------------------------------------------------------------
    # CSV / JSON 出力
    # ------------------------------------------------------------------
    ranked = sorted(results, key=lambda r: r.mean_score, reverse=True)

    csv_path = os.path.join(args.out_dir, "results.csv")
    import csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "category", "prompt", "mean_score", "std_score",
                          "min_score", "max_score", "n_samples", "audio_paths"])
        for rank, r in enumerate(ranked, start=1):
            writer.writerow([rank, r.category, r.prompt, f"{r.mean_score:.6f}",
                              f"{r.std_score:.6f}", f"{r.min_score:.6f}", f"{r.max_score:.6f}",
                              len(r.scores), ";".join(r.audio_paths)])
    print(f"\nCSV出力: {csv_path}")

    json_path = os.path.join(args.out_dir, "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in ranked], f, ensure_ascii=False, indent=2)
    print(f"JSON出力: {json_path}")

    # ------------------------------------------------------------------
    # サマリ
    # ------------------------------------------------------------------
    print(f"\n--- 上位{args.top_k}件（AudioLDMが得意なプロンプト） ---")
    for r in ranked[:args.top_k]:
        print(f"  {r.mean_score:.4f}  [{r.category}] {r.prompt}")

    print(f"\n--- 下位{args.top_k}件（AudioLDMが苦手なプロンプト） ---")
    for r in ranked[-args.top_k:][::-1]:
        print(f"  {r.mean_score:.4f}  [{r.category}] {r.prompt}")

    print("\n--- カテゴリ別平均スコア ---")
    for cat in categories:
        cat_scores = [r.mean_score for r in results if r.category == cat]
        if cat_scores:
            print(f"  {cat:20s} mean={np.mean(cat_scores):.4f}  n={len(cat_scores)}")

    # ------------------------------------------------------------------
    # CLAP空間マップ (PCA 2D)
    # ------------------------------------------------------------------
    if args.map:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA

        embs = np.stack([
            pipeline._encode_text_single(r.prompt).squeeze().cpu().numpy() for r in results
        ])
        coords = PCA(n_components=2).fit_transform(embs)
        scores = np.array([r.mean_score for r in results])

        fig, ax = plt.subplots(figsize=(10, 8))
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=scores, cmap="viridis", s=40)
        plt.colorbar(sc, label="CLAP self-consistency score")

        # 上位/下位のプロンプトにラベルを付ける
        order = np.argsort(scores)
        for i in list(order[:5]) + list(order[-5:]):
            ax.annotate(results[i].prompt, (coords[i, 0], coords[i, 1]), fontsize=7, alpha=0.8)

        ax.set_title("AudioLDM CLAP self-consistency map (PCA of prompt text embeddings)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

        map_path = os.path.join(args.out_dir, "clap_map.png")
        fig.savefig(map_path, dpi=150, bbox_inches="tight")
        print(f"\nマップ出力: {map_path}")


if __name__ == "__main__":
    main()

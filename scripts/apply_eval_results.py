#!/usr/bin/env python3
"""
評価結果を prompt_pool.py に反映するスクリプト。

Usage:
  python scripts/apply_eval_results.py \
      --results scripts/outputs/prompt_clap_consistency_m/results.json \
      --min_score 0.25 --max_std 0.07
"""
import argparse
import json
import sys
import os

sys.path.insert(0, ".")

CANDIDATES_GENRES = [
    "trip hop beat",
    "synthwave music",
    "drum and bass music",
    "new age ambient music",
    "big band jazz",
    "baroque music",
]

CANDIDATES_COMPOUND = [
    "dark trip hop beat with heavy bass",
    "melancholic piano melody with ambient strings",
    "atmospheric dark jazz with muted trumpet",
    "driving synthesizer arpeggio with drum machine",
    "eerie theremin melody with strings",
    "upbeat marimba ensemble with bass",
]

ALL_CANDIDATES = set(CANDIDATES_GENRES + CANDIDATES_COMPOUND)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="scripts/outputs/prompt_clap_consistency_m/results.json")
    parser.add_argument("--min_score", type=float, default=0.25)
    parser.add_argument("--max_std", type=float, default=0.07)
    args = parser.parse_args()

    with open(args.results, encoding="utf-8") as f:
        results = json.load(f)

    score_map = {r["prompt"]: r for r in results}

    print("=" * 60)
    print("候補プロンプト 評価結果")
    print(f"  採用基準: mean_score >= {args.min_score} and std_score < {args.max_std}")
    print("=" * 60)

    adopt = []
    reject = []
    for prompt in sorted(ALL_CANDIDATES):
        if prompt not in score_map:
            print(f"  [未評価] {prompt}")
            continue
        r = score_map[prompt]
        mean = sum(r["scores"]) / len(r["scores"])
        std = (sum((s - mean) ** 2 for s in r["scores"]) / len(r["scores"])) ** 0.5
        ok = mean >= args.min_score and std < args.max_std
        status = "採用" if ok else "除外"
        cat = "GENRES" if prompt in CANDIDATES_GENRES else "COMPOUND_PHRASES"
        print(f"  [{status}] ({cat}) '{prompt}': score={mean:.4f} std={std:.4f}")
        if ok:
            adopt.append((prompt, cat, mean, std))
        else:
            reject.append((prompt, cat, mean, std))

    print(f"\n採用: {len(adopt)}件 / 除外: {len(reject)}件")
    print("\n--- 全プロンプト 新スコア (上位20) ---")
    ranked = sorted(results, key=lambda r: sum(r["scores"]) / len(r["scores"]), reverse=True)
    for r in ranked[:20]:
        mean = sum(r["scores"]) / len(r["scores"])
        std = (sum((s - mean) ** 2 for s in r["scores"]) / len(r["scores"])) ** 0.5
        marker = " ★NEW" if r["prompt"] in ALL_CANDIDATES else ""
        print(f"  {mean:.4f} (±{std:.4f})  {r['prompt']}{marker}")

    print("\n--- 全プロンプト 新スコア (下位10) ---")
    for r in ranked[-10:]:
        mean = sum(r["scores"]) / len(r["scores"])
        std = (sum((s - mean) ** 2 for s in r["scores"]) / len(r["scores"])) ** 0.5
        marker = " ★NEW" if r["prompt"] in ALL_CANDIDATES else ""
        print(f"  {mean:.4f} (±{std:.4f})  {r['prompt']}{marker}")

    # CLAP_SCORES 更新用スニペットを出力
    print("\n--- CLAP_SCORES 更新用スニペット (audioldm-m-full スコア) ---")
    print("# 採用候補分のみ抜粋:")
    for prompt, cat, mean, std in adopt:
        print(f'    "{prompt}": {mean:.6f},')


if __name__ == "__main__":
    main()

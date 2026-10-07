"""意味方向プールをコーパスから構築する。

現行の意味方向プール（`audioldm/prompt_pool.py` の PROMPT_POOL, 99語）は人手構成であり、
(a) カテゴリ構成の恣意性が残る (b) 到達範囲がその99方向に制約される、という問題がある。
本スクリプトは既存の音楽キャプションコーパスから方向語彙を構築し、これを置き換える。

**なぜ MusicCaps か**
  - AudioCaps / Clotho は環境音・音声中心（例:「A woman talks nearby as water pours」）で、
    音楽を対象とする本研究には不適。注入方向として使うと対象ジャンルからの逸脱を助長する。
  - MusicCaps は AudioSet 由来の音楽 5,521 クリップに人手で付けた記述で、
    `aspect_list` に短い音楽記述語（例: 'mellow piano melody', 'sustained strings melody'）
    が入っている。現行プールと同じ「短い方向語」の形式で、桁違いに広い。
  - 人手で書かれた実音楽の記述なので、「その方向が実在する音に対応する」ことが
    データセット側で担保される（現行の CLAP 自己整合スコアによる足切りの代替）。

**残る限界**: コーパスに変えても到達範囲が有限のプールに縛られること自体は解消しない。
ただし 99 語と数千語では質が異なり、「実データの分布からサンプルしている」という
正当化が立つ。

使用例:
    # 語彙の構築だけ（モデル不要・数秒）
    python scripts/build_semantic_pool.py --min-freq 5

    # CLAP 埋め込みまで事前計算（要モデル。ワークステーションで実行）
    python scripts/build_semantic_pool.py --min-freq 5 --embed
"""

from __future__ import annotations

import argparse
import ast
import collections
import csv
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

MUSICCAPS_URL = (
    "https://huggingface.co/datasets/google/MusicCaps/resolve/main/musiccaps-public.csv"
)
DEFAULT_OUT = Path("scripts/outputs/semantic_pool")

# 録音品質・メタ情報の語。方向として引いても「好みの音を探す」助けにならず、
# むしろ音質を劣化させる方向へ誘導するため除外する。
# （MusicCaps の高頻度語には 'low quality' 1220件, 'noisy' 628件, 'amateur recording' 518件
#   などが含まれ、素通しにすると注入の相当割合がこれらになる）
BLOCK_SUBSTRINGS = (
    "low quality", "poor quality", "bad quality", "high quality", "audio quality",
    "noisy", "noise", "amateur", "recording", "recorded", "mono", "stereo",
    "muffled", "distorted audio", "clipping", "compressed", "bitrate",
    "live performance", "studio", "microphone", "mic ", "reverb heavy",
    "background hum", "hiss", "static",
)


def download(url: str, dest: Path) -> Path:
    """未取得ならダウンロードする。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  キャッシュを使用: {dest}")
        return dest
    print(f"  ダウンロード中: {url}")
    urllib.request.urlretrieve(url, dest)
    print(f"  保存: {dest} ({dest.stat().st_size / 1024:.0f} KB)")
    return dest


def is_blocked(term: str) -> bool:
    return any(b in term for b in BLOCK_SUBSTRINGS)


def extract_aspects(csv_path: Path) -> collections.Counter:
    """MusicCaps の aspect_list から方向語を集計する。"""
    cnt: collections.Counter = collections.Counter()
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                aspects = ast.literal_eval(row.get("aspect_list", "[]"))
            except (ValueError, SyntaxError):
                continue
            for a in aspects:
                term = str(a).strip().lower()
                if term:
                    cnt[term] += 1
    return cnt


def main() -> int:
    ap = argparse.ArgumentParser(description="意味方向プールをコーパスから構築する")
    ap.add_argument("--min-freq", type=int, default=5,
                    help="この回数以上出現した方向語だけを採用する (既定: 5)")
    ap.add_argument("--max-words", type=int, default=6,
                    help="方向語の最大単語数。長すぎる記述は方向として過剰に特定的 (既定: 6)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--keep-legacy", action="store_true",
                    help="現行の99語プールを必ず含める（比較・後方互換用）")
    ap.add_argument("--embed", action="store_true",
                    help="CLAP テキスト埋め込みを事前計算して保存する（要モデル）")
    ap.add_argument("--backend", type=str, default="audioldm",
                    choices=["audioldm", "musicldm"],
                    help="埋め込みを計算するバックエンド。CLAP はモデルごとに別物なので、\n"
                         "使用するバックエンドごとに計算する必要がある")
    ap.add_argument("--model", type=str, default=None,
                    help="モデル名。未指定ならバックエンドの既定")
    ap.add_argument("--no-block", action="store_true",
                    help="録音品質語のブロックリストを適用しない")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.out_dir / "raw"

    print("=" * 64)
    print(" 意味方向プールの構築（MusicCaps）")
    print("=" * 64)

    csv_path = download(MUSICCAPS_URL, cache_dir / "musiccaps-public.csv")
    cnt = extract_aspects(csv_path)
    print(f"\n異なり方向語: {len(cnt):,}")

    terms = [t for t, c in cnt.items() if c >= args.min_freq]
    print(f"  出現 {args.min_freq} 回以上: {len(terms):,}")

    terms = [t for t in terms if len(t.split()) <= args.max_words]
    print(f"  {args.max_words} 単語以内: {len(terms):,}")

    if not args.no_block:
        blocked = [t for t in terms if is_blocked(t)]
        terms = [t for t in terms if not is_blocked(t)]
        print(f"  録音品質・メタ語を除外: -{len(blocked)} → {len(terms):,}")
        if blocked:
            print(f"    除外例: {blocked[:8]}")

    terms = sorted(set(terms))

    if args.keep_legacy:
        from audioldm.prompt_pool import PROMPT_POOL
        before = len(terms)
        terms = sorted(set(terms) | set(p.lower() for p in PROMPT_POOL))
        print(f"  既存99語を統合: +{len(terms) - before} → {len(terms):,}")

    pool_path = args.out_dir / "semantic_pool.json"
    meta = {
        "source": "google/MusicCaps (aspect_list)",
        "source_url": MUSICCAPS_URL,
        # 派生物なので出典とライセンスをファイル自身に埋めておく（CC BY-SA 4.0 は継承義務あり）
        "source_license": "CC BY-SA 4.0",
        "source_citation": (
            "Agostinelli et al., MusicLM: Generating Music From Text, arXiv:2301.11325 (2023)"
        ),
        "derived_work_license": "CC BY-SA 4.0 (inherited from MusicCaps)",
        "min_freq": args.min_freq,
        "max_words": args.max_words,
        "blocklist_applied": not args.no_block,
        "keep_legacy": args.keep_legacy,
        "size": len(terms),
        "terms": terms,
    }
    with open(pool_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\n✅ プールを保存: {pool_path}（{len(terms):,} 方向）")

    print("\n--- 採用例（頻度順 上位20）---")
    for t, c in sorted(((t, cnt[t]) for t in terms), key=lambda x: -x[1])[:20]:
        print(f"  {c:5d}  {t}")

    if not args.embed:
        print("\n埋め込みの事前計算は --embed --backend {audioldm,musicldm} で行う")
        return 0

    # ------------------------------------------------------------------
    print("\n" + "=" * 64)
    print(" CLAP 埋め込みの事前計算")
    print("=" * 64)
    import torch

    from audioldm.backends import build_backend

    model = args.model or (
        "audioldm-m-full" if args.backend == "audioldm" else "ucsd-reach/musicldm")
    be = build_backend(backend=args.backend, model_name=model,
                       device="cuda" if torch.cuda.is_available() else "cpu",
                       duration=None, guidance_scale=2.5, ddim_steps=200)
    embs = []
    for i, t in enumerate(terms, 1):
        embs.append(be.encode_text(t).cpu())
        if i % 200 == 0 or i == len(terms):
            print(f"  {i:,}/{len(terms):,}")

    from audioldm.prompt_pool import pool_embeddings_path

    emb_path = Path(pool_embeddings_path(str(pool_path), args.backend))
    # model_name と backend を必ず記録する。これが無いと読み込み側で拒否される
    # （別モデルの埋め込みを使うと注入帯の判定が静かに壊れるため fail-closed）
    torch.save({
        "terms": terms,
        "embeddings": torch.cat(embs, dim=0),
        "model_name": be.model_name,
        "backend": args.backend,
    }, emb_path)
    size_mb = emb_path.stat().st_size / 1024 ** 2
    print(f"\n✅ 埋め込みを保存: {emb_path}（{size_mb:.1f} MB）")
    print(f"   backend={args.backend} / model={be.model_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

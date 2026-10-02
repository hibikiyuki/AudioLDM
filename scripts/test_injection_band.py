"""多様性注入の条件づけ（injection_band）の検証。

変異・注入でプールから方向を引く際、c* との cos 類似度が帯 [lo, hi] に入るものだけを
候補にする挙動を、実モデルをロードせずに検証する。

オープンキャンパスで確認した第二の課題（語彙が初期の方向性と整合しない場合、
多様性の注入が対象ジャンルからの逸脱を引き起こす）への対処。

検証すること:
  1. injection_band=None（既定）では従来どおりプール全体から引く
  2. 帯を設定すると、引かれた方向の類似度がすべて帯の中に入る
  3. 帯が狭すぎて候補が足りない場合でも、必ず n 件返る（探索が停止しない）
  4. ref_embedding が None のときは帯を適用しない（初期化フェーズ用）
  5. プール埋め込みはキャッシュされ、2回目以降は再エンコードしない

使用方法:
    python scripts/test_injection_band.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from audioldm.iec_pipeline import AudioLDM_IEC
from audioldm.prompt_pool import PROMPT_POOL


def make_system(band=None) -> AudioLDM_IEC:
    """モデルをロードせずに、プールサンプリングに必要な器だけ用意する。"""
    iec = object.__new__(AudioLDM_IEC)
    iec.device = "cpu"
    iec.injection_band = band
    iec._pool_embedding_cache = {}
    iec.prompt_pool = None
    iec.encode_calls = 0

    def fake_encode(prompt: str) -> torch.Tensor:
        """プロンプト文字列から決定的なダミー embedding を作る。"""
        iec.encode_calls += 1
        h = abs(hash(prompt)) % (2**31)
        g = torch.Generator().manual_seed(h)
        return torch.randn(1, 1, 512, generator=g)

    iec._encode_text_single = fake_encode
    return iec


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = torch.nn.functional.normalize(a.flatten().float(), dim=0)
    b = torch.nn.functional.normalize(b.flatten().float(), dim=0)
    return float(torch.dot(a, b))


def check(cond: bool, label: str) -> bool:
    print(f"  {'✅' if cond else '❌'} {label}")
    return cond


def main() -> int:
    ok = True
    ref = torch.randn(1, 1, 512, generator=torch.Generator().manual_seed(0))

    # ------------------------------------------------------------------
    print("\n[1] 帯なし（既定）: プール全体から引く")
    iec = make_system(band=None)
    got = iec._sample_pool_prompts(5, ref_embedding=ref)
    ok &= check(len(got) == 5, "5件引けた")
    ok &= check(all(p in PROMPT_POOL for p in got), "全てプール由来")
    ok &= check(iec.encode_calls == 0, "帯なしでは埋め込みを計算しない（コスト増なし）")

    # ------------------------------------------------------------------
    print("\n[2] 帯あり: 引いた方向の類似度が全て帯の中に入る")
    # ダミー埋め込みでの実際の類似度分布に合わせて帯を決める
    iec2 = make_system(band=(0.0, 1.0))          # 一旦全域で分布を調べる
    embs = iec2._get_pool_embeddings()
    sims = sorted(cos(ref, embs[p]) for p in PROMPT_POOL)
    lo, hi = sims[len(sims) // 4], sims[3 * len(sims) // 4]   # 中央50%を帯にする
    print(f"  （類似度分布 min={sims[0]:.3f} max={sims[-1]:.3f} 帯=[{lo:.3f}, {hi:.3f}]）")

    iec2.injection_band = (lo, hi)
    got2 = iec2._sample_pool_prompts(8, ref_embedding=ref)
    sims2 = [cos(ref, embs[p]) for p in got2]
    ok &= check(len(got2) == 8, "8件引けた")
    ok &= check(all(lo <= s <= hi for s in sims2),
                f"全て帯の中（実測 {min(sims2):.3f}〜{max(sims2):.3f}）")
    ok &= check(len(set(got2)) == 8, "重複なし")

    # 帯の外の方向が確かに除外されていること
    excluded = [p for p in PROMPT_POOL if not (lo <= cos(ref, embs[p]) <= hi)]
    ok &= check(len(excluded) > 0 and not any(p in got2 for p in excluded),
                f"帯の外の {len(excluded)} 件は引かれていない")

    # ------------------------------------------------------------------
    print("\n[3] 帯が狭すぎる場合でも n 件返る（探索が停止しない）")
    iec3 = make_system(band=(0.999, 1.0))        # ほぼ誰も入らない帯
    got3 = iec3._sample_pool_prompts(6, ref_embedding=ref)
    ok &= check(len(got3) == 6, "候補が足りなくても 6 件返る")
    ok &= check(len(set(got3)) == 6, "重複なし")

    # ------------------------------------------------------------------
    print("\n[4] ref_embedding=None では帯を適用しない（初期化フェーズ用）")
    iec4 = make_system(band=(0.999, 1.0))
    got4 = iec4._sample_pool_prompts(6, ref_embedding=None)
    ok &= check(len(got4) == 6, "6件引けた")
    ok &= check(iec4.encode_calls == 0, "埋め込みを計算していない＝帯を適用していない")

    # ------------------------------------------------------------------
    print("\n[5] プール埋め込みのキャッシュ")
    iec5 = make_system(band=(lo, hi))
    iec5._sample_pool_prompts(3, ref_embedding=ref)
    first = iec5.encode_calls
    iec5._sample_pool_prompts(3, ref_embedding=ref)
    second = iec5.encode_calls
    ok &= check(first == len(PROMPT_POOL), f"初回はプール全件をエンコード（{first}件）")
    ok &= check(second == first, "2回目は再エンコードしない（キャッシュが効いている）")

    # ------------------------------------------------------------------
    print("\n[6] exclude が効く")
    iec6 = make_system(band=(lo, hi))
    victim = [p for p in PROMPT_POOL if lo <= cos(ref, embs[p]) <= hi][0]
    got6 = iec6._sample_pool_prompts(5, exclude=[victim], ref_embedding=ref)
    ok &= check(victim not in got6, f"除外した '{victim}' は引かれない")

    print("\n" + "=" * 56)
    print("✅ 全て成功" if ok else "❌ 失敗あり")
    print("=" * 56)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

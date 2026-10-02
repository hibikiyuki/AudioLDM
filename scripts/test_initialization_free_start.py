"""初期化フェーズ（プロンプト入力なしの開始）の検証。

`initialize_population_seed_selection(per_candidate_c=True)` と、それを受ける
`select_seed_winner` の引き継ぎロジックを、実モデルをロードせずに検証する。

検証すること:
  1. per_candidate_c=True で、候補ごとに**異なる** c（プールから引いたプロンプト）が割り当たる
  2. per_candidate_c=False（従来の x_Tガチャ）では、全候補が**同一**の c を共有する
  3. 初期化フェーズの勝者は、自身の embedding が c* として第2段階へ引き継がれる
     かつ世代番号を**進めない**（第0世代から始める）
  4. 第1段階（CLAP-IEC からガチャへ復帰）では従来どおり c* を共有し、
     世代番号を**進める**
  5. 候補数 N は IEC 個体数と独立に指定できる

使用方法:
    python scripts/test_initialization_free_start.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from audioldm.iec_pipeline import AudioLDM_IEC
from audioldm.prompt_pool import PROMPT_POOL

POP_SIZE = 6
LATENT_SHAPE = (8, 64, 16)   # (C, T, F) 相当のダミー。形だけ合っていればよい


class FakePopulation:
    def __init__(self):
        self.generation_number = 0
        self.current_generation: List = []
        self.history: List = []


def make_system() -> AudioLDM_IEC:
    """モデルをロードせずに AudioLDM_IEC の器だけを用意する。"""
    iec = object.__new__(AudioLDM_IEC)          # __init__ を呼ばない
    iec.device = "cpu"
    iec.population_size = POP_SIZE
    iec.latent_shape = LATENT_SHAPE
    iec.population = FakePopulation()
    iec.ga_mode = "conditioning"
    iec._rng = None
    iec._x_T_seed = None
    iec._conditioning_x_T = None
    iec._conditioning_c_base = None
    iec._seed_selection_population = []
    iec._seed_selection_base_embedding = None
    iec._seed_selection_per_candidate_c = False
    iec.prompt_pool = None
    iec.injection_band = None
    iec._pool_embedding_cache = {}

    # --- モデル依存部分のスタブ ---
    def fake_encode(prompt: str) -> torch.Tensor:
        """プロンプト文字列から決定的なダミー embedding を作る。"""
        h = abs(hash(prompt)) % (2**31)
        g = torch.Generator().manual_seed(h)
        return torch.randn(1, 1, 512, generator=g)

    def fake_generate(genotypes, **kwargs):
        return [np.zeros(1024, dtype=np.float32) for _ in genotypes]

    iec._encode_text_single = fake_encode
    iec._generate_audio_batch_conditioning = fake_generate

    # select_seed_winner が呼ぶ第2段階の初期化を捕捉する
    iec._captured: Dict = {}

    def fake_init_conditioning(**kwargs):
        iec._captured = dict(kwargs)
        if kwargs.get("advance_generation"):
            iec.population.generation_number += 1
        results = []
        for _ in range(POP_SIZE):
            from audioldm.iec import ConditioningGenotype
            g = ConditioningGenotype(
                embedding=torch.zeros(1, 1, 512),
                x_T=torch.zeros(1, *LATENT_SHAPE),
                source_prompt="fake",
            )
            results.append((g, np.zeros(1024, dtype=np.float32)))
        iec.population.history.append([g for g, _ in results])
        return results

    iec.initialize_population_conditioning = fake_init_conditioning
    return iec


def check(cond: bool, label: str) -> bool:
    print(f"  {'✅' if cond else '❌'} {label}")
    return cond


def main() -> int:
    ok = True

    # ------------------------------------------------------------------
    print("\n[1] 初期化フェーズ: 候補ごとに異なる c が割り当たる")
    iec = make_system()
    n = 8                                   # IEC個体数(6)と違う値をわざと使う
    results = iec.initialize_population_seed_selection(
        n_candidates=n, per_candidate_c=True)

    prompts = [g.metadata["base_prompt"] for g, _ in results]
    embeds = [g.embedding for g, _ in results]
    x_T_seeds = [g.seed for g, _ in results]

    ok &= check(len(results) == n, f"候補数 N={n} で生成された（IEC個体数 {POP_SIZE} と独立）")
    ok &= check(len(set(prompts)) == n, f"候補ごとに異なるプロンプト: {len(set(prompts))}/{n} 種")
    ok &= check(all(p in PROMPT_POOL for p in prompts), "全プロンプトが語彙プール由来")
    ok &= check(
        not torch.allclose(embeds[0], embeds[1]),
        "候補0と候補1の embedding が異なる（c が個体ごとに違う）")
    ok &= check(len(set(x_T_seeds)) == n, "x_T も個体ごとに異なる")
    ok &= check(
        all(g.metadata.get("per_candidate_c") for g, _ in results),
        "metadata に per_candidate_c=True が記録されている")
    ok &= check(
        all(g.metadata.get("pool_prompt") for g, _ in results),
        "metadata に pool_prompt（出発方向）が記録されている")
    ok &= check(iec._seed_selection_per_candidate_c is True, "システム側のフラグが立っている")

    # ------------------------------------------------------------------
    print("\n[2] 従来の x_Tガチャ: 全候補が同一の c を共有する")
    iec2 = make_system()
    results2 = iec2.initialize_population_seed_selection(
        prompt="electric guitar hard rock", n_candidates=6, per_candidate_c=False)

    prompts2 = [g.metadata["base_prompt"] for g, _ in results2]
    embeds2 = [g.embedding for g, _ in results2]

    ok &= check(len(set(prompts2)) == 1, "全候補が同一プロンプト")
    ok &= check(prompts2[0] == "electric guitar hard rock", "指定したプロンプトが使われている")
    ok &= check(
        all(torch.allclose(embeds2[0], e) for e in embeds2),
        "全候補の embedding が同一（c 固定＝音色差のみ比較できる）")
    ok &= check(
        len(set(g.seed for g, _ in results2)) == 6, "x_T は個体ごとに異なる")
    ok &= check(iec2._seed_selection_per_candidate_c is False, "フラグが立っていない")

    # ------------------------------------------------------------------
    print("\n[3] 初期化フェーズの勝者 → 第2段階へ c* を引き継ぐ / 世代は進めない")
    winner_idx = 3
    winner_embed = results[winner_idx][0].embedding.clone()
    winner_prompt = prompts[winner_idx]
    winner_seed = x_T_seeds[winner_idx]
    gen_before = iec.population.generation_number

    iec.select_seed_winner(winner_index=winner_idx)
    cap = iec._captured

    ok &= check(cap.get("base_embedding") is not None, "base_embedding が渡された")
    ok &= check(
        cap.get("base_embedding") is not None
        and torch.allclose(cap["base_embedding"], winner_embed),
        "渡された c* が勝者自身の embedding と一致する")
    ok &= check(cap.get("prompt") == winner_prompt,
                f"ラベルが勝者のプロンプト '{winner_prompt}' になっている")
    ok &= check(cap.get("x_T_seed") == winner_seed, "勝者の x_T seed が引き継がれた")
    ok &= check(cap.get("x_T_mode") == "shared", "x_T は shared で固定される")
    ok &= check(cap.get("advance_generation") is False, "世代番号を進めない")
    ok &= check(iec.population.generation_number == gen_before,
                f"世代番号が据え置き（{gen_before}）")

    # ------------------------------------------------------------------
    print("\n[4] 第1段階（ガチャ復帰）: c* を共有し、世代を進める")
    iec3 = make_system()
    c_star = torch.randn(1, 1, 512)
    results3 = iec3.initialize_population_seed_selection(
        n_candidates=5, base_embedding=c_star,
        base_prompt_label="my direction", per_candidate_c=True)  # per_candidate_c は無視される

    embeds3 = [g.embedding for g, _ in results3]
    ok &= check(
        all(torch.allclose(c_star, e) for e in embeds3),
        "base_embedding 指定時は per_candidate_c を無視して c* を共有する")
    ok &= check(iec3._seed_selection_per_candidate_c is False,
                "base_embedding 指定時はフラグが立たない")

    gen_before3 = iec3.population.generation_number
    iec3.select_seed_winner(winner_index=0)
    cap3 = iec3._captured
    ok &= check(cap3.get("advance_generation") is True, "ガチャ復帰では世代番号を進める")
    ok &= check(iec3.population.generation_number == gen_before3 + 1,
                f"世代番号が {gen_before3} → {iec3.population.generation_number}")
    ok &= check(torch.allclose(cap3["base_embedding"], c_star), "c* がそのまま引き継がれる")

    # ------------------------------------------------------------------
    print("\n[5] 候補数の独立性")
    for n_try in (2, 3, 10):
        iec4 = make_system()
        r = iec4.initialize_population_seed_selection(
            n_candidates=n_try, per_candidate_c=True)
        ok &= check(len(r) == n_try, f"N={n_try} で生成できる")

    print("\n" + "=" * 56)
    print("✅ 全て成功" if ok else "❌ 失敗あり")
    print("=" * 56)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

"""出発点 c* が初期個体群に残ることの検証（モデル不要）。

利用者が初期化／第1段階で選んだ音そのものが第2段階の候補群に
1個体として存在すること、個体数が維持されることを確かめる。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from audioldm.iec_pipeline import AudioLDM_IEC, ConditioningGenotype  # noqa: E402


class StubIEC(AudioLDM_IEC):
    """モデルを読み込まずに初期化ロジックだけを動かす。"""

    def __init__(self, population_size: int = 6) -> None:
        self.population_size = population_size
        self.device = "cpu"
        self.latent_shape = (8, 16, 16)
        self.prompt_pool = [f"dir{i:03d}" for i in range(50)]
        self.pool_embeddings = None
        self.injection_band = None
        self._x_T_seed = None
        self._rng = np.random.default_rng(0)
        self._conditioning_x_T = None
        self._conditioning_c_base = None
        from audioldm.iec import IECPopulation
        self.population = IECPopulation(population_size=population_size)

    def _encode_text_single(self, text: str) -> torch.Tensor:
        # 語ごとに決定論的な単位ベクトル
        g = torch.Generator().manual_seed(abs(hash(text)) % (2**31))
        v = torch.randn((1, 1, 512), generator=g)
        return v / v.norm(dim=-1, keepdim=True)

    def _make_x_T(self):
        seed = int(self._rng.integers(0, 2**31 - 1))
        g = torch.Generator().manual_seed(seed)
        return torch.randn((1,) + self.latent_shape, generator=g), seed

    def _generate_audio_batch_conditioning(self, genotypes):
        return [np.zeros(16, dtype=np.float32) for _ in genotypes]


def main() -> int:
    ok = True

    def check(label: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        print(f"  {'✅' if cond else '❌'} {label}{(' / ' + detail) if detail else ''}")
        ok = ok and cond

    print("[1] base_embedding あり → 個体0が出発点そのまま")
    iec = StubIEC()
    c_star = iec._encode_text_single("chosen start point")
    res = iec.initialize_population_conditioning(
        prompt="chosen start point", slerp_alpha=0.4,
        x_T_seed=1234, base_embedding=c_star, x_T_mode="shared")
    gs = [g for g, _ in res]
    check("個体数が維持される", len(gs) == 6, f"{len(gs)}個体")
    check("個体0が base_kept", gs[0].metadata.get("initialization") == "base_kept",
          str(gs[0].metadata.get("initialization")))
    cos = torch.nn.functional.cosine_similarity(
        gs[0].embedding.flatten(), c_star.flatten(), dim=0).item()
    check("個体0の embedding が c* と一致", cos > 0.9999, f"cos={cos:.6f}")
    check("残り5個体は slerp_prompt",
          all(g.metadata["initialization"] == "slerp_prompt" for g in gs[1:]))
    others = [torch.nn.functional.cosine_similarity(
        g.embedding.flatten(), c_star.flatten(), dim=0).item() for g in gs[1:]]
    check("残り5個体は c* と一致しない", all(c < 0.9999 for c in others),
          f"cos={[round(c, 3) for c in others]}")
    check("方向語が重複しない",
          len({g.metadata["rand_prompt"] for g in gs[1:]}) == 5)

    print("\n[2] keep_base=False → 従来どおり全個体が注入個体")
    iec = StubIEC()
    res = iec.initialize_population_conditioning(
        prompt="chosen start point", slerp_alpha=0.4, x_T_seed=1234,
        base_embedding=c_star, x_T_mode="shared", keep_base=False)
    gs = [g for g, _ in res]
    check("個体数が維持される", len(gs) == 6, f"{len(gs)}個体")
    check("base_kept が無い",
          all(g.metadata["initialization"] == "slerp_prompt" for g in gs))

    print("\n[3] 出発点なし（自由探索）→ keep_base は無効")
    iec = StubIEC()
    res = iec.initialize_population_conditioning(
        prompt="", slerp_alpha=0.4, x_T_seed=1234, x_T_mode="shared")
    gs = [g for g, _ in res]
    check("個体数が維持される", len(gs) == 6, f"{len(gs)}個体")
    check("全個体が注入個体",
          all(g.metadata["initialization"] == "slerp_prompt" for g in gs))

    print("\n[4] x_T_mode='inherit' でも個体0は成立する")
    iec = StubIEC()
    res = iec.initialize_population_conditioning(
        prompt="chosen start point", slerp_alpha=0.4, x_T_seed=1234,
        base_embedding=c_star, x_T_mode="inherit")
    gs = [g for g, _ in res]
    check("個体数が維持される", len(gs) == 6, f"{len(gs)}個体")
    check("個体0に x_T が割り当てられている", gs[0].x_T is not None)
    check("x_T が個体ごとに異なる",
          len({int(g.seed) for g in gs}) == 6, f"{[int(g.seed) for g in gs]}")

    print("\n" + "=" * 56)
    print("✅ 全て成功" if ok else "❌ 失敗あり")
    print("=" * 56)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

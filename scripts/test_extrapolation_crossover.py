"""第2段階の外挿交叉の検証（モデル不要）。

- slerp_conditioning が外挿でも単位ノルムを保つ（ほぼ平行の枝を含む）
- 外挿の子は 非選択 → 選択 の測地線上で、選択個体の先にある
- 選択1体でも外挿が成立する／全個体選択なら内挿にフォールバックする
- cos(子, c*) の下限で alpha が縮む
- crossover_type="interpolate" は従来どおり
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from audioldm.iec import slerp_conditioning  # noqa: E402
from test_initialization_keep_base import StubIEC  # noqa: E402


def unit(v: torch.Tensor) -> torch.Tensor:
    return F.normalize(v.flatten().float(), dim=0)


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(unit(a), unit(b)))


def angle(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(np.degrees(np.arccos(np.clip(cos(a, b), -1.0, 1.0))))


def main() -> int:
    ok = True

    def check(label: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        print(f"  {'✅' if cond else '❌'} {label}{(' / ' + detail) if detail else ''}")
        ok = ok and cond

    print("[1] slerp_conditioning は外挿でも単位ノルム")
    g = torch.Generator().manual_seed(0)
    a = F.normalize(torch.randn(1, 1, 512, generator=g), dim=-1)
    b = F.normalize(torch.randn(1, 1, 512, generator=g), dim=-1)
    near = F.normalize(a + 1e-4 * torch.randn(1, 1, 512, generator=g), dim=-1)
    norms = [float(slerp_conditioning(x, y, t).norm())
             for x, y in ((a, b), (a, near))
             for t in (-0.5, 0.0, 0.5, 1.0, 1.3, 1.6)]
    check("全 alpha でノルム 1", all(abs(n - 1) < 1e-5 for n in norms),
          f"min={min(norms):.6f} max={max(norms):.6f}")
    th = angle(a, b)
    v = slerp_conditioning(a, b, 1.5)
    check("alpha=1.5 は b から θ/2 先", abs(angle(v, b) - th / 2) < 0.05,
          f"θ={th:.1f}° / angle(v,b)={angle(v, b):.2f}°")
    check("alpha=1.5 は a から 1.5θ", abs(angle(v, a) - 1.5 * th) < 0.05 or 1.5 * th > 180,
          f"angle(v,a)={angle(v, a):.2f}°")
    w = slerp_conditioning(a, b, 0.4)
    old = (np.sin(0.6 * np.radians(th)) * a + np.sin(0.4 * np.radians(th)) * b) / np.sin(np.radians(th))
    check("内挿の結果は変更前の式と一致", float((w - old).abs().max()) < 1e-5)

    def setup(seed=1):
        iec = StubIEC()
        c_star = iec._encode_text_single("start point")
        iec.initialize_population_conditioning(
            prompt="start point", slerp_alpha=0.4, x_T_seed=seed,
            base_embedding=c_star, x_T_mode="shared")
        return iec, c_star

    print("\n[2] 外挿の子は 非選択 → 選択 の先にある")
    iec, c_star = setup()
    prev = [g.clone() for g in iec.population.current_generation]
    res, conv = iec.evolve_population_conditioning(
        [1, 3], p_mut=0.0, elite_count=1, random_sample_count=2,
        x_T_mode="shared", crossover_type="extrapolate",
        extrap_alpha_range=(1.2, 1.5), anchor_floor=None)
    gs = [g for g, _ in res]
    ext = [g for g in gs if g.metadata.get("operation") == "crossover_conditioning_extrapolate"]
    check("個体数 6", len(gs) == 6)
    check("外挿個体 3", len(ext) == 3, f"{len(ext)}体")
    check("収束情報に crossover_type=extrapolate", conv.get("crossover_type") == "extrapolate")
    for gg in ext:
        f, t = gg.metadata["extrap_from_pop_index"], gg.metadata["extrap_to_pop_index"]
        al = gg.metadata["crossover_alpha"]
        pa, pt = prev[f].embedding, prev[t].embedding
        th = angle(pa, pt)
        check(f"  非選択{f}→選択{t}  α={al:.2f}",
              f not in (1, 3) and t in (1, 3)
              and abs(angle(gg.embedding, pt) - (al - 1) * th) < 0.05
              and abs(angle(gg.embedding, pa) - al * th) < 0.05,
              f"θ={th:.1f}° 選択から {angle(gg.embedding, pt):.2f}° 先")
    check("外挿個体に elite フラグが残っていない",
          not any(gg.metadata.get("elite") for gg in ext))

    print("\n[3] 変異付き外挿")
    iec, _ = setup(2)
    res, _ = iec.evolve_population_conditioning(
        [0, 2], p_mut=0.7, elite_count=1, random_sample_count=2,
        mutation_mu_range=(0.15, 0.35), x_T_mode="shared", crossover_type="extrapolate")
    ops = [g.metadata.get("operation") for g, _ in res]
    check("3枠のうち2体が extrapolate_then_mutate",
          ops.count("extrapolate_then_mutate") == 2, str(ops))
    m = [g for g, _ in res if g.metadata.get("operation") == "extrapolate_then_mutate"]
    check("変異後も外挿の出所が残る",
          all("extrap_from_pop_index" in g.metadata and "mu" in g.metadata for g in m))

    print("\n[4] 選択1体でも外挿が成立する")
    iec, _ = setup(3)
    res, conv = iec.evolve_population_conditioning(
        [4], p_mut=0.0, elite_count=1, random_sample_count=2,
        x_T_mode="shared", crossover_type="extrapolate")
    ops = [g.metadata.get("operation") for g, _ in res]
    check("外挿 3体", ops.count("crossover_conditioning_extrapolate") == 3, str(ops))
    check("向かう先は全て個体4", all(
        g.metadata["extrap_to_pop_index"] == 4 for g, _ in res
        if g.metadata.get("operation") == "crossover_conditioning_extrapolate"))

    print("\n[5] 全個体選択なら内挿にフォールバック")
    iec, _ = setup(4)
    res, conv = iec.evolve_population_conditioning(
        list(range(6)), p_mut=0.0, elite_count=1, random_sample_count=2,
        x_T_mode="shared", crossover_type="extrapolate")
    ops = [g.metadata.get("operation") for g, _ in res]
    check("外挿個体なし", "crossover_conditioning_extrapolate" not in ops, str(ops))
    check("収束情報に crossover_type=interpolate", conv.get("crossover_type") == "interpolate")

    print("\n[6a] 選択個体自体が下限を下回るなら縮めない（利用者がそちらへ向かっている）")
    iec, c_star = setup(5)
    sel_cos = [cos(iec.population.current_generation[i].embedding, c_star) for i in (1, 2)]
    res, _ = iec.evolve_population_conditioning(
        [1, 2], p_mut=0.0, elite_count=1, random_sample_count=2,
        x_T_mode="shared", crossover_type="extrapolate",
        extrap_alpha_range=(1.5, 1.5), anchor_floor=0.93)
    ext = [g for g, _ in res if g.metadata.get("operation") == "crossover_conditioning_extrapolate"]
    check("選択個体の cos(·, c*) は下限 0.93 未満", all(c < 0.93 for c in sel_cos),
          str([round(c, 3) for c in sel_cos]))
    check("縮めていない", not any(g.metadata["anchor_clamped"] for g in ext)
          and all(g.metadata["crossover_alpha"] == 1.5 for g in ext))

    print("\n[6b] cos(子, c*) の下限で alpha が縮む")
    iec, c_star = setup(5)
    floor = 0.97      # 個体0（= c*）へ向かう外挿は c* を通り過ぎて下限を割る
    res, _ = iec.evolve_population_conditioning(
        [0], p_mut=0.0, elite_count=1, random_sample_count=2,
        x_T_mode="shared", crossover_type="extrapolate",
        extrap_alpha_range=(1.5, 1.5), anchor_floor=floor)
    ext = [g for g, _ in res if g.metadata.get("operation") == "crossover_conditioning_extrapolate"]
    clamped = [g for g in ext if g.metadata["anchor_clamped"]]
    check("縮んだ個体がある", len(clamped) > 0, f"{len(clamped)}/{len(ext)}体")
    check("縮んだ個体は下限を満たし alpha<1.5", all(
        cos(g.embedding, c_star) >= floor - 1e-6 and g.metadata["crossover_alpha"] < 1.5
        for g in clamped),
        str([(round(g.metadata["crossover_alpha"], 3), round(cos(g.embedding, c_star), 3))
             for g in clamped]))
    check("記録した anchor_cos が実測と一致", all(
        abs(g.metadata["anchor_cos"] - cos(g.embedding, c_star)) < 1e-4 for g in ext))

    print("\n[7] crossover_type='interpolate' は従来どおり")
    iec, _ = setup(6)
    res, conv = iec.evolve_population_conditioning(
        [1, 3], p_mut=0.0, elite_count=1, random_sample_count=1,
        x_T_mode="shared", crossover_type="interpolate")
    ops = [g.metadata.get("operation") for g, _ in res]
    check("内挿 4体", ops.count("crossover_conditioning_slerp") == 4, str(ops))
    check("外挿なし", "crossover_conditioning_extrapolate" not in ops)

    print("\n" + "=" * 56)
    print("✅ 全て成功" if ok else "❌ 失敗あり")
    print("=" * 56)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

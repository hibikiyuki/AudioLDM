#!/usr/bin/env python3
"""
z0 IEC 機能テスト — モデル不要、純 PyTorch で動作

検証項目:
  1. 遺伝子演算子の正確性 (clone / mutate / crossover / slerp)
  2. 個体群多様性の維持 (fresh_count の有無で比較)
  3. 交叉の有効性 (補間性・空間的分散)
  4. 変異の有効性 (強度とシフト量の比例関係)
  5. 新鮮注入メタデータの確認

Usage:
    python scripts/test_z0_iec.py
"""

import sys
import math
import numpy as np
import torch

sys.path.insert(0, ".")

# audioldm/__init__.py は pipeline.py を経由して librosa を import するが、
# この環境では numba キャッシュエラーが発生する。
# iec.py は純粋な torch ロジックのみなので直接ファイルから import する。
import importlib.util as _ilu

def _load_module(name: str, path: str):
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_iec = _load_module("audioldm.iec", "audioldm/iec.py")
LatentZ0Genotype    = _iec.LatentZ0Genotype
crossover_z0_slerp  = _iec.crossover_z0_slerp
mutate_z0_gaussian  = _iec.mutate_z0_gaussian
slerp               = _iec.slerp
IECPopulation       = _iec.IECPopulation

# -----------------------------------------------------------------
# テスト設定
# -----------------------------------------------------------------
# AudioLDM-S の典型的な潜在空間 (C=8, T=256, F=16) より小さい値でも
# ロジックは同一なので高速化のため縮小する
LATENT_SHAPE = (8, 32, 16)   # (C, T, F)
POP_SIZE = 6
N_GENERATIONS = 8
torch.manual_seed(0)
np.random.seed(0)

PASS_COUNT = 0
FAIL_COUNT = 0


# -----------------------------------------------------------------
# ユーティリティ
# -----------------------------------------------------------------

def report(label: str, passed: bool, detail: str = "") -> bool:
    global PASS_COUNT, FAIL_COUNT
    status = "PASS" if passed else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {label}{suffix}")
    if passed:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1
    return passed


def mean_pairwise_cosine_dist(genotypes: list) -> float:
    """個体群の平均ペアワイズコサイン距離 (多様性指標、0=同一, 1=直交, 2=逆向き)"""
    vecs = [g.z0.reshape(-1).float() for g in genotypes]
    n = len(vecs)
    if n < 2:
        return 0.0
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = torch.dot(vecs[i], vecs[j]) / (
                torch.norm(vecs[i]) * torch.norm(vecs[j]) + 1e-8
            )
            dists.append(1.0 - sim.item())
    return float(np.mean(dists))


def mean_pairwise_l2(genotypes: list) -> float:
    vecs = [g.z0.reshape(-1).float() for g in genotypes]
    n = len(vecs)
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(torch.norm(vecs[i] - vecs[j]).item())
    return float(np.mean(dists)) if dists else 0.0


def make_random_pop(n: int = POP_SIZE) -> list:
    return [
        LatentZ0Genotype(
            z0=torch.randn((1,) + LATENT_SHAPE),
            seed=np.random.randint(0, 2**31),
            metadata={"prompt": "test"},
        )
        for _ in range(n)
    ]


def simulate_evolution(
    population: list,
    selected_indices: list,
    mutation_strength: float,
    elite_count: int,
    fresh_count: int,
    n_gen: int,
) -> list:
    """
    モデルなしで evolve_population_z0 のロジックをシミュレート。
    DDIM 新鮮注入はランダム z0 で代替（同等の多様性を注入）。
    """
    history = [list(population)]
    current = list(population)

    for _ in range(n_gen):
        sel = [current[i] for i in selected_indices if i < len(current)]
        if not sel:
            sel = [current[0]]
        prompt = sel[0].metadata.get("prompt", "")

        next_gen: list = []

        # エリート保存
        for i in range(min(elite_count, len(sel))):
            e = sel[i].clone()
            e.generation += 1
            e.metadata["elite"] = True
            next_gen.append(e)

        # DDIM 新鮮注入（ランダム z0 で代替）
        actual_fresh = min(fresh_count, len(current) - len(next_gen))
        for _ in range(actual_fresh):
            g = LatentZ0Genotype(
                z0=torch.randn((1,) + LATENT_SHAPE),
                metadata={"operation": "ddim_fresh_injection", "prompt": prompt},
            )
            g.generation = current[0].generation + 1
            next_gen.append(g)

        # 交叉 + 変異で残りを埋める
        while len(next_gen) < len(current):
            if len(sel) == 1:
                child = mutate_z0_gaussian(sel[0], mutation_strength)
            else:
                p1, p2 = np.random.choice(sel, size=2, replace=False)
                alpha = np.random.uniform(0.1, 0.9)
                child = crossover_z0_slerp(p1, p2, alpha)
                child = mutate_z0_gaussian(child, mutation_strength)
            child.metadata["prompt"] = prompt
            next_gen.append(child)

        current = next_gen[: len(population)]
        history.append(list(current))

    return history


# -----------------------------------------------------------------
# Section 1: 遺伝子演算子の正確性
# -----------------------------------------------------------------

def section_operators():
    print("\n" + "=" * 60)
    print("Section 1: 遺伝子演算子の正確性")
    print("=" * 60)

    # 1-1: clone は独立したメモリを持つ
    g = LatentZ0Genotype(z0=torch.ones((1,) + LATENT_SHAPE))
    gc = g.clone()
    gc.z0[0, 0, 0, 0] = 999.0
    report("clone() は深いコピー (メモリ独立)", not torch.allclose(g.z0, gc.z0))

    # 1-2: mutate は z0 を変化させる
    g0 = LatentZ0Genotype(z0=torch.zeros((1,) + LATENT_SHAPE))
    gm = mutate_z0_gaussian(g0, mutation_strength=0.15)
    delta = torch.norm(gm.z0 - g0.z0).item()
    report("mutate_z0_gaussian: z0 が変化する", delta > 0.0, f"||Δz0||={delta:.4f}")

    # 1-3: mutation_strength と変位の比例性
    #   期待: ||Δz0|| ≈ strength × √n (Gaussianノイズの期待ノルム)
    n_elem = math.prod(LATENT_SHAPE)
    results_str = []
    all_ok = True
    for strength in [0.05, 0.15, 0.50]:
        base = LatentZ0Genotype(z0=torch.zeros((1,) + LATENT_SHAPE))
        # 複数回平均して分散を抑える
        deltas = []
        for _ in range(10):
            m = mutate_z0_gaussian(base, mutation_strength=strength)
            deltas.append(torch.norm(m.z0 - base.z0).item())
        actual = float(np.mean(deltas))
        expected = strength * math.sqrt(n_elem)
        ratio = actual / expected
        ok = 0.85 < ratio < 1.15
        all_ok = all_ok and ok
        results_str.append(f"σ={strength}: 実測={actual:.1f}/期待≈{expected:.1f}(比={ratio:.2f})")
    report("mutation_strength に比例した変位", all_ok, " | ".join(results_str))

    # 1-4: crossover_z0_slerp — 子は両親より近い距離にある
    z0_a = torch.randn((1,) + LATENT_SHAPE)
    z0_b = torch.randn((1,) + LATENT_SHAPE)
    ga = LatentZ0Genotype(z0=z0_a)
    gb = LatentZ0Genotype(z0=z0_b)
    d_ab = torch.norm(ga.z0 - gb.z0).item()
    all_ok = True
    for alpha in [0.1, 0.5, 0.9]:
        gc = crossover_z0_slerp(ga, gb, alpha=alpha)
        d_ca = torch.norm(gc.z0 - ga.z0).item()
        d_cb = torch.norm(gc.z0 - gb.z0).item()
        ok = d_ca < d_ab and d_cb < d_ab
        all_ok = all_ok and ok
    report(
        "crossover_z0_slerp: 子は両親より近い (補間性)",
        all_ok,
        f"d(A,B)={d_ab:.2f}  child(α=0.5): d_ca={torch.norm(crossover_z0_slerp(ga,gb,0.5).z0-ga.z0):.2f}, d_cb={torch.norm(crossover_z0_slerp(ga,gb,0.5).z0-gb.z0):.2f}",
    )

    # 1-5: slerp(v, v, α) = v（同一ベクトルは変化なし）
    v = torch.randn(128)
    for alpha in [0.0, 0.5, 1.0]:
        result = slerp(v, v, alpha)
        ok = torch.allclose(result, v, atol=1e-5)
        report(f"slerp(v, v, {alpha}) = v", ok)

    # 1-6: slerp の端点確認 slerp(a, b, 0)=a, slerp(a, b, 1)=b
    v0 = torch.randn(64)
    v1 = torch.randn(64)
    r0 = slerp(v0, v1, 0.0)
    r1 = slerp(v0, v1, 1.0)
    report("slerp(a, b, 0.0) ≈ a", torch.allclose(r0, v0, atol=1e-5))
    report("slerp(a, b, 1.0) ≈ b", torch.allclose(r1, v1, atol=1e-5))

    # 1-7: crossover の alpha が子の親への近さに反映される
    # alpha 小さい → 親 A 寄り、alpha 大きい → 親 B 寄り
    gc_near_a = crossover_z0_slerp(ga, gb, alpha=0.1)
    gc_near_b = crossover_z0_slerp(ga, gb, alpha=0.9)
    d_nearA_to_A = torch.norm(gc_near_a.z0 - ga.z0).item()
    d_nearB_to_A = torch.norm(gc_near_b.z0 - ga.z0).item()
    report(
        "crossover alpha=0.1 は A 寄り、alpha=0.9 は B 寄り",
        d_nearA_to_A < d_nearB_to_A,
        f"d(α=0.1, A)={d_nearA_to_A:.2f} < d(α=0.9, A)={d_nearB_to_A:.2f}",
    )


# -----------------------------------------------------------------
# Section 2: 個体群多様性の維持
# -----------------------------------------------------------------

def section_diversity():
    print("\n" + "=" * 60)
    print("Section 2: 個体群多様性の維持")
    print("=" * 60)

    pop_init = make_random_pop()
    div_init_cos = mean_pairwise_cosine_dist(pop_init)
    div_init_l2 = mean_pairwise_l2(pop_init)
    report(
        "初期個体群はランダムに多様 (cosine dist > 0.3)",
        div_init_cos > 0.3,
        f"cosine dist={div_init_cos:.4f}, L2={div_init_l2:.2f}",
    )

    # --- fresh_count=0: 1個体選択 → 収束するはず ---
    pop = make_random_pop()
    hist_no_fresh = simulate_evolution(
        pop, selected_indices=[0], mutation_strength=0.15,
        elite_count=1, fresh_count=0, n_gen=N_GENERATIONS,
    )
    div_end_no_fresh = mean_pairwise_cosine_dist(hist_no_fresh[-1])
    report(
        f"fresh_count=0, 1個体選択 → {N_GENERATIONS}世代後に多様性が低下する",
        div_end_no_fresh < div_init_cos * 0.7,
        f"初期={div_init_cos:.4f} → {N_GENERATIONS}世代後={div_end_no_fresh:.4f}",
    )

    # --- fresh_count=1: 同じ条件でも多様性を維持 ---
    pop = make_random_pop()
    hist_with_fresh = simulate_evolution(
        pop, selected_indices=[0], mutation_strength=0.15,
        elite_count=1, fresh_count=1, n_gen=N_GENERATIONS,
    )
    div_end_with_fresh = mean_pairwise_cosine_dist(hist_with_fresh[-1])
    report(
        f"fresh_count=1, 1個体選択 → {N_GENERATIONS}世代後に多様性が維持される (> 0.2)",
        div_end_with_fresh > 0.2,
        f"{N_GENERATIONS}世代後={div_end_with_fresh:.4f}",
    )

    # --- 比較 ---
    report(
        "fresh_count=1 の多様性 > fresh_count=0 の多様性",
        div_end_with_fresh > div_end_no_fresh,
        f"with={div_end_with_fresh:.4f} > no fresh={div_end_no_fresh:.4f}",
    )

    # --- 世代ごとの多様性トレース ---
    print()
    print("  [世代ごとの多様性トレース (cosine dist)]")
    print(f"  {'Gen':>4}  {'fresh=0':>9}  {'fresh=1':>9}")
    for gen in range(N_GENERATIONS + 1):
        d0 = mean_pairwise_cosine_dist(hist_no_fresh[gen])
        d1 = mean_pairwise_cosine_dist(hist_with_fresh[gen])
        bar0 = "#" * int(d0 * 30)
        bar1 = "#" * int(d1 * 30)
        print(f"  {gen:>4}  {d0:>9.4f}  {d1:>9.4f}  {bar0} / {bar1}")

    # --- 複数親選択の場合 ---
    pop = make_random_pop()
    hist_multi = simulate_evolution(
        pop, selected_indices=[0, 1, 2], mutation_strength=0.15,
        elite_count=1, fresh_count=0, n_gen=N_GENERATIONS,
    )
    div_multi = mean_pairwise_cosine_dist(hist_multi[-1])
    report(
        f"3個体選択, fresh_count=0 → 1個体選択より多様性が高い",
        div_multi > div_end_no_fresh,
        f"3親={div_multi:.4f} > 1親={div_end_no_fresh:.4f}",
    )


# -----------------------------------------------------------------
# Section 3: 交叉の有効性
# -----------------------------------------------------------------

def section_crossover():
    print("\n" + "=" * 60)
    print("Section 3: 交叉の有効性")
    print("=" * 60)

    # 3-1: 交叉子は変異子より親から遠い位置に分布する（空間的分散）
    ga = LatentZ0Genotype(z0=torch.randn((1,) + LATENT_SHAPE))
    gb = LatentZ0Genotype(z0=torch.randn((1,) + LATENT_SHAPE))

    N = 50
    mut_dists_from_a = []
    cross_dists_from_a = []
    for _ in range(N):
        gm = mutate_z0_gaussian(ga, mutation_strength=0.15)
        gc = crossover_z0_slerp(ga, gb, alpha=np.random.uniform(0.1, 0.9))
        mut_dists_from_a.append(torch.norm(gm.z0 - ga.z0).item())
        cross_dists_from_a.append(torch.norm(gc.z0 - ga.z0).item())

    mean_mut = float(np.mean(mut_dists_from_a))
    mean_cross = float(np.mean(cross_dists_from_a))
    report(
        "交叉子は変異子より親 A から遠い (より広い探索)",
        mean_cross > mean_mut,
        f"crossover avg dist={mean_cross:.2f} > mutation avg dist={mean_mut:.2f}",
    )

    # 3-2: alpha=0.5 の子は両親からほぼ等距離
    gc_mid = crossover_z0_slerp(ga, gb, alpha=0.5)
    d_ca = torch.norm(gc_mid.z0 - ga.z0).item()
    d_cb = torch.norm(gc_mid.z0 - gb.z0).item()
    ratio = min(d_ca, d_cb) / max(d_ca, d_cb)
    report(
        "alpha=0.5 の交叉子は両親からほぼ等距離 (距離比 > 0.7)",
        ratio > 0.7,
        f"d(child,A)={d_ca:.2f}, d(child,B)={d_cb:.2f}, ratio={ratio:.3f}",
    )

    # 3-3: 交叉 + 変異子の多様性 > 変異のみの多様性
    ga_pop = [LatentZ0Genotype(z0=torch.randn((1,) + LATENT_SHAPE)) for _ in range(3)]
    mut_only_children = [mutate_z0_gaussian(ga_pop[0], mutation_strength=0.15) for _ in range(POP_SIZE)]
    cross_mut_children = []
    for _ in range(POP_SIZE):
        p1, p2 = np.random.choice(ga_pop, size=2, replace=False)
        child = crossover_z0_slerp(p1, p2, alpha=np.random.uniform(0.1, 0.9))
        child = mutate_z0_gaussian(child, mutation_strength=0.15)
        cross_mut_children.append(child)

    div_mut_only = mean_pairwise_cosine_dist(mut_only_children)
    div_cross_mut = mean_pairwise_cosine_dist(cross_mut_children)
    report(
        "交叉+変異の多様性 > 変異のみの多様性",
        div_cross_mut > div_mut_only,
        f"cross+mut={div_cross_mut:.4f} > mut_only={div_mut_only:.4f}",
    )

    # 3-4: 交叉子のメタデータに両親 ID が記録される
    gc_check = crossover_z0_slerp(ga, gb, alpha=0.5)
    ok = "parent1_id" in gc_check.metadata and "parent2_id" in gc_check.metadata
    report("交叉子のメタデータに parent1_id, parent2_id が存在する", ok)


# -----------------------------------------------------------------
# Section 4: 変異の有効性
# -----------------------------------------------------------------

def section_mutation():
    print("\n" + "=" * 60)
    print("Section 4: 変異の有効性")
    print("=" * 60)

    base = LatentZ0Genotype(z0=torch.randn((1,) + LATENT_SHAPE))

    # 4-1: 異なる strength → 異なる変位 (単調増加)
    strengths = [0.02, 0.10, 0.30]
    mean_deltas = []
    for strength in strengths:
        deltas = [torch.norm(mutate_z0_gaussian(base, strength).z0 - base.z0).item() for _ in range(20)]
        mean_deltas.append(float(np.mean(deltas)))
    monotone = all(mean_deltas[i] < mean_deltas[i + 1] for i in range(len(mean_deltas) - 1))
    detail = "  ".join(f"σ={s}: {d:.2f}" for s, d in zip(strengths, mean_deltas))
    report("mutation_strength が大きいほど変位が大きい (単調性)", monotone, detail)

    # 4-2: 同じ親から複数の変異子 → 異なる結果（確率性）
    children = [mutate_z0_gaussian(base, mutation_strength=0.15) for _ in range(5)]
    z0s = [c.z0.reshape(-1) for c in children]
    all_different = all(
        not torch.allclose(z0s[i], z0s[j], atol=1e-6)
        for i in range(len(z0s)) for j in range(i + 1, len(z0s))
    )
    report("同一親からの複数変異子はすべて異なる (確率性)", all_different)

    # 4-3: 小さい strength の変異子は親の「近く」にある
    gm_small = mutate_z0_gaussian(base, mutation_strength=0.02)
    gm_large = mutate_z0_gaussian(base, mutation_strength=0.50)
    cos_small = float(torch.nn.functional.cosine_similarity(
        base.z0.reshape(1, -1), gm_small.z0.reshape(1, -1)
    ).item())
    cos_large = float(torch.nn.functional.cosine_similarity(
        base.z0.reshape(1, -1), gm_large.z0.reshape(1, -1)
    ).item())
    report(
        "小 strength の変異子は親に近い (コサイン類似度)",
        cos_small > cos_large,
        f"cos(base, mut_small)={cos_small:.4f} > cos(base, mut_large)={cos_large:.4f}",
    )

    # 4-4: 変異子のメタデータが正しく記録される
    gm = mutate_z0_gaussian(base, mutation_strength=0.15)
    ok = (
        gm.metadata.get("operation") == "mutate_z0_gaussian"
        and "parent_id" in gm.metadata
        and gm.metadata.get("mutation_strength") == 0.15
    )
    report("変異子のメタデータが正しく記録される", ok, str(gm.metadata))


# -----------------------------------------------------------------
# Section 5: 新鮮注入の確認
# -----------------------------------------------------------------

def section_fresh_injection():
    print("\n" + "=" * 60)
    print("Section 5: 新鮮注入 (fresh_count) の確認")
    print("=" * 60)

    def count_fresh(gen: list) -> int:
        return sum(
            1 for g in gen if g.metadata.get("operation") == "ddim_fresh_injection"
        )

    # 5-1: fresh_count=1 → 各世代に1体の新鮮個体
    pop = make_random_pop()
    hist = simulate_evolution(
        pop, selected_indices=[0], mutation_strength=0.15,
        elite_count=1, fresh_count=1, n_gen=N_GENERATIONS,
    )
    fresh_counts = [count_fresh(gen) for gen in hist[1:]]  # 初代は除く
    all_one = all(c == 1 for c in fresh_counts)
    report(
        f"fresh_count=1 → 全世代で新鮮個体が 1 体",
        all_one,
        f"各世代の注入数: {fresh_counts}",
    )

    # 5-2: fresh_count=0 → 新鮮個体なし
    pop = make_random_pop()
    hist_no = simulate_evolution(
        pop, selected_indices=[0], mutation_strength=0.15,
        elite_count=1, fresh_count=0, n_gen=N_GENERATIONS,
    )
    no_fresh = all(count_fresh(gen) == 0 for gen in hist_no[1:])
    report("fresh_count=0 → 新鮮個体なし", no_fresh)

    # 5-3: fresh_count=2 → 各世代に2体
    pop = make_random_pop()
    hist2 = simulate_evolution(
        pop, selected_indices=[0], mutation_strength=0.15,
        elite_count=1, fresh_count=2, n_gen=N_GENERATIONS,
    )
    fresh_counts2 = [count_fresh(gen) for gen in hist2[1:]]
    all_two = all(c == 2 for c in fresh_counts2)
    report(
        f"fresh_count=2 → 全世代で新鮮個体が 2 体",
        all_two,
        f"各世代の注入数: {fresh_counts2}",
    )

    # 5-4: エリート + fresh が population_size を超えない
    pop = make_random_pop(POP_SIZE)
    hist_check = simulate_evolution(
        pop, selected_indices=[0, 1], mutation_strength=0.15,
        elite_count=2, fresh_count=2, n_gen=3,
    )
    size_ok = all(len(gen) == POP_SIZE for gen in hist_check)
    report(
        "エリート + fresh があっても個体数は population_size を維持",
        size_ok,
        f"各世代のサイズ: {[len(g) for g in hist_check]}",
    )


# -----------------------------------------------------------------
# Section 6: 収束検知 (DIV 閾値による判定)
# -----------------------------------------------------------------

def section_convergence_detection():
    print("\n" + "=" * 60)
    print("Section 6: 収束検知")
    print("=" * 60)

    CONVERGENCE_THRESHOLD = 0.05  # cosine dist がこれ以下なら収束とみなす

    # 6-1: 同一 z0 を持つ個体群 → 収束として検知できる
    z_same = torch.randn((1,) + LATENT_SHAPE)
    converged_pop = [LatentZ0Genotype(z0=z_same.clone()) for _ in range(POP_SIZE)]
    div = mean_pairwise_cosine_dist(converged_pop)
    report(
        "同一 z0 個体群は収束として検知される (cosine dist ≈ 0)",
        div < CONVERGENCE_THRESHOLD,
        f"cosine dist = {div:.6f}",
    )

    # 6-2: fresh_count=1 → 収束から回復できる
    # 全個体が同一 z0 の状態から fresh_count=1 で進化させ、多様性が戻るか
    pop = [LatentZ0Genotype(z0=z_same.clone(), metadata={"prompt": "test"}) for _ in range(POP_SIZE)]
    hist_recover = simulate_evolution(
        pop, selected_indices=[0], mutation_strength=0.15,
        elite_count=1, fresh_count=1, n_gen=N_GENERATIONS,
    )
    div_recovered = mean_pairwise_cosine_dist(hist_recover[-1])
    report(
        f"完全収束状態から fresh_count=1 で多様性を回復 (cosine dist > 0.15)",
        div_recovered > 0.15,
        f"{N_GENERATIONS}世代後 cosine dist = {div_recovered:.4f}",
    )

    # 6-3: fresh_count=0 → 収束状態のまま脱出できない
    pop = [LatentZ0Genotype(z0=z_same.clone(), metadata={"prompt": "test"}) for _ in range(POP_SIZE)]
    hist_stuck = simulate_evolution(
        pop, selected_indices=[0], mutation_strength=0.15,
        elite_count=1, fresh_count=0, n_gen=N_GENERATIONS,
    )
    div_stuck = mean_pairwise_cosine_dist(hist_stuck[-1])
    report(
        f"完全収束状態で fresh_count=0 → {N_GENERATIONS}世代後も多様性が低い (< 0.15)",
        div_stuck < 0.15,
        f"{N_GENERATIONS}世代後 cosine dist = {div_stuck:.4f}",
    )


# -----------------------------------------------------------------
# メイン
# -----------------------------------------------------------------

def main():
    print("\n" + "=" * 60)
    print("  z0 IEC 機能テスト")
    print(f"  LATENT_SHAPE={LATENT_SHAPE}, POP_SIZE={POP_SIZE}, N_GEN={N_GENERATIONS}")
    print("=" * 60)

    section_operators()
    section_diversity()
    section_crossover()
    section_mutation()
    section_fresh_injection()
    section_convergence_detection()

    total = PASS_COUNT + FAIL_COUNT
    print("\n" + "=" * 60)
    print(f"  結果: {PASS_COUNT}/{total} PASS  ({FAIL_COUNT} FAIL)")
    print("=" * 60)

    if FAIL_COUNT > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
AudioLDM-IEC 簡易デモ・テストスクリプト

IECコア機能のテスト用スクリプト
"""

import torch
import numpy as np
from audioldm.iec import (
    AudioGenotype,
    IECPopulation,
    slerp,
    crossover_slerp,
    mutate_gaussian,
    adaptive_mutation_rate
)


def test_slerp():
    """球面線形補間のテスト"""
    print("=" * 70)
    print("Slerp（球面線形補間）のテスト")
    print("=" * 70)
    
    # テストベクトル
    v0 = torch.randn(1, 128)
    v1 = torch.randn(1, 128)
    
    # 補間パラメータ
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    print(f"v0の形状: {v0.shape}")
    print(f"v1の形状: {v1.shape}")
    print()
    
    for alpha in alphas:
        result = slerp(v0, v1, alpha)
        print(f"alpha={alpha:.2f}: 形状={result.shape}, ノルム={torch.norm(result).item():.4f}")
    
    print("✅ Slerpテスト成功\n")


def test_genotype():
    """遺伝子型クラスのテスト"""
    print("=" * 70)
    print("AudioGenotype（遺伝子型）のテスト")
    print("=" * 70)
    
    # 遺伝子型の作成
    latent_shape = (1, 8, 128, 16)
    latent_noise = torch.randn(latent_shape)
    conditioning = torch.randn(1, 77, 768)
    
    genotype = AudioGenotype(
        latent_noise=latent_noise,
        conditioning=conditioning,
        seed=42,
        metadata={"test": True}
    )
    
    print(f"ID: {genotype.id}")
    print(f"世代: {genotype.generation}")
    print(f"潜在ノイズ形状: {genotype.latent_noise.shape}")
    print(f"条件付けベクトル形状: {genotype.conditioning.shape}")
    print(f"適応度: {genotype.fitness}")
    print()
    
    # クローンのテスト
    clone = genotype.clone()
    print(f"クローンID: {clone.id}")
    print(f"元のIDと異なる: {clone.id != genotype.id}")
    
    print("✅ Genotypeテスト成功\n")


def test_crossover():
    """交叉のテスト"""
    print("=" * 70)
    print("Crossover（交叉）のテスト")
    print("=" * 70)
    
    # 親個体の作成
    latent_shape = (1, 8, 128, 16)
    
    parent1 = AudioGenotype(
        latent_noise=torch.randn(latent_shape),
        conditioning=torch.randn(1, 77, 768),
        seed=1
    )
    parent1.generation = 0
    
    parent2 = AudioGenotype(
        latent_noise=torch.randn(latent_shape),
        conditioning=torch.randn(1, 77, 768),
        seed=2
    )
    parent2.generation = 0
    
    print(f"親1 ID: {parent1.id[:20]}...")
    print(f"親2 ID: {parent2.id[:20]}...")
    print()
    
    # 交叉
    child = crossover_slerp(parent1, parent2, alpha=0.5)
    
    print(f"子 ID: {child.id[:20]}...")
    print(f"子の世代: {child.generation}")
    print(f"メタデータ: {child.metadata}")
    print(f"潜在ノイズ形状: {child.latent_noise.shape}")
    
    print("✅ Crossoverテスト成功\n")


def test_mutation():
    """突然変異のテスト"""
    print("=" * 70)
    print("Mutation（突然変異）のテスト")
    print("=" * 70)
    
    # 個体の作成
    latent_shape = (1, 8, 128, 16)
    individual = AudioGenotype(
        latent_noise=torch.randn(latent_shape),
        conditioning=torch.randn(1, 77, 768),
        seed=42
    )
    
    print(f"元の個体 ID: {individual.id[:20]}...")
    print(f"元の潜在ノイズのノルム: {torch.norm(individual.latent_noise).item():.4f}")
    print()
    
    # 突然変異
    mutant = mutate_gaussian(individual, mutation_rate=1.0, mutation_strength=0.2)
    
    print(f"変異個体 ID: {mutant.id[:20]}...")
    print(f"変異個体の世代: {mutant.generation}")
    print(f"変異後の潜在ノイズのノルム: {torch.norm(mutant.latent_noise).item():.4f}")
    print(f"メタデータ: {mutant.metadata}")
    
    # 変化量の確認
    diff = torch.norm(mutant.latent_noise - individual.latent_noise).item()
    print(f"変化量（ユークリッド距離）: {diff:.4f}")
    
    print("✅ Mutationテスト成功\n")


def test_population():
    """個体群管理のテスト"""
    print("=" * 70)
    print("IECPopulation（個体群管理）のテスト")
    print("=" * 70)
    
    # 個体群の初期化
    population = IECPopulation(population_size=4)
    print(f"個体群サイズ: {population.population_size}")
    print()
    
    # ランダム初期化
    latent_shape = (1, 8, 128, 16)
    individuals = population.initialize_random(latent_shape, device="cpu")
    
    print(f"第{population.generation_number}世代:")
    for i, ind in enumerate(individuals):
        print(f"  個体{i}: {ind.id[:20]}... (世代{ind.generation})")
    print()
    
    # 次世代の生成
    selected_indices = [0, 2]  # 個体0と2を選択
    print(f"選択された個体: {selected_indices}")
    
    next_gen = population.evolve_next_generation(
        selected_indices=selected_indices,
        mutation_rate=0.3,
        mutation_strength=0.15,
        elite_count=1
    )
    
    print(f"\n第{population.generation_number}世代:")
    for i, ind in enumerate(next_gen):
        print(f"  個体{i}: {ind.id[:20]}... (世代{ind.generation})")
        if "elite" in ind.metadata:
            print(f"    ↑ エリート個体")
    print()
    
    # ロールバックのテスト
    print("1世代ロールバック...")
    rolled_back = population.rollback_generation(steps=1)
    print(f"現在の世代: {population.generation_number}")
    print(f"個体数: {len(rolled_back)}")
    
    print("✅ Populationテスト成功\n")


def test_adaptive_mutation():
    """適応的突然変異率のテスト"""
    print("=" * 70)
    print("Adaptive Mutation Rate（適応的突然変異率）のテスト")
    print("=" * 70)
    
    generations = [0, 5, 10, 20, 50]
    convergence_scores = [0.0, 0.5, 0.8]
    
    print("世代数と収束スコアによる突然変異率の変化:")
    print()
    
    for conv in convergence_scores:
        print(f"収束スコア = {conv}")
        for gen in generations:
            rate = adaptive_mutation_rate(gen, conv)
            print(f"  世代{gen:3d}: 突然変異率 = {rate:.4f}")
        print()
    
    print("✅ Adaptive Mutation Rateテスト成功\n")


def main():
    """全テストを実行"""
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + " " * 15 + "AudioLDM-IEC コア機能テスト" + " " * 23 + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    print("\n")
    
    try:
        test_slerp()
        test_genotype()
        test_crossover()
        test_mutation()
        test_population()
        test_adaptive_mutation()
        
        print("=" * 70)
        print("✅ 全てのテストが成功しました！")
        print("=" * 70)
        print("\nIECコア機能は正常に動作しています。")
        print("次のステップ: Gradio UIを起動してください")
        print("  $ python scripts/launch_iec_gradio.py")
        print()
        
    except Exception as e:
        print("=" * 70)
        print("❌ テストが失敗しました")
        print("=" * 70)
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

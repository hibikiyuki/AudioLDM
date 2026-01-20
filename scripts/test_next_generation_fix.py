"""
次世代生成アルゴリズムの修正をテストするスクリプト
"""

import torch
import numpy as np
from audioldm.iec import AudioGenotype, crossover_slerp, mutate_gaussian, IECPopulation


def test_seed_consistency():
    """seedが適切に設定されているかをテスト"""
    print("=" * 60)
    print("テスト1: seedの一貫性テスト")
    print("=" * 60)
    
    # テスト用の親個体を作成
    device = "cpu"
    latent_shape = (8, 16, 8)
    
    # 親1
    seed1 = 42
    gen1 = torch.Generator(device=device).manual_seed(seed1)
    latent1 = torch.randn(latent_shape, generator=gen1, device=device)
    parent1 = AudioGenotype(latent_noise=latent1, seed=seed1)
    
    # 親2
    seed2 = 100
    gen2 = torch.Generator(device=device).manual_seed(seed2)
    latent2 = torch.randn(latent_shape, generator=gen2, device=device)
    parent2 = AudioGenotype(latent_noise=latent2, seed=seed2)
    
    print(f"親1のseed: {parent1.seed}")
    print(f"親2のseed: {parent2.seed}")
    
    # 交叉テスト
    child = crossover_slerp(parent1, parent2, alpha=0.5)
    print(f"\n交叉後の子のseed: {child.seed}")
    print(f"子のメタデータ: {child.metadata}")
    
    assert child.seed is not None, "❌ 子のseedがNoneです"
    assert child.seed != parent1.seed, "❌ 子のseedが親1と同じです"
    assert child.seed != parent2.seed, "❌ 子のseedが親2と同じです"
    assert "parent1_seed" in child.metadata, "❌ 親1のseedがメタデータに記録されていません"
    assert "parent2_seed" in child.metadata, "❌ 親2のseedがメタデータに記録されていません"
    
    print("✅ 交叉後のseed設定: OK")
    
    # 変異テスト
    mutant = mutate_gaussian(parent1, mutation_rate=1.0, mutation_strength=0.1)
    print(f"\n変異後の子のseed: {mutant.seed}")
    print(f"変異後のメタデータ: {mutant.metadata}")
    
    assert mutant.seed is not None, "❌ 変異後の子のseedがNoneです"
    assert mutant.seed != parent1.seed, "❌ 変異後の子のseedが親と同じです"
    assert "parent_seed" in mutant.metadata, "❌ 親のseedがメタデータに記録されていません"
    
    print("✅ 変異後のseed設定: OK")


def test_conditioning_preservation():
    """条件付けベクトルが適切に保持されているかをテスト"""
    print("\n" + "=" * 60)
    print("テスト2: 条件付けベクトルの保持テスト")
    print("=" * 60)
    
    device = "cpu"
    latent_shape = (8, 16, 8)
    conditioning_shape = (1, 768)
    
    # 条件付けベクトルを持つ親個体を作成
    latent1 = torch.randn(latent_shape, device=device)
    conditioning = torch.randn(conditioning_shape, device=device)
    parent1 = AudioGenotype(latent_noise=latent1, conditioning=conditioning, seed=42)
    
    latent2 = torch.randn(latent_shape, device=device)
    parent2 = AudioGenotype(latent_noise=latent2, conditioning=conditioning, seed=100)
    
    print(f"親の条件付けベクトル形状: {conditioning.shape}")
    print(f"親の条件付けベクトルの平均値: {conditioning.mean().item():.6f}")
    
    # 交叉テスト
    child = crossover_slerp(parent1, parent2, alpha=0.5)
    print(f"\n交叉後の子の条件付けベクトル形状: {child.conditioning.shape}")
    print(f"交叉後の子の条件付けベクトルの平均値: {child.conditioning.mean().item():.6f}")
    
    # 条件付けベクトルが親1と同じであることを確認（補間されていない）
    assert torch.allclose(child.conditioning, parent1.conditioning), "❌ 条件付けベクトルが変更されています"
    print("✅ 交叉時の条件付けベクトル保持: OK（補間されていない）")
    
    # 変異テスト
    mutant = mutate_gaussian(parent1, mutation_rate=1.0, mutation_strength=0.5)
    print(f"\n変異後の条件付けベクトル形状: {mutant.conditioning.shape}")
    print(f"変異後の条件付けベクトルの平均値: {mutant.conditioning.mean().item():.6f}")
    
    # 条件付けベクトルが変異されていないことを確認
    assert torch.allclose(mutant.conditioning, parent1.conditioning), "❌ 条件付けベクトルが変異されています"
    print("✅ 変異時の条件付けベクトル保持: OK（変異されていない）")


def test_population_seed_usage():
    """個体群初期化時にseedが使用されているかをテスト"""
    print("\n" + "=" * 60)
    print("テスト3: 個体群初期化時のseed使用テスト")
    print("=" * 60)
    
    device = "cpu"
    latent_shape = (8, 16, 8)
    population_size = 4
    
    population = IECPopulation(population_size=population_size)
    
    # 初期化
    genotypes = population.initialize_random(latent_shape, device=device)
    
    print(f"個体数: {len(genotypes)}")
    
    # 全個体がseedを持つことを確認
    for i, genotype in enumerate(genotypes):
        print(f"個体{i}: seed={genotype.seed}")
        assert genotype.seed is not None, f"❌ 個体{i}のseedがNoneです"
    
    print("✅ 全個体がseedを持っています")
    
    # seedが重複していないことを確認
    seeds = [g.seed for g in genotypes]
    unique_seeds = set(seeds)
    assert len(unique_seeds) == len(seeds), "❌ seedが重複しています"
    print("✅ 全個体のseedがユニークです")


def test_seed_based_regeneration():
    """seedを使って潜在ノイズを再生成できることをテスト"""
    print("\n" + "=" * 60)
    print("テスト4: seedベースの潜在ノイズ再生成テスト")
    print("=" * 60)
    
    device = "cpu"
    latent_shape = (8, 16, 8)
    seed = 12345
    
    # 最初の生成
    gen1 = torch.Generator(device=device).manual_seed(seed)
    latent1 = torch.randn(latent_shape, generator=gen1, device=device)
    
    # seedを使って再生成
    gen2 = torch.Generator(device=device).manual_seed(seed)
    latent2 = torch.randn(latent_shape, generator=gen2, device=device)
    
    # 完全に一致することを確認
    assert torch.allclose(latent1, latent2), "❌ 同じseedから異なる潜在ノイズが生成されました"
    print(f"seed={seed}から生成された潜在ノイズ:")
    print(f"  1回目の平均値: {latent1.mean().item():.6f}")
    print(f"  2回目の平均値: {latent2.mean().item():.6f}")
    print(f"  差分: {torch.abs(latent1 - latent2).max().item():.10f}")
    print("✅ seedベースの再生成: OK（完全に一致）")


def test_crossover_properties():
    """交叉の性質をテスト"""
    print("\n" + "=" * 60)
    print("テスト5: 交叉の性質テスト")
    print("=" * 60)
    
    device = "cpu"
    latent_shape = (8, 16, 8)
    
    # 親個体を作成
    latent1 = torch.randn(latent_shape, device=device)
    latent2 = torch.randn(latent_shape, device=device)
    parent1 = AudioGenotype(latent_noise=latent1, seed=42)
    parent2 = AudioGenotype(latent_noise=latent2, seed=100)
    
    # alpha=0の場合、親1に近いはず
    child0 = crossover_slerp(parent1, parent2, alpha=0.0)
    dist0_to_p1 = torch.norm(child0.latent_noise - parent1.latent_noise).item()
    dist0_to_p2 = torch.norm(child0.latent_noise - parent2.latent_noise).item()
    print(f"alpha=0.0: 親1への距離={dist0_to_p1:.6f}, 親2への距離={dist0_to_p2:.6f}")
    assert dist0_to_p1 < dist0_to_p2, "❌ alpha=0で親1に近くありません"
    
    # alpha=1の場合、親2に近いはず
    child1 = crossover_slerp(parent1, parent2, alpha=1.0)
    dist1_to_p1 = torch.norm(child1.latent_noise - parent1.latent_noise).item()
    dist1_to_p2 = torch.norm(child1.latent_noise - parent2.latent_noise).item()
    print(f"alpha=1.0: 親1への距離={dist1_to_p1:.6f}, 親2への距離={dist1_to_p2:.6f}")
    assert dist1_to_p2 < dist1_to_p1, "❌ alpha=1で親2に近くありません"
    
    # alpha=0.5の場合、中間のはず
    child_mid = crossover_slerp(parent1, parent2, alpha=0.5)
    dist_mid_to_p1 = torch.norm(child_mid.latent_noise - parent1.latent_noise).item()
    dist_mid_to_p2 = torch.norm(child_mid.latent_noise - parent2.latent_noise).item()
    print(f"alpha=0.5: 親1への距離={dist_mid_to_p1:.6f}, 親2への距離={dist_mid_to_p2:.6f}")
    
    print("✅ 交叉の性質: OK")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("次世代生成アルゴリズムの修正テスト")
    print("=" * 60)
    
    try:
        test_seed_consistency()
        test_conditioning_preservation()
        test_population_seed_usage()
        test_seed_based_regeneration()
        test_crossover_properties()
        
        print("\n" + "=" * 60)
        print("✅ 全てのテストが成功しました！")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ テスト失敗: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n❌ エラー: {e}")
        import traceback
        traceback.print_exc()

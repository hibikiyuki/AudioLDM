"""
Interactive Evolutionary Computation (IEC) for AudioLDM
対話型進化計算による効果音生成システム
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Dict
import json
import os
from datetime import datetime


class AudioGenotype:
    """
    音声生成のための遺伝子型クラス
    拡散モデルのノイズと条件付けベクトルを遺伝子として保持
    """
    
    def __init__(
        self,
        latent_noise: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Args:
            latent_noise: 拡散モデルの初期ノイズ (潜在空間でのノイズ)
            conditioning: 条件付けベクトル (テキストエンベディング等)
            seed: ランダムシード
            metadata: メタデータ (生成パラメータ、世代番号等)
        """
        self.latent_noise = latent_noise.clone()
        self.conditioning = conditioning.clone() if conditioning is not None else None
        self.seed = seed
        self.metadata = metadata or {}
        self.fitness = 0.0  # ユーザー評価スコア
        self.generation = 0
        self.id = self._generate_id()
    
    def _generate_id(self) -> str:
        """個体IDの生成"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"genotype_{timestamp}"
    
    def clone(self) -> 'AudioGenotype':
        """個体のクローンを作成"""
        return AudioGenotype(
            latent_noise=self.latent_noise.clone(),
            conditioning=self.conditioning.clone() if self.conditioning is not None else None,
            seed=self.seed,
            metadata=self.metadata.copy()
        )


def slerp(v0: torch.Tensor, v1: torch.Tensor, t: float, DOT_THRESHOLD: float = 0.9995) -> torch.Tensor:
    """
    球面線形補間 (Spherical Linear Interpolation)
    高次元潜在空間での補間に適している
    
    Args:
        v0: 開始ベクトル
        v1: 終了ベクトル
        t: 補間パラメータ [0, 1]
        DOT_THRESHOLD: 線形補間に切り替える閾値
    
    Returns:
        補間されたベクトル
    """
    # 元の形状を保存
    original_shape = v0.shape
    
    # テンソルを平坦化
    v0_flat = v0.reshape(-1)
    v1_flat = v1.reshape(-1)
    
    # 正規化
    v0_norm = v0_flat / torch.norm(v0_flat)
    v1_norm = v1_flat / torch.norm(v1_flat)
    
    # 内積を計算
    dot = torch.dot(v0_norm, v1_norm)
    
    # ベクトルが非常に近い場合は線形補間
    if torch.abs(dot).item() > DOT_THRESHOLD:
        result = (1.0 - t) * v0_flat + t * v1_flat
        return result.reshape(original_shape)
    
    # 球面線形補間
    theta = torch.acos(torch.clamp(dot, -1.0, 1.0))
    sin_theta = torch.sin(theta)
    
    s0 = torch.sin((1.0 - t) * theta) / sin_theta
    s1 = torch.sin(t * theta) / sin_theta
    
    result = s0 * v0_flat + s1 * v1_flat
    return result.reshape(original_shape)


def crossover_slerp(
    parent1: AudioGenotype,
    parent2: AudioGenotype,
    alpha: float = 0.5
) -> AudioGenotype:
    """
    球面線形補間を用いた交叉
    
    Args:
        parent1: 親個体1
        parent2: 親個体2
        alpha: 補間パラメータ [0, 1]
    
    Returns:
        子個体
    """
    # 潜在ノイズの補間
    child_latent = slerp(parent1.latent_noise, parent2.latent_noise, alpha)
    
    # 条件付けベクトルは補間せず、親1のものをそのまま使用
    # (テキストエンベディングの補間は意味的整合性を損なうため)
    child_conditioning = None
    if parent1.conditioning is not None:
        child_conditioning = parent1.conditioning.clone()
    elif parent2.conditioning is not None:
        child_conditioning = parent2.conditioning.clone()
    
    # seedは新規生成（潜在ノイズが補間されているため、元のseedは無効）
    child_seed = np.random.randint(0, 2**32 - 1)
    
    # 子個体の生成
    child = AudioGenotype(
        latent_noise=child_latent,
        conditioning=child_conditioning,
        seed=child_seed,
        metadata={
            "parent1_id": parent1.id,
            "parent2_id": parent2.id,
            "parent1_seed": parent1.seed,
            "parent2_seed": parent2.seed,
            "crossover_alpha": alpha,
            "operation": "crossover_slerp"
        }
    )
    child.generation = max(parent1.generation, parent2.generation) + 1
    
    return child


def mutate_gaussian(
    individual: AudioGenotype,
    mutation_rate: float = 0.1,
    mutation_strength: float = 0.1
) -> AudioGenotype:
    """
    ガウシアンノイズを用いた突然変異
    
    Args:
        individual: 変異させる個体
        mutation_rate: 変異確率 [0, 1]
        mutation_strength: 変異の強度 (ノイズの標準偏差)
    
    Returns:
        変異した個体
    """
    mutant = individual.clone()
    
    # 変異するかどうかの判定
    if np.random.random() > mutation_rate:
        return mutant
    
    # 潜在ノイズに対する変異
    noise = torch.randn_like(mutant.latent_noise) * mutation_strength
    mutant.latent_noise = mutant.latent_noise + noise
    
    # 条件付けベクトルは変異させない
    # (テキストエンベディングを変異させると、元のプロンプトの意味が変わってしまうため)
    # mutant.conditioningはそのまま保持
    
    # seedは新規生成（潜在ノイズが変更されているため、元のseedは無効）
    mutant.seed = np.random.randint(0, 2**32 - 1)
    
    # メタデータの更新
    mutant.metadata = {
        "parent_id": individual.id,
        "parent_seed": individual.seed,
        "mutation_rate": mutation_rate,
        "mutation_strength": mutation_strength,
        "operation": "mutate_gaussian"
    }
    mutant.generation = individual.generation + 1
    
    return mutant


class IECPopulation:
    """
    IEC用の個体群管理クラス
    """
    
    def __init__(self, population_size: int = 6):
        """
        Args:
            population_size: 1世代あたりの個体数
        """
        self.population_size = population_size
        self.current_generation: List[AudioGenotype] = []
        self.generation_number = 0
        self.history: List[List[AudioGenotype]] = []
        self.best_individuals: List[AudioGenotype] = []
    
    def initialize_random(
        self,
        latent_shape: Tuple[int, ...],
        device: str = "cuda"
    ) -> List[AudioGenotype]:
        """
        ランダムな初期個体群を生成
        
        Args:
            latent_shape: 潜在ベクトルの形状
            device: デバイス (cuda/cpu)
        
        Returns:
            初期個体群
        """
        self.current_generation = []
        for i in range(self.population_size):
            # 各個体に固有のseedを生成
            seed = np.random.randint(0, 2**32 - 1)
            
            # seedを使用して潜在ノイズを生成
            generator = torch.Generator(device=device).manual_seed(seed)
            latent_noise = torch.randn(latent_shape, device=device, generator=generator)
            
            genotype = AudioGenotype(
                latent_noise=latent_noise,
                seed=seed,
                metadata={"initialization": "random"}
            )
            genotype.generation = self.generation_number
            self.current_generation.append(genotype)
        
        self.history.append([g.clone() for g in self.current_generation])
        return self.current_generation
    
    def initialize_from_prompt(
        self,
        base_latent: torch.Tensor,
        base_conditioning: Optional[torch.Tensor] = None,
        variation_strength: float = 0.3
    ) -> List[AudioGenotype]:
        """
        プロンプトベースの初期個体群を生成
        基本となる潜在ベクトルに変異を加えた個体群を生成
        
        Args:
            base_latent: 基本となる潜在ベクトル
            base_conditioning: 基本となる条件付けベクトル
            variation_strength: 初期変異の強度
        
        Returns:
            初期個体群
        """
        self.current_generation = []
        for i in range(self.population_size):
            # ベースに変異を加える
            noise = torch.randn_like(base_latent) * variation_strength
            latent_noise = base_latent + noise
            
            conditioning = None
            if base_conditioning is not None:
                cond_noise = torch.randn_like(base_conditioning) * (variation_strength * 0.5)
                conditioning = base_conditioning + cond_noise
            
            genotype = AudioGenotype(
                latent_noise=latent_noise,
                conditioning=conditioning,
                seed=np.random.randint(0, 2**32 - 1),
                metadata={"initialization": "from_prompt", "variation_strength": variation_strength}
            )
            genotype.generation = self.generation_number
            self.current_generation.append(genotype)
        
        self.history.append([g.clone() for g in self.current_generation])
        return self.current_generation
    
    def evolve_next_generation(
        self,
        selected_indices: List[int],
        mutation_rate: float = 0.3,
        mutation_strength: float = 0.15,
        elite_count: int = 1
    ) -> List[AudioGenotype]:
        """
        次世代の個体群を生成
        
        Args:
            selected_indices: ユーザーが選択した個体のインデックス
            mutation_rate: 突然変異率
            mutation_strength: 突然変異の強度
            elite_count: エリート保存数
        
        Returns:
            次世代の個体群
        """
        if len(selected_indices) == 0:
            raise ValueError("少なくとも1つの個体を選択してください")
        
        # 選択された個体を取得
        selected = [self.current_generation[i] for i in selected_indices]
        
        # 最良個体を記録
        if len(selected) > 0:
            self.best_individuals.append(selected[0].clone())
        
        next_generation = []
        
        # エリート保存: 最良個体をそのまま次世代に
        for i in range(min(elite_count, len(selected))):
            elite = selected[i].clone()
            elite.generation = self.generation_number + 1
            elite.metadata["elite"] = True
            next_generation.append(elite)
        
        # 残りの個体を生成
        while len(next_generation) < self.population_size:
            if len(selected) == 1:
                # 1つしか選択されていない場合は変異のみ
                parent = selected[0]
                child = mutate_gaussian(parent, mutation_rate, mutation_strength)
            else:
                # 複数選択されている場合は交叉と変異
                parent1, parent2 = np.random.choice(selected, size=2, replace=False)
                alpha = np.random.uniform(0.3, 0.7)  # 両親の特徴をバランスよく継承
                child = crossover_slerp(parent1, parent2, alpha)
                
                # 交叉後に変異を適用
                if np.random.random() < mutation_rate:
                    child = mutate_gaussian(child, 1.0, mutation_strength)
            
            next_generation.append(child)
        
        # 世代番号を更新
        self.generation_number += 1
        self.current_generation = next_generation[:self.population_size]
        self.history.append([g.clone() for g in self.current_generation])
        
        return self.current_generation
    
    def rollback_generation(self, steps: int = 1) -> List[AudioGenotype]:
        """
        指定世代数だけ戻る
        
        Args:
            steps: 戻る世代数
        
        Returns:
            ロールバック後の個体群
        """
        if len(self.history) <= steps:
            print("警告: 十分な履歴がありません。最初の世代に戻ります。")
            steps = len(self.history) - 1
        
        if steps < 1:
            return self.current_generation
        
        self.generation_number -= steps
        self.current_generation = [g.clone() for g in self.history[self.generation_number]]
        
        # 履歴を調整
        self.history = self.history[:self.generation_number + 1]
        
        return self.current_generation
    
    def save_history(self, filepath: str):
        """
        進化の履歴をJSON形式で保存
        
        Args:
            filepath: 保存先のファイルパス
        """
        history_data = {
            "generation_number": self.generation_number,
            "population_size": self.population_size,
            "history": []
        }
        
        for gen_idx, generation in enumerate(self.history):
            gen_data = {
                "generation": gen_idx,
                "individuals": []
            }
            for individual in generation:
                ind_data = {
                    "id": individual.id,
                    "generation": individual.generation,
                    "fitness": individual.fitness,
                    "metadata": individual.metadata
                }
                gen_data["individuals"].append(ind_data)
            history_data["history"].append(gen_data)
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)
        
        print(f"履歴を保存しました: {filepath}")


class TransformGenotype:
    """
    ノイズ変換行列を遺伝子とする個体。
    固定ベースノイズ z_T に C×C 直交行列 Q(A) を適用して変換ノイズ z'_T を生成する。
    z'_T = Q(A) @ z_T  （論文 Algorithm 2、Sec. 3.1 に準拠）
    """

    def __init__(
        self,
        transform_matrix: torch.Tensor,
        base_noise: torch.Tensor,
        seed: Optional[int] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Args:
            transform_matrix: チャンネル間混合行列 shape: (C, C)
            base_noise: 固定ベースノイズ shape: (1, C, T, F) — セッション全体で共有
            seed: ランダムシード
            metadata: メタデータ
        """
        self.transform_matrix = transform_matrix.clone()
        self.base_noise = base_noise  # 参照共有（書き換え禁止）
        self.seed = seed
        self.metadata = metadata or {}
        self.fitness = 0.0
        self.generation = 0
        self.id = self._generate_id()

    def _generate_id(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"transform_{timestamp}"

    def get_transformed_noise(self) -> torch.Tensor:
        """
        QR分解で直交行列 Q(A) を抽出し、ベースノイズに適用して x_T を返す。
        z'_T = Q(A) @ z_base (論文 Algorithm 2 に準拠)
        直交変換なので ||z'_T|| = ||z_base|| が保証され、ガウスシェル上に留まる。

        Returns:
            shape: (1, C, T, F)
        """
        C = self.base_noise.shape[1]
        T = self.base_noise.shape[2]
        F_size = self.base_noise.shape[3]

        # QR分解で直交行列 Q を抽出（チャンネル次元のみ、論文 Sec. 3.1 に準拠）
        Q, _ = torch.linalg.qr(self.transform_matrix)

        x = self.base_noise[0].reshape(C, -1).to(Q.device)
        x_transformed = Q @ x  # (C, C) @ (C, T*F) -> (C, T*F)

        return x_transformed.reshape(1, C, T, F_size)

    def clone(self) -> 'TransformGenotype':
        cloned = TransformGenotype(
            transform_matrix=self.transform_matrix.clone(),
            base_noise=self.base_noise,
            seed=self.seed,
            metadata=self.metadata.copy()
        )
        cloned.fitness = self.fitness
        cloned.generation = self.generation
        return cloned



def crossover_matrix_uniform(
    parent1: 'TransformGenotype',
    parent2: 'TransformGenotype',
    p: float = 0.5
) -> 'TransformGenotype':
    """
    座標ごとの uniform crossover (論文 Sec. B.3, B.5 に準拠)。
    各要素を確率 p で parent1 から、確率 (1-p) で parent2 から独立に選択する。
    直接ノイズ探索の uniform crossover と同じ操作を行列要素に適用。
    """
    mask = torch.bernoulli(
        torch.full_like(parent1.transform_matrix, p)
    ).bool()
    child_matrix = torch.where(mask, parent1.transform_matrix, parent2.transform_matrix)

    child = TransformGenotype(
        transform_matrix=child_matrix,
        base_noise=parent1.base_noise,
        seed=np.random.randint(0, 2**32 - 1),
        metadata={
            "parent1_id": parent1.id,
            "parent2_id": parent2.id,
            "crossover_p": p,
            "operation": "crossover_matrix_uniform",
            "prompt": parent1.metadata.get("prompt", "")
        }
    )
    child.generation = max(parent1.generation, parent2.generation) + 1
    return child


def mutate_transform_gaussian(
    individual: 'TransformGenotype',
    mutation_strength: float = 0.1
) -> 'TransformGenotype':
    """
    ガウシアンノイズによる変換行列の突然変異。
    """
    mutant = individual.clone()
    noise = torch.randn_like(mutant.transform_matrix) * mutation_strength
    mutant.transform_matrix = mutant.transform_matrix + noise
    mutant.seed = np.random.randint(0, 2**32 - 1)
    mutant.metadata = {
        "parent_id": individual.id,
        "mutation_strength": mutation_strength,
        "operation": "mutate_transform_gaussian",
        "prompt": individual.metadata.get("prompt", "")
    }
    mutant.generation = individual.generation + 1
    return mutant


def adaptive_mutation_rate(generation: int, convergence_score: float = 0.0) -> float:
    """
    世代数と収束状況に応じて突然変異率を動的に調整
    
    Args:
        generation: 現在の世代番号
        convergence_score: 収束度スコア [0, 1] (1に近いほど収束)
    
    Returns:
        調整された突然変異率
    """
    # 初期世代では高い変異率で探索
    base_rate = 0.5 * np.exp(-generation / 10.0) + 0.1
    
    # 収束している場合は変異率を上げて多様性を維持
    if convergence_score > 0.7:
        base_rate *= 1.5
    
    return min(base_rate, 0.8)

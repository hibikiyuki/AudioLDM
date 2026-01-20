"""
AudioLDMとIECを統合したパイプライン
"""

import torch
import numpy as np
from typing import List, Optional, Tuple, Dict
import os
from tqdm import tqdm

from audioldm import LatentDiffusion
from audioldm.pipeline import build_model, make_batch_for_text_to_audio, duration_to_latent_t_size
from audioldm.latent_diffusion.ddim import DDIMSampler
from audioldm.iec import AudioGenotype, IECPopulation
from audioldm.utils import save_wave


class AudioLDM_IEC:
    """
    AudioLDMを用いた対話型進化的音声生成システム
    """
    
    def __init__(
        self,
        model_name: str = "audioldm-s-full-v2",
        ckpt_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        population_size: int = 6,
        duration: float = 5.0,
        guidance_scale: float = 2.5,
        ddim_steps: int = 200,
        n_candidate_gen_per_text: int = 1,
    ):
        """
        Args:
            model_name: AudioLDMのモデル名
            ckpt_path: チェックポイントのパス
            device: 使用デバイス
            population_size: 1世代あたりの個体数
            duration: 生成する音声の長さ (秒)
            guidance_scale: ガイダンススケール
            ddim_steps: DDIMサンプリングのステップ数
            n_candidate_gen_per_text: テキストごとの候補生成数
        """
        self.device = device
        self.population_size = population_size
        self.duration = duration
        self.guidance_scale = guidance_scale
        self.ddim_steps = ddim_steps
        self.n_candidate_gen_per_text = n_candidate_gen_per_text
        
        # AudioLDMモデルのロード
        print(f"AudioLDMモデルをロード中: {model_name}")
        self.latent_diffusion = build_model(
            ckpt_path=ckpt_path,
            model_name=model_name
        )
        self.latent_diffusion = self.latent_diffusion.to(device)
        self.latent_diffusion.eval()
        
        # 潜在空間のサイズを設定
        self.latent_diffusion.latent_t_size = duration_to_latent_t_size(duration)
        self.latent_diffusion.cond_stage_model.embed_mode = "text"
        
        # 潜在空間の形状
        self.latent_shape = (
            self.latent_diffusion.channels,
            self.latent_diffusion.latent_t_size,
            self.latent_diffusion.latent_f_size
        )
        
        # IEC個体群の初期化
        self.population = IECPopulation(population_size=population_size)
        
        print(f"初期化完了: 個体数={population_size}, 音声長={duration}秒, デバイス={device}")
        print(f"潜在空間形状: {self.latent_shape}")
    
    def _generate_audio_from_genotype(
        self,
        genotype: AudioGenotype,
        text: str = "",
        unconditional_guidance_scale: Optional[float] = None
    ) -> np.ndarray:
        """
        遺伝子型から音声を生成
        
        Args:
            genotype: 音声遺伝子型（潜在ベクトルを含む）
            text: プロンプトテキスト
            unconditional_guidance_scale: ガイダンススケール
        
        Returns:
            生成された音声波形 (numpy array)
        """
        if unconditional_guidance_scale is None:
            unconditional_guidance_scale = self.guidance_scale
        
        with torch.no_grad():
            # 潜在ベクトルを取得
            x_T = genotype.latent_noise.to(self.device)
            
            # テキスト条件付けを取得
            if text:
                c = self.latent_diffusion.get_learned_conditioning([text])
            else:
                c = self.latent_diffusion.get_learned_conditioning([" "])
            
            # Unconditional conditioning
            uc = self.latent_diffusion.get_learned_conditioning([" "])
            
            # sample_logを直接呼び出し（x_Tを渡せる）
            samples, _ = self.latent_diffusion.sample_log(
                cond=c,
                batch_size=1,
                ddim=True,
                ddim_steps=self.ddim_steps,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=uc,
                x_T=x_T,
            )
            
            # サンプルのクリッピング
            if torch.max(torch.abs(samples)) > 1e2:
                samples = torch.clip(samples, min=-10, max=10)
            
            # 潜在空間からメルスペクトログラムへデコード
            mel = self.latent_diffusion.decode_first_stage(samples)
            
            # メルスペクトログラムから音声波形へ変換
            waveform = self.latent_diffusion.mel_spectrogram_to_waveform(mel)
        
        return waveform
    
    def initialize_population(
        self,
        prompt: Optional[str] = None,
        variation_strength: float = 0.3
    ) -> List[Tuple[AudioGenotype, np.ndarray]]:
        """
        初期個体群を生成
        
        Args:
            prompt: 初期プロンプト (Noneの場合はランダム生成=空プロンプト)
            variation_strength: プロンプトからの初期変異強度
        
        Returns:
            (遺伝子型, 音声波形) のリスト
        """
        print(f"第{self.population.generation_number}世代の個体群を生成中...")
        
        if prompt is None:
            prompt = ""
            print("ランダム（無条件）生成")
        else:
            print(f"プロンプト: '{prompt}'")
        
        # プロンプトベースの初期個体群を生成
        if prompt:
            # テキスト条件付けベクトルを取得
            text_embedding = self.latent_diffusion.get_learned_conditioning([prompt])
            
            # ベース潜在ベクトルを生成
            base_latent = torch.randn((1,) + self.latent_shape, device=self.device)
            
            genotypes = []
            for i in range(self.population_size):
                # ベースに変異を加える
                noise = torch.randn_like(base_latent) * variation_strength
                latent_noise = base_latent + noise
                
                # 条件付けにも変異を加える（オプション）
                cond_noise = torch.randn_like(text_embedding) * (variation_strength * 0.3)
                conditioning = text_embedding + cond_noise
                
                genotype = AudioGenotype(
                    latent_noise=latent_noise,
                    conditioning=conditioning,
                    seed=np.random.randint(0, 2**31 - 1),
                    metadata={"initialization": "from_prompt", "prompt": prompt, "variation_strength": variation_strength}
                )
                genotype.generation = self.population.generation_number
                genotypes.append(genotype)
        else:
            # ランダム初期化
            genotypes = []
            for i in range(self.population_size):
                latent_noise = torch.randn((1,) + self.latent_shape, device=self.device)
                genotype = AudioGenotype(
                    latent_noise=latent_noise,
                    conditioning=None,
                    seed=np.random.randint(0, 2**31 - 1),
                    metadata={"initialization": "random", "prompt": prompt}
                )
                genotype.generation = self.population.generation_number
                genotypes.append(genotype)
        
        self.population.current_generation = genotypes
        self.population.history.append([g.clone() for g in genotypes])
        
        # 各個体から音声を生成
        results = []
        for i, genotype in enumerate(tqdm(genotypes, desc="音声生成中")):
            waveform = self._generate_audio_from_genotype(genotype, text=prompt)
            results.append((genotype, waveform))
        
        return results
    
    def evolve_population(
        self,
        selected_indices: List[int],
        mutation_rate: float = 0.3,
        mutation_strength: float = 0.15,
        elite_count: int = 1
    ) -> List[Tuple[AudioGenotype, np.ndarray]]:
        """
        選択された個体から次世代を生成
        
        Args:
            selected_indices: 選択された個体のインデックスリスト
            mutation_rate: 突然変異率
            mutation_strength: 突然変異強度
            elite_count: エリート保存数
        
        Returns:
            (遺伝子型, 音声波形) のリスト
        """
        print(f"\n第{self.population.generation_number + 1}世代を生成中...")
        print(f"選択された個体: {selected_indices}")
        
        if len(selected_indices) == 0:
            raise ValueError("少なくとも1つの個体を選択してください")
        
        # 選択された個体を取得
        selected = [self.population.current_generation[i] for i in selected_indices]
        
        # プロンプトを取得（最初の選択個体から）
        prompt = selected[0].metadata.get("prompt", "")
        
        # IECの進化関数を使用
        from audioldm.iec import crossover_slerp, mutate_gaussian
        
        next_generation = []
        
        # エリート保存
        for i in range(min(elite_count, len(selected))):
            elite = selected[i].clone()
            elite.generation = self.population.generation_number + 1
            elite.metadata["elite"] = True
            next_generation.append(elite)
        
        # 残りの個体を生成
        while len(next_generation) < self.population_size:
            if len(selected) == 1:
                # 1つしか選択されていない場合は変異のみ
                parent = selected[0]
                child = mutate_gaussian(parent, mutation_rate, mutation_strength)
            else:
                # 2つの親から交叉
                parent1, parent2 = np.random.choice(selected, size=2, replace=False)
                alpha = np.random.uniform(0.3, 0.7)
                child = crossover_slerp(parent1, parent2, alpha)
                
                # 交叉後に変異を適用
                if np.random.random() < mutation_rate:
                    child = mutate_gaussian(child, 1.0, mutation_strength)
            
            child.generation = self.population.generation_number + 1
            child.metadata["prompt"] = prompt
            next_generation.append(child)
        
        # 世代番号を更新
        self.population.generation_number += 1
        self.population.current_generation = next_generation[:self.population_size]
        self.population.history.append([g.clone() for g in self.population.current_generation])
        
        # 各個体から音声を生成
        results = []
        for i, genotype in enumerate(tqdm(self.population.current_generation, desc="音声生成中")):
            waveform = self._generate_audio_from_genotype(genotype, text=prompt)
            results.append((genotype, waveform))
        
        return results
    
    def save_generation_audio(
        self,
        generation_results: List[Tuple[AudioGenotype, np.ndarray]],
        output_dir: str,
        prefix: str = "gen"
    ) -> List[str]:
        """
        世代の音声を保存
        
        Args:
            generation_results: (遺伝子型, 音声波形) のリスト
            output_dir: 出力ディレクトリ
            prefix: ファイル名のプレフィックス
        
        Returns:
            保存されたファイルパスのリスト
        """
        os.makedirs(output_dir, exist_ok=True)
        
        saved_paths = []
        gen_num = self.population.generation_number
        
        for i, (genotype, waveform) in enumerate(generation_results):
            filename = f"{prefix}_gen{gen_num:03d}_ind{i:02d}.wav"
            filepath = os.path.join(output_dir, filename)
            
            # 音声を保存
            import soundfile as sf
            # waveformの形状: (batch, samples) または (batch, 1, samples)
            # モノラル音声として保存
            if len(waveform.shape) == 3:
                # (batch, 1, samples) -> (samples,)
                audio_data = waveform[0, 0, :]
            elif len(waveform.shape) == 2:
                # (batch, samples) -> (samples,)
                audio_data = waveform[0, :]
            else:
                # すでに1次元の場合
                audio_data = waveform
            
            sf.write(filepath, audio_data, samplerate=16000)
            saved_paths.append(filepath)
        
        print(f"音声を保存しました: {output_dir}")
        return saved_paths
    
    def get_generation_info(self) -> Dict:
        """
        現在の世代情報を取得
        
        Returns:
            世代情報の辞書
        """
        return {
            "generation_number": self.population.generation_number,
            "population_size": self.population_size,
            "history_length": len(self.population.history),
            "best_count": len(self.population.best_individuals)
        }


def run_iec_session(
    prompt: Optional[str] = None,
    model_name: str = "audioldm-s-full-v2",
    population_size: int = 6,
    duration: float = 5.0,
    output_dir: str = "./output/iec_session",
    max_generations: int = 10
):
    """
    IECセッションを実行 (CLIベース)
    
    Args:
        prompt: 初期プロンプト
        model_name: モデル名
        population_size: 個体数
        duration: 音声長
        output_dir: 出力ディレクトリ
        max_generations: 最大世代数
    """
    # システムの初期化
    iec_system = AudioLDM_IEC(
        model_name=model_name,
        population_size=population_size,
        duration=duration
    )
    
    # 初期個体群を生成
    results = iec_system.initialize_population(prompt=prompt)
    
    # 音声を保存
    saved_paths = iec_system.save_generation_audio(
        results,
        output_dir=output_dir,
        prefix="initial"
    )
    
    print(f"\n初期個体群を生成しました: {len(saved_paths)}個")
    print("音声を聴いて、気に入った個体の番号を選択してください (スペース区切り)")
    
    for generation in range(max_generations):
        print(f"\n--- 第{iec_system.population.generation_number}世代 ---")
        for i, path in enumerate(saved_paths):
            print(f"  [{i}] {os.path.basename(path)}")
        
        # ユーザー入力
        user_input = input("\n選択する個体番号 (例: 0 2 3): ").strip()
        
        if user_input.lower() in ['q', 'quit', 'exit']:
            print("セッションを終了します。")
            break
        
        try:
            selected_indices = [int(x) for x in user_input.split()]
            
            if len(selected_indices) == 0:
                print("少なくとも1つの個体を選択してください。")
                continue
            
            # 次世代を生成
            results = iec_system.evolve_population(selected_indices)
            
            # 音声を保存
            saved_paths = iec_system.save_generation_audio(
                results,
                output_dir=output_dir,
                prefix=f"gen{iec_system.population.generation_number}"
            )
            
        except ValueError as e:
            print(f"エラー: {e}")
            continue
    
    # 履歴を保存
    history_path = os.path.join(output_dir, "iec_history.json")
    iec_system.population.save_history(history_path)
    
    print(f"\n=== IECセッション完了 ===")
    print(f"総世代数: {iec_system.population.generation_number}")
    print(f"出力ディレクトリ: {output_dir}")


if __name__ == "__main__":
    # テスト実行
    run_iec_session(
        prompt="爆発音",
        population_size=4,
        duration=3.0,
        max_generations=5
    )

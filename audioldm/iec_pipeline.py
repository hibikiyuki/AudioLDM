"""
AudioLDMとIECを統合したパイプライン
"""

import torch
import numpy as np
import soundfile as sf
from typing import List, Optional, Tuple, Dict
import os

from audioldm.pipeline import build_model, duration_to_latent_t_size
from audioldm.iec import (
    AudioGenotype, IECPopulation,
    TransformGenotype, crossover_matrix_uniform, mutate_transform_gaussian,
    LatentZ0Genotype, crossover_z0_slerp, mutate_z0_gaussian,
)


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
        guidance_scale: float = 7.5, # ガイダンススケールのデフォルト値を7.5に設定
        ddim_steps: int = 200,
        n_candidate_gen_per_text: int = 3,
        ga_mode: str = "latent",
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
            ga_mode: "latent" (潜在ノイズGA) または "transform" (変換行列GA)
        """
        self.device = device
        self.population_size = population_size
        self.duration = duration
        self.guidance_scale = guidance_scale
        self.ddim_steps = ddim_steps
        self.n_candidate_gen_per_text = n_candidate_gen_per_text
        self.ga_mode = ga_mode
        self._shared_base_noise: Optional[torch.Tensor] = None
        
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
        
        print(f"初期化完了: 個体数={population_size}, 音声長={duration}秒, デバイス={device}, モード={ga_mode}")
        print(f"潜在空間形状: {self.latent_shape}")

    # ------------------------------------------------------------------
    # 共通ヘルパー
    # ------------------------------------------------------------------

    def _get_or_create_base_noise(self, seed: Optional[int] = None) -> torch.Tensor:
        """変換行列モード用の共有ベースノイズを取得・生成する。"""
        if self._shared_base_noise is None:
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
                self._shared_base_noise = torch.randn(
                    (1,) + self.latent_shape, device=self.device, generator=generator
                )
            else:
                self._shared_base_noise = torch.randn(
                    (1,) + self.latent_shape, device=self.device
                )
            print(f"ベースノイズを生成しました (seed={seed})")
        return self._shared_base_noise

    def _sample_audio(
        self,
        x_T: torch.Tensor,
        text: str = "",
        unconditional_guidance_scale: Optional[float] = None,
        ddim_eta: float = 0.0,
    ) -> np.ndarray:
        """x_T とテキストから音声波形を生成する共通ルーチン。"""
        return self._sample_audio_batch([x_T], text=text,
                                        unconditional_guidance_scale=unconditional_guidance_scale,
                                        ddim_eta=ddim_eta)[0]

    def _sample_audio_batch(
        self,
        x_T_list: List[torch.Tensor],
        text: str = "",
        unconditional_guidance_scale: Optional[float] = None,
        ddim_eta: float = 0.0,
    ) -> List[np.ndarray]:
        """複数のx_Tをバッチ処理して音声波形リストを返す。テキスト条件・DDIMは1回だけ実行。"""
        if unconditional_guidance_scale is None:
            unconditional_guidance_scale = self.guidance_scale

        n = len(x_T_list)

        with self.latent_diffusion.ema_scope("IEC Generation"):
            with torch.no_grad():
                if text:
                    c_single = self.latent_diffusion.cond_stage_model([text, text])
                    c_single = c_single[0:1]
                else:
                    c_single = self.latent_diffusion.cond_stage_model([" ", " "])
                    c_single = c_single[0:1]

                c = torch.cat([c_single] * n, dim=0)

                uc = (
                    self.latent_diffusion.cond_stage_model.get_unconditional_condition(n)
                    if unconditional_guidance_scale != 1.0
                    else None
                )

                x_T_batch = torch.cat(x_T_list, dim=0)

                samples, _ = self.latent_diffusion.sample_log(
                    cond=c,
                    batch_size=n,
                    ddim=True,
                    ddim_steps=self.ddim_steps,
                    eta=ddim_eta,
                    unconditional_guidance_scale=unconditional_guidance_scale,
                    unconditional_conditioning=uc,
                    x_T=x_T_batch,
                )

                if torch.max(torch.abs(samples)) > 1e2:
                    samples = torch.clip(samples, min=-10, max=10)

                mel = self.latent_diffusion.decode_first_stage(samples)
                waveform_batch = self.latent_diffusion.mel_spectrogram_to_waveform(mel)

        return [waveform_batch[i:i+1] for i in range(n)]

    def _generate_audio_from_any_genotype(
        self,
        genotype,
        text: str = "",
        unconditional_guidance_scale: Optional[float] = None,
        ddim_eta: float = 0.0,
    ) -> np.ndarray:
        """AudioGenotype / TransformGenotype / LatentZ0Genotype を問わず音声を生成するディスパッチャ。"""
        if isinstance(genotype, LatentZ0Genotype):
            return self._generate_audio_from_z0_genotype(genotype)
        if isinstance(genotype, TransformGenotype):
            return self._generate_audio_from_transform_genotype(
                genotype, text=text,
                unconditional_guidance_scale=unconditional_guidance_scale,
                ddim_eta=ddim_eta,
            )
        return self._generate_audio_from_genotype(
            genotype, text=text,
            unconditional_guidance_scale=unconditional_guidance_scale,
            ddim_eta=ddim_eta,
        )

    # ------------------------------------------------------------------
    # 潜在ノイズ GA（既存モード）
    # ------------------------------------------------------------------

    def _generate_audio_from_genotype(
        self,
        genotype: AudioGenotype,
        text: str = "",
        unconditional_guidance_scale: Optional[float] = None,
        ddim_eta: float = 0.0
    ) -> np.ndarray:
        """
        遺伝子型から音声を生成
        
        Args:
            genotype: 音声遺伝子型（潜在ベクトルを含む）
            text: プロンプトテキスト
            unconditional_guidance_scale: ガイダンススケール
            ddim_eta: DDIMサンプリングのeta（確率性を制御）
        
        Returns:
            生成された音声波形 (numpy array)
        """
        if genotype.seed is not None:
            torch.manual_seed(genotype.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(genotype.seed)
            np.random.seed(genotype.seed % (2**32))

        x_T = genotype.latent_noise.to(self.device)
        return self._sample_audio(x_T, text=text,
                                  unconditional_guidance_scale=unconditional_guidance_scale,
                                  ddim_eta=ddim_eta)

    # ------------------------------------------------------------------
    # 変換行列 GA（新モード）
    # ------------------------------------------------------------------

    def _generate_audio_from_transform_genotype(
        self,
        genotype: TransformGenotype,
        text: str = "",
        unconditional_guidance_scale: Optional[float] = None,
        ddim_eta: float = 0.0,
    ) -> np.ndarray:
        """変換行列遺伝子型から音声を生成する。"""
        x_T = genotype.get_transformed_noise().to(self.device)
        return self._sample_audio(x_T, text=text,
                                  unconditional_guidance_scale=unconditional_guidance_scale,
                                  ddim_eta=ddim_eta)

    def initialize_population_transform(
        self,
        prompt: Optional[str] = None,
        base_noise_seed: Optional[int] = None,
    ) -> List[Tuple[TransformGenotype, np.ndarray]]:
        """
        変換行列モードで初期個体群を生成する。

        各個体の変換行列は N(0, I) からサンプルした C×C 行列で初期化する（論文 Sec. C.1 に準拠）。
        ベースノイズ z_T はセッション全体で共有される。
        """
        print(f"[変換行列モード] 第{self.population.generation_number}世代を生成中...")
        if prompt is None:
            prompt = ""

        base_noise = self._get_or_create_base_noise(seed=base_noise_seed)
        C = self.latent_shape[0]

        genotypes: List[TransformGenotype] = []
        for _ in range(self.population_size):
            seed = np.random.randint(0, 2**32 - 1)
            torch.manual_seed(seed)
            M = torch.randn(C, C, device=self.device)  # N(0,I) → QR → ランダム直交行列 (論文準拠)

            g = TransformGenotype(
                transform_matrix=M,
                base_noise=base_noise,
                seed=seed,
                metadata={
                    "ga_mode": "transform",
                    "initialization": "random_gaussian",
                    "prompt": prompt,
                    "base_noise_seed": base_noise_seed,
                }
            )
            g.generation = self.population.generation_number
            genotypes.append(g)

        self.population.current_generation = genotypes
        self.population.history.append([g.clone() for g in genotypes])

        x_T_list = [g.get_transformed_noise().to(self.device) for g in genotypes]
        waveforms = self._sample_audio_batch(x_T_list, text=prompt)
        return list(zip(genotypes, waveforms))

    def evolve_population_transform(
        self,
        selected_indices: List[int],
        mutation_strength: float = 0.1,
        elite_count: int = 1,
    ) -> List[Tuple[TransformGenotype, np.ndarray]]:
        """
        変換行列モードで次世代を生成する。

        交叉: uniform crossover（論文 Sec. B.3, B.5 に準拠）
        突然変異: 行列にガウスノイズを加算（σ = mutation_strength）
        """
        print(f"\n[変換行列モード] 第{self.population.generation_number + 1}世代を生成中...")
        if not selected_indices:
            raise ValueError("少なくとも1つの個体を選択してください")

        selected = [self.population.current_generation[i] for i in selected_indices]
        prompt = selected[0].metadata.get("prompt", "")

        next_gen: List[TransformGenotype] = []

        for i in range(min(elite_count, len(selected))):
            elite = selected[i].clone()
            elite.generation = self.population.generation_number + 1
            elite.metadata["elite"] = True
            next_gen.append(elite)

        while len(next_gen) < self.population_size:
            if len(selected) == 1:
                child = mutate_transform_gaussian(selected[0], mutation_strength)
            else:
                p1, p2 = np.random.choice(selected, size=2, replace=False)
                child = crossover_matrix_uniform(p1, p2)  # uniform crossover (論文 Sec. B.3, B.5)
                child = mutate_transform_gaussian(child, mutation_strength)

            child.generation = self.population.generation_number + 1
            child.metadata["prompt"] = prompt
            next_gen.append(child)

        self.population.generation_number += 1
        self.population.current_generation = next_gen[:self.population_size]
        self.population.history.append([g.clone() for g in self.population.current_generation])

        genotypes = self.population.current_generation
        x_T_list = [g.get_transformed_noise().to(self.device) for g in genotypes]
        waveforms = self._sample_audio_batch(x_T_list, text=prompt)
        return list(zip(genotypes, waveforms))

    # ------------------------------------------------------------------
    # z_0 GA（新モード: DDIM逆拡散で得られた潜在表現を個体とする）
    # ------------------------------------------------------------------

    def _sample_z0_batch(
        self,
        x_T_list: List[torch.Tensor],
        text: str = "",
        unconditional_guidance_scale: Optional[float] = None,
        ddim_eta: float = 0.0,
    ) -> List[torch.Tensor]:
        """複数の z_T から DDIM で z_0 をバッチ生成し、z_0 テンソルのリストを返す。"""
        if unconditional_guidance_scale is None:
            unconditional_guidance_scale = self.guidance_scale
        n = len(x_T_list)
        with self.latent_diffusion.ema_scope("Z0 DDIM Sampling"):
            with torch.no_grad():
                if text:
                    c_single = self.latent_diffusion.cond_stage_model([text, text])[0:1]
                else:
                    c_single = self.latent_diffusion.cond_stage_model([" ", " "])[0:1]
                c = torch.cat([c_single] * n, dim=0)
                uc = (
                    self.latent_diffusion.cond_stage_model.get_unconditional_condition(n)
                    if unconditional_guidance_scale != 1.0 else None
                )
                x_T_batch = torch.cat(x_T_list, dim=0)
                samples, _ = self.latent_diffusion.sample_log(
                    cond=c, batch_size=n, ddim=True,
                    ddim_steps=self.ddim_steps, eta=ddim_eta,
                    unconditional_guidance_scale=unconditional_guidance_scale,
                    unconditional_conditioning=uc, x_T=x_T_batch,
                )
        return [samples[i:i+1] for i in range(n)]

    def _decode_z0_to_audio(self, z0: torch.Tensor) -> np.ndarray:
        """z_0 を VAE デコードして音声波形を返す（DDIM 不要）。"""
        with self.latent_diffusion.ema_scope("Z0 Decode"):
            with torch.no_grad():
                if torch.max(torch.abs(z0)) > 1e2:
                    z0 = torch.clip(z0, min=-10, max=10)
                mel = self.latent_diffusion.decode_first_stage(z0)
                waveform = self.latent_diffusion.mel_spectrogram_to_waveform(mel)
        return waveform[0:1]

    def _decode_z0_batch(self, z0_list: List[torch.Tensor]) -> List[np.ndarray]:
        """複数の z_0 をバッチ VAE デコードして音声波形リストを返す。"""
        with self.latent_diffusion.ema_scope("Z0 Batch Decode"):
            with torch.no_grad():
                z0_batch = torch.cat(z0_list, dim=0)
                if torch.max(torch.abs(z0_batch)) > 1e2:
                    z0_batch = torch.clip(z0_batch, min=-10, max=10)
                mel = self.latent_diffusion.decode_first_stage(z0_batch)
                waveform_batch = self.latent_diffusion.mel_spectrogram_to_waveform(mel)
        n = len(z0_list)
        return [waveform_batch[i:i+1] for i in range(n)]

    def _generate_audio_from_z0_genotype(self, genotype: LatentZ0Genotype) -> np.ndarray:
        """z_0 を直接 VAE デコードして音声生成（DDIM 不要）。"""
        return self._decode_z0_to_audio(genotype.z0.to(self.device))

    def initialize_population_z0(
        self,
        prompt: Optional[str] = None,
    ) -> List[Tuple[LatentZ0Genotype, np.ndarray]]:
        """
        z_0 モードの初期個体群を生成する。

        各個体: ランダム z_T → DDIM サンプリング → z_0 を個体として保持。
        以降の子個体の評価は VAE decode のみで済む（DDIM 不要）。
        """
        print(f"[z0モード] 第{self.population.generation_number}世代を生成中...")
        if prompt is None:
            prompt = ""

        x_T_list = []
        seeds = []
        for _ in range(self.population_size):
            seed = np.random.randint(0, 2**32 - 1)
            seeds.append(seed)
            generator = torch.Generator(device=self.device).manual_seed(seed)
            x_T = torch.randn((1,) + self.latent_shape, device=self.device, generator=generator)
            x_T_list.append(x_T)

        print(f"  DDIM サンプリングで z_0 を生成中 (個体数={self.population_size})...")
        z0_list = self._sample_z0_batch(x_T_list, text=prompt)

        genotypes = []
        for z0, seed in zip(z0_list, seeds):
            g = LatentZ0Genotype(
                z0=z0,
                seed=seed,
                metadata={"initialization": "ddim_sampling", "prompt": prompt, "ga_mode": "z0"}
            )
            g.generation = self.population.generation_number
            genotypes.append(g)

        self.population.current_generation = genotypes
        self.population.history.append([g.clone() for g in genotypes])

        print(f"  VAE デコードで音声を生成中...")
        waveforms = self._decode_z0_batch([g.z0.to(self.device) for g in genotypes])
        return list(zip(genotypes, waveforms))

    def evolve_population_z0(
        self,
        selected_indices: List[int],
        mutation_strength: float = 0.15,
        elite_count: int = 1,
        fresh_count: int = 1,
    ) -> List[Tuple[LatentZ0Genotype, np.ndarray]]:
        """
        z_0 空間での交叉・変異 → VAE デコードで音声生成。

        子の評価に DDIM は不要なため、進化世代の生成が高速。

        Args:
            fresh_count: 毎世代 DDIM で完全新規生成する個体数。
                         slerp 補間だけだと個体が選択親の重心に収束するため、
                         fresh_count > 0 にすることで多様性を維持できる。
        """
        print(f"\n[z0モード] 第{self.population.generation_number + 1}世代を生成中...")
        if not selected_indices:
            raise ValueError("少なくとも1つの個体を選択してください")

        selected = [self.population.current_generation[i] for i in selected_indices]
        prompt = selected[0].metadata.get("prompt", "")

        next_gen: List[LatentZ0Genotype] = []

        # エリート保存
        for i in range(min(elite_count, len(selected))):
            elite = selected[i].clone()
            elite.generation = self.population.generation_number + 1
            elite.metadata["elite"] = True
            next_gen.append(elite)

        # DDIM 新鮮注入: slerp だけだと全子が親の「間」に収束するため、
        # 完全ランダムな z_T から DDIM で生成した個体を混入して多様性を維持する。
        actual_fresh = min(fresh_count, self.population_size - len(next_gen))
        if actual_fresh > 0:
            print(f"  DDIM 新鮮注入: {actual_fresh} 体を DDIM で新規生成中...")
            fresh_x_T_list = []
            fresh_seeds = []
            for _ in range(actual_fresh):
                seed = np.random.randint(0, 2**32 - 1)
                fresh_seeds.append(seed)
                generator = torch.Generator(device=self.device).manual_seed(seed)
                x_T = torch.randn((1,) + self.latent_shape, device=self.device, generator=generator)
                fresh_x_T_list.append(x_T)
            fresh_z0_list = self._sample_z0_batch(fresh_x_T_list, text=prompt)
            for z0, seed in zip(fresh_z0_list, fresh_seeds):
                g = LatentZ0Genotype(
                    z0=z0, seed=seed,
                    metadata={"operation": "ddim_fresh_injection", "prompt": prompt,
                               "ga_mode": "z0", "generation": self.population.generation_number + 1}
                )
                g.generation = self.population.generation_number + 1
                next_gen.append(g)

        # 交叉 + 突然変異で残りを埋める
        while len(next_gen) < self.population_size:
            if len(selected) == 1:
                child = mutate_z0_gaussian(selected[0], mutation_strength)
            else:
                p1, p2 = np.random.choice(selected, size=2, replace=False)
                # alpha 範囲を広げ（[0.1, 0.9]）、親の中心から離れた点も探索する
                alpha = np.random.uniform(0.1, 0.9)
                child = crossover_z0_slerp(p1, p2, alpha)
                child = mutate_z0_gaussian(child, mutation_strength)

            child.generation = self.population.generation_number + 1
            child.metadata["prompt"] = prompt
            next_gen.append(child)

        self.population.generation_number += 1
        self.population.current_generation = next_gen[:self.population_size]
        self.population.history.append([g.clone() for g in self.population.current_generation])

        # 新鮮注入個体は DDIM 済みなので VAE decode だけで済む
        print(f"  VAE デコードで音声を生成中 (DDIM 不要)...")
        waveforms = self._decode_z0_batch(
            [g.z0.to(self.device) for g in self.population.current_generation]
        )
        return list(zip(self.population.current_generation, waveforms))

    # ------------------------------------------------------------------
    # 公開 API（モードディスパッチ）
    # ------------------------------------------------------------------

    def initialize_population(
        self,
        prompt: Optional[str] = None,
        variation_strength: float = 0.3,
        ga_mode: Optional[str] = None,
        base_noise_seed: Optional[int] = None,
    ) -> List[Tuple]:
        """
        初期個体群を生成する。ga_mode を渡すと self.ga_mode を上書きできる。

        Args:
            prompt: 初期プロンプト (None で無条件生成)
            variation_strength: 未使用（互換性のため残存）
            ga_mode: "latent" または "transform"（省略時は self.ga_mode を使用）
            base_noise_seed: 変換行列モード用ベースノイズの seed
        """
        if ga_mode is not None:
            self.ga_mode = ga_mode
            # モード切替時はベースノイズをリセット
            self._shared_base_noise = None

        if self.ga_mode == "transform":
            return self.initialize_population_transform(
                prompt=prompt, base_noise_seed=base_noise_seed
            )

        if self.ga_mode == "z0":
            return self.initialize_population_z0(prompt=prompt)

        print(f"[潜在ノイズモード] 第{self.population.generation_number}世代の個体群を生成中...")
        
        if prompt is None:
            prompt = ""
            print("ランダム（無条件）生成")
        else:
            print(f"プロンプト: '{prompt}'")
        
        # プロンプトベースの初期個体群を生成
        if prompt:
            # テキスト条件付けベクトルを取得（全個体で共通）
            # 通常のAudioLDMと同じ方法：単一テキストは2つ複製してから1つ取る
            text_embedding = self.latent_diffusion.cond_stage_model([prompt, prompt])
            text_embedding = text_embedding[0:1]
            
            genotypes = []
            for _ in range(self.population_size):
                # 各個体に完全に独立したseedを生成
                seed = np.random.randint(0, 2**32 - 1)

                # seedを使用してランダムノイズを生成
                generator = torch.Generator(device=self.device).manual_seed(seed)
                latent_noise = torch.randn((1,) + self.latent_shape, device=self.device, generator=generator)

                genotype = AudioGenotype(
                    latent_noise=latent_noise,
                    conditioning=text_embedding,  # 条件付けは共通（ノイズを加えない）
                    seed=seed,
                    metadata={"initialization": "from_prompt", "prompt": prompt}
                )
                genotype.generation = self.population.generation_number
                genotypes.append(genotype)
        else:
            # ランダム初期化（各個体に完全に独立したランダムノイズ）
            genotypes = []
            for _ in range(self.population_size):
                # 各個体に完全に独立したseedを生成
                seed = np.random.randint(0, 2**32 - 1)
                
                # seedを使用してランダムノイズを生成
                generator = torch.Generator(device=self.device).manual_seed(seed)
                latent_noise = torch.randn((1,) + self.latent_shape, device=self.device, generator=generator)
                
                genotype = AudioGenotype(
                    latent_noise=latent_noise,
                    conditioning=None,
                    seed=seed,
                    metadata={"initialization": "random", "prompt": prompt}
                )
                genotype.generation = self.population.generation_number
                genotypes.append(genotype)
        
        self.population.current_generation = genotypes
        self.population.history.append([g.clone() for g in genotypes])

        x_T_list = [g.latent_noise.to(self.device) for g in genotypes]
        waveforms = self._sample_audio_batch(x_T_list, text=prompt)
        return list(zip(genotypes, waveforms))

    def evolve_population(
        self,
        selected_indices: List[int],
        mutation_rate: float = 0.3,
        mutation_strength: float = 0.15,
        elite_count: int = 1,
        fresh_count: int = 1,
    ) -> List[Tuple]:
        """選択された個体から次世代を生成する（モードに応じてディスパッチ）。"""
        if self.ga_mode == "transform":
            return self.evolve_population_transform(
                selected_indices,
                mutation_strength=mutation_strength,
                elite_count=elite_count,
            )

        if self.ga_mode == "z0":
            return self.evolve_population_z0(
                selected_indices,
                mutation_strength=mutation_strength,
                elite_count=elite_count,
                fresh_count=fresh_count,
            )

        print(f"\n[潜在ノイズモード] 第{self.population.generation_number + 1}世代を生成中...")
        print(f"選択された個体: {selected_indices}")

        if not selected_indices:
            raise ValueError("少なくとも1つの個体を選択してください")

        selected = [self.population.current_generation[i] for i in selected_indices]
        prompt = selected[0].metadata.get("prompt", "")

        from audioldm.iec import crossover_slerp, mutate_gaussian

        next_generation = []

        for i in range(min(elite_count, len(selected))):
            elite = selected[i].clone()
            elite.generation = self.population.generation_number + 1
            elite.metadata["elite"] = True
            next_generation.append(elite)

        while len(next_generation) < self.population_size:
            if len(selected) == 1:
                child = mutate_gaussian(selected[0], mutation_rate, mutation_strength)
            else:
                parent1, parent2 = np.random.choice(selected, size=2, replace=False)
                alpha = np.random.uniform(0.3, 0.7)
                child = crossover_slerp(parent1, parent2, alpha)
                if np.random.random() < mutation_rate:
                    child = mutate_gaussian(child, 1.0, mutation_strength)

            child.generation = self.population.generation_number + 1
            child.metadata["prompt"] = prompt
            next_generation.append(child)

        self.population.generation_number += 1
        self.population.current_generation = next_generation[:self.population_size]
        self.population.history.append([g.clone() for g in self.population.current_generation])

        genotypes = self.population.current_generation
        x_T_list = [g.latent_noise.to(self.device) for g in genotypes]
        waveforms = self._sample_audio_batch(x_T_list, text=prompt)
        return list(zip(genotypes, waveforms))
    
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
        
        for i, (_, waveform) in enumerate(generation_results):
            filename = f"{prefix}_gen{gen_num:03d}_ind{i:02d}.wav"
            filepath = os.path.join(output_dir, filename)
            
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
    
    for _ in range(max_generations):
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

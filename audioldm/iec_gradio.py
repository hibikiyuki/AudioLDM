"""
GradioベースのWebインターフェース for AudioLDM-IEC
対話型進化的効果音生成システムのUI
"""

import gradio as gr
import numpy as np
import torch
import os
from typing import List, Optional, Dict, Tuple
from datetime import datetime
import json

from audioldm.iec_pipeline import AudioLDM_IEC
from audioldm.iec import StyleTransferGenotype


class IECInterface:
    """
    Gradio UIのステート管理とロジック
    """
    
    def __init__(
        self,
        model_name: str = "audioldm-s-full-v2",
        population_size: int = 6,
        duration: float = 5.0,
        output_dir: str = "./output/iec_gradio"
    ):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        print("AudioLDM-IECシステムを初期化中...")
        self.iec_system = AudioLDM_IEC(
            model_name=model_name,
            population_size=population_size,
            duration=duration,
            ga_mode="latent",
        )
        
        # セッション状態
        self.current_results: List[Tuple] = []
        self.current_audio_paths: List[str] = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(output_dir, f"session_{self.session_id}")
        os.makedirs(self.session_dir, exist_ok=True)

        # スタイル転送用の状態
        self.st_results: List[Tuple] = []
        self.st_audio_paths: List[str] = []
        self.st_ranked_words: List[Tuple[str, float]] = []

        # ログ
        self.interaction_log = []
        
        print(f"セッションID: {self.session_id}")
        print(f"出力ディレクトリ: {self.session_dir}")
    
    def initialize_generation(
        self,
        prompt: str,
        variation_strength: float,
        ga_mode: str = "latent",
        transform_init_std: float = 0.3,
        base_noise_seed_str: str = "",
        progress=gr.Progress()
    ) -> Tuple[List, str, str]:
        """
        初期世代を生成
        
        Returns:
            (音声リスト, 情報テキスト, ステータスメッセージ)
        """
        progress(0, desc="初期個体群を生成中...")

        try:
            # 変換行列モード用パラメータを反映
            if ga_mode == "transform":
                self.iec_system.transform_init_std = transform_init_std
                self.iec_system._shared_base_noise = None  # 初期化時はリセット

            base_noise_seed: Optional[int] = None
            if base_noise_seed_str.strip():
                try:
                    base_noise_seed = int(base_noise_seed_str.strip())
                except ValueError:
                    pass

            effective_prompt = prompt.strip() if prompt.strip() else None

            self.current_results = self.iec_system.initialize_population(
                prompt=effective_prompt,
                variation_strength=variation_strength,
                ga_mode=ga_mode,
                base_noise_seed=base_noise_seed,
            )

            mode_labels = {"transform": "変換行列モード", "z0": "z0潜在表現モード", "latent": "潜在ノイズモード"}
            mode_label = mode_labels.get(ga_mode, "潜在ノイズモード")
            if effective_prompt:
                message = f"[{mode_label}] プロンプト '{effective_prompt}' から初期個体群を生成しました"
            else:
                message = f"[{mode_label}] ランダムな初期個体群を生成しました"
            
            progress(0.5, desc="音声を保存中...")
            
            # 音声を保存
            self.current_audio_paths = self.iec_system.save_generation_audio(
                self.current_results,
                output_dir=self.session_dir,
                prefix=f"gen{self.iec_system.population.generation_number:03d}"
            )
            
            # ログに記録
            self.interaction_log.append({
                "timestamp": datetime.now().isoformat(),
                "action": "initialize",
                "prompt": prompt,
                "generation": self.iec_system.population.generation_number,
                "population_size": len(self.current_results)
            })
            
            progress(1.0, desc="完了!")
            
            # 音声コンポーネント用のリストを作成
            audio_list = [path for path in self.current_audio_paths]
            
            # 情報テキスト
            info = self._get_generation_info()
            
            return audio_list, info, message
            
        except Exception as e:
            error_msg = f"エラーが発生しました: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return [], "", error_msg
    
    def evolve_generation(
        self,
        selected_checkboxes: List[int],
        mutation_rate: float,
        mutation_strength: float,
        elite_count: int,
        fresh_count: int = 1,
        crossover_mode: str = "z0",
        sdedit_strength: float = 0.0,
        progress=gr.Progress()
    ) -> Tuple[List, str, str]:
        """
        次世代を生成
        
        Returns:
            (音声リスト, 情報テキスト, ステータスメッセージ)
        """
        progress(0, desc="選択を確認中...")
        
        if not selected_checkboxes or len(selected_checkboxes) == 0:
            return self.current_audio_paths, self._get_generation_info(), "⚠️ 少なくとも1つの個体を選択してください"
        
        try:
            progress(0.2, desc=f"{len(selected_checkboxes)}個の親個体から次世代を生成中...")
            
            # 次世代を生成
            self.current_results = self.iec_system.evolve_population(
                selected_indices=selected_checkboxes,
                mutation_rate=mutation_rate,
                mutation_strength=mutation_strength,
                elite_count=elite_count,
                fresh_count=fresh_count,
                crossover_mode=crossover_mode,
                sdedit_strength=sdedit_strength,
            )
            
            progress(0.7, desc="音声を保存中...")
            
            # 音声を保存
            self.current_audio_paths = self.iec_system.save_generation_audio(
                self.current_results,
                output_dir=self.session_dir,
                prefix=f"gen{self.iec_system.population.generation_number:03d}"
            )
            
            # ログに記録
            self.interaction_log.append({
                "timestamp": datetime.now().isoformat(),
                "action": "evolve",
                "selected_indices": selected_checkboxes,
                "generation": self.iec_system.population.generation_number,
                "mutation_rate": mutation_rate,
                "mutation_strength": mutation_strength,
                "elite_count": elite_count,
                "fresh_count": fresh_count,
                "crossover_mode": crossover_mode,
                "sdedit_strength": sdedit_strength,
            })
            
            progress(1.0, desc="完了!")
            
            # 音声コンポーネント用のリストを作成
            audio_list = [path for path in self.current_audio_paths]
            
            # 情報テキスト
            info = self._get_generation_info()
            
            message = f"✅ 第{self.iec_system.population.generation_number}世代を生成しました (親個体: {len(selected_checkboxes)}個)"
            
            return audio_list, info, message
            
        except Exception as e:
            error_msg = f"エラーが発生しました: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return self.current_audio_paths, self._get_generation_info(), error_msg
    
    def rollback_generation(self, steps: int = 1) -> Tuple[List, str, str]:
        """
        指定世代数だけ戻る
        """
        try:
            genotypes = self.iec_system.population.rollback_generation(steps)
            
            self.current_results = []
            for genotype in genotypes:
                prompt = genotype.metadata.get("prompt", "")
                waveform = self.iec_system._generate_audio_from_any_genotype(genotype, text=prompt)
                self.current_results.append((genotype, waveform))
            
            # 音声を保存
            self.current_audio_paths = self.iec_system.save_generation_audio(
                self.current_results,
                output_dir=self.session_dir,
                prefix=f"gen{self.iec_system.population.generation_number:03d}_rollback"
            )
            
            # ログに記録
            self.interaction_log.append({
                "timestamp": datetime.now().isoformat(),
                "action": "rollback",
                "steps": steps,
                "generation": self.iec_system.population.generation_number
            })
            
            audio_list = [path for path in self.current_audio_paths]
            info = self._get_generation_info()
            message = f"🔙 第{self.iec_system.population.generation_number}世代にロールバックしました"
            
            return audio_list, info, message
            
        except Exception as e:
            error_msg = f"エラーが発生しました: {str(e)}"
            print(error_msg)
            return self.current_audio_paths, self._get_generation_info(), error_msg
    
    def save_session(self) -> str:
        """
        セッションを保存
        """
        try:
            # 履歴を保存
            history_path = os.path.join(self.session_dir, "iec_history.json")
            self.iec_system.population.save_history(history_path)
            
            # インタラクションログを保存
            log_path = os.path.join(self.session_dir, "interaction_log.json")
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(self.interaction_log, f, indent=2, ensure_ascii=False)
            
            message = f"✅ セッションを保存しました\n"
            message += f"📁 {self.session_dir}\n"
            message += f"- 履歴: {os.path.basename(history_path)}\n"
            message += f"- ログ: {os.path.basename(log_path)}\n"
            message += f"- 音声ファイル: {len(self.current_audio_paths)}個"
            
            return message
            
        except Exception as e:
            return f"エラー: {str(e)}"
    
    def _get_generation_info(self) -> str:
        """
        現在の世代情報を取得
        """
        info = self.iec_system.get_generation_info()

        mode_labels = {"transform": "変換行列GA", "z0": "z0潜在表現GA", "latent": "潜在ノイズGA"}
        mode_label = mode_labels.get(self.iec_system.ga_mode, "潜在ノイズGA")
        text = f"""
### 📊 現在の状態

- **世代番号**: {info['generation_number']}
- **個体数**: {info['population_size']}
- **履歴**: {info['history_length']} 世代
- **GA モード**: {mode_label}
- **セッションID**: {self.session_id}
- **インタラクション数**: {len(self.interaction_log)}
        """

        return text.strip()

    @staticmethod
    def _get_individual_status(genotype, index: int) -> str:
        """
        個体の生成元情報を人間が読めるテキストで返す
        """
        meta = genotype.metadata
        lines = []

        op = meta.get("operation", "")
        init = meta.get("initialization", "")
        is_elite = meta.get("elite", False)

        def _fmt_pop_idx(idx):
            return f"個体{idx}" if idx is not None else "?"

        if is_elite:
            parent_idx = meta.get("elite_parent_pop_index")
            src = f" (前世代の{_fmt_pop_idx(parent_idx)})" if parent_idx is not None else ""
            lines.append(f"⭐ エリート保存{src}")

        if init in ("random", "random_gaussian"):
            lines.append("🎲 初期ランダム生成")
        elif init == "from_prompt":
            lines.append("📝 プロンプトから初期生成")
        elif init == "ddim_sampling":
            lines.append("🎲 DDIM初期生成")
        elif op == "ddim_fresh_injection":
            lines.append("✨ DDIM新鮮注入")
        elif op == "crossover_then_mutate":
            p1_idx = _fmt_pop_idx(meta.get("crossover_parent1_pop_index"))
            p2_idx = _fmt_pop_idx(meta.get("crossover_parent2_pop_index"))
            strength = meta.get("mutation_strength", 0)
            ctype = meta.get("crossover_type", "slerp")
            if ctype == "uniform":
                cp = meta.get("crossover_p", 0.5)
                lines.append(f"🧬 一様交叉＋突然変異")
                lines.append(f"　親: {p1_idx} × {p2_idx}  交叉p={cp:.2f}  変異強度={strength:.3f}")
            else:
                alpha = meta.get("crossover_alpha", 0.5)
                sdedit = meta.get("sdedit_strength", 0.0)
                sdedit_tag = f"  SDEdit={sdedit:.2f}" if sdedit > 0 else ""
                lines.append(f"🧬 SLERP交叉＋突然変異")
                lines.append(f"　親: {p1_idx} × {p2_idx}  α={alpha:.2f}  変異強度={strength:.3f}{sdedit_tag}")
        elif op in ("mutate_gaussian", "mutate_transform_gaussian", "mutate_z0_gaussian"):
            parent_idx = _fmt_pop_idx(meta.get("parent_pop_index"))
            strength = meta.get("mutation_strength", 0)
            lines.append(f"🔀 突然変異")
            lines.append(f"　親: {parent_idx}  強度={strength:.3f}")
        elif op == "crossover_zt_slerp":
            alpha = meta.get("crossover_alpha", 0.5)
            strength = meta.get("sdedit_strength", 0.0)
            p1_idx = _fmt_pop_idx(meta.get("crossover_parent1_pop_index"))
            p2_idx = _fmt_pop_idx(meta.get("crossover_parent2_pop_index"))
            lines.append(f"🧬 z_t空間交叉")
            lines.append(f"　親: {p1_idx} × {p2_idx}  α={alpha:.2f}  ノイズ強度={strength:.2f}")
        elif op in ("crossover_slerp", "crossover_z0_slerp"):
            alpha = meta.get("crossover_alpha", 0.5)
            p1_idx = _fmt_pop_idx(meta.get("crossover_parent1_pop_index"))
            p2_idx = _fmt_pop_idx(meta.get("crossover_parent2_pop_index"))
            lines.append(f"🧬 SLERP交叉")
            lines.append(f"　親: {p1_idx} × {p2_idx}  α={alpha:.2f}")
        elif op == "crossover_matrix_uniform":
            cp = meta.get("crossover_p", 0.5)
            p1_idx = _fmt_pop_idx(meta.get("crossover_parent1_pop_index"))
            p2_idx = _fmt_pop_idx(meta.get("crossover_parent2_pop_index"))
            lines.append(f"🧬 一様交叉")
            lines.append(f"　親: {p1_idx} × {p2_idx}  p={cp:.2f}")

        if isinstance(genotype, StyleTransferGenotype):
            lines.append("🎨 スタイル転送")
            lines.append(f"  ノイズ強度: {genotype.noise_strength:.3f}")
            lines.append(f"  ガイダンス: {genotype.guidance_scale:.1f}")
            lines.append(f"  マスク: [{genotype.mask_start:.2f}, {genotype.mask_end:.2f}]")
            if is_elite:
                lines.append("⭐ エリート")
            return "\n".join(lines) if lines else "情報なし"

        seed = getattr(genotype, "seed", None)
        if seed is not None:
            lines.append(f"🌱 seed={seed}")

        return "\n".join(lines) if lines else "情報なし"

    def initialize_style_transfer(
        self,
        audio_a_path: Optional[str],
        audio_b_path: Optional[str],
        base_prompt: str,
        top_k: int,
        noise_min: float,
        noise_max: float,
        gs_min: float,
        gs_max: float,
        progress=gr.Progress(),
    ) -> Tuple[List, str, str, str]:
        """スタイル転送の初期個体群を生成する。

        Returns:
            (audio_paths, style_words_md, info, status_message)
        """
        if not audio_a_path or not audio_b_path:
            return self.st_audio_paths, "", "", "⚠️ 音声AとBの両方をアップロードしてください"

        try:
            progress(0.1, desc="スタイル語をランキング中...")
            self.st_results, self.st_ranked_words = \
                self.iec_system.initialize_style_transfer_population(
                    audio_a_path=audio_a_path,
                    audio_b_path=audio_b_path,
                    base_prompt=base_prompt,
                    top_k_styles=int(top_k),
                    noise_strength_range=(noise_min, noise_max),
                    guidance_scale_range=(gs_min, gs_max),
                )
            progress(0.9, desc="音声を保存中...")
            self.st_audio_paths = self.iec_system.save_generation_audio(
                self.st_results,
                output_dir=self.session_dir,
                prefix="st_gen000",
            )
            style_words_md = "| スタイル語 | スコア |\n|---|---|\n"
            style_words_md += "\n".join(
                f"| {w} | {s:.3f} |" for w, s in self.st_ranked_words)
            info = f"**第0世代** / 個体数={len(self.st_results)} / " \
                   f"スタイルプロンプト: _{self.st_results[0][0].style_prompt}_"
            progress(1.0, desc="完了!")
            return self.st_audio_paths, style_words_md, info, "✅ スタイル転送の初期個体群を生成しました"
        except Exception as e:
            import traceback; traceback.print_exc()
            return self.st_audio_paths, "", "", f"エラー: {str(e)}"

    def evolve_style_transfer(
        self,
        selected_labels: List[str],
        mutation_noise: float,
        mutation_gs: float,
        mutation_mask: float,
        elite_count: int,
        progress=gr.Progress(),
    ) -> Tuple[List, str, str, str]:
        """スタイル転送個体群を進化させる。

        Returns:
            (audio_paths, style_words_md, info, status_message)
        """
        if not self.st_results:
            return [], "", "", "⚠️ 先にスタイル転送を初期化してください"
        if not selected_labels:
            return self.st_audio_paths, "", "", "⚠️ 少なくとも1つの個体を選択してください"

        selected_indices = [int(label.split()[-1]) for label in selected_labels]
        try:
            progress(0.2, desc="次世代を生成中...")
            self.st_results = self.iec_system.evolve_style_transfer_population(
                selected_indices=selected_indices,
                mutation_noise_sigma=mutation_noise,
                mutation_gs_sigma=mutation_gs,
                mutation_mask_sigma=mutation_mask,
                elite_count=int(elite_count),
            )
            progress(0.9, desc="音声を保存中...")
            gen = self.iec_system.population.generation_number
            self.st_audio_paths = self.iec_system.save_generation_audio(
                self.st_results, output_dir=self.session_dir,
                prefix=f"st_gen{gen:03d}",
            )
            style_words_md = "| スタイル語 | スコア |\n|---|---|\n"
            style_words_md += "\n".join(
                f"| {w} | {s:.3f} |" for w, s in self.st_ranked_words)
            info = f"**第{gen}世代** / 個体数={len(self.st_results)}"
            progress(1.0, desc="完了!")
            return self.st_audio_paths, style_words_md, info, \
                   f"✅ 第{gen}世代を生成しました"
        except Exception as e:
            import traceback; traceback.print_exc()
            return self.st_audio_paths, "", "", f"エラー: {str(e)}"


def create_gradio_interface(
    model_name: str = "audioldm-s-full-v2",
    population_size: int = 6,
    duration: float = 5.0
) -> gr.Blocks:
    """
    Gradio UIを作成
    """
    # インターフェースの初期化
    interface = IECInterface(
        model_name=model_name,
        population_size=population_size,
        duration=duration
    )
    
    # カスタムCSS
    custom_css = """
    .audio-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 20px;
        margin: 20px 0;
    }
    .individual-card {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
        background-color: #f9f9f9;
    }
    .selected {
        border-color: #4CAF50;
        background-color: #e8f5e9;
    }
    """
    
    with gr.Blocks(title="Demo", css=custom_css) as demo:
        gr.Markdown("""
        ## 使い方
        1. **初期化**: プロンプトを入力して初期個体群を生成
        2. **選択**: 気に入った音声を選択
        3. **進化**: 次世代を生成してステップ2-3を繰り返す
        """)

        with gr.Tabs():

            # ===== 通常生成タブ =====
            with gr.Tab("通常生成"):
                with gr.Row():
                    with gr.Column(scale=2):
                        with gr.Group():
                            gr.Markdown("### 🚀 初期化")
                            prompt_input = gr.Textbox(
                                label="プロンプト (空欄でランダム生成)",
                                placeholder="",
                                value=""
                            )
                            ga_mode_radio = gr.Radio(
                                choices=["latent", "transform", "z0"],
                                value="latent",
                                label="GA モード",
                                info="latent: 潜在ノイズを直接進化 / transform: 変換行列を進化 / z0: DDIM逆拡散潜在表現を進化（子の評価が高速）"
                            )
                            variation_strength_slider = gr.Slider(
                                visible=False,
                                minimum=0.0, maximum=1.0, value=0.3, step=0.05,
                                label="初期変異強度",
                            )
                            with gr.Group() as transform_init_group:
                                transform_init_std_slider = gr.Slider(
                                    minimum=0.01, maximum=1.0, value=0.3, step=0.05,
                                    label="変換行列の初期摂動強度",
                                )
                                base_noise_seed_input = gr.Textbox(
                                    label="ベースノイズ seed (空欄でランダム)",
                                    placeholder="例: 42", value=""
                                )
                            init_button = gr.Button("🎲 初期個体群を生成", variant="primary", size="lg")

                        with gr.Group(visible=True):
                            gr.Markdown("### ⚙️ 進化パラメータ")
                            mutation_rate_slider = gr.Slider(
                                minimum=0.0, maximum=1.0, value=0.3, step=0.05,
                                label="突然変異率 (潜在ノイズモード用)",
                            )
                            mutation_strength_slider = gr.Slider(
                                minimum=0.0, maximum=0.5, value=0.15, step=0.05,
                                label="突然変異強度",
                            )
                            elite_count_slider = gr.Slider(
                                minimum=0, maximum=3, value=1, step=1,
                                label="エリート保存数",
                            )
                            fresh_count_slider = gr.Slider(
                                minimum=0, maximum=3, value=1, step=1,
                                label="DDIM 新鮮注入数 (z0モード専用)",
                            )
                            crossover_mode_radio = gr.Radio(
                                choices=["z0", "zt"], value="z0",
                                label="交叉モード (z0モード専用)",
                            )
                            sdedit_strength_slider = gr.Slider(
                                minimum=0.0, maximum=0.5, value=0.0, step=0.05,
                                label="SDEdit強度 (z0モード専用)",
                            )

                        with gr.Group():
                            gr.Markdown("### 🎮 コントロール")
                            evolve_button = gr.Button("🧬 次世代を生成", variant="primary", size="lg")
                            with gr.Row():
                                rollback_button = gr.Button("🔙 1世代戻る")
                                save_button = gr.Button("💾 セッション保存")

                    with gr.Column(scale=1):
                        info_display = gr.Markdown("### 📊 情報\n準備中...")
                        status_display = gr.Textbox(
                            label="ステータス", value="システム準備完了", interactive=False
                        )

                gr.Markdown("### 🎧 個体群 (音声を聴いて選択してください)")
                selection_group = gr.CheckboxGroup(
                    choices=[], label="選択する個体",
                    info="気に入った音声を選択してください(複数選択可)"
                )
                audio_components = []
                status_components = []
                with gr.Row():
                    for i in range(population_size):
                        with gr.Column():
                            audio = gr.Audio(label=f"個体 {i}", type="filepath", interactive=False)
                            status = gr.Textbox(
                                value="", label="生成元", interactive=False, lines=3, max_lines=4)
                            audio_components.append(audio)
                            status_components.append(status)

                audio_paths_state = gr.State([])

                def _get_status_list():
                    statuses = []
                    for i in range(population_size):
                        if i < len(interface.current_results):
                            genotype = interface.current_results[i][0]
                            statuses.append(IECInterface._get_individual_status(genotype, i))
                        else:
                            statuses.append("")
                    return statuses

                def _pack_outputs(audio_list, info, message):
                    choices = [f"個体 {i}" for i in range(len(audio_list))]
                    audio_outputs = audio_list + [None] * (population_size - len(audio_list))
                    status_outputs = _get_status_list()
                    return (audio_outputs
                            + [gr.CheckboxGroup(choices=choices, value=[]), info, message, audio_list]
                            + status_outputs)

                def init_wrapper(prompt, variation_strength, ga_mode, transform_init_std, base_noise_seed_str):
                    audio_list, info, message = interface.initialize_generation(
                        prompt, variation_strength, ga_mode, transform_init_std, base_noise_seed_str)
                    return _pack_outputs(audio_list, info, message)

                def evolve_wrapper(selected_labels, mutation_rate, mutation_strength,
                                   elite_count, fresh_count, crossover_mode, sdedit_strength):
                    selected_indices = [int(label.split()[-1]) for label in selected_labels]
                    audio_list, info, message = interface.evolve_generation(
                        selected_indices, mutation_rate, mutation_strength, elite_count, fresh_count,
                        crossover_mode=crossover_mode, sdedit_strength=sdedit_strength)
                    return _pack_outputs(audio_list, info, message)

                def rollback_wrapper():
                    audio_list, info, message = interface.rollback_generation(steps=1)
                    return _pack_outputs(audio_list, info, message)

                def save_wrapper():
                    return interface.save_session()

                all_outputs = (audio_components
                               + [selection_group, info_display, status_display, audio_paths_state]
                               + status_components)

                init_button.click(
                    fn=init_wrapper,
                    inputs=[prompt_input, variation_strength_slider, ga_mode_radio,
                            transform_init_std_slider, base_noise_seed_input],
                    outputs=all_outputs
                )
                evolve_button.click(
                    fn=evolve_wrapper,
                    inputs=[selection_group, mutation_rate_slider, mutation_strength_slider,
                            elite_count_slider, fresh_count_slider,
                            crossover_mode_radio, sdedit_strength_slider],
                    outputs=all_outputs
                )
                rollback_button.click(fn=rollback_wrapper, inputs=[], outputs=all_outputs)
                save_button.click(fn=save_wrapper, inputs=[], outputs=[status_display])

            # ===== スタイル変換タブ =====
            with gr.Tab("スタイル変換"):
                with gr.Row():
                    with gr.Column(scale=2):
                        with gr.Group():
                            gr.Markdown("### 🎵 音声入力")
                            st_audio_a = gr.Audio(label="音声A (コンテンツ元)", type="filepath")
                            st_audio_b = gr.Audio(label="音声B (スタイル参照)", type="filepath")
                            st_base_prompt = gr.Textbox(
                                label="ベースプロンプト (任意)", placeholder="例: music", value="")
                            st_topk_slider = gr.Slider(
                                minimum=3, maximum=10, value=5, step=1,
                                label="スタイル語数 Top-K",
                                info="音声Bから選出するスタイル語の数"
                            )

                        with gr.Group():
                            gr.Markdown("### ⚙️ 初期パラメータ範囲")
                            with gr.Row():
                                st_noise_min = gr.Slider(0.05, 0.5, value=0.1, step=0.05,
                                                         label="ノイズ強度 最小")
                                st_noise_max = gr.Slider(0.05, 0.5, value=0.4, step=0.05,
                                                         label="ノイズ強度 最大")
                            with gr.Row():
                                st_gs_min = gr.Slider(1.0, 15.0, value=3.0, step=0.5,
                                                      label="ガイダンス 最小")
                                st_gs_max = gr.Slider(1.0, 15.0, value=10.0, step=0.5,
                                                      label="ガイダンス 最大")

                        with gr.Group():
                            gr.Markdown("### 🔀 進化パラメータ")
                            st_mutation_noise = gr.Slider(0.0, 0.1, value=0.05, step=0.01,
                                                          label="突然変異 ノイズ強度σ")
                            st_mutation_gs = gr.Slider(0.0, 3.0, value=1.0, step=0.1,
                                                       label="突然変異 ガイダンスσ")
                            st_mutation_mask = gr.Slider(0.0, 0.1, value=0.05, step=0.01,
                                                         label="突然変異 マスクσ")
                            st_elite_count = gr.Slider(0, 3, value=1, step=1, label="エリート保存数")

                        with gr.Group():
                            gr.Markdown("### 🎮 コントロール")
                            st_init_button = gr.Button(
                                "🎨 スタイル転送を初期化", variant="primary", size="lg")
                            st_evolve_button = gr.Button(
                                "🧬 次世代を生成", variant="secondary", size="lg")

                    with gr.Column(scale=1):
                        st_style_words_display = gr.Markdown(
                            "#### スタイル語 (音声Bの特徴)\n初期化後に表示されます")
                        st_info_display = gr.Markdown("#### 情報\n準備中...")
                        st_status_display = gr.Textbox(
                            label="ステータス",
                            value="音声A・Bをアップロードして初期化してください",
                            interactive=False
                        )

                gr.Markdown("### 🎧 個体群 (スタイル変換結果)")
                st_selection_group = gr.CheckboxGroup(
                    choices=[], label="選択する個体",
                    info="気に入った結果を選択してください(複数選択可)"
                )
                st_audio_components = []
                st_status_components = []
                with gr.Row():
                    for i in range(population_size):
                        with gr.Column():
                            st_audio = gr.Audio(
                                label=f"個体 {i}", type="filepath", interactive=False)
                            st_status = gr.Textbox(
                                value="", label="遺伝子値", interactive=False,
                                lines=4, max_lines=5)
                            st_audio_components.append(st_audio)
                            st_status_components.append(st_status)

                def _get_st_status_list():
                    statuses = []
                    for i in range(population_size):
                        if i < len(interface.st_results):
                            genotype = interface.st_results[i][0]
                            statuses.append(IECInterface._get_individual_status(genotype, i))
                        else:
                            statuses.append("")
                    return statuses

                def _pack_st_outputs(audio_paths, style_words_md, info, message):
                    choices = [f"個体 {i}" for i in range(len(audio_paths))]
                    audio_outputs = audio_paths + [None] * (population_size - len(audio_paths))
                    status_outputs = _get_st_status_list()
                    return (audio_outputs
                            + [gr.CheckboxGroup(choices=choices, value=[]),
                               style_words_md, info, message]
                            + status_outputs)

                def st_init_wrapper(a_path, b_path, base_prompt, top_k,
                                    noise_min, noise_max, gs_min, gs_max):
                    paths, words_md, info, msg = interface.initialize_style_transfer(
                        a_path, b_path, base_prompt, top_k,
                        noise_min, noise_max, gs_min, gs_max)
                    return _pack_st_outputs(paths, words_md, info, msg)

                def st_evolve_wrapper(selected_labels, mutation_noise, mutation_gs,
                                      mutation_mask, elite_count):
                    paths, words_md, info, msg = interface.evolve_style_transfer(
                        selected_labels, mutation_noise, mutation_gs,
                        mutation_mask, elite_count)
                    return _pack_st_outputs(paths, words_md, info, msg)

                st_all_outputs = (st_audio_components
                                  + [st_selection_group,
                                     st_style_words_display,
                                     st_info_display,
                                     st_status_display]
                                  + st_status_components)

                st_init_button.click(
                    fn=st_init_wrapper,
                    inputs=[st_audio_a, st_audio_b, st_base_prompt, st_topk_slider,
                            st_noise_min, st_noise_max, st_gs_min, st_gs_max],
                    outputs=st_all_outputs
                )
                st_evolve_button.click(
                    fn=st_evolve_wrapper,
                    inputs=[st_selection_group, st_mutation_noise, st_mutation_gs,
                            st_mutation_mask, st_elite_count],
                    outputs=st_all_outputs
                )

    return demo


def launch_interface(
    model_name: str = "audioldm-s-full-v2",
    population_size: int = 6,
    duration: float = 5.0,
    share: bool = False,
    server_port: int = 7860
):
    """
    Gradioインターフェースを起動
    
    Args:
        model_name: AudioLDMモデル名
        population_size: 個体数
        duration: 音声長(秒)
        share: 公開リンクを生成するか
        server_port: サーバーポート
    """
    demo = create_gradio_interface(
        model_name=model_name,
        population_size=population_size,
        duration=duration
    )
    
    demo.launch(
        share=share,
        server_port=server_port,
        server_name="0.0.0.0"
    )


if __name__ == "__main__":
    # デフォルト設定で起動
    launch_interface(
        population_size=6,
        duration=5.0,
        share=False
    )

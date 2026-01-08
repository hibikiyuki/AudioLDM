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
        
        # IECシステムの初期化
        print("AudioLDM-IECシステムを初期化中...")
        self.iec_system = AudioLDM_IEC(
            model_name=model_name,
            population_size=population_size,
            duration=duration
        )
        
        # セッション状態
        self.current_results: List[Tuple] = []
        self.current_audio_paths: List[str] = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(output_dir, f"session_{self.session_id}")
        os.makedirs(self.session_dir, exist_ok=True)
        
        # ログ
        self.interaction_log = []
        
        print(f"セッションID: {self.session_id}")
        print(f"出力ディレクトリ: {self.session_dir}")
    
    def initialize_generation(
        self,
        prompt: str,
        variation_strength: float,
        progress=gr.Progress()
    ) -> Tuple[List, str, str]:
        """
        初期世代を生成
        
        Returns:
            (音声リスト, 情報テキスト, ステータスメッセージ)
        """
        progress(0, desc="初期個体群を生成中...")
        
        try:
            # 初期個体群を生成
            if prompt.strip() == "":
                self.current_results = self.iec_system.initialize_population(
                    prompt=None
                )
                message = "ランダムな初期個体群を生成しました"
            else:
                self.current_results = self.iec_system.initialize_population(
                    prompt=prompt,
                    variation_strength=variation_strength
                )
                message = f"プロンプト '{prompt}' から初期個体群を生成しました"
            
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
                elite_count=elite_count
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
                "elite_count": elite_count
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
            
            # 音声を再生成
            self.current_results = []
            for genotype in genotypes:
                waveform = self.iec_system._generate_audio_from_genotype(genotype)
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
        
        text = f"""
### 📊 現在の状態

- **世代番号**: {info['generation_number']}
- **個体数**: {info['population_size']}
- **履歴**: {info['history_length']} 世代
- **セッションID**: {self.session_id}
- **インタラクション数**: {len(self.interaction_log)}
        """
        
        return text.strip()


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
    
    with gr.Blocks(title="AudioLDM-IEC: 対話型進化的効果音生成", css=custom_css) as demo:
        gr.Markdown("""
        # 🎵 AudioLDM-IEC: 対話型進化的効果音生成システム
        
        このシステムは、進化計算を用いて理想の効果音を探索します。
        
        ## 使い方
        1. **初期化**: プロンプトを入力して初期個体群を生成
        2. **選択**: 気に入った音声を選択
        3. **進化**: 次世代を生成してステップ2-3を繰り返す
        4. **保存**: 満足したらセッションを保存
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                # 初期化セクション
                with gr.Group():
                    gr.Markdown("### 🚀 初期化")
                    prompt_input = gr.Textbox(
                        label="プロンプト (空欄でランダム生成)",
                        placeholder="例: 爆発音、雷の音、ドアが閉まる音",
                        value=""
                    )
                    variation_strength_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.3,
                        step=0.05,
                        label="初期変異強度",
                        info="プロンプトからの変化の大きさ"
                    )
                    init_button = gr.Button("🎲 初期個体群を生成", variant="primary", size="lg")
                
                # 進化パラメータ
                with gr.Group():
                    gr.Markdown("### ⚙️ 進化パラメータ")
                    mutation_rate_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.3,
                        step=0.05,
                        label="突然変異率",
                        info="変異が起こる確率"
                    )
                    mutation_strength_slider = gr.Slider(
                        minimum=0.0,
                        maximum=0.5,
                        value=0.15,
                        step=0.05,
                        label="突然変異強度",
                        info="変異の大きさ"
                    )
                    elite_count_slider = gr.Slider(
                        minimum=0,
                        maximum=3,
                        value=1,
                        step=1,
                        label="エリート保存数",
                        info="優秀個体をそのまま次世代に残す数"
                    )
                
                # コントロールボタン
                with gr.Group():
                    gr.Markdown("### 🎮 コントロール")
                    evolve_button = gr.Button("🧬 次世代を生成", variant="primary", size="lg")
                    
                    with gr.Row():
                        rollback_button = gr.Button("🔙 1世代戻る")
                        save_button = gr.Button("💾 セッション保存")
            
            with gr.Column(scale=1):
                # 情報表示
                info_display = gr.Markdown("### 📊 情報\n準備中...")
                status_display = gr.Textbox(
                    label="ステータス",
                    value="システム準備完了",
                    interactive=False
                )
        
        # 音声表示エリア
        gr.Markdown("### 🎧 個体群 (音声を聴いて選択してください)")
        
        # チェックボックスグループ
        selection_group = gr.CheckboxGroup(
            choices=[],
            label="選択する個体",
            info="気に入った音声を選択してください(複数選択可)"
        )
        
        # 音声プレーヤー(動的に生成)
        audio_components = []
        with gr.Row():
            for i in range(population_size):
                with gr.Column():
                    audio = gr.Audio(
                        label=f"個体 {i}",
                        type="filepath",
                        interactive=False
                    )
                    audio_components.append(audio)
        
        # 状態管理
        audio_paths_state = gr.State([])
        
        # イベントハンドラ
        def init_wrapper(prompt, variation_strength):
            audio_list, info, message = interface.initialize_generation(
                prompt, variation_strength
            )
            # チェックボックスの選択肢を更新
            choices = [f"個体 {i}" for i in range(len(audio_list))]
            # 音声コンポーネントを更新
            outputs = audio_list + [None] * (population_size - len(audio_list))
            # CheckboxGroupの更新: choices, value, info, message, state
            return outputs + [gr.CheckboxGroup(choices=choices, value=[]), info, message, audio_list]
        
        def evolve_wrapper(selected_labels, mutation_rate, mutation_strength, elite_count):
            # ラベルからインデックスを抽出
            selected_indices = [int(label.split()[-1]) for label in selected_labels]
            
            audio_list, info, message = interface.evolve_generation(
                selected_indices, mutation_rate, mutation_strength, elite_count
            )
            # チェックボックスの選択肢を更新
            choices = [f"個体 {i}" for i in range(len(audio_list))]
            # 音声コンポーネントを更新
            outputs = audio_list + [None] * (population_size - len(audio_list))
            # CheckboxGroupの更新: choices, value, info, message, state
            return outputs + [gr.CheckboxGroup(choices=choices, value=[]), info, message, audio_list]
        
        def rollback_wrapper():
            audio_list, info, message = interface.rollback_generation(steps=1)
            choices = [f"個体 {i}" for i in range(len(audio_list))]
            outputs = audio_list + [None] * (population_size - len(audio_list))
            # CheckboxGroupの更新: choices, value, info, message, state
            return outputs + [gr.CheckboxGroup(choices=choices, value=[]), info, message, audio_list]
        
        def save_wrapper():
            message = interface.save_session()
            return message
        
        # イベントの接続
        init_button.click(
            fn=init_wrapper,
            inputs=[prompt_input, variation_strength_slider],
            outputs=audio_components + [selection_group, info_display, status_display, audio_paths_state]
        )
        
        evolve_button.click(
            fn=evolve_wrapper,
            inputs=[selection_group, mutation_rate_slider, mutation_strength_slider, elite_count_slider],
            outputs=audio_components + [selection_group, info_display, status_display, audio_paths_state]
        )
        
        rollback_button.click(
            fn=rollback_wrapper,
            inputs=[],
            outputs=audio_components + [selection_group, info_display, status_display, audio_paths_state]
        )
        
        save_button.click(
            fn=save_wrapper,
            inputs=[],
            outputs=[status_display]
        )
        
        # 使用方法の説明
        gr.Markdown("""
        ---
        ## 📖 詳細ガイド
        
        ### パラメータの説明
        
        - **初期変異強度**: プロンプトから生成される初期個体群の多様性を制御します。大きいほど多様な音が生成されます。
        - **突然変異率**: 次世代で変異が起こる確率です。高いと探索範囲が広がりますが、収束が遅くなります。
        - **突然変異強度**: 変異の大きさです。高いと大きく変化します。
        - **エリート保存数**: 優秀な個体をそのまま次世代に残す数です。1-2が推奨です。
        
        ### Tips
        
        - 初期世代で多様性が低い場合は、突然変異率を上げてみてください。
        - 良い音が見つかったら、エリート保存で確実に残しましょう。
        - 探索が行き詰まったら、ロールバックして別の選択を試してみてください。
        - 定期的にセッションを保存することをお勧めします。
        
        ### 研究計画書について
        
        本システムは、**主観的評価に基づく対話型進化的効果音生成**の研究プロトタイプです。
        言語化困難な「理想の音」を、聴覚フィードバックと進化計算により探索します。
        """)
    
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

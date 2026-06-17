"""
CLAP-IEC デモ専用 Gradio UI
============================

二段階交互探索（x_Tガチャ ↔ CLAP-IEC）のデモシナリオを円滑に実行するための
専用インターフェース。conditioning モードに特化し、過去の GA モード／スタイル変換の
UI は一切含まない。

設計方針:
  - 左半分 = x_Tガチャ（音の質感を選ぶ）。プロンプト/seed/N を決めて候補を生成し、
    候補を1つ選んで IEC を開始する。
  - 右半分 = CLAP-IEC（音楽的方向を進化させる）。個体群を提示し、進化パラメータを
    設定して次世代を生成。必要なら「ガチャに戻る」で質感探索へ往復する。
  - スクロール最小・一画面完結を目指したコンパクトな二段組レイアウト。
  - 候補/個体は固定サイズのグリッドセルに表示し、個体数によらず1セルあたりの
    表示面積が常に一定（CSS Grid の固定トラックで担保）。
  - 選択は専用コンポーネントではなく、各セルのボタンをクリックして行う
    （選択中はボタンが primary 表示 + ✓ ラベルに変わる）。

ロジックは既存の :class:`audioldm.iec_gradio.IECInterface` をそのまま再利用する。
"""

import gradio as gr

from audioldm.iec_gradio import IECInterface


# x_Tガチャ候補の最大数（2列グリッド × 5行）。N スライダー範囲(4-10)に合わせる。
MAX_CANDIDATES = 10
# 候補グリッドの列数
CAND_COLS = 2


CUSTOM_CSS = """
/* 候補グリッド: 2列固定トラック。表示数によらず各セル幅 = 1/2 で一定 */
.cand-grid {
    display: grid !important;
    grid-template-columns: repeat(2, 1fr) !important;
    gap: 8px !important;
}
/* 個体グリッド: 2列固定トラック。個体数によらず各セル幅 = 1/2 で一定 */
.pop-grid {
    display: grid !important;
    grid-template-columns: repeat(2, 1fr) !important;
    gap: 8px !important;
}
/* セル共通: 固定の枠。min-width:0 でグリッド内の伸長を防ぐ */
.cell {
    border: 1px solid var(--border-color-primary) !important;
    border-radius: 8px !important;
    padding: 5px !important;
    min-width: 0 !important;
    overflow: visible !important;
}
.cell .wrap, .cell > div { min-width: 0 !important; overflow: visible !important; }
/* 個体の生成元キャプション: 高さ固定でセル高さを一定に保つ */
.cell-cap {
    font-size: 11px !important;
    line-height: 1.25 !important;
    height: 52px !important;
    overflow: hidden !important;
    margin: 2px 0 0 0 !important;
    color: var(--body-text-color-subdued) !important;
}
.cell-cap p { margin: 0 !important; }
/* 選択ボタンをコンパクトに */
.cell button { min-height: 30px !important; padding: 2px 4px !important; font-size: 12px !important; }
/* パネル見出し */
.panel-head { font-weight: 600; margin-bottom: 4px; }
footer { display: none !important; }
"""


def create_demo_interface(
    model_name: str = "audioldm-m-full",
    population_size: int = 6,
    duration: float = 5.0,
) -> gr.Blocks:
    """CLAP-IEC デモ専用インターフェースを構築する。"""
    interface = IECInterface(
        model_name=model_name,
        population_size=population_size,
        duration=duration,
    )
    # デモは常に conditioning モードで動作する
    interface.iec_system.ga_mode = "conditioning"
    POP = population_size

    with gr.Blocks(title="AudioLDM-IEC Demo", css=CUSTOM_CSS) as demo:
        gr.Markdown(
            "## 🎼 AudioLDM-IEC\n"
            "**左**で音の質感（x_T）を選び → **右**で音楽的方向（CLAP）を進化させる。"
        )

        # 状態: ガチャ選択 = 単一 index (未選択は -1), IEC選択 = index のリスト
        cand_selected_state = gr.State(-1)
        pop_selected_state = gr.State([])

        with gr.Row(equal_height=False):

            # ============================================================
            # 左パネル: x_Tガチャ（音の質感）
            # ============================================================
            with gr.Column(scale=1):
                gr.Markdown("### 🎯 x_Tガチャ — 音の質感を選ぶ", elem_classes=["panel-head"])

                with gr.Row():
                    prompt_box = gr.Textbox(
                        label="プロンプト", value="lofi hip hop",
                        placeholder="例: atmospheric dark jazz with muted trumpet",
                        scale=3,
                    )
                    n_slider = gr.Slider(
                        minimum=4, maximum=MAX_CANDIDATES, value=6, step=1,
                        label="候補数 N", scale=1,
                    )
                with gr.Row():
                    seed_box = gr.Textbox(
                        label="x_T seed (空欄でランダム)", value="", placeholder="例: 42",
                        scale=2,
                    )
                    gacha_button = gr.Button("🎯 x_Tを生成", variant="primary", scale=1)

                # --- 候補グリッド（固定2列） ---
                cand_cells, cand_audios, cand_buttons = [], [], []
                with gr.Column(elem_classes=["cand-grid"]):
                    for i in range(MAX_CANDIDATES):
                        with gr.Group(elem_classes=["cell"], visible=False) as cell:
                            audio = gr.Audio(
                                type="filepath", show_label=False,
                                show_download_button=True, show_share_button=False,
                                interactive=False, waveform_options=gr.WaveformOptions(
                                    show_recording_waveform=False),
                            )
                            btn = gr.Button(f"候補 {i}", variant="secondary", size="sm")
                        cand_cells.append(cell)
                        cand_audios.append(audio)
                        cand_buttons.append(btn)

                start_iec_button = gr.Button(
                    "✅ 選んだx_TでIECを開始 →", variant="primary", size="lg")
                gacha_status = gr.Textbox(
                    label="ガチャ状況", value="プロンプトを決めて「x_Tを生成」を押してください",
                    interactive=False, lines=2,
                )

            # ============================================================
            # 右パネル: CLAP-IEC（音楽的方向）
            # ============================================================
            with gr.Column(scale=1):
                gr.Markdown("### 🧬 CLAP-IEC — 音楽的方向を進化", elem_classes=["panel-head"])

                with gr.Accordion("⚙️ 進化パラメータ", open=False):
                    with gr.Row():
                        alpha_slider = gr.Slider(
                            minimum=0.0, maximum=0.7, value=0.4, step=0.01,
                            label="初期多様性 α (B-2 SLERP)",
                        )
                        elite_slider = gr.Slider(
                            minimum=0, maximum=3, value=1, step=1, label="エリート保存数",
                        )
                    with gr.Row():
                        p_mut_slider = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.5, step=0.05, label="変異確率 p_mut",
                        )
                        rand_slider = gr.Slider(
                            minimum=0, maximum=3, value=1, step=1, label="ランダム注入数",
                        )
                    with gr.Row():
                        mu_min_slider = gr.Slider(
                            minimum=0.01, maximum=0.30, value=0.10, step=0.01, label="変異 μ 下限",
                        )
                        mu_max_slider = gr.Slider(
                            minimum=0.01, maximum=0.40, value=0.25, step=0.01, label="変異 μ 上限",
                        )
                    weighted_checkbox = gr.Checkbox(
                        value=True, label="加重B-2サンプリング (CLAP高スコアのプロンプトを優先)",
                    )

                # --- 個体グリッド（固定2列） ---
                pop_cells, pop_audios, pop_buttons, pop_caps = [], [], [], []
                with gr.Column(elem_classes=["pop-grid"]):
                    for i in range(POP):
                        with gr.Group(elem_classes=["cell"], visible=False) as cell:
                            audio = gr.Audio(
                                type="filepath", show_label=False,
                                show_download_button=True, show_share_button=False,
                                interactive=False, waveform_options=gr.WaveformOptions(
                                    show_recording_waveform=False),
                            )
                            btn = gr.Button(f"個体 {i}", variant="secondary", size="sm")
                            cap = gr.Markdown("", elem_classes=["cell-cap"])
                        pop_cells.append(cell)
                        pop_audios.append(audio)
                        pop_buttons.append(btn)
                        pop_caps.append(cap)

                with gr.Row():
                    evolve_button = gr.Button("🧬 次世代を生成", variant="primary", scale=2)
                    back_button = gr.Button("🔄 ガチャに戻る", variant="secondary", scale=1)
                iec_info = gr.Markdown("")
                iec_status = gr.Textbox(
                    label="IEC状況", value="左でx_Tを選んでIECを開始してください",
                    interactive=False, lines=2,
                )
                convergence = gr.Markdown("")

                gr.Markdown("---")
                with gr.Row():
                    long_dur_slider = gr.Slider(
                        minimum=5, maximum=60, value=20, step=5,
                        label="長尺生成 長さ (秒)", scale=3,
                    )
                    long_gen_button = gr.Button("🎵 長尺生成", variant="secondary", scale=1)
                long_audio_output = gr.Audio(
                    label="長尺出力", type="filepath",
                    show_download_button=True, show_share_button=False,
                    interactive=False,
                )
                long_status = gr.Textbox(
                    label="", value="", interactive=False, lines=1, visible=False,
                )

        # ================================================================
        # 出力リスト定義
        # ================================================================
        gacha_outputs = (
            cand_cells + cand_audios + cand_buttons
            + [cand_selected_state, gacha_status, seed_box]
        )
        pop_outputs = (
            pop_cells + pop_audios + pop_buttons + pop_caps
            + [pop_selected_state, iec_status, iec_info, convergence]
        )

        # ================================================================
        # 出力ビルダー
        # ================================================================
        def build_gacha_outputs(audio_list, msg, seed):
            n = len(audio_list)
            cells, audios, btns = [], [], []
            for j in range(MAX_CANDIDATES):
                if j < n:
                    cells.append(gr.update(visible=True))
                    audios.append(gr.update(value=audio_list[j]))
                else:
                    cells.append(gr.update(visible=False))
                    audios.append(gr.update(value=None))
                btns.append(gr.update(variant="secondary", value=f"候補 {j}"))
            return cells + audios + btns + [-1, msg, seed]

        def build_pop_outputs(audio_list, info, msg, conv=""):
            n = len(audio_list)
            cells, audios, btns, caps = [], [], [], []
            for j in range(POP):
                if j < n:
                    geno = interface.current_results[j][0]
                    cap_text = IECInterface._get_individual_status(geno, j)
                    cells.append(gr.update(visible=True))
                    audios.append(gr.update(value=audio_list[j]))
                    caps.append(gr.update(value=cap_text.replace("\n", "  \n")))
                else:
                    cells.append(gr.update(visible=False))
                    audios.append(gr.update(value=None))
                    caps.append(gr.update(value=""))
                btns.append(gr.update(variant="secondary", value=f"個体 {j}"))
            return cells + audios + btns + caps + [[], msg, info, conv]

        # ================================================================
        # アクション
        # ================================================================
        def do_generate_gacha(prompt, n, seed_str):
            audio_list, msg, new_seed = interface.run_seed_selection(prompt, int(n), seed_str)
            return build_gacha_outputs(audio_list, msg, new_seed)

        def do_start_iec(cand_sel, alpha, weighted):
            if cand_sel is None or cand_sel < 0:
                noop = [gr.update() for _ in range(POP * 4)]
                return noop + [[], "⚠️ x_T候補を1つ選択してください", gr.update(), gr.update()]
            audio_list, info, msg, _seed = interface.select_seed_winner(
                f"候補 {cand_sel}", alpha, weighted_b2=weighted)
            return build_pop_outputs(audio_list, info, msg, "")

        def do_evolve(pop_sel, elite, p_mut, mu_min, mu_max, rand_n, weighted):
            if not pop_sel:
                noop = [gr.update() for _ in range(POP * 4)]
                return noop + [pop_sel, "⚠️ 少なくとも1つの個体を選択してください",
                               gr.update(), gr.update()]
            audio_list, info, msg, conv = interface.evolve_generation(
                pop_sel,
                mutation_rate=0.0, mutation_strength=0.0,
                elite_count=int(elite), fresh_count=0,
                crossover_mode="z0", sdedit_strength=0.0,
                p_mut=float(p_mut), mutation_mu_min=float(mu_min),
                mutation_mu_max=float(mu_max),
                random_sample_count=int(rand_n), random_b1_count=0,
                x_T_mode="shared", weighted_b2=weighted,
            )
            return build_pop_outputs(audio_list, info, msg, conv)

        def do_back_to_gacha(pop_sel, n):
            if not pop_sel or len(pop_sel) != 1:
                noop_cells = [gr.update() for _ in range(MAX_CANDIDATES * 3)]
                return noop_cells + [
                    -1, "⚠️ c*として継承する個体を1つだけ選択してください", gr.update()]
            audio_list, msg, seed = interface.return_to_gacha(pop_sel, int(n))
            return build_gacha_outputs(audio_list, msg, seed)

        def do_generate_long(pop_sel, dur):
            path, msg = interface.generate_long_audio(pop_sel, float(dur))
            return path, gr.update(value=msg, visible=True)

        # --- 選択トグル（候補: 単一選択） ---
        def make_cand_select(idx):
            def _fn():
                btns = []
                for j in range(MAX_CANDIDATES):
                    if j == idx:
                        btns.append(gr.update(variant="primary", value=f"✓ 候補 {j}"))
                    else:
                        btns.append(gr.update(variant="secondary", value=f"候補 {j}"))
                return btns + [idx]
            return _fn

        for i, btn in enumerate(cand_buttons):
            btn.click(fn=make_cand_select(i), inputs=[],
                      outputs=cand_buttons + [cand_selected_state])

        # --- 選択トグル（個体: 複数選択） ---
        def make_pop_toggle(idx):
            def _fn(selected):
                s = set(selected or [])
                if idx in s:
                    s.discard(idx)
                else:
                    s.add(idx)
                s = sorted(s)
                btns = []
                for j in range(POP):
                    if j in s:
                        btns.append(gr.update(variant="primary", value=f"✓ 個体 {j}"))
                    else:
                        btns.append(gr.update(variant="secondary", value=f"個体 {j}"))
                return btns + [s]
            return _fn

        for i, btn in enumerate(pop_buttons):
            btn.click(fn=make_pop_toggle(i), inputs=[pop_selected_state],
                      outputs=pop_buttons + [pop_selected_state])

        # --- メインアクションの結線 ---
        gacha_button.click(
            fn=do_generate_gacha,
            inputs=[prompt_box, n_slider, seed_box],
            outputs=gacha_outputs,
        )
        start_iec_button.click(
            fn=do_start_iec,
            inputs=[cand_selected_state, alpha_slider, weighted_checkbox],
            outputs=pop_outputs,
        )
        evolve_button.click(
            fn=do_evolve,
            inputs=[pop_selected_state, elite_slider, p_mut_slider,
                    mu_min_slider, mu_max_slider, rand_slider, weighted_checkbox],
            outputs=pop_outputs,
        )
        back_button.click(
            fn=do_back_to_gacha,
            inputs=[pop_selected_state, n_slider],
            outputs=gacha_outputs,
        )
        long_gen_button.click(
            fn=do_generate_long,
            inputs=[pop_selected_state, long_dur_slider],
            outputs=[long_audio_output, long_status],
        )

    return demo


def launch_demo_interface(
    model_name: str = "audioldm-m-full",
    population_size: int = 6,
    duration: float = 2.5,
    share: bool = False,
    server_port: int = 8080,
):
    """デモ専用インターフェースを起動する。"""
    demo = create_demo_interface(
        model_name=model_name,
        population_size=population_size,
        duration=duration,
    )
    demo.launch(share=share, server_port=server_port, server_name="0.0.0.0")


if __name__ == "__main__":
    launch_demo_interface()

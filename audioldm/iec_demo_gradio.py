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
from audioldm.prompt_pool import DEFAULT_DEMO_PROMPT, DEMO_PROMPT_EXAMPLES


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

/* ===== 区間スライダー（つまみ2つ・自作） ===== */
.rs-box { margin: 6px 4px 2px 4px; }
.rs-label {
    font-size: 12px; color: var(--body-text-color-subdued);
    display: flex; justify-content: space-between; align-items: baseline;
    margin-bottom: 2px;
}
.rs-readout { font-weight: 600; color: var(--body-text-color); }
.rs-wrap { position: relative; height: 30px; }
.rs-track {
    position: absolute; top: 12px; left: 0; right: 0; height: 6px;
    border-radius: 3px; background: var(--border-color-primary);
}
.rs-range {
    position: absolute; top: 12px; height: 6px; border-radius: 3px;
    background: var(--color-accent, #f97316);
}
.rs-wrap input[type=range] {
    position: absolute; top: 4px; left: 0; width: 100%; height: 22px; margin: 0;
    -webkit-appearance: none; appearance: none;
    background: transparent; pointer-events: none;
}
.rs-wrap input[type=range]:focus { outline: none; }
/* トラックはCSSで描くので入力自体のトラックは透明。つまみだけ操作可能にする */
.rs-wrap input[type=range]::-webkit-slider-runnable-track {
    background: transparent; border: none; height: 22px;
}
.rs-wrap input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none; pointer-events: auto;
    width: 16px; height: 16px; margin-top: 3px; border-radius: 50%;
    background: var(--color-accent, #f97316); border: 2px solid #fff;
    cursor: pointer; box-shadow: 0 1px 3px rgba(0,0,0,.4);
}
.rs-wrap input[type=range]::-moz-range-track {
    background: transparent; border: none; height: 22px;
}
.rs-wrap input[type=range]::-moz-range-thumb {
    pointer-events: auto;
    width: 14px; height: 14px; border-radius: 50%;
    background: var(--color-accent, #f97316); border: 2px solid #fff;
    cursor: pointer;
}
/* Gradioの数値入力はDOMに残したまま視覚的に隠す（JSから値を同期するため） */
.rs-hidden {
    position: absolute !important; width: 1px !important; height: 1px !important;
    padding: 0 !important; margin: -1px !important; overflow: hidden !important;
    clip: rect(0,0,0,0) !important; white-space: nowrap !important; border: 0 !important;
}
"""

# 区間スライダー本体。Gradio 4.44 には RangeSlider が無いため自作する。
# 2本の input[type=range] を重ね、つまみだけ pointer-events:auto にして
# 「1ウィジェットで開始・終了の両端を指定」を実現する。
RANGE_SLIDER_HTML = """
<div class="rs-box" id="regen_range_widget">
  <div class="rs-label">
    <span>再生成する区間（区間外は固定）</span>
    <span class="rs-readout" id="rsReadout">50% – 100%</span>
  </div>
  <div class="rs-wrap">
    <div class="rs-track"></div>
    <div class="rs-range" id="rsRange"></div>
    <input type="range" id="rsStart" min="0" max="100" step="5" value="50"
           aria-label="再生成する区間の開始">
    <input type="range" id="rsEnd" min="0" max="100" step="5" value="100"
           aria-label="再生成する区間の終了">
  </div>
</div>
"""

# 区間スライダー → 隠しGradio数値入力への同期。
# Gradio はコンポーネントを非同期マウントするため、要素が現れるまでポーリングする。
RANGE_SLIDER_JS = """
() => {
  const MINGAP = 5;
  function setGradioNumber(wrapId, val) {
    const wrap = document.getElementById(wrapId);
    if (!wrap) return;
    const input = wrap.querySelector('input');
    if (!input) return;
    if (String(input.value) === String(val)) return;
    // ネイティブsetter経由で入れないとフレームワーク側が変更を検知しないことがある
    const setter = Object.getOwnPropertyDescriptor(
      window.HTMLInputElement.prototype, 'value').set;
    setter.call(input, String(val));
    input.dispatchEvent(new Event('input', { bubbles: true }));
    input.dispatchEvent(new Event('change', { bubbles: true }));
  }
  function init() {
    const s = document.getElementById('rsStart');
    const e = document.getElementById('rsEnd');
    const bar = document.getElementById('rsRange');
    const out = document.getElementById('rsReadout');
    if (!s || !e || !bar || !out) return false;
    if (s.dataset.rsBound === '1') return true;
    s.dataset.rsBound = '1';
    function render() {
      const a = parseInt(s.value, 10), b = parseInt(e.value, 10);
      bar.style.left = a + '%';
      bar.style.width = Math.max(0, b - a) + '%';
      out.textContent = a + '% – ' + b + '%';
      setGradioNumber('regen_start_num', a);
      setGradioNumber('regen_end_num', b);
    }
    s.addEventListener('input', () => {
      let a = parseInt(s.value, 10);
      const b = parseInt(e.value, 10);
      if (a > b - MINGAP) { a = Math.max(0, b - MINGAP); s.value = a; }
      render();
    });
    e.addEventListener('input', () => {
      let b = parseInt(e.value, 10);
      const a = parseInt(s.value, 10);
      if (b < a + MINGAP) { b = Math.min(100, a + MINGAP); e.value = b; }
      render();
    });
    render();
    return true;
  }
  if (!init()) {
    const iv = setInterval(() => { if (init()) clearInterval(iv); }, 200);
    setTimeout(() => clearInterval(iv), 15000);
  }
}
"""


def create_demo_interface(
    model_name: str = "audioldm-m-full",
    population_size: int = 6,
    duration: float = 5.0,
    mode: str = "two_axis",
    condition: str = None,
    participant_id: str = None,
    target_prompt_id: str = None,
    order: str = None,
    output_dir: str = "./output/iec_gradio",
    injection_band=None,
    prompt_pool=None,
    translate_backend="auto",
) -> gr.Blocks:
    """CLAP-IEC デモ専用インターフェースを構築する。

    mode:
      - "two_axis"   : 提案手法（x_Tガチャ ↔ CLAP-IEC の2軸交互探索）。
      - "single_axis": 対照条件（CLAP-IEC 単独）。x_T はシステムが
        ランダムに1つ固定（shared）し、ユーザは意味軸 c のみを進化させる。
        左パネル（x_Tガチャ）と「ガチャに戻る」を無効化する。
    """
    is_single = (mode == "single_axis")
    interface = IECInterface(
        model_name=model_name,
        population_size=population_size,
        duration=duration,
        output_dir=output_dir,
        condition=condition,
        participant_id=participant_id,
        target_prompt_id=target_prompt_id,
        order=order,
        injection_band=injection_band,
        prompt_pool=prompt_pool,
        translate_backend=translate_backend,
    )
    # デモは常に conditioning モードで動作する
    interface.iec_system.ga_mode = "conditioning"
    POP = population_size

    with gr.Blocks(title="AudioLDM-IEC Demo", css=CUSTOM_CSS,
                   js=RANGE_SLIDER_JS) as demo:
        if is_single:
            gr.Markdown(
                "## 🎼 AudioLDM-IEC（方法2）\n"
                "プロンプトを決めて開始し、提示された候補から好みを選んで進化させる。"
            )
        else:
            gr.Markdown(
                "## 🎼 AudioLDM-IEC（方法1）\n"
                "**左**で音の質感（x_T）を選び → **右**で音楽的方向（CLAP）を進化させる。"
            )

        # 状態: ガチャ選択 = 単一 index (未選択は -1), IEC選択 = index のリスト
        cand_selected_state = gr.State(-1)
        pop_selected_state = gr.State([])

        with gr.Row(equal_height=False):

            # ============================================================
            # 左パネル: x_Tガチャ（音の質感）— 単軸条件(方法2)では非表示
            # ============================================================
            with gr.Column(scale=1, visible=not is_single):
                gr.Markdown("### 🎯 初期化 / x_Tガチャ", elem_classes=["panel-head"])

                # 開始方法。「おまかせ」は初期化フェーズ（候補ごとに異なる c をプールから引く）。
                # プロンプト入力を一度も要求せずに出発点を選べる。
                start_mode_radio = gr.Radio(
                    choices=[
                        ("おまかせ（プロンプト入力なし）", "free"),
                        ("プロンプトから", "prompt"),
                    ],
                    value="free",
                    label="開始方法",
                )

                with gr.Row(visible=False) as prompt_row:
                    prompt_box = gr.Textbox(
                        label="プロンプト", value=DEFAULT_DEMO_PROMPT,
                        placeholder="例: acoustic guitar, warm and gentle",
                        scale=3,
                    )
                    # 英語を思いつけない来場者向け。選ぶと prompt_box に流し込むだけで、
                    # prompt_box 自体は手入力可能なまま残す。
                    preset_dropdown = gr.Dropdown(
                        choices=DEMO_PROMPT_EXAMPLES, value=DEFAULT_DEMO_PROMPT,
                        label="例から選ぶ", filterable=False,
                        allow_custom_value=False, scale=2,
                    )
                with gr.Row():
                    seed_box = gr.Textbox(
                        label="x_T seed (空欄でランダム)", value="", placeholder="例: 42",
                        scale=2,
                    )
                    # 候補数 N は第2階段（IEC個体数 POP）とは独立。初期化・ガチャは
                    # 1回だけの操作なので、毎世代の負担が乗る第2段階とは設計要件が違う。
                    n_slider = gr.Slider(
                        minimum=2, maximum=MAX_CANDIDATES, value=POP, step=1,
                        label=f"候補数 N（IEC個体数 {POP} とは独立）", scale=1,
                    )
                    gacha_button = gr.Button("🎲 候補を生成", variant="primary", scale=1)
                # 再生成する区間 [開始%, 終了%]。区間外は固定される。
                #   50-100=後半 / 30-70=中間(両端固定) / 0-50=前半
                # 見た目はつまみ2つの自作スライダー、値は下の隠し Number が保持する。
                gr.HTML(RANGE_SLIDER_HTML)
                regen_start_num = gr.Number(
                    value=50, elem_id="regen_start_num",
                    elem_classes=["rs-hidden"], show_label=False, container=False)
                regen_end_num = gr.Number(
                    value=100, elem_id="regen_end_num",
                    elem_classes=["rs-hidden"], show_label=False, container=False)
                variation_button = gr.Button(
                    "🔁 選んだ区間を振り直す", variant="secondary")

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
                    "✅ 選んだ候補でIECを開始 →", variant="primary", size="lg")
                gacha_status = gr.Textbox(
                    label="状況", value="「候補を生成」を押してください（プロンプト入力は不要）",
                    interactive=False, lines=3,
                )

            # ============================================================
            # 右パネル: CLAP-IEC（音楽的方向）
            # ============================================================
            with gr.Column(scale=1):
                gr.Markdown("### 🧬 CLAP-IEC — 音楽的方向を進化", elem_classes=["panel-head"])

                # 単軸条件(方法2)用の開始パネル。x_T をランダム固定して直接IEC開始する。
                with gr.Row(visible=is_single) as single_start_row:
                    single_prompt_box = gr.Textbox(
                        label="プロンプト", value=DEFAULT_DEMO_PROMPT,
                        placeholder="例: acoustic guitar, warm and gentle",
                        scale=3,
                    )
                    single_preset_dropdown = gr.Dropdown(
                        choices=DEMO_PROMPT_EXAMPLES, value=DEFAULT_DEMO_PROMPT,
                        label="例から選ぶ", filterable=False,
                        allow_custom_value=False, scale=2,
                    )
                    single_start_button = gr.Button(
                        "▶ 開始", variant="primary", scale=1)

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
                    back_button = gr.Button(
                        "🔄 ガチャに戻る", variant="secondary", scale=1, visible=not is_single)
                    # 区間は左パネルの regen_start/end スライダーを共用する
                    # （スライダーを重複配置すると被験者が混乱するため）
                    lock_head_checkbox = gr.Checkbox(
                        label="区間外を固定", value=False, scale=1, visible=not is_single)
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
        def do_generate_gacha(prompt, n, seed_str, start_mode):
            # "free" = 初期化フェーズ（候補ごとに異なる c をプールから引く）
            per_candidate_c = (start_mode == "free")
            audio_list, msg, new_seed = interface.run_seed_selection(
                prompt, int(n), seed_str, per_candidate_c=per_candidate_c)
            return build_gacha_outputs(audio_list, msg, new_seed)

        def toggle_start_mode(start_mode):
            """「プロンプトから」を選んだときだけプロンプト入力欄を出す。"""
            return gr.update(visible=(start_mode == "prompt"))

        def do_variation_gacha(cand_sel, n, regen_start_pct, regen_end_pct):
            # インペイントには参照となる既存の候補が必要なので、必ず1つ選ばせる
            if cand_sel is None or cand_sel < 0:
                noop = [gr.update() for _ in range(MAX_CANDIDATES * 3)]
                return noop + [-1, "⚠️ 塗り直したい候補を1つ選択してください", gr.update()]
            audio_list, msg, seed = interface.run_variation_gacha(
                int(cand_sel), int(n),
                float(regen_start_pct) / 100.0, float(regen_end_pct) / 100.0)
            return build_gacha_outputs(audio_list, msg, seed)

        def do_start_iec(cand_sel, alpha, weighted):
            if cand_sel is None or cand_sel < 0:
                noop = [gr.update() for _ in range(POP * 4)]
                return noop + [[], "⚠️ x_T候補を1つ選択してください", gr.update(), gr.update()]
            audio_list, info, msg, _seed = interface.select_seed_winner(
                f"候補 {cand_sel}", alpha, weighted_b2=weighted)
            return build_pop_outputs(audio_list, info, msg, "")

        def do_start_single(prompt, alpha, weighted):
            # 単軸条件(方法2): x_T をランダムに1つ固定(shared)し、conditioning個体群を初期化する。
            # ガチャ(Phase 1)を経由せず、ユーザは意味軸 c のみを進化させる。
            audio_list, info, msg, _seed = interface.initialize_generation(
                prompt=prompt,
                variation_strength=0.0,
                ga_mode="conditioning",
                cond_slerp_alpha=float(alpha),
                cond_x_T_seed_str="",
                x_T_mode="shared",
                weighted_b2=weighted,
            )
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

        def do_back_to_gacha(pop_sel, n, lock_region, regen_start_pct, regen_end_pct):
            if not pop_sel or len(pop_sel) != 1:
                noop_cells = [gr.update() for _ in range(MAX_CANDIDATES * 3)]
                return noop_cells + [
                    -1, "⚠️ c*として継承する個体を1つだけ選択してください", gr.update()]
            audio_list, msg, seed = interface.return_to_gacha(
                pop_sel, int(n), lock_region=bool(lock_region),
                regen_start=float(regen_start_pct) / 100.0,
                regen_end=float(regen_end_pct) / 100.0)
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

        # --- プロンプト例プルダウン → テキストボックス ---
        # プルダウンは「入力補助」であり真の入力源ではない。値を流し込んだあとは
        # テキストボックス側を自由に書き換えられる（生成時に読むのは常に textbox）。
        def apply_preset(choice):
            return gr.update() if not choice else gr.update(value=choice)

        preset_dropdown.change(
            fn=apply_preset, inputs=[preset_dropdown], outputs=[prompt_box])
        single_preset_dropdown.change(
            fn=apply_preset, inputs=[single_preset_dropdown],
            outputs=[single_prompt_box])

        # --- メインアクションの結線 ---
        start_mode_radio.change(
            fn=toggle_start_mode,
            inputs=[start_mode_radio],
            outputs=[prompt_row],
        )
        gacha_button.click(
            fn=do_generate_gacha,
            inputs=[prompt_box, n_slider, seed_box, start_mode_radio],
            outputs=gacha_outputs,
        )
        variation_button.click(
            fn=do_variation_gacha,
            inputs=[cand_selected_state, n_slider, regen_start_num, regen_end_num],
            outputs=gacha_outputs,
        )
        start_iec_button.click(
            fn=do_start_iec,
            inputs=[cand_selected_state, alpha_slider, weighted_checkbox],
            outputs=pop_outputs,
        )
        single_start_button.click(
            fn=do_start_single,
            inputs=[single_prompt_box, alpha_slider, weighted_checkbox],
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
            inputs=[pop_selected_state, n_slider, lock_head_checkbox,
                    regen_start_num, regen_end_num],
            outputs=gacha_outputs,
        )
        long_gen_button.click(
            fn=do_generate_long,
            inputs=[pop_selected_state, long_dur_slider],
            outputs=[long_audio_output, long_status],
        )

    return demo


def create_text_baseline_interface(
    model_name: str = "audioldm-m-full",
    population_size: int = 6,
    duration: float = 5.0,
    condition: str = None,
    participant_id: str = None,
    target_prompt_id: str = None,
    order: str = None,
    output_dir: str = "./output/iec_gradio",
    injection_band=None,
    prompt_pool=None,
    translate_backend="auto",
) -> gr.Blocks:
    """テキスト手打ちベースライン専用インターフェース（ユーザスタディ 条件B）。

    提案手法と同じ AudioLDM バックエンドで、プロンプト入力 → 生成（毎回新seedで
    population_size 個）→ 試聴 → 書き換え → 再生成 のループを提供する。進化・選択の
    次世代反映は行わない純粋な text-to-audio。ユーザは気に入った候補を最終候補として
    確保し、最後に確定する。
    """
    interface = IECInterface(
        model_name=model_name,
        population_size=population_size,
        duration=duration,
        output_dir=output_dir,
        condition=condition,
        participant_id=participant_id,
        target_prompt_id=target_prompt_id,
        order=order,
        injection_band=injection_band,
        prompt_pool=prompt_pool,
        translate_backend=translate_backend,
    )
    interface.iec_system.ga_mode = "conditioning"
    POP = population_size

    with gr.Blocks(title="AudioLDM Text Baseline", css=CUSTOM_CSS) as demo:
        gr.Markdown(
            "## ⌨️ AudioLDM（方法2）\n"
            "プロンプトを入力して生成 → 気に入った音を最終候補に。"
            "思う音にならなければ**プロンプトを書き換えて再生成**してください。"
        )

        final_pick_state = gr.State(None)

        with gr.Row():
            prompt_box = gr.Textbox(
                label="プロンプト（日本語で入力できます）", value="",
                placeholder="欲しい音を言葉で表現してください（例: 暖かくてやわらかいピアノ／warm mellow piano）",
                scale=4,
            )
            gen_button = gr.Button("🎲 生成", variant="primary", scale=1)

        # --- 候補グリッド（固定2列） ---
        cells, audios, buttons = [], [], []
        with gr.Column(elem_classes=["pop-grid"]):
            for i in range(POP):
                with gr.Group(elem_classes=["cell"], visible=False) as cell:
                    audio = gr.Audio(
                        type="filepath", show_label=False,
                        show_download_button=True, show_share_button=False,
                        interactive=False, waveform_options=gr.WaveformOptions(
                            show_recording_waveform=False),
                    )
                    btn = gr.Button(f"★ 最終候補にする {i}", variant="secondary", size="sm")
                cells.append(cell)
                audios.append(audio)
                buttons.append(btn)

        status = gr.Textbox(
            label="状況", value="プロンプトを入力して「生成」を押してください",
            interactive=False, lines=2,
        )
        info_md = gr.Markdown("")

        gr.Markdown("---\n### ✅ 最終候補")
        final_audio = gr.Audio(
            label="現在の最終候補（再生成しても保持されます）", type="filepath",
            show_download_button=True, show_share_button=False, interactive=False,
        )
        finalize_button = gr.Button("✅ これで決定して保存", variant="primary")
        final_status = gr.Textbox(label="", value="", interactive=False, lines=2, visible=False)

        gen_outputs = cells + audios + buttons + [status, info_md]

        def build_outputs(audio_list, msg, info):
            n = len(audio_list)
            cs, aus, bs = [], [], []
            for j in range(POP):
                if j < n:
                    cs.append(gr.update(visible=True))
                    aus.append(gr.update(value=audio_list[j]))
                else:
                    cs.append(gr.update(visible=False))
                    aus.append(gr.update(value=None))
                bs.append(gr.update(variant="secondary", value=f"★ 最終候補にする {j}"))
            return cs + aus + bs + [msg, info]

        def do_gen(prompt):
            audio_list, info, msg = interface.generate_text_baseline(prompt)
            return build_outputs(audio_list, msg, info)

        gen_button.click(fn=do_gen, inputs=[prompt_box], outputs=gen_outputs)

        # --- 最終候補の単一選択（再生成をまたいで path で保持）---
        def make_pick(idx):
            def _fn(prompt):
                paths = interface.text_baseline_audio_paths
                if idx >= len(paths):
                    return [gr.update() for _ in range(POP)] + [gr.update(), gr.update()]
                pick = {
                    "prompt": (prompt or "").strip(),
                    "round": interface._text_baseline_round,
                    "index": idx,
                    "path": paths[idx],
                }
                btns = []
                for j in range(POP):
                    if j == idx:
                        btns.append(gr.update(variant="primary", value=f"✓ 最終候補 {j}"))
                    else:
                        btns.append(gr.update(variant="secondary", value=f"★ 最終候補にする {j}"))
                return btns + [pick, gr.update(value=paths[idx])]
            return _fn

        for i, btn in enumerate(buttons):
            btn.click(fn=make_pick(i), inputs=[prompt_box],
                      outputs=buttons + [final_pick_state, final_audio])

        def do_finalize(final_pick):
            if not final_pick:
                return gr.update(value="⚠️ 先に最終候補を1つ選んでください", visible=True)
            interface.finalize_text_baseline(final_pick)
            save_msg = interface.save_session()
            return gr.update(
                value=f"✅ 最終候補を確定し、セッションを保存しました\n{save_msg}",
                visible=True,
            )

        finalize_button.click(fn=do_finalize, inputs=[final_pick_state], outputs=[final_status])

    return demo


def launch_demo_interface(
    model_name: str = "audioldm-m-full",
    population_size: int = 6,
    duration: float = 2.5,
    share: bool = False,
    server_port: int = 8080,
    mode: str = "two_axis",
    condition: str = None,
    participant_id: str = None,
    target_prompt_id: str = None,
    order: str = None,
    output_dir: str = "./output/iec_gradio",
    injection_band=None,
    prompt_pool=None,
    translate_backend="auto",
):
    """デモ専用インターフェースを起動する（mode によりUIを切り替える）。"""
    if mode == "text_baseline":
        # 手打ちベースラインは意味方向プールを使わないため injection_band は渡さない
        demo = create_text_baseline_interface(
            model_name=model_name,
            population_size=population_size,
            duration=duration,
            condition=condition,
            participant_id=participant_id,
            target_prompt_id=target_prompt_id,
            order=order,
            output_dir=output_dir,
            translate_backend=translate_backend,
        )
    else:
        demo = create_demo_interface(
            model_name=model_name,
            population_size=population_size,
            duration=duration,
            mode=mode,
            condition=condition,
            participant_id=participant_id,
            target_prompt_id=target_prompt_id,
            order=order,
            output_dir=output_dir,
            injection_band=injection_band,
            prompt_pool=prompt_pool,
            translate_backend=translate_backend,
        )
    demo.launch(share=share, server_port=server_port, server_name="0.0.0.0")


if __name__ == "__main__":
    launch_demo_interface()

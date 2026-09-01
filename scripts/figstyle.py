"""発表スライド用の作図スタイル（配色・フォント・ヘルパ）。

配色の役割は発表全体で固定する。図をまたいで同じ意味に同じ色を割り当てることで、
「意味軸とテクスチャ軸を分けて探索する」という主張がそのまま絵になる。

    BLUE   … 提案・採用した設定・意味軸 (c)・到達点
    ORANGE … 対照・比較対象・音響テクスチャ軸 (x_T)・出発点
    AQUA   … 3系列目のみ（凡例か直接ラベルを必ず併記すること）
    INK 系 … 中立・補助・軸まわり

配色はカテゴリカル3スロットとして検証済み（白背景・全ペア）。
    slot1 #2a78d6  L=0.575 C=0.163 contrast 4.42
    slot2 #eb6834  L=0.671 C=0.175 contrast 3.20
    slot3 #1baf7a  L=0.669 C=0.141 contrast 2.82 → 3:1 未満のため直接ラベル必須
    最悪ペア: 通常視 24.0 / CVD(protan,deutan) 9.2  … いずれも基準を満たす

注: ハンドアウト（冊子体）に載せる図1・図2は白黒印刷を想定して別管理。
本モジュールはスライド専用の図に使う。
"""
from __future__ import annotations

import matplotlib
import matplotlib.font_manager as fm

# --- カテゴリカル3スロット -------------------------------------------------
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
SERIES = (BLUE, ORANGE, AQUA)

# --- インク・クローム -------------------------------------------------------
INK = "#0b0b0b"          # 主テキスト
INK_2 = "#52514e"        # 副テキスト
MUTED = "#898781"        # 軸ラベル・注記
GRID = "#e1e0d9"         # グリッド線
AXIS = "#c3c2b7"         # 軸線・ベースライン
SURFACE = "#ffffff"      # スライドの地色
PANEL = "#f4f4f1"        # パネル背景（薄いグレー）

_CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def use(size: float = 12) -> None:
    """日本語フォントと共通の rcParams を設定する。"""
    fm.fontManager.addfont(_CJK)
    jp = fm.FontProperties(fname=_CJK).get_name()
    matplotlib.rcParams.update({
        "font.size": size,
        "font.family": jp,
        "axes.unicode_minus": False,
        "mathtext.fontset": "cm",
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    })


def tint(hex_color: str, amount: float) -> str:
    """色を地色（白）へ amount だけ寄せた淡色を返す。amount=0 で原色、1 で白。

    図中のボックス塗りに使う。縁は原色のまま残すので、識別はパレットの
    ドキュメント値が担う。
    """
    h = hex_color.lstrip("#")
    rgb = [int(h[i:i + 2], 16) for i in (0, 2, 4)]
    mixed = [round(c + (255 - c) * amount) for c in rgb]
    return "#{:02x}{:02x}{:02x}".format(*mixed)

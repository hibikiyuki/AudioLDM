#!/usr/bin/env python3
"""S3: 同一プロンプトから初期ノイズだけを変えた候補の Mel スペクトログラム。

「テキストでは書き分けられない側面がある」ことを1枚で示すための図。
第1段階（音響テクスチャ選択）の候補群、すなわち c を共有し x_T のみ異なる
音声を横に並べる。

使い方:
  python scripts/make_fig_xt_variation.py [wav ...] [--prompt "..."] [-o out.png]

  wav を省略した場合は output/iec_gradio 以下の seed_selection_gen000_ind*.wav
  （＝第1段階の候補群）のうち最新セッションのものを先頭4件使う。

出力: local/fig_xt_variation.png (300 dpi)
"""
import argparse
import glob
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np

_cjk = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(_cjk)
_jp = fm.FontProperties(fname=_cjk).get_name()
plt.rcParams.update({
    "font.family": _jp, "axes.unicode_minus": False, "mathtext.fontset": "cm",
    "font.size": 12,
})


def default_wavs(n=4):
    """第1段階の候補群を含む最新セッションから n 件拾う。"""
    sessions = sorted(
        {os.path.dirname(p)
         for p in glob.glob("output/iec_gradio/*/seed_selection_gen000_ind*.wav")})
    if not sessions:
        raise SystemExit(
            "第1段階の候補音声が見つかりません。wav を引数で指定してください。")
    latest = sessions[-1]
    wavs = sorted(glob.glob(os.path.join(latest, "seed_selection_gen000_ind*.wav")))
    print(f"using session: {latest}")
    return wavs[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wavs", nargs="*", help="同一 c・異なる x_T の wav（4件推奨）")
    ap.add_argument("-o", "--out", default="local/fig_xt_variation.png")
    ap.add_argument("--prompt", default=None,
                    help="図中に表示するプロンプト文字列（省略可）")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--n_mels", type=int, default=128)
    ap.add_argument("--gray", action="store_true",
                    help="白黒印刷向けグレースケール出力")
    args = ap.parse_args()

    wavs = args.wavs or default_wavs(4)
    n = len(wavs)
    cmap = "gray_r" if args.gray else "magma"

    fig, axes = plt.subplots(1, n, figsize=(2.5 * n + 0.8, 3.1))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.74, bottom=0.16,
                        wspace=0.16)
    axes = np.atleast_1d(axes)

    img = None
    for i, (ax, path) in enumerate(zip(axes, wavs)):
        y, sr = librosa.load(path, sr=args.sr)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=args.n_mels)
        S_db = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_db, sr=sr, x_axis="time",
                                       y_axis="mel", ax=ax, cmap=cmap,
                                       vmin=-80, vmax=0)
        ax.set_title(f"$x_T^{{({i + 1})}}$", fontsize=13, pad=6)
        ax.set_xlabel("時間 [s]", fontsize=10)
        ax.set_ylabel("周波数 [Hz]" if i == 0 else "", fontsize=10)
        if i > 0:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=8.5)

    head = "プロンプトは同一。初期ノイズ $x_T$ だけが異なる"
    if args.prompt:
        head += f"（プロンプト: “{args.prompt}”）"
    fig.text(0.5, 0.95, head, ha="center", va="center", fontsize=13,
             fontweight="bold")
    fig.text(0.5, 0.87,
             "音色・空間性・録音の質感が候補ごとに変わる。これはテキストで書き分けられない",
             ha="center", va="center", fontsize=10.5, color="0.20")

    fig.savefig(args.out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

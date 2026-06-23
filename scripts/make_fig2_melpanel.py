#!/usr/bin/env python3
"""図2: 生成結果の Mel スペクトログラム 2x2 パネルを生成する。

4つの WAV (1.初期個体 / 2.IEC後 / 3.ガチャ戻り / 4.再IEC後) から
Mel スペクトログラム(カラー, magma)を 2x2 で並べ、共有カラーバーを付す。
注: 冊子体は白黒印刷のため、必要に応じて --gray でグレースケール出力に切替可。

使い方:
  python scripts/make_fig2_melpanel.py w1.wav w2.wav w3.wav w4.wav [-o out.png]
  引数省略時は tmp/individual_0..3.wav をプレースホルダとして使用する。
本番では実際の ①②③④ の音声に差し替えること。
"""
import argparse
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
    "font.size": 13,
})

LABELS = [
    "①初期個体",
    "②IEC後",
    "③テクスチャ選択に回帰",
    "④最終（再IEC）",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wavs", nargs="*", help="4 wav files (1..4)")
    ap.add_argument("-o", "--out", default="local/fig2_melpanel.png")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--n_mels", type=int, default=128)
    ap.add_argument("--gray", action="store_true",
                    help="白黒印刷向けグレースケール出力")
    args = ap.parse_args()

    wavs = args.wavs or [f"tmp/individual_{i}.wav" for i in range(4)]
    assert len(wavs) == 4, "exactly 4 wav files are required"

    cmap = "gray_r" if args.gray else "magma"
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.8),
                              gridspec_kw={"hspace": 0.55, "wspace": 0.22})

    img = None
    for ax, wav, label in zip(axes.ravel(), wavs, LABELS):
        y, sr = librosa.load(wav, sr=args.sr)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=args.n_mels)
        S_db = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(
            S_db, sr=sr, x_axis="time", y_axis="mel",
            cmap=cmap, vmin=-80, vmax=0, ax=ax)
        ax.set_title(label, fontsize=21, pad=6)
        ax.set_xlabel("時間 [s]", fontsize=8)
        ax.set_ylabel("周波数 (mel)", fontsize=8)
        ax.tick_params(labelsize=9)

    fig.tight_layout(rect=(0, 0, 0.92, 1), pad=0.6)
    cax = fig.add_axes([0.94, 0.12, 0.018, 0.76])
    cb = fig.colorbar(img, cax=cax, format="%+d")
    cb.set_label("音圧 [dB]", fontsize=15)
    cb.ax.tick_params(labelsize=9)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()

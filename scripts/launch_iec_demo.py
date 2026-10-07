#!/usr/bin/env python3
"""
CLAP-IEC デモ専用 Gradio Web Interface Launcher

二段階交互探索（x_Tガチャ ↔ CLAP-IEC）のデモシナリオに特化した
専用インターフェースの起動スクリプト。conditioning モードのみを扱う。
"""

import argparse
import os
import sys

from audioldm.iec_demo_gradio import launch_demo_interface


def main():
    parser = argparse.ArgumentParser(
        description="CLAP-IEC デモ: x_Tガチャ ↔ 意味空間IEC の二段階交互探索",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # デフォルト設定で起動（audioldm-m-full, 6個体, 5秒, ポート8080）
  NUMBA_CACHE_DIR=/tmp/numba_cache HF_HOME=/tmp/huggingface_cache \\
    python scripts/launch_iec_demo.py

  # カスタム設定
  NUMBA_CACHE_DIR=/tmp/numba_cache HF_HOME=/tmp/huggingface_cache \\
    python scripts/launch_iec_demo.py --model_name audioldm-m-full --duration 5.0 --port 8080

  # Winからは http://192.168.100.16:8080/ にアクセス
        """
    )
    parser.add_argument("--model_name", type=str, default="audioldm-m-full",
                        help="AudioLDMモデル名 (デフォルト: audioldm-m-full)")
    parser.add_argument("--population_size", type=int, default=6,
                        help="1世代あたりの個体数 (デフォルト: 6)")
    parser.add_argument("--backend", type=str, default="audioldm",
                        choices=["audioldm", "musicldm"],
                        help="生成バックエンド (デフォルト: audioldm)")
    parser.add_argument("--duration", type=float, default=None,
                        help="生成する音声の長さ(秒)。未指定なら各バックエンドの\n"
                             "訓練長（audioldm: 10.0 / musicldm: 10.24）。\n"
                             "訓練長から外れると品質が落ちるので原則そのままにする")
    parser.add_argument("--show_variation", action="store_true",
                        help="バリエーションガチャ（区間の振り直し）のUIを表示する。\n"
                             "既定は非表示（評価対象外・手打ち条件に同等機能がないため）")
    parser.add_argument("--port", type=int, default=8080,
                        help="サーバーポート番号 (デフォルト: 8080)")
    parser.add_argument("--share", action="store_true",
                        help="Gradioの公開リンクを生成する")
    parser.add_argument("--mode", type=str, default="two_axis",
                        choices=["two_axis", "single_axis", "text_baseline"],
                        help="two_axis: 2軸交互探索(提案/方法1) / "
                             "single_axis: 単軸CLAP-IEC(温存・非使用) / "
                             "text_baseline: テキスト手打ちベースライン(対照/方法2) "
                             "(デフォルト: two_axis)")
    parser.add_argument("--condition", type=str, default=None,
                        help="ユーザスタディの条件ラベル (例: A / B)")
    parser.add_argument("--participant_id", type=str, default=None,
                        help="被験者ID。指定時は output/<participant>/<condition>/ に保存")
    parser.add_argument("--target_prompt_id", type=str, default=None,
                        help="お題(ターゲットプロンプト)の識別子")
    parser.add_argument("--order", type=str, default=None,
                        help="提示順 (例: AB / BA)")
    parser.add_argument("--translate_backend", type=str, default="auto",
                        choices=["auto", "google", "marian", "none"],
                        help="手打ち条件の日→英翻訳 (既定: auto)。\n"
                             "google: Cloud Translation API v2。要 GOOGLE_TRANSLATE_API_KEY。\n"
                             "        逆翻訳も表示できるのでユーザスタディ推奨\n"
                             "marian: ローカル opus-mt-ja-en。品質不十分（誤訳の実例あり）\n"
                             "none:   素通し")
    parser.add_argument("--semantic_pool", type=str, default=None,
                        help="意味方向プールのJSON (scripts/build_semantic_pool.py の出力)。\n"
                             "未指定なら人手構成の99語プールを使う")
    parser.add_argument("--injection_band", type=str, default=None,
                        help="変異・注入でプールから引く方向を、c* との cos 類似度が\n"
                             "この帯に入るものに限る。**推奨: 0.50,0.80（両バックエンド共通）**\n"
                             "MusicCaps プール1,407方向での実測（c*100個×3シード）:\n"
                             "  audioldm 中央値 0.32 / 90%%tile 0.59 → 該当 17-20%%\n"
                             "  musicldm 中央値 0.27 / 90%%tile 0.58 → 該当 14-16%%\n"
                             "中央値は違うが上側の裾はほぼ同じなので帯は共通で使える。\n"
                             "未指定ならプール全体から引く")
    parser.add_argument("--output_dir", type=str, default="./output/iec_gradio",
                        help="セッション保存先のルート (ユーザスタディ時は ./output/iec_user_study 推奨)")
    args = parser.parse_args()

    injection_band = None
    if args.injection_band:
        lo, hi = (float(v) for v in args.injection_band.split(","))
        if not (lo < hi):
            parser.error("--injection_band は lo,hi で lo < hi にしてください")
        injection_band = (lo, hi)

    semantic_pool = None
    if args.semantic_pool:
        from audioldm.prompt_pool import load_pool_json
        semantic_pool = load_pool_json(args.semantic_pool)
        print(f"意味方向プール: {args.semantic_pool} ({len(semantic_pool)} 方向)")
        # 埋め込みの読み込みはパイプライン側で行う。バックエンド構築後でないと
        # 実際に動いているモデル名と照合できないため（launcher は実モデルを知らない）。

    try:
        launch_demo_interface(
            model_name=args.model_name,
            population_size=args.population_size,
            duration=args.duration,
            backend=args.backend,
            show_variation=args.show_variation,
            share=args.share,
            server_port=args.port,
            mode=args.mode,
            condition=args.condition,
            participant_id=args.participant_id,
            target_prompt_id=args.target_prompt_id,
            order=args.order,
            output_dir=args.output_dir,
            injection_band=injection_band,
            prompt_pool=semantic_pool,
            pool_embeddings_path=args.semantic_pool,
            translate_backend=args.translate_backend,
        )
    except KeyboardInterrupt:
        print("\n\nサーバーを停止しました。")
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

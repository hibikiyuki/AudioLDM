#!/usr/bin/env python3
"""
CLAP-IEC デモ専用 Gradio Web Interface Launcher

二段階交互探索（x_Tガチャ ↔ CLAP-IEC）のデモシナリオに特化した
専用インターフェースの起動スクリプト。conditioning モードのみを扱う。
"""

import argparse
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
    parser.add_argument("--duration", type=float, default=5.0,
                        help="生成する音声の長さ(秒) (デフォルト: 5.0)")
    parser.add_argument("--port", type=int, default=8080,
                        help="サーバーポート番号 (デフォルト: 8080)")
    parser.add_argument("--share", action="store_true",
                        help="Gradioの公開リンクを生成する")
    args = parser.parse_args()

    print("=" * 70)
    print("CLAP-IEC デモ: x_Tガチャ ↔ 意味空間IEC")
    print("=" * 70)
    print(f"モデル: {args.model_name}")
    print(f"個体数: {args.population_size}")
    print(f"音声長: {args.duration}秒")
    print(f"ポート: {args.port}")
    print(f"公開リンク: {'有効' if args.share else '無効'}")
    print("=" * 70)
    print()

    try:
        launch_demo_interface(
            model_name=args.model_name,
            population_size=args.population_size,
            duration=args.duration,
            share=args.share,
            server_port=args.port,
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

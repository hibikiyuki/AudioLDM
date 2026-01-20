#!/usr/bin/env python3
"""
AudioLDM-IEC Gradio Web Interface Launcher

対話型進化的効果音生成システムのWebインターフェース起動スクリプト
"""

import argparse
import os
import sys

# AudioLDMモジュールをインポート
from audioldm.iec_gradio import launch_interface


def main():
    parser = argparse.ArgumentParser(
        description="AudioLDM-IEC: 対話型進化的効果音生成システム",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # デフォルト設定で起動
  python scripts/launch_iec_gradio.py

  # カスタム設定で起動
  python scripts/launch_iec_gradio.py --model_name audioldm-m-full --population_size 6 --duration 2.5 --port 8080

  # 公開リンクを生成
  python scripts/launch_iec_gradio.py --share
        """
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="audioldm-s-full-v2",
        help="AudioLDMモデル名 (デフォルト: audioldm-s-full-v2)"
    )
    
    parser.add_argument(
        "--population_size",
        type=int,
        default=6,
        help="1世代あたりの個体数 (デフォルト: 6)"
    )
    
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="生成する音声の長さ(秒) (デフォルト: 5.0)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="サーバーポート番号 (デフォルト: 7860)"
    )
    
    parser.add_argument(
        "--share",
        action="store_true",
        help="Gradioの公開リンクを生成する"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("AudioLDM-IEC: 対話型進化的効果音生成システム")
    print("=" * 70)
    print(f"モデル: {args.model_name}")
    print(f"個体数: {args.population_size}")
    print(f"音声長: {args.duration}秒")
    print(f"ポート: {args.port}")
    print(f"公開リンク: {'有効' if args.share else '無効'}")
    print("=" * 70)
    print()
    
    try:
        launch_interface(
            model_name=args.model_name,
            population_size=args.population_size,
            duration=args.duration,
            share=args.share,
            server_port=args.port
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

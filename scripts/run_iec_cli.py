#!/usr/bin/env python3
"""
AudioLDM-IEC CLI版

コマンドラインインターフェースで対話型進化計算を実行
"""

import argparse
import sys

from audioldm.iec_pipeline import run_iec_session


def main():
    parser = argparse.ArgumentParser(
        description="AudioLDM-IEC CLI: コマンドライン版対話型進化的効果音生成",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # プロンプトを指定して実行
  python scripts/run_iec_cli.py --prompt "爆発音" --population_size 4 --duration 3.0

  # ランダム初期化で実行
  python scripts/run_iec_cli.py --population_size 6 --max_generations 10
        """
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="初期プロンプト (指定しない場合はランダム生成)"
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
        "--max_generations",
        type=int,
        default=10,
        help="最大世代数 (デフォルト: 10)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/iec_cli_session",
        help="出力ディレクトリ (デフォルト: ./output/iec_cli_session)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("AudioLDM-IEC CLI: 対話型進化的効果音生成システム")
    print("=" * 70)
    print(f"プロンプト: {args.prompt if args.prompt else 'ランダム'}")
    print(f"モデル: {args.model_name}")
    print(f"個体数: {args.population_size}")
    print(f"音声長: {args.duration}秒")
    print(f"最大世代数: {args.max_generations}")
    print(f"出力先: {args.output_dir}")
    print("=" * 70)
    print()
    
    try:
        run_iec_session(
            prompt=args.prompt,
            model_name=args.model_name,
            population_size=args.population_size,
            duration=args.duration,
            output_dir=args.output_dir,
            max_generations=args.max_generations
        )
    except KeyboardInterrupt:
        print("\n\nセッションを中断しました。")
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

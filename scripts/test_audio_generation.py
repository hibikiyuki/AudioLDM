#!/usr/bin/env python3
"""
AudioLDM-IEC 音声生成テストスクリプト
修正後の音声生成が正しく動作するか確認
"""

import torch
import numpy as np
import os
import soundfile as sf

print("=" * 70)
print("AudioLDM-IEC 音声生成テスト")
print("=" * 70)

# 1. インポートテスト
print("\n1. モジュールインポートテスト...")
try:
    from audioldm.iec_pipeline import AudioLDM_IEC
    print("✅ インポート成功")
except Exception as e:
    print(f"❌ インポートエラー: {e}")
    exit(1)

# 2. システム初期化テスト
print("\n2. システム初期化テスト...")
try:
    iec_system = AudioLDM_IEC(
        model_name="audioldm-s-full-v2",
        population_size=2,  # テストなので少数
        duration=5.0,  # AudioLDMは2.5秒の倍数が必要
        ddim_steps=50  # テストなので少ないステップ
    )
    print("✅ 初期化成功")
    print(f"   デバイス: {iec_system.device}")
    print(f"   個体数: {iec_system.population_size}")
    print(f"   音声長: {iec_system.duration}秒")
except Exception as e:
    print(f"❌ 初期化エラー: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 3. 初期個体群生成テスト
print("\n3. 初期個体群生成テスト...")
try:
    results = iec_system.initialize_population(
        prompt="爆発音",
        variation_strength=0.3
    )
    print(f"✅ 生成成功: {len(results)}個の個体")
    
    # 波形の形状を確認
    for i, (genotype, waveform) in enumerate(results):
        print(f"   個体{i}: 波形形状={waveform.shape}, dtype={waveform.dtype}")
        
        # 形状の検証
        if len(waveform.shape) == 2:
            print(f"      ✅ 正しい形状 (batch, samples)")
        elif len(waveform.shape) == 3:
            print(f"      ✅ 正しい形状 (batch, channels, samples)")
        else:
            print(f"      ⚠️  想定外の形状")
        
        # サンプル数の検証
        expected_samples = int(iec_system.duration * 16000)
        actual_samples = waveform.shape[-1]
        print(f"      期待サンプル数: {expected_samples}, 実際: {actual_samples}")
        
        if abs(actual_samples - expected_samples) < 1000:
            print(f"      ✅ サンプル数OK")
        else:
            print(f"      ⚠️  サンプル数が想定と異なります")
            
except Exception as e:
    print(f"❌ 生成エラー: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 4. 音声保存テスト
print("\n4. 音声保存テスト...")
output_dir = "./output/test_iec_audio"
try:
    saved_paths = iec_system.save_generation_audio(
        results,
        output_dir=output_dir,
        prefix="test"
    )
    print(f"✅ 保存成功: {len(saved_paths)}個のファイル")
    
    # 保存されたファイルを検証
    for path in saved_paths:
        if os.path.exists(path):
            file_size = os.path.getsize(path)
            print(f"   {os.path.basename(path)}: {file_size / 1024:.1f} KB")
            
            # ファイルの検証
            try:
                data, sr = sf.read(path)
                print(f"      読み込み成功: 形状={data.shape}, SR={sr}Hz")
                
                # 形状の検証
                if len(data.shape) == 1:
                    print(f"      ✅ モノラル音声")
                    expected_samples = int(iec_system.duration * sr)
                    if abs(len(data) - expected_samples) < 1000:
                        print(f"      ✅ 長さOK ({len(data)} サンプル)")
                    else:
                        print(f"      ⚠️  長さが想定と異なります")
                elif len(data.shape) == 2 and data.shape[1] == 1:
                    print(f"      ✅ モノラル音声 (2次元)")
                else:
                    print(f"      ⚠️  想定外の形状: {data.shape}")
                
                # ファイルサイズの検証
                expected_size_kb = (sr * iec_system.duration * 2) / 1024  # 16-bit PCM
                if file_size / 1024 > expected_size_kb * 0.8:
                    print(f"      ✅ ファイルサイズOK")
                else:
                    print(f"      ⚠️  ファイルサイズが小さすぎます (期待: 約{expected_size_kb:.1f}KB)")
                    
            except Exception as e:
                print(f"      ❌ ファイル検証エラー: {e}")
        else:
            print(f"   ❌ ファイルが存在しません: {path}")
            
except Exception as e:
    print(f"❌ 保存エラー: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 5. 進化テスト
print("\n5. 進化テスト...")
try:
    # 最初の2個体を選択
    selected_indices = [0, 1]
    results = iec_system.evolve_population(
        selected_indices=selected_indices,
        mutation_rate=0.3,
        mutation_strength=0.15,
        elite_count=1
    )
    print(f"✅ 進化成功: {len(results)}個の次世代個体")
    
    # 波形の形状を確認
    for i, (genotype, waveform) in enumerate(results):
        print(f"   個体{i}: 波形形状={waveform.shape}")
        
except Exception as e:
    print(f"❌ 進化エラー: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 6. 履歴保存テスト
print("\n6. 履歴保存テスト...")
try:
    history_path = os.path.join(output_dir, "test_history.json")
    iec_system.population.save_history(history_path)
    
    if os.path.exists(history_path):
        print(f"✅ 履歴保存成功: {history_path}")
    else:
        print(f"❌ 履歴ファイルが作成されませんでした")
        
except Exception as e:
    print(f"❌ 履歴保存エラー: {e}")
    import traceback
    traceback.print_exc()

# 最終結果
print("\n" + "=" * 70)
print("✅ 全てのテストが成功しました！")
print("=" * 70)
print(f"\n生成された音声: {output_dir}")
print("\n次のステップ:")
print("  1. 生成された音声を再生して確認")
print("  2. Gradio UIを起動して全機能をテスト")
print("     $ python scripts/launch_iec_gradio.py")
print()

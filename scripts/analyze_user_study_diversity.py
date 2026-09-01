#!/usr/bin/env python3
"""
ユーザスタディ 客観的多様性の自動算出
=====================================

提案システム(2軸分離)のユーザスタディで保存されたセッション
(音声WAV + iec_history.json + session_meta.json)から、探索の客観的多様性を算出する。
主観評価（音色が豊か／退屈）の裏付けとして用いる。

算出する指標:
  - 音色多様性 (timbral diversity): 各音声の MFCC（時間方向の平均・標準偏差）を特徴量とし、
    集団内の平均ペアワイズ・ユークリッド距離。モデル不要で常に算出できる主指標。
  - 意味多様性 (semantic diversity, --clap 指定時のみ): 各音声を CLAP オーディオ
    エンコーダで埋め込み、集団内の平均ペアワイズ余弦距離。
    （scripts/exp_common.pairwise_cosine_distance を再利用）

集計単位:
  - 世代ごと (generation): 各世代に提示された個体群の多様性。
  - all  : セッションで探索した全音声の多様性。
  - final: 最終世代の多様性。

読み方: 第2段階では x_T を固定するため、世代を経ても音色多様性は大きくは伸びず、
意味多様性が探索とともに動くことが期待される。session_meta.json に condition が
入っている場合は、その値ごとの平均も併せて表示する（探索設定を変えて記録した
セッション群を並べて見るための補助であり、比較条件の指定ではない）。

使用例:
  # 1セッション
  python scripts/analyze_user_study_diversity.py output/iec_user_study/P01/session_xxxx

  # 被験者ルート以下を再帰的に集計し、CSV/JSON出力
  python scripts/analyze_user_study_diversity.py output/iec_user_study \\
      --recursive --out-csv out/diversity.csv --out-json out/diversity.json

  # CLAP意味多様性も算出（モデルをロードする）
  python scripts/analyze_user_study_diversity.py <session_dir> --clap
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.exp_common import pairwise_cosine_distance  # noqa: E402

# 音声ファイル名 "<prefix>_gen<NNN>_ind<MM>.wav" から世代・個体番号を抽出する
_WAV_RE = re.compile(r"_gen(\d+)_ind(\d+)\.wav$")
SR = 16000
N_MFCC = 20


@dataclass
class GenerationDiversity:
    generation: int
    n_individuals: int
    timbral_diversity: float
    semantic_diversity: float = -1.0  # -1 は未算出（--clap 未指定）


@dataclass
class SessionDiversity:
    session_dir: str
    condition: Optional[str] = None
    participant_id: Optional[str] = None
    target_prompt_id: Optional[str] = None
    order: Optional[str] = None
    n_generations: int = 0
    # all: 探索した全音声 / final: 最終世代
    timbral_diversity_all: float = 0.0
    timbral_diversity_final: float = 0.0
    semantic_diversity_all: float = -1.0
    semantic_diversity_final: float = -1.0
    per_generation: List[GenerationDiversity] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 特徴量・距離
# ---------------------------------------------------------------------------

def mfcc_feature(wav_path: str) -> np.ndarray:
    """音声から MFCC ベースの音色特徴ベクトルを抽出する。

    n_mfcc 次元の時間平均と時間標準偏差を連結した (2*n_mfcc,) のベクトル。
    スペクトル包絡の中心とその時間変動の両方を捉える。
    """
    import librosa  # 遅延import（解析時のみ依存）

    y, _ = librosa.load(wav_path, sr=SR)
    if y.size == 0:
        return np.zeros(2 * N_MFCC, dtype=np.float32)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)  # (n_mfcc, frames)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)]).astype(np.float32)


def mean_pairwise_euclidean(feats: List[np.ndarray]) -> float:
    """特徴ベクトル集合の平均ペアワイズ・ユークリッド距離。"""
    if len(feats) < 2:
        return 0.0
    arr = np.stack(feats)
    dists = []
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            dists.append(float(np.linalg.norm(arr[i] - arr[j])))
    return float(np.mean(dists)) if dists else 0.0


# ---------------------------------------------------------------------------
# セッション収集
# ---------------------------------------------------------------------------

def find_session_dirs(root: Path, recursive: bool) -> List[Path]:
    """音声WAVを含むセッションディレクトリを列挙する。"""
    def has_wavs(d: Path) -> bool:
        return any(_WAV_RE.search(p.name) for p in d.glob("*.wav"))

    if not recursive:
        return [root] if has_wavs(root) else []
    dirs = set()
    for p in root.rglob("*.wav"):
        if _WAV_RE.search(p.name):
            dirs.add(p.parent)
    return sorted(dirs)


def group_wavs_by_generation(session_dir: Path) -> Dict[int, List[Path]]:
    groups: Dict[int, List[Path]] = {}
    for p in sorted(session_dir.glob("*.wav")):
        m = _WAV_RE.search(p.name)
        if not m:
            continue
        gen = int(m.group(1))
        groups.setdefault(gen, []).append(p)
    return groups


def load_session_meta(session_dir: Path) -> dict:
    meta_path = session_dir / "session_meta.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def analyze_session(session_dir: Path, use_clap: bool, clap_model=None) -> Optional[SessionDiversity]:
    groups = group_wavs_by_generation(session_dir)
    if not groups:
        return None
    meta = load_session_meta(session_dir)
    result = SessionDiversity(
        session_dir=str(session_dir),
        condition=meta.get("condition"),
        participant_id=meta.get("participant_id"),
        target_prompt_id=meta.get("target_prompt_id"),
        order=meta.get("order"),
        n_generations=len(groups),
    )

    # 全音声の特徴量をキャッシュ
    all_timbre: List[np.ndarray] = []
    all_clap = []  # torch.Tensor のリスト
    final_gen = max(groups.keys())
    final_timbre: List[np.ndarray] = []
    final_clap = []

    for gen in sorted(groups.keys()):
        wavs = groups[gen]
        timbre_feats = [mfcc_feature(str(p)) for p in wavs]
        gen_div = GenerationDiversity(
            generation=gen,
            n_individuals=len(wavs),
            timbral_diversity=mean_pairwise_euclidean(timbre_feats),
        )
        all_timbre.extend(timbre_feats)
        if gen == final_gen:
            final_timbre.extend(timbre_feats)

        if use_clap and clap_model is not None:
            clap_feats = [clap_embed_audio(clap_model, str(p)) for p in wavs]
            gen_div.semantic_diversity = pairwise_cosine_distance(clap_feats)
            all_clap.extend(clap_feats)
            if gen == final_gen:
                final_clap.extend(clap_feats)

        result.per_generation.append(gen_div)

    result.timbral_diversity_all = mean_pairwise_euclidean(all_timbre)
    result.timbral_diversity_final = mean_pairwise_euclidean(final_timbre)
    if use_clap and all_clap:
        result.semantic_diversity_all = pairwise_cosine_distance(all_clap)
        result.semantic_diversity_final = pairwise_cosine_distance(final_clap)
    return result


# ---------------------------------------------------------------------------
# CLAP（オプション）
# ---------------------------------------------------------------------------

def load_clap_model(model_name: str = "audioldm-m-full"):
    """CLAP オーディオ埋め込みのため AudioLDM をロードする。"""
    from audioldm.iec_pipeline import AudioLDM_IEC

    print(f"[clap] {model_name} をロード中（意味多様性算出のため）...")
    return AudioLDM_IEC(model_name=model_name, population_size=1, duration=2.5)


def clap_embed_audio(clap_model, wav_path: str):
    """音声を CLAP オーディオ埋め込み (torch.Tensor) に変換する。"""
    import librosa
    import torch

    y, _ = librosa.load(wav_path, sr=SR)
    # AudioLDM_IEC が音声→CLAP埋め込みヘルパを提供している場合はそれを使う。
    for attr in ("encode_audio_clap", "_encode_audio_clap", "clap_audio_embedding"):
        fn = getattr(clap_model, attr, None)
        if callable(fn):
            emb = fn(y)
            return emb.detach().cpu() if hasattr(emb, "detach") else torch.tensor(emb)
    raise RuntimeError(
        "CLAP オーディオ埋め込み用のメソッドが AudioLDM_IEC に見つかりません。"
        " --clap を外して音色多様性のみで実行してください。"
    )


# ---------------------------------------------------------------------------
# 出力
# ---------------------------------------------------------------------------

def write_csv(results: List[SessionDiversity], path: str) -> None:
    rows = []
    for r in results:
        for g in r.per_generation:
            rows.append({
                "participant_id": r.participant_id,
                "condition": r.condition,
                "target_prompt_id": r.target_prompt_id,
                "order": r.order,
                "session_dir": r.session_dir,
                **asdict(g),
            })
    if not rows:
        print("[analyze] 出力対象がありません")
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[analyze] CSV 出力: {path}")


def write_json(results: List[SessionDiversity], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(r) for r in results]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[analyze] JSON 出力: {path}")


def print_summary(results: List[SessionDiversity]) -> None:
    print("\n=== セッション別サマリ ===")
    for r in results:
        sem = (f"{r.semantic_diversity_all:.4f}" if r.semantic_diversity_all >= 0 else "n/a")
        print(
            f"[{r.condition or '?'}] {r.participant_id or r.session_dir} | "
            f"世代数={r.n_generations} | "
            f"音色(all)={r.timbral_diversity_all:.4f} 音色(final)={r.timbral_diversity_final:.4f} | "
            f"意味(all)={sem}"
        )

    # session_meta.json に condition がある場合のみ、その値ごとの平均も出す
    by_cond: Dict[str, List[SessionDiversity]] = {}
    for r in results:
        by_cond.setdefault(r.condition or "NA", []).append(r)
    if len(by_cond) > 1:
        print("\n=== 条件別平均（音色多様性 all） ===")
        for cond, rs in sorted(by_cond.items()):
            vals = [r.timbral_diversity_all for r in rs]
            print(f"  条件 {cond}: mean={np.mean(vals):.4f}  n={len(vals)}")


def main():
    parser = argparse.ArgumentParser(description="ユーザスタディの客観的多様性を算出")
    parser.add_argument("path", type=str, help="セッションディレクトリ、または被験者ルート")
    parser.add_argument("--recursive", action="store_true",
                        help="path 以下を再帰的に探索し、全セッションを集計する")
    parser.add_argument("--clap", action="store_true",
                        help="CLAPオーディオ埋め込みによる意味多様性も算出（モデルをロード）")
    parser.add_argument("--model_name", type=str, default="audioldm-m-full",
                        help="--clap 時に使うAudioLDMモデル名")
    parser.add_argument("--out-csv", type=str, default=None, help="CSV出力先")
    parser.add_argument("--out-json", type=str, default=None, help="JSON出力先")
    args = parser.parse_args()

    root = Path(args.path)
    if not root.exists():
        print(f"パスが存在しません: {root}")
        sys.exit(1)

    session_dirs = find_session_dirs(root, args.recursive)
    if not session_dirs:
        print(f"音声WAVを含むセッションが見つかりません: {root}"
              " （--recursive を試してください）")
        sys.exit(1)
    print(f"対象セッション数: {len(session_dirs)}")

    clap_model = load_clap_model(args.model_name) if args.clap else None

    results: List[SessionDiversity] = []
    for d in session_dirs:
        r = analyze_session(d, args.clap, clap_model)
        if r is not None:
            results.append(r)

    print_summary(results)
    if args.out_csv:
        write_csv(results, args.out_csv)
    if args.out_json:
        write_json(results, args.out_json)


if __name__ == "__main__":
    main()

"""Exp 5 パイロット事前検証: interaction_log のタイミング計測フィールドを検証する

`IECInterface` にこの session で追加した `selection_seconds` / `compute_seconds`
（および `random_baseline` アクションのログ）が、実モデルを起動せずに想定通り
記録されることを検証するユニットテスト。

モデルのロードが重いため、`AudioLDM_IEC` をフェイクに差し替えて
`IECInterface` のロギングロジックのみを検証する（シミュレーションモード）。

使用方法:
    python scripts/test_exp5_interaction_log.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from audioldm.iec_gradio import IECInterface

POP_SIZE = 6
COMPUTE_DELAY = 0.05   # 生成計算をシミュレートする遅延（秒）
VIEW_DELAY = 0.08      # ユーザーが個体群を聴取・比較している時間をシミュレート（秒）
TIME_TOLERANCE = 0.05  # 計測値の許容誤差（秒）

NOOP_PROGRESS = lambda *args, **kwargs: None


class FakeGenotype:
    def __init__(self, metadata: Optional[Dict] = None):
        self.metadata = metadata or {}


class FakePopulation:
    def __init__(self):
        self.generation_number = 0
        self.history: List = []
        self.best_individuals: List = []

    def rollback_generation(self, steps: int):
        self.generation_number = max(0, self.generation_number - steps)
        return [FakeGenotype({"prompt": "fake prompt"}) for _ in range(POP_SIZE)]


class FakeIECSystem:
    """AudioLDM_IEC のロード不要なスタブ。ログの計測対象になる時間だけ再現する。"""

    def __init__(self, population_size: int = POP_SIZE):
        self.population = FakePopulation()
        self.population_size = population_size
        self.ga_mode = "conditioning"

    def initialize_population_conditioning(self, prompt, slerp_alpha, x_T_seed=None, x_T_mode="elite_keep", **kwargs):
        time.sleep(COMPUTE_DELAY)
        return [
            (FakeGenotype({"prompt": prompt, "initialization": "slerp_b2"}), np.zeros(10))
            for _ in range(self.population_size)
        ]

    def evolve_population_conditioning(self, selected_indices, mutation_mu_range,
                                        p_mut, elite_count, random_sample_count,
                                        x_T_mode="elite_keep", **kwargs):
        time.sleep(COMPUTE_DELAY)
        self.population.generation_number += 1
        results = [
            (FakeGenotype({"operation": "crossover", "elite": i < elite_count}), np.zeros(10))
            for i in range(self.population_size)
        ]
        conv_info = {
            "centroid_converged": False,
            "diversity_low": False,
            "centroid_dist": 0.05,
            "diversity": 0.2,
        }
        return results, conv_info

    def generate_random_baseline_population(self, prompt=None, x_T_seed=None):
        time.sleep(COMPUTE_DELAY)
        if x_T_seed is None:
            x_T_seed = int(np.random.randint(0, 2**32 - 1))
        return [
            (FakeGenotype({"initialization": "random_baseline", "x_T_seed": x_T_seed}), np.zeros(10))
            for _ in range(self.population_size)
        ]

    def save_generation_audio(self, results, output_dir, prefix="gen"):
        return [os.path.join(output_dir, f"{prefix}_{i}.wav") for i in range(len(results))]

    def get_generation_info(self) -> Dict:
        return {
            "generation_number": self.population.generation_number,
            "population_size": self.population_size,
            "history_length": len(self.population.history),
            "best_count": len(self.population.best_individuals),
        }

    def _generate_audio_from_any_genotype(self, genotype, text=""):
        return np.zeros(10)


def make_interface(tmp_dir: str) -> IECInterface:
    """重いモデルロードを回避し、IECInterface インスタンスを直接組み立てる。"""
    interface = object.__new__(IECInterface)
    interface.output_dir = tmp_dir
    interface.iec_system = FakeIECSystem()
    interface.current_results = []
    interface.current_audio_paths = []
    interface.session_id = "pilot_dryrun"
    interface.session_dir = tmp_dir
    interface.st_results = []
    interface.st_audio_paths = []
    interface.st_ranked_words = []
    interface.baseline_results = []
    interface.baseline_audio_paths = []
    interface._baseline_round = 0
    interface.interaction_log = []
    interface._presented_at = None
    return interface


def check(label: str, condition: bool, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f" ({detail})" if detail else ""))
    return condition


def run() -> bool:
    print("=" * 60)
    print("Exp 5 パイロット事前検証: interaction_log タイミング計測")
    print("=" * 60)

    all_ok = True

    with tempfile.TemporaryDirectory() as tmp_dir:
        interface = make_interface(tmp_dir)

        # --- 1. initialize_generation ---------------------------------
        print("\n[1] initialize_generation")
        audio_list, info, message = interface.initialize_generation(
            prompt="tense orchestral strings",
            variation_strength=0.3,
            ga_mode="conditioning",
            progress=NOOP_PROGRESS,
        )
        log0 = interface.interaction_log[-1]
        all_ok &= check("action == 'initialize'", log0.get("action") == "initialize")
        all_ok &= check("compute_seconds が記録されている", "compute_seconds" in log0,
                        f"値={log0.get('compute_seconds')}")
        all_ok &= check(
            "compute_seconds がシミュレートした計算時間と整合",
            log0["compute_seconds"] >= COMPUTE_DELAY - TIME_TOLERANCE,
            f"compute_seconds={log0['compute_seconds']:.4f}s, 期待 >= {COMPUTE_DELAY:.2f}s",
        )
        all_ok &= check("_presented_at が設定されている", interface._presented_at is not None)

        # ユーザーが個体群を聴取・比較している時間をシミュレート
        time.sleep(VIEW_DELAY)

        # --- 2. evolve_generation（選好時間の計測） ---------------------
        print("\n[2] evolve_generation (1回目: 選好時間計測)")
        presented_at_before = interface._presented_at
        audio_list, info, message, conv_text = interface.evolve_generation(
            selected_checkboxes=[0, 1],
            mutation_rate=0.3,
            mutation_strength=0.3,
            elite_count=2,
            progress=NOOP_PROGRESS,
        )
        log1 = interface.interaction_log[-1]
        all_ok &= check("action == 'evolve'", log1.get("action") == "evolve")
        all_ok &= check("selection_seconds が記録されている", log1.get("selection_seconds") is not None,
                        f"値={log1.get('selection_seconds')}")
        all_ok &= check(
            "selection_seconds が「提示〜選択」の経過時間を反映",
            log1["selection_seconds"] >= VIEW_DELAY - TIME_TOLERANCE,
            f"selection_seconds={log1['selection_seconds']:.4f}s, 期待 >= {VIEW_DELAY:.2f}s",
        )
        all_ok &= check("compute_seconds が記録されている", "compute_seconds" in log1,
                        f"値={log1.get('compute_seconds')}")
        all_ok &= check(
            "_presented_at が次の提示時刻に更新されている",
            interface._presented_at is not None and interface._presented_at != presented_at_before,
        )

        # --- 3. evolve_generation（2世代目、selection_seconds が新しい起点を使う） ---
        time.sleep(VIEW_DELAY * 2)
        print("\n[3] evolve_generation (2回目: 起点更新の確認)")
        audio_list, info, message, conv_text = interface.evolve_generation(
            selected_checkboxes=[0],
            mutation_rate=0.3,
            mutation_strength=0.3,
            elite_count=2,
            progress=NOOP_PROGRESS,
        )
        log2 = interface.interaction_log[-1]
        all_ok &= check(
            "selection_seconds が前回より長い待機を反映している",
            log2["selection_seconds"] > log1["selection_seconds"],
            f"1回目={log1['selection_seconds']:.4f}s, 2回目={log2['selection_seconds']:.4f}s",
        )

        # --- 4. rollback_generation（_presented_at のリセット） ----------
        print("\n[4] rollback_generation")
        presented_at_before_rollback = interface._presented_at
        audio_list, info, message = interface.rollback_generation(steps=1)
        log3 = interface.interaction_log[-1]
        all_ok &= check("action == 'rollback'", log3.get("action") == "rollback")
        all_ok &= check(
            "rollback 後に _presented_at がリセットされている（選好時間計測の起点更新）",
            interface._presented_at is not None and interface._presented_at != presented_at_before_rollback,
        )

        # rollback 後、再選択までの時間が新しい _presented_at から計測されることを確認
        time.sleep(VIEW_DELAY)
        audio_list, info, message, conv_text = interface.evolve_generation(
            selected_checkboxes=[0],
            mutation_rate=0.3,
            mutation_strength=0.3,
            elite_count=2,
            progress=NOOP_PROGRESS,
        )
        log4 = interface.interaction_log[-1]
        all_ok &= check(
            "rollback 後の selection_seconds が rollback 時刻からの経過を反映",
            VIEW_DELAY - TIME_TOLERANCE <= log4["selection_seconds"] <= VIEW_DELAY + VIEW_DELAY,
            f"selection_seconds={log4['selection_seconds']:.4f}s",
        )

        # --- 5. generate_random_baseline（ベースライン条件のログ） -------
        print("\n[5] generate_random_baseline")
        audio_list, info, message, seed_str = interface.generate_random_baseline(
            prompt="ambient soundscape",
            progress=NOOP_PROGRESS,
        )
        log5 = interface.interaction_log[-1]
        all_ok &= check("action == 'random_baseline'", log5.get("action") == "random_baseline")
        all_ok &= check("round が記録されている", log5.get("round") == 1, f"値={log5.get('round')}")
        all_ok &= check("population_size が記録されている", log5.get("population_size") == POP_SIZE)
        all_ok &= check("compute_seconds が記録されている", "compute_seconds" in log5,
                        f"値={log5.get('compute_seconds')}")
        all_ok &= check(
            "random_baseline は selection_seconds を持たない（進化条件と独立）",
            "selection_seconds" not in log5,
        )
        all_ok &= check("x_T_seed が記録されている", log5.get("x_T_seed") is not None,
                        f"値={log5.get('x_T_seed')}")
        all_ok &= check(
            "x_T seed入力欄への返り値が記録seedと一致",
            seed_str == str(log5.get("x_T_seed")),
            f"seed_str={seed_str}, log5.x_T_seed={log5.get('x_T_seed')}",
        )

        # --- 6. save_session で JSON シリアライズ可能か ------------------
        print("\n[6] interaction_log の JSON シリアライズ確認")
        import json
        try:
            json.dumps(interface.interaction_log, ensure_ascii=False)
            all_ok &= check("interaction_log が JSON シリアライズ可能", True)
        except TypeError as e:
            all_ok &= check("interaction_log が JSON シリアライズ可能", False, str(e))

    print("\n" + "=" * 60)
    print(f"総合判定: {'PASS' if all_ok else 'FAIL'}")
    print("=" * 60)
    return all_ok


if __name__ == "__main__":
    success = run()
    sys.exit(0 if success else 1)

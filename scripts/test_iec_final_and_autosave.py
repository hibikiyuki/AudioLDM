"""提案条件のログ自動保存・個体の出自・最終音声の確定の検証（モデル不要）。

2026-10-15 に、提案条件のログが一度もディスクに書かれていなかったことが判明した。
- 操作のたびに interaction_log.json / iec_history.json が書かれること
- 進化ログに、提示した個体群と選んだ個体の出自が残ること
- 最終候補が世代をまたいで保持され、確定時に出自とともに記録されること
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from audioldm.iec import genotype_provenance  # noqa: E402
from test_exp5_interaction_log import make_interface  # noqa: E402
from test_initialization_keep_base import StubIEC  # noqa: E402


class StubSystem(StubIEC):
    """実際の初期化・進化ロジックを動かし、音声の保存だけを偽装する。"""

    def __init__(self) -> None:
        super().__init__()
        self.ga_mode = "conditioning"
        self.presentation_source = {}

    def save_generation_audio(self, results, output_dir, prefix="gen"):
        paths = []
        for i in range(len(results)):
            full = os.path.join(output_dir, f"{prefix}_ind{i:02d}.wav")
            present = full.replace(".wav", "_present.wav")
            self.presentation_source[present] = full
            paths.append(present)
        return paths


def main() -> int:
    ok = True

    def check(label: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        print(f"  {'✅' if cond else '❌'} {label}{(' / ' + detail) if detail else ''}")
        ok = ok and cond

    tmp = tempfile.mkdtemp()
    itf = make_interface(tmp)
    itf.iec_system = StubSystem()
    sysm = itf.iec_system
    log_path = os.path.join(tmp, "interaction_log.json")
    hist_path = os.path.join(tmp, "iec_history.json")

    c_star = sysm._encode_text_single("start point")
    itf.current_results = sysm.initialize_population_conditioning(
        prompt="start point", slerp_alpha=0.4, x_T_seed=1,
        base_embedding=c_star, x_T_mode="shared")
    itf.current_audio_paths = sysm.save_generation_audio(itf.current_results, tmp, "gen000")

    print("[1] 進化1回目: 自動保存と出自")
    check("保存前はログファイルが無い", not os.path.exists(log_path))
    itf.evolve_generation([1, 3], 0.0, 0.0, 1, p_mut=0.7, mutation_mu_min=0.15,
                          mutation_mu_max=0.35, random_sample_count=2, x_T_mode="shared")
    check("interaction_log.json が書かれた", os.path.exists(log_path))
    disk = json.load(open(log_path, encoding="utf-8"))
    ev = [e for e in disk if e.get("action") == "evolve"]
    check("ディスク上に evolve が1件", len(ev) == 1)
    e = ev[0]
    check("presented_origins（第2段階の初期個体群）",
          e.get("presented_origins") == ["start_point"] + ["init_inject"] * 5,
          str(e.get("presented_origins")))
    check("selected_origins が選択に対応",
          e.get("selected_origins") == ["init_inject", "init_inject"], str(e.get("selected_origins")))
    check("cond_crossover=extrapolate が記録", e.get("cond_crossover") == "extrapolate")
    check("diversity が記録", isinstance(e.get("diversity"), float))
    hist = json.load(open(hist_path, encoding="utf-8"))
    check("iec_history.json が JSON として読める（新しいメタデータ込み）",
          len(hist["history"]) == 2, f"{len(hist['history'])}世代")

    print("\n[2] 進化後の個体の出自")
    origins = [genotype_provenance(g) for g, _ in itf.current_results]
    kinds = sorted(o["origin"] for o in origins)
    check("エリート1・進化3・注入2",
          kinds.count("evolved") == 3 and kinds.count("inject") == 2
          and kinds.count("init_inject") == 1, str(kinds))
    check("新しい個体は born_generation=1",
          all(o["born_generation"] == 1 for o in origins if o["origin"] in ("evolved", "inject")))
    elite = [o for o in origins if o["elite"]]
    check("エリートは元の出自と生まれた世代を保つ",
          len(elite) == 1 and elite[0]["origin"] == "init_inject"
          and elite[0]["born_generation"] == 0, str(elite))

    print("\n[3] 最終候補は世代をまたいで保持され、確定時に出自とともに記録される")
    ev_idx = next(i for i, o in enumerate(origins) if o["origin"] == "evolved")
    inj_idx = next(i for i, o in enumerate(origins) if o["origin"] == "inject")
    pick_inj = itf.make_iec_final_pick(inj_idx)
    check("注入個体の出自は inject", pick_inj["provenance"]["origin"] == "inject")
    pick = itf.make_iec_final_pick(ev_idx)
    check("進化個体の出自は evolved", pick["provenance"]["origin"] == "evolved")
    check("提示用と全長のパスを両方持つ",
          pick["path"].endswith("_present.wav") and pick["full_path"].endswith(".wav")
          and not pick["full_path"].endswith("_present.wav"),
          f"{os.path.basename(pick['path'])} / {os.path.basename(pick['full_path'])}")
    # 最終候補を控えたまま、もう1世代進める
    itf.evolve_generation([0], 0.0, 0.0, 1, p_mut=0.7, mutation_mu_min=0.15,
                          mutation_mu_max=0.35, random_sample_count=2, x_T_mode="shared")
    check("次の世代に進んだ", sysm.population.generation_number == 2)
    check("範囲外は None", itf.make_iec_final_pick(99) is None)
    itf.finalize_iec(pick)
    disk = json.load(open(log_path, encoding="utf-8"))
    fin = [e for e in disk if e.get("action") == "iec_final"]
    check("ディスク上に iec_final が1件", len(fin) == 1)
    f = fin[0]
    check("確定した世代は控えた時点の第1世代", f.get("generation") == 1, str(f.get("generation")))
    check("出自 evolved・born_generation=1",
          f["provenance"]["origin"] == "evolved" and f["provenance"]["born_generation"] == 1,
          str(f.get("provenance")))
    check("全長パスが記録されている", f.get("final_full_audio_path") == pick["full_path"])
    ev2 = [e for e in disk if e.get("action") == "evolve"][-1]
    check("2回目の evolve は選択1体でも外挿", ev2.get("cond_crossover") == "extrapolate")
    check("2回目の presented_origins に evolved と inject が並ぶ",
          {"evolved", "inject"} <= set(ev2.get("presented_origins", [])),
          str(ev2.get("presented_origins")))

    print("\n[4] 確定前に未選択なら何も記録しない")
    n = len(itf.interaction_log)
    itf.finalize_iec(None)
    check("ログが増えない", len(itf.interaction_log) == n)

    print("\n" + "=" * 56)
    print("✅ 全て成功" if ok else "❌ 失敗あり")
    print("=" * 56)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

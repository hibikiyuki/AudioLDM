"""プール埋め込みのバックエンド読み分けと照合ガードの検証。

CLAP はバックエンドごとに別物なので、別モデルで計算した埋め込みを使うと
注入帯の判定が**静かに壊れる**（例外にならず、無関係な方向が「帯に入った」として
選ばれる）。これを防ぐ仕組みを検証する。

検証すること:
  1. `pool_embeddings_path` がバックエンドごとに別のパスを返す
  2. モデル名が一致すれば読める
  3. モデル名が**不一致**なら例外（別バックエンドの埋め込みを誤用しない）
  4. モデル名が**未記録**なら例外（出所不明のファイルを使わない＝fail-closed）
  5. `allow_unknown=True` のときだけ未記録を許す

モデルのロードは不要。

使用方法:
    python scripts/test_pool_embeddings_guard.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from audioldm.prompt_pool import load_pool_embeddings, pool_embeddings_path


def check(cond: bool, label: str) -> bool:
    print(f"  {'✅' if cond else '❌'} {label}")
    return cond


def write_pt(path: Path, model_name, n: int = 5):
    data = {"terms": [f"t{i}" for i in range(n)],
            "embeddings": torch.randn(n, 512)}
    if model_name is not None:
        data["model_name"] = model_name
    torch.save(data, path)


def main() -> int:
    ok = True

    print("\n[1] バックエンドごとにパスが分かれる")
    base = "scripts/outputs/semantic_pool/semantic_pool.json"
    pa = pool_embeddings_path(base, "audioldm")
    pm = pool_embeddings_path(base, "musicldm")
    print(f"  audioldm → {pa}")
    print(f"  musicldm → {pm}")
    ok &= check(pa != pm, "2つのバックエンドで別のパスになる")
    ok &= check(pa.endswith("_embeddings_audioldm.pt"), "audioldm のパス命名")
    ok &= check(pm.endswith("_embeddings_musicldm.pt"), "musicldm のパス命名")
    ok &= check(pool_embeddings_path("x/foo.pt", "musicldm") == "x/foo.pt",
                ".pt を直接指定したらそのまま返す")

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)

        print("\n[2] モデル名が一致すれば読める")
        f_ok = d / "ok.pt"
        write_pt(f_ok, "ucsd-reach/musicldm")
        got = load_pool_embeddings(str(f_ok), expect_model="ucsd-reach/musicldm")
        ok &= check(len(got) == 5, f"5 件読めた")
        ok &= check(tuple(next(iter(got.values())).shape) == (1, 512), "形状が (1, 512)")

        print("\n[3] モデル名が不一致なら例外")
        try:
            load_pool_embeddings(str(f_ok), expect_model="audioldm-m-full")
            ok &= check(False, "例外が出るべきだった")
        except ValueError as e:
            ok &= check("一致しません" in str(e), f"拒否された: {str(e).splitlines()[0][:50]}")

        print("\n[4] モデル名が未記録なら例外（fail-closed）")
        f_unknown = d / "unknown.pt"
        write_pt(f_unknown, None)
        try:
            load_pool_embeddings(str(f_unknown), expect_model="ucsd-reach/musicldm")
            ok &= check(False, "例外が出るべきだった")
        except ValueError as e:
            ok &= check("記録されていません" in str(e),
                        f"拒否された: {str(e).splitlines()[0][:50]}")

        print("\n[5] allow_unknown=True のときだけ未記録を許す")
        got = load_pool_embeddings(str(f_unknown), expect_model="ucsd-reach/musicldm",
                                   allow_unknown=True)
        ok &= check(len(got) == 5, "明示的に許可すれば読める")

        print("\n[6] expect_model を渡さなくても未記録は拒否される")
        try:
            load_pool_embeddings(str(f_unknown))
            ok &= check(False, "例外が出るべきだった")
        except ValueError:
            ok &= check(True, "照合相手が無くても出所不明なら拒否")

    print("\n" + "=" * 56)
    print("✅ 全て成功" if ok else "❌ 失敗あり")
    print("=" * 56)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

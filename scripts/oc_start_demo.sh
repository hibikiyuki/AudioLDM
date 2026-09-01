#!/usr/bin/env bash
# =====================================================================
# オープンキャンパス2026 デモ起動スクリプト
#
#   bash scripts/oc_start_demo.sh
#
# ・環境変数のセットと起動コマンドをまとめただけのもの。
# ・落ちても自動で再起動する（Ctrl+C を2回押すと完全停止）。
# ・接続先の表示に使うアドレスは HOST_IP で上書きできる。
#     HOST_IP=192.168.0.10 PORT=8080 bash scripts/oc_start_demo.sh
#   未指定なら実行中ホストのIPを自動検出する。
# =====================================================================
set -u

cd "$(dirname "$0")/.." || exit 1

export NUMBA_CACHE_DIR=/tmp/numba_cache
export HF_HOME=/tmp/huggingface_cache

PORT="${PORT:-8080}"
# 表示用アドレス。未指定なら既定経路のIPを拾う（取れなければホスト名を出す）
HOST_IP="${HOST_IP:-$(ip route get 1.1.1.1 2>/dev/null | awk '{print $7; exit}')}"
HOST_IP="${HOST_IP:-$(hostname)}"
DURATION="${DURATION:-2.5}"
POP="${POP:-6}"

echo "======================================================"
echo " オープンキャンパス デモサーバ"
echo " ノートPCからは http://${HOST_IP}:${PORT}/ で接続"
echo " 停止するには Ctrl+C を2回押す"
echo "======================================================"

while true; do
    python scripts/launch_iec_demo.py \
        --model_name audioldm-m-full \
        --population_size "${POP}" \
        --duration "${DURATION}" \
        --port "${PORT}" \
        --mode two_axis

    code=$?
    if [ $code -eq 130 ]; then
        echo "終了しました。"
        break
    fi
    echo ""
    echo "!! サーバが落ちました (exit=${code})。5秒後に自動で再起動します。"
    echo "!! 完全に止めたい場合は今すぐ Ctrl+C を押してください。"
    sleep 5
done

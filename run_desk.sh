#!/usr/bin/env bash
# 重跑交易台：回归测试 → 基准阶梯（五个区）→ 渲染 docs/index.html
#
#   bash run_desk.sh
#
# 可选环境变量：
#   WQ_DATA_DIR  数据库位置，默认 ~/aus-weather-quant/data
#   TRAIN_DAYS   滚动训练窗（天），默认 180；电价历史不足时可临时调小
set -euo pipefail
cd "$(dirname "$0")"

export WQ_DATA_DIR="${WQ_DATA_DIR:-${HOME}/aus-weather-quant/data}"
TRAIN_DAYS="${TRAIN_DAYS:-180}"

if [ ! -d "${WQ_DATA_DIR}/truth" ]; then
  echo "找不到数据库：${WQ_DATA_DIR}/truth"
  echo "用 WQ_DATA_DIR=/你的/data bash run_desk.sh 指定位置"
  exit 1
fi

echo "数据库：${WQ_DATA_DIR}"
echo "训练窗：${TRAIN_DAYS} 天"
echo

echo "== 1/3 回归测试 =="
python -m pytest l1/test_asof.py -q

echo
echo "== 2/3 基准阶梯 =="
python l1/ladder.py --region all --hours all --train-days "${TRAIN_DAYS}" --out docs/ladder.json

echo
echo "== 3/3 交易台 =="
python l1/desk.py --in docs/ladder.json --out docs/index.html

echo
echo "完成：docs/index.html（本地用浏览器打开即可）"
echo "发布：bash publish.sh"

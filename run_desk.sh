#!/usr/bin/env bash
# 重跑交易台：回归测试 → 基准阶梯（五个区）→ 稳健性检验 → 渲染两页
#   docs/index.html  交易台
#   docs/method.html 方法与证据
#
#   bash run_desk.sh
#
# 可选环境变量：
#   WQ_DATA_DIR    数据库位置，默认 ~/aus-weather-quant/data
#   TRAIN_DAYS     主训练窗（天），默认 180
#   ALT_WINDOWS    稳健性检验用的其他训练窗，默认 "90 120"；设成空字符串就跳过（快很多）
set -euo pipefail
cd "$(dirname "$0")"

export WQ_DATA_DIR="${WQ_DATA_DIR:-${HOME}/aus-weather-quant/data}"
TRAIN_DAYS="${TRAIN_DAYS:-180}"
ALT_WINDOWS="${ALT_WINDOWS-90 120}"

if [ ! -d "${WQ_DATA_DIR}/truth" ]; then
  echo "找不到数据库：${WQ_DATA_DIR}/truth"
  echo "用 WQ_DATA_DIR=/你的/data bash run_desk.sh 指定位置"
  exit 1
fi

echo "数据库：${WQ_DATA_DIR}"
echo "训练窗：${TRAIN_DAYS} 天；稳健性对照：${ALT_WINDOWS:-无}"
echo

echo "== 1/5 回归测试 =="
python -m pytest l1/test_asof.py -q

echo
echo "== 2/5 基准阶梯 =="
python l1/ladder.py --region all --hours all --train-days "${TRAIN_DAYS}" --out docs/ladder.json

echo
echo "== 3/5 稳健性检验 =="
mkdir -p docs/windows
ALT_ARGS=""
for W in ${ALT_WINDOWS}; do
  echo "-- 训练窗 ${W} 天（几分钟）"
  python l1/ladder.py --region all --hours all --train-days "${W}" --out "docs/windows/ladder_${W}.json" > "docs/windows/ladder_${W}.log" 2>&1
  ALT_ARGS="${ALT_ARGS} ${W}=docs/windows/ladder_${W}.json"
done
python l1/robust.py --in docs/ladder.json --alt ${ALT_ARGS} --out docs/robust.json

echo
echo "== 4/5 期货公允价值 =="
python l1/futures.py --out docs/futures.json

echo
echo "== 5/5 交易台 =="
python l1/desk.py --in docs/ladder.json --out docs/index.html

echo
echo "完成：docs/index.html 交易台、docs/method.html 方法与证据（本地用浏览器打开即可）"
echo "发布：bash publish.sh"
echo "期货信号（只在本机）：python l1/signals.py --template 填价格，再 python l1/signals.py"

#!/bin/bash
# 安装前推纸面交易的定时任务（只需要运行一次）：
#
#   bash paper_install.sh
#
# 做三件事：
#   1. 记下当前终端里的 python / gh / git 位置和数据目录，写到 ~/.nemdesk.env
#      （定时任务的环境很干净，找不到 conda 里的 python，所以要提前记下来）
#   2. 往 crontab 加一行：每 10 分钟调用一次 paper_cron.sh
#   3. 跑一次自检，确认 python 能导入、数据目录在
set -euo pipefail
cd "$(dirname "$0")"
REPO="$(pwd)"
PY="$(command -v python || command -v python3)"
GH="$(command -v gh || true)"
GIT="$(command -v git)"
DATA="${WQ_DATA_DIR:-${HOME}/aus-weather-quant/data}"
LOGDIR="$(dirname "${DATA}")"

PATHS="$(dirname "${PY}"):$(dirname "${GIT}")"
if [ -n "${GH}" ]; then
  PATHS="${PATHS}:$(dirname "${GH}")"
fi

cat > "${HOME}/.nemdesk.env" <<ENV
export PATH="${PATHS}:/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export WQ_DATA_DIR="${DATA}"
ENV
echo "已写 ${HOME}/.nemdesk.env"
echo "  python: ${PY}"
echo "  gh:     ${GH:-（没找到，推送会用 git 自己的登录）}"
echo "  数据:   ${DATA}"

LINE="*/10 * * * * /bin/bash \"${REPO}/paper_cron.sh\" >> \"${LOGDIR}/paper_cron.log\" 2>&1"
CUR="$(crontab -l 2>/dev/null || true)"
if printf '%s\n' "${CUR}" | grep -q "paper_cron.sh"; then
  echo "crontab 里已经有这一行了，不重复加"
else
  printf '%s\n%s\n' "${CUR}" "${LINE}" | sed '/^$/d' | crontab -
  echo "已加入 crontab：${LINE}"
fi

echo
echo "自检："
. "${HOME}/.nemdesk.env"
python l1/paper.py --status
echo
echo "完成。日志在 ${LOGDIR}/paper_cron.log"
echo "macOS 需要给 cron 完全磁盘访问权限，否则它读不了“文稿”里的仓库："
echo "  系统设置 → 隐私与安全性 → 完全磁盘访问权限 → 点 + → 按 Cmd+Shift+G 输入 /usr/sbin/cron → 打开"
echo "电脑要在每天 13:05–13:29（NEM，墨尔本冬令时的 13:05，夏令时的 14:05）开机且没有睡眠。"

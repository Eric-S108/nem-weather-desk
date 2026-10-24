#!/bin/bash
# 定时任务入口：每 10 分钟被调用一次；只有在 NEM 13:05–13:29 之间、且明天的信号还没出时才真正干活。
# 由 paper_install.sh 写进 crontab，不需要手动运行。
cd "$(dirname "$0")" || exit 1
if [ -f "${HOME}/.nemdesk.env" ]; then
  . "${HOME}/.nemdesk.env"
fi
export WQ_DATA_DIR="${WQ_DATA_DIR:-${HOME}/aus-weather-quant/data}"
exec python l1/paper.py --cron

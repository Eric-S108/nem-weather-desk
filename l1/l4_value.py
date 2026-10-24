"""
L4 决策价值 —— 已并入 ladder.py 的经济评估，本文件只保留为兼容入口。
====================================================================
为什么退役：

  原来的 l4_value 拿 l3_signal 的**水平**信号（"今天电价偏高还是偏低"）
  去排**日内**充放电计划（"今天哪个小时充、哪个小时放"）。水平信号回答不了
  排序问题，于是计划挑出来的时刻基本是噪声——2026-10 审计实跑：

      naive  89.29 $/天     ours  −26.89 $/天     perfect  133.54 $/天
      捕获率 −262.5%，比什么都不做还亏

  ladder.py 的经济评估现在用同样的电池规格（1 MW / 2 MWh、η=0.85、
  吞吐成本 $5/MWh、先充后放、按 NEM 交易日），但计划直接来自**分小时**
  的价格预测，并且和 naive / 自回归 / +天气 各层放在同一批日子上比。

所以这里不再自己算，直接转给 ladder：

    python l4_value.py --region QLD1
    # 等价于
    python ladder.py --region QLD1 --hours all
"""
from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="（已并入 ladder.py）储能决策价值")
    ap.add_argument("--region", default="QLD1")
    ap.add_argument("--train-days", type=int, default=180)
    ap.add_argument("--step-days", type=int, default=7)
    ap.add_argument("--embargo-days", type=int, default=3)
    ap.add_argument("--out", default=None)
    # 旧参数照收不报错，但已经不起作用
    ap.add_argument("--win-days", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--cost", type=float, default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)

    print("l4_value 已并入 ladder.py 的经济评估（原实现捕获率 −262.5%，已退役）。")
    print("下面直接运行：ladder.py --hours all\n")
    import ladder
    fwd = ["--region", a.region, "--hours", "all",
           "--train-days", str(a.train_days), "--step-days", str(a.step_days),
           "--embargo-days", str(a.embargo_days)]
    if a.out:
        fwd += ["--out", a.out]
    return ladder.main(fwd)


if __name__ == "__main__":
    sys.exit(main())

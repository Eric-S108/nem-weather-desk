# NEM 天气交易台

**天气预报，在价格自回归和 AEMO 官方预调度之上，还能多赚多少？**

页面：<https://eric-s108.github.io/nem-weather-desk/>

这不是"天气和电价相不相关"的研究——那个问题的答案永远是"相关"，没有交易含义。
这里问的是**增量**：已经知道最近的价格、也知道 AEMO 公开的预调度，再加上天气，
一块电池每天能多赚多少，而且这个"多"能不能和运气分开。

---

## 方法

按 Lago & Weron 的 EPF 综述搭一个四层阶梯，每层只比上一层多一组特征，
全部在**同一批样本外小时**上比较：

| 层 | 特征 | 含义 |
|---|---|---|
| B0 naive | P(t−168h) | rMAE 的分母 |
| B1 自回归 | + 决策时刻已结算的最新价、最近两天同一钟点、一周前、24h 均价 | 主导预测因子 |
| B2 +官方 | + AEMO 预调度 | 市场已经定价的部分 |
| B3 +天气 | + GHI / 风速 / 气温预报的滚动分位 | **天气的增量在这一格** |

一个区判为"可用"要**四道全过**：

1. 样本外 ≥ 90 天
2. rMAE 相对上一层有改善
3. Giacomini-White p ≤ 0.05（滚动重拟合下 DM 检验的前提不成立）
4. 一块 1 MW / 2 MWh 电池的日收益增量，95% 区间下界 > 0

### 几条硬约束

- **固定时刻决策：NEM 日 D 的 24 小时，在 D−1 天 13:30（NEM）一次性定计划。**
  这是电池运营商排次日计划的方式，也是 EPF 综述的 day-ahead 设定。AEMO 预调度
  最远只看到次日 04:00、约 12:30 之后才覆盖下一天，所以"每小时滚动提前 24 小时"
  会让 04:00–12:30 的小时拿不到官方预测。可用 `--decide-at` 调整。
- **每个特征在决策时刻都必须已可见。** 一小时的结算价要过 1h10m 才发布；天气和
  预调度都只取决策时刻之前发出的版本里最新的一版。回归测试会在这条被破坏时直接失败；
  合成数据里还埋了一个"决策时刻之后发出、数值等于实际价"的预调度版本当前视陷阱。
- **每个 hour-of-day 各拟合一个模型。** 一个模型跨 24 小时会把日内曲线压平：
  水平贴得准，挑出来的充放电时刻却是噪声。
- **asinh 变换后拟合，预测夹回训练期见过的范围。** 电价重尾、会穿零。
- **经济评估**：每个 NEM 交易日只用预测价排计划（先充后放各 2 小时，η=0.85，
  吞吐成本 $5/MWh，预测不赚钱就不动），按实际价结算。完美预知用同一优化器喂实际价，
  是同一约束下的上限。

---

## 运行

```
pip install -r requirements.txt
bash run_desk.sh
bash publish.sh
```

`run_desk.sh` 跑回归测试 → 五个区的基准阶梯 → 生成两页：
`docs/index.html` 给交易员（每个区用不用天气、用哪个模型排电池、一天赚多少、累计曲线），
`docs/method.html` 放方法与统计证据（rMAE、GW 检验、误差解剖、特征取舍）。

两种用法分开评估：

- **排电池**（1 MW / 2 MWh，先充后放）只吃日内排序；
- **赌 AEMO 预调度的偏差**：每小时预测高于预调度做多 1 MW、低于做空，按实际价结算。
  吃的是整天的价格水平。基线是"一直做空预调度"——前一天的预调度系统性偏高，
  这一条本身就"赚"很多，但那是口径差，不是信息；模型要和它比。

天气变量的取舍用**嵌套选择**：每个滚动窗口只用训练段（前段拟合、最后 28 天验证）
挑一次，测试段完全不可见。第二页那张逐变量对比表是在测试集上算的，只作诊断。

单次运行有时长限制的机器上，可以一个区一个区跑再合并：

```
python l1/ladder.py --region VIC1 --hours all --out VIC1.json
python l1/ladder.py --merge NSW1.json QLD1.json VIC1.json SA1.json TAS1.json --hours all --out docs/ladder.json
python l1/desk.py --in docs/ladder.json --out docs/index.html
```

`run_desk.sh` 默认还会用 90 天、120 天训练窗各跑一遍，给稳健性检验用（多花几分钟）；
不需要时用 `ALT_WINDOWS= bash run_desk.sh` 跳过。
默认读 `~/aus-weather-quant/data`，可用 `WQ_DATA_DIR` 改。

### 补数据（要联网，在本机跑）

```
export WQ_DATA_DIR="$HOME/aus-weather-quant/data"
cd l1
python aemo.py --archive --start 2024-03 --end 2026-09
python aemo.py --daily-archive --start 2026-10-01 --end 2026-10-23
python aemo.py --inspect-predispatch
python aemo.py --predispatch-archive --start 2025-10-25 --end 2026-10-23 --lead-lo 12 --lead-hi 40
python ingest.py --backfill-forecast --method previous-runs --model ecmwf_ifs025 --start 2026-08-20 --end 2026-10-23
python ingest.py --backfill-truth --start 2026-08-20 --end 2026-10-17
```

第一行很重要：采集脚本和 `run_desk.sh` 必须指向同一个数据库。
预调度归档每天一个大文件、只滚动保留约 13 个月；中途断了直接重跑同一条命令，
已完成的日子会跳过。


## 前推纸面交易

回测之外唯一不能作弊的检验：每天 13:05–13:29（NEM）自动补数据、给**次日** 24 小时出信号，
写进 `docs/paper/signals.csv` 并提交到 GitHub——提交时间就是信号先于结果存在的证据。
同一次运行里把已有实际价的日子结算掉（`docs/paper/summary.json`），交易台第一页显示累计结果。

安装一次即可（之后每 10 分钟被调用一次，窗口外直接退出）：

```
bash paper_install.sh
```

手动操作：

```
python l1/paper.py --status
python l1/paper.py --signal --refresh --push
```

第一版信号就是正式版，同一天再出一次不会覆盖；决策时刻之后才生成的信号标为 late，不计成绩。

---

## 目录

| 文件 | 作用 |
|---|---|
| `l1/ladder.py` | 核心：基准阶梯、GW 检验、储能经济评估、误差解剖、特征取舍 |
| `l1/desk.py` | 把 `ladder.json` 渲染成两页；第二页“方法与证据”在这里 |
| `l1/trader.py` | 第一页“交易台”：结论、行动表、森林图、累计收益 |
| `l1/paper.py` | 前推纸面交易：每日出信号、结算、写 summary |
| `l1/robust.py` | 稳健性检验：按月拆分、Holm 多重比较、按天下注、换训练窗（共同样本段） |
| `l1/aemo.py` | NEMWeb：电价、需求、预调度 |
| `l1/ingest.py` `l1/openmeteo.py` | Open-Meteo：天气预报与再分析 |
| `l1/store.py` `l1/features.py` `l1/realdata.py` | 时点存储与 as-of 读取 |
| `l1/test_asof.py` | 回归测试 |
| `l1/price_link.py` `l1/l3_signal.py` | 诊断层（相关性、分层、消融），不进交易台 |
| `l1/l4_value.py` | 已退役，转给 `ladder.py --hours all` |
| `l1/l6_report.py` `l1/build_data_real.py` | 旧版页面，保留备查 |

## 出处

- Lago, Marcjasz, De Schutter, Weron. *Forecasting day-ahead electricity prices:
  A review of state-of-the-art algorithms, best practices and an open-access benchmark.*
  [arXiv 2204.11735](https://arxiv.org/pdf/2204.11735)
- *The Role of Weather Predictions in Electricity Price Forecasting Beyond the
  Day-Ahead Horizon.* [IEEE 9788043](https://ieeexplore.ieee.org/document/9788043/)
- *Optimising Battery Energy Storage System Trading via Energy Market Operator
  Price Forecast.* [arXiv 2510.03657](https://arxiv.org/html/2510.03657)

数据：[AEMO NEMWeb](https://nemweb.com.au)、[Open-Meteo](https://open-meteo.com)。

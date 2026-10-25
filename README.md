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




## 给金融交易员：期货公允价值、信号与风险

ASX 的 NEM 期货按合约期内现货均价结算，所以公允价值 = 合约期内现货均价的期望。
`l1/futures.py` 只用 AEMO 公开现货价：已过去的小时按实际价，剩下的从最近 90 天整块抽样，
区间按历史校准放宽（样本外 90% 区间覆盖约 88–100%）。产品：基荷、峰荷、早峰、晚峰、$300 cap（按小时价算，是下限）。

**ASX Energy 网站条款禁止爬取、下载、转载其价格**，所以：

- 公开页面只放我们的公允价值和区间，不放任何 ASX 价格；
- 市场价由你自己在本机录入，信号、仓位和风险只生成在本机、不提交：

```
python l1/signals.py --template
python l1/signals.py
```

第一条生成 `private/futures_prices.csv`（填 bid / ask）、`private/risk.json`（单日亏损上限等）、
`private/positions.csv`（现有持仓）。第二条生成 `private/desk.html`。`private/` 已在 `.gitignore` 里。

信号：胜率 ≥ 70% 且扣费后优势 ≥ $1/MWh 才买入 / 卖出；数量 = min(每合约上限, 剩余单日亏损额度 ÷ 每 MW 的 95% 单日亏损)。
$300 cap 只出买入信号（公允价值是下限）。这个规则**没有在历史期货价上回测过**（没有公开的历史期货价）。

## 能下单的产品与交易成本

ASX 的 NEM 电力合约只有月度、季度、年度（基荷 / 峰荷 / $300 cap / 早晚峰），**没有日度或周度合约**，
日前信号在交易所上表达不出来。`l1/products.py` 把预测变成按现货价结算的实物决策、扣掉交易成本：

| 产品 | 规格 | 吃什么 |
|---|---|---|
| 储能套利 | 1 MW / 2 MWh，每天先充后放一个循环 | 日内排序 |
| 可调负荷 | 每天用够 8 MWh、最大 1 MW，挑便宜的小时 | 日内排序 |
| 日前定机组的调峰 | 1 MW 燃气机组，前一天定运行小时，开机至少 3 小时 | 价格水平 |

成本：AEMO 市场费用 $0.55/MWh（双向）；储能效率 85%、衰减 $15/MWh（敏感性 0–50）；
调峰机组边际成本 $130/MWh（2026 Q1 东海岸气价 $10.61/GJ × 热耗 11.5 + 运维）、启动 $60/MW/次。
每个产品都和"不建模型、直接按 AEMO 预调度排"比；最好的预测扣完成本都不显著赚钱时，推荐"不做"。

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


### 每日任务跑在 GitHub 上（不依赖 Mac 开机）

`.github/workflows/paper-daily.yml` 每天 UTC 03:05 / 03:13 / 03:21（= NEM 13:05 / 13:13 / 13:21）触发：
补数据 → 出信号 → 结算 → 更新期货公允价值和页面 → 提交推送。数据用仓库里的精简数据仓 `data_live/`
（`l1/livestore.py --export` 从完整数据仓导出，约 17 MB；和完整仓出的信号逐位一致，已核对）。
每次运行写 `docs/paper/health.json`，页面顶部显示数据体检结果；数据出问题、错过窗口、信号没出全时任务以失败结束，
GitHub 会发邮件。在 GitHub 网页的 Actions 页也可以手动点 Run workflow（只补数据、结算、更新页面）。

第一版信号就是正式版，同一天再出一次不会覆盖；决策时刻之后才生成的信号标为 late，不计成绩。

---

## 目录

| 文件 | 作用 |
|---|---|
| `l1/ladder.py` | 核心：基准阶梯、GW 检验、储能经济评估、误差解剖、特征取舍 |
| `l1/desk.py` | 把 `ladder.json` 渲染成两页；第二页“方法与证据”在这里 |
| `l1/trader.py` | 第一页“交易台”：结论、行动表、森林图、累计收益 |
| `l1/futures.py` | 期货公允价值：结算价分布、历史校准、单日风险、天气指标、明日现货展望 |
| `l1/signals.py` | 本机私有：按你录入的报价出信号、仓位和风险（不提交） |
| `l1/products.py` | 能下单的产品：储能、可调负荷、调峰机组，含交易成本和敏感性 |
| `l1/livestore.py` | 精简数据仓：导出 / 合并每日任务用的 data_live/ |
| `l1/prob.py` | 概率预测：分位数、尖峰概率及其检验 |
| `l1/futures_bt.py` | 期货信号回测（本机私有，用你提供的历史期货价） |
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


## 概率预测、混合模型、供需基本面、期货回测、组合风险（2026-10-25）

- `l1/prob.py`：分位数（q10/q50/q90）和尖峰概率（> $300）= 点预测 + 过去 90 天同一小时的样本外误差。
  第二页"概率预测与新模型"给 pinball、80% 覆盖、Brier，对照 AEMO、"预调度 > 300"和历史频率。
- 混合模型 `B2 混合`：每天把我们模型的 24 个预测值按 AEMO 预调度的日内顺序重新分配到各小时
  （水平和振幅是我们的，哪个小时贵是 AEMO 的）。
- 供需基本面：预调度解析器多出需求、可用发电、联络线、风光可用出力。历史要重跑一遍预调度归档
  （完成记录换成 v2，已有的电价行按主键去重），然后用 `l1/livestore.py --sync data_live` 带给云上任务。
- `l1/futures_bt.py`（本机私有）：把你拿到的历史期货结算价放进 `private/futures_history.csv`，
  回测风险溢价、公允价值的信息量和信号规则。
- `l1/signals.py` 加组合风险：同一批历史日期算各腿盈亏再加总（带相关性），VaR95 / ES95 / 历史最差一天；
  新信号按"期望盈亏 ÷ 单日风险"排序分配额度，组合 ES95 不超过单日上限。

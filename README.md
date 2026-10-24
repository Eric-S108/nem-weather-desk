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
| B1 自回归 | + t−26/27/48/168h 的价格、24h 均价 | 主导预测因子 |
| B2 +官方 | + AEMO 预调度 | 市场已经定价的部分 |
| B3 +天气 | + GHI / 风速 / 气温预报的滚动分位 | **天气的增量在这一格** |

一个区判为"可用"要**四道全过**：

1. 样本外 ≥ 90 天
2. rMAE 相对上一层有改善
3. Giacomini-White p ≤ 0.05（滚动重拟合下 DM 检验的前提不成立）
4. 一块 1 MW / 2 MWh 电池的日收益增量，95% 区间下界 > 0

### 几条硬约束

- **决策时刻 t−24h，每个特征在那一刻都必须已可见。** 一小时的结算价要过
  1h10m 才发布，所以最新可用价格是 t−26h；天气和预调度都只取 known_at ≤ t−24h
  的版本里最新的一版。回归测试会在这条被破坏时直接失败。
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

`run_desk.sh` 跑回归测试 → 五个区的基准阶梯 → 生成 `docs/index.html`。
默认读 `~/aus-weather-quant/data`，可用 `WQ_DATA_DIR` 改。

### 补数据（要联网，在本机跑）

```
cd l1
python aemo.py --archive --start 2024-03 --end 2026-09
python aemo.py --daily-archive --start 2026-10-01 --end 2026-10-23
python aemo.py --inspect-predispatch
python aemo.py --predispatch-archive --start 2025-10-25 --end 2026-10-23 --lead-lo 24 --lead-hi 30
python ingest.py --backfill-forecast --method previous-runs --model ecmwf_ifs025 --start 2026-08-20 --end 2026-10-23
python ingest.py --backfill-truth --start 2026-08-20 --end 2026-10-17
```

预调度归档每天一个大文件、只滚动保留约 13 个月；中途断了直接重跑同一条命令，
已完成的日子会跳过。

---

## 目录

| 文件 | 作用 |
|---|---|
| `l1/ladder.py` | 核心：基准阶梯、GW 检验、储能经济评估、误差解剖、特征取舍 |
| `l1/desk.py` | 把 `ladder.json` 渲染成交易台 |
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

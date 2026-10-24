# L1 采集与时点存储 · 以及基准阶梯

《NSW1 电价预测系统骨架》第 L1 层的实现：**只追加不覆盖，每条记录带 `interval_time` / `known_at` / `ingested_at`**。
外加 L2 的唯一入口 `get_features(as_of, horizons)`，以及 2026-09 重写的 **L3–L6 基准阶梯**。

这一层的全部意义就一句话：**让"回测时误用了未来信息"从一个隐蔽的 bug 变成一次崩溃。**

---

## 先跑这个：基准阶梯 + 交易台

2026-09 按 Lago & Weron 的 EPF 综述重写了评估口径。核心变化：不再问"天气和电价相不相关"
（那个问题的答案永远是"相关"，没有交易含义），改问**天气在价格自回归和 AEMO 官方预调度之上还剩多少增量**。

```bash
python ladder.py --region all --hours 10-16 --out ladder.json
python desk.py --in ladder.json --out 交易台.html
open 交易台.html
```

四层阶梯，每层只比上一层多一组特征：

| 层 | 特征 | 含义 |
|---|---|---|
| `B0 naive` | P(d−7,h) | rMAE 的分母 |
| `B1 自回归` | + P(t−24/25/48/168h)、前一日均价 | 综述里的主导预测因子 |
| `B2 +官方` | + AEMO 预调度（同时效） | 市场已经定价的部分 |
| `B3 +天气` | + GHI / 风速 / 气温预报的滚动分位 | **天气的增量在这一格** |

判为"可用"要**三道全过**：rMAE 有改善、Giacomini-White p ≤ 0.05、储能套利日收益增量的
95% 区间下界 > 0。任何一道不过就是"不可用"。

### 口径上的几个硬约束

- **rMAE，不用 MAPE。** 电价会穿零，MAPE 在那里没有定义。MAE 和 RMSE 同时报。
- **Giacomini-White，不用 Diebold-Mariano。** 滚动重拟合下校准窗口一直在变，
  DM 的前提不成立。方差做 Newey-West 修正，否则损失差分的自相关会把 p 值压到荒谬的小。
- **asinh 变换后拟合，预测夹回训练期见过的范围。** 电价重尾，原尺度上几个尖峰就能主导整条
  回归线；而 sinh 是指数级的，不夹住的话一个离群输入能吐出上万块的预测——
  fixture 上实测：不夹时 B2 的 RMSE 从 124 炸到 8171。
- **价格滞后按时间戳取，不按下标平移。** 样本有洞时下标平移会把"24 行前"当成"24 小时前"。
- **训练集和测试集之间留 3 天禁运带。**
- **经济评估是必须项，不是加分项。** 综述原话：统计误差
  "may not necessarily reflect the economic value of reducing prediction errors"。

### 没有 AEMO 预调度时

`B2` 会显式跳过，`B3` 的增量改成相对 `B1` 算，页面上打黄色警示条。
这时候的数字是**上界不是 alpha**：官方预调度里已经含了 AEMO 自己的新能源出力预测，
那部分信息市场已经定价。想拿到真正的增量，先把预调度接上：

```bash
python aemo.py --inspect PREDISPATCHPRICE --start 2026-03
```

---

## 快速开始（数据层）

```bash
pip install -r requirements.txt

python ingest.py --selftest      # 不联网，验证存储语义与 as-of 边界
pytest -q test_asof.py           # 50 个回归测试，同样不联网

# 回填：先用便宜的 previous-runs 铺一遍（几分钟）
python ingest.py --backfill-truth    --start 2024-03-14 --end 2026-08-20
python ingest.py --backfill-forecast --method previous-runs \
                 --model ecmwf_ifs025 --start 2024-03-14 --end 2026-08-20

# 训练/回测要用的那段，再用 single-runs 精确重抓
python ingest.py --backfill-forecast --method single-runs \
                 --model ecmwf_ifs025 --start 2026-05-01 --end 2026-08-20 --runs 00,12 --yes

python ingest.py --live          # 增量，挂 cron
python ingest.py --status        # 表里有什么
```

回填和增量是**同一张表的两端，可以并行**：先开 `--live` 写今天的，同时慢慢回填历史，
两边在中间自然接上（重复键被反连接剔掉）。上线不需要等回填跑完。

---

## `known_at` 的三种来源

`known_at` = 这条数值**最早可用**的时刻。取错就是骨架点名的第一号自欺，所以语义只在 `ingest.py` 里定义一次。

| method | 来源 | 定义 | 用途 |
|---|---|---|---|
| `exact` | Single Runs API | `run` 参数（模型初始化时刻）+ 发布延迟 | 训练 / 回测的主力 |
| `request` | 实时 forecast API | 请求时刻 | 增量；它 ≥ 真实可见时刻，**只会更严格，永不泄漏** |
| `approx` | Previous Runs API | `interval_time − N×24h + 发布延迟` | 时效-技能曲线；`strict=True` 下默认排除 |

**发布延迟**很关键：00Z 初始化的 ECMWF HRES 不是 00Z 就能用的。`PUBLISH_DELAY_H` 里按模型给了
4–6 小时，宁可保守。

`approx` 的推导取的是可能区间 `[T−(N+1)×24h, T−N×24h]` 的**最晚端**，所以是保守上界。
副作用很漂亮：`day0` 算出来的 `known_at > interval_time`，也就是"目标时刻之前不可见"——
完全正确，`day0` 对历史时刻本质就是分析场，本来就不该当特征。

---

## 真值的两段式

ERA5 有约 5 天延迟，所以最近几天没有最终真值。硬扛会让最近的评分变成假的，绕开的办法是分两段：

| source | quality | 延迟 | 说明 |
|---|---|---|---|
| `era5` | `final` | ~5 天 | 最终口径，评估以它为准 |
| `ifs_analysis` | `provisional` | ~6 小时 | ECMWF IFS 9km 分析场，先顶着算 |

`truth` 表的主键含 `source`，所以两条会**并存**而不是互相覆盖；`get_labels()` 优先取 `final`。
`--live` 每次顺手抓临时真值，定期跑一次 `--backfill-truth --source era5` 把最终真值补齐，
两者自动对账。

注意 `truth` 也有 `known_at`（= `interval_time` + 延迟）：**评估时刻能看到的标签也是滞后的**，
这一点必须进回测，否则最近 5 天的分数是自己骗自己。

---

## 存储：Parquet 为准，DuckDB 当引擎

```
data/
  forecast/dt=2026-08-27/part-<run_id>.parquet
  truth/   dt=2026-08-27/part-<run_id>.parquet
  ingest_log/...
```

"只追加不覆盖"在这里是**文件级的物理事实**，不是约定：

- 每次采集写**新文件**，既有文件永不改写、永不删除
- 重复键在写之前用反连接剔掉，同一条事实只落一次
- 回滚 = 删掉某个 `run_id` 的那些文件

`store.connect()` 把每张表注册成 Parquet 之上的 DuckDB 视图，外加 `forecast_nem` / `truth_nem`
两个 NEM 时间视图。**磁盘上一律 UTC**；NEM 时间 = UTC+10 固定无夏令时，只在读的时候转。

---

## L2 入口

```python
import store, features
con = store.connect()

# 站在 2026-08-20 06:00 UTC，取对未来 1..48h 的全部可见预报
X = features.get_features(con, "2026-08-20 06:00", horizons=range(1, 49),
                          sites=["NSW1_CWO"], strict=True)

# 当时已经落地的标签（ERA5 5 天延迟会体现在这里）
y = features.get_labels(con, "2026-08-20 06:00", final_only=True)

# 逐时刻重建的训练集
df = features.make_training_frame(con, "2026-08-20 06:00", horizons=range(1, 25))
```

三重保险：SQL 里 `WHERE known_at <= as_of` → `QUALIFY` 只留当时最新的一版 →
返回前 `assert_no_leak()` 再验一遍，越界抛异常而不是静默返回。

---

## 调用量与配额

单次调用可以带多个坐标，5 个站点合成一次请求。

| 任务 | 调用数 | 时间 |
|---|---|---|
| ERA5 真值，2.5 年 | ~8（按 120 天分块） | 分钟级 |
| previous-runs 回填，2.5 年 × 1 模型 | ~8 | 分钟级 |
| single-runs 回填，1 年 × 2 run/天 | ~730 | 按 `WQ_PACE_SECONDS` 节流，几十分钟 |
| `--live` 一次 | 2 | 秒级 |

免费非商用档有速率限制。客户端已内置指数退避 + `Retry-After` + `WQ_PACE_SECONDS` 节流
（默认 0.35s/次）。大批量回填建议分几天跑，或先只回填要训练的那段。

`--backfill-forecast --method single-runs` 超过 400 次调用会要求 `--yes` 确认。

---

## 接到现有系统上

`engine.py` 里的 `truth_series()` / `make_forecast()` 是合成的。换成真数据只动两个地方：

1. 真值 ← `features.get_labels(con, as_of, final_only=True)`
2. 预报 ← `features.get_features(con, as_of, horizons)`

下游的共形校准、DM 检验、MCS、BESS DP、POT-GPD 全部原样复用——它们只认 `(预报, 真值)` 配对。

---

## 已知边界

- **HTTP 层没有在本地跑通过**：写这套脚本的沙箱不能访问 Open-Meteo（出网白名单只有包源）。
  存储语义、as-of 逻辑、去重、NEM 时区换算都有离线测试覆盖（`--selftest` 8 项 + `pytest` 20 项），
  但**参数名和返回结构需要你第一次联网时核对一遍**。建议先跑一个小区间：
  `python ingest.py --backfill-truth --start 2026-08-01 --end 2026-08-05`，
  再 `python ingest.py --status` 看看行数和时间范围对不对。
- 存档起点按 Open-Meteo 文档写在 `config.py` 的 `archive_from`：
  ECMWF IFS HRES 9km 从 2024-03-14，其余模型从 2026-04-02。想回填更长历史，
  目前只有 ECMWF 这一条路。
- AEMO 侧（NEMSEER / NEMOSIS）还没接，等电价管道要用时再加一个 `aemo.py`，
  写进同一张 `forecast` / `truth` 表即可，schema 不用变。

---

# 迁移：从合成数据切到真实数据

三个新文件，把 L1 存储接到评估管道上。**`engine.py` 一行没改**——里面的 DM 检验、
MCS、共形、pinball、CRPS、BESS DP、POT-GPD 只认 `(预报, 真值)` 数组。

| 文件 | 作用 |
|---|---|
| `realdata.py` | 适配层：把时点存储的行装配成 `engine.py` 期望的数组 |
| `build_data_real.py` | 主管道：产出运行台的 `data.json` |
| `make_fixture.py` | 造一份**格式真、内容合成**的存储，用来离线端到端验证 |

## 先用 fixture 验证，再切真数据

```bash
cd l1
python make_fixture.py --days 400            # 写到 ../data_fixture
WQ_DATA_DIR=../data_fixture python build_data_real.py --out ../data_real.json
cd .. && cp data_real.json data.json && python build.py
open WeatherQuant-NEM-运行台.html
```

fixture 跑通说明装配逻辑（时效分桶、样本对齐、缺口处理、分位计算）没问题。
然后把 `WQ_DATA_DIR` 指回真的 `data/`，同一条命令即可：

```bash
python build_data_real.py --coverage-only     # 先看每个站点能装配出多少小时
python build_data_real.py --out ../data.json
```

## 三条装配规矩

1. **固定预报时效**（`--lead-day`，默认 D+1）。模型之间必须在同一时效上比，
   否则排行榜比的是时效不是模型。
2. **样本完全重合**。只保留所有模型 + 真值都有值的时刻，DM 检验与 MCS 才成立。
3. **真值默认只用 `final`**（ERA5）。`--allow-provisional` 可放宽，但评分会掺进
   未定稿的标签。

## 模型清单只列真实存在的东西

| 键 | 来源 | 需要训练 |
|---|---|---|
| `persist` / `clim` | 由真值直接构造 | 否 |
| `gem` / `icon` / `gfs` / `ecmwf` | NWP 原始输出 | 否 |
| `mm` | 多模式等权平均 | 否 |
| `qra` | 多模式中位数 + 分裂共形 | 否 |

LightGBM 残差修正、PatchTST、DQR 在训练之前**不会出现在排行榜里**。
`--strict` 只用 `exact`/`request` 的 `known_at`；默认允许 `approx`，
因为它是保守上界，配合 lead 分桶是安全的。

## 已知取舍

- 只有 4 个 NWP 成员，`q5`/`q95` 是 min/max 再经共形放大，不是 21 成员那种平滑分位。
  共形的 λ 会替你把覆盖率拉回名义值，但锐度不如大集合。
- 存档回填只含历史目标时刻，所以**顶部运行窗口没有未来预报**。
  跑一次 `ingest.py --live` 就能补满，`meta.window_ok` 会告诉你够不够 48h。
- 时效衰减曲线的点数 = 存储里实际有的时效桶数（`--max-day` 决定）。

---

# AEMO 接入与「分水岭」分析

## `aemo.py` —— NEMWeb → 同一张时点存储表

schema 一行没改。约定：**市场数据的 `site_id` 直接用 NEM 区代码**（NSW1 / QLD1 …），
和气象站点 id（NSW1_CWO …）天然区分。

```bash
python aemo.py --selftest                                  # 12 项离线检查，不联网
python aemo.py --probe                                     # 探测 NEMWeb 可达性与文件名
python aemo.py --archive --start 2025-01 --end 2026-07     # MMSDM 月度归档回填
python aemo.py --current --hours 24                        # Reports/Current 增量（2 天窗口）
python aemo.py --predispatch                               # 官方预测 → forecast 表
```

拿三样：`DISPATCH/PRICE`（RRP）、`DISPATCH/REGIONSUM`（需求、半调度出力）→ truth；
`PREDISPATCH` / `P5MIN` → forecast（`model = aemo_predispatch` / `aemo_p5min`）。
后者是骨架第 2 条要的**官方预测基准线**。

### 三个写死在代码里的坑

1. **SETTLEMENTDATE 是区间结束时刻，且是 NEM 时间（UTC+10）。**
   标着 00:05 的那条覆盖 00:00–00:05，所以 UTC 小时是 `(SD − 5min − 10h).floor('h')`。
   少减那 5 分钟会整体错一格。
2. **必须过滤 `INTERVENTION = 0`。** 干预运行的价格不是市场出清价。
3. **`known_at ≠ interval_time`。** 调度价在区间结束后约 1 分钟发布，这里保守取 +10 分钟；
   预调度取 `RUN_DATETIME + 5min`。

`--selftest` 用一段人造 AEMO 格式 CSV 验证解析、C/I/D 分段、嵌套 zip、干预过滤、
时区与区间端点、小时聚合、known_at ——全部离线。

## `price_link.py` —— 这个系统能不能赚钱

```bash
python price_link.py --spike 300 --out ../price_link.json
```

三个层次，一层比一层严：

| 层 | 问的问题 |
|---|---|
| ① 真值缺口 → RRP | 物理关系存不存在。不存在，后面都不用谈 |
| ② 预报缺口 → RRP | 你实际能拿到的信号还剩多少。**这才是可交易的那个数** |
| ③ 预报缺口 → (RRP − AEMO 预调度) | 相对官方预测的增量。骨架第 2 条：真正的 alpha 是预测 AEMO 预测的偏差 |

**控时段尖峰提升**：不控制时段会被日循环骗——缺口最大的时刻多在夜间，
而夜间需求低本来就不出尖峰，于是会冒出"顶档尖峰率反而更低"这种反直觉结果。
做法是在每个整点内部给缺口排名，各取前 20%，再和同一批小时的基准率比。

## 结算价切换

```bash
python build_data_real.py --price-source aemo   # 强制用真实 RRP
python build_data_real.py --price-source proxy  # 只用净负荷代理
python build_data_real.py                       # auto：有 RRP 就用
```

排程仍按净负荷代理做（它是天气→价格的可微映射），**收益按 `p_real` 结算**。
接上真实 RRP 之后价值捕获率会掉——这是对的，代理价那个数本来就是上界。

## 数据模式自动识别

`build_data_real.py` 会读 truth 表的 `source`，只要有 `fixture*` 就把
`meta.data_mode` 标成 `fixture`，网页顶栏显示「FIXTURE · 格式真内容合成」，
并在缺口↔RRP 那一行加警告：合成 RRP 本来就由净负荷生成，ρ 必然高，
它证明的是**分析代码正确**，不是市场里真有这么强的关系。

---

# L6 修正记录：从"数字能跑出来"到"数字可信"

真实 RRP 接进来之后，第一版结果是**负相关**——风光越差电价越低，物理上讲不通。
显著的错方向不是"没信号"，是代码有病。下面是查出来的四处，以及为什么每一处
都不会报错、只会安静地把结论带偏。

## ① 基准用了全样本年均值（致命）

```python
norm_h = 每个整点在全样本上的 VRE 均值   # 2.5 年
def = clip(norm_h - vre, 0, None)
```

基准是两年半的年均值，能和电价对上的样本只有 1–6 月。于是盛夏几乎恒为 0、
冬天恒为正——这个"缺口"测的是**季节**，不是"今天风光差不差"。
`clip(...,0,None)` 还把一半样本压成并列的 0，Spearman 在大量并列上会失真。

**改**：基准换成"同一整点、过去 30 天"的滚动均值，随季节自己漂移；不再 clip，
用带符号的异常值。

## ② 相关没有控混杂

日循环和季节都是**公开信息**，交易上不值钱。不控掉，ρ 测的是它们。

**改**：按 `(整点 × 年月)` 分层，组内转秩再求相关。值钱的是"同一个 6 月的
下午 3 点，今天比往常差多少"。

## ③ 基准里掺了看不到的真值（前视泄漏）

滚动基准是用 ERA5 算的，而 ERA5 有约 5 天延迟。要在 t−24h 决策，那一刻能拿到的
真值只到 t−6 天。不后移，基准里就有"决策时还看不到的东西"。

**改**：`_clim_trailing(..., lag_days=6)`。①② 共用同一个基准——两者只应差在
"用真值 VRE 还是预报 VRE"，否则 ① 不再是 ② 的上限。
（实测这一项对数值影响很小，但它是对错问题，不是大小问题。）

## ④ ③ 层拿 AEMO 的临报当对手（不公平比较）

原来取 `known_at` 最晚的一版预调度——那是提前半小时的临报。拿我们提前 24 小时的
预报去比它，等于让对手先看半天答案，③ 被系统性压低。

**改**：取 lead 最接近本面板时效的那一版，同时效对比。

## 顺带修掉的

- **自检污染真库**：`ingest.py --selftest` 一直往真实 `data/` 写 63 行
  `source='selftest'` 的人造行（`NSW1_CWO` 比别的站多 63 行就是它）。改写临时目录。
- **残缺小时**：一小时不足 10 个 5 分钟区间也照样求均值——得到一个看起来正常、
  其实代表不了这一小时的数。现在直接丢弃，并去重防止重复行"凑满"。
- **`df.get("INTERVENTION","0").astype(str)`**：列缺失时返回 Python 字符串，
  后面 `.astype` 会炸。改成显式判断。
- **p 值下溢**：`(p or 1) < 0.01` 在 `p == 0.0` 时走 `or` 的右支，
  最显著的结果反而丢了星号。
- **尖峰用错列**：一次 $10000 的 10 分钟尖峰，摊到小时均价只剩 $900。
  尖峰改用小时内最高价 `rrp_max`，相关仍用均价。

## 新增的两件工具

**时间对齐体检 `_lag_scan`**：把电价整体前后平移 ±12 小时，看相关峰值落在哪。
缺口和电价是同一时刻的物理量，峰值就该在 0。落在 ±10 → NEM(UTC+10) 偏移用反；
落在 ±1 → 区间起点/终点记反。一个数暴露一整类静默错误，比反复读时区代码可靠。

**移动块自助区间**：电价和天气强自相关，普通 p 值把 4344 个小时当成 4344 个独立
观测，显著性虚高。块长 24h、B=400 的区间才是能信的那个。区间跨过 0 就是和零分不开。
5 个区 × 5 个时段 = 25 次检验，不看区间只看 p<0.01 一定会捡到假阳性。

## 还没解决的

- **③ 缺历史预调度**。`--current` 只有两天。MMSDM 里对应 `PREDISPATCHPRICE`，
  可以试 `--tables PREDISPATCHPRICE`（解析器复用 `PREDISPATCH/REGION_PRICES`，
  但这个表名我没有联网验证过）。文件比 DISPATCHPRICE 大很多。
- **尖峰样本太少**。半年里 QLD1 只有个位数小时超 $300。NEM 真实尖峰率本就是
  零点几个百分点，做尖峰方向需要多年回填，不是半年。
- **只有 ECMWF 一家**。GFS/ICON/GEM 的 previous-runs 存档只到 2026-04，
  多模型集成与 MCS 只能在那之后的窗口上做。

---

# L3：可交易信号 `l3_signal.py`

诊断代码不能直接当信号用。`price_link` 为了回答"关系存不存在"，可以用全样本分层、
可以一次性切半段；`l3_signal` 要每天给出一个数，所以每一步只能用**决策时刻已可见**
的数据。诊断代码直接拿去当信号，是这类项目最常见的死法：分层用了全样本的秩、
标准化用了全样本的均值，回测漂亮，实盘归零。

三处 as-of 纪律：

1. **滚动分位**代替组内秩。组内秩要看完整组才能算，其中含未来；
   `trailing_pct` 只回看"过去 N 天、同一整点"的窗口。
2. **真值类特征后移** `TRUTH_LAG_D` 天，让开 ERA5 的发布延迟。
3. **滚动重拟合**：每 7 天用此前 120 天重估一次权重，训练窗与预测段之间留 3 天禁运
   （天气过程 3~7 天，训练窗末尾紧挨预测段开头等于看了同一场天气的另一半）。

```
python l3_signal.py --region QLD1 --hours 10-16
```

输出里最该看的不是 ρ，而是**权重稳定性**：`同号比例 100%` 意味着 38 次重拟合
每一次都给出同方向的权重。同号比例掉到 70% 以下，说明这个信号在换季时会翻脸。

**文件名不叫 `signal.py`**——那会盖住标准库的 `signal`：`l1/` 在 `sys.path[0]`，
任何库执行 `from signal import Signals` 都会拿到我们的文件，进程里一大片东西直接崩。
这个坑踩过一次，pytest 整个跑不起来。

## 仍未解决

`l3_signal` 只做到第 ② 层。**③ 层（相对 AEMO 预调度的增量）还没有数据。**
在 ③ 出来之前，② 再高也可能只是把公开信息重算了一遍，不构成 alpha。

---

# 独立审计（第三方复核）修掉的问题

让一个没参与开发的审计者复核 `price_link.py` / `l3_signal.py` 的统计逻辑，
查出 14 条，其中 5 条会**改变结论**。最严重的一条我自己踩了两次。

## 致命：自助块长短于序列自身的记忆

块长写死"一天"或"一周"，而信号和目标都是 `win_days` 天的滚动分位，按构造带
几十天的记忆。块比记忆短，重采样破坏不了自相关，区间就虚窄。

**实测**：特征与电价完全独立时，旧代码把 **1/3** 的试验判成「可用」，
名义应该是 5%。修好后 **1/24 = 4%**。

修法不是改成另一个写死的数，而是**量出来**：算积分自相关时间 τ，块长取 3τ，
上界 n/12（`_boot_ci` 要求 n ≥ 块×10，取到 n/8 会让它一律返回 None）。
真数据上 τ ≈ 12 行 ≈ 2 天，块长 ≈ 6 天——正好是天气过程的尺度，是个好旁证。

同一个坑在 `drivers_day` 里还埋着一份：`24*7=168` 行只在"整点全留"时等于一周，
日间只留 6 小时/天，168 行就是 **28 天**，CV 的禁运带被撑成 4 周。

## 空对照只抽 20 次

`np.percentile(x, 95)` 在 20 个点上实质就是最大值，自身抽样误差和被估的量同量级。
而这是把关的那个数。改成 200 次。

## 十分位用全样本分位切档

对外宣称的 `$/MWh 价差` 是这么算出来的：用**整段回测**的分位点切档。
在 t 时刻你不可能知道今天的信号在整段里排第几——这是前视。
改成滚动分位切档，窗口没攒够的行不进任何档。

## `spk > spike` 是布尔数组，NaN 变 False

`np.nanmean(spk > spike)` 里已经没有 NaN 可跳，缺失的小时被静静算作"没有尖峰"。
`spk=[100,nan,nan,nan,500]` 报 20%，真实是 50%。
`neg_share` / `n_spikes` / 各时段尖峰率四处同病。

## 提升倍数的分子分母来自不同样本

分子只用拟合非空的行，分母用全样本。两批样本无条件尖峰率的差异会冒充成"提升"。

## 其余

- `_clim_trailing` 的 `expanding(min_periods=1)` 兜底会拿 1~7 个观测冒充
  "30 天气候态"，每个整点各 7 行，且正好落进十分位的极端档 → 改成给 NaN。
- p 值自由度是错的（逐小时强自相关 + 组内去均值消耗几百自由度），全表必然带星
  → 星号改由**自助区间是否含 0** 决定。
- `_ranks_for_boot` 少了组大小过滤，区间和它标定的 ρ 算在两批样本上。
- `_lift_hour_adjusted` 把 `rrp_max` 传进 `rrp` 位，JSON 里的中位价标签是错的。
- 分季切分 `tr_end = max(100, h - gap)` 会在样本紧张时**悄悄取消禁运** → 改成拒绝出数。
- "信号最强"用带符号的 max，且 `rho=None` 时会崩。

## 审计确认没问题的部分

`trailing_pct` 无泄漏（扰动测试逐点验证）、`walk_forward` 的训练/禁运/测试划分正确、
`drivers_decomp` 的 `_pos` 回填在组过滤后仍然正确、`_block_shuffle` 是真置换、
`_fit_eval` 的两侧禁运逻辑正确、消融的同样本控制正确、AEMO 与气象的时间对齐一致。

---

# 判定台页面 `l6_report.py`

把 `price_link.py` / `l3_signal.py` 的 JSON 渲染成一页 HTML。页面只回答一个问题：
**这条信号现在能不能拿去下单。**

```
python price_link.py --out ../price_link.json
python l3_signal.py --region QLD1 --hours 10-16 --ablation --out ../signal.json
python l6_report.py --link ../price_link.json --signal ../signal.json --out ../L6判定台.html
```

三条设计约束，都是被之前的错误逼出来的：

1. **每个 ρ 都画在它自己的噪声地板上。** 单看数字说明不了任何事——
   「0.35 而地板 0.06」和「0.35 而地板 0.34」是完全相反的结论。刻度固定 0~0.7，跨区可比。
2. **数据来源跟着结果走。** JSON 里带 `is_fixture`，页面顶部据此挂横幅。
   fixture 的合成 RRP 由净负荷生成，相关必然偏高；不标出来，页面就会拿合成数字冒充结论。
3. **页面上每一句断言都由数字生成。** 曾经写死"手填权重全线失效"，
   而在 fixture 上手填恰好有效——那句话就在自己的表格旁边说谎。
   现在结论文本从 `rows` 里数出来。

## 交易台 vs 研究报告

页面的问题是「今天该怎么下手」，不是「这些数据之间有什么关系」。所以：

- **前瞻档位表**：`l3_signal.py --forward` 给**还没出价的小时**打分。
  用截至最后一个有价小时的 `train_days` 天拟合，中间留禁运；
  档位边界取**训练段**的分位——用预测段自己的分位切档，等于拿未来定义档位。
- **档位阶梯**：每一档历史上的中位价、尖峰率、对应动作。这是**条件分布**不是点预测，
  差价来自反复在两端建仓，不来自赌任何单独一小时。
- **风险闸门置顶**：③ 层没算出来就只能纸上跟踪。
  闸门**先看数据真假再看数字大小**——fixture 上 ③ 必然显著（预调度也是合成的），
  在那种数据上说"可以谈仓位"是这一页最不该犯的错。

```
python l3_signal.py --region QLD1 --hours 10-16 --ablation --forward --out ../signal.json
python l6_report.py --link ../price_link.json --signal ../signal.json --out ../L6交易台.html
```

## PREDISPATCH 运行时刻：不猜，从数据标定

真实事故：MMSDM 的 `PREDISPATCHPRICE` 没有 `RUN_DATETIME` 列，代码退回按
`PREDISPATCHSEQNO(YYYYMMDDPP)` 的 "PP × 30 分钟" 反推——但 PP 是**交易日内**的
运行序号，交易日从 04:00 开始不是午夜。算出来的运行时刻整体偏 4 小时，
时效全部掉出 6~48h 带，**87600 行解析出来、写进去 0 行，还不报错**。

现在改成：同一个 seqno 就是同一次运行，它覆盖的最早目标时刻必然紧跟运行时刻之后，
所以 `run ≈ 该组 min(DATETIME) − 30min`。不依赖任何列名约定。
并且逐段打印行数（解析 → 时效带内），静默丢光不可能再发生。

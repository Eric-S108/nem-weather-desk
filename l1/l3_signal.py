"""
L3：把三个气象量拟合成一个**可交易**的信号。
=============================================
和 price_link 的分工要分清楚：

  price_link 是**诊断**——关系存不存在、符号对不对、是不是噪声。
               它可以用全样本分层、可以一次性切半段，因为它只回答"有没有"。
  signal     是**生产**——每天给出一个数，将来要拿去下单。
               所以每一步都必须只用**决策时刻已经可见**的数据。

诊断代码直接拿来当信号用，是这类项目最常见的死法：分层用了全样本的秩、
标准化用了全样本的均值，回测漂亮，实盘归零。这个文件的每个函数都为此让路。

三处 as-of 纪律：
  ① 秩变换只对**过去同一整点**的滚动窗口做，不对全样本做
  ② 由真值构造的特征必须整体后移，让开 ERA5 的约 5 天发布延迟。
     **目前 build_features 只用预报量（t−24h 即可见），所以 lag_days=0 是对的。**
     将来若加入气候基准之类的真值特征，必须显式传 lag_days≥6，别指望这里替你做。
  ③ 滚动重拟合：训练窗只含 t 之前的数据，且与预测段之间留禁运间隔

    python l3_signal.py --region QLD1 --hours 10-16
    python l3_signal.py --region QLD1 --hours 10-16 --out ../signal_qld1.json

文件名不叫 signal.py：那会盖住 Python 标准库的 signal 模块——l1/ 在 sys.path[0]，
任何库执行 `from signal import Signals` 都会拿到这个文件，进程里一大片东西直接崩。
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import price_link as PL  # noqa: E402
import realdata as R  # noqa: E402
import store  # noqa: E402
from config import SITES  # noqa: E402

FEATURES = ["ghi", "wind", "temp"]


def trailing_pct(x, hod, days=60, lag_days=0, min_obs=12):
    """x[t] 在「过去 days 天、同一整点」里的分位（0~1）。

    这是 price_link 里"组内秩"的 as-of 干净版本。区别很关键：
    组内秩要看完这一组的全部样本才能算，其中包含未来；
    这里只回看过去，所以每个时刻的值在当时就能算出来。

    lag_days>0 用于由真值构造的特征（ERA5 有发布延迟）；
    预报量本身在 t−24h 就可见，lag_days=0。
    """
    x = np.asarray(x, float)
    hod = np.asarray(hod)
    out = np.full(len(x), np.nan)
    for h in np.unique(hod):
        idx = np.where(hod == h)[0]
        if len(idx) < min_obs:
            continue
        # 窗口内小于当前值的比例。窗口严格取自过去，且再往前推 lag_days 天。
        vals = x[idx]
        n = len(idx)
        res = np.full(n, np.nan)
        for i in range(n):
            lo = max(0, i - days - lag_days)
            hi = i - lag_days
            if hi - lo < min_obs:
                continue
            w = vals[lo:hi]
            res[i] = float(np.mean(w < vals[i]))
        out[idx] = res
    return out


def build_features(con, region: str, site_id: str, lead_day: int = 1,
                   win_days: int = 60, keep_future: bool = False) -> pd.DataFrame:
    """装配 (特征, 电价) 表。特征全部是 as-of 干净的滚动分位。"""
    p = R.fill_persist_head(R.load_backtest(con, site_id, lead_day=lead_day))
    q = p.preds["qra"]
    raw = {"ghi": q["ghi"], "wind": q["wind100"], "temp": q["temp"]}

    d = pd.DataFrame({"interval_time": pd.to_datetime(p.t), "hod": p.hod})
    for k, v in raw.items():
        # 预报量在 t−24h 就可见，不需要 lag
        d[k] = trailing_pct(v, p.hod, days=win_days, lag_days=0)

    mk = con.execute("""
        SELECT interval_time, variable, value FROM truth
        WHERE site_id = ? AND variable IN ('rrp','rrp_max')
        QUALIFY row_number() OVER (PARTITION BY variable, interval_time
                                   ORDER BY (quality='final') DESC, known_at DESC) = 1
    """, [region]).df()
    if mk.empty:
        raise ValueError(f"{region}: truth 表里没有 rrp。先跑 aemo.py --archive。")
    mw = mk.pivot_table(index="interval_time", columns="variable",
                        values="value", aggfunc="last").reset_index()
    # keep_future=True 时用左连接：**气象预报已经有、电价还没出**的那些小时要留下来，
    # 它们正是要拿去下单的时段。内连接会把它们静静丢掉，只剩回测。
    how = "left" if keep_future else "inner"
    d = d.merge(mw, on="interval_time", how=how).sort_values("interval_time")
    d = d.reset_index(drop=True)
    # 目标也做同样的滚动分位。用绝对价格拟合，会被几次尖峰主导；
    # 用分位，问的是"比同一时刻的常态贵多少"，那才是这个信号能回答的问题。
    d["y"] = trailing_pct(d["rrp"].to_numpy(float), d["hod"].to_numpy(),
                          days=win_days, lag_days=0)
    return d


ALL_FEATURES = ["ghi", "wind", "temp"]


def walk_forward(d: pd.DataFrame, train_days: int = 120, step_days: int = 7,
                 embargo_days: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """滚动重拟合。每 step_days 用**此前** train_days 的数据重估一次权重，
    预测接下来的 step_days。训练窗与预测段之间留 embargo_days 的空白。

    禁运的理由和 price_link 里一样：天气过程 3~7 天，训练窗末尾紧挨预测段开头，
    等于让模型看到了同一场天气的另一半。
    """
    # 永远按**全部**特征去空行，即使这次只用其中一个。
    # 否则单特征子集会比全量多出几行，消融比较的就不是同一批样本了。
    have = [c for c in ALL_FEATURES if c in d.columns]
    d = d.dropna(subset=have + ["y"]).reset_index(drop=True)
    if d.empty:
        raise ValueError("特征全为空：滚动分位窗口还没攒够，把 --win-days 调小或回填更长区间")
    t = pd.to_datetime(d["interval_time"])
    day = (t - t.iloc[0]).dt.days.to_numpy()
    # 门槛要按**天**算。筛掉时段后每天只剩几个小时，用小时数判断会误报"样本不够"——
    # 6 小时/天 × 400 天 = 2400 小时，看着少，其实横跨的时间足够训练窗滚动。
    span_d = int(day.max()) + 1
    if span_d < train_days + embargo_days + step_days:
        raise ValueError(f"样本只横跨 {span_d} 天，不够跑 "
                         f"train={train_days} + embargo={embargo_days} + step={step_days}")
    X = d[FEATURES].to_numpy(float)
    y = d["y"].to_numpy(float)

    pred = np.full(len(d), np.nan)
    rows = []
    start = train_days + embargo_days
    for d0 in range(start, int(day.max()) + 1, step_days):
        tr = day < (d0 - embargo_days)
        tr &= day >= (d0 - embargo_days - train_days)
        te = (day >= d0) & (day < d0 + step_days)
        if tr.sum() < 200 or te.sum() < 12:
            continue
        A = np.column_stack([X[tr], np.ones(tr.sum())])
        beta, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
        pred[te] = X[te] @ beta[:-1] + beta[-1]
        rows.append({"day": int(d0), "n_train": int(tr.sum()), "n_test": int(te.sum()),
                     **{f"w_{k}": round(float(b), 4) for k, b in zip(FEATURES, beta[:-1])}})

    d = d.assign(signal=pred)
    return d, pd.DataFrame(rows)


def forward(con, region: str, site_id: str, lead_day: int, win_days: int,
            train_days: int, embargo_days: int, hours: tuple[int, int] | None) -> dict:
    """给**还没出价的那些小时**打分——这才是拿去下单的东西。

    用截至最后一个有价小时的 train_days 天拟合（中间留 embargo_days 禁运），
    再去给之后的小时打分。回测回答"这套办法过去行不行"，
    这里回答"明天那几个小时该怎么办"。
    """
    d = build_features(con, region, site_id, lead_day, win_days, keep_future=True)
    if hours:
        d = d[(d["hod"] >= hours[0]) & (d["hod"] < hours[1])]
    d = d.dropna(subset=ALL_FEATURES).sort_values("interval_time").reset_index(drop=True)
    if d.empty:
        return {"status": "no_features"}

    has_y = d["y"].notna()
    if not has_y.any():
        return {"status": "no_price"}
    t = pd.to_datetime(d["interval_time"])
    last_priced = t[has_y].max()
    fut = d[t > last_priced]
    if fut.empty:
        return {"status": "no_forward", "last_priced": str(last_priced)}

    # 训练集：最后一个有价时刻往前 train_days，且与预测段留出禁运
    cut = last_priced - pd.Timedelta(days=embargo_days)
    tr = d[has_y & (t <= cut) & (t > cut - pd.Timedelta(days=train_days))]
    if len(tr) < 200:
        return {"status": "too_few_train", "n_train": int(len(tr))}
    X = tr[ALL_FEATURES].to_numpy(float)
    beta, *_ = np.linalg.lstsq(np.column_stack([X, np.ones(len(X))]),
                               tr["y"].to_numpy(float), rcond=None)
    sig = fut[ALL_FEATURES].to_numpy(float) @ beta[:-1] + beta[-1]

    # 档位边界取**训练段**的分位，不用预测段自己的——后者等于用未来定义档位
    qs = np.quantile(tr[ALL_FEATURES].to_numpy(float) @ beta[:-1] + beta[-1],
                     np.linspace(0, 1, 11))
    dec = np.clip(np.searchsorted(qs, sig, side="right"), 1, 10)

    rows = [{"t": str(ts), "hod": int(h), "signal": round(float(v), 4), "decile": int(k)}
            for ts, h, v, k in zip(fut["interval_time"], fut["hod"], sig, dec)]
    return {"status": "ok", "region": region,
            "trained_to": str(cut), "last_priced": str(last_priced),
            "n_train": int(len(tr)), "n_forward": len(rows),
            "weights": {k: round(float(b), 4) for k, b in zip(ALL_FEATURES, beta[:-1])},
            "rows": rows}


def _rows_per_day(t) -> float:
    """样本里平均每天有几行。筛过时段之后这不是 24。

    用**实际出现过的日历日数**做分母，不用首尾跨度：滚动重拟合会跳过样本不足的折，
    留下 NaN，跨度分母会把这些空洞算进去，rows_per_day 偏小，
    进而让下面所有块长都偏短——块长偏短正是自助区间虚窄的根源。
    """
    t = pd.to_datetime(pd.Series(t))
    days = max(1, int(t.dt.normalize().nunique()))
    return len(t) / days


def acf_time(x, max_lag=None) -> float:
    """积分自相关时间：ACF 累加到首次转负为止。

    块长不该靠猜。写死一天 → 破坏不了自相关，区间虚窄（实测两条独立序列也能
    "显著"，假阳性约 1/3）；写死 win_days 天 → 又过于保守，两年样本都凑不出 6 个块。
    正确做法是**量出来**：τ 就是这条序列自己的记忆长度，块长取它的若干倍。
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 50:
        return 1.0
    x = x - x.mean()
    v = float(np.dot(x, x))
    if v <= 0:
        return 1.0
    max_lag = max_lag or min(n // 4, 2000)
    tau = 1.0
    for k in range(1, max_lag):
        r = float(np.dot(x[:-k], x[k:])) / v
        if r <= 0:
            break
        tau += 2.0 * r
    return max(1.0, tau)


def _rate(flag, finite) -> float:
    """在**有观测**的行上算比例（百分数）。缺失不算作 False。"""
    n = int(np.sum(finite))
    return float(np.sum(flag & finite)) / n * 100.0 if n else float("nan")


def evaluate(d: pd.DataFrame, spike: float = 300.0, win_days: int = 60) -> dict:
    """样本外评估。所有数字都来自滚动重拟合的预测，没有一处用到未来。"""
    m = np.isfinite(d["signal"]) & np.isfinite(d["y"])
    if m.sum() < 200:
        return {"status": "too_few", "n": int(m.sum())}
    s = d.loc[m, "signal"].to_numpy(float)
    y = d.loc[m, "y"].to_numpy(float)
    rrp = d.loc[m, "rrp"].to_numpy(float)
    spk = (d.loc[m, "rrp_max"].to_numpy(float)
           if "rrp_max" in d.columns else rrp)
    # spk > spike 是布尔数组：NaN 比较后变 False，nanmean 已经没有 NaN 可跳，
    # 于是缺失的小时被当成"没有尖峰"，基准尖峰率被系统性压低。必须先掩码。

    # 块长必须按**实际行数**折算，不能写死 24 / 24*7。
    # 筛掉时段之后，一天只剩几行：10-16 时段是 6 行/天，一周 = 42 行，不是 168 行。
    # 用 168 行当"一周"，276 行的样本里就只有 2 个块——打乱两个块几乎等于没打乱，
    # 空对照于是逼近真值本身（这次看到 地板 0.486 == ρ 0.486，就是这个 bug）。
    rpd = _rows_per_day(d.loc[m, "interval_time"])
    # 块长必须**长过序列自身的记忆**，否则自助/打乱破坏不了自相关，区间虚窄、地板虚低。
    # 这里的记忆长度不是一天也不是一周：信号和目标都是 win_days 天的滚动分位，
    # 按构造就带 win_days 天的记忆。用一天的块，两条互相独立的序列也能得出
    # 95% 区间不含 0 的"显著"结论——实测假阳性率约 1/3。
    # 块长 = 两条序列里较长的那个记忆时间的 3 倍，并且至少覆盖一天、不超过样本的 1/8
    tau = max(acf_time(s), acf_time(y))
    # 上界取 n//12：_boot_ci 要求 n >= block*10，取到 n//8 会让它一律返回 None，
    # 表面看是"算不出区间"，其实是块长上界和自助的前提互相打架。
    blk = int(np.clip(round(3 * tau), max(2, round(rpd)), max(2, len(s) // 12)))
    rho = float(stats.spearmanr(s, y)[0])
    ci = PL._boot_ci(s, y, block=blk, B=400)

    # 空对照同样按 blk 打乱。20 次抽样的 95 分位实质就是最大值，
    # 自身抽样误差和被估的量同量级——这是把关的那个数，必须抽够。
    n_blocks = len(s) // blk
    if n_blocks >= 6:
        nulls = [abs(float(stats.spearmanr(
            PL._block_shuffle(s, block=blk, seed=201 + i), y)[0])) for i in range(200)]
        floor = float(np.percentile(nulls, 95))
    else:
        floor = None

    # 分档数按样本量定：每档少于 30 个样本，中位价就是噪声，
    # 十档非单调只是抽样波动，看着却像"信号在高档失效"，会误导判断。
    # 切档必须用**滚动分位**。用全样本分位切，等于在 t 时刻就知道这个信号值
    # 在整段回测里排第几——那是未来信息，而它正好决定了对外宣称的 $/MWh 价差。
    nb = int(np.clip(len(s) // 30, 3, 10))
    s_rank = trailing_pct(s, d.loc[m, "hod"].to_numpy(), days=win_days, lag_days=0)
    dec = []
    for i in range(nb):
        sel = (s_rank > i / nb) & (s_rank <= (i + 1) / nb)
        if sel.sum() < 10:
            continue
        dec.append({"decile": i + 1, "n": int(sel.sum()),
                    "rrp_median": round(float(np.median(rrp[sel])), 1),
                    "rrp_mean": round(float(np.mean(rrp[sel])), 1),
                    "spike_rate": round(_rate(spk[sel] > spike, np.isfinite(spk[sel])), 2)})

    top = dec[-1] if dec else {}
    bot = dec[0] if dec else {}
    return {"status": "ok", "n": int(m.sum()),
            "span": [str(d.loc[m, "interval_time"].min()),
                     str(d.loc[m, "interval_time"].max())],
            "rho_oos": round(float(rho), 3), "ci": ci, "block_rows": int(blk),
            "acf_tau_rows": round(float(tau), 1),
            "null_floor": None if floor is None else round(floor, 3),
            "n_blocks": int(n_blocks), "rows_per_day": round(rpd, 1),
            "verdict": _verdict(rho, ci, floor, int(m.sum()), n_blocks),
            "deciles": dec, "n_buckets": nb,
            "spread_median": (round(top.get("rrp_median", 0) - bot.get("rrp_median", 0), 1)
                              if dec else None)}


def ablation(d: pd.DataFrame, train_days: int, step_days: int, embargo_days: int,
             spike: float, win_days: int = 60) -> list[dict]:
    """逐个特征单独跑一遍完整的滚动重拟合，再跑一次全量。

    为什么必须做：权重里 ghi 是 −0.60、wind 只有 −0.10，很可能整条信号
    就是"太阳大不大"一个变量，另外两个只是陪跑。真是这样的话，
    这个信号的价值就要重新估——预测正午光伏出力是 AEMO 自己最擅长的事。

    每个子集都走**同一套滚动重拟合**，不是在同一份拟合结果上删列，
    否则比较的不是"少这个特征行不行"，而是"少这个系数行不行"。
    """
    out = []
    subsets = [[k] for k in FEATURES] + [list(FEATURES)]
    saved = list(FEATURES)
    try:
        for sub in subsets:
            globals()["FEATURES"] = sub
            try:
                dd, _ = walk_forward(d.copy(), train_days, step_days, embargo_days)
                ev = evaluate(dd, spike, win_days)
            except Exception as e:
                out.append({"features": "+".join(sub), "rho": None, "err": str(e)[:60]})
                continue
            out.append({"features": "+".join(sub),
                        "rho": ev.get("rho_oos"), "ci": ev.get("ci"),
                        "spread": ev.get("spread_median")})
    finally:
        globals()["FEATURES"] = saved
    return out


def _verdict(rho, ci, floor, n, n_blocks) -> str:
    """判定要把"没测出来"和"样本不够所以测不了"分开说。
    两者都不能下单，但下一步动作完全不同：前者换特征，后者去补数据。"""
    if n < 500 or n_blocks < 6:
        return f"样本不足（{n} 行 / {n_blocks} 个周块），结论不可信——先回填更长的价格历史"
    if floor is None:
        return "无法估计噪声地板"
    if ci is None or ci[0] * ci[1] <= 0:
        return "95%区间跨过 0，与零分不开"
    if abs(rho) <= floor:
        return "未超过噪声地板"
    return "可用"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="L3：滚动重拟合的可交易信号")
    ap.add_argument("--region", default="QLD1")
    ap.add_argument("--hours", default="10-16", help="NEM 时段，如 10-16；all 表示全天")
    ap.add_argument("--lead-day", type=int, default=1)
    ap.add_argument("--win-days", type=int, default=60, help="滚动分位窗口")
    ap.add_argument("--train-days", type=int, default=120)
    ap.add_argument("--step-days", type=int, default=7)
    ap.add_argument("--embargo-days", type=int, default=3)
    ap.add_argument("--spike", type=float, default=300.0)
    ap.add_argument("--forward", action="store_true",
                    help="给还没出价的小时打分（拿去下单的那一张表）")
    ap.add_argument("--ablation", action="store_true",
                    help="逐特征单独重跑，看信号是不是只靠一个变量")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    site = next((s[0] for s in SITES if s[1] == a.region), None)
    if site is None:
        print(f"未知区 {a.region}"); return 2

    con = store.connect()
    try:
        d = build_features(con, a.region, site, a.lead_day, a.win_days)
        if a.hours != "all":
            lo, hi = (int(x) for x in a.hours.split("-"))
            d = d[(d["hod"] >= lo) & (d["hod"] < hi)].reset_index(drop=True)
        d, w = walk_forward(d, a.train_days, a.step_days, a.embargo_days)
        ev = evaluate(d, a.spike, a.win_days)
    finally:
        con.close()

    print("\n" + "=" * 72)
    print(f"{a.region}  时段 {a.hours}  ·  滚动重拟合信号（全部样本外）")
    print("=" * 72)
    if ev["status"] != "ok":
        print(f"  {ev}"); return 1
    print(f"  样本 {ev['n']} 小时   {ev['span'][0][:10]} → {ev['span'][1][:10]}")
    fl = ev.get("null_floor")
    print(f"  样本外 ρ = {ev['rho_oos']:+.3f}   95%区间 {ev['ci']}   "
          f"噪声地板 {('%.3f' % fl) if fl is not None else '—（周块太少，估不出来）'}")
    print(f"  每天 {ev['rows_per_day']} 行 · 自相关时间 {ev['acf_tau_rows']} 行 · 块长 {ev['block_rows']} 行"
          f"（≈{ev['block_rows']/max(ev['rows_per_day'],1):.0f} 天）· {ev['n_blocks']} 个块"
          f"（<6 个块时空对照没有意义）")
    print(f"  判定：{ev['verdict']}")
    print(f"\n  重拟合 {len(w)} 次，权重的稳定性（越稳越可信）：")
    if len(w):
        for k in FEATURES:
            c = w[f"w_{k}"]
            print(f"    {k:5s}  中位数 {c.median():+.3f}   四分位距 {c.quantile(.75)-c.quantile(.25):.3f}"
                  f"   同号比例 {max((c>0).mean(),(c<0).mean())*100:.0f}%")
    print(f"\n  信号分 {ev['n_buckets']} 档 → 真实电价：")
    print(f"    {'档':>3s}{'样本':>7s}{'中位价':>9s}{'均价':>9s}{'尖峰率':>8s}")
    for r in ev["deciles"]:
        print(f"    {r['decile']:>3d}{r['n']:>7d}{r['rrp_median']:>9.1f}"
              f"{r['rrp_mean']:>9.1f}{r['spike_rate']:>7.2f}%")
    if ev.get("spread_median") is not None:
        print(f"\n  顶档 − 底档 中位价差 = ${ev['spread_median']}/MWh")
    print("\n  注意：这只是第 ② 层。③ 层（相对 AEMO 预调度的增量）还没有数据——")
    print("  在 ③ 出来之前，② 再高也可能只是把公开信息重算了一遍，不构成 alpha。\n")

    fw = None
    if a.forward:
        con = store.connect()
        try:
            hrs = None if a.hours == "all" else tuple(int(x) for x in a.hours.split("-"))
            fw = forward(con, a.region, site, a.lead_day, a.win_days,
                         a.train_days, a.embargo_days, hrs)
        finally:
            con.close()
        if fw.get("status") != "ok":
            print(f"  前瞻打分不可用：{fw}")
        else:
            print(f"\n  前瞻打分：{fw['n_forward']} 个还没出价的小时"
                  f"（训练截至 {fw['trained_to'][:16]}，最后有价 {fw['last_priced'][:16]}）")
            byd = {}
            for r_ in fw["rows"]:
                byd.setdefault(r_["t"][:10], []).append(r_["decile"])
            for day, ds in list(byd.items())[:10]:
                bars = "".join("▁▂▃▄▅▆▇█"[min(7, (k - 1) * 8 // 10)] for k in ds)
                print(f"    {day}  {bars}  档位 {min(ds)}–{max(ds)}")
            if len(byd) > 10:
                print(f"    …… 共 {len(byd)} 天")

    ab = None
    if a.ablation:
        con = store.connect()
        try:
            d2 = build_features(con, a.region, site, a.lead_day, a.win_days)
            if a.hours != "all":
                lo, hi = (int(x) for x in a.hours.split("-"))
                d2 = d2[(d2["hod"] >= lo) & (d2["hod"] < hi)].reset_index(drop=True)
        finally:
            con.close()
        ab = ablation(d2, a.train_days, a.step_days, a.embargo_days, a.spike, a.win_days)
        print("  逐特征消融（每个子集都走同一套滚动重拟合）：")
        print(f"    {'特征':<22s}{'样本外ρ':>10s}{'顶底价差':>10s}")
        for r in ab:
            rr = f"{r['rho']:+.3f}" if r.get("rho") is not None else "  —  "
            sp = f"${r['spread']}" if r.get("spread") is not None else " — "
            print(f"    {r['features']:<22s}{rr:>10s}{sp:>10s}")
        # 结论必须由数字决定，不能写死一句话——那样无论结果如何都在"确认"我的猜测
        full = next((r for r in ab if r["features"] == "+".join(FEATURES)), None)
        best1 = max((r for r in ab if "+" not in r["features"]
                     and r.get("rho") is not None), key=lambda r: abs(r["rho"]), default=None)
        if full and best1 and full.get("rho"):
            share = abs(best1["rho"]) / abs(full["rho"])
            if share > 0.9:
                print(f"    → {best1['features']} 一个变量就拿到全量的 {share*100:.0f}%，"
                      f"这条信号实质上只有一个自由度。")
            elif share > 0.7:
                print(f"    → {best1['features']} 占了全量的 {share*100:.0f}%，"
                      f"另外两个是边际补充。")
            else:
                print(f"    → 最强单变量只占 {share*100:.0f}%，三个特征确实各有贡献。")
        print()

    if a.out:
        with open(a.out, "w") as f:
            json.dump({"region": a.region, "hours": a.hours, "eval": ev,
                       "ablation": ab, "forward": fw,
                       "weights": w.to_dict("records")}, f, ensure_ascii=False, indent=1)
        print(f"→ {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

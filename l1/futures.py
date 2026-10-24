"""
期货公允价值 —— 给只做金融合约的交易员用。
=========================================
ASX 的 NEM 电力期货按**合约期内现货价的算术平均**现金结算（基荷月度/季度、峰荷、早晚峰、$300 cap）。
所以一张期货的公允价值 = 合约期内现货均价的期望；已经过去的小时是确定的，剩下的才有不确定性。

这里只用 AEMO 的公开现货价，不碰 ASX 的价格：
  · ASX Energy 网站条款禁止爬取、下载、转载其价格（2026-06-12 版，Prohibited uses (a)(c)(e)）；
  · 所以公开页面只放**我们的公允价值和区间**；
  · 市场价由交易员自己在本机录入 private/futures_prices.csv，信号和仓位只生成在本机的 private/desk.html，
    不提交、不发布（见 signals.py）。

方法（全部可在页面上看到口径）：
  · 已结算部分：合约期内已经有实际价的小时，按产品时段直接算；
  · 未结算部分：从决策日之前 90 天的完整交易日里**按 3 天一块整块抽样**（保留连续高价/低价日的聚集），
    逐日拼出剩余天数，按目标日期的星期几套产品时段，模拟 4000 次 → 结算价分布；
  · 校准：在过去每个完整的月份和季度上，从开始前一天到结束前几天，每周站一次位置算区间，
    看实际结算价落在 50% / 90% 区间里的比例 —— 区间不准，页面会直接说；
  · 仓位：持有到交割时，每过一天就有一天的现货结算进来，单日盈亏 ≈ 当天产品小时数 × (当天均价 − 开仓价)。
    用抽样池里单日均价的 5% / 95% 分位算"95% 情况下单日最多亏多少"，按交易员给的单日亏损上限反推最大 MW。
    没算市场价每天的盯市波动（没有公开的历史期货价）。
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "futures.json")
NEM = pd.Timedelta(hours=10)
REGIONS = ["NSW1", "VIC1", "QLD1", "SA1"]          # ASX 只挂这四个区
POOL_DAYS = 90
BLOCK = 3
N_SIMS = 4000
CAP_STRIKE = 300.0

# 产品时段（NEM 时间，小时起点）。peak = 工作日 07:00–22:00（未扣公共假日）。
PRODUCTS = {
    "base": ("基荷", "全天 24 小时"),
    "peak": ("峰荷", "工作日 07:00–22:00"),
    "morning": ("早峰", "每天 06:00–09:00"),
    "evening": ("晚峰", "每天 16:00–21:00"),
    "cap300": ("$300 cap", "全天，每小时付 max(价格 − 300, 0)"),
}
FEE_PER_MWH = 0.05      # 交易所 + 清算费的粗估（$/MWh），交易员可在私有文件里改


def hour_mask(product: str, hours: np.ndarray, weekday: np.ndarray) -> np.ndarray:
    h = np.asarray(hours)
    if product in ("base", "cap300"):
        return np.ones(len(h), bool)
    if product == "peak":
        return (np.asarray(weekday) < 5) & (h >= 7) & (h < 22)
    if product == "morning":
        return (h >= 6) & (h < 9)
    if product == "evening":
        return (h >= 16) & (h < 21)
    raise KeyError(product)


def payoff(product: str, p: np.ndarray) -> np.ndarray:
    return np.maximum(p - CAP_STRIKE, 0.0) if product == "cap300" else p


# ------------------------------------------------------------------ 数据
def load_prices(con, region: str) -> pd.Series:
    """逐小时 RRP，索引 = NEM 时间的小时起点（无时区）。"""
    df = con.execute("""
        SELECT interval_time AS t, value FROM truth
        WHERE site_id = ? AND variable = 'rrp'
        QUALIFY row_number() OVER (PARTITION BY interval_time
                                   ORDER BY (quality='final') DESC, known_at DESC) = 1
    """, [region]).df()
    s = df.set_index(pd.to_datetime(df["t"]) + NEM)["value"].sort_index()
    return s[~s.index.duplicated()]


def day_matrix(s: pd.Series) -> pd.DataFrame:
    """完整交易日 × 24 小时。缺小时的日子不进抽样池。"""
    d = s.to_frame("p")
    d["day"] = d.index.normalize()
    d["h"] = d.index.hour
    m = d.pivot_table(index="day", columns="h", values="p", aggfunc="mean")
    m = m.reindex(columns=range(24))
    return m


# ------------------------------------------------------------------ 合约期
def contract_periods(asof_day: pd.Timestamp) -> list[dict]:
    """当月、下月、当季、下季。asof_day = 已结算到的最后一个完整交易日。"""
    d = pd.Timestamp(asof_day).normalize() + pd.Timedelta(days=1)   # 第一个还没结算的日子
    out = []
    m0 = d.to_period("M")
    for k, tag in ((0, "当月"), (1, "下月")):
        p = m0 + k
        out.append({"kind": "month", "tag": tag, "label": p.strftime("%b %y"),
                    "start": p.start_time.normalize(), "end": p.end_time.normalize()})
    q0 = d.to_period("Q")
    for k, tag in ((0, "当季"), (1, "下季")):
        p = q0 + k
        out.append({"kind": "quarter", "tag": tag, "label": f"Q{p.quarter}{str(p.year)[2:]}",
                    "start": p.start_time.normalize(), "end": p.end_time.normalize()})
    return out


# ------------------------------------------------------------------ 公允价值
def _sim_days(pool: np.ndarray, n_days: int, n_sims: int, rng) -> np.ndarray:
    """移动块自助：从池子里整块（BLOCK 天）抽，拼出 n_days 天。返回 (n_sims, n_days, 24)。"""
    n = len(pool)
    if n_days == 0:
        return np.zeros((n_sims, 0, 24))
    b = min(BLOCK, n)
    nb = int(np.ceil(n_days / b))
    starts = rng.integers(0, n - b + 1, size=(n_sims, nb))
    idx = (starts[:, :, None] + np.arange(b)[None, None, :]).reshape(n_sims, -1)[:, :n_days]
    return pool[idx]


def fair_value(mat: pd.DataFrame, period: dict, asof_day: pd.Timestamp, product: str,
               pool_days: int = POOL_DAYS, n_sims: int = N_SIMS, seed: int = 7) -> dict | None:
    """合约期结算价的分布。已结算的日子用实际价；之后的日子从决策日前 pool_days 天抽样。"""
    asof_day = pd.Timestamp(asof_day).normalize()
    start, end = period["start"], period["end"]
    days = pd.date_range(start, end, freq="D")
    if len(days) == 0:
        return None
    known_days = days[days <= asof_day]
    rest_days = days[days > asof_day]
    pool = mat.loc[(mat.index > asof_day - pd.Timedelta(days=pool_days)) & (mat.index <= asof_day)]
    pool = pool.dropna()
    if len(pool) < 30:
        return None
    hrs = np.arange(24)
    # 已结算部分：产品小时上的支付之和、小时数
    k_sum, k_n = 0.0, 0
    for d in known_days:
        if d not in mat.index:
            continue
        row = mat.loc[d].to_numpy(float)
        msk = hour_mask(product, hrs, np.full(24, d.weekday())) & np.isfinite(row)
        k_sum += float(payoff(product, row[msk]).sum())
        k_n += int(msk.sum())
    # 未结算部分
    rng = np.random.default_rng(seed)
    sims = _sim_days(pool.to_numpy(float), len(rest_days), n_sims, rng)
    if len(rest_days):
        wd = np.array([d.weekday() for d in rest_days])
        M = np.stack([hour_mask(product, hrs, np.full(24, w)) for w in wd])   # (n_days, 24)
        pay = payoff(product, sims) * M[None, :, :]
        r_sum = pay.sum(axis=(1, 2))
        r_n = int(M.sum())
    else:
        r_sum, r_n = np.zeros(n_sims), 0
    tot_n = k_n + r_n
    if tot_n == 0:
        return None
    v = (k_sum + r_sum) / tot_n
    q = np.percentile(v, [5, 25, 50, 75, 95])
    # 单日风险：抽样池里单日产品均价的分布（持有到交割，每天一天的现货结算进来）
    pm = np.stack([hour_mask(product, hrs, np.full(24, d.weekday())) for d in pool.index])
    pv = pool.to_numpy(float)
    dn = pm.sum(1)
    dmean = np.where(dn > 0, (payoff(product, pv) * pm).sum(1) / np.maximum(dn, 1), np.nan)
    ok = dn > 0
    return {
        "mean": round(float(v.mean()), 2), "p5": round(float(q[0]), 2), "p25": round(float(q[1]), 2),
        "p50": round(float(q[2]), 2), "p75": round(float(q[3]), 2), "p95": round(float(q[4]), 2),
        "settled_share": round(k_n / tot_n, 3), "settled_avg": round(k_sum / k_n, 2) if k_n else None,
        "hours_total": tot_n, "hours_left": r_n, "days_left": int(len(rest_days)),
        "day_hours": round(float(dn[ok].mean()), 2) if ok.any() else 0.0,
        "day_p05": round(float(np.nanpercentile(dmean[ok], 5)), 2) if ok.any() else None,
        "day_p95": round(float(np.nanpercentile(dmean[ok], 95)), 2) if ok.any() else None,
        "_samples": v,
    }


def daily_risk_per_mw(fv: dict, price: float, side: int) -> float:
    """持有到交割、95% 情况下单日最多亏多少（$/MW）。side=+1 做多，−1 做空。"""
    if fv["days_left"] == 0:
        return 0.0
    worst = fv["day_p05"] if side > 0 else fv["day_p95"]
    return max(0.0, side * (price - worst)) * fv["day_hours"]


# ------------------------------------------------------------------ 校准
def calibrate(mat: pd.DataFrame, product: str, kind: str, offsets=(0, 7, 14, 21, 28, 45, 60, 75),
              n_sims: int = 1500) -> dict:
    """在过去每个完整的合约期上，从开始前一天起每隔若干天站一次位置，看区间准不准。"""
    first, last = mat.dropna().index.min(), mat.dropna().index.max()
    per = pd.period_range(first + pd.Timedelta(days=POOL_DAYS + 1), last, freq="M" if kind == "month" else "Q")
    rows = []
    for p in per:
        st, en = p.start_time.normalize(), p.end_time.normalize()
        if en > last:
            continue
        span = (en - st).days + 1
        act = fair_value(mat, {"start": st, "end": en}, en, product, n_sims=10)
        if act is None:
            continue
        actual = act["mean"]
        for off in offsets:
            asof = st - pd.Timedelta(days=1) + pd.Timedelta(days=off)
            if asof >= en - pd.Timedelta(days=1):
                continue
            fv = fair_value(mat, {"start": st, "end": en}, asof, product, n_sims=n_sims, seed=11 + off)
            if fv is None:
                continue
            v = fv["_samples"]
            rows.append({"period": str(p), "asof": str(asof.date()), "left": fv["days_left"] / span,
                         "actual": actual, "mean": fv["mean"],
                         "p5": fv["p5"], "p25": fv["p25"], "p75": fv["p75"], "p95": fv["p95"],
                         "in50": bool(fv["p25"] <= actual <= fv["p75"]),
                         "in90": bool(fv["p5"] <= actual <= fv["p95"]),
                         "pit": float((v < actual).mean()),
                         "naive": fv["settled_avg"]})
    if not rows:
        return {"n": 0}
    df = pd.DataFrame(rows)
    df["bucket"] = pd.cut(df["left"], [-0.01, 0.25, 0.5, 0.75, 1.01],
                          labels=["剩 ≤25%", "剩 25–50%", "剩 50–75%", "剩 75–100%"])
    by = []
    for b, g in df.groupby("bucket", observed=True):
        nv = g.dropna(subset=["naive"])
        by.append({"bucket": str(b), "n": int(len(g)),
                   "cov50": round(float(g["in50"].mean()), 3), "cov90": round(float(g["in90"].mean()), 3),
                   "mae": round(float((g["mean"] - g["actual"]).abs().mean()), 2),
                   "mae_naive": round(float((nv["naive"] - nv["actual"]).abs().mean()), 2) if len(nv) else None})
    k50, k90 = widen_factors(df)
    oos = widened_oos(df)
    return {"n": int(len(df)), "periods": int(df["period"].nunique()),
            "cov50": round(float(df["in50"].mean()), 3), "cov90": round(float(df["in90"].mean()), 3),
            "k50": k50, "k90": k90, **oos, "by_left": by}


def _z(df: pd.DataFrame, lo: str, hi: str) -> np.ndarray:
    """实际值相对点估计的偏离，以对应区间的半边宽为单位（偏上用上半边，偏下用下半边）。"""
    up = (df["actual"] - df["mean"]) / (df[hi] - df["mean"]).clip(lower=1e-6)
    dn = (df["mean"] - df["actual"]) / (df["mean"] - df[lo]).clip(lower=1e-6)
    return np.where(df["actual"] >= df["mean"], up, dn).astype(float)


def widen_factors(df: pd.DataFrame) -> tuple[float, float]:
    """抽样区间历史上偏窄时放宽多少倍，才能让 50% / 90% 区间真的盖住 50% / 90%。不收窄（下限 1）。"""
    if len(df) < 8:
        return 1.0, 1.0
    k50 = float(np.quantile(_z(df, "p25", "p75"), 0.5))
    k90 = float(np.quantile(_z(df, "p5", "p95"), 0.9))
    return round(max(1.0, k50), 2), round(max(1.0, k90), 2)


def widened_oos(df: pd.DataFrame, min_rows: int = 8) -> dict:
    """放宽系数只用**更早的合约期**估，再到后面的合约期上检验 —— 样本外覆盖率。"""
    df = df.sort_values(["period", "asof"]).reset_index(drop=True)
    hit50, hit90 = [], []
    for per in df["period"].unique():
        past = df[df["period"] < per]
        if len(past) < min_rows:
            continue
        k50, k90 = widen_factors(past)
        cur = df[df["period"] == per]
        for _, r in cur.iterrows():
            lo50 = r["mean"] + k50 * (r["p25"] - r["mean"]); hi50 = r["mean"] + k50 * (r["p75"] - r["mean"])
            lo90 = r["mean"] + k90 * (r["p5"] - r["mean"]); hi90 = r["mean"] + k90 * (r["p95"] - r["mean"])
            hit50.append(lo50 <= r["actual"] <= hi50)
            hit90.append(lo90 <= r["actual"] <= hi90)
    if not hit90:
        return {"oos_n": 0}
    return {"oos_n": len(hit90), "oos_cov50": round(float(np.mean(hit50)), 3),
            "oos_cov90": round(float(np.mean(hit90)), 3)}


def apply_widen(fv: dict, k50: float, k90: float, floor: float | None = None) -> dict:
    """把放宽系数用到一个公允价值上：分位数相对均值按比例拉开；抽样样本按 90% 的系数拉开（算胜率用）。"""
    m = fv["mean"]
    out = dict(fv)
    for q, k in (("p5", k90), ("p95", k90), ("p25", k50), ("p75", k50)):
        v = m + k * (fv[q] - m)
        out[q] = round(v if floor is None else max(floor, v), 2)
    if "_samples" in fv:
        smp = m + k90 * (fv["_samples"] - m)
        out["_samples"] = smp if floor is None else np.maximum(floor, smp)
    out["widen"] = [k50, k90]
    return out


# ------------------------------------------------------------------ 天气指标
def weather_outlook(con, site_id: str, region_prices: pd.Series, days_ahead: int = 7) -> dict | None:
    """未来几天风速、气温相对同期常态的偏离，以及历史上这种偏离对应的日均价差（描述性，不是预测）。"""
    f = con.execute("""
        SELECT interval_time AS t, variable, value, known_at FROM forecast
        WHERE site_id = ? AND model = 'ecmwf_ifs025' AND variable IN ('wind100','temp')
        QUALIFY row_number() OVER (PARTITION BY variable, interval_time ORDER BY known_at DESC) = 1
    """, [site_id]).df()
    tr = con.execute("""
        SELECT interval_time AS t, variable, value FROM truth
        WHERE site_id = ? AND variable IN ('wind100','temp')
        QUALIFY row_number() OVER (PARTITION BY variable, interval_time
                                   ORDER BY (quality='final') DESC, known_at DESC) = 1
    """, [site_id]).df()
    if f.empty or tr.empty:
        return None
    now = pd.Timestamp.now("UTC").tz_localize(None)
    f = f[(pd.to_datetime(f["t"]) >= now) & (pd.to_datetime(f["t"]) < now + pd.Timedelta(days=days_ahead))]
    if f.empty:
        return None
    fw = f.pivot_table(index="t", columns="variable", values="value")
    tw = tr.pivot_table(index="t", columns="variable", values="value")
    tw.index = pd.to_datetime(tw.index) + NEM
    fw.index = pd.to_datetime(fw.index) + NEM
    td = tw.resample("D").mean().dropna()
    td["doy"] = td.index.dayofyear
    out = {"days": int(fw.index.normalize().nunique()),
           "from": str(fw.index.min().date()), "to": str(fw.index.max().date())}
    fd = fw.resample("D").mean().dropna()
    clim = {}
    for v in ("wind100", "temp"):
        doys = fd.index.dayofyear
        ref = []
        for dy in doys:
            w = td[(np.abs(td["doy"] - dy) <= 15) | (np.abs(td["doy"] - dy) >= 350)][v]
            ref.append((w.mean(), w.std()))
        mu = np.array([r[0] for r in ref]); sd = np.array([r[1] for r in ref])
        clim[v] = (float(fd[v].mean()), float(mu.mean()), float(np.mean((fd[v].to_numpy() - mu) / sd)))
    out["wind"] = {"fc": round(clim["wind100"][0], 2), "norm": round(clim["wind100"][1], 2),
                   "z": round(clim["wind100"][2], 2)}
    out["temp"] = {"fc": round(clim["temp"][0], 1), "norm": round(clim["temp"][1], 1),
                   "z": round(clim["temp"][2], 2)}
    # 历史描述：日均价对风速偏离的回归（同月份内比较），只作参考
    dp = region_prices.resample("D").mean().rename("p").to_frame().join(td, how="inner").dropna()
    if len(dp) > 120:
        dp["m"] = dp.index.month
        for v in ("wind100", "temp"):
            dp[v + "_a"] = dp[v] - dp.groupby("m")[v].transform("mean")
        dp["p_a"] = dp["p"] - dp.groupby("m")["p"].transform("median")
        X = np.column_stack([dp["wind100_a"], np.abs(dp["temp_a"]), np.ones(len(dp))])
        beta = np.linalg.lstsq(X, dp["p_a"].to_numpy(), rcond=None)[0]
        out["sens_wind"] = round(float(beta[0]), 2)       # $/MWh 每 1 km/h（Open-Meteo 默认单位）
        out["sens_temp"] = round(float(beta[1]), 2)       # $/MWh 每偏离 1 °C
        out["implied"] = round(float(beta[0] * (out["wind"]["fc"] - out["wind"]["norm"])
                                     + beta[1] * abs(out["temp"]["fc"] - out["temp"]["norm"])), 2)
        out["sens_days"] = int(len(dp))
    return out


# ------------------------------------------------------------------ 明日现货展望
def spot_outlook(con, region: str, mat: pd.DataFrame) -> dict | None:
    """明天：AEMO **最新一版**预调度的均价/最高价、纸面交易里我们的预测、历史同月尖峰频率。
    预调度没覆盖到明天（每日任务还没跑）就标 stale，页面上照实说。"""
    today = (pd.Timestamp.now("UTC").tz_localize(None) + NEM).normalize()
    nxt = today + pd.Timedelta(days=1)
    lo, hi = nxt - NEM, nxt + pd.Timedelta(days=1) - NEM
    df = con.execute("""
        SELECT interval_time AS t, value, known_at FROM forecast
        WHERE site_id = ? AND variable = 'rrp' AND model = 'aemo_predispatch'
          AND interval_time >= ? AND interval_time < ?
        QUALIFY row_number() OVER (PARTITION BY interval_time ORDER BY known_at DESC) = 1
    """, [region, lo, hi]).df()
    out = {"day": str(nxt.date())}
    hist = mat.dropna()
    hist = hist[hist.index.month == nxt.month]
    if len(hist):
        out["spike_freq"] = round(float((hist.max(axis=1) > CAP_STRIKE).mean()), 3)
        out["spike_days"] = int(len(hist))
    if len(df) < 20:
        last = con.execute("""SELECT max(interval_time) FROM forecast
                              WHERE site_id = ? AND model = 'aemo_predispatch'""", [region]).fetchone()[0]
        out["stale"] = True
        out["pd_last"] = None if last is None else str((pd.Timestamp(last) + NEM))[:16]
        return out
    out.update({"aemo_avg": round(float(df["value"].mean()), 1), "aemo_max": round(float(df["value"].max()), 1),
                "aemo_hours_over_300": int((df["value"] > CAP_STRIKE).sum()),
                "pd_known_at": str(pd.to_datetime(df["known_at"]).max() + NEM)[:16]})
    sig = os.path.join(ROOT, "docs", "paper", "signals.csv")
    if os.path.exists(sig):
        sg = pd.read_csv(sig)
        sg = sg[(sg["region"] == region) & (sg["target_day"] == out["day"]) & (sg["status"] == "ok")]
        if len(sg) >= 20:
            out["ours_avg"] = round(float(pd.to_numeric(sg["p_b2"]).mean()), 1)
            out["ours_max"] = round(float(pd.to_numeric(sg["p_b2"]).max()), 1)
    return out


# ------------------------------------------------------------------ 汇总
def build(con, calib: bool = True) -> dict:
    from config import SITES
    res = {"generated_at": pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M"),
           "pool_days": POOL_DAYS, "block": BLOCK, "n_sims": N_SIMS,
           "products": {k: {"name": v[0], "desc": v[1]} for k, v in PRODUCTS.items()},
           "regions": {}}
    for region in REGIONS:
        s = load_prices(con, region)
        mat = day_matrix(s)
        full = mat.dropna()
        if len(full) < 60:
            continue
        asof = full.index.max()
        reg = {"asof": str(asof.date()), "last_hour": str(s.index.max()), "contracts": []}
        cal = {}
        if calib:
            cal = {f"{k}_{p}": calibrate(mat, p, k) for k in ("month", "quarter") for p in ("base", "peak")}
            reg["calibration"] = cal
        for per in contract_periods(asof):
            row = {"tag": per["tag"], "label": per["label"], "kind": per["kind"],
                   "start": str(per["start"].date()), "end": str(per["end"].date()), "products": {}}
            for prod in PRODUCTS:
                fv = fair_value(mat, per, asof, prod)
                if fv is None:
                    continue
                c_ = cal.get(f"{per['kind']}_{prod if prod in ('base', 'peak') else 'peak'}") or {}
                if c_.get("k90"):
                    fv = apply_widen(fv, c_["k50"], c_["k90"], floor=0.0 if prod == "cap300" else None)
                fv.pop("_samples", None)
                fv["risk_long"] = round(daily_risk_per_mw(fv, fv["mean"], +1), 1)
                fv["risk_short"] = round(daily_risk_per_mw(fv, fv["mean"], -1), 1)
                row["products"][prod] = fv
            reg["contracts"].append(row)
        site = next((x[0] for x in SITES if x[1] == region), None)
        try:
            reg["weather"] = weather_outlook(con, site, s) if site else None
        except Exception as e:                                       # noqa: BLE001
            reg["weather"] = {"error": str(e)[:120]}
        try:
            reg["spot"] = spot_outlook(con, region, mat)
        except Exception as e:                                       # noqa: BLE001
            reg["spot"] = {"error": str(e)[:120]}
        res["regions"][region] = reg
        print(f"  {region}: 截至 {asof.date()}，" + "；".join(
            f"{c['label']} 基荷 {c['products']['base']['mean']:.1f} [{c['products']['base']['p5']:.0f}, {c['products']['base']['p95']:.0f}]"
            for c in reg["contracts"] if "base" in c["products"]))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="期货公允价值（只用 AEMO 现货价）")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--no-calib", action="store_true", help="跳过历史校准（快）")
    a = ap.parse_args(argv)
    import store
    con = store.connect()
    res = build(con, calib=not a.no_calib)
    con.close()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    print(f"→ {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

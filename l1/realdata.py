"""
L1 → 评估管道 的适配层。
========================
把时点存储里的行，装配成 engine.py 期望的 (预报, 真值) 数组。
engine.py 的评估函数一行不改——它们只认数组。

三条必须守住的规矩：
  ① 回测取**固定预报时效**（默认 D+1），不是"最新可见的那版"。
     模型之间必须在同一时效上比，否则排行榜比的是时效不是模型。
  ② DM 检验和 MCS 要求各模型样本完全相同，所以只保留**所有模型都有值**的时刻。
  ③ 真值优先 final（ERA5）；只有 provisional 的时刻单独标出来，默认排除出评估。
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import store
from config import NEM_OFFSET, SITES, SITE_BY_ID, STRICT_METHODS, VARIABLES

VARS = list(VARIABLES.keys())

# ------------------------------------------------------------------ 真实模型清单
# 只列真实存在的东西。LightGBM / PatchTST 在训练之前不出现在这里。
NWP = ["ecmwf_ifs025", "gfs_global", "icon_global", "gem_global"]
MODEL_META = {
    "persist":      ("Persistence (D-1 同时刻)", "朴素基准", False),
    "clim":         ("气候态 Climatology",       "朴素基准", False),
    "gfs_global":   ("GFS 原始输出",             "NWP",      False),
    "icon_global":  ("ICON 原始输出",            "NWP",      False),
    "gem_global":   ("GEM 原始输出",             "NWP",      False),
    "ecmwf_ifs025": ("ECMWF-IFS 原始输出",       "NWP",      False),
    "mm_mean":      ("多模式等权平均",           "集成",     False),
    "qra":          ("多模式中位数 + 在线共形",  "集成/校准", True),
}
BASELINE = "ecmwf_ifs025"          # 技能分的分母，骨架要求用官方预报而非 persistence


# ------------------------------------------------------------------ 太阳几何（与 engine.py 同一套公式）
def cos_zenith(lat, lon, doy, clock_hour_nem):
    doy = np.asarray(doy, dtype=float)
    ch = np.asarray(clock_hour_nem, dtype=float)
    B = 2 * np.pi * (doy - 81) / 364.0
    eot = 9.87 * np.sin(2 * B) - 7.53 * np.cos(B) - 1.5 * np.sin(B)
    solar_time = ch + (lon - 150.0) / 15.0 + eot / 60.0
    H = np.radians(15.0 * (solar_time - 12.0))
    dec = np.radians(23.44 * np.sin(2 * np.pi * (284 + doy) / 365.0))
    la = math.radians(lat)
    return np.sin(la) * np.sin(dec) + np.cos(la) * np.cos(dec) * np.cos(H)


def clearsky_ghi(cosz):
    cz = np.clip(np.asarray(cosz, dtype=float), 0.0, None)
    out = np.zeros_like(cz)
    m = cz > 0.01
    out[m] = 1098.0 * cz[m] * np.exp(-0.059 / cz[m])
    return out


def wind_cf(v):
    v = np.asarray(v, dtype=float)
    cf = np.where(v < 3.0, 0.0, np.where(v >= 12.0, 1.0, (v ** 3 - 27.0) / (1728.0 - 27.0)))
    return np.where(v >= 25.0, 0.0, cf)


def solar_cf(g, t):
    g = np.asarray(g, dtype=float); t = np.asarray(t, dtype=float)
    return np.clip((g / 1000.0) * (1 - 0.004 * (t + 0.028 * g - 25.0)), 0.0, None)


# ------------------------------------------------------------------ 装配结果
@dataclass
class Panel:
    site_id: str
    region: str
    name: str
    lat: float
    lon: float
    solar_w: float
    wind_w: float
    vre_share: float
    t: np.ndarray                      # interval_time (UTC, datetime64)
    hod: np.ndarray                    # NEM 时区的小时
    doy: np.ndarray
    cosz: np.ndarray
    clearsky: np.ndarray
    truth: dict = field(default_factory=dict)          # var -> array
    preds: dict = field(default_factory=dict)          # model -> {var -> array}
    lead_h: dict = field(default_factory=dict)         # model -> array
    cov: dict = field(default_factory=dict)            # 覆盖率诊断

    def __len__(self):
        return len(self.t)

    def kt(self):
        """晴空指数 kt = GHI / GHI_clearsky。骨架里 GHI 的归一化口径。"""
        cs = np.maximum(self.clearsky, 1e-6)
        out = np.where(self.clearsky > 20.0, self.truth["ghi"] / cs, np.nan)
        return out

    def vre(self, ghi, wind, temp):
        return self.solar_w * solar_cf(ghi, temp) + self.wind_w * wind_cf(wind)


# ------------------------------------------------------------------ 决策时刻
# 交易口径：NEM 日历日 D 的全部 24 小时，在 D−1 天的固定时刻（NEM 时间）一次性定计划。
# 这是电池运营商排次日计划的真实做法：AEMO 大约 12:30 之后才发布覆盖下一天的
# 预调度，在那之后一次性排好明天。也和 EPF 综述的 day-ahead 设定一致——
# 同一天的 24 个小时共用一个信息集，时效随小时变（约 10–35 小时）。
#
# 之前用的是"滚动提前 24 小时"：每个小时各自往前推 24 小时。那样对 04:00–12:30
# 的目标小时根本拿不到预调度（AEMO 那时还没发布覆盖它们的版本），
# 加了官方预测这一层后这半天会被整段丢掉——恰好是午间光伏低谷的前半段。
DECIDE_AT_NEM = "13:30"


def _clock_minutes(clock: str) -> int:
    hh, mm = clock.split(":")
    m = int(hh) * 60 + int(mm)
    if not 0 <= m < 24 * 60:
        raise ValueError(f"决策时刻要是 HH:MM（NEM 时间），收到 {clock!r}")
    return m


def decision_time(t_utc, clock: str = DECIDE_AT_NEM):
    """目标小时（UTC）→ 它的决策时刻（UTC）。

    目标落在 NEM 日历日 D，决策时刻就是 D−1 天的 clock（NEM），再换回 UTC。
    """
    t = pd.to_datetime(t_utc)
    off = pd.Timedelta(hours=NEM_OFFSET.total_seconds() / 3600)
    nem_day = (t + off).normalize() if isinstance(t, pd.Timestamp) \
        else (pd.Series(t) + off).dt.normalize()
    out = nem_day - pd.Timedelta(days=1) + pd.Timedelta(minutes=_clock_minutes(clock)) - off
    return out if isinstance(t, pd.Timestamp) else out.to_numpy()


def decision_sql(col: str, clock: str = DECIDE_AT_NEM) -> str:
    """decision_time 的 SQL 版本（DuckDB），两者必须一致——有测试盯着。"""
    off_h = int(NEM_OFFSET.total_seconds() // 3600)
    return (f"(date_trunc('day', {col} + INTERVAL {off_h} HOUR) - INTERVAL 1 DAY "
            f"+ INTERVAL {_clock_minutes(clock)} MINUTE - INTERVAL {off_h} HOUR)")


# ------------------------------------------------------------------ 读取
_WARNED: set = set()


def _warn_once(key) -> bool:
    """同一个站点会被逐时效反复装配，同样的告警刷十几遍只会淹没真正的问题。"""
    if key in _WARNED:
        return False
    _WARNED.add(key)
    return True


def _time_cols(t_utc, lat, lon):
    t = pd.to_datetime(pd.Series(t_utc))
    nem = t + NEM_OFFSET
    hod = nem.dt.hour.to_numpy()
    doy = nem.dt.dayofyear.to_numpy()
    cz = np.clip(cos_zenith(lat, lon, doy, hod + 0.5), 0.0, None)
    return hod, doy, cz, clearsky_ghi(cz)


def load_backtest(con, site_id: str, start=None, end=None, lead_day: int = 1,
                  lead_tol_h: int = 12, strict: bool = False,
                  final_only: bool = True, models=None,
                  min_lead_h: float | None = None,
                  decide_at: str | None = None,
                  require_truth: bool = True) -> Panel:
    """按固定预报时效装配回测面板。

    lead_day=1 表示取"目标时刻前约 24h 发出"的那一版，容差 ±lead_tol_h。
    strict=False 时允许 previous-runs 回填的 approx 行——它的 known_at 是**保守上界**，
    用于 as-of 过滤是安全的，只是时效归属不精确，所以这里靠 lead 分桶而非 known_at 精度。

    min_lead_h：**单侧**时效约束，给有"决策时刻"的下游用（ladder.py）。
    设了它就只取 known_at ≤ interval_time − min_lead_h 的版本，并在其中取**最新**的一版。
    默认的 ±lead_tol_h 双侧窗口会选到"离 24h 最近"的那版——可能是 21h 前发出的，
    比决策时刻还新，对决策回测来说就是前视。时效按秒精确算，不用 date_diff('hour')
    （后者数的是跨过的整点个数，12:01 → 次日 12:00 会算成 24 小时）。

    decide_at（"HH:MM"，NEM）：**固定时刻**口径，优先于 lead_day / min_lead_h。
    目标落在 NEM 日 D，只取 known_at ≤ D−1 天 decide_at 的版本，并取其中最新的一版。
    """
    models = models or NWP
    lead_c = lead_day * 24
    lo_h, hi_h = lead_c - lead_tol_h, lead_c + lead_tol_h
    meth = list(STRICT_METHODS) if strict else ["exact", "request", "approx"]
    s = SITE_BY_ID[site_id]

    where = ["site_id = ?", f"model IN ({', '.join(['?'] * len(models))})",
             f"known_at_method IN ({', '.join(['?'] * len(meth))})"]
    params: list = [site_id, *models, *meth]
    if start is not None:
        where.append("interval_time >= ?"); params.append(pd.Timestamp(start))
    if end is not None:
        where.append("interval_time <= ?"); params.append(pd.Timestamp(end))

    if decide_at is not None:
        tau = decision_sql("interval_time", decide_at)
        f = con.execute(f"""
            WITH f AS (
                SELECT site_id, model, variable, interval_time, value, known_at,
                       {tau} AS tau
                FROM forecast WHERE {' AND '.join(where)}
            )
            SELECT site_id, model, variable, interval_time, value,
                   date_diff('second', known_at, interval_time) / 3600.0 AS lead_h
            FROM f
            WHERE known_at <= tau AND known_at >= tau - INTERVAL 4 DAY
            QUALIFY row_number() OVER (
                PARTITION BY model, variable, interval_time
                ORDER BY known_at DESC
            ) = 1
        """, params).df()
    elif min_lead_h is None:
        f = con.execute(f"""
            WITH f AS (
                SELECT site_id, model, variable, interval_time, value,
                       date_diff('hour', known_at, interval_time) AS lead_h
                FROM forecast WHERE {' AND '.join(where)}
            )
            SELECT * FROM f
            WHERE lead_h BETWEEN {lo_h} AND {hi_h}
            QUALIFY row_number() OVER (
                PARTITION BY model, variable, interval_time
                ORDER BY abs(lead_h - {lead_c}), lead_h
            ) = 1
        """, params).df()
    else:
        # 时效按**整数秒**算、整数秒排序，最后才换成小时。
        # 用浮点小时做 QUALIFY 排序 + 区间过滤，会触发 DuckDB 1.5.x 的优化器内部错误
        # （"Failed to cast expression to type"），在 Mac 上实测复现过。
        lo_s = int(round(float(min_lead_h) * 3600))
        hi_s = int(round((float(min_lead_h) + 2 * lead_tol_h) * 3600))
        f = con.execute(f"""
            WITH f AS (
                SELECT site_id, model, variable, interval_time, value,
                       date_diff('second', known_at, interval_time) AS lead_s
                FROM forecast WHERE {' AND '.join(where)}
            )
            SELECT site_id, model, variable, interval_time, value,
                   lead_s / 3600.0 AS lead_h
            FROM f
            WHERE lead_s >= {lo_s} AND lead_s <= {hi_s}
            QUALIFY row_number() OVER (
                PARTITION BY model, variable, interval_time
                ORDER BY lead_s
            ) = 1
        """, params).df()

    tw = ["site_id = ?"]
    tp: list = [site_id]
    if final_only:
        tw.append("quality = 'final'")
    if start is not None:
        tw.append("interval_time >= ?"); tp.append(pd.Timestamp(start))
    if end is not None:
        tw.append("interval_time <= ?"); tp.append(pd.Timestamp(end))
    t = con.execute(f"""
        SELECT variable, interval_time, value, quality FROM truth
        WHERE {' AND '.join(tw)}
        QUALIFY row_number() OVER (PARTITION BY variable, interval_time
                                   ORDER BY (quality='final') DESC, known_at DESC) = 1
    """, tp).df()

    if f.empty or t.empty:
        raise ValueError(f"{site_id}: 预报 {len(f)} 行 / 真值 {len(t)} 行，装配不出面板。"
                         f"先跑 ingest.py --backfill-* 或放宽 lead_day / final_only。")

    tw_ = t.pivot_table(index="interval_time", columns="variable", values="value", aggfunc="last")
    fw = f.pivot_table(index="interval_time", columns=["model", "variable"],
                       values="value", aggfunc="last")
    lw = f.pivot_table(index="interval_time", columns="model", values="lead_h", aggfunc="last")

    need_t = [v for v in VARS if v in tw_.columns]
    if len(need_t) < len(VARS):
        raise ValueError(f"{site_id}: 真值缺变量 {set(VARS) - set(need_t)}")
    # 名册自适应：只保留**所有变量都齐**的模型。硬编码 4 个 NWP 会让整段回测
    # 因为某一家临时缺一个变量而整体失败——但 DM / MCS 要求的是"同样本对比"，
    # 不是"必须四家都在"。所以这里降级而不是报错，并把真正用到的名册写进 cov。
    have = [m for m in models if all((m, v) in fw.columns for v in VARS)]
    dropped = [m for m in models if m not in have]
    if dropped and _warn_once(("drop", site_id, tuple(dropped))):
        print(f"  [warn] {site_id}: 预报不全，剔除模型 {dropped}；实际使用 {have}")
    if not have:
        avail = sorted({c[0] for c in fw.columns})
        raise ValueError(f"{site_id}: 没有任何模型的三个变量是齐的。"
                         f"存储里出现过的模型：{avail}。先跑 ingest.py 回填。")
    if len(have) == 1 and _warn_once(("solo", site_id, have[0])):
        # 单模型也要能跑。缺口↔电价这类分析只需要一条净负荷预报，
        # 需要 ≥2 的是集成、DM 检验和 MCS——那是 L3/L4 的事，不该卡住 L6。
        # 此时 mm_mean / qra 退化成该模型本身，模型对比无意义，cov 里标出来。
        print(f"  [warn] {site_id}: 只有 {have[0]} 一家。集成/DM/MCS 无意义，"
              f"但缺口与价格分析照常可跑。")
    models = have
    need_f = [(m, v) for m in models for v in VARS]

    n_t, n_f = len(tw_), len(fw)
    # ② 只保留所有模型 + 真值都齐的时刻，DM / MCS 才是同一样本
    if require_truth:
        idx = tw_.dropna(subset=need_t).index.intersection(fw.dropna(subset=need_f).index)
    else:
        # 实时出信号用：目标小时还没有真值，只要预报齐就保留（真值留 NaN）
        idx = fw.dropna(subset=need_f).index
    idx = idx.sort_values()
    if len(idx) < 24 * 30:
        raise ValueError(f"{site_id}: 完整重合样本只有 {len(idx)} 小时（<30 天），不足以回测。"
                         f"真值 {n_t} / 预报 {n_f}。")

    tw_, fw, lw = tw_.reindex(idx), fw.loc[idx], lw.loc[idx]
    t_utc = idx.to_numpy()
    hod, doy, cz, cs = _time_cols(t_utc, s[3], s[4])

    p = Panel(site_id=site_id, region=s[1], name=s[2], lat=s[3], lon=s[4],
              solar_w=s[5], wind_w=s[6], vre_share=s[7],
              t=t_utc, hod=hod, doy=doy, cosz=cz, clearsky=cs)
    p.truth = {v: tw_[v].to_numpy(dtype=float) for v in VARS}
    for m in models:
        p.preds[m] = {v: fw[(m, v)].to_numpy(dtype=float) for v in VARS}
        p.lead_h[m] = lw[m].to_numpy(dtype=float)

    # ---- 朴素基准（由真值直接构造，不占 API 调用）----
    p.preds["persist"] = {v: _shift24(p.truth[v], t_utc) for v in VARS}
    p.preds["clim"] = {v: _climatology(p.truth[v], doy, hod) for v in VARS}
    # ---- 多模式集成 ----
    p.preds["mm_mean"] = {v: np.mean([p.preds[m][v] for m in models], axis=0) for v in VARS}
    p.preds["qra"] = {v: np.median([p.preds[m][v] for m in models], axis=0) for v in VARS}
    for k in ("persist", "clim", "mm_mean", "qra"):
        p.lead_h[k] = np.full(len(idx), float(lead_c))
    p.preds["qra"]["ghi"] = np.clip(p.preds["qra"]["ghi"], 0, None)
    p.preds["mm_mean"]["ghi"] = np.clip(p.preds["mm_mean"]["ghi"], 0, None)

    p.cov = {"hours": int(len(idx)), "truth_rows": int(n_t), "forecast_rows": int(n_f),
             "kept_share": round(len(idx) / max(n_t, 1) * 100, 1),
             "lead_day": lead_day, "lead_h_target": int(lead_c),
             "models_used": list(models),
             "models_dropped": dropped, "single_model": len(models) == 1,
             "lead_h_median": {m: float(np.median(p.lead_h[m])) for m in models},
             "span": [str(pd.Timestamp(t_utc[0])), str(pd.Timestamp(t_utc[-1]))],
             "gaps_h": int(_gap_hours(t_utc))}
    return p


def _shift24(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """D-1 同时刻。样本可能有洞，所以按时间戳对齐而不是按下标平移。"""
    s = pd.Series(y, index=pd.to_datetime(t))
    return s.reindex(s.index - pd.Timedelta(hours=24)).to_numpy()


def _climatology(y, doy, hod) -> np.ndarray:
    """逐 (月 × 时刻) 均值。避免把年循环当成技能。"""
    mon = ((np.asarray(doy) - 1) // 30).astype(int)
    key = mon * 24 + np.asarray(hod)
    out = np.empty(len(y))
    for k in np.unique(key):
        m = key == k
        out[m] = np.nanmean(y[m])
    return out


def _gap_hours(t: np.ndarray) -> int:
    idx = pd.DatetimeIndex(t)
    if len(idx) < 2:
        return 0
    d = idx.to_series().diff().dt.total_seconds().div(3600.0).dropna()
    return int((d[d > 1.0] - 1.0).sum())


def fill_persist_head(p: Panel) -> Panel:
    """persist 的前 24h 是 NaN，统一裁掉，保证所有模型样本一致。"""
    bad = np.zeros(len(p), dtype=bool)
    for m in p.preds:
        for v in VARS:
            bad |= ~np.isfinite(p.preds[m][v])
    for v in VARS:
        bad |= ~np.isfinite(p.truth[v])
    if not bad.any():
        return p
    keep = ~bad
    p.t, p.hod, p.doy = p.t[keep], p.hod[keep], p.doy[keep]
    p.cosz, p.clearsky = p.cosz[keep], p.clearsky[keep]
    p.truth = {v: a[keep] for v, a in p.truth.items()}
    p.preds = {m: {v: a[keep] for v, a in d.items()} for m, d in p.preds.items()}
    p.lead_h = {m: a[keep] for m, a in p.lead_h.items()}
    p.cov["hours"] = int(keep.sum())
    p.cov["dropped_incomplete"] = int((~keep).sum())
    return p


# ------------------------------------------------------------------ 运行窗口（-24h .. +48h）
def load_window(con, site_id: str, as_of=None, past_h: int = 24, future_h: int = 48,
                models=None) -> dict:
    """给运行台顶部那张扇形图用：观测段 + 最新一版预报 + 多模式离散度。"""
    models = models or NWP
    s = SITE_BY_ID[site_id]
    if as_of is None:
        as_of = con.execute("SELECT max(known_at) FROM forecast WHERE site_id = ?",
                            [site_id]).fetchone()[0]
        if as_of is None:
            raise ValueError(f"{site_id}: forecast 表是空的")
    as_of = pd.Timestamp(as_of).floor("h")
    t0, t1 = as_of - pd.Timedelta(hours=past_h), as_of + pd.Timedelta(hours=future_h)

    f = con.execute(f"""
        SELECT model, variable, interval_time, value,
               date_diff('hour', known_at, interval_time) AS lead_h
        FROM forecast
        WHERE site_id = ? AND known_at <= ? AND interval_time BETWEEN ? AND ?
          AND model IN ({', '.join(['?'] * len(models))})
        QUALIFY row_number() OVER (PARTITION BY model, variable, interval_time
                                   ORDER BY known_at DESC) = 1
    """, [site_id, as_of, t0, t1, *models]).df()
    t = con.execute("""
        SELECT variable, interval_time, value FROM truth
        WHERE site_id = ? AND interval_time BETWEEN ? AND ?
        QUALIFY row_number() OVER (PARTITION BY variable, interval_time
                                   ORDER BY (quality='final') DESC, known_at DESC) = 1
    """, [site_id, t0, as_of]).df()

    hours = pd.date_range(t0, t1, freq="h")
    out = {"site_id": site_id, "region": s[1], "name": s[2], "lat": s[3], "lon": s[4],
           "as_of": str(as_of), "series": {}}
    fw = (f.pivot_table(index="interval_time", columns=["model", "variable"],
                        values="value", aggfunc="last").reindex(hours)
          if not f.empty else pd.DataFrame(index=hours))
    tw = (t.pivot_table(index="interval_time", columns="variable",
                        values="value", aggfunc="last").reindex(hours)
          if not t.empty else pd.DataFrame(index=hours))

    fut = hours[hours >= as_of]
    horizon = 0
    for v in VARS:
        cols = [(m, v) for m in models if (m, v) in fw.columns]
        ens = fw.loc[fut, cols].to_numpy(dtype=float) if cols else np.empty((len(fut), 0))
        have = np.isfinite(ens).any(axis=1) if ens.shape[1] else np.zeros(len(fut), bool)
        horizon = max(horizon, int(have.sum()))
        med = np.full(len(fut), np.nan); lo = med.copy(); hi = med.copy()
        if ens.shape[1] and have.any():
            med[have] = np.nanmedian(ens[have], axis=1)
            lo[have] = np.nanmin(ens[have], axis=1)
            hi[have] = np.nanmax(ens[have], axis=1)
        n_mem = int(np.isfinite(ens).sum(axis=1).max()) if ens.shape[1] and have.any() else 0
        obs_idx = hours[hours <= as_of]
        obs = (tw[v].reindex(obs_idx).to_numpy(dtype=float)
               if v in tw.columns else np.full(len(obs_idx), np.nan))
        base = (fw.loc[fut, (BASELINE, v)].to_numpy(dtype=float)
                if (BASELINE, v) in fw.columns else med)
        gfs = (fw.loc[fut, ("gfs_global", v)].to_numpy(dtype=float)
               if ("gfs_global", v) in fw.columns else med)
        out["series"][v] = {"obs": obs, "median": med, "lo": lo, "hi": hi,
                            "n_members": n_mem, "ecmwf": base, "gfs": gfs,
                            "have": have}
    out["horizon_h"] = horizon
    out["ok"] = horizon >= future_h
    out["hours"] = hours
    out["future_hours"] = fut
    return out


# ------------------------------------------------------------------ 巡检
def coverage_report(con, lead_day: int = 1) -> pd.DataFrame:
    """迁移前先看一眼：每个站点能装配出多少小时的完整样本。"""
    rows = []
    for s in SITES:
        try:
            p = fill_persist_head(load_backtest(con, s[0], lead_day=lead_day))
            rows.append({"site_id": s[0], "region": s[1], "hours": len(p),
                         "days": round(len(p) / 24, 1), "span": " → ".join(p.cov["span"]),
                         "kept_%": p.cov["kept_share"], "gaps_h": p.cov["gaps_h"],
                         "status": "ok"})
        except Exception as e:
            rows.append({"site_id": s[0], "region": s[1], "hours": 0, "days": 0,
                         "span": "", "kept_%": 0, "gaps_h": 0,
                         "status": f"{type(e).__name__}: {str(e)[:90]}"})
    return pd.DataFrame(rows)


def inventory(con) -> pd.DataFrame:
    """存储里到底有什么：站点 × 模型 × 变量 的行数与时间跨度。

    "预报缺 gfs_global/ghi" 这类报错，八成是 ingest 某几家没跑成功。
    先看这张表，再决定补哪一家，比盲目重跑整个 backfill 省几个小时。
    """
    return con.execute("""
        SELECT site_id, model, variable, count(*) AS rows,
               min(interval_time) AS first_t, max(interval_time) AS last_t,
               count(DISTINCT known_at_method) AS methods
        FROM forecast GROUP BY 1,2,3 ORDER BY 1,2,3
    """).df()


if __name__ == "__main__":
    con = store.connect()
    try:
        inv = inventory(con)
        print("=== 预报库存（站点 × 模型 × 变量）===")
        print(inv.to_string(index=False) if len(inv) else "  (空)")
        print("\n=== 可装配面板 ===")
        print(coverage_report(con).to_string(index=False))
    finally:
        con.close()

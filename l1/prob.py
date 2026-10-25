"""
概率预测：分位数和尖峰概率。
==========================
NEM 的钱大部分在尾部赚和亏（cap 合约、晚峰、做空），只给一个均值不够用。

方法（经验残差法，不另外拟合模型）：
  · 点预测 = 价格 + AEMO 模型（B2）的样本外预测；
  · 每个测试日，取它**之前** POOL_DAYS 天里同一小时的**样本外**残差（实际 − 预测），
    加到当天的点预测上 → 这一小时的预测分布。只用已经发生过的样本外误差，不偷看；
  · 分位数 q10 / q50 / q90；尖峰概率 = P(价格 > $300)。

对照：
  · 同样的做法套在 AEMO 预调度上（AEMO 预调度 + 它自己的历史残差）；
  · 尖峰概率还对照"历史同小时尖峰频率"（气候态）和"预调度 > $300 就算 100%"。

评分：分位数用 pinball loss（越小越好）和区间覆盖率；尖峰用 Brier 分数（越小越好）和可靠性（预测 20% 的那批小时，实际是不是 20% 左右）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

POOL_DAYS = 90
MIN_DAYS = 30
QS = (0.1, 0.5, 0.9)
SPIKE = 300.0
NEM = pd.Timedelta(hours=10)


def _pinball(y, q, tau):
    d = y - q
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))


def rolling_dist(y: np.ndarray, p: np.ndarray, t, pool_days: int = POOL_DAYS) -> dict:
    """对每个测试日，用之前 pool_days 天同一小时的样本外残差给出分位数和尖峰概率。"""
    t = pd.to_datetime(pd.Series(t))
    nem = t + NEM
    day = nem.dt.floor("D").to_numpy()
    hod = nem.dt.hour.to_numpy()
    res = y - p
    udays = np.unique(day)
    n = len(y)
    qs = {q: np.full(n, np.nan) for q in QS}
    ps = np.full(n, np.nan)
    for dd in udays:
        cur = day == dd
        past = (day < dd) & (day >= dd - np.timedelta64(pool_days, "D"))
        if len(np.unique(day[past])) < MIN_DAYS:
            continue
        for h in np.unique(hod[cur]):
            r = res[past & (hod == h)]
            if len(r) < MIN_DAYS:
                continue
            idx = np.where(cur & (hod == h))[0]
            for i in idx:
                smp = p[i] + r
                for q in QS:
                    qs[q][i] = float(np.quantile(smp, q))
                ps[i] = float((smp > SPIKE).mean())
    return {"q": qs, "p_spike": ps}


def climatology_spike(y: np.ndarray, t, pool_days: int = POOL_DAYS) -> np.ndarray:
    t = pd.to_datetime(pd.Series(t))
    nem = t + NEM
    day = nem.dt.floor("D").to_numpy()
    hod = nem.dt.hour.to_numpy()
    out = np.full(len(y), np.nan)
    for dd in np.unique(day):
        cur = day == dd
        past = (day < dd) & (day >= dd - np.timedelta64(pool_days, "D"))
        for h in np.unique(hod[cur]):
            r = y[past & (hod == h)]
            if len(r) >= MIN_DAYS:
                out[cur & (hod == h)] = float((r > SPIKE).mean())
    return out


def score(y: np.ndarray, dist: dict, ok: np.ndarray) -> dict:
    q = dist["q"]
    out = {"pinball": {str(k): round(_pinball(y[ok], q[k][ok], k), 2) for k in QS},
           "cov80": round(float(((y >= q[0.1]) & (y <= q[0.9]))[ok].mean()), 3)}
    ps = dist["p_spike"][ok]
    yy = (y[ok] > SPIKE).astype(float)
    out["brier"] = round(float(np.mean((ps - yy) ** 2)), 5)
    return out


def reliability(p: np.ndarray, y: np.ndarray, bins=(0, .02, .05, .1, .2, .4, 1.01)) -> list[dict]:
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() >= 20:
            out.append({"bin": f"{lo:.0%}–{min(hi, 1):.0%}", "n": int(m.sum()),
                        "pred": round(float(p[m].mean()), 3), "obs": round(float(y[m].mean()), 3)})
    return out


def probabilistic(res: dict) -> dict:
    if res.get("status") != "ok" or "_pred" not in res or "B2 +官方" not in res["_pred"]:
        return {"status": "unavailable"}
    a = res.get("_aemo")
    if a is None:
        return {"status": "unavailable"}
    y = np.asarray(res["_y"], float)
    t = res["_t"]
    ours = rolling_dist(y, np.asarray(res["_pred"]["B2 +官方"], float), t)
    off = rolling_dist(y, np.asarray(a, float), t)
    clim = climatology_spike(y, t)
    ok = (np.isfinite(ours["p_spike"]) & np.isfinite(off["p_spike"]) & np.isfinite(clim))
    if ok.sum() < 24 * 30:
        return {"status": "too_short", "n": int(ok.sum())}
    yy = (y > SPIKE).astype(float)
    hard = (np.asarray(a, float) > SPIKE).astype(float)
    out = {"status": "ok", "hours": int(ok.sum()), "spike_rate": round(float(yy[ok].mean()), 4),
           "ours": score(y, ours, ok), "aemo": score(y, off, ok),
           "brier_clim": round(float(np.mean((clim[ok] - yy[ok]) ** 2)), 5),
           "brier_aemo_hard": round(float(np.mean((hard[ok] - yy[ok]) ** 2)), 5),
           "reliability": reliability(ours["p_spike"][ok], yy[ok])}
    # 给实时用：最近 POOL_DAYS 天每个小时的样本外残差分位数（纸面交易明天的分布用它）
    tt = pd.to_datetime(pd.Series(t)) + NEM
    last = tt.max().normalize()
    m = (tt >= last - pd.Timedelta(days=POOL_DAYS)).to_numpy()
    r = y - np.asarray(res["_pred"]["B2 +官方"], float)
    hod = tt.dt.hour.to_numpy()
    out["resid"] = {int(h): [round(float(v), 1) for v in np.quantile(r[m & (hod == h)], np.linspace(0, 1, 41))]
                    for h in range(24) if (m & (hod == h)).sum() >= MIN_DAYS}
    return out


def tomorrow(pred: np.ndarray, hours: np.ndarray, resid: dict) -> dict | None:
    """明天每小时的 q10/q50/q90 和尖峰概率：点预测 + 存下来的残差分位数（41 个点近似分布）。"""
    if not resid:
        return None
    q10, q50, q90, ps = [], [], [], []
    for p_, h in zip(pred, hours):
        r = np.asarray(resid.get(int(h)) or resid.get(str(int(h))) or [], float)
        if len(r) == 0:
            return None
        smp = p_ + r
        q10.append(round(float(np.quantile(smp, .1)), 1))
        q50.append(round(float(np.quantile(smp, .5)), 1))
        q90.append(round(float(np.quantile(smp, .9)), 1))
        ps.append(round(float((smp > SPIKE).mean()), 3))
    return {"q10": q10, "q50": q50, "q90": q90, "p_spike": ps}

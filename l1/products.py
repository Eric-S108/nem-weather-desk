"""
能真正下单的产品 —— 把预测变成按现货价结算的实物决策，扣掉交易成本。
=====================================================================
为什么不是交易所合约：ASX 的 NEM 电力衍生品只有月度、季度和年度（基荷 / 峰荷 / $300 cap /
早晚峰），**没有日度或周度合约**（ASX 2025-06 合约说明）。"明天哪几个小时贵"这种日前信号，
在交易所上表达不出来。能把它变成钱的，是**按现货价结算的实物决策**：

  1. 储能套利        1 MW / 2 MWh，每天先充后放一个循环           —— 吃日内排序
  2. 可调负荷        每天必须用够 8 MWh、最大 1 MW，挑小时用电       —— 吃日内排序
  3. 日前定机组的调峰 1 MW 燃气机组，前一天定好哪几个小时运行          —— 吃价格水平
                     （燃料要提前订、有启动成本、开了至少连续运行 3 小时）

"AEMO 偏差交易"保留为**信号检验**：它说明我们对 AEMO 的修正方向对不对，
但预调度价不是成交价，没有对手方，不能直接下单。

每个产品都用同一组预测来源做决策、按实际价结算：
  一周前同期 / AEMO 预调度直接用 / 价格模型 / 价格 + AEMO / + 天气 / + 挑过的天气
"AEMO 直接用"是最重要的对照：不建任何模型、直接拿官方预测排计划能赚多少。

成本假设（全部可改，页面上列出来源，并做敏感性）：
  · AEMO 市场费用    $0.55/MWh，充、放两个方向都收（FY27 起按总电量计费；
                     FY26 市场用户各项 $/MWh 费率合计约 $0.53）
  · 电池衰减          $15/MWh 放电量（敏感性 0–50）
  · 往返效率          85%
  · 调峰机组边际成本  $130/MWh（2026 Q1 东海岸气价均值 $10.61/GJ × 热耗 11.5 GJ/MWh
                     ≈ $122，加可变运维 $8）；启动成本 $60/MW/次；最小连续运行 3 小时
  · 全部按"价格接受者"算：规模放大时自己的出力会压价，这里没算。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NEM_OFFSET_H = 10

COSTS = {
    "fee": 0.55,            # $/MWh，AEMO 市场费用，按总电量（两个方向）
    "eta": 0.85,            # 储能往返效率
    "degr": 15.0,           # $/MWh 放电量，电池衰减
    "batt_mw": 1.0, "batt_h": 2,
    "load_mwh": 8, "load_mw": 1.0,
    "srmc": 130.0,          # $/MWh，调峰机组边际成本
    "start": 60.0,          # $/MW/次
    "min_up": 3,            # 小时
}
SOURCES = {
    "fee": "AEMO FY26 Budget & Fees（市场用户 $/MWh 各项合计约 0.53）；FY27 起按总电量双向计费",
    "gas": "AEMO Quarterly Energy Dynamics Q1 2026：东海岸气价均值 $10.61/GJ",
    "asx": "ASX Australian Electricity fact sheet（2025-06）：只有月度/季度/年度合约，无日度、周度",
}
MODEL_ORDER = ["B0 naive", "AEMO 直用", "B1 自回归", "B2 +官方", "B3 +天气", "B3 选特征"]
REC_ORDER = ["AEMO 直用", "B0 naive", "B1 自回归", "B2 +官方", "B3 +天气", "B3 选特征"]


# ------------------------------------------------------------------ 三个产品的单日决策
def battery_plan(p: np.ndarray, k: int, eta: float, cyc_cost_per_h: float):
    """先充后放一个循环。cyc_cost_per_h：每个充放小时对摊的成本（费用 + 衰减）。"""
    n = len(p)
    best = (0.0, None, None)
    if n < 2 * k:
        return best
    for s in range(k, n - k + 1):
        ci = np.argsort(p[:s], kind="stable")[:k]
        di = s + np.argsort(-p[s:], kind="stable")[:k]
        v = eta * float(p[di].sum()) - float(p[ci].sum()) - cyc_cost_per_h * k
        if v > best[0]:
            best = (v, ci, di)
    return best


def battery_cost_per_h(c: dict) -> float:
    """1 MW 充 1 小时买 1 MWh、放 1 小时卖 η MWh：费用两个方向都收，衰减按放电量。"""
    return c["fee"] * (1 + c["eta"]) + c["degr"] * c["eta"]


def battery_day(p, y, c) -> float:
    v, ci, di = battery_plan(p, c["batt_h"], c["eta"], battery_cost_per_h(c))
    if ci is None:
        return 0.0
    return c["batt_mw"] * (c["eta"] * float(y[di].sum()) - float(y[ci].sum())
                           - battery_cost_per_h(c) * c["batt_h"])


def flexload_day(p, y, c) -> float:
    """每天必须用够 load_mwh，最大 load_mw。按预测挑最便宜的小时用电。
    价值 = 相对"全天平均用电"省下的钱。市场费用两边一样，不影响差额。"""
    k = int(round(c["load_mwh"] / c["load_mw"]))
    pick = np.argsort(p, kind="stable")[:k]
    flat = c["load_mwh"] * float(np.mean(y))
    return flat - c["load_mw"] * float(y[pick].sum())


def peaker_plan(p: np.ndarray, srmc: float, fee: float, start: float, m: int) -> np.ndarray:
    """日前定机组：动态规划选运行小时，最大化 Σ(p − srmc − fee) − start × 启动次数。
    开机后至少连续运行 m 小时；当天结束时要么已关机、要么已满 m 小时。"""
    n = len(p)
    mar = p - srmc - fee
    NEG = -1e18
    # 状态 0 = 关机；状态 r (1..m) = 已连续运行 r 小时（m 表示 ≥ m）
    V = np.full((n + 1, m + 1), NEG)
    B = np.zeros((n + 1, m + 1), int)
    V[0, 0] = 0.0
    for t in range(n):
        for s in range(m + 1):
            if V[t, s] <= NEG / 2:
                continue
            # 关机
            if s == 0 or s == m:
                if V[t, s] > V[t + 1, 0]:
                    V[t + 1, 0], B[t + 1, 0] = V[t, s], s
            # 运行
            ns = min(s + 1, m)
            v = V[t, s] + mar[t] - (start if s == 0 else 0.0)
            if v > V[t + 1, ns]:
                V[t + 1, ns], B[t + 1, ns] = v, s
    end = 0 if V[n, 0] >= V[n, m] else m
    on = np.zeros(n, bool)
    s = end
    for t in range(n, 0, -1):
        prev = B[t, s]
        on[t - 1] = s != 0
        s = prev
    return on


def peaker_day(p, y, c) -> float:
    on = peaker_plan(p, c["srmc"], c["fee"], c["start"], c["min_up"])
    starts = int(on[0]) + int(np.sum(on[1:] & ~on[:-1]))
    return float(np.sum((y - c["srmc"] - c["fee"])[on])) - c["start"] * starts


PRODUCTS = {
    "battery": ("储能套利", "1 MW / 2 MWh，每天先充后放一个循环", battery_day),
    "flexload": ("可调负荷", "每天用够 8 MWh、最大 1 MW，挑便宜的小时", flexload_day),
    "peaker": ("日前定机组的调峰", "1 MW 燃气机组，前一天定好运行小时，开机至少 3 小时", peaker_day),
}


# ------------------------------------------------------------------ 汇总
def _groups(t):
    day = (pd.Series(pd.to_datetime(t)) + pd.Timedelta(hours=NEM_OFFSET_H)).dt.floor("D").to_numpy()
    udays, inv = np.unique(day, return_inverse=True)
    order = np.argsort(np.asarray(t), kind="stable")
    gs = [(udays[i], order[inv[order] == i]) for i in range(len(udays))]
    gs = [(d_, g) for d_, g in gs if len(g) >= 20]
    return [str(pd.Timestamp(d_).date()) for d_, _ in gs], [g for _, g in gs]


def _sources(res: dict) -> dict[str, np.ndarray]:
    src = {}
    for m in MODEL_ORDER:
        if m == "AEMO 直用":
            a = res.get("_aemo")
            if a is not None and np.isfinite(a).all():
                src[m] = np.asarray(a, float)
        elif m in res.get("_pred", {}) and np.isfinite(res["_pred"][m]).all():
            src[m] = np.asarray(res["_pred"][m], float)
    return src


def daily_pnl(res: dict, key: str, c: dict | None = None) -> tuple[list[str], dict[str, np.ndarray]]:
    c = {**COSTS, **(c or {})}
    fn = PRODUCTS[key][2]
    y = np.asarray(res["_y"], float)
    dates, groups = _groups(res["_t"])
    src = _sources(res)
    out = {m: np.array([fn(p[g], y[g], c) for g in groups]) for m, p in src.items()}
    out["完美预知"] = np.array([fn(y[g], y[g], c) for g in groups])
    return dates, out


def _ci(x):
    import ladder as L
    ci = L._block_boot(np.asarray(x, float))
    return None if ci is None else [round(v, 2) for v in ci]


def product_summary(res: dict, key: str, c: dict | None = None, keep_daily: bool = True) -> dict:
    dates, s = daily_pnl(res, key, c)
    pf = float(s["完美预知"].mean())
    rows = []
    for m in MODEL_ORDER:
        if m not in s:
            continue
        g = s[m]
        r = {"model": m, "daily": round(float(g.mean()), 2), "ci": _ci(g),
             "capture": round(float(g.mean()) / pf, 3) if pf > 0 else None,
             "win_days": round(float((g > 0).mean()), 3)}
        for ref, tag in (("B2 +官方", "vs_b2"), ("AEMO 直用", "vs_aemo"), ("B1 自回归", "vs_b1")):
            if m != ref and ref in s:
                d = g - s[ref]
                r[tag] = round(float(d.mean()), 2)
                r[tag + "_ci"] = _ci(d)
        rows.append(r)
    # 推荐：
    #   · 最好的那个扣完成本都不显著赚钱 → "不做"（这个区不碰这个产品）；
    #   · 否则在"均值为正、且不显著差于最好那个"的里面挑最简单的。
    #     简单的顺序：AEMO 直接用（不建模型）→ 一周前同期 → 价格模型 → 价格 + AEMO → + 天气 → + 挑过的天气。
    best = max(rows, key=lambda r_: r_["daily"])["model"]
    bci = _ci(s[best])
    rec = None
    if bci is None or bci[0] <= 0:
        rec = "不做"
    else:
        for m in REC_ORDER:
            if m not in s or s[m].mean() <= 0:
                continue
            if m == best:
                rec = m
                break
            ci = _ci(s[m] - s[best])
            if ci is not None and ci[1] >= 0:
                rec = m
                break
        rec = rec or best
    out = {"name": PRODUCTS[key][0], "desc": PRODUCTS[key][1], "days": len(dates),
           "perfect": round(pf, 2), "rows": rows, "recommended": rec, "best": best}
    if keep_daily:
        out["daily"] = {"dates": dates, **{k: [round(float(v), 2) for v in x] for k, x in s.items()}}
    return out


def sensitivity(res: dict) -> dict:
    """成本敏感性：电池衰减成本、调峰机组边际成本。看天气增量和最好模型的收益怎么变。"""
    out = {"battery_degr": [], "peaker_srmc": []}
    for dg in (0.0, 5.0, 15.0, 30.0, 50.0):
        sm = product_summary(res, "battery", {"degr": dg}, keep_daily=False)
        rr = {r["model"]: r for r in sm["rows"]}
        out["battery_degr"].append({
            "x": dg, "b2": rr.get("B2 +官方", {}).get("daily"),
            "b3": rr.get("B3 +天气", {}).get("daily"),
            "aemo": rr.get("AEMO 直用", {}).get("daily"),
            "best": sm["best"], "best_daily": max(r["daily"] for r in sm["rows"]),
            "w_gain": rr.get("B3 +天气", {}).get("vs_b2"), "w_ci": rr.get("B3 +天气", {}).get("vs_b2_ci")})
    for sc in (100.0, 130.0, 180.0):
        sm = product_summary(res, "peaker", {"srmc": sc}, keep_daily=False)
        rr = {r["model"]: r for r in sm["rows"]}
        out["peaker_srmc"].append({
            "x": sc, "b2": rr.get("B2 +官方", {}).get("daily"),
            "b3": rr.get("B3 +天气", {}).get("daily"),
            "aemo": rr.get("AEMO 直用", {}).get("daily"),
            "perfect": sm["perfect"],
            "best": sm["best"], "best_daily": max(r["daily"] for r in sm["rows"]),
            "w_gain": rr.get("B3 +天气", {}).get("vs_b2"), "w_ci": rr.get("B3 +天气", {}).get("vs_b2_ci")})
    return out


def products(res: dict) -> dict:
    if res.get("status") != "ok" or "_pred" not in res:
        return {"status": "unavailable"}
    return {"status": "ok", "costs": dict(COSTS), "sources": SOURCES,
            "items": {k: product_summary(res, k) for k in PRODUCTS},
            "sensitivity": sensitivity(res)}

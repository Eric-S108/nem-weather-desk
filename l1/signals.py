"""
本机私有：期货信号、仓位和风险。**不提交、不发布。**
=================================================
ASX Energy 网站条款禁止爬取、下载、转载其价格，所以市场价由交易员自己录入：

    python l1/signals.py --template        # 生成 private/futures_prices.csv（列出当前所有合约，价格留空）
    （打开 CSV，从你自己的交易屏幕填 bid / ask）
    python l1/signals.py                   # 生成 private/desk.html 并在终端打印信号

可选：
    private/risk.json        风险参数（单日亏损上限、每个合约最大 MW、最低胜率等）
    private/positions.csv    现有持仓：region,contract,product,side,mw,entry
                             side 为 long / short；用来算组合层面的风险

信号规则（全部写在页面上）：
  · 胜率 = 我们的结算价分布里，按 ask 买入（加费用）后结算更高的概率；卖出同理；
  · 胜率 ≥ min_prob 且 期望优势 ≥ min_edge（$/MWh，已扣费用）才出"买入 / 卖出"，否则"观望"；
  · 仓位 = min(每合约上限, 单日亏损上限 ÷ 每 MW 的 95% 单日亏损)，取整到 1 MW；
  · 风险：5% 最差结算下的总盈亏、单日 95% 亏损、已结算比例、尖峰暴露、区间历史覆盖率。
"""
from __future__ import annotations

import argparse
import html
import json
import os

import numpy as np
import pandas as pd

import futures as F

ROOT = F.ROOT
PRIV = os.path.join(ROOT, "private")
PRICES = os.path.join(PRIV, "futures_prices.csv")
RISK = os.path.join(PRIV, "risk.json")
POS = os.path.join(PRIV, "positions.csv")
OUT = os.path.join(PRIV, "desk.html")
DEFAULT_RISK = {"daily_loss_limit": 20000.0, "max_mw_per_contract": 20, "min_prob": 0.70,
                "min_edge": 1.0, "fee_per_mwh": F.FEE_PER_MWH}


def period_from_label(label: str) -> dict:
    lab = label.strip()
    if lab[:1].upper() == "Q" and len(lab) == 4:
        q, yy = int(lab[1]), int(lab[2:])
        p = pd.Period(year=2000 + yy, quarter=q, freq="Q")
        kind = "quarter"
    else:
        p = pd.Period(pd.to_datetime(lab, format="%b %y"), freq="M")
        kind = "month"
    return {"label": lab, "kind": kind, "start": p.start_time.normalize(), "end": p.end_time.normalize()}


def load_risk() -> dict:
    r = dict(DEFAULT_RISK)
    if os.path.exists(RISK):
        with open(RISK) as f:
            r.update(json.load(f))
    return r


def decide(samples: np.ndarray, mean: float, bid: float | None, ask: float | None, fee: float,
           min_prob: float, min_edge: float, allow_sell: bool = True) -> dict:
    """按 ask 买、按 bid 卖，各算胜率和优势；都不够就观望。

    allow_sell=False 用于 $300 cap：我们按小时均价算 cap，5 分钟尖峰被平均掉，公允价值是**下限**。
    下限比市场价低不代表市场贵，所以 cap 只允许买入信号（买入时下限都高于 ask，真实优势只会更大）。"""
    out = {"side": 0, "action": "观望", "price": None, "edge": None, "prob": None}
    cands = []
    if ask is not None and np.isfinite(ask):
        cands.append((+1, ask, float(mean - ask - fee), float((samples > ask + fee).mean())))
    if allow_sell and bid is not None and np.isfinite(bid):
        cands.append((-1, bid, float(bid - mean - fee), float((samples < bid - fee).mean())))
    if not cands:
        return out
    side, px, edge, prob = max(cands, key=lambda c: c[2])
    out.update({"price": px, "edge": round(edge, 2), "prob": round(prob, 3), "best_side": side})
    if prob >= min_prob and edge >= min_edge:
        out.update({"side": side, "action": "买入" if side > 0 else "卖出"})
    return out


def size(fv: dict, side: int, price: float, risk: dict) -> tuple[int, float]:
    per_mw = F.daily_risk_per_mw(fv, price, side)
    if per_mw <= 0:
        return int(risk["max_mw_per_contract"]), 0.0
    mw = int(min(risk["max_mw_per_contract"], np.floor(risk["daily_loss_limit"] / per_mw)))
    return max(mw, 0), per_mw


def evaluate(con, rows: pd.DataFrame, risk: dict, fut: dict | None) -> pd.DataFrame:
    mats, out = {}, []
    for _, r in rows.iterrows():
        reg, prod = str(r["region"]), str(r["product"])
        if reg not in mats:
            s = F.load_prices(con, reg)
            mats[reg] = (F.day_matrix(s), s)
        mat, _ = mats[reg]
        asof = mat.dropna().index.max()
        per = period_from_label(str(r["contract"]))
        fv = F.fair_value(mat, per, asof, prod)
        if fv is None:
            continue
        cal = (((fut or {}).get("regions") or {}).get(reg) or {}).get("calibration") or {}
        c_ = cal.get(f"{per['kind']}_{prod if prod in ('base', 'peak') else 'peak'}") or {}
        if c_.get("k90"):
            fv = F.apply_widen(fv, c_["k50"], c_["k90"], floor=0.0 if prod == "cap300" else None)
        bid = pd.to_numeric(r.get("bid"), errors="coerce")
        ask = pd.to_numeric(r.get("ask"), errors="coerce")
        fee = float(risk["fee_per_mwh"])
        d = decide(fv["_samples"], fv["mean"], bid, ask, fee, risk["min_prob"], risk["min_edge"],
                   allow_sell=prod != "cap300")
        rec = {"region": reg, "contract": per["label"], "product": prod, "bid": bid, "ask": ask,
               "fair": fv["mean"], "p5": fv["p5"], "p25": fv["p25"], "p75": fv["p75"], "p95": fv["p95"],
               "settled_share": fv["settled_share"], "days_left": fv["days_left"],
               "hours_total": fv["hours_total"], "cov90": c_.get("oos_cov90"), **d}
        side = d["side"] or d.get("best_side") or 0
        if side and d["price"] is not None:
            mw, per_mw = size(fv, side, d["price"], risk)
            rec["mw_max"] = mw
            rec["risk_day_per_mw"] = round(per_mw, 0)
            if d["side"]:
                v = fv["_samples"]
                pnl = side * (v - d["price"]) - fee
                rec["mw"] = mw
                rec["exp_pnl"] = round(float(pnl.mean() * fv["hours_total"] * mw), 0)
                rec["p5_pnl"] = round(float(np.percentile(pnl, 5) * fv["hours_total"] * mw), 0)
                rec["risk_day"] = round(per_mw * mw, 0)
        out.append(rec)
    return pd.DataFrame(out)


def positions_risk(con, risk: dict, fut: dict | None) -> pd.DataFrame | None:
    if not os.path.exists(POS):
        return None
    p = pd.read_csv(POS)
    if p.empty:
        return None
    rows = []
    for _, r in p.iterrows():
        reg, prod = str(r["region"]), str(r["product"])
        s = F.load_prices(con, reg)
        mat = F.day_matrix(s)
        asof = mat.dropna().index.max()
        per = period_from_label(str(r["contract"]))
        fv = F.fair_value(mat, per, asof, prod)
        if fv is None:
            continue
        side = 1 if str(r["side"]).lower().startswith("l") else -1
        mw, entry = float(r["mw"]), float(r["entry"])
        v = fv["_samples"]
        pnl = side * (v - entry) * fv["hours_total"] * mw
        rows.append({"region": reg, "contract": per["label"], "product": prod,
                     "side": "多" if side > 0 else "空", "mw": mw, "entry": entry, "fair": fv["mean"],
                     "exp_pnl": round(float(pnl.mean()), 0), "p5_pnl": round(float(np.percentile(pnl, 5)), 0),
                     "risk_day": round(F.daily_risk_per_mw(fv, entry, side) * mw, 0),
                     "settled_share": fv["settled_share"]})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 组合风险
def leg_daily_pnl(mat: pd.DataFrame, product: str, side: int, mw: float, price: float,
                  days: pd.DatetimeIndex) -> pd.Series:
    """一条腿在历史每个交易日的盈亏：side × MW × 当日产品小时数 × (当日产品均价 − 开仓价)。
    用**同一批历史日期**算每条腿，组合加总时各区之间的相关性就自然带进来了。"""
    hrs = np.arange(24)
    sub = mat.reindex(days)
    out = []
    for d, row in sub.iterrows():
        msk = F.hour_mask(product, hrs, np.full(24, d.weekday()))
        v = row.to_numpy(float)
        if not np.isfinite(v).all() or msk.sum() == 0:
            out.append(0.0 if msk.sum() == 0 else np.nan)
            continue
        out.append(side * mw * msk.sum() * (float(F.payoff(product, v[msk]).mean()) - price))
    return pd.Series(out, index=days)


def portfolio_risk(mats: dict, legs: list[dict], lookback: int = 365) -> dict | None:
    """组合在历史上每个交易日会亏多少：VaR95、ES95（最差 5% 日子的平均亏损）、历史最差一天（压力重放）。"""
    if not legs:
        return None
    ends = [mats[l_["region"]].dropna().index.max() for l_ in legs]
    end = min(ends)
    days = pd.date_range(end - pd.Timedelta(days=lookback - 1), end, freq="D")
    tot = pd.Series(0.0, index=days)
    for l_ in legs:
        tot = tot + leg_daily_pnl(mats[l_["region"]], l_["product"], l_["side"], l_["mw"], l_["price"], days)
    tot = tot.dropna()
    if len(tot) < 60:
        return None
    q = float(np.quantile(tot, 0.05))
    full_days = mats[legs[0]["region"]].dropna().index
    for l_ in legs[1:]:
        full_days = full_days.intersection(mats[l_["region"]].dropna().index)
    allp = pd.Series(0.0, index=full_days)
    for l_ in legs:
        allp = allp + leg_daily_pnl(mats[l_["region"]], l_["product"], l_["side"], l_["mw"], l_["price"], full_days)
    worst = allp.idxmin()
    return {"days": int(len(tot)), "var95": round(-q, 0), "es95": round(-float(tot[tot <= q].mean()), 0),
            "worst": round(float(allp.min()), 0), "worst_day": str(worst.date()),
            "sum_naive": None}


def allocate(mats: dict, pos_legs: list[dict], new_legs: list[dict], limit: float,
             lookback: int = 365) -> list[int]:
    """组合限额下分配新信号的 MW：按"期望盈亏 ÷ 单日风险"从高到低一笔一笔加，
    每加一笔检查"现有持仓 + 已加的"组合 ES95，超了就把这一笔减到刚好不超。
    每条腿先算 1 MW 的历史逐日盈亏，之后只是线性缩放，算得很快。"""
    if not new_legs:
        return []
    ends = [mats[l_["region"]].dropna().index.max() for l_ in pos_legs + new_legs]
    end = min(ends)
    days = pd.date_range(end - pd.Timedelta(days=lookback - 1), end, freq="D")
    unit = lambda l_: leg_daily_pnl(mats[l_["region"]], l_["product"], l_["side"], 1.0, l_["price"], days)  # noqa: E731
    base = pd.Series(0.0, index=days)
    for l_ in pos_legs:
        base = base + unit(l_) * l_["mw"]

    def es(x: pd.Series) -> float:
        x = x.dropna()
        if len(x) < 60:
            return 0.0
        q = np.quantile(x, 0.05)
        return -float(x[x <= q].mean())

    order = sorted(range(len(new_legs)), key=lambda i: -new_legs[i].get("score", 0.0))
    mw = [0] * len(new_legs)
    cur = base.copy()
    for i in order:
        u = unit(new_legs[i])
        want = int(new_legs[i]["mw"])
        lo, hi = 0, want
        while lo < hi:                       # 最大的整数 MW，使 ES95 ≤ 上限
            mid = (lo + hi + 1) // 2
            if es(cur + u * mid) <= limit:
                lo = mid
            else:
                hi = mid - 1
        mw[i] = lo
        cur = cur + u * lo
    return mw


def template(con) -> int:
    os.makedirs(PRIV, exist_ok=True)
    rows = []
    for reg in F.REGIONS:
        mat = F.day_matrix(F.load_prices(con, reg))
        asof = mat.dropna().index.max()
        for per in F.contract_periods(asof):
            for prod in ("base", "peak", "cap300"):
                rows.append({"region": reg, "contract": per["label"], "product": prod, "bid": "", "ask": ""})
    if os.path.exists(PRICES):
        print(f"{PRICES} 已存在，不覆盖。要重建先删掉它。")
    else:
        pd.DataFrame(rows).to_csv(PRICES, index=False)
        print(f"→ {PRICES}：{len(rows)} 行，填 bid / ask（$/MWh），不交易的合约留空即可")
    if not os.path.exists(RISK):
        with open(RISK, "w") as f:
            json.dump(DEFAULT_RISK, f, ensure_ascii=False, indent=1)
        print(f"→ {RISK}：风险参数（单日亏损上限 ${DEFAULT_RISK['daily_loss_limit']:,.0f} 等），按需修改")
    if not os.path.exists(POS):
        pd.DataFrame(columns=["region", "contract", "product", "side", "mw", "entry"]).to_csv(POS, index=False)
        print(f"→ {POS}：现有持仓（可留空）")
    _ensure_gitignore()
    return 0


def _ensure_gitignore():
    gi = os.path.join(ROOT, ".gitignore")
    line = "private/"
    cur = open(gi).read() if os.path.exists(gi) else ""
    if line not in cur.split():
        with open(gi, "a") as f:
            f.write(("\n" if cur and not cur.endswith("\n") else "") + line + "\n")


def _mw(r) -> int:
    v = r.get("mw_scaled")
    return int(v) if v is not None and np.isfinite(v) else int(r.get("mw", 0) or 0)


def _pct(v) -> str:
    return "—" if v is None or not np.isfinite(v) else f"{v:.0%}"


def _port_html(port: dict | None, m) -> str:
    if not port:
        return ""
    rows = []
    for k, lab in (("pos", "只有现有持仓"), ("all", "现有持仓 + 建议的新信号")):
        r = port.get(k)
        if not r:
            continue
        rows.append(f"<tr><td>{lab}</td><td class='num'>{m(r['var95'])}</td><td class='num'>{m(r['es95'])}</td>"
                    f"<td class='num'>{m(r['worst'], 0, True)}<span class='hint'>{r['worst_day']}</span></td>"
                    f"<td class='num'>{r['days']} 天</td></tr>")
    if not rows:
        return ""
    note = ("" if not port.get("cut") else
            f"<b>组合限额生效：{port['cut']}/{port['n_new']} 笔新信号被减量或清零</b>"
            f"（按期望盈亏 ÷ 单日风险从高到低分配额度；信号表里的 MW 是分配后的）。")
    return f"""<h2>组合风险</h2>
<p class="lead2">用同一批历史交易日算每条腿的盈亏再加总，各区之间的相关性（NSW 和 QLD、VIC 和 SA 常常一起涨跌）自然带进来，
不是简单相加。ES95 = 最差 5% 日子的平均亏损，要求不超过单日上限 {m(port['limit'])}。{note}
"历史最差一天"= 把这个组合放回有数据以来每一天，亏得最多的那天（压力重放）。</p>
<div class="card"><div class="scroll"><table class="act"><thead><tr><th>组合</th><th class="num">VaR95（单日）</th>
<th class="num">ES95（单日）</th><th class="num">历史最差一天</th><th class="num">样本</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div></div>"""


def render(sig: pd.DataFrame, pos: pd.DataFrame | None, risk: dict, asof: str, port: dict | None = None) -> str:
    import trader as T
    esc = html.escape
    pn = {k: v[0] for k, v in F.PRODUCTS.items()}

    def m(v, nd=0, sign=False):
        return T._money(None if v is None or (isinstance(v, float) and not np.isfinite(v)) else v, sign, nd)

    act = sig[sig["action"] != "观望"] if len(sig) else sig
    trs = []
    for _, r in (act if len(act) else sig.head(0)).iterrows():
        chip = (f'<span class="chip {"good" if r["side"] > 0 else "critical"}"><span aria-hidden="true">'
                f'{"▲" if r["side"] > 0 else "▼"}</span>{esc(r["action"])}</span>')
        trs.append(f"""<tr><td class="rg">{esc(r['region'])}</td><td>{esc(r['contract'])} {esc(pn.get(r['product'], r['product']))}</td>
<td>{chip}</td>
<td class="num"><span class="big2">{_mw(r)} MW</span><span class="hint">单笔上限 {int(r.get('mw_max', 0))} MW</span></td>
<td class="num">{m(r['price'], 2)}<span class="hint">公允 {m(r['fair'], 2)}</span></td>
<td class="num">{m(r['edge'], 2, True)}<span class="hint">胜率 {r['prob']:.0%}</span></td>
<td class="num">{m(r.get('exp_pnl'), 0, True)}<span class="hint">5% 最差 {m(r.get('p5_pnl'), 0, True)}</span></td>
<td class="num">{m(r.get('risk_day'))}<span class="hint">每 MW {m(r.get('risk_day_per_mw'))}</span></td>
<td class="why">已结算 {r['settled_share']:.0%}，剩 {int(r['days_left'])} 天；区间历史覆盖 {('—' if r['cov90'] is None or not np.isfinite(r['cov90']) else f"{r['cov90']:.0%}")}</td></tr>""")
    allr = []
    for _, r in sig.iterrows():
        allr.append(f"<tr><td>{esc(r['region'])}</td><td>{esc(r['contract'])}</td><td>{esc(pn.get(r['product'], r['product']))}</td>"
                    f"<td class='num'>{m(r['bid'], 2)}</td><td class='num'>{m(r['ask'], 2)}</td>"
                    f"<td class='num'>{m(r['fair'], 2)}</td><td class='num'>{m(r['p5'], 0)} – {m(r['p95'], 0)}</td>"
                    f"<td class='num'>{_pct(r['prob'])}</td>"
                    f"<td>{esc(r['action'])}</td></tr>")
    ptab = ""
    if pos is not None and len(pos):
        prs = "".join(f"<tr><td>{esc(r['region'])}</td><td>{esc(r['contract'])} {esc(pn.get(r['product'], r['product']))}</td>"
                      f"<td>{esc(r['side'])} {r['mw']:g} MW @ {m(r['entry'], 2)}</td><td class='num'>{m(r['fair'], 2)}</td>"
                      f"<td class='num'>{m(r['exp_pnl'], 0, True)}</td><td class='num'>{m(r['p5_pnl'], 0, True)}</td>"
                      f"<td class='num'>{m(r['risk_day'])}</td></tr>" for _, r in pos.iterrows())
        tot_day = float(pos["risk_day"].sum())
        ptab = f"""<h2>现有持仓</h2>
<p class="lead2">单日 95% 亏损合计 <b>{m(tot_day)}</b>（各合约直接相加，没算分散效应，偏保守）；
单日上限 {m(risk['daily_loss_limit'])}。{'<b>已超上限，先减仓。</b>' if tot_day > risk['daily_loss_limit'] else ''}</p>
<div class="card"><div class="scroll"><table class="act"><thead><tr><th>区</th><th>合约</th><th>持仓</th><th class="num">公允</th>
<th class="num">期望盈亏</th><th class="num">5% 最差</th><th class="num">单日 95% 亏损</th></tr></thead><tbody>{prs}</tbody></table></div></div>"""
    n_act = len(act)
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>私有交易台 · 期货信号</title>
<style>{T.CSS}</style></head><body><div class="wrap">
<div class="top"><div class="brand">私有交易台 · 期货信号</div><nav class="nav"><a href="../docs/index.html">公开交易台</a></nav></div>
<div class="asof">现货结算到 {esc(asof)} · 市场价来自你录入的 private/futures_prices.csv · 本页只在本机，不提交、不发布</div>
<section class="hero"><div><p class="q">现在有信号的合约</p><div class="big">{n_act}<small>/ {len(sig)}</small></div></div>
<p class="say">胜率 ≥ {risk['min_prob']:.0%} 且扣完费用的优势 ≥ ${risk['min_edge']:g}/MWh 才出信号。
仓位 = min({risk['max_mw_per_contract']} MW, 剩余单日亏损额度 ÷ 每 MW 的 95% 单日亏损)；剩余额度 = 上限 {m(risk['daily_loss_limit'])} − 现有持仓已占用的部分。
每个信号的数量是<b>各自单独</b>按剩余额度算的，同时做几笔要按比例缩小。</p></section>
<h2>信号</h2>
<div class="card"><div class="scroll"><table class="act"><thead><tr><th>区</th><th>合约</th><th>动作</th><th class="num">数量</th>
<th class="num">成交价</th><th class="num">优势</th><th class="num">期望盈亏</th><th class="num">单日 95% 亏损</th><th>风险</th></tr></thead>
<tbody>{''.join(trs) or '<tr><td colspan="9" class="why">没有达到门槛的合约。</td></tr>'}</tbody></table></div></div>
{ptab}
{_port_html(port, m)}
<h2>全部合约</h2>
<div class="card"><div class="scroll"><table class="act"><thead><tr><th>区</th><th>合约</th><th>产品</th><th class="num">bid</th><th class="num">ask</th>
<th class="num">公允</th><th class="num">90% 区间</th><th class="num">胜率</th><th>动作</th></tr></thead><tbody>{''.join(allr)}</tbody></table></div></div>
<h2>这些数字没有包含的风险</h2>
<div class="how">
  <div><b>盯市风险</b>：仓位按"持有到交割、每天一天现货结算进来"算单日亏损；期货价格本身每天的波动没算（没有公开的历史期货价）。</div>
  <div><b>模型风险</b>：公允价值来自过去 90 天现货的整块抽样，区间已按历史放宽；结构性变化（新机组、断电、气价跳变）不在样本里。</div>
  <div><b>尖峰与 $300 cap</b>：结算价按小时均价近似，5 分钟尖峰被平均掉，cap 的公允价值是<b>下限</b>，所以 cap 只出买入信号、不出卖出信号。</div>
  <div><b>流动性与执行</b>：月度合约成交稀，bid/ask 价差可能比优势还大；本页假设按你录入的价格全部成交。</div>
</div>
<footer>不是投资建议。信号只基于公开的 AEMO 现货价与你录入的报价，没有在历史期货价上回测过。</footer>
</div></body></html>"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="本机私有期货信号")
    ap.add_argument("--template", action="store_true", help="生成价格录入模板、风险参数和持仓文件")
    a = ap.parse_args(argv)
    import store
    con = store.connect()
    if a.template:
        return template(con)
    _ensure_gitignore()
    if not os.path.exists(PRICES):
        print("还没有价格文件。先运行：python l1/signals.py --template")
        return 1
    px = pd.read_csv(PRICES)
    px = px[pd.to_numeric(px["bid"], errors="coerce").notna() | pd.to_numeric(px["ask"], errors="coerce").notna()]
    if px.empty:
        print(f"{PRICES} 里还没有填任何 bid / ask。")
        return 1
    risk = load_risk()
    fut = None
    if os.path.exists(F.OUT):
        with open(F.OUT) as f:
            fut = json.load(f)
    pos = positions_risk(con, risk, fut)
    used = float(pos["risk_day"].sum()) if pos is not None and len(pos) else 0.0
    # 新信号只能用剩下的单日风险额度（现有持仓先占用）
    risk_left = dict(risk, daily_loss_limit=max(0.0, risk["daily_loss_limit"] - used))
    sig = evaluate(con, px, risk_left, fut)
    # 组合层面：用同一批历史日期算各腿盈亏再加总（带上各区之间的相关性），新信号整体缩到 ES95 ≤ 上限
    mats = {}
    for reg in set(list(sig["region"]) + ([] if pos is None or not len(pos) else list(pos["region"]))):
        mats[reg] = F.day_matrix(F.load_prices(con, reg))
    pos_legs = [] if pos is None or not len(pos) else [
        {"region": r["region"], "product": r["product"], "side": 1 if r["side"] == "多" else -1,
         "mw": float(r["mw"]), "price": float(r["entry"])} for _, r in pos.iterrows()]
    act = sig[(sig["action"] != "观望") & (sig.get("mw", 0) > 0)] if len(sig) else sig
    new_legs = [{"region": r["region"], "product": r["product"], "side": int(r["side"]),
                 "mw": float(r["mw"]), "price": float(r["price"])} for _, r in act.iterrows()]
    for l_, (_, r) in zip(new_legs, act.iterrows()):
        rd = float(r.get("risk_day") or 0) or 1.0
        l_["score"] = float(r.get("exp_pnl") or 0) / rd
    alloc = allocate(mats, pos_legs, new_legs, risk["daily_loss_limit"])
    if new_legs:
        sig.loc[act.index, "mw_scaled"] = alloc
    cut = sum(1 for l_, a_ in zip(new_legs, alloc) if a_ < l_["mw"])
    port = {"pos": portfolio_risk(mats, pos_legs),
            "all": portfolio_risk(mats, pos_legs + [dict(l_, mw=a_) for l_, a_ in zip(new_legs, alloc) if a_ > 0]),
            "cut": cut, "n_new": len(new_legs), "limit": risk["daily_loss_limit"]}
    asof = str(F.day_matrix(F.load_prices(con, F.REGIONS[0])).dropna().index.max().date())
    con.close()
    os.makedirs(PRIV, exist_ok=True)
    with open(OUT, "w") as f:
        f.write(render(sig, pos, risk, asof, port))
    for _, r in sig.iterrows():
        if r["action"] != "观望":
            print(f"  {r['action']} {r['region']} {r['contract']} {r['product']} {_mw(r)} MW @ {r['price']:.2f}"
                  f"（公允 {r['fair']:.2f}，胜率 {r['prob']:.0%}，期望 {r.get('exp_pnl', 0):+,.0f}，5% 最差 {r.get('p5_pnl', 0):+,.0f}）")
    print(f"→ {OUT}（只在本机）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

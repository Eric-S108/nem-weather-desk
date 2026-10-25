"""
期货信号回测 —— 用你自己拿到的历史期货结算价（本机私有，不提交、不发布）。
====================================================================
公开渠道拿不到 ASX 的历史期货价，所以"市场价偏离公允价值就交易"这条规则一直没回测过。
拿到数据后（ASX Energy 30 天试用、经纪商导出、或你自己的成交记录），放进：

    private/futures_history.csv
        date,region,contract,product,settle
        2025-11-03,VIC1,Q126,base,112.50
        ...
    date = 交易日；contract 和页面同一写法（Q126、Nov 25）；product = base / peak / cap300

    python l1/futures_bt.py                  # 每周一个站位（快）
    python l1/futures_bt.py --every 1        # 每个交易日（慢）

它回答三件事：
  1. **风险溢价**：期货价比最终结算价平均高多少（按距离交割的远近分组）。
     如果市场价系统性高于结算价，那是对冲方付的保险费，不是我们的机会 —— 公允价值偏低时
     "卖出"信号会一直响，回测会照实显示它赚的是溢价还是信息。
  2. **公允价值有没有信息**：我们的优势（公允 − 市场价）和实际结果（结算 − 市场价）的相关性；
     对照组是"把公允价值换成已结算均价"的朴素估计。
  3. **规则本身**：和页面上一样的门槛（胜率、优势），每个信号 1 MW 持有到交割，
     看命中率、每 MWh 平均盈亏、按合约聚类的 t 值。

严格不偷看：站位日 d 的公允价值只用 d 之前（含 d）已结算的现货；区间放宽系数只用 d 之前已经交割完的合约期估。
"""
from __future__ import annotations

import argparse
import html
import json
import os

import numpy as np
import pandas as pd

import futures as F
import signals as S

HIST = os.path.join(S.PRIV, "futures_history.csv")
OUT_HTML = os.path.join(S.PRIV, "futures_bt.html")
OUT_JSON = os.path.join(S.PRIV, "futures_bt.json")


def _widen_asof(cal_rows: pd.DataFrame, asof: pd.Timestamp) -> tuple[float, float]:
    if cal_rows is None or cal_rows.empty:
        return 1.0, 1.0
    past = cal_rows[cal_rows["end"] < asof]
    return F.widen_factors(past)


def backtest(mats: dict, hist: pd.DataFrame, risk: dict, every: int = 5, n_sims: int = 800) -> pd.DataFrame:
    hist = hist.copy()
    hist["date"] = pd.to_datetime(hist["date"]).dt.normalize()
    dates = sorted(hist["date"].unique())
    keep = set(dates[::max(1, every)])
    hist = hist[hist["date"].isin(keep)]
    cals: dict = {}
    rows = []
    for _, r in hist.iterrows():
        reg, prod = str(r["region"]), str(r["product"])
        if reg not in mats:
            continue
        mat = mats[reg]
        per = S.period_from_label(str(r["contract"]))
        d = pd.Timestamp(r["date"])
        if d >= per["end"]:
            continue
        final = F.fair_value(mat, per, per["end"] + pd.Timedelta(days=1), prod, n_sims=5)
        if final is None or final["settled_share"] < 0.99:
            continue                                     # 合约还没交割完，算不出结果
        fv = F.fair_value(mat, per, d, prod, n_sims=n_sims)
        if fv is None:
            continue
        key = (reg, per["kind"], prod if prod in ("base", "peak") else "peak")
        if key not in cals:
            cals[key] = F.calibrate(mat, key[2], per["kind"], n_sims=600, return_rows=True)[0]
        k50, k90 = _widen_asof(cals[key], d)
        fv = F.apply_widen(fv, k50, k90, floor=0.0 if prod == "cap300" else None)
        px = float(r["settle"])
        fee = float(risk["fee_per_mwh"])
        dd = S.decide(fv["_samples"], fv["mean"], px, px, fee, risk["min_prob"], risk["min_edge"],
                      allow_sell=prod != "cap300")
        rows.append({"date": d, "region": reg, "contract": per["label"], "product": prod,
                     "days_to_end": (per["end"] - d).days, "price": px, "fair": fv["mean"],
                     "naive": fv["settled_avg"], "final": final["mean"], "hours": final["hours_total"],
                     "side": dd["side"], "prob": dd["prob"],
                     "pnl_mwh": (dd["side"] * (final["mean"] - px) - fee) if dd["side"] else 0.0})
    return pd.DataFrame(rows)


def summarize(bt: pd.DataFrame) -> dict:
    if bt.empty:
        return {"n": 0}
    bt = bt.copy()
    bt["premium"] = bt["price"] - bt["final"]
    bt["edge"] = bt["fair"] - bt["price"]
    bt["outcome"] = bt["final"] - bt["price"]
    bt["bucket"] = pd.cut(bt["days_to_end"], [-1, 30, 90, 180, 10000],
                          labels=["≤30 天", "31–90 天", "91–180 天", ">180 天"])
    prem = [{"bucket": str(b), "n": int(len(g)), "mean": round(float(g["premium"].mean()), 2),
             "median": round(float(g["premium"].median()), 2)}
            for b, g in bt.groupby("bucket", observed=True)]
    corr = float(np.corrcoef(bt["edge"], bt["outcome"])[0, 1]) if len(bt) > 5 else None
    nv = bt.dropna(subset=["naive"])
    corr_naive = (float(np.corrcoef(nv["naive"] - nv["price"], nv["outcome"])[0, 1])
                  if len(nv) > 5 else None)
    tr = bt[bt["side"] != 0]
    out = {"n": int(len(bt)), "contracts": int(bt[["region", "contract", "product"]].drop_duplicates().shape[0]),
           "premium": prem, "corr_edge_outcome": None if corr is None else round(corr, 3),
           "corr_naive_outcome": None if corr_naive is None else round(corr_naive, 3),
           "mae_fair": round(float((bt["fair"] - bt["final"]).abs().mean()), 2),
           "mae_market": round(float((bt["price"] - bt["final"]).abs().mean()), 2),
           "trades": int(len(tr))}
    if len(tr):
        g = tr.groupby(["region", "contract", "product"])["pnl_mwh"].mean()   # 按合约聚类：同一合约的多次信号高度相关
        se = float(g.std(ddof=1) / np.sqrt(len(g))) if len(g) > 1 else None
        out.update({"hit": round(float((tr["pnl_mwh"] > 0).mean()), 3),
                    "pnl_mwh": round(float(tr["pnl_mwh"].mean()), 2),
                    "clusters": int(len(g)),
                    "t_cluster": None if not se else round(float(g.mean() / se), 2),
                    "buy": int((tr["side"] > 0).sum()), "sell": int((tr["side"] < 0).sum()),
                    "pnl_buy": round(float(tr.loc[tr["side"] > 0, "pnl_mwh"].mean()), 2) if (tr["side"] > 0).any() else None,
                    "pnl_sell": round(float(tr.loc[tr["side"] < 0, "pnl_mwh"].mean()), 2) if (tr["side"] < 0).any() else None})
    return out


def render(sm: dict) -> str:
    import trader as T
    esc = html.escape
    if not sm.get("n"):
        body = "<p class='lead2'>历史文件里没有能回测的行（合约要已经交割完）。</p>"
    else:
        prem = "".join(f"<tr><td>{esc(p['bucket'])}</td><td class='num'>{p['n']}</td>"
                       f"<td class='num'>{T._money(p['mean'], True, 2)}</td><td class='num'>{T._money(p['median'], True, 2)}</td></tr>"
                       for p in sm["premium"])
        tr = ("<p class='lead2'>没有达到门槛的信号。</p>" if not sm.get("trades") else
              f"""<div class="how"><div><b>信号数</b> {sm['trades']}（买 {sm['buy']} / 卖 {sm['sell']}），
              {sm['clusters']} 个合约</div><div><b>命中率</b> {sm['hit']:.0%}</div>
              <div><b>每 MWh 平均盈亏</b> {T._money(sm['pnl_mwh'], True, 2)}（买 {T._money(sm.get('pnl_buy'), True, 2)}，
              卖 {T._money(sm.get('pnl_sell'), True, 2)}）</div>
              <div><b>按合约聚类的 t 值</b> {sm.get('t_cluster') if sm.get('t_cluster') is not None else '—'}
              （|t| &lt; 2 就和运气分不开）</div></div>""")
        body = f"""<h2>1. 风险溢价：期货价 − 最终结算价</h2>
<p class="lead2">正数 = 市场价平均高于最后实际结算。这部分是对冲方付的保险费；如果它占了大头，
"公允价值低于市场价就卖"赚的是溢价，不是我们的信息。</p>
<div class="card"><div class="scroll"><table class="act"><thead><tr><th>距离交割</th><th class="num">站位数</th>
<th class="num">平均</th><th class="num">中位数</th></tr></thead><tbody>{prem}</tbody></table></div></div>
<h2>2. 公允价值有没有信息</h2>
<div class="how"><div><b>优势和结果的相关</b> {sm['corr_edge_outcome']}（我们的公允价值）
 vs {sm['corr_naive_outcome']}（朴素：已结算均价）</div>
<div><b>和最终结算的平均差</b> 我们 {T._money(sm['mae_fair'], nd=2)} vs 市场价 {T._money(sm['mae_market'], nd=2)}
（市场价更准就说明市场已经知道我们知道的东西）</div></div>
<h2>3. 规则回测（每个信号 1 MW 持有到交割，扣费）</h2>{tr}"""
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>期货信号回测</title>
<style>{T.CSS}</style></head><body><div class="wrap">
<div class="top"><div class="brand">期货信号回测（本机）</div><nav class="nav"><a href="desk.html">私有交易台</a></nav></div>
<div class="asof">{sm.get('n', 0)} 个站位、{sm.get('contracts', 0)} 个合约 · 数据来自你提供的 private/futures_history.csv · 不提交、不发布</div>
{body}
<footer>站位日的公允价值只用当天及以前的现货；区间放宽系数只用当天以前交割完的合约期估。</footer>
</div></body></html>"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="期货信号回测（本机私有）")
    ap.add_argument("--every", type=int, default=5, help="每隔几个交易日站一次位（默认 5 ≈ 每周）")
    a = ap.parse_args(argv)
    if not os.path.exists(HIST):
        print(f"没有 {HIST}。格式：date,region,contract,product,settle（见文件开头说明）")
        return 1
    hist = pd.read_csv(HIST)
    import store
    con = store.connect()
    mats = {r: F.day_matrix(F.load_prices(con, r)) for r in sorted(hist["region"].unique()) if r in F.REGIONS}
    con.close()
    risk = S.load_risk()
    bt = backtest(mats, hist, risk, every=a.every)
    sm = summarize(bt)
    os.makedirs(S.PRIV, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(sm, f, ensure_ascii=False, indent=1)
    with open(OUT_HTML, "w") as f:
        f.write(render(sm))
    print(json.dumps(sm, ensure_ascii=False, indent=1)[:1500])
    print(f"→ {OUT_HTML}（只在本机）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

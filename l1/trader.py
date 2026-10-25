"""
交易台首页 —— 给交易员看的那一页。
================================
只回答交易员每天要问的三件事，按这个顺序：

  1. 天气预报现在值不值得进决策？           → 一句话 + 一个数
  2. 每个区该用哪个模型排电池、一天赚多少？   → 一张表
  3. 证据：加天气每天多赚还是少赚、累计曲线   → 两张图

统计检验、误差解剖、特征取舍这些"为什么"，全部放到第二页 method.html。
这一页不出现 rMAE、GW p 这类术语。

图表规范（dataviz）：
  · 三条模型线用验证过的分类色 1–3（蓝/橙/青，明暗两套都过了 CVD 校验），
    基线"一周前同期"用中性灰退到背景；
  · 状态（用 / 不用 / 亏钱 / 样本不足）用保留的状态色，并且永远带图标 + 文字；
  · 数字和标签用文字色，不用系列色；每张图都有表格视图；悬停有提示；
  · 明暗两种主题各自取色，不是自动反色。
"""
from __future__ import annotations

import html
import json
import math

import numpy as np

import desk as D

MODEL_NAME = {
    "B0 naive": "一周前同期",
    "B1 自回归": "价格模型",
    "B2 +官方": "价格 + AEMO",
    "B3 +天气": "价格 + AEMO + 天气",
    "AEMO 直用": "AEMO 预调度直接用",
    "B3 选特征": "价格 + AEMO + 挑过的天气",
    "完美预知": "完美预知",
    "不做": "这个区不做",
    "B2 混合": "价格 + AEMO（AEMO 日内形状）",
    "B2 +供需": "价格 + AEMO + 供需",
}
MODEL_HINT = {
    "B0 naive": "直接照抄上周同一天、同一小时的价格",
    "B1 自回归": "只看最近的电价",
    "B2 +官方": "最近电价 + AEMO 官方预调度",
    "B3 +天气": "最近电价 + AEMO 预调度 + 天气预报",
    "AEMO 直用": "不建模型，直接按官方预测排",
    "B3 选特征": "天气变量只在训练段里挑",
    "不做": "最好的预测扣完成本也不显著赚钱",
    "B2 混合": "日均水平用我们的模型，日内高低照 AEMO 预调度",
    "B2 +供需": "再加预调度里的需求、可用发电、风光可用出力",
}
SERIES = ["B1 自回归", "B2 +官方", "B3 +天气"]           # 分类色 1–3
STATUS = {                                               # (类名, 图标, 文字)
    "yes": ("good", "✓", "用"),
    "no": ("neutral", "—", "不用"),
    "loss": ("critical", "✕", "亏钱"),
    "none": ("warning", "?", "样本不足"),
}


# ------------------------------------------------------------------ 数据整理
def _rows(r: dict) -> dict:
    ec = r.get("economics") or {}
    return {x["tier"]: x for x in ec.get("rows", [])}


def weather_ref(r: dict) -> str:
    return "B2 +官方" if r.get("has_official") else "B1 自回归"


def weather_state(r: dict) -> str:
    """用 / 不用 / 亏钱 / 样本不足 —— 和 desk.verdict 同一套门槛，只是多分出"亏钱"。"""
    lvl, _, _ = D.verdict(r)
    if lvl == "yes":
        return "yes"
    if lvl == "none":
        return "none"
    w = _rows(r).get("B3 +天气") or {}
    ci = w.get("gain_ci")
    if ci and ci[1] < 0:
        return "loss"
    return "no"


def _ci_state(ci, days) -> str:
    if days is not None and days < D.MIN_TEST_DAYS:
        return "none"
    if not ci:
        return "none"
    return "yes" if ci[0] > 0 else "loss" if ci[1] < 0 else "no"


def region_summary(name: str, r: dict) -> dict:
    ec = r.get("economics") or {}
    rows = _rows(r)
    rec = ec.get("recommended") or ("B0 naive" if rows else None)
    w = rows.get("B3 +天气") or {}
    bt = r.get("bias") or {}
    brows = {x["tier"]: x for x in bt.get("rows", [])}
    bw = brows.get("B3 +天气") or {}
    bref = brows.get(bt.get("ref")) or {}
    en = ec.get("nested") or {}
    return {
        "b_ok": bt.get("status") == "ok",
        "b_days": bt.get("days"),
        "b_ref": bt.get("ref"),
        "b_ref_daily": bref.get("daily"), "b_ref_ci": bref.get("ci"), "b_ref_hit": bref.get("hit"),
        "b_short": (brows.get("做空 AEMO") or {}).get("daily"),
        "b_ref_vs_short": bref.get("vs_short"), "b_ref_vs_short_ci": bref.get("vs_short_ci"),
        "b_gain": bw.get("gain"), "b_ci": bw.get("gain_ci"), "b_trim": bw.get("gain_trim"),
        "b_hit": bw.get("hit"),
        "b_state": _ci_state(bw.get("gain_ci"), bt.get("days")) if bt.get("status") == "ok" else "none",
        "n_gain": en.get("gain"), "n_ci": en.get("gain_ci"),
        "n_state": _ci_state(en.get("gain_ci"), ec.get("days")) if en else "none",
        "n_freq": (r.get("nested") or {}).get("freq"),
        "region": name,
        "ok": r.get("status") == "ok" and ec.get("status") == "ok",
        "state": weather_state(r) if r.get("status") == "ok" else "none",
        "rec": rec,
        "rec_daily": (rows.get(rec) or {}).get("daily"),
        "rec_capture": (rows.get(rec) or {}).get("capture"),
        "perfect": ec.get("perfect_daily"),
        "w_gain": w.get("gain"),
        "w_ci": w.get("gain_ci"),
        "w_ref": weather_ref(r),
        "days": ec.get("days"),
        "daily": ec.get("daily"),
        "has_official": bool(r.get("has_official")),
        "reason": plain_reason(r),
    }


def plain_reason(r: dict) -> str:
    """把四道门槛翻译成交易员的话：准不准、准得是不是运气、钱多没多。不出现 rMAE / p 值。"""
    if r.get("status") != "ok":
        return D.verdict(r)[2]
    b3 = next((x for x in r.get("tiers", []) if x["tier"] == "B3 +天气"), None)
    if not b3 or "gain_rmae" not in b3:
        return "天气这一层没跑出可比结果。"
    w = _rows(r).get("B3 +天气") or {}
    ci, g = w.get("gain_ci"), w.get("gain")
    nd = D.test_days(r) or 0
    p = b3.get("gw_p")
    more = b3["gain_rmae"] > 0
    sig = more and p is not None and p <= 0.05
    pct = abs(b3["gain_rmae"]) * 100
    if not more:
        acc = "加天气后价格预测没有更准"
    elif sig:
        acc = f"加天气后价格预测准了约 {pct:.1f}%（不是运气）"
    else:
        acc = f"加天气后价格预测准了约 {pct:.1f}%，但和运气分不开"
    if ci is None or g is None:
        money = "收益增量算不出区间"
    elif ci[0] > 0:
        money = f"电池每天稳定多赚 {_money(g, nd=2)}"
    elif ci[1] < 0:
        money = f"但排电池每天反而少赚 {_money(-g, nd=2)}，而且不是运气"
    else:
        money = f"电池收益 {_money(g, True, 2)}/天，和 0 分不开"
    out = f"{acc}；{money}。"
    if nd < D.MIN_TEST_DAYS:
        out = f"只有 {nd:.0f} 天样本（要 {D.MIN_TEST_DAYS} 天），先不下结论。" + out
    if not r.get("has_official"):
        out += "没有 AEMO 预调度，这是上界。"
    return out


# ------------------------------------------------------------------ 小工具
def _esc(x) -> str:
    return html.escape(str(x))


def _money(v, sign=False, nd=0) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    s = f"{abs(v):,.{nd}f}"
    if sign:
        return ("+" if v > 0 else "−" if v < 0 else "±") + "$" + s
    return ("−" if v < 0 else "") + "$" + s


def _nice(v: float) -> float:
    if v <= 0:
        return 1.0
    e = 10 ** math.floor(math.log10(v))
    for m in (1, 2, 2.5, 5, 10):
        if v <= m * e:
            return m * e
    return 10 * e


def _ticks(lo: float, hi: float, n: int = 4) -> list[float]:
    step = _nice((hi - lo) / max(n, 1))
    a = math.floor(lo / step) * step
    out, x = [], a
    while x <= hi + 1e-9:
        out.append(round(x, 6))
        x += step
    return out


def _compact(v: float) -> str:
    a = abs(v)
    s = f"{a / 1000:.0f}K" if a >= 10000 else f"{a / 1000:.1f}K" if a >= 1000 else f"{a:.0f}"
    s = s.replace(".0K", "K")
    return ("−" if v < 0 else "") + "$" + s


def chip(state: str) -> str:
    cls, icon, text = STATUS[state]
    return f'<span class="chip {cls}"><span aria-hidden="true">{icon}</span>{text}</span>'


# ------------------------------------------------------------------ 图 1：森林图
def forest(items: list[dict], key_mean: str, key_ci: str, title: str, sub: str,
           fig_id: str) -> str:
    """每个区一行：点 = 日均增量，须 = 95% 区间，0 线是"没区别"。
    用 HTML 网格而不是一整张 SVG：窄屏上字号不跟着缩。
    区间太宽的行在边缘截断并画箭头，数值照写在右侧，不靠图读数。"""
    rows = [x for x in items if x.get(key_mean) is not None and x.get(key_ci)]
    if not rows:
        return ""
    bounds = sorted(abs(v) for x in rows for v in x[key_ci])
    q = bounds[int(0.75 * (len(bounds) - 1))]
    # 刻度按多数区的区间宽度定，个别极端区（点或须超出）截在边上画箭头，数值照写在右侧
    ms = sorted(abs(x[key_mean]) for x in rows)
    lim = _nice(max(q * 1.5, ms[int(0.75 * (len(ms) - 1))] * 1.2, 2.0))
    P = lambda v: (max(-lim, min(lim, v)) + lim) / (2 * lim) * 100  # noqa: E731
    ticks = "".join(
        f'<span class="ft" style="left:{P(tv):.2f}%">{_compact(tv) if tv else "$0"}</span>'
        for tv in _ticks(-lim, lim, 4))
    grid = "".join(f'<i class="fg" style="left:{P(tv):.2f}%"></i>' for tv in _ticks(-lim, lim, 4))
    out = []
    for x in rows:
        lo, hi = x[key_ci]
        m = x[key_mean]
        st = "good" if lo > 0 else "critical" if hi < 0 else "neutral"
        tip = (f"<b>{_esc(x['region'])}</b><br>日均 {_money(m, True, 2)}<br>"
               f"95% 区间 [{_money(lo, True, 1)}, {_money(hi, True, 1)}]")
        arrows = ""
        if lo < -lim:
            arrows += '<i class="fa l"></i>'
        if hi > lim:
            arrows += '<i class="fa r"></i>'
        out.append(f"""
    <div class="fr" tabindex="0" data-tip="{_esc(tip)}">
      <span class="fl">{_esc(x['region'])}</span>
      <span class="ftrack {st}">{grid}<i class="f0" style="left:{P(0):.2f}%"></i>
        <i class="fw" style="left:{P(lo):.2f}%;width:{max(P(hi) - P(lo), 0.4):.2f}%">{arrows}</i>
        <i class="fd{' clip' if abs(m) > lim else ''}" style="left:{P(m):.2f}%"></i></span>
      <span class="fv"><b>{_money(m, True, 2)}</b><small>[{_money(lo, True, 1)}, {_money(hi, True, 1)}]</small></span>
    </div>""")
    table = "".join(
        f"<tr><td>{_esc(x['region'])}</td><td class='num'>{_money(x[key_mean], True, 2)}</td>"
        f"<td class='num'>[{_money(x[key_ci][0], True, 1)}, {_money(x[key_ci][1], True, 1)}]</td></tr>"
        for x in rows)
    return f"""
<figure class="fig" id="{fig_id}">
  <figcaption><h3>{title}</h3><p>{sub}</p></figcaption>
  <div class="forest" role="img" aria-label="{_esc(title)}">
    <div class="fr fhead"><span class="fl"></span>
      <span class="ftrack"><span class="fdir l">← 少赚</span><span class="fdir r">多赚 →</span></span>
      <span class="fv"></span></div>
    {''.join(out)}
    <div class="fr faxis"><span class="fl"></span><span class="ftrack">{ticks}</span><span class="fv"></span></div>
  </div>
  <details class="tv"><summary>表格视图</summary>
    <table class="dt"><thead><tr><th>区</th><th class="num">日均</th><th class="num">95% 区间</th></tr></thead>
    <tbody>{table}</tbody></table></details>
</figure>"""


# ------------------------------------------------------------------ 图 2：累计收益小多图
def cumulative(items: list[dict], decide_at: str = "13:30", mw: float = 1, mwh: float = 2,
               base: str = "B0 naive", title: str | None = None, sub: str | None = None,
               fig_id: str = "fig-cum") -> str:
    panels, tables = [], []
    for x in items:
        dl = x.get("daily")
        if not dl or not dl.get("dates"):
            continue
        dates = dl["dates"]
        keys = [base] + [k for k in SERIES if k in dl]
        cum = {}
        for k in keys:
            acc, out = 0.0, []
            for v in dl[k]:
                acc += v
                out.append(round(acc, 1))
            cum[k] = out
        allv = [v for k in keys for v in cum[k]] + [0.0]
        lo, hi = min(allv), max(allv)
        ticks = _ticks(lo, hi, 4)
        lo, hi = min(ticks[0], lo), max(ticks[-1], hi)
        W, H, L, R, T, B = 500, 250, 52, 16, 14, 28
        pw, ph = W - L - R, H - T - B
        n = len(dates)
        X = lambda i: L + (i / max(n - 1, 1)) * pw          # noqa: E731
        Y = lambda v: T + (1 - (v - lo) / (hi - lo or 1)) * ph  # noqa: E731
        g = []
        for tv in ticks:
            g.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{Y(tv):.1f}" y2="{Y(tv):.1f}"/>')
            g.append(f'<text class="tick" x="{L - 6}" y="{Y(tv) + 4:.1f}" text-anchor="end">{_compact(tv)}</text>')
        if lo < 0 < hi:
            g.append(f'<line class="zero" x1="{L}" x2="{W - R}" y1="{Y(0):.1f}" y2="{Y(0):.1f}"/>')
        for i, anc in ((0, "start"), (n // 2, "middle"), (n - 1, "end")):
            g.append(f'<text class="tick" x="{X(i):.1f}" y="{H - 8}" text-anchor="{anc}">{dates[i][2:]}</text>')
        for k in keys:
            pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(cum[k]))
            cls = "base" if k == base else f"s{SERIES.index(k) + 1}"
            g.append(f'<polyline class="ln {cls}" points="{pts}"/>')
        rec = x["rec"]
        if rec in cum:
            cls = "base" if rec == base else f"s{SERIES.index(rec) + 1}"
            g.append(f'<circle class="end {cls}" cx="{X(n - 1):.1f}" cy="{Y(cum[rec][-1]):.1f}" r="4"/>')
        g.append(f'<line class="cross" x1="0" x2="0" y1="{T}" y2="{T + ph}" style="display:none"/>')
        g.append(f'<rect class="hitarea" x="{L}" y="{T}" width="{pw}" height="{ph}"/>')
        payload = {"dates": dates, "series": [[MODEL_NAME[k], cum[k], k] for k in keys],
                   "L": L, "pw": pw}
        diff = None
        ref = x["w_ref"]
        if ref in cum and "B3 +天气" in cum:
            diff = cum["B3 +天气"][-1] - cum[ref][-1]
        head = (f"加天气比 {MODEL_NAME[ref]} 累计 <b>{_money(diff, True)}</b>"
                if diff is not None else "")
        panels.append(f"""
  <div class="panel">
    <div class="phead"><span class="pname">{_esc(x['region'])}</span>{chip(x['state'])}
      <span class="pnote">{head}</span></div>
    <svg class="cum" viewBox="0 0 {W} {H}" role="img" data-cum='{_esc(json.dumps(payload, ensure_ascii=False))}'
         aria-label="{_esc(x['region'])} 累计收益">{''.join(g)}</svg>
  </div>""")
        tables.append("<tr><td>{}</td>{}</tr>".format(
            _esc(x["region"]),
            "".join(f"<td class='num'>{_money(cum[k][-1])}</td>" if k in cum else "<td>—</td>"
                    for k in [base] + SERIES)))
    if not panels:
        return ""
    legend = "".join(
        f'<span class="key"><i class="sw {("base" if k == base else "s" + str(SERIES.index(k) + 1))}"></i>'
        f'{MODEL_NAME[k]}</span>' for k in [base] + SERIES)
    head_cells = "".join(f"<th class='num'>{MODEL_NAME[k]}</th>" for k in [base] + SERIES)
    return f"""
<figure class="fig" id="{fig_id}">
  <figcaption><h3>{title or f"累计收益：一块 {mw:g} MW / {mwh:g} MWh 电池"}</h3>
    <p>{sub or f"每天 {decide_at} 用各模型的预测排好次日计划，按实际电价结算后的累计美元。线越高越好；圆点是该区建议使用的模型。"}</p></figcaption>
  <div class="legend">{legend}</div>
  <div class="grid5">{''.join(panels)}</div>
  <details class="tv"><summary>表格视图（期末累计）</summary>
    <table class="dt"><thead><tr><th>区</th>{head_cells}</tr></thead>
    <tbody>{''.join(tables)}</tbody></table></details>
</figure>"""


# ------------------------------------------------------------------ 系统状态
def status_strip(data: dict) -> str:
    hl = data.get("health")
    pp = data.get("paper") or {}
    if not hl:
        return ""
    lv = hl.get("level", "warn")
    st = {"ok": ("good", "✓", "数据正常"), "warn": ("warning", "!", "数据有提醒"),
          "fail": ("critical", "✕", "数据有问题")}[lv]
    chipx = f'<span class="chip {st[0]}"><span aria-hidden="true">{st[1]}</span>{st[2]}</span>'
    lags = [v.get("price_lag_h") for v in hl.get("regions", {}).values() if v.get("price_lag_h") is not None]
    bits = [f"体检 {_esc(hl.get('checked_at', ''))} UTC" + ("（云上每日任务）" if hl.get("ci") else "")]
    if lags:
        bits.append(f"现货价最新落后 {min(lags):.0f}–{max(lags):.0f} 小时")
    lt = pp.get("latest") or {}
    if lt:
        bits.append(f"最新信号 {_esc(lt.get('target_day'))}" + ("（late）" if lt.get("late") else ""))
    bits += [_esc(n) for n in hl.get("notes", [])]
    bad = [f"{k}: 预调度 {v['predispatch_hours']} h / 天气 {v['weather_hours']} h"
           for k, v in hl.get("regions", {}).items() if v.get("level") != "ok"]
    if bad:
        bits.append("明天输入不全 —— " + "；".join(_esc(b) for b in bad))
    return f'<div class="status">{chipx}<span>{" · ".join(bits)}</span></div>'


# ------------------------------------------------------------------ 顶部
def hero_tiles(fut: dict | None, verdict_big: str, verdict_say: str) -> str:
    tiles = []
    for reg, r in ((fut or {}).get("regions") or {}).items():
        c = next((c for c in r.get("contracts", []) if c["tag"] == "当季"), None)
        fv = (c or {}).get("products", {}).get("base")
        if not fv:
            continue
        tiles.append(f"""<div class="tile"><div class="tl">{_esc(reg)} · {_esc(c['label'])} 基荷公允</div>
<div class="tv">{_money(fv['mean'], nd=2)}</div><div class="tn">50% {_money(fv['p25'])}–{_money(fv['p75'])} · 已结算 {fv['settled_share']:.0%}</div></div>""")
    if not tiles:
        return f"""<section class="hero"><div><p class="q">扣完交易成本，加天气后显著多赚的（区 × 产品）</p>
<div class="big">{verdict_big}</div></div><p class="say">{verdict_say}</p></section>"""
    return f"""<section class="hero hero2">
<div class="tiles">{''.join(tiles)}</div>
<p class="say">拿你屏幕上的当季基荷价和上面比：低于 50% 区间偏便宜、高于偏贵。买卖信号、数量和风险在本机运行
<code>python l1/signals.py</code> 生成（市场价不出你的电脑）。<br>
<span class="dim">天气对能下单的东西：扣完成本后显著多赚的（区 × 产品）{verdict_big.replace('<small>/', ' / ').replace('</small>', '')}。</span></p>
</section>"""


# ------------------------------------------------------------------ 期货：公允价值
FUT_PROD = [("base", "基荷"), ("peak", "峰荷"), ("cap300", "$300 cap*")]


def _range_row(label: str, fv: dict, lo: float, hi: float, note: str) -> str:
    P = lambda v: (min(max(v, lo), hi) - lo) / ((hi - lo) or 1) * 100   # noqa: E731
    tip = (f"<b>{_esc(label)}</b><br>公允 {_money(fv['mean'], nd=2)}<br>50% 区间 {_money(fv['p25'], nd=1)} – {_money(fv['p75'], nd=1)}"
           f"<br>90% 区间 {_money(fv['p5'], nd=1)} – {_money(fv['p95'], nd=1)}<br>已结算 {fv['settled_share']:.0%}，剩 {fv['days_left']} 天")
    return f"""<div class="fr" tabindex="0" data-tip="{_esc(tip)}">
  <span class="fl fl2">{_esc(label)}</span>
  <span class="ftrack rng"><i class="r90" style="left:{P(fv['p5']):.1f}%;width:{max(P(fv['p95']) - P(fv['p5']), .5):.1f}%"></i>
    <i class="r50" style="left:{P(fv['p25']):.1f}%;width:{max(P(fv['p75']) - P(fv['p25']), .5):.1f}%"></i>
    <i class="fd" style="left:{P(fv['mean']):.1f}%"></i></span>
  <span class="fv"><b>{_money(fv['mean'], nd=2)}</b><small>{_esc(note)}</small></span></div>"""


def futures_block(fut: dict | None) -> str:
    if not fut or not fut.get("regions"):
        return ""
    panels, rows = [], []
    for reg, r in fut["regions"].items():
        cs = [c for c in r.get("contracts", []) if "base" in c.get("products", {})]
        if not cs:
            continue
        vals = [c["products"]["base"][k] for c in cs for k in ("p5", "p95")]
        lo, hi = _nice_floor(min(vals)), _nice_ceil(max(vals))
        cal = r.get("calibration") or {}
        cm = cal.get("month_base") or {}
        cq = cal.get("quarter_base") or {}
        body = "".join(_range_row(f"{c['label']}", c["products"]["base"], lo, hi,
                                  f"{_money(c['products']['base']['p5'])}–{_money(c['products']['base']['p95'])}")
                       for c in cs)
        cov = []
        if cm.get("oos_cov90") is not None:
            cov.append(f"月度 {cm['oos_cov90']:.0%}")
        if cq.get("oos_cov90") is not None:
            cov.append(f"季度 {cq['oos_cov90']:.0%}")
        panels.append(f"""<div class="panel"><div class="phead"><span class="pname">{_esc(reg)}</span>
  <span class="pnote">基荷 · 截至 {_esc(r['asof'])} · 90% 区间历史覆盖 {'、'.join(cov) or '—'}</span></div>
  <div class="forest">{body}
  <div class="fr faxis"><span class="fl fl2"></span><span class="ftrack"><span class="ft" style="left:0%">{_money(lo)}</span>
  <span class="ft" style="left:100%">{_money(hi)}</span></span><span class="fv"></span></div></div></div>""")
        for c in r.get("contracts", []):
            cells = ""
            for k, _ in FUT_PROD:
                fv = c["products"].get(k)
                cells += ("<td>—</td>" if not fv else
                          f"<td class='num' data-l='{dict(FUT_PROD)[k]}'><span class='big2'>{_money(fv['mean'], nd=2)}</span>"
                          f"<span class='hint'>50% {_money(fv['p25'])}–{_money(fv['p75'])} · 90% {_money(fv['p5'])}–{_money(fv['p95'])}</span></td>")
            b = c["products"].get("base") or {}
            rows.append(f"""<tr><td class="rg">{_esc(reg)}</td><td data-l="合约"><b>{_esc(c['label'])}</b><span class="hint">{_esc(c['tag'])} · 剩 {b.get('days_left', '—')} 天 · 已结算 {b.get('settled_share', 0):.0%}</span></td>
{cells}<td class="num" data-l="每 MW 单日 95% 亏损">{_money(b.get('risk_long'))}<span class="hint">做空 {_money(b.get('risk_short'))}</span></td></tr>""")
    return f"""
<h2>期货公允价值</h2>
<p class="lead2">ASX 的 NEM 期货按合约期内现货均价结算。这里只用 AEMO 公开的现货价算：已经过去的小时按实际价，
剩下的从最近 {fut.get('pool_days', 90)} 天里整块抽样 {fut.get('n_sims', 4000):,} 次，区间再按历史校准放宽。
<b>页面上不放 ASX 的价格</b>（ASX Energy 网站条款禁止转载）——拿你自己屏幕上的价格对照：
市场价低于 50% 区间下沿 → 偏便宜；高于上沿 → 偏贵。要买卖信号和仓位，在本机运行 <code>python l1/signals.py</code>。</p>
<div class="grid5">{''.join(panels)}</div>
<div class="card" style="margin-top:16px"><div class="scroll"><table class="act">
<thead><tr><th>区</th><th>合约</th>{''.join(f'<th class="num">{n} 公允</th>' for _, n in FUT_PROD)}<th class="num">基荷每 MW 单日 95% 亏损（做多）</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div></div>
<p class="lead2">* $300 cap 按小时均价算，5 分钟尖峰被平均掉，是<b>下限</b>。"单日 95% 亏损"= 持有到交割时，
95% 的交易日里单日最多亏多少（按开仓价 = 公允价、每天一天现货结算进来算；不含期货价本身的盯市波动）。</p>"""


def _nice_floor(v: float) -> float:
    st = _nice(max(abs(v), 1) / 4)
    return float(np.floor(v / st) * st)


def _nice_ceil(v: float) -> float:
    st = _nice(max(abs(v), 1) / 4)
    return float(np.ceil(v / st) * st)


def _spike_note(regs: dict) -> str:
    """尖峰概率的历史检验结果：不比"历史同小时频率"好的区，照实提醒。"""
    worse = [k for k, r in regs.items()
             if (r.get("prob") or {}).get("status") == "ok"
             and r["prob"]["ours"]["brier"] > r["prob"]["brier_clim"] * 1.05]
    if not worse:
        return ""
    return (f"<b>注意：</b>历史检验里，{'、'.join(worse)} 我们的尖峰概率还不如\"历史同小时频率\"准"
            f"（尖峰太少，误差分布估不稳），这些区先看最后一列的历史频率。")


def spot_block(fut: dict | None, regs: dict | None = None) -> str:
    if not fut:
        return ""
    trs = []
    for reg, r in fut.get("regions", {}).items():
        sp = r.get("spot") or {}
        if not sp or "error" in sp:
            continue
        if sp.get("stale"):
            trs.append(f"""<tr><td class="rg">{_esc(reg)}</td><td data-l="日期">{_esc(sp['day'])}</td>
<td colspan="4" class="why">预调度只更新到 {_esc(sp.get('pd_last') or '—')}（NEM），还没覆盖明天——每日任务跑过之后才有。</td>
<td class="num" data-l="本月历史尖峰日">{'—' if sp.get('spike_freq') is None else f"{sp['spike_freq']:.0%}"}</td></tr>""")
            continue
        ours = ("—" if sp.get("ours_avg") is None else
                f"{_money(sp['ours_avg'])}<span class='hint'>最高 {_money(sp.get('ours_max'))}</span>")
        spk = ("—" if sp.get("p_spike_max") is None else
               f"{sp['p_spike_max']:.0%}<span class='hint'>{sp.get('spike_hours', 0)} 个小时 &gt; 20%</span>")
        trs.append(f"""<tr><td class="rg">{_esc(reg)}</td><td data-l="日期">{_esc(sp['day'])}</td>
<td class="num" data-l="AEMO 预调度均价">{_money(sp['aemo_avg'])}<span class="hint">最高 {_money(sp['aemo_max'])}</span></td>
<td class="num" data-l="我们的预测均价">{ours}</td>
<td class="num" data-l="预调度 > $300 的小时">{sp['aemo_hours_over_300']}</td>
<td class="num" data-l="我们的尖峰概率（最高的小时）">{spk}</td>
<td class="num" data-l="本月历史尖峰日">{'—' if sp.get('spike_freq') is None else f"{sp['spike_freq']:.0%}"}<span class="hint">{sp.get('spike_days', '')} 天里</span></td></tr>""")
    if not trs:
        return ""
    return f"""
<h2>明日现货展望</h2>
<p class="lead2">AEMO 预调度取最新一版（前一天的预调度系统性偏高，均价要打折看）；"我们的预测"是纸面交易每天 13:30 前出的价格 + AEMO 模型（没有就显示 —）。
"我们的尖峰概率"= 明天价格超过 $300 的概率里最高的那个小时（点预测 + 过去 90 天同一小时的样本外误差），
历史检验见第二页"概率预测"。"本月历史尖峰日"= 历史上同一个月里，日内最高小时均价超过 $300 的天数占比。
{_spike_note(regs or {})}</p>
<div class="card"><div class="scroll"><table class="act"><thead><tr><th>区</th><th>日期</th><th class="num">AEMO 预调度均价</th>
<th class="num">我们的预测均价</th><th class="num">预调度 &gt; $300 的小时</th><th class="num">我们的尖峰概率</th><th class="num">本月历史尖峰日</th></tr></thead>
<tbody>{''.join(trs)}</tbody></table></div></div>"""


def weather_block(fut: dict | None) -> str:
    if not fut:
        return ""
    trs = []
    for reg, r in fut.get("regions", {}).items():
        w = r.get("weather") or {}
        if not w or "error" in w or "wind" not in w:
            continue
        wz, tz = w["wind"]["z"], w["temp"]["z"]
        wtxt = "偏弱" if wz < -0.5 else "偏强" if wz > 0.5 else "正常"
        ttxt = "偏冷" if tz < -0.5 else "偏热" if tz > 0.5 else "正常"
        imp = w.get("implied")
        trs.append(f"""<tr><td class="rg">{_esc(reg)}</td><td data-l="区间">{_esc(w['from'][5:])} – {_esc(w['to'][5:])}</td>
<td class="num" data-l="100 米风速">{w['wind']['fc']:.1f} km/h<span class="hint">常态 {w['wind']['norm']:.1f} · {wtxt}（{wz:+.1f}σ）</span></td>
<td class="num" data-l="气温">{w['temp']['fc']:.1f} °C<span class="hint">常态 {w['temp']['norm']:.1f} · {ttxt}（{tz:+.1f}σ）</span></td>
<td class="num" data-l="历史上对应的日均价偏离">{_money(imp, True, 1)}<span class="hint">风每少 1 km/h {_money(-(w.get('sens_wind') or 0), True, 1)}</span></td></tr>""")
    if not trs:
        return ""
    return f"""
<h2>天气指标（未来几天）</h2>
<p class="lead2">ECMWF 最新一版预报相对同期常态（ERA5 同日 ±15 天）的偏离。最后一列是历史上同月份里这种偏离对应的日均价偏离——
<b>描述性的，不是预测</b>：前面的检验已经说明，加上天气并没有让能下单的东西显著多赚。把它当成"市场可能在交易什么"的背景信息。</p>
<div class="card"><div class="scroll"><table class="act"><thead><tr><th>区</th><th>区间</th><th class="num">100 米风速</th>
<th class="num">气温</th><th class="num">历史上对应的日均价偏离</th></tr></thead><tbody>{''.join(trs)}</tbody></table></div></div>"""


def risk_block(fut: dict | None, data: dict) -> str:
    items = []
    if fut:
        lasts = [r.get("last_hour", "")[:16] for r in fut.get("regions", {}).values()]
        if lasts:
            items.append(f"<div><b>数据新鲜度</b>：现货价最后一个小时 {_esc(min(lasts))}（NEM）。超过两天没更新就别用公允价值。</div>")
        covs = [((r.get("calibration") or {}).get("month_base") or {}).get("oos_cov90")
                for r in fut.get("regions", {}).values()]
        covs = [c for c in covs if c is not None]
        if covs:
            items.append(f"<div><b>区间可信度</b>：放宽后的 90% 区间在样本外盖住了 {min(covs):.0%}–{max(covs):.0%} 的实际结算价"
                         f"（月度基荷，各区）。低于 90% 说明还偏窄。</div>")
    items += [
        "<div><b>没有历史期货价</b>：公允价值的准确度检验过，但“市场价偏离公允价值就交易”这个规则<b>没有在历史期货价上回测过</b>。</div>",
        "<div><b>结构性变化</b>：抽样池只看最近 90 天；新机组投运、长时间检修、气价跳变、极端天气都可能让未来和最近不一样。</div>",
        "<div><b>尾部</b>：NEM 有市场价格上限（每 5 分钟一万多澳元/MWh）。卖出 cap、做空晚峰在尖峰日的亏损可以远超“单日 95% 亏损”。</div>",
        "<div><b>流动性</b>：月度和峰荷合约成交稀，bid/ask 价差可能吃掉全部优势。</div>",
    ]
    return f"""
<h2>风险</h2>
<div class="how">{''.join(items)}</div>"""


# ------------------------------------------------------------------ 能下单的产品
PRODUCT_KEYS = ["battery", "flexload", "peaker"]


def product_rows(regs: dict, key: str) -> list[dict]:
    out = []
    for name, r in regs.items():
        pr = (r.get("products") or {})
        it = (pr.get("items") or {}).get(key) if pr.get("status") == "ok" else None
        if not it:
            out.append({"region": name, "ok": False, "reason": "没有可比结果"})
            continue
        rows = {x["model"]: x for x in it["rows"]}
        w = rows.get("B3 +天气") or {}
        b2 = rows.get("B2 +官方") or {}
        rec = it.get("recommended")
        days = it.get("days") or 0
        st = _ci_state(w.get("vs_b2_ci"), days) if w else "none"
        dl = it.get("daily") or {}
        out.append({
            "region": name, "ok": True, "state": st, "rec": rec,
            "rec_daily": 0.0 if rec == "不做" else (rows.get(rec) or {}).get("daily"),
            "rec_capture": None if rec == "不做" else (rows.get(rec) or {}).get("capture"),
            "perfect": it.get("perfect"), "days": days,
            "w_gain": w.get("vs_b2"), "w_ci": w.get("vs_b2_ci"),
            "m_gain": b2.get("vs_aemo"), "m_ci": b2.get("vs_aemo_ci"),
            "aemo_daily": (rows.get("AEMO 直用") or {}).get("daily"),
            "daily": dl, "w_ref": "B2 +官方", "has_official": bool(r.get("has_official")),
        })
    return out


def _ci_txt(ci, nd=1) -> str:
    if not ci:
        return ""
    return f"[{_money(ci[0], True, nd)}, {_money(ci[1], True, nd)}]"


def products_block(regs: dict, decide_at: str) -> tuple[str, dict]:
    """三个能下单的产品，一个产品一个标签页：行动表 + 天气森林图 + 累计收益。"""
    tabs, panes, stats = [], [], {}
    first = True
    for key in PRODUCT_KEYS:
        rows = product_rows(regs, key)
        oks = [x for x in rows if x["ok"]]
        if not oks:
            continue
        it0 = next(((r.get("products") or {}).get("items") or {}).get(key)
                    for r in regs.values() if (r.get("products") or {}).get("status") == "ok")
        name, desc = it0["name"], it0["desc"]
        stats[key] = {"name": name, "yes": [x["region"] for x in oks if x["state"] == "yes"],
                      "loss": [x["region"] for x in oks if x["state"] == "loss"],
                      "model_yes": [x["region"] for x in oks if x["m_ci"] and x["m_ci"][0] > 0],
                      "recs": {x["region"]: x["rec"] for x in oks}}
        trs = []
        for x in rows:
            if not x["ok"]:
                trs.append(f"<tr><td class='rg'>{_esc(x['region'])}</td><td>{chip('none')}</td>"
                           f"<td colspan='5' class='why'>{_esc(x['reason'])}</td></tr>")
                continue
            cap = x["rec_capture"]
            trs.append(f"""<tr>
  <td class="rg">{_esc(x['region'])}</td>
  <td class="st">{chip(x['state'])}</td>
  <td class="mdl" data-l="用什么预测"><span class="model">{MODEL_NAME.get(x['rec'], '—')}</span><span class="hint">{MODEL_HINT.get(x['rec'], '')}</span></td>
  <td class="num" data-l="每天净赚"><span class="big2">{_money(x['rec_daily'], True)}</span><span class="hint">已扣成本</span></td>
  <td class="num" data-l="抓到上限的">{'—' if cap is None else f'{cap:.0%}'}<span class="hint">上限 {_money(x['perfect'])}</span></td>
  <td class="num" data-l="建模比直接用 AEMO 多"><span class="big2">{_money(x['m_gain'], True, 1)}</span><span class="hint">{_ci_txt(x['m_ci'])}</span></td>
  <td class="num" data-l="再加天气"><span class="big2">{_money(x['w_gain'], True, 1)}</span><span class="hint">{_ci_txt(x['w_ci'])}</span></td>
</tr>""")
        days = max(x["days"] for x in oks)
        fig = forest([dict(x, m=x["w_gain"], ci=x["w_ci"]) for x in oks if x["w_gain"] is not None],
                     "m", "ci", f"{name}：加上天气每天多赚还是少赚？",
                     f"点 = {days} 天的日均差额（相对价格 + AEMO 模型），横线 = 95% 区间。已扣交易成本。",
                     f"fig-w-{key}")
        cum = cumulative(oks, decide_at, base="AEMO 直用", fig_id=f"fig-cum-{key}",
                         title=f"{name}：累计净收益",
                         sub=f"灰线 = 不建模型、直接按 AEMO 预调度排；彩线 = 我们的三个模型。"
                             f"每天 {decide_at} 定次日计划，按实际电价结算，已扣交易成本。圆点 = 建议用的预测。")
        hid = "" if first else " hidden"
        tabs.append(f'<button class="tab{" on" if first else ""}" data-pane="pane-{key}" '
                    f'aria-selected="{"true" if first else "false"}" role="tab">{name}</button>')
        panes.append(f"""<div class="pane" id="pane-{key}" role="tabpanel"{hid}>
<p class="lead2"><b>{name}</b>：{_esc(desc)}。</p>
<div class="card"><div class="scroll"><table class="act">
<thead><tr><th>区</th><th>天气</th><th>用什么预测</th><th class="num">每天净赚</th><th class="num">抓到上限的</th>
<th class="num">建模比直接用 AEMO 多</th><th class="num">再加天气</th></tr></thead>
<tbody>{''.join(trs)}</tbody></table></div></div>
{fig}{cum}</div>""")
        first = False
    if not panes:
        return "", stats
    return f"""
<h2>有实物资产时：现货决策</h2>
<p class="lead2">ASX 没有日度或周度合约，"明天哪几个小时贵"只能通过实物资产按现货价变现。
下面三种都按 1 MW 算、<b>已扣交易成本</b>，和"不建模型、直接按 AEMO 预调度排"比。纯金融交易员可以跳过这一节。</p>
<div class="tabs" role="tablist">{''.join(tabs)}</div>
{''.join(panes)}""", stats


def costs_block(regs: dict) -> str:
    pr = next(((r.get("products") or {}) for r in regs.values()
               if (r.get("products") or {}).get("status") == "ok"), None)
    if not pr:
        return ""
    c, src = pr["costs"], pr.get("sources", {})
    bd_rows, pk_rows = [], []
    for name, r in regs.items():
        ss = ((r.get("products") or {}).get("sensitivity") or {})
        if ss.get("battery_degr"):
            bd_rows.append(f"<tr><td class='tier'>{_esc(name)}</td>" + "".join(
                f"<td class='num'>{_money(x['best_daily'], True)}<br><small>{_esc(MODEL_NAME.get(x['best'], x['best']))}</small></td>"
                for x in ss["battery_degr"]) + "</tr>")
        if ss.get("peaker_srmc"):
            pk_rows.append(f"<tr><td class='tier'>{_esc(name)}</td>" + "".join(
                f"<td class='num'>{_money(x['best_daily'], True)}<br><small>天气 {_money(x['w_gain'], True, 1)}</small></td>"
                for x in ss["peaker_srmc"]) + "</tr>")
    any_ss = next(((r.get("products") or {}).get("sensitivity") for r in regs.values()
                   if (r.get("products") or {}).get("sensitivity")), {})
    bd_head = "".join(f"<th class='num'>衰减 ${x['x']:g}</th>" for x in any_ss.get("battery_degr", []))
    pk_head = "".join(f"<th class='num'>边际成本 ${x['x']:g}</th>" for x in any_ss.get("peaker_srmc", []))
    return f"""
<h2>交易成本假设</h2>
<div class="how">
  <div><b>AEMO 市场费用</b> ${c['fee']:.2f}/MWh，买、卖两个方向都收。{_esc(src.get('fee', ''))}。</div>
  <div><b>储能</b>：往返效率 {c['eta']:.0%}，衰减 ${c['degr']:g}/MWh 放电量（下表做敏感性）。预计扣完成本不赚钱的日子不动。</div>
  <div><b>调峰机组</b>：边际成本 ${c['srmc']:g}/MWh（{_esc(src.get('gas', ''))}，热耗约 11.5 GJ/MWh，加可变运维），
    启动 ${c['start']:g}/MW/次，开机至少连续 {c['min_up']} 小时。快速启动、能实时报价的机组用不上日前预测，这里假设的是要提前一天定计划的机组。</div>
  <div><b>没算进去的</b>：自己的出力对价格的影响（规模放大会压价）、网损系数、网络费、辅助服务收入、融资成本。</div>
</div>
<div class="card"><div class="scroll"><table class="act sens">
<thead><tr><th>储能：最好的预测每天净赚</th>{bd_head}</tr></thead>
<tbody>{''.join(bd_rows)}</tbody></table></div></div>
<div class="card"><div class="scroll"><table class="act sens">
<thead><tr><th>调峰机组：最好的预测每天净赚</th>{pk_head}</tr></thead>
<tbody>{''.join(pk_rows)}</tbody></table></div></div>"""


# ------------------------------------------------------------------ 第二种用法
def bias_section(ok: list[dict], days: int) -> str:
    xs = [x for x in ok if x["b_ok"]]
    if not xs:
        return ""
    trs = []
    for x in xs:
        ci = x["b_ci"] or [None, None]
        vs = x["b_ref_vs_short_ci"] or [None, None]
        trs.append(f"""<tr>
  <td class="rg">{_esc(x['region'])}</td>
  <td class="st">{chip(x['b_state'])}</td>
  <td class="num" data-l="一直做空 AEMO"><span class="big2">{_money(x['b_short'], True)}</span>
      <span class="hint">/ 天 · 预调度偏高的口径差</span></td>
  <td class="num" data-l="{MODEL_NAME.get(x['b_ref'], '')} 比它多"><span class="big2">{_money(x['b_ref_vs_short'], True)}</span>
      <span class="hint">[{_money(vs[0], True)}, {_money(vs[1], True)}] · 命中 {'—' if x['b_ref_hit'] is None else f"{x['b_ref_hit']:.0%}"}</span></td>
  <td class="num" data-l="再加天气每天"><span class="big2">{_money(x['b_gain'], True, 1)}</span>
      <span class="hint">[{_money(ci[0], True, 0)}, {_money(ci[1], True, 0)}]</span></td>
  <td class="num" data-l="去掉尖峰日">{_money(x['b_trim'], True, 1)}
      <span class="hint">最好/最差各 2% 的日子不算</span></td>
</tr>""")
    fig = forest(
        [dict(x, m=x["b_gain"], ci=x["b_ci"]) for x in xs if x["b_gain"] is not None and x["b_ci"]],
        "m", "ci", "赌 AEMO 偏差时，加上天气每天多赚还是少赚？",
        f"点 = {days} 天里的日均差额（1 MW，相对不用天气的价格 + AEMO 模型），横线 = 95% 区间。"
        "空心点 / 箭头 = 超出刻度，看右侧数字。", "fig-bias")
    return f"""
<h2>信号检验：赌 AEMO 预调度的偏差（不能直接下单）</h2>
<p class="lead2">电池只吃<b>日内排序</b>（哪个小时最便宜、哪个最贵），而天气——尤其是风——主要移动的是
<b>整天的价格水平</b>。所以单独量一次水平信号：每个小时，我们的预测比 AEMO 预调度高就做多 1 MW、
低就做空 1 MW，按实际电价结算。NEM 没有日前市场，这不是一张能直接下的单；
它衡量的是"我们对 AEMO 的修正值多少钱"，现实里对应调整对冲头寸、报价，或在期货 / 互换上表达看法。</p>
<p class="lead2"><b>先看第一列。</b>前一天下午的预调度系统性偏高（按报价排序、不含之后的重新报价，尖峰常常不兑现），
所以什么模型都不用、一直做空它就"赚"很多——那是预调度的口径，不是信息，第二列起才是模型的本事。
金额按 1 MW 计，每天 24 MWh；头寸放大多少倍，数字就放大多少倍。</p>
<div class="card"><div class="scroll"><table class="act">
<thead><tr><th>区</th><th>天气</th><th class="num">一直做空 AEMO</th><th class="num">价格 + AEMO 模型比它多</th>
<th class="num">再加天气每天</th><th class="num">去掉尖峰日</th></tr></thead>
<tbody>{''.join(trs)}</tbody></table></div></div>
{fig}"""


# ------------------------------------------------------------------ 前推纸面交易
def paper_section(pp: dict | None, rb: dict | None) -> str:
    if not pp:
        return ""
    regs = pp.get("regions") or {}
    lt = pp.get("latest")
    focus = [k for k, v in ((rb or {}).get("regions") or {}).items()
             if (v.get("checks") or {}).get("holm")] or ["VIC1"]
    n_set = max((v.get("n", 0) for v in regs.values()), default=0)
    start = pp.get("start")
    if not start:
        return ("<h2>前推纸面交易（实盘检验）</h2><p class=\"lead2\">还没有开始。装好定时任务后，"
                "每天 13:05–13:29（NEM）自动给次日出信号、提交到 GitHub，第二天按实际电价结算，结果显示在这里。</p>")
    head = (f"从 {start} 开始，每天 13:30（NEM）之前自动给次日出信号并提交到 GitHub，"
            f"提交时间就是信号先于结果存在的证据。已结算 <b>{n_set}</b> 天。")
    if n_set < 20:
        head += " 样本还很少，先看方向，不看显著性；至少要攒到 60 天以上再下结论。"
    rows = []
    for k, v in regs.items():
        if not v.get("n"):
            continue
        rows.append(f"""<tr><td class="rg">{_esc(k)}</td>
  <td class="num" data-l="已结算">{v['n']} 天</td>
  <td class="num" data-l="价格+AEMO 比一直做空">{_money((v.get('bias_b2') or 0) - (v.get('short') or 0), True)}</td>
  <td class="num" data-l="加天气累计"><span class="big2">{_money(v.get('w_gain_total'), True)}</span>
    <span class="hint">每天 {_money(v.get('w_gain_per_day'), True, 1)} · 赢 {'—' if v.get('w_gain_win_days') is None else f"{v['w_gain_win_days']:.0%}"} 的天</span></td>
  <td class="num" data-l="电池（价格模型）">{_money(v.get('batt_b1'), True)}<span class="hint">上限 {_money(v.get('batt_perfect'))}</span></td>
</tr>""")
    table = "" if not rows else f"""<div class="card"><div class="scroll"><table class="act">
<thead><tr><th>区</th><th class="num">已结算</th><th class="num">价格+AEMO 比一直做空</th>
<th class="num">加天气累计（偏差交易）</th><th class="num">电池（价格模型）</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div></div>"""
    prow = []
    names = {"battery": "储能", "flexload": "可调负荷", "peaker": "调峰机组"}
    for k, v in regs.items():
        pv = v.get("products") or {}
        if not v.get("n") or not pv:
            continue
        cells = ""
        for key in PRODUCT_KEYS:
            x = pv.get(key) or {}
            b2, b3, ae = x.get("b2"), x.get("b3"), x.get("aemo")
            dw = None if b2 is None or b3 is None else b3 - b2
            dm = None if b2 is None or ae is None else b2 - ae
            cells += (f'<td class="num" data-l="{names[key]}">{_money(b2, True)}'
                      f'<span class="hint">比 AEMO 直用 {_money(dm, True)} · 加天气 {_money(dw, True)}</span></td>')
        prow.append(f'<tr><td class="rg">{_esc(k)}</td>{cells}</tr>')
    if prow:
        table += f"""<div class="card"><div class="scroll"><table class="act">
<thead><tr><th>区</th>{''.join(f'<th class="num">{names[k]}（价格+AEMO，累计）</th>' for k in PRODUCT_KEYS)}</tr></thead>
<tbody>{''.join(prow)}</tbody></table></div></div>"""
    strip = ""
    if lt:
        for k in focus:
            x = (lt.get("regions") or {}).get(k)
            if not x:
                continue
            cells_h, cells_a, cells_p, cells_d = [], [], [], []
            for h, a, b2, b3 in zip(x["hour"], x["aemo"], x["b2"], x["b3"]):
                up = b3 > a
                flip = (b3 > a) != (b2 > a)
                cells_h.append(f"<th>{h:02d}</th>")
                cells_a.append(f"<td>{a:,.0f}</td>")
                cells_p.append(f"<td>{b3:,.0f}</td>")
                cells_d.append(f'<td class="dir {"up" if up else "dn"}{" flip" if flip else ""}">'
                               f'{"多" if up else "空"}{"*" if flip else ""}</td>')
            late = "（晚于决策时刻生成，不计成绩）" if lt.get("late") else ""
            strip += f"""
<div class="fig"><figcaption><h3>{_esc(k)} · {lt['target_day']} 的信号</h3>
<p>生成于 {_esc(lt.get('generated_at_utc'))} UTC{late}。每小时：AEMO 预调度价、我们的预测（价格 + AEMO + 天气）、方向。
带 * 的小时 = 天气把方向翻过来了（不加天气时方向相反）。挑中的天气变量：{_esc(x.get('sel'))}。</p></figcaption>
<div class="scroll"><table class="strip"><thead><tr><th>时（NEM）</th>{''.join(cells_h)}</tr></thead>
<tbody><tr><th>AEMO $</th>{''.join(cells_a)}</tr><tr><th>我们 $</th>{''.join(cells_p)}</tr>
<tr><th>方向</th>{''.join(cells_d)}</tr></tbody></table></div></div>"""
    cum = ""
    for k in focus:
        v = regs.get(k) or {}
        if len(v.get("w_gain_cum") or []) >= 2:
            ys = v["w_gain_cum"]
            lo, hi = min(ys + [0.0]), max(ys + [0.0])
            W, H, Lp, T = 500, 140, 52, 12
            pw, ph = W - Lp - 16, H - T - 26
            X = lambda i: Lp + i / (len(ys) - 1) * pw                 # noqa: E731
            Y = lambda q: T + (1 - (q - lo) / ((hi - lo) or 1)) * ph   # noqa: E731
            pts = " ".join(f"{X(i):.1f},{Y(q):.1f}" for i, q in enumerate(ys))
            cum += f"""<div class="fig"><figcaption><h3>{_esc(k)} · 实盘里加天气的累计增量（偏差交易，1 MW）</h3></figcaption>
<svg class="spark" viewBox="0 0 {W} {H}" role="img" aria-label="累计增量">
<line class="zero" x1="{Lp}" x2="{W - 16}" y1="{Y(0):.1f}" y2="{Y(0):.1f}"/>
<text class="tick" x="{Lp - 6}" y="{Y(0) + 4:.1f}" text-anchor="end">$0</text>
<text class="tick" x="{Lp - 6}" y="{Y(ys[-1]) + 4:.1f}" text-anchor="end">{_compact(ys[-1])}</text>
<polyline class="ln s3" points="{pts}"/>
<text class="tick" x="{Lp}" y="{H - 6}">{v['w_gain_dates'][0][5:]}</text>
<text class="tick" x="{W - 16}" y="{H - 6}" text-anchor="end">{v['w_gain_dates'][-1][5:]}</text></svg></div>"""
    fails = pp.get("failures") or []
    fail = ""
    if fails:
        f0 = fails[-1]
        fail = f'<p class="lead2">最近一次没出成：{_esc(f0["target_day"])} {_esc(f0["region"])} —— {_esc(str(f0["status"])[:120])}</p>'
    return f"""
<h2>前推纸面交易（实盘检验）</h2>
<p class="lead2">{head}</p>
{table}{strip}{cum}{fail}"""


# ------------------------------------------------------------------ 稳健性检查单
CHECKS = [
    ("holm", "校正后仍显著", "5 个区一起检验，做了多重比较校正"),
    ("months", "多数月份为正", "至少 70% 的月份是正的"),
    ("leave_out", "去掉最好的月仍为正", "不是一个月撑起来的"),
    ("windows", "换训练窗都为正", "90 / 120 / 180 天，同一段样本"),
    ("not_concentrated", "不靠少数几天", "最好的 5 天占总增量不到一半"),
    ("baseload", "按天下注也成立", "一天押一次全天均价，更接近能交易的单"),
]


def _tick(ok: bool) -> str:
    return ('<span class="tk good" title="通过"><span aria-hidden="true">✓</span><span class="sr">通过</span></span>'
            if ok else
            '<span class="tk critical" title="没通过"><span aria-hidden="true">✕</span><span class="sr">没通过</span></span>')


def robust_checks(rb: dict | None) -> str:
    if not rb or not rb.get("regions"):
        return ""
    rows, good = [], []
    for k, v in rb["regions"].items():
        c, w = v.get("checks"), v.get("weather")
        if not c or not w:
            continue
        n_pass = sum(1 for key, *_ in CHECKS if c.get(key))
        if c.get("holm"):
            good.append((k, n_pass, c))
        cells = "".join(
            f'<td class="ck" data-l="{label}">{_tick(bool(c.get(key)))}'
            + (f'<span class="hint">{c["windows_sig"]}/{c["windows_n"]} 显著</span>' if key == "windows" else "")
            + "</td>" for key, label, _ in CHECKS)
        rows.append(f"""<tr><td class="rg">{_esc(k)}</td>
  <td class="num" data-l="加天气每天"><span class="big2">{_money(w['per_day'], True, 1)}</span>
    <span class="hint">{n_pass}/{len(CHECKS)} 项通过</span></td>{cells}</tr>""")
    if good:
        k, n, c = max(good, key=lambda x: x[1])
        miss = [label for key, label, _ in CHECKS if not c.get(key)]
        say = (f"<b>{k}</b> 是唯一校正后仍显著的区，{n}/{len(CHECKS)} 项通过。"
               + (f"没过的是：{'、'.join(miss)}。" if miss else ""))
        if not c.get("not_concentrated"):
            say += "钱主要来自少数几个高价日，平常日子的增量很小。"
        if not c.get("baseload"):
            say += ("天气改变的是<b>哪几个小时</b>预调度会错，而不是对<b>整天</b>的判断——"
                    "要用它，需要能按小时调整的头寸（灵活负荷、调峰机组的报价、按小时的对冲），"
                    "日基荷合约吃不到。")
    else:
        say = "做了多重比较校正后，没有一个区的天气增量还显著。"
    head = "".join(f'<th class="ck" title="{_esc(tip)}">{label}</th>' for _, label, tip in CHECKS)
    return f"""
<h2>这个信号靠得住吗</h2>
<p class="lead2">{say}</p>
<div class="card"><div class="scroll"><table class="act chk">
<thead><tr><th>区</th><th class="num">加天气每天</th>{head}</tr></thead>
<tbody>{''.join(rows)}</tbody></table></div></div>"""


# ------------------------------------------------------------------ 页面
CSS = """
:root{
 color-scheme:light;
 --page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
 --grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10);
 --s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--base:#9b9a94;
 --good:#0ca30c;--good-ink:#006300;--warning:#fab219;--critical:#d03b3b;
 --chipbg-good:#e6f4e6;--chipbg-critical:#fbe9e9;--chipbg-warning:#fdf3dc;--chipbg-neutral:#efeeea;
}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme=light])){
 color-scheme:dark;
 --page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;
 --grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);
 --s1:#3987e5;--s2:#d95926;--s3:#199e70;--base:#6f6e69;
 --good-ink:#0ca30c;
 --chipbg-good:#13301a;--chipbg-critical:#3a1717;--chipbg-warning:#3a2d0a;--chipbg-neutral:#2a2a28;
}}
:root[data-theme=dark]{
 color-scheme:dark;
 --page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;
 --grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);
 --s1:#3987e5;--s2:#d95926;--s3:#199e70;--base:#6f6e69;
 --good-ink:#0ca30c;
 --chipbg-good:#13301a;--chipbg-critical:#3a1717;--chipbg-warning:#3a2d0a;--chipbg-neutral:#2a2a28;
}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);
 font:15px/1.55 system-ui,-apple-system,"Segoe UI","PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;
 -webkit-font-smoothing:antialiased}
a{color:var(--s1)}
.wrap{max-width:1120px;margin:0 auto;padding:0 20px 64px}
.top{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;
 padding:16px 0;border-bottom:1px solid var(--grid)}
.brand{font-weight:700;font-size:17px;letter-spacing:.01em}
.nav{display:flex;gap:4px;flex-wrap:wrap}
.nav a{padding:6px 12px;border-radius:999px;text-decoration:none;color:var(--ink2);font-size:14px}
.nav a.on{background:var(--surface);color:var(--ink);box-shadow:0 0 0 1px var(--ring)}
.asof{color:var(--muted);font-size:13px;padding:10px 0 0}
.fixture{margin-top:10px;padding:8px 12px;border-radius:8px;background:var(--chipbg-critical);
 color:var(--ink);font-size:13px}
.hero{display:grid;grid-template-columns:auto 1fr;gap:28px;align-items:center;margin:28px 0 8px;
 background:var(--surface);border-radius:14px;padding:24px 28px;box-shadow:0 0 0 1px var(--ring)}
.hero .q{color:var(--ink2);font-size:14px;margin:0 0 4px}
.hero .big{font-size:52px;font-weight:700;line-height:1.05;letter-spacing:-.01em}
.hero .big small{font-size:18px;font-weight:600;color:var(--ink2);margin-left:6px}
.hero .say{font-size:16px;margin:0}
.hero2{grid-template-columns:1fr}
.status{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:10px;font-size:13px;color:var(--ink2)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px}
@media (max-width:640px){.tile .tv{font-size:24px}.panel .fr{grid-template-columns:48px 1fr}.panel .fr .fv{grid-column:2}}
.tile{padding:4px 0}
.tile .tl{font-size:13px;color:var(--ink2)}
.tile .tv{font-size:30px;font-weight:700;letter-spacing:-.01em;font-variant-numeric:tabular-nums}
.tile .tn{font-size:12.5px;color:var(--muted)}
code{font-family:ui-monospace,Menlo,monospace;font-size:.92em;background:var(--chipbg-neutral);padding:1px 5px;border-radius:4px}
.hero .say b{font-weight:650}
@media (max-width:640px){.hero{grid-template-columns:1fr;gap:12px;padding:18px}.hero .big{font-size:44px}}
h2{font-size:18px;margin:36px 0 12px}
.lead2{font-size:14px;color:var(--ink2);margin:-4px 0 14px;max-width:820px}
.lead2 b{color:var(--ink)}
.card{background:var(--surface);border-radius:14px;box-shadow:0 0 0 1px var(--ring);overflow:hidden;margin-bottom:20px}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
table.act{width:100%;border-collapse:collapse;min-width:760px}
table.act th{font-size:12px;font-weight:600;color:var(--muted);text-align:left;padding:12px 14px;
 border-bottom:1px solid var(--grid);white-space:nowrap}
table.act td{padding:14px;border-bottom:1px solid var(--grid);vertical-align:middle}
table.act tr:last-child td{border-bottom:0}
table.act .num,table.dt .num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
table.act .rg{font-weight:700;font-size:16px}
table.act .model{font-weight:600}
table.act .hint{display:block;font-size:12px;color:var(--muted);font-weight:400}
table.act .why{font-size:12.5px;color:var(--ink2);max-width:300px}
.big2{font-size:18px;font-weight:650}
svg.spark{max-width:620px}
.tabs{display:flex;gap:6px;flex-wrap:wrap;margin:4px 0 14px}
.tab{font:inherit;font-size:14px;padding:7px 14px;border-radius:999px;border:0;cursor:pointer;
 background:transparent;color:var(--ink2);box-shadow:0 0 0 1px var(--ring)}
.tab.on{background:var(--ink);color:var(--page);box-shadow:none}
.pane[hidden]{display:none}
table.sens td small{color:var(--muted)}
table.strip{border-collapse:collapse;font-size:12.5px;font-variant-numeric:tabular-nums}
table.strip th,table.strip td{padding:5px 6px;text-align:right;border-bottom:1px solid var(--grid);white-space:nowrap}
table.strip thead th{color:var(--muted);font-weight:600}
table.strip tbody th{text-align:left;color:var(--ink2);font-weight:600;position:sticky;left:0;background:var(--surface)}
table.strip td.dir{font-weight:700;text-align:center}
table.strip td.dir.up{color:var(--ink)}table.strip td.dir.dn{color:var(--muted)}
table.strip td.dir.flip{background:var(--chipbg-warning)}
table.chk th.ck{white-space:normal;text-align:center;min-width:76px;line-height:1.3}
table.chk td.ck{text-align:center}
.tk{display:inline-flex;width:24px;height:24px;border-radius:50%;align-items:center;justify-content:center;
 font-weight:700;font-size:13px}
.tk.good{background:var(--chipbg-good);color:var(--good-ink)}
.tk.critical{background:var(--chipbg-critical);color:var(--critical)}
.sr{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0)}
table.chk td.ck .hint{display:block;font-size:11.5px;color:var(--muted)}
@media (max-width:700px){
 table.act{min-width:0}table.act thead{display:none}
 table.act,table.act tbody{display:block}
 table.act tr{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px 12px;padding:14px 16px;
  border-bottom:1px solid var(--grid)}
 table.act tr:last-child{border-bottom:0}
 table.act td{padding:0;border:0}
 table.act td.rg{grid-column:1/3;align-self:center}
 table.act td.st{grid-column:3;justify-self:end}
 table.act td.mdl,table.act td.why{grid-column:1/-1}
 table.chk tr{grid-template-columns:1fr 1fr}
 table.chk td.rg{grid-column:1}
 table.chk td.num{grid-column:2;text-align:right}
 table.chk td.ck{text-align:left;display:flex;align-items:center;gap:6px;flex-wrap:wrap}
 table.chk td.ck::before{margin:0;flex:1 1 auto}
 table.chk td.ck .hint{display:inline;flex-basis:100%;text-align:right}
 table.act td.why{max-width:none}
 table.act td.num{text-align:left;white-space:normal}
 table.act td[data-l]::before{content:attr(data-l);display:block;font-size:11.5px;color:var(--muted);
  font-weight:500;margin-bottom:2px}
 table.act .big2{font-size:16px}
}
.dim{color:var(--muted)}
.chip{display:inline-flex;align-items:center;gap:5px;padding:3px 10px 3px 8px;border-radius:999px;
 font-size:13px;font-weight:600;white-space:nowrap;color:var(--ink)}
.chip.good{background:var(--chipbg-good)}.chip.good span{color:var(--good)}
.chip.critical{background:var(--chipbg-critical)}.chip.critical span{color:var(--critical)}
.chip.warning{background:var(--chipbg-warning)}.chip.warning span{color:#b07800}
.chip.neutral{background:var(--chipbg-neutral)}.chip.neutral span{color:var(--muted)}
.fig{margin:0 0 24px;background:var(--surface);border-radius:14px;box-shadow:0 0 0 1px var(--ring);
 padding:18px 20px}
.fig figcaption h3{margin:0;font-size:16px}
.fig figcaption p{margin:4px 0 12px;font-size:13px;color:var(--ink2)}
.figbody{overflow-x:auto}
svg{display:block;width:100%;height:auto;font-family:inherit}
svg .grid{stroke:var(--grid);stroke-width:1}
svg .zero{stroke:var(--axis);stroke-width:1.5}
svg .tick{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}
svg .axislab{fill:var(--muted);font-size:11px}
@media (max-width:640px){svg.cum .tick{font-size:16px}}
svg .rlab{fill:var(--ink);font-size:13px;font-weight:600}
svg .rval{font-size:12.5px;font-variant-numeric:tabular-nums}
svg .rval .strong{fill:var(--ink);font-weight:650}
svg .rval .muted{fill:var(--muted)}
.forest{display:flex;flex-direction:column}
.fl2{font-weight:600;font-size:13px}
.panel .fr{grid-template-columns:56px 1fr 150px;gap:10px}
.panel .fv small{display:block;margin-top:-2px}
.rng .r90{position:absolute;top:50%;height:4px;margin-top:-2px;border-radius:2px;background:var(--s1);opacity:.35}
.rng .r50{position:absolute;top:50%;height:10px;margin-top:-5px;border-radius:3px;background:var(--s1);opacity:.75}
.rng .fd{color:var(--ink)}
.fr{display:grid;grid-template-columns:64px 1fr 190px;align-items:center;gap:14px;min-height:40px;
 border-radius:8px;outline:none}
.fr[data-tip]:hover,.fr[data-tip]:focus-visible{background:color-mix(in srgb,var(--grid) 45%,transparent)}
.fl{font-weight:700;text-align:right}
.ftrack{position:relative;height:40px;display:block}
.fhead .ftrack,.faxis .ftrack{height:22px}
.fhead,.faxis{min-height:22px}
.fg{position:absolute;top:0;bottom:0;width:1px;background:var(--grid)}
.f0{position:absolute;top:0;bottom:0;width:2px;margin-left:-1px;background:var(--axis)}
.fw{position:absolute;top:50%;height:2px;margin-top:-1px;border-radius:2px;background:currentColor}
.fd{position:absolute;top:50%;width:12px;height:12px;margin:-6px 0 0 -6px;border-radius:50%;
 background:currentColor;box-shadow:0 0 0 2px var(--surface)}
.fd.clip{background:var(--surface);box-shadow:inset 0 0 0 2px currentColor,0 0 0 2px var(--surface)}
.fa{position:absolute;top:-4px;width:0;height:0;border-top:5px solid transparent;border-bottom:5px solid transparent}
.fa.l{left:-2px;border-right:8px solid currentColor}
.fa.r{right:-2px;border-left:8px solid currentColor}
.ftrack.good{color:var(--good)}.ftrack.critical{color:var(--critical)}.ftrack.neutral{color:var(--muted)}
.ft{position:absolute;top:2px;transform:translateX(-50%);font-size:12px;color:var(--muted);
 font-variant-numeric:tabular-nums;white-space:nowrap}
.fdir{position:absolute;top:0;font-size:12px;color:var(--muted)}.fdir.l{left:0}.fdir.r{right:0}
.fv{font-variant-numeric:tabular-nums;white-space:nowrap}
.fv b{font-weight:650;margin-right:6px}.fv small{color:var(--muted);font-size:12.5px}
@media (max-width:640px){.fr{grid-template-columns:44px 1fr;gap:8px}
 .fr .fv{grid-column:2;margin-top:-8px;padding-bottom:6px}.fhead .fv,.faxis .fv{display:none}}
svg .whisk{stroke-width:2;stroke-linecap:round}
svg .whisk.good,svg .dot.good,svg .arrow.good{stroke:var(--good);fill:var(--good)}
svg .whisk.critical,svg .dot.critical,svg .arrow.critical{stroke:var(--critical);fill:var(--critical)}
svg .whisk.neutral,svg .dot.neutral,svg .arrow.neutral{stroke:var(--muted);fill:var(--muted)}
svg .dot{stroke:var(--surface)!important;stroke-width:2}
svg .arrow{stroke:none!important}
svg .hit{fill:transparent}
svg .frow:hover .hit,svg .frow:focus .hit{fill:var(--grid);opacity:.45}
svg .frow:focus{outline:none}
svg .ln{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
svg .ln.s1{stroke:var(--s1)}svg .ln.s2{stroke:var(--s2)}svg .ln.s3{stroke:var(--s3)}
svg .ln.base{stroke:var(--base)}
svg .end{stroke:var(--surface);stroke-width:2}
svg .end.s1{fill:var(--s1)}svg .end.s2{fill:var(--s2)}svg .end.s3{fill:var(--s3)}svg .end.base{fill:var(--base)}
svg .cross{stroke:var(--axis);stroke-width:1}
svg .hitarea{fill:transparent;cursor:crosshair}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;margin:0 0 10px;font-size:13px;color:var(--ink2)}
.key{display:inline-flex;align-items:center;gap:6px}
.sw{display:inline-block;width:16px;height:3px;border-radius:2px}
.sw.s1{background:var(--s1)}.sw.s2{background:var(--s2)}.sw.s3{background:var(--s3)}.sw.base{background:var(--base)}
.grid5{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,440px),1fr));gap:14px}
.panel{border-radius:10px;box-shadow:0 0 0 1px var(--ring);padding:10px 10px 4px}
.phead{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin:0 0 4px}
.pname{font-weight:700}
.pnote{font-size:12.5px;color:var(--ink2);margin-left:auto}
details.tv{margin-top:10px;font-size:13px}
details.tv summary{cursor:pointer;color:var(--ink2)}
table.dt{border-collapse:collapse;margin-top:8px;font-size:13px}
table.dt th,table.dt td{padding:5px 12px 5px 0;border-bottom:1px solid var(--grid);text-align:left}
.how{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px;margin-top:12px}
.how div{background:var(--surface);border-radius:12px;box-shadow:0 0 0 1px var(--ring);padding:14px 16px;
 font-size:13.5px;color:var(--ink2)}
.how b{color:var(--ink)}
.next{margin-top:28px;padding:16px 18px;border-radius:12px;background:var(--surface);
 box-shadow:0 0 0 1px var(--ring);display:flex;justify-content:space-between;gap:12px;align-items:center;
 flex-wrap:wrap}
.next a{font-weight:600;text-decoration:none}
.tip{position:fixed;pointer-events:none;z-index:10;background:var(--surface);color:var(--ink);
 box-shadow:0 2px 10px rgba(0,0,0,.18),0 0 0 1px var(--ring);border-radius:8px;padding:8px 10px;
 font-size:12.5px;line-height:1.5;display:none;font-variant-numeric:tabular-nums;max-width:280px}
.tip .k{display:inline-block;width:10px;height:3px;border-radius:2px;margin-right:6px;vertical-align:middle}
footer{margin-top:40px;color:var(--muted);font-size:12.5px}
"""

JS = r"""
(function(){
  document.querySelectorAll('.tab').forEach(function(b){
    b.addEventListener('click',function(){
      document.querySelectorAll('.tab').forEach(function(x){x.classList.remove('on');x.setAttribute('aria-selected','false');});
      document.querySelectorAll('.pane').forEach(function(p){p.hidden=true;});
      b.classList.add('on');b.setAttribute('aria-selected','true');
      document.getElementById(b.getAttribute('data-pane')).hidden=false;});
  });
  var tip=document.createElement('div');tip.className='tip';document.body.appendChild(tip);
  function show(html,x,y){tip.innerHTML=html;tip.style.display='block';
    var w=tip.offsetWidth,h=tip.offsetHeight,vx=window.innerWidth,vy=window.innerHeight;
    tip.style.left=Math.min(x+14,vx-w-8)+'px';tip.style.top=Math.min(y+14,vy-h-8)+'px';}
  function hide(){tip.style.display='none';}
  document.querySelectorAll('.fr[data-tip]').forEach(function(g){
    g.addEventListener('mousemove',function(e){show(g.getAttribute('data-tip'),e.clientX,e.clientY);});
    g.addEventListener('mouseleave',hide);
    g.addEventListener('focus',function(){var r=g.getBoundingClientRect();show(g.getAttribute('data-tip'),r.left+20,r.top);});
    g.addEventListener('blur',hide);
  });
  var cls={'B0 naive':'base','AEMO 直用':'base','B1 自回归':'s1','B2 +官方':'s2','B3 +天气':'s3'};
  document.querySelectorAll('svg.cum').forEach(function(svg){
    var d=JSON.parse(svg.getAttribute('data-cum')),hit=svg.querySelector('.hitarea'),
        cross=svg.querySelector('.cross'),n=d.dates.length;
    hit.addEventListener('mousemove',function(e){
      var pt=svg.createSVGPoint();pt.x=e.clientX;pt.y=e.clientY;
      var p=pt.matrixTransform(svg.getScreenCTM().inverse());
      var i=Math.round((p.x-d.L)/d.pw*(n-1));i=Math.max(0,Math.min(n-1,i));
      var x=d.L+i/(n-1)*d.pw;cross.setAttribute('x1',x);cross.setAttribute('x2',x);cross.style.display='';
      var h='<b>'+d.dates[i]+'</b> 累计';
      d.series.forEach(function(s){var v=s[1][i];
        h+='<br><span class="k" style="background:var(--'+cls[s[2]]+')"></span>'+s[0]+'：'+
           (v<0?'−$':'$')+Math.abs(Math.round(v)).toLocaleString();});
      show(h,e.clientX,e.clientY);});
    hit.addEventListener('mouseleave',function(){cross.style.display='none';hide();});
  });
})();
"""


def render_trader(data: dict) -> str:
    regs = data.get("regions", {})
    items = [region_summary(k, v) for k, v in regs.items()]
    ok = [x for x in items if x["ok"]]
    is_fix = bool(data.get("is_fixture"))
    decide_at = data.get("decide_at", "13:30")
    last = max((x["daily"]["dates"][-1] for x in ok if x.get("daily")), default="")
    first = min((x["daily"]["dates"][0] for x in ok if x.get("daily")), default="")
    days = max((x["days"] or 0 for x in ok), default=0)
    prod_html, pst = products_block(regs, decide_at)

    # ---- 一句话结论（按能下单的产品）----
    n_cells = sum(len([x for x in product_rows(regs, k) if x["ok"]]) for k in pst)
    n_yes = sum(len(v["yes"]) for v in pst.values())
    if not pst:
        verdict_big, verdict_say = "—", "还没有一个区跑出可比结果。"
    elif days < D.MIN_TEST_DAYS:
        verdict_big = "未定"
        verdict_say = f"样本外只有 {days} 天，不到 {D.MIN_TEST_DAYS} 天，<b>先不下结论</b>。"
    else:
        verdict_big = f"{n_yes}<small>/ {n_cells}</small>"
        parts = []
        for k, v in pst.items():
            t = f"<b>{v['name']}</b>："
            t += (f"天气在 <b>{'、'.join(v['yes'])}</b> 扣完成本后显著多赚" if v["yes"]
                  else "加天气没有一个区显著多赚")
            if v["loss"]:
                t += f"，<b>{'、'.join(v['loss'])}</b> 反而显著少赚"
            grp = {}
            for rg, m in v["recs"].items():
                grp.setdefault("不做" if m == "不做" else MODEL_NAME.get(m, m), []).append(rg)
            t += "。建议：" + "；".join(f"{'、'.join(r)} {m}" for m, r in grp.items())
            parts.append(t + "。")
        verdict_say = "<br>".join(parts)
    nest = [x for x in ok if x.get("n_ci")]
    bias = [x for x in ok if x["b_ok"] and x["b_ci"]]
    if bias:
        by = [x["region"] for x in bias if x["b_state"] == "yes"]
        verdict_say += ("<br><span class='dim'>信号检验（不能直接下单）：赌 AEMO 偏差时天气"
                        + (f"在 {'、'.join(by)} 显著有用" if by else "没有显著用处") + "。</span>")

    fixture = ('<div class="fixture">合成数据（fixture）——数字都是假的，只用来检查版面。</div>'
               if is_fix else "")
    gen = str(data.get("generated_at", ""))[:16].replace("T", " ")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NEM 天气交易台</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
<div class="top">
  <div class="brand">NEM 天气交易台</div>
  <nav class="nav"><a class="on" href="index.html">交易台</a><a href="method.html">方法与证据</a>
    <a href="{D.REPO_URL}">代码</a></nav>
</div>
<div class="asof">样本外 {first} → {last}（{days} 天）· 每天 {decide_at}（NEM）定次日 24 小时计划 ·
  按现货价结算、已扣交易成本 · 生成于 {gen} UTC</div>
{fixture}
{status_strip(data)}

{hero_tiles(data.get("futures"), verdict_big, verdict_say)}

{futures_block(data.get("futures"))}
{spot_block(data.get("futures"), regs)}
{weather_block(data.get("futures"))}
{risk_block(data.get("futures"), data)}

{prod_html}
{costs_block(regs)}

{bias_section(ok, days)}
{robust_checks(data.get("robust"))}
{paper_section(data.get("paper"), data.get("robust"))}

<h2>怎么读这一页</h2>
<div class="how">
  <div><b>预测来源</b>：AEMO 预调度直接用（不建模型）→ 价格模型（最近电价）→ 价格 + AEMO →
    价格 + AEMO + 天气。天气的价值 = 最后一步多赚的钱；建模的价值 = 价格 + AEMO 比直接用 AEMO 多赚的钱。</div>
  <div><b>"用什么预测"</b>选的是：收益不显著低于最好那个的、<b>最简单</b>的那个。多看一样东西，必须用显著更多的钱来换。</div>
  <div><b>天气"用"</b>：样本外 ≥ {D.MIN_TEST_DAYS} 天，而且加天气每天多赚的钱（已扣成本）95% 区间整段大于 0。</div>
  <div><b>信号检验</b>那一节（赌 AEMO 偏差）说明预测方向对不对，但预调度价不是成交价，不能直接下单，只作参考。</div>
</div>

<div class="next"><span>统计检验、误差解剖、哪些天气变量有用——都在第二页。</span>
  <a href="method.html">方法与证据 →</a></div>

<footer>数据：AEMO NEMWeb（电价、预调度）、Open-Meteo（ECMWF 预报、ERA5）。
  所有特征都只用决策时刻之前已公布的信息。· <a href="{D.REPO_URL}">{D.REPO_URL}</a></footer>
</div>
<script>{JS}</script>
</body>
</html>"""

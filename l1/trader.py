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

import desk as D

MODEL_NAME = {
    "B0 naive": "一周前同期",
    "B1 自回归": "价格模型",
    "B2 +官方": "价格 + AEMO",
    "B3 +天气": "价格 + AEMO + 天气",
}
MODEL_HINT = {
    "B0 naive": "直接照抄上周同一天、同一小时的价格",
    "B1 自回归": "只看最近的电价",
    "B2 +官方": "最近电价 + AEMO 官方预调度",
    "B3 +天气": "最近电价 + AEMO 预调度 + 天气预报",
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
    lim = _nice(max(q * 1.5, 2.0))
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
def cumulative(items: list[dict], decide_at: str = "13:30", mw: float = 1, mwh: float = 2) -> str:
    panels, tables = [], []
    for x in items:
        dl = x.get("daily")
        if not dl or not dl.get("dates"):
            continue
        dates = dl["dates"]
        keys = ["B0 naive"] + [k for k in SERIES if k in dl]
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
            cls = "base" if k == "B0 naive" else f"s{SERIES.index(k) + 1}"
            g.append(f'<polyline class="ln {cls}" points="{pts}"/>')
        rec = x["rec"]
        if rec in cum:
            cls = "base" if rec == "B0 naive" else f"s{SERIES.index(rec) + 1}"
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
                    for k in ["B0 naive"] + SERIES)))
    if not panels:
        return ""
    legend = "".join(
        f'<span class="key"><i class="sw {("base" if k == "B0 naive" else "s" + str(SERIES.index(k) + 1))}"></i>'
        f'{MODEL_NAME[k]}</span>' for k in ["B0 naive"] + SERIES)
    head_cells = "".join(f"<th class='num'>{MODEL_NAME[k]}</th>" for k in ["B0 naive"] + SERIES)
    return f"""
<figure class="fig" id="fig-cum">
  <figcaption><h3>累计收益：一块 {mw:g} MW / {mwh:g} MWh 电池</h3>
    <p>每天 {decide_at} 用各模型的预测排好次日计划，按实际电价结算后的累计美元。线越高越好；
    圆点是该区建议使用的模型。</p></figcaption>
  <div class="legend">{legend}</div>
  <div class="grid5">{''.join(panels)}</div>
  <details class="tv"><summary>表格视图（期末累计）</summary>
    <table class="dt"><thead><tr><th>区</th>{head_cells}</tr></thead>
    <tbody>{''.join(tables)}</tbody></table></details>
</figure>"""


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
<h2>另一种用法：赌 AEMO 预调度的偏差</h2>
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
.grid5{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,440px),1fr));gap:14px}
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
  var cls={'B0 naive':'base','B1 自回归':'s1','B2 +官方':'s2','B3 +天气':'s3'};
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
    n_yes = sum(1 for x in ok if x["state"] == "yes")
    n_loss = [x["region"] for x in ok if x["state"] == "loss"]
    n_short = [x["region"] for x in ok if x["state"] == "none"]
    is_fix = bool(data.get("is_fixture"))
    decide_at = data.get("decide_at", "13:30")
    bt = data.get("battery") or {"mw": 1, "hours": 2}
    last = max((x["daily"]["dates"][-1] for x in ok if x.get("daily")), default="")
    first = min((x["daily"]["dates"][0] for x in ok if x.get("daily")), default="")
    days = max((x["days"] or 0 for x in ok), default=0)

    # ---- 一句话结论 ----
    if not ok:
        verdict_big, verdict_say = "—", "还没有一个区跑出可比结果。"
    elif len(n_short) == len(ok):
        verdict_big = "未定"
        verdict_say = f"样本外只有 {days} 天，不到 {D.MIN_TEST_DAYS} 天，<b>先不下结论</b>。"
    else:
        verdict_big = f"{n_yes}<small>/ {len(ok)} 个区</small>"
        if n_yes == 0:
            verdict_say = "加上天气预报后，<b>没有一个区</b>的电池收益显著提高"
        else:
            verdict_say = f"<b>{n_yes} 个区</b>加上天气后电池收益显著提高"
        if n_loss:
            verdict_say += f"；<b>{'、'.join(n_loss)}</b> 反而显著少赚"
        verdict_say += "。"
        recs = {}
        for x in ok:
            recs.setdefault(MODEL_NAME.get(x["rec"], "—"), []).append(x["region"])
        verdict_say += "<br>排电池建议用：" + "；".join(
            f"<b>{m}</b>（{'、'.join(r)}）" for m, r in recs.items()) + "。"
    nest = [x for x in ok if x.get("n_ci")]
    if nest:
        ny = [x["region"] for x in nest if x["n_state"] == "yes"]
        verdict_say += ("<br>只用训练段挑天气变量（不偷看测试集）后："
                        + (f"<b>{'、'.join(ny)}</b> 显著多赚。" if ny else "仍然<b>没有</b>区显著多赚。"))
    bias = [x for x in ok if x["b_ok"] and x["b_ci"]]
    if bias:
        by = [x["region"] for x in bias if x["b_state"] == "yes"]
        bl = [x["region"] for x in bias if x["b_state"] == "loss"]
        verdict_say += ("<br>换成赌 AEMO 预调度的偏差（吃整天水平，不是日内排序）："
                        + (f"天气在 <b>{'、'.join(by)}</b> 显著多赚" if by else "天气<b>没有</b>一个区显著多赚")
                        + (f"，<b>{'、'.join(bl)}</b> 显著少赚" if bl else "") + "。")
    if ok and not all(x["has_official"] for x in ok):
        verdict_say += " 部分区缺 AEMO 预调度，天气增量没扣掉官方预测，只能当上界。"

    # ---- 行动表 ----
    trs = []
    for x in items:
        if not x["ok"]:
            trs.append(f"<tr><td class='rg'>{_esc(x['region'])}</td><td>{chip('none')}</td>"
                       f"<td colspan='5' class='why'>{_esc(x['reason'])}</td></tr>")
            continue
        rec = x["rec"]
        cap = x["rec_capture"]
        trs.append(f"""<tr>
  <td class="rg">{_esc(x['region'])}</td>
  <td class="st">{chip(x['state'])}</td>
  <td class="mdl" data-l="排电池用"><span class="model">{MODEL_NAME.get(rec, '—')}</span><span class="hint">{MODEL_HINT.get(rec, '')}</span></td>
  <td class="num" data-l="日收益"><span class="big2">{_money(x['rec_daily'])}</span><span class="hint">/ 天</span></td>
  <td class="num" data-l="抓到上限的">{'—' if cap is None else f'{cap:.0%}'}<span class="hint">上限 {_money(x['perfect'])}</span></td>
  <td class="num" data-l="加天气每天"><span class="big2">{_money(x['w_gain'], True, 2)}</span>
      <span class="hint">[{_money((x['w_ci'] or [None])[0], True, 1)}, {_money((x['w_ci'] or [None, None])[1], True, 1)}]</span></td>
  <td class="why">{_esc(x['reason'])}</td>
</tr>""")

    fixture = ('<div class="fixture">合成数据（fixture）——数字都是假的，只用来检查版面。</div>'
               if is_fix else "")
    gen = str(data.get("generated_at", ""))[:16].replace("T", " ")
    forest_html = forest(
        [dict(x, m=x["w_gain"], ci=x["w_ci"]) for x in ok], "m", "ci",
        "加上天气预报，电池每天多赚还是少赚？",
        "点 = 168 天里的日均差额（相对不用天气的同一套模型），横线 = 95% 区间。"
        "横线整段在 0 右边才算真的多赚，整段在左边是真的少赚，跨过 0 就是和运气分不开。空心点 / 箭头 = 超出刻度，看右侧数字。"
        .replace("168", str(days)),
        "fig-weather")

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
  {bt.get('mw', 1):g} MW / {bt.get('mw', 1) * bt.get('hours', 2):g} MWh 电池 · 生成于 {gen} UTC</div>
{fixture}

<section class="hero">
  <div><p class="q">排电池：加上天气后显著多赚的区</p><div class="big">{verdict_big}</div></div>
  <p class="say">{verdict_say}</p>
</section>

<h2>每个区怎么做</h2>
<div class="card"><div class="scroll"><table class="act">
<thead><tr><th>区</th><th>天气</th><th>排电池用</th><th class="num">日收益</th>
<th class="num">抓到上限的</th><th class="num">加天气每天</th><th>原因</th></tr></thead>
<tbody>{''.join(trs)}</tbody></table></div></div>

<h2>证据</h2>
{forest_html}
{cumulative(ok, decide_at, bt.get('mw', 1), bt.get('mw', 1) * bt.get('hours', 2))}

{bias_section(ok, days)}

<h2>怎么读这一页</h2>
<div class="how">
  <div><b>四个模型</b>，一个比一个多看一样东西：一周前同期 → 价格模型（最近电价）→
    价格 + AEMO（再加官方预调度）→ 价格 + AEMO + 天气。天气的价值 = 最后一步多赚的钱。</div>
  <div><b>电池规则</b>：每天 {decide_at} 只用预测排好明天，先充 {bt.get('hours', 2):g} 小时、后放
    {bt.get('hours', 2):g} 小时，算上 85% 往返效率和每 MWh $5 的损耗；预计不赚钱的日子不动。
    然后按实际电价结算。</div>
  <div><b>"排电池用"</b>选的是：收益不显著低于最好那个模型的、<b>最简单</b>的那个。
    多看一样东西，必须用显著更多的钱来换。</div>
  <div><b>天气"用"</b>要同时满足：样本外 ≥ {D.MIN_TEST_DAYS} 天、预测更准、准得不是运气、
    而且每天多赚的钱 95% 区间整段大于 0。任何一条不过就是"不用"。</div>
</div>

<div class="next"><span>统计检验、误差解剖、哪些天气变量有用——都在第二页。</span>
  <a href="method.html">方法与证据 →</a></div>

<footer>数据：AEMO NEMWeb（电价、预调度）、Open-Meteo（ECMWF 预报、ERA5）。
  所有特征都只用决策时刻之前已公布的信息。· <a href="{D.REPO_URL}">{D.REPO_URL}</a></footer>
</div>
<script>{JS}</script>
</body>
</html>"""

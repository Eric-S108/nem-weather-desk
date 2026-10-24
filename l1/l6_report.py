"""
把 price_link.py / l3_signal.py 的 JSON 渲染成一页判定台。
==========================================================
页面只回答一个问题：**这条信号现在能不能拿去下单。**

所以每个 ρ 都画在它自己的噪声地板上——单看数字说明不了任何事，
"0.35 而地板是 0.06" 和 "0.35 而地板是 0.34" 是完全相反的结论。

    python price_link.py --out ../price_link.json
    python l3_signal.py --region QLD1 --hours 10-16 --ablation --out ../signal.json
    python l6_report.py --link ../price_link.json --signal ../signal.json --out ../L6判定台.html
"""
from __future__ import annotations
import argparse
import html
import json
import os

REGION_CN = {"NSW1": "新南威尔士", "QLD1": "昆士兰", "VIC1": "维多利亚",
             "SA1": "南澳", "TAS1": "塔斯马尼亚"}


def esc(x):
    return html.escape(str(x))


def num(v, fmt="{:+.3f}", dash="—"):
    return dash if v is None else fmt.format(v)


def bar(rho, floor, w=132):
    """ρ 对噪声地板的条形。地板画成一道竖线——越过它才算测到东西。

    刻度固定 0~0.7：跨区可比才有意义，自适应刻度会让弱信号看起来和强信号一样长。
    """
    if rho is None:
        return '<span class="bar bar--none">—</span>'
    hi = 0.7
    a = min(abs(rho), hi) / hi
    f = min(abs(floor or 0), hi) / hi
    good = abs(rho) > abs(floor or 0)
    cls = "is-pass" if good else "is-fail"
    return (f'<span class="bar {cls}" style="--w:{a*100:.1f}%;--f:{f*100:.1f}%" '
            f'role="img" aria-label="rho {rho:+.3f}，噪声地板 {floor or 0:.3f}"></span>')


def page(link: dict, sig: dict | None) -> str:
    regions = [r for r in link.get("regions", []) if r.get("status") == "ok"]
    spike = link.get("spike", 300)

    rows = []
    for r in regions:
        code = r["region"]
        dv = r.get("drivers") or {}
        fit = (dv.get("fit") or {}).get("oos_blocked")
        chrono = (dv.get("fit") or {}).get("oos_chrono")
        floor = (dv.get("fit_null") or {}).get("abs_p95")
        hand = (r["levels"].get("forecast_vs_rrp") or {}).get("rho")
        l3 = (r.get("levels") or {}).get("forecast_vs_aemo_residual") or {}
        ha = r.get("lift_hour_adj") or {}
        pd_ = dv.get("per_driver") or {}
        dday = (r.get("drivers_day") or {}).get("fit") or {}
        rows.append({
            "code": code, "cn": REGION_CN.get(code, code), "n": r.get("n"),
            "fit": fit, "chrono": chrono, "floor": floor, "hand": hand,
            "l3": l3.get("rho"), "lift": ha.get("lift"),
            "spikes": r.get("n_spikes"),
            "ghi": (pd_.get("ghi") or {}).get("rho"),
            "wind": (pd_.get("wind") or {}).get("rho"),
            "temp": (pd_.get("temp") or {}).get("rho"),
            "day": dday.get("oos_blocked"),
            "pass": (fit is not None and floor is not None and abs(fit) > floor),
        })
    rows.sort(key=lambda x: -(x["fit"] or 0))

    span = regions[0].get("span") if regions else ["—", "—"]

    def tr(r):
        stab = ("稳" if (r["chrono"] is not None and r["fit"] is not None
                         and abs(r["chrono"] - r["fit"]) < 0.06) else "漂移")
        return f"""<tr>
      <th scope="row"><span class="code">{esc(r['code'])}</span><span class="cn">{esc(r['cn'])}</span></th>
      <td class="n">{num(r['fit'])}</td>
      <td class="barcell">{bar(r['fit'], r['floor'])}</td>
      <td class="n dim">{num(r['floor'], '{:.3f}')}</td>
      <td class="n dim">{num(r['hand'])}</td>
      <td class="n">{num(r['day'])}</td>
      <td class="tag"><span class="chip chip--{'ok' if stab=='稳' else 'warn'}">{stab}</span></td>
      <td class="n">{num(r['lift'], '{:.1f}×', '—')}</td>
      <td class="n dim">{r['spikes'] if r['spikes'] is not None else '—'}</td>
      <td class="n">{num(r['l3'])}</td>
    </tr>"""

    # ---- 消融 ----
    abl_html = ""
    if sig and sig.get("ablation"):
        ab = sig["ablation"]
        full = next((a for a in ab if "+" in a["features"]), None)
        singles = [a for a in ab if "+" not in a["features"] and a.get("rho") is not None]
        top = max(singles, key=lambda a: abs(a["rho"])) if singles else None
        share = (abs(top["rho"]) / abs(full["rho"]) * 100) if (top and full and full.get("rho")) else None
        ev = sig.get("eval") or {}
        items = "".join(
            f"""<tr><td class="feat">{esc(a['features'])}</td>
                <td class="n">{num(a.get('rho'))}</td>
                <td class="barcell">{bar(a.get('rho'), ev.get('null_floor'))}</td>
                <td class="n dim">{'$'+str(a['spread']) if a.get('spread') is not None else '—'}</td></tr>"""
            for a in ab)
        verdict = (f"<strong>{esc(top['features'])} 一个变量就拿到全量的 {share:.0f}%</strong>——"
                   f"这条信号实质上只有一个自由度。"
                   if share and share > 90 else
                   f"最强单变量占全量的 {share:.0f}%。" if share else "")
        abl_html = f"""
    <section class="block" id="ablation">
      <h2>消融：这条信号靠的是哪个变量</h2>
      <p class="lede">每个子集都走<b>同一套</b>滚动重拟合，不是在同一份拟合结果上删列——
      后者比较的是"少个系数"，不是"少个特征"。样本行数也强制一致。</p>
      <div class="scroll"><table class="tbl tbl--abl">
        <thead><tr><th>特征</th><th class="n">样本外 ρ</th><th>对噪声地板</th><th class="n">顶底价差</th></tr></thead>
        <tbody>{items}</tbody>
      </table></div>
      <p class="finding">{verdict}
      而<b>日前光伏出力恰恰是 AEMO 预调度预测得最好的东西</b>。所以在 ③ 层算出来之前，
      不能假设这里面有 alpha。</p>
    </section>"""

    # ---- 十分位 ----
    dec_html = ""
    if sig and (sig.get("eval") or {}).get("deciles"):
        ev = sig["eval"]
        dec = ev["deciles"]
        mx = max(abs(d["rrp_median"]) for d in dec) or 1
        items = "".join(
            f"""<tr><td class="n dim">{d['decile']}</td>
                <td class="n dim">{d['n']}</td>
                <td class="n">{d['rrp_median']:.1f}</td>
                <td class="barcell">{_pricebar(d['rrp_median'], mx)}</td>
                <td class="n">{d['spike_rate']:.2f}%</td></tr>"""
            for d in dec)
        dec_html = f"""
    <section class="block" id="deciles">
      <h2>信号分档 → 真实电价</h2>
      <p class="lede">档位用<b>滚动分位</b>切，不是全样本分位。用全样本分位，等于在当天就知道
      这个信号值在整段回测里排第几——那是未来信息，而它正好决定了下面的 $/MWh。</p>
      <div class="scroll"><table class="tbl">
        <thead><tr><th class="n">档</th><th class="n">样本</th><th class="n">中位价</th>
        <th>　</th><th class="n">尖峰率</th></tr></thead>
        <tbody>{items}</tbody>
      </table></div>
      <p class="finding">顶档 − 底档中位价差 <strong>${ev.get('spread_median')}/MWh</strong>。
      尖峰率指小时内最高 5 分钟价超过 ${spike:.0f}——用小时均价会把 10 分钟的尖峰摊平，系统性漏掉。</p>
    </section>"""

    # 结论由数字生成。写死一句"手填全线失效"，在手填恰好有效的数据上就是在说谎——
    # 页面上每一句断言都必须能被它旁边那张表证实。
    n_pass = sum(1 for r in rows if r["pass"])
    n_hand_bad = sum(1 for r in rows
                     if r["hand"] is not None and r["fit"] is not None
                     and abs(r["hand"]) < abs(r["fit"]) - 0.05)
    if n_hand_bad >= max(1, len(rows) * 0.6):
        verdict_fit = (f"<b>{n_hand_bad}/{len(rows)} 个区里，手填的物理权重明显弱于拟合权重</b>——"
                       f"问题在权重，不在特征。")
    elif n_hand_bad == 0 and rows:
        verdict_fit = ("这份数据上手填权重和拟合权重差不多。"
                       "注意 fixture 的合成电价本来就是用同一套物理式生成的，"
                       "手填口径在这里占了便宜；真数据上不是这样。")
    else:
        verdict_fit = f"{n_hand_bad}/{len(rows)} 个区的手填权重明显弱于拟合权重。"
    verdict_fit += f" {n_pass}/{len(rows)} 个区的拟合值站上了各自的噪声地板。"

    # ---------- 交易台：前瞻档位 + 档位→钱的阶梯 ----------
    desk_html = ""
    fw = (sig or {}).get("forward") or {}
    dec_map = {d["decile"]: d for d in ((sig or {}).get("eval") or {}).get("deciles", [])}
    if fw.get("status") == "ok" and dec_map:
        byday = {}
        for r_ in fw["rows"]:
            byday.setdefault(r_["t"][:10], []).append(r_)
        days = list(byday.items())[:14]
        # 每天一行：逐小时的档位方块，颜色深浅 = 档位高低
        drows = []
        for day, items in days:
            cells = "".join(
                f'<span class="cell d{it["decile"]}" title="{esc(it["t"][11:16])} 档 {it["decile"]}">'
                f'{it["decile"]}</span>' for it in items)
            ks = [it["decile"] for it in items]
            hi = dec_map.get(max(ks), {})
            lo = dec_map.get(min(ks), {})
            span_ = (hi.get("rrp_median", 0) - lo.get("rrp_median", 0))
            drows.append(
                f'<tr><td class="day">{esc(day)}</td><td class="cells">{cells}</td>'
                f'<td class="n">{min(ks)}–{max(ks)}</td>'
                f'<td class="n">${span_:.0f}</td></tr>')
        # 档位阶梯：每一档历史上值多少钱
        lrows = "".join(
            f'<tr><td class="n dim">{d["decile"]}</td><td class="n dim">{d["n"]}</td>'
            f'<td class="n">{d["rrp_median"]:.1f}</td>'
            f'<td class="n">{d["spike_rate"]:.2f}%</td>'
            f'<td class="act">{_action(d["decile"])}</td></tr>'
            for d in sorted(dec_map.values(), key=lambda x: x["decile"]))
        w = fw.get("weights") or {}
        desk_html = f"""
  <section class="block" id="desk">
    <h2>{esc(fw.get('region',''))} · 未来 {fw.get('n_forward')} 个小时的档位</h2>
    <p class="lede">训练截至 <b>{esc(fw.get('trained_to','')[:10])}</b>（与预测段留了禁运间隔），
    最后一个有价小时是 <b>{esc(fw.get('last_priced','')[:10])}</b>。
    档位边界取<b>训练段</b>的分位——用预测段自己的分位切档，等于拿未来定义档位。
    权重 {esc(' '.join(f'{k}={v:+.2f}' for k, v in w.items()))}。</p>
    <div class="scroll"><table class="tbl tbl--desk">
      <thead><tr><th>日期</th><th>逐小时档位</th><th class="n">区间</th><th class="n">日内价差</th></tr></thead>
      <tbody>{''.join(drows)}</tbody>
    </table></div>
    <h3 class="sub-h">每一档历史上值多少钱</h3>
    <div class="scroll"><table class="tbl tbl--ladder">
      <thead><tr><th class="n">档</th><th class="n">样本</th><th class="n">中位价</th>
      <th class="n">尖峰率</th><th>对应动作</th></tr></thead>
      <tbody>{lrows}</tbody>
    </table></div>
    <p class="finding">这张表是<b>条件分布</b>，不是点预测。第 10 档的中位价
    ${dec_map.get(10,{}).get('rrp_median',0):.1f}，第 1 档 ${dec_map.get(1,{}).get('rrp_median',0):.1f}——
    差价来自<b>反复在两端建仓</b>，不来自赌任何单独一小时。</p>
  </section>"""
    elif fw:
        desk_html = f"""
  <section class="block" id="desk">
    <h2>没有可下单的时段</h2>
    <p class="lede">前瞻打分返回 <code class="mono">{esc(fw.get('status'))}</code>。
    最常见的原因是气象预报没有超出电价覆盖的区间——
    跑 <code class="mono">ingest.py --live</code> 把未来 48 小时的预报抓进来。</p>
  </section>"""

    # 风险闸门：③ 有没有算出来，直接决定这页能不能拿去下真金白银
    l3_vals = [r["l3"] for r in rows if r["l3"] is not None]
    if link.get("is_fixture"):
        # fixture 的预调度也是合成的，③ 必然"显著"。在合成数据上说"可以谈仓位"，
        # 正是这一页最不该犯的错——闸门必须先看数据真假，再看数字大小。
        gate_h = "合成数据 — 任何仓位结论都不成立"
        gate_p = ("这一页跑在 fixture 上：RRP 和 AEMO 预调度都是由净负荷生成的，"
                  "③ 层必然显著。它证明的是<b>整条管道算得对</b>，"
                  "不是市场里真有这个机会。换成真数据重跑本页再看闸门。")
        gate_cmd = ""
    elif not l3_vals:
        gate_h = "只能纸上跟踪 — ③ 层还没有数据"
        gate_p = ("下面的档位是<b>相对历史</b>的排序，不是相对市场的优势。"
                  "AEMO 的预调度已经公开发布了同一份日前光伏预报，市场大概率已经定价。"
                  "<b>③ 层量的就是「扣掉 AEMO 说过的部分，你还剩多少」</b>——"
                  "它算出来之前，按这张表下单等于赌自己比公开信息快，而没有证据支持这一点。")
        gate_cmd = ('<div class="cmd">python aemo.py --archive --start 2025-07 '
                    '--end 2026-06 --tables PREDISPATCHPRICE</div>')
    else:
        mx3 = max(abs(v) for v in l3_vals)
        strong = [r["code"] for r in rows
                  if r["l3"] is not None and r["floor"] is not None
                  and abs(r["l3"]) > r["floor"]]
        if strong:
            gate_h = f"③ 层有增量：{', '.join(strong)}"
            gate_p = ("这些区在扣掉 AEMO 预调度已经说过的部分之后，信号仍然显著。"
                      "这才是可以谈仓位的证据。先小仓影子跟踪 30~60 天，"
                      "对齐实际成交与这张表的档位，再谈放大。")
        else:
            gate_h = "③ 层不显著 — 这条信号只是复算了公开信息"
            gate_p = (f"最强的 ③ 只有 {mx3:.3f}，没有越过噪声地板。"
                      "说明这套天气特征提供的东西，AEMO 的预调度里已经有了。"
                      "别在这个方向加仓；换特征（互联线余量、机组检修、晚高峰爬坡）。")
        gate_cmd = ""

    gen = link.get("generated_at", "")[:16].replace("T", " ")
    srcs = link.get("sources") or []
    if link.get("is_fixture"):
        badge = ('<div class="badge badge--fixture"><b>FIXTURE · 格式真、内容合成</b>'
                 '合成 RRP 由净负荷生成，相关必然偏高。这一页证明的是<b>分析代码正确</b>，'
                 '不是市场里真有这么强的关系。换成真数据重跑本页即可。</div>')
    else:
        badge = (f'<div class="badge badge--real"><b>真实数据</b>'
                 f'truth 来源：{esc(", ".join(srcs)) or "—"}</div>')
    return f"""<title>NEM 天气交易台</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {{
  --paper:#F5F6F8; --card:#FFFFFF; --ink:#111721; --slate:#5C6878;
  --rule:#DCE1E8; --rule-soft:#EAEEF3;
  --signal:#1F6079; --signal-soft:#DDEAF0;
  --pass:#2C6E52; --fail:#A63F2A; --warn:#8A6516;
  --shadow:0 1px 2px rgba(17,23,33,.05), 0 8px 24px -16px rgba(17,23,33,.18);
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --paper:#0D1117; --card:#151B23; --ink:#E4E9F0; --slate:#93A1B3;
    --rule:#242D38; --rule-soft:#1C242E;
    --signal:#63B3D0; --signal-soft:#17313C;
    --pass:#65BE96; --fail:#E0836B; --warn:#D3A44A;
    --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 24px -16px rgba(0,0,0,.6);
  }}
}}
:root[data-theme="dark"] {{
  --paper:#0D1117; --card:#151B23; --ink:#E4E9F0; --slate:#93A1B3;
  --rule:#242D38; --rule-soft:#1C242E;
  --signal:#63B3D0; --signal-soft:#17313C;
  --pass:#65BE96; --fail:#E0836B; --warn:#D3A44A;
  --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 24px -16px rgba(0,0,0,.6);
}}
* {{ box-sizing:border-box; }}
body {{
  margin:0; background:var(--paper); color:var(--ink);
  font-family:"Source Serif 4",Georgia,"Songti SC",serif;
  font-size:16px; line-height:1.65; -webkit-font-smoothing:antialiased;
}}
.wrap {{ max-width:1000px; margin:0 auto; padding:clamp(28px,5vw,64px) clamp(18px,4vw,40px) 96px; }}
h1,h2,h3,.eyebrow,.chip,.code,thead th {{ font-family:Archivo,"PingFang SC",system-ui,sans-serif; }}
.n,.code,.bar,.mono {{ font-family:"IBM Plex Mono","SFMono-Regular",monospace; font-variant-numeric:tabular-nums; }}

.eyebrow {{ font-size:11px; letter-spacing:.14em; text-transform:uppercase;
  color:var(--slate); font-weight:600; margin:0 0 10px; }}
h1 {{ font-size:clamp(30px,4.6vw,46px); line-height:1.22; margin:0 0 14px;
  font-weight:700; text-wrap:balance; }}
/* 负字距是给拉丁字母收紧用的；中文方块字会直接叠在一起 */
h2 {{ letter-spacing:0; }}
h2 {{ font-size:20px; font-weight:600; margin:0 0 10px; }}
.sub {{ color:var(--slate); max-width:62ch; margin:0 0 6px; }}
.meta {{ color:var(--slate); font-size:13px; margin:18px 0 0;
  display:flex; flex-wrap:wrap; gap:6px 20px; }}
.meta b {{ color:var(--ink); font-weight:600; }}

.badge {{ font-family:Archivo,"PingFang SC",system-ui,sans-serif; font-size:13px;
  padding:10px 14px; border-radius:3px; margin:0 0 22px; max-width:74ch;
  display:flex; flex-wrap:wrap; gap:4px 10px; align-items:baseline; }}
.badge b {{ font-weight:700; letter-spacing:.03em; }}
.badge--fixture {{ background:var(--warn); color:#fff; }}
.badge--real {{ background:var(--signal-soft); color:var(--ink); }}
.gate {{ margin:36px 0 0; padding:20px 22px; border-radius:3px; background:var(--card);
  border:1px solid var(--rule); border-left:4px solid var(--warn); box-shadow:var(--shadow); }}
.gate h2 {{ margin:0 0 8px; }}
.gate p {{ margin:0; color:var(--slate); max-width:66ch; }}
.gate .cmd {{ margin-top:14px; }}

.block {{ margin:48px 0 0; }}
.lede {{ color:var(--slate); max-width:66ch; margin:0 0 18px; }}
.finding {{ margin:16px 0 0; padding:14px 16px; background:var(--signal-soft);
  border-radius:3px; max-width:70ch; font-size:15px; }}
.finding strong {{ font-weight:600; }}

.scroll {{ overflow-x:auto; border:1px solid var(--rule); border-radius:3px;
  background:var(--card); box-shadow:var(--shadow); }}
.tbl {{ width:100%; border-collapse:collapse; font-size:14px; min-width:820px; }}
.tbl--abl {{ min-width:520px; }}
thead th {{ font-size:11px; letter-spacing:.06em; text-transform:uppercase; font-weight:600;
  color:var(--slate); text-align:left; padding:12px 14px; border-bottom:1px solid var(--rule);
  white-space:nowrap; }}
thead th.n {{ text-align:right; }}
tbody td, tbody th {{ padding:11px 14px; border-bottom:1px solid var(--rule-soft); }}
tbody tr:last-child td, tbody tr:last-child th {{ border-bottom:0; }}
tbody th {{ text-align:left; font-weight:400; white-space:nowrap; }}
td.n {{ text-align:right; white-space:nowrap; }}
td.dim {{ color:var(--slate); }}
td.feat {{ font-family:"IBM Plex Mono",monospace; font-size:13px; }}
.code {{ font-size:13px; font-weight:600; letter-spacing:.02em; }}
.cn {{ color:var(--slate); font-size:12px; margin-left:8px; }}

.bar {{ display:block; position:relative; width:132px; height:9px;
  background:var(--rule-soft); border-radius:1px; overflow:visible; }}
.bar::before {{ content:""; position:absolute; inset:0 auto 0 0; width:var(--w);
  background:var(--signal); border-radius:1px; }}
.bar::after {{ content:""; position:absolute; top:-3px; bottom:-3px; left:var(--f);
  width:2px; background:var(--ink); opacity:.55; }}
.bar.is-fail::before {{ background:var(--fail); }}
.bar--none {{ background:none; color:var(--slate); }}
.pricebar {{ display:block; position:relative; width:120px; height:9px; }}
.pricebar i {{ position:absolute; top:0; bottom:0; background:var(--signal); border-radius:1px; }}
.pricebar i.neg {{ background:var(--warn); }}
.pricebar .axis {{ position:absolute; top:-3px; bottom:-3px; left:50%; width:1px;
  background:var(--rule); }}

.chip {{ display:inline-block; font-size:11px; font-weight:600; letter-spacing:.04em;
  padding:2px 8px; border-radius:2px; border:1px solid; }}
.chip--ok {{ color:var(--pass); border-color:var(--pass); }}
.chip--warn {{ color:var(--warn); border-color:var(--warn); }}

.cmd {{ font-family:"IBM Plex Mono",monospace; font-size:13px; background:var(--card);
  border:1px solid var(--rule); border-radius:3px; padding:10px 13px; overflow-x:auto;
  white-space:pre; color:var(--ink); }}

.sub-h {{ font-family:Archivo,"PingFang SC",system-ui,sans-serif; font-size:14px;
  font-weight:600; margin:26px 0 10px; color:var(--slate); }}
.tbl--desk {{ min-width:640px; }} .tbl--ladder {{ min-width:520px; }}
td.day {{ font-family:"IBM Plex Mono",monospace; font-size:12.5px; white-space:nowrap; }}
td.cells {{ padding:8px 14px; }}
.cell {{ display:inline-flex; align-items:center; justify-content:center;
  width:20px; height:20px; margin:1px; border-radius:2px; font-size:10.5px;
  font-family:"IBM Plex Mono",monospace; color:var(--ink); background:var(--rule-soft); }}
.cell.d1,.cell.d2 {{ background:color-mix(in srgb,var(--signal) 12%,transparent); }}
.cell.d3,.cell.d4 {{ background:color-mix(in srgb,var(--signal) 24%,transparent); }}
.cell.d5,.cell.d6 {{ background:color-mix(in srgb,var(--signal) 40%,transparent); }}
.cell.d7,.cell.d8 {{ background:color-mix(in srgb,var(--warn) 45%,transparent); }}
.cell.d9,.cell.d10 {{ background:var(--warn); color:#fff; font-weight:600; }}
td.act {{ font-size:13px; color:var(--slate); white-space:nowrap; }}
.notes {{ margin:44px 0 0; padding-top:22px; border-top:1px solid var(--rule); }}
.notes h2 {{ margin-bottom:14px; }}
.notes ul {{ margin:0; padding-left:1.1em; color:var(--slate); max-width:70ch; }}
.notes li {{ margin:0 0 9px; }}
.notes b {{ color:var(--ink); font-weight:600; }}
.foot {{ margin:40px 0 0; color:var(--slate); font-size:12.5px; }}
@media (prefers-reduced-motion:reduce) {{ * {{ animation:none!important; transition:none!important; }} }}
</style>

<div class="wrap">
  {badge}
  <p class="eyebrow">NEM · 天气驱动的日前价格信号</p>
  <h1>今天该怎么下手</h1>
  <p class="sub">这条信号给出的是<b>某一小时的电价会落在历史哪一档</b>。
  它不预测价位，只排序。所以用法是分档建仓，不是看点位。</p>
  <p class="meta">
    <span>样本 <b>{regions[0]['n'] if regions else '—'}</b> 小时</span>
    <span><b>{esc(span[0][:10])}</b> → <b>{esc(span[1][:10])}</b></span>
    <span>尖峰阈值 <b>${spike:.0f}/MWh</b></span>
    <span>生成于 {esc(gen)} UTC</span>
  </p>

  <div class="gate">
    <h2>{gate_h}</h2>
    <p>{gate_p}</p>
    {gate_cmd}
  </div>

{desk_html}
  <section class="block">
    <h2>各区信号强度</h2>
    <p class="lede">② 拟合 = 让数据自己挑权重、按周分块 5 折交叉验证（训练集剔掉紧挨测试块的周）
    的样本外 ρ。② 手填 = <code class="mono">engine.demand_shape</code> 的物理权重，已证实不可用，留作对照。</p>
    <div class="scroll"><table class="tbl">
      <thead><tr>
        <th>区</th><th class="n">② 拟合</th><th>对噪声地板</th><th class="n">地板</th>
        <th class="n">② 手填</th><th class="n">日间 10–16</th><th>跨季</th>
        <th class="n">尖峰提升</th><th class="n">尖峰 h</th><th class="n">③ 超 AEMO</th>
      </tr></thead>
      <tbody>{''.join(tr(r) for r in rows)}</tbody>
    </table></div>
    <p class="finding">{verdict_fit}
    「跨季」比较前后两半分别训练的结果：差距小说明权重不随季节漂移。</p>
  </section>
{abl_html}
{dec_html}
  <section class="block notes">
    <h2>这份数字为什么可信</h2>
    <ul>
      <li><b>块长按自相关量出来</b>，不写死。曾经写死"一天"，而信号是 60 天滚动分位、
      自带几十天记忆；块比记忆短，重采样破坏不了自相关，区间虚窄——
      实测两条<b>完全独立</b>的序列有 1/3 的概率被判成「可用」。改成 3× 积分自相关时间后，
      同一实验的假阳性率降到 4%（名义 5%）。</li>
      <li><b>分层</b>按 (整点 × 年月)。日循环和季节是公开信息，不控掉的话 ρ 测的是季节不是天气。</li>
      <li><b>禁运</b>：训练集剔掉紧挨测试块的周。天气过程 3~7 天，不剔掉隔壁周的信息会漏进训练集。</li>
      <li><b>星号来自自助区间是否含 0</b>，不用 p 值——逐小时强自相关下 p 的自由度是错的，
      全表都会带星，没有信息量。</li>
      <li><b>缺失不算作 False</b>。<code class="mono">spk &gt; 300</code> 是布尔数组，NaN 比较后变 False，
      直接求均值会把没观测到的小时算成"没有尖峰"，基准率被系统性压低。</li>
    </ul>
  </section>

  <section class="block notes">
    <h2>还没解决的</h2>
    <ul>
      <li><b>③ 层缺历史预调度</b>——唯一的决策关口。</li>
      <li><b>只有 ECMWF 一家</b>。GFS/ICON/GEM 的 previous-runs 存档只到 2026-04，
      集成与模型筛选只能在那之后的窗口上做。</li>
      <li><b>需求项是近似的</b>。<code class="mono">demand_shape</code> 的气温项在 16~24℃ 之间恒为 0，
      而 27 个月样本显示气温根本不是驱动。真实需求在 <code class="mono">DISPATCHREGIONSUM</code> 里。</li>
    </ul>
  </section>

  <p class="foot">由 <code class="mono">price_link.py</code> 与 <code class="mono">l3_signal.py</code> 的
  JSON 输出生成。改了数据重跑 <code class="mono">l6_report.py</code> 即可刷新本页。</p>
</div>"""


def _action(k: int) -> str:
    """档位 → 动作。只给方向和相对力度，不给数量——仓位大小取决于你的资产和风险预算。"""
    if k >= 9:
        return "偏紧：放电 / 减少用电 / 不锁定"
    if k >= 7:
        return "偏紧，力度小"
    if k <= 2:
        return "偏松：充电 / 可锁定负荷"
    if k <= 4:
        return "偏松，力度小"
    return "中性，不动"


def _pricebar(v, mx, w=120):
    half = w / 2
    frac = min(abs(v) / mx, 1.0) * half
    if v >= 0:
        style = f"left:{half:.0f}px;width:{frac:.0f}px"
        cls = ""
    else:
        style = f"left:{half-frac:.0f}px;width:{frac:.0f}px"
        cls = " neg"
    return (f'<span class="pricebar" role="img" aria-label="中位价 {v:.1f}">'
            f'<span class="axis"></span><i class="{cls.strip()}" style="{style}"></i></span>')


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--link", default=os.path.join(here, "price_link.json"))
    ap.add_argument("--signal", default=os.path.join(here, "signal.json"))
    ap.add_argument("--out", default=os.path.join(here, "L6判定台.html"))
    a = ap.parse_args(argv)

    link = json.load(open(a.link, encoding="utf-8"))
    sig = None
    if a.signal and os.path.exists(a.signal):
        sig = json.load(open(a.signal, encoding="utf-8"))
    else:
        print(f"  [warn] 没找到 {a.signal}，消融与分档两节会省略")

    with open(a.out, "w", encoding="utf-8") as f:
        f.write(page(link, sig))
    print(f"→ {a.out}  ({os.path.getsize(a.out)/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

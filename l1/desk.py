"""
交易台 —— 把基准阶梯的结果渲染成一页可以每天看的东西。
=====================================================
这一页只回答一个问题：**今天，天气预报值不值得进我的决策？**

所以版面按决策顺序排，不按研究顺序排：
  1. 每个区一张牌，一个裁决（可用 / 不可用 / 数据不足），一个金额
  2. 展开才是证据：阶梯、经济评估、特征取舍
  3. 口径和出处放最后，随时可查，但不挡路

裁决的门槛是**四道全过**才算"可用"：
  · 样本外至少 MIN_TEST_DAYS 天（太短的样本什么都证明不了）
  · rMAE 相对上一层有改善
  · Giacomini-White p ≤ 0.05（不是运气）
  · 储能套利日收益增量的 95% 区间下界 > 0（能变现）

任何一道不过就是"不可用"。宁可漏掉真信号，也不能把噪声当 alpha 报出去。

    python desk.py --in ladder.json --out 交易台.html
"""
from __future__ import annotations
import argparse
import html
import json
from datetime import datetime, timezone

TIERS = ["B0 naive", "B1 自回归", "B2 +官方", "B3 +天气"]
MIN_TEST_DAYS = 90
REPO_URL = "https://github.com/Eric-S108/nem-weather-desk"
TIER_NOTE = {
    "B0 naive": "一周前同一时刻的价格。所有 rMAE 的分母。",
    "B1 自回归": "加上决策时刻已结算的最新价、最近两天同一钟点、一周前同一钟点和 24h 均价。综述里的主导预测因子。",
    "B2 +官方": "加上 AEMO 预调度（决策时刻前发出的最新一版）。这部分市场已经定价。",
    "B3 +天气": "加上 GHI / 风速 / 气温预报的滚动分位。天气的增量在这一行。",
}


def test_days(r: dict) -> float | None:
    ec = r.get("economics") or {}
    if ec.get("status") == "ok":
        return float(ec["days"])
    if r.get("status") == "ok" and r.get("span"):
        a, b = (str(x)[:10] for x in r["span"])
        from datetime import date
        return float((date.fromisoformat(b) - date.fromisoformat(a)).days + 1)
    return None


def verdict(r: dict) -> tuple[str, str, str]:
    """（等级, 标签, 一句话）。四道门槛全过才算可用。"""
    if r.get("status") != "ok":
        s = r.get("status", "?")
        msg = {"no_rows": "特征表是空的——预报或真值没接上。",
               "too_short": "样本跨度不够一个训练窗 + 禁运带。",
               "too_few_test": "样本外行数太少，比不出东西。",
               "error": r.get("error", "")}.get(s, str(s))
        return "none", "数据不足", msg
    b3 = next((x for x in r["tiers"] if x["tier"] == "B3 +天气"), None)
    if not b3 or "gain_rmae" not in b3:
        return "none", "数据不足", "天气层没有跑出可比结果。"
    ec = r.get("economics") or {}
    erow = next((x for x in ec.get("rows", []) if x["tier"] == "B3 +天气"), None)
    gain_ci = erow.get("gain_ci") if erow else None
    p = b3.get("gw_p")

    fails = []
    nd = test_days(r)
    short = nd is not None and nd < MIN_TEST_DAYS
    if short:
        fails.append(f"样本外只有 {nd:.0f} 天（门槛 {MIN_TEST_DAYS} 天）")
    if b3["gain_rmae"] <= 0:
        fails.append(f"rMAE 没改善（{b3['gain_rmae']:+.4f}）")
    if p is None or p > 0.05:
        fails.append(f"GW p={'—' if p is None else f'{p:.3f}'} 不显著")
    if gain_ci is None:
        fails.append("经济增益算不出区间")
    elif gain_ci[1] < 0:
        fails.append(f"日收益增量区间 [{gain_ci[0]:.1f}, {gain_ci[1]:.1f}] 整体在 0 以下——显著亏钱")
    elif gain_ci[0] <= 0:
        fails.append(f"日收益增量区间 [{gain_ci[0]:.1f}, {gain_ci[1]:.1f}] 跨 0")

    if fails:
        if short:
            return "none", "样本不足", "；".join(fails) + "。"
        return "no", "不可用", "；".join(fails) + "。"
    tail = "" if r.get("has_official") else "但**没有扣掉 AEMO 预调度**，见下方警示。"
    return "yes", "可用", (
        f"在 {b3['vs_prev']} 之上 rMAE 改善 {b3['gain_rmae']:.4f}，GW p={p:.4f}，"
        f"电池日收益多 ${erow['gain']:.2f}/天。{tail}")


def _n(v, fmt="{:.3f}", dash="—"):
    return dash if v is None else fmt.format(v)


def card(region: str, r: dict) -> str:
    lvl, tag, why = verdict(r)
    ec = r.get("economics") or {}
    erow = next((x for x in ec.get("rows", []) if x["tier"] == "B3 +天气"), None)
    g_ = erow.get("gain") if erow else None
    big = "—" if g_ is None else ("+" if g_ > 0 else "−" if g_ < 0 else "") + f"${abs(g_):.2f}"
    sub = ("每天 · 1 MW / 2 MWh 电池" if erow and "gain" in erow else "没有可用估计")
    b3 = next((x for x in (r.get("tiers") or []) if x["tier"] == "B3 +天气"), None)
    rmae = _n(b3.get("gain_rmae") if b3 else None, "{:+.4f}")
    return f"""
    <a class="card {lvl}" href="#r-{region}">
      <div class="chip">{tag}</div>
      <h3>{region}</h3>
      <div class="big">{big}</div>
      <div class="sub">{sub}</div>
      <div class="mini">rMAE 增量 {rmae}</div>
      <p class="why">{html.escape(why).replace('**', '')}</p>
    </a>"""


def ladder_table(r: dict) -> str:
    rows = []
    for t in r.get("tiers", []):
        if t.get("skipped"):
            rows.append(f"""<tr class="skip"><td>{t['tier']}</td>
              <td colspan="7">跳过 —— {t['skipped']}</td></tr>""")
            continue
        g = _n(t.get("gain_rmae"), "{:+.4f}")
        p = _n(t.get("gw_p"), "{:.4f}")
        hi = ' class="hl"' if t["tier"] == "B3 +天气" else ""
        bar = min(100, max(0, (1.0 - t["rmae"]) * 250))
        rows.append(f"""<tr{hi}>
          <td class="tier">{t['tier']}<span class="note">{TIER_NOTE.get(t['tier'],'')}</span></td>
          <td class="num">{t['mae']:.2f}</td>
          <td class="num rm">{t['rmae']:.3f}<i style="width:{bar:.0f}%"></i></td>
          <td class="num dim">{_n(t.get('rmdae'))}</td>
          <td class="num">{t['rmse']:.1f}</td>
          <td class="num">{t['rho']:.3f}</td>
          <td class="num">{g}</td><td class="num">{p}</td></tr>""")
    return f"""<div class="tblwrap"><table class="lad"><thead><tr>
      <th>层</th><th>MAE</th><th>rMAE</th><th>rMdAE</th><th>RMSE</th><th>ρ</th>
      <th>相对上层</th><th>GW p</th></tr></thead>
      <tbody>{''.join(rows)}</tbody></table></div>"""


RMDAE_NOTE = """<p class="cap"><b>rMdAE</b> 是同样口径的<b>中位</b>误差比。
  电价的 MAE 常常被极少数尖峰主导——这时 rMAE 讲的是"尖峰上表现如何"，
  rMdAE 讲的是"平常一小时表现如何"。两个背离得厉害，就看下面的误差解剖：
  多半是几个时刻炸了，不是模型整体不行。</p>"""


def econ_table(r: dict) -> str:
    ec = r.get("economics") or {}
    if ec.get("status") != "ok":
        return f'<p class="empty">经济评估不可用（{ec.get("status","—")}）。</p>'
    rows = []
    for x in ec["rows"]:
        ci = f"[{x['ci'][0]:.1f}, {x['ci'][1]:.1f}]"
        cap = f"{x['capture']:.1%}" if x.get("capture") is not None else "—"
        g = f"{x['gain']:+.2f}" if "gain" in x else "—"
        gci = (f"[{x['gain_ci'][0]:.1f}, {x['gain_ci'][1]:.1f}]"
               if "gain_ci" in x else "—")
        hi = ' class="hl"' if x["tier"] == "B3 +天气" else ""
        rr = _n(x.get("rank_rho"))
        sp = _n(x.get("spread"), "{:.2f}")
        flat = x.get("spread") is not None and x["spread"] < 0.4
        rows.append(f"""<tr{hi}><td class="tier">{x['tier']}</td>
          <td class="num">{x['daily']:.2f}</td><td class="num dim">{ci}</td>
          <td class="num">{cap}</td><td class="num">{rr}</td>
          <td class="num{' warnnum' if flat else ''}">{sp}</td>
          <td class="num">{g}</td>
          <td class="num dim">{gci}</td></tr>""")
    rows.append(f"""<tr class="perfect"><td class="tier">完美预知</td>
      <td class="num">{ec['perfect_daily']:.2f}</td><td class="num dim">收益上限</td>
      <td class="num">100.0%</td><td class="num">1.000</td><td class="num">1.00</td>
      <td class="num">—</td><td class="num dim">—</td></tr>""")
    hpd = ec.get("hours_per_day", 0)
    extra = ""
    best = max(ec["rows"], key=lambda r_: r_["daily"]) if ec["rows"] else None
    if best is not None and best["tier"] == ec["rows"][0]["tier"] and len(ec["rows"]) > 1:
        extra += """<div class="warn"><b>赚钱最多的是 naive——统计上更好的层反而更少。</b>
          这不是矛盾，是两个目标不一样：MAE 奖励"贴近价格水平"，套利要的是"排对日内顺序"。
          看<b>振幅比</b>那一列，回归常常把日内曲线压平（振幅比远小于 1），
          水平贴得很准，但 argmax/argmin 挑出来的时刻基本是噪声。</div>"""
    if hpd and hpd < 12:
        extra += f"""<div class="warn"><b>每天只有 {hpd} 个可选小时，套利被窗口卡住了。</b>
          储能真正的套利是"午间充电、晚峰放电"，跨越的正是这个时段之外的小时。
          这里的数字只说明"在这个窗口内部"能捕多少，不是真实经济价值——
          那个要用 <code>--hours all</code> 单独跑。</div>"""
    bt = ec.get("battery") or {"mw": 1, "hours": 1, "cost": 0}
    return f"""<p class="cap">一块 {bt['mw']:g} MW / {bt['mw'] * bt['hours']:g} MWh 电池，
      每个 NEM 交易日<b>只用预测价</b>定计划：先充 {bt['hours']} 小时、后放 {bt['hours']} 小时，
      往返效率 η={ec['eta']}，吞吐成本 ${bt['cost']:g}/MWh，预测收益 ≤ 0 的日子不动；
      然后按<b>实际</b>价结算。共 {ec['days']} 个交易日。完美预知用同一个优化器喂实际价，
      是同一约束下的上限。区间是按日块自助（2000 次）算的。</p>
      <div class="tblwrap"><table class="lad wide"><thead><tr><th>层</th>
      <th>日均 $/天</th><th>95% 区间</th><th>捕获率</th><th>日内秩 ρ</th>
      <th>振幅比</th><th>相对上层</th><th>增量区间</th></tr></thead>
      <tbody>{''.join(rows)}</tbody></table></div>
      <p class="cap"><b>日内秩 ρ</b> = 每天 pred 与 actual 的秩相关的均值；
      <b>振幅比</b> = 日内预测振幅 / 实际振幅的中位数。套利只吃日内排序，
      这两个数才是它的直接驱动，MAE 不是。</p>{extra}"""


def var_table(r: dict) -> str:
    vs = r.get("variants") or []
    if not vs:
        return ""
    rows = []
    for v in vs:
        keep = v["gain_rmae"] > 0 and v.get("gw_p") is not None and v["gw_p"] <= 0.10
        rows.append(f"""<tr class="{'keep' if keep else 'drop'}">
          <td>{'留' if keep else '删'}</td><td class="tier">{html.escape(v['name'])}</td>
          <td class="num">{v['rmae']:.3f}</td>
          <td class="num">{v['gain_rmae']:+.4f}</td>
          <td class="num">{_n(v.get('gw_p'), '{:.3f}')}</td></tr>""")
    return f"""<p class="cap">每一组都和 <b>{vs[0]['vs']}</b> 比。判为"留"要同时满足
      rMAE 有改善且 GW p≤0.10。没过的特征应当删掉——多一个自由度，少一分稳定，
      而且会在样本内制造好看的假象。</p>
      <div class="tblwrap"><table class="lad var"><thead><tr><th></th>
      <th>特征组</th><th>rMAE</th>
      <th>相对基准</th><th>GW p</th></tr></thead>
      <tbody>{''.join(rows)}</tbody></table></div>"""


def nested_table(r: dict) -> str:
    nd = r.get("nested")
    if not nd:
        return ""
    en = (r.get("economics") or {}).get("nested") or {}
    fq = "".join(f"<tr><td>{html.escape(k)}</td><td class='num'>{v:.0%}</td></tr>"
                 for k, v in nd["freq"].items())
    cb = "、".join(f"{html.escape(k)} ×{v}" for k, v in list(nd["combos"].items())[:4])
    gi = en.get("gain_ci")
    money = ("—" if not gi else
             f"{en['gain']:+.2f} $/天，95% 区间 [{gi[0]:.1f}, {gi[1]:.1f}]")
    return f"""<p class="cap"><b>上面那张表是在测试集上比出来的，只能当诊断，不能拿来决定留哪个变量</b>——
      那等于偷看了答案。真正的取舍在这里：每个滚动窗口<b>只用训练段</b>挑一次——
      训练窗前段拟合、最后 {nd['val_days']} 天验证，枚举天气变量的全部组合（含"不用"），
      验证误差在最好的 {nd['tol']:.1%} 以内时取变量更少的那个。挑完用整段训练窗重拟合，再预测测试段。</p>
      <div class="tblwrap"><table class="lad"><thead><tr><th>相对 {html.escape(nd['vs'])}</th>
      <th>rMAE</th><th>增量</th><th>GW p</th><th>电池日收益增量</th></tr></thead>
      <tbody><tr class="hl"><td class="tier">{html.escape(nd['name'])}</td>
      <td class="num">{nd['rmae']:.3f}</td><td class="num">{nd['gain_rmae']:+.4f}</td>
      <td class="num">{_n(nd.get('gw_p'), '{:.3f}')}</td><td class="num">{money}</td></tr></tbody></table></div>
      <p class="cap">{nd['windows']} 个窗口里各变量被选中的比例：</p>
      <div class="tblwrap"><table class="lad var"><tbody>{fq}</tbody></table></div>
      <p class="cap">最常见的组合：{cb}。</p>"""


def bias_table(r: dict) -> str:
    bt = r.get("bias") or {}
    if bt.get("status") != "ok":
        return '<p class="cap">没有 AEMO 预调度，做不了这一项。</p>'
    rows = []
    for x in bt["rows"]:
        g = ("—" if "gain" not in x else
             f"{x['gain']:+.1f} [{x['gain_ci'][0]:.1f}, {x['gain_ci'][1]:.1f}]")
        tr = "—" if "gain_trim" not in x else f"{x['gain_trim']:+.1f}"
        vs = ("—" if "vs_short" not in x else
              f"{x['vs_short']:+.0f} [{x['vs_short_ci'][0]:.0f}, {x['vs_short_ci'][1]:.0f}]")
        hi = ' class="hl"' if x["tier"] == "B3 +天气" else ""
        rows.append(f"""<tr{hi}><td class="tier">{html.escape(x['tier'])}</td>
          <td class="num">{x['daily']:.1f}</td>
          <td class="num">[{x['ci'][0]:.0f}, {x['ci'][1]:.0f}]</td>
          <td class="num">{x['median_day']:.1f}</td>
          <td class="num">{'—' if x['hit'] is None else format(x['hit'], '.1%')}</td>
          <td class="num">{vs}</td><td class="num">{g}</td><td class="num">{tr}</td></tr>""")
    return f"""<p class="cap">电池只吃<b>日内排序</b>，而天气（尤其风）主要移动的是<b>整天的水平</b>。
      这一项量水平信号：每个小时，预测高于 AEMO 预调度就做多 1 MW、低于就做空，
      按 (实际 − 预调度) 结算。它回答"我们对 AEMO 的修正，方向对不对、值多少钱"。
      NEM 没有日前市场，预调度不是成交价，所以这是<b>信号价值</b>，
      现实里对应调整对冲头寸 / 报价或在期货、互换上表达看法。
      电价重尾，均值会被几个尖峰日支配，所以另给日中位数、方向命中率（|实际 − 预调度| &lt; $1 的小时不计），
      以及去掉最好、最差各 2% 日子后的增量。
      第一行"做空 AEMO"是基线：不看任何模型、每小时都做空预调度。前一天的预调度系统性偏高，
      光这一条就"赚"很多，那是口径偏差，不是信息；模型要和它比。</p>
      <div class="tblwrap"><table class="lad"><thead><tr><th>层</th><th>日均 $</th><th>95% 区间</th>
      <th>日中位 $</th><th>命中率</th><th>相对一直做空</th><th>相对 {html.escape(bt['ref'])}</th><th>去尖峰后</th></tr></thead>
      <tbody>{''.join(rows)}</tbody></table></div>"""


def _pc(v) -> str:
    return "—" if v is None else f"{v:.0%}"


def prob_section(regs: dict) -> str:
    """概率预测（分位数、尖峰）和两个新模型（混合、供需）的检验。"""
    rows, mrows = [], []
    for k, r in regs.items():
        pb = r.get("prob") or {}
        if pb.get("status") == "ok":
            o_, a_ = pb["ours"], pb["aemo"]
            rows.append(f"<tr><td class='tier'>{k}</td><td class='num'>{pb['spike_rate']:.1%}</td>"
                        f"<td class='num'>{o_['pinball']['0.1']:.1f} / {o_['pinball']['0.5']:.1f} / {o_['pinball']['0.9']:.1f}</td>"
                        f"<td class='num'>{a_['pinball']['0.1']:.1f} / {a_['pinball']['0.5']:.1f} / {a_['pinball']['0.9']:.1f}</td>"
                        f"<td class='num'>{o_['cov80']:.0%} / {a_['cov80']:.0%}</td>"
                        f"<td class='num'>{o_['brier']:.4f}</td><td class='num'>{a_['brier']:.4f}</td>"
                        f"<td class='num'>{pb['brier_aemo_hard']:.4f}</td><td class='num'>{pb['brier_clim']:.4f}</td></tr>")
        for key, lab in (("hybrid", "价格 + AEMO（AEMO 日内形状）"), ("fund", "价格 + AEMO + 供需")):
            h = r.get(key)
            if h:
                mrows.append(f"<tr><td class='tier'>{k}</td><td>{lab}</td><td class='num'>{h['rmae']:.3f}</td>"
                             f"<td class='num'>{h['gain_rmae']:+.4f}</td><td class='num'>{_n(h.get('gw_p'), '{:.3f}')}</td></tr>")
        if not r.get("has_fund") and r.get("status") == "ok":
            mrows.append(f"<tr><td class='tier'>{k}</td><td>价格 + AEMO + 供需</td>"
                         f"<td colspan='3' class='cap'>还没有足够的历史（预调度归档里的供需要重新回填）</td></tr>")
    if not rows and not mrows:
        return ""
    return f"""<section class="reg" id="prob">
      <h2>概率预测与新模型</h2>
      <p class="lede">NEM 的钱大部分在尾部。这里检验两件事：给出的分位数和尖峰概率准不准；
        两个新模型（AEMO 日内形状 + 我们的水平、加供需基本面）有没有让预测更准。</p>
      <p class="cap">分位数 = 点预测 + 过去 90 天同一小时的<b>样本外</b>误差。pinball loss 越小越好；
        80% 区间覆盖应接近 80%。尖峰 = 价格 &gt; $300；Brier 越小越好，对照"AEMO 预调度 + 它自己的误差"、
        "预调度 &gt; $300 就当 100%"和"历史同小时频率"。</p>
      <div class="tblwrap"><table class="lad"><thead><tr><th>区</th><th>尖峰小时占比</th>
        <th>pinball q10/q50/q90（我们）</th><th>pinball（AEMO）</th><th>80% 覆盖 我们/AEMO</th>
        <th>Brier 我们</th><th>Brier AEMO+误差</th><th>Brier 预调度&gt;300</th><th>Brier 历史频率</th></tr></thead>
        <tbody>{''.join(rows)}</tbody></table></div>
      <h4>新模型：相对 价格 + AEMO</h4>
      <div class="tblwrap"><table class="lad"><thead><tr><th>区</th><th>模型</th><th>rMAE</th><th>增量</th><th>GW p</th></tr></thead>
        <tbody>{''.join(mrows)}</tbody></table></div>
    </section>"""


def futures_section(fut: dict | None) -> str:
    """期货公允价值的历史校准：抽样区间准不准、放宽多少、按剩余天数分的误差。"""
    if not fut or not fut.get("regions"):
        return ""
    rows, det = [], []
    for reg, r in fut["regions"].items():
        for key, c in (r.get("calibration") or {}).items():
            if not c.get("n"):
                continue
            kind, prod = key.split("_")
            rows.append(f"<tr><td class='tier'>{reg}</td><td>{'月度' if kind == 'month' else '季度'}{'基荷' if prod == 'base' else '峰荷'}</td>"
                        f"<td class='num'>{c['periods']}</td><td class='num'>{c['n']}</td>"
                        f"<td class='num'>{c['cov50']:.0%}</td><td class='num'>{c['cov90']:.0%}</td>"
                        f"<td class='num'>{c['k50']:.2f} / {c['k90']:.2f}</td>"
                        f"<td class='num'>{_pc(c.get('oos_cov50'))}</td>"
                        f"<td class='num'>{_pc(c.get('oos_cov90'))}</td></tr>")
            if key == "month_base":
                for b in c.get("by_left", []):
                    det.append(f"<tr><td class='tier'>{reg}</td><td>{html.escape(b['bucket'])}</td><td class='num'>{b['n']}</td>"
                               f"<td class='num'>{b['mae']:.1f}</td><td class='num'>{_n(b['mae_naive'], '{:.1f}')}</td>"
                               f"<td class='num'>{b['cov90']:.0%}</td></tr>")
    return f"""<section class="reg" id="futures">
      <h2>期货公允价值：历史校准</h2>
      <p class="lede">在过去每个完整的月份和季度上，从开始前一天起每隔一周站一次位置，算结算价分布，
        再看实际结算价落在 50% / 90% 区间里的比例。抽样区间偏窄时按历史放宽（只放宽、不收窄），
        放宽系数只用<b>更早的</b>合约期估、在后面的合约期上检验，得到样本外覆盖率。</p>
      <p class="cap">抽样：决策日前 {fut.get('pool_days', 90)} 天的完整交易日，{fut.get('block', 3)} 天一块，{fut.get('n_sims', 4000)} 次。
        结算按小时均价，$300 cap 因此偏低。没有用 ASX 的任何价格。</p>
      <div class="tblwrap"><table class="lad"><thead><tr><th>区</th><th>合约</th><th>合约期数</th><th>站位次数</th>
        <th>50% 覆盖（抽样）</th><th>90% 覆盖（抽样）</th><th>放宽系数 50/90</th><th>50% 覆盖（放宽，样本外）</th><th>90% 覆盖（放宽，样本外）</th></tr></thead>
        <tbody>{''.join(rows)}</tbody></table></div>
      <h4>月度基荷：按剩余比例看误差</h4>
      <p class="cap">MAE = 公允价值（均值）和实际结算价的平均绝对差，$/MWh；"只看已结算"= 假设剩下的日子和已经过去的一样。</p>
      <div class="tblwrap"><table class="lad"><thead><tr><th>区</th><th>站位时剩余</th><th>次数</th><th>MAE</th><th>MAE（只看已结算）</th><th>90% 覆盖（抽样）</th></tr></thead>
        <tbody>{''.join(det)}</tbody></table></div>
    </section>"""


def products_section(regs: dict) -> str:
    """三个能下单的产品，逐区逐预测来源的日均净收益（已扣交易成本）。"""
    import products as PR
    blocks = []
    for key in PR.PRODUCTS:
        rows, name = [], None
        for k, r in regs.items():
            pr = r.get("products") or {}
            it = (pr.get("items") or {}).get(key) if pr.get("status") == "ok" else None
            if not it:
                continue
            name = it["name"]
            rr = {x["model"]: x for x in it["rows"]}
            cells = "".join(
                f"<td class='num'>{rr[m]['daily']:.1f}</td>" if m in rr else "<td>—</td>"
                for m in PR.MODEL_ORDER)
            w, b2 = rr.get("B3 +天气", {}), rr.get("B2 +官方", {})
            wci, mci = w.get("vs_b2_ci"), b2.get("vs_aemo_ci")
            rows.append(f"<tr><td class='tier'>{k}</td>{cells}<td class='num'>{it['perfect']:.1f}</td>"
                        f"<td class='num'>{w.get('vs_b2', float('nan')):+.1f}<br><small>"
                        f"{'' if not wci else f'[{wci[0]:.1f}, {wci[1]:.1f}]'}</small></td>"
                        f"<td class='num'>{b2.get('vs_aemo', float('nan')):+.1f}<br><small>"
                        f"{'' if not mci else f'[{mci[0]:.1f}, {mci[1]:.1f}]'}</small></td>"
                        f"<td>{html.escape(it.get('recommended') or '—')}</td></tr>")
        if rows:
            head = "".join(f"<th>{html.escape(m)}</th>" for m in PR.MODEL_ORDER)
            blocks.append(f"""<h4>{html.escape(name)}：{html.escape(PR.PRODUCTS[key][1])}</h4>
      <div class="tblwrap"><table class="lad"><thead><tr><th>区</th>{head}<th>完美预知</th>
      <th>B3 − B2</th><th>B2 − AEMO 直用</th><th>推荐</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>""")
    if not blocks:
        return ""
    c = PR.COSTS
    return f"""<section class="reg" id="products">
      <h2>能下单的产品：按现货价结算、已扣交易成本</h2>
      <p class="lede">ASX 的 NEM 电力合约只有月度/季度/年度，没有日度或周度合约，日前信号在交易所上表达不出来。
        这里把预测变成按现货价结算的实物决策，单位 $/天、1 MW。</p>
      <p class="cap">成本：AEMO 市场费用 ${c['fee']}/MWh（双向）；储能往返效率 {c['eta']:.0%}、衰减 ${c['degr']:g}/MWh 放电量；
        调峰机组边际成本 ${c['srmc']:g}/MWh、启动 ${c['start']:g}/MW/次、最小连续运行 {c['min_up']} 小时；
        可调负荷每天 {c['load_mwh']} MWh、最大 {c['load_mw']:g} MW，和全天平均用电比。
        "AEMO 直用"= 不建模型、直接按预调度排计划，是最重要的对照。区间为 5 天移动块自助 95% 区间。</p>
      {''.join(blocks)}
    </section>"""


def robust_section(rb: dict | None) -> str:
    """偏差交易里天气增量的稳健性：按月、多重比较、按天下注、换训练窗。"""
    if not rb or not rb.get("regions"):
        return ""
    regs = rb["regions"]
    months = sorted({m["month"] for v in regs.values() if v.get("weather")
                     for m in v["weather"]["months"]})
    head = "".join(f"<th>{m[2:]}</th>" for m in months)
    mrows = []
    for k, v in regs.items():
        w = v.get("weather")
        if not w:
            continue
        cell = {m["month"]: m for m in w["months"]}
        tds = "".join(
            f"<td class='num'>{cell[m]['per_day']:+.0f}</td>" if m in cell else "<td>—</td>"
            for m in months)
        lo = w["leave_one_month_out"]
        mrows.append(f"<tr><td class='tier'>{k}</td>{tds}<td class='num'>{w['per_day']:+.1f}</td>"
                     f"<td class='num'>{_n(w['p'], '{:.3f}')}</td><td class='num'>{_n(w['p_holm'], '{:.3f}')}</td>"
                     f"<td class='num'>{'—' if w['top5_share'] is None else ('&gt;100%' if w['top5_share'] > 1 else format(w['top5_share'], '.0%'))}</td>"
                     f"<td class='num'>{lo['per_day']:+.1f}<br><small>去 {lo['month']}</small></td></tr>")
    brows = []
    for k, v in regs.items():
        b = v.get("baseload")
        if not b:
            continue
        brows.append(f"<tr><td class='tier'>{k}</td><td class='num'>{b['per_day']:+.1f}</td>"
                     f"<td class='num'>[{b['ci'][0]:.1f}, {b['ci'][1]:.1f}]</td>"
                     f"<td class='num'>{_n(b['p_holm'], '{:.3f}')}</td><td class='num'>{b['flips']}</td>"
                     f"<td class='num'>{b['ref_hit']:.0%} → {b['hit']:.0%}</td></tr>")
    tds = sorted(rb.get("windows", {}), key=int)
    whead = "".join(f"<th>训练 {t} 天</th>" for t in tds)
    wrows = []
    for k in regs:
        cells = []
        for t in tds:
            x = (rb["windows"][t] or {}).get(k)
            c = (x or {}).get("common")
            cells.append("<td>—</td>" if not c else
                         f"<td class='num'>{c['gain']:+.1f}<br><small>[{c['ci'][0]:.0f}, {c['ci'][1]:.0f}]</small></td>")
        wrows.append(f"<tr><td class='tier'>{k}</td>{''.join(cells)}</tr>")
    start = rb.get("common_start") or ""
    return f"""<section class="reg" id="robust">
      <h2>稳健性：偏差交易里天气多赚的钱，是真的吗</h2>
      <p class="lede">数字都是"加天气（B3）比不加（B2）每天多赚的 $"，1 MW，赌 AEMO 预调度偏差的小时版。</p>
      <h4>一、按月拆开 + 多重比较</h4>
      <p class="cap">p 是移动块自助（5 天一块）的双侧 p；5 个区各检验一次，Holm 校正后的 p 才能直接和 0.05 比。
        "前 5 天占"= 最好的 5 天贡献了总增量的多少，越高越靠少数尖峰日；"去掉最好的月"= 拿掉贡献最大的那个月后剩下的日均。</p>
      <div class="tblwrap"><table class="lad"><thead><tr><th>区</th>{head}<th>全段</th><th>p</th><th>Holm p</th>
        <th>前 5 天占</th><th>去掉最好的月</th></tr></thead><tbody>{''.join(mrows)}</tbody></table></div>
      <h4>二、按天下注</h4>
      <p class="cap">一天只押一次方向：明天全天均价比预调度的全天均价高还是低，按实际全天均价结算。
        这更接近能交易的东西（日基荷互换、调整次日对冲量）。"方向改了"= 加上天气后有多少天押的方向和不加时不同。</p>
      <div class="tblwrap"><table class="lad"><thead><tr><th>区</th><th>加天气 $/天</th><th>95% 区间</th>
        <th>Holm p</th><th>方向改了（天）</th><th>命中率</th></tr></thead><tbody>{''.join(brows)}</tbody></table></div>
      <h4>三、换训练窗长度</h4>
      <p class="cap">训练窗越短，样本外开始得越早；直接比全段等于同时换了样本。这里只比<b>共同样本段</b>（{start} 起），
        只有参数在变。</p>
      <div class="tblwrap"><table class="lad"><thead><tr><th>区</th>{whead}</tr></thead>
        <tbody>{''.join(wrows)}</tbody></table></div>
    </section>"""


def diag_table(r: dict) -> str:
    dg = r.get("diag") or {}
    if dg.get("status") != "ok":
        return ""
    rows, flags = [], []
    for x in dg["tiers"]:
        sh = "—" if x["top1_share"] is None else f"{x['top1_share']:.1%}"
        hot = x["top1_share"] is not None and x["top1_share"] > 0.35
        if hot:
            flags.append(x)
        hi = ' class="hl"' if x["tier"] == "B3 +天气" else ""
        rows.append(f"""<tr{hi}><td class="tier">{x['tier']}</td>
          <td class="num">{x['mae']:.2f}</td><td class="num">{x['med_ae']:.2f}</td>
          <td class="num">{x['p90_ae']:.2f}</td><td class="num">{x['p99_ae']:.2f}</td>
          <td class="num">{x['bias']:+.2f}</td>
          <td class="num{' warnnum' if hot else ''}">{sh}</td></tr>""")
    note = ""
    for x in flags:
        w = "".join(f"<li><code>{html.escape(v['t'])}</code> 实际 "
                    f"<b>{v['actual']:.1f}</b>，预测 {v['pred']:.1f}</li>"
                    for v in x["worst"][:4])
        note += (f'<div class="warn"><b>{html.escape(x["tier"])}：最坏 1% 的行吃掉了 '
                 f'{x["top1_share"]:.0%} 的总误差。</b>这不是"模型不好"，是几个时刻炸了——'
                 f'该查的是那几个时刻，不是换模型。最坏的几行：<ul>{w}</ul></div>')
    return f"""<p class="cap">MAE 是个均值，会把"整体偏"和"几天炸"混成同一个数。
      中位数和分位数拆开看：中位误差就大 = 模型或特征错了；
      中位数正常但最坏 1% 占比高 = 尖峰时刻的外推问题。
      偏差是平均<b>带符号</b>误差，正数表示系统性高估。</p>
      <div class="tblwrap"><table class="lad"><thead><tr><th>层</th><th>MAE</th>
      <th>中位 AE</th><th>p90</th><th>p99</th><th>偏差</th><th>最坏 1% 占比</th>
      </tr></thead><tbody>{''.join(rows)}</tbody></table></div>{note}"""


def section(region: str, r: dict) -> str:
    lvl, tag, why = verdict(r)
    warn = ""
    if r.get("status") == "ok" and not r.get("has_official"):
        warn = """<div class="warn"><b>没有 AEMO 预调度数据。</b>
          B2 这一层是空的，所以 B3 的增量是相对<b>自回归</b>算的，
          没有扣掉官方公开预测。官方预调度里已经含了 AEMO 自己的新能源出力预测，
          那部分信息市场已经定价，不能算作天气带来的优势。
          在补上预调度之前，下面的数字是<b>上界</b>，不是 alpha。</div>"""
    nd = test_days(r)
    if r.get("status") == "ok" and nd is not None and nd < MIN_TEST_DAYS:
        warn += f"""<div class="warn"><b>样本外只有 {nd:.0f} 天。</b>
          少于 {MIN_TEST_DAYS} 天时，日收益的区间宽到几乎什么都包含，
          这一区不下任何结论——数字只用来确认流程跑通。补齐历史电价后重跑。</div>"""
    if r.get("status") != "ok":
        return f"""<section id="r-{region}" class="reg">
          <h2>{region} <span class="chip {lvl}">{tag}</span></h2>
          <p class="empty">{html.escape(why)}</p></section>"""
    return f"""<section id="r-{region}" class="reg">
      <h2>{region} <span class="chip {lvl}">{tag}</span></h2>
      <p class="lede">{html.escape(why).replace('**','')}</p>
      {warn}
      <div class="meta">样本外 {r['n_test']} 小时 · {r['span'][0][:10]} → {r['span'][1][:10]}
        · {r.get('n_refit','?')} 次滚动重拟合 ·
        {('预调度 ' + str((r.get('official_span') or ['', ''])[0])[:10] + ' 起，区间内覆盖 '
          + format(r.get('official_cov_in_range') or 0, '.0%') + ' ·') if r.get('official_span') else '无预调度 ·'}
        {'分小时拟合（24 个模型）' if r.get('per_hour') else '单一共享模型（对照）'}</div>
      <h4>一、基准阶梯</h4>{ladder_table(r)}{RMDAE_NOTE}
      <h4>二、经济评估</h4>{econ_table(r)}
      <h4>三、误差解剖</h4>{diag_table(r)}
      <h4>四、特征取舍</h4>{var_table(r)}{nested_table(r)}
      <h4>五、赌 AEMO 预调度的偏差</h4>{bias_table(r)}
    </section>"""


CSS = """
:root{--bg:#f7f6f3;--fg:#16181c;--dim:#6b7280;--line:#e2e0da;--card:#fff;
 --ac:#0f5d70;--ok:#0f7a4f;--no:#9a3412;--none:#6b7280;--hl:#eef6f8}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){
 --bg:#0e1113;--fg:#e8e6e1;--dim:#9aa1a9;--line:#242a2e;--card:#161a1d;
 --ac:#4fb3c9;--ok:#3fbf87;--no:#e08a5a;--none:#8b939b;--hl:#12262c}}
:root[data-theme=dark]{--bg:#0e1113;--fg:#e8e6e1;--dim:#9aa1a9;--line:#242a2e;
 --card:#161a1d;--ac:#4fb3c9;--ok:#3fbf87;--no:#e08a5a;--none:#8b939b;--hl:#12262c}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
 font:15px/1.6 "Archivo",-apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif;
 -webkit-font-smoothing:antialiased}
.wrap{max-width:1080px;margin:0 auto;padding:24px 24px 80px}
.pnav{display:flex;gap:18px;align-items:center;font-size:14px;margin-bottom:22px;
 padding-bottom:12px;border-bottom:1px solid var(--line)}
.pnav a{color:var(--ac);text-decoration:none}.pnav a:hover{text-decoration:underline}
.pnav .on{color:var(--fg);font-weight:600}
@media(max-width:640px){.wrap{padding:16px 16px 60px}}
header{border-bottom:2px solid var(--fg);padding-bottom:18px;margin-bottom:28px}
h1{font-size:30px;margin:0 0 6px;letter-spacing:-.02em;font-weight:700}
.tagline{color:var(--dim);margin:0;font-size:14px}
.stamp{margin-top:14px;font:12px/1.5 "IBM Plex Mono",ui-monospace,Menlo,monospace;
 color:var(--dim)}
.fixture{display:inline-block;margin-top:12px;padding:8px 14px;border-radius:4px;
 background:#7c2d12;color:#fff;font-weight:700;font-size:13px}
.q{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--ac);
 padding:16px 20px;margin:24px 0;border-radius:0 6px 6px 0}
.q p{margin:0;font:italic 16px/1.6 "Source Serif 4",Georgia,serif}
.grid{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));
 margin:26px 0 36px}
.card{display:block;background:var(--card);border:1px solid var(--line);border-radius:8px;
 padding:16px;text-decoration:none;color:inherit;border-top:3px solid var(--none);
 transition:transform .12s ease,box-shadow .12s ease}
.card:hover{transform:translateY(-2px);box-shadow:0 6px 18px rgba(0,0,0,.09)}
.card.yes{border-top-color:var(--ok)}.card.no{border-top-color:var(--no)}
.card h3{margin:6px 0 10px;font-size:17px;letter-spacing:.02em}
.chip{display:inline-block;font-size:11px;font-weight:700;padding:2px 8px;border-radius:99px;
 background:var(--none);color:#fff;letter-spacing:.04em}
.card.yes .chip,.chip.yes{background:var(--ok)}
.card.no .chip,.chip.no{background:var(--no)}
.chip.none{background:var(--none)}
.big{font:700 30px/1 "IBM Plex Mono",ui-monospace,monospace;letter-spacing:-.02em}
.card.no .big,.card.none .big{color:var(--dim)}
.card.yes .big{color:var(--ok)}
.sub{color:var(--dim);font-size:12px;margin-top:3px}
.mini{margin-top:8px;font:12px "IBM Plex Mono",ui-monospace,monospace;color:var(--dim)}
.why{font-size:12.5px;color:var(--dim);margin:10px 0 0;line-height:1.5}
.reg{margin:44px 0;padding-top:22px;border-top:1px solid var(--line)}
.reg h2{font-size:23px;margin:0 0 8px;display:flex;align-items:center;gap:10px}
.reg h4{font-size:13px;text-transform:uppercase;letter-spacing:.08em;color:var(--dim);
 margin:28px 0 10px;font-weight:700}
.lede{margin:0 0 12px;font-size:15px}
.meta{font:12px "IBM Plex Mono",ui-monospace,monospace;color:var(--dim);margin-bottom:6px}
.warn ul{margin:8px 0 0;padding-left:20px}
.warn li{margin:3px 0}
.warn{background:#fef3c7;color:#5b3a00;border-radius:6px;padding:14px 16px;
 font-size:13.5px;line-height:1.6;margin:12px 0}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]) .warn{
 background:#3a2d0a;color:#f6dfa0}}
:root[data-theme=dark] .warn{background:#3a2d0a;color:#f6dfa0}
.tblwrap{overflow-x:auto;-webkit-overflow-scrolling:touch}
.tblwrap table{min-width:560px}
.tblwrap table.wide{min-width:700px}
@media(max-width:560px){td.tier .note{max-width:200px}}
table.lad{width:100%;border-collapse:collapse;font-size:13.5px;margin:6px 0 4px}
table.lad th{text-align:right;font-size:11.5px;letter-spacing:.03em;
 color:var(--dim);border-bottom:1px solid var(--line);padding:6px 8px;font-weight:700}
table.lad th:first-child,table.lad th:nth-child(2){text-align:left}
table.lad td{padding:9px 8px;border-bottom:1px solid var(--line);vertical-align:top}
table.lad td.num{text-align:right;font:13px "IBM Plex Mono",ui-monospace,monospace;
 white-space:nowrap}
td.dim{color:var(--dim)}
td.warnnum{color:var(--no);font-weight:700}
td.tier{font-weight:600;min-width:130px}
td.tier .note{display:block;font-weight:400;font-size:11.5px;color:var(--dim);
 margin-top:2px;max-width:330px}
tr.hl td{background:var(--hl)}
tr.skip td{color:var(--dim);font-style:italic}
tr.perfect td{color:var(--dim);border-top:1px solid var(--line)}
td.rm{position:relative}
td.rm i{display:block;height:3px;background:var(--ac);margin-top:4px;border-radius:2px;
 opacity:.75}
tr.keep td:first-child{color:var(--ok);font-weight:700}
tr.drop td:first-child{color:var(--dim)}
.cap{font-size:12.5px;color:var(--dim);line-height:1.6;margin:4px 0 10px}
.card.none .chip,.chip.none{background:var(--none)}
.empty{color:var(--dim);font-style:italic}
footer{margin-top:60px;padding-top:24px;border-top:1px solid var(--line);
 font-size:13px;color:var(--dim);line-height:1.7}
footer h3{font-size:13px;text-transform:uppercase;letter-spacing:.08em;color:var(--fg);
 margin:22px 0 8px}
footer ul{margin:6px 0;padding-left:20px}
footer a{color:var(--ac)}
code{font:12.5px "IBM Plex Mono",ui-monospace,monospace;background:var(--hl);
 padding:1px 5px;border-radius:3px}
"""

FOOT = """
<footer>
<h3>口径</h3>
<ul>
<li><b>决策时刻</b>：NEM 日 D 的全部 24 小时，在 D−1 天 <b>{decide_at}</b>（NEM）一次性定计划——
    电池运营商排次日计划的真实做法，也和 EPF 综述的 day-ahead 设定一致。
    AEMO 预调度最远只看到次日 04:00，而且约 12:30 之后才开始覆盖下一天，
    所以"每小时滚动提前 24 小时"的口径会让 04:00–12:30 的小时拿不到官方预测。
    <b>每一个特征在决策时刻都必须已经可见</b>：一小时的结算价要等这一小时结束再过 10 分钟
    才发布；天气预报和 AEMO 预调度都取决策时刻之前发出的版本里最新的一版。
    价格滞后按<b>时间戳</b>取，不按下标平移；天气分位用滚动窗口，不用全样本分位。</li>
<li><b>每个 hour-of-day 各拟合一个模型</b>，不是一个模型跨 24 小时。共享模型会把
    日内曲线压平（NSW1 实测振幅比 0.09），水平贴得准，挑出来的充放电时刻却是噪声。</li>
<li><b>rMAE</b>，不是 MAPE。电价会穿零，MAPE 在那里没有定义。分母固定是
    naive P(d−7,h)。MAE 和 RMSE 同时报——只报一个会掩盖尖峰上的表现。</li>
<li><b>Giacomini-White</b>，不是 Diebold-Mariano。滚动重拟合下校准窗口一直在变，
    DM 的前提（预测方法固定）不成立。方差用 Newey-West 修正，
    否则损失差分的自相关会把 p 值压到荒谬的小。</li>
<li><b>asinh 变换</b>后再拟合，预测夹回训练期见过的范围。电价重尾，
    直接在原尺度上做最小二乘，几个尖峰就能主导整条回归线；
    而 sinh 是指数级的，不夹住的话一个离群输入就能吐出上万块的预测。</li>
<li><b>禁运带</b>：训练集和测试集之间留 {embargo} 天空隙，避免相邻小时把信息漏过去。
    训练窗 {train} 天、每 {step} 天滚动重拟合一次。</li>
<li><b>四道门槛</b>全过才判"可用"：样本外 ≥ {min_days} 天、rMAE 有改善、
    GW p ≤ 0.05、储能日收益增量的 95% 区间下界 &gt; 0。宁可漏掉真信号，
    也不把噪声当 alpha。</li>
</ul>
<h3>为什么是这个结构</h3>
<p>之前问的是"天气和电价相不相关"。那个问题的答案永远是"相关"——天气驱动供给，
供给驱动价格。但它不是交易问题。交易问题是：我已经知道最近的价格，也知道 AEMO
公开发布的预调度，再加上天气，还能<b>多</b>赚多少。所以每一层只比上一层多一组特征，
天气的价值就是最后那一格的增量。</p>
<h3>出处</h3>
<ul>
<li>Lago, Marcjasz, De Schutter, Weron，
  <a href="https://arxiv.org/pdf/2204.11735">Forecasting day-ahead electricity prices:
  A review of state-of-the-art algorithms, best practices and an open-access benchmark</a>
  —— 基准阶梯、rMAE、GW 检验、"统计误差不等于经济价值"都出自这篇。</li>
<li><a href="https://ieeexplore.ieee.org/document/9788043/">The Role of Weather Predictions
  in Electricity Price Forecasting Beyond the Day-Ahead Horizon</a></li>
<li><a href="https://arxiv.org/html/2510.03657">Optimising Battery Energy Storage System
  Trading via Energy Market Operator Price Forecast</a> —— 经济评估用储能套利的依据。</li>
</ul>
<p style="margin-top:20px">{stamp}</p>
<p>代码与方法：<a href="{repo}">{repo}</a></p>
</footer>"""


def render(data: dict) -> str:
    regs = data.get("regions", {})
    is_fix = bool(data.get("is_fixture")) or "fixture" in str(data.get("data_dir", "")).lower()
    cards = "".join(card(k, v) for k, v in regs.items())
    secs = "".join(section(k, v) for k, v in regs.items())
    n_ok = sum(1 for v in regs.values() if verdict(v)[0] == "yes")
    n_run = sum(1 for v in regs.values() if v.get("status") == "ok")
    no_off = [k for k, v in regs.items()
              if v.get("status") == "ok" and not v.get("has_official")]

    short = [k for k, v in regs.items() if v.get("status") == "ok"
             and (test_days(v) or 0) < MIN_TEST_DAYS]
    if n_run == 0:
        head = "还没有一个区跑出可比结果。先把数据接上。"
    elif len(short) == n_run:
        nd = min(test_days(regs[k]) or 0 for k in short)
        head = (f"跑通 {n_run} 个区，但样本外都只有 {nd:.0f} 天左右（门槛 {MIN_TEST_DAYS} 天），"
                "全部判为<b>样本不足</b>。这一版只证明整条流程在真实数据上跑通，"
                "<b>不下任何交易结论</b>。")
    elif n_ok == 0:
        head = (f"跑通 {n_run} 个区，<b>没有一个</b>通过全部四道门槛。"
                "按现在的数据，天气预报还不能作为独立的交易优势。")
    else:
        head = (f"跑通 {n_run} 个区，其中 <b>{n_ok} 个</b>通过全部四道门槛："
                f"样本外 ≥ {MIN_TEST_DAYS} 天、rMAE 有改善、GW 显著、电池日收益增量区间不跨 0。")
    if short and len(short) < n_run:
        head += (f" {'、'.join(short)} 的样本外不足 {MIN_TEST_DAYS} 天，"
                 "判为<b>样本不足</b>。")
    if no_off:
        who = "所有区" if len(no_off) == n_run else "、".join(no_off)
        head += (f" {who}都还没有 AEMO 预调度，天气增量<b>没有扣掉官方预测</b>，"
                 "只能当上界看。")
    spans = [v["span"] for v in regs.values() if v.get("status") == "ok" and v.get("span")]
    window = (f"样本外 {min(s_[0] for s_ in spans)[:10]} → {max(s_[1] for s_ in spans)[:10]}"
              if spans else "")

    fix = ('<div class="fixture">合成数据（fixture）。下面每一个数字都是假的，'
           '只用来验证版面和流程。不要据此下单。</div>' if is_fix else '')
    gen = str(data.get("generated_at", ""))[:16].replace("T", " ")
    stamp = " · ".join(x for x in (
        f"生成于 {gen} UTC", window,
        f"评估时段 {html.escape(str(data.get('hours', '')))}",
        f"前一天 {data.get('decide_at', '13:30')} 定计划",
        "分小时拟合" if data.get("per_hour", True) else "共享模型") if x)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NEM 天气交易台 · 方法与证据</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Archivo:wght@400;600;700&family=IBM+Plex+Mono:wght@400;600&family=Source+Serif+4:ital@1&display=swap" rel="stylesheet">
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
<nav class="pnav"><a href="index.html">← 交易台</a><span class="on">方法与证据</span><a href="{REPO_URL}">代码</a></nav>
<header>
  <h1>方法与证据</h1>
  <p class="tagline">天气预报在价格自回归与 AEMO 官方预调度之上，还剩多少可交易的增量</p>
  {fix}
  <div class="stamp">{stamp}</div>
</header>
<div class="q"><p>“Statistical error measures … may not necessarily reflect
the economic value of reducing prediction errors.”<br>
—— Lago &amp; Weron, EPF 综述。所以这一页每个区都配一块电池实际能多赚多少钱，
不只是误差。</p></div>
<p class="lede">{head}</p>
<div class="grid">{cards}</div>
{futures_section(data.get("futures"))}
{prob_section(regs)}
{products_section(regs)}
{robust_section(data.get("robust"))}
{secs}
{FOOT.format(decide_at=html.escape(str(data.get('decide_at', '13:30'))), stamp=stamp,
             embargo=data.get('embargo_days', 3), train=data.get('train_days', 180),
             step=data.get('step_days', 7), min_days=MIN_TEST_DAYS, repo=REPO_URL)}
</div>
</body>
</html>"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="把 ladder.json 渲染成交易台")
    ap.add_argument("--in", dest="src", default="ladder.json")
    ap.add_argument("--out", default="index.html",
                    help="第一页（交易台）；第二页 method.html 写在同一目录")
    a = ap.parse_args(argv)
    with open(a.src) as f:
        data = json.load(f)
    import os
    import trader
    # 稳健性检验（robust.py 的输出）放在 ladder.json 旁边，有就一起渲染
    rp = os.path.join(os.path.dirname(os.path.abspath(a.src)), "robust.json")
    if os.path.exists(rp) and "robust" not in data:
        with open(rp) as f:
            data["robust"] = json.load(f)
    fp = os.path.join(os.path.dirname(os.path.abspath(a.src)), "futures.json")
    if os.path.exists(fp) and "futures" not in data:
        with open(fp) as f:
            data["futures"] = json.load(f)
    hp = os.path.join(os.path.dirname(os.path.abspath(a.src)), "paper", "health.json")
    if os.path.exists(hp) and "health" not in data:
        with open(hp) as f:
            data["health"] = json.load(f)
    pp = os.path.join(os.path.dirname(os.path.abspath(a.src)), "paper", "summary.json")
    if os.path.exists(pp) and "paper" not in data:
        with open(pp) as f:
            data["paper"] = json.load(f)
    method = os.path.join(os.path.dirname(os.path.abspath(a.out)), "method.html")
    with open(a.out, "w") as f:
        f.write(trader.render_trader(data))
    with open(method, "w") as f:
        f.write(render(data))
    regs = data.get("regions", {})
    ok = [k for k, v in regs.items() if verdict(v)[0] == "yes"]
    print(f"→ {a.out}（交易台）")
    print(f"→ {method}（方法与证据）")
    print(f"  {len(regs)} 个区，{len(ok)} 个判为可用" + (f"：{'、'.join(ok)}" if ok else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

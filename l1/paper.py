"""
前推纸面交易 —— 每天在决策时刻之前出明天的信号，记下来，等实际价出来再结算。
=========================================================================
回测再干净也是事后看的。这里做的是唯一不能作弊的检验：

  · 每天 13:05–13:29（NEM）之间：补数据 → 用截至此刻已公开的信息给**明天**
    24 小时出信号 → 写进 docs/paper/signals.csv → git 提交并推送。
    提交时间戳就是"信号在结果出来之前就已经存在"的证据。
  · 同一次运行里，把已经有实际价的日子结算掉，写 docs/paper/summary.json，
    交易台第一页读它。

信号的口径和回测完全一样（ladder.build(live=True)）：同一个决策时刻、同一套 as-of 规则、
同一组特征、同样的分小时岭回归、同样 180 天训练窗和 3 天禁运带。

用法：
    python l1/paper.py --cron                 # 定时任务调用：自己判断现在该不该干活
    python l1/paper.py --signal --force       # 现在就给明天出信号（不在窗口内会标成 late）
    python l1/paper.py --settle               # 只结算 + 写 summary.json
    python l1/paper.py --status
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

import ladder as L
import realdata as R

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER_DIR = os.path.join(ROOT, "docs", "paper")
SIGNALS = os.path.join(PAPER_DIR, "signals.csv")
SUMMARY = os.path.join(PAPER_DIR, "summary.json")
LOCK = os.path.join(os.environ.get("WQ_DATA_DIR", ROOT), ".paper.lock")

MODELS = ["B0 naive", "B1 自回归", "B2 +官方", "B3 +天气", L.NESTED_NAME]
COL = {"B0 naive": "p_b0", "B1 自回归": "p_b1", "B2 +官方": "p_b2",
       "B3 +天气": "p_b3", L.NESTED_NAME: "p_b3sel"}
WINDOW = ("13:05", "13:29")          # NEM；13:00 那一版预调度要过几分钟才发布
TRAIN_DAYS, EMBARGO_DAYS = 180, 3
NEM = pd.Timedelta(hours=L.NEM_OFFSET_H)
PD_LEAD = (12.0, 40.0)               # 和预调度归档入库时用的时效带一致


def now_utc() -> pd.Timestamp:
    return pd.Timestamp.now("UTC").tz_localize(None)


def nem_day(t) -> pd.Series:
    return (pd.to_datetime(pd.Series(t)) + NEM).dt.floor("D")


def tau_for(target_day: pd.Timestamp, decide_at: str = L.DECIDE_AT_NEM) -> pd.Timestamp:
    """目标 NEM 日的决策时刻（UTC）：前一天 decide_at（NEM）。"""
    hh, mm = (int(x) for x in decide_at.split(":"))
    return pd.Timestamp(target_day) - pd.Timedelta(days=1) + pd.Timedelta(hours=hh, minutes=mm) - NEM


# ------------------------------------------------------------------ 出信号
def signal_from_frame(d: pd.DataFrame, target_day, decide_at: str = L.DECIDE_AT_NEM,
                      train_days: int = TRAIN_DAYS, embargo_days: int = EMBARGO_DAYS
                      ) -> tuple[pd.DataFrame | None, str]:
    """给 target_day（NEM 日）的 24 小时出各模型的预测。d 是 ladder.build(live=True) 的输出。

    只用决策时刻 τ 已经可见的东西：
      · 训练行 = 目标日之前 embargo_days 天再往前 train_days 天，且**结算价在 τ 已发布**；
      · 目标行的特征由 build 按 as-of 规则装好，这里不碰目标行的 rrp。
    """
    target_day = pd.Timestamp(target_day).normalize()
    tau = tau_for(target_day, decide_at)
    d = d.copy()
    d["nd"] = nem_day(d["t"]).to_numpy()
    tgt = d[d["nd"] == target_day].sort_values("t").reset_index(drop=True)
    feats = L.AR_COLS + L.WX_COLS + L.OFF_COLS
    if len(tgt) < 24:
        return None, f"目标日只有 {len(tgt)} 个小时有特征（天气预报或面板不全）"
    miss = {c: int(tgt[c].isna().sum()) for c in feats if tgt[c].isna().any()}
    if miss:
        return None, "目标日特征缺失：" + "、".join(f"{k} {v} 小时" for k, v in miss.items())

    published = d["t"] + L.PRICE_PUBLISH <= tau          # 这一小时的结算价在 τ 已发布
    lo = target_day - pd.Timedelta(days=embargo_days + train_days)
    hi = target_day - pd.Timedelta(days=embargo_days)
    tr = d[(d["nd"] >= lo) & (d["nd"] < hi) & published].dropna(subset=feats + ["rrp"])
    if len(tr) < 24 * 60:
        return None, f"训练行只有 {len(tr)}（< 60 天）"

    out = pd.DataFrame({
        "target_day": str(target_day.date()),
        "t_utc": tgt["t"].dt.strftime("%Y-%m-%d %H:%M"),
        "hour_nem": ((tgt["t"] + NEM).dt.hour).astype(int),
        "aemo": tgt["aemo"].round(2),
    })
    fp = L._fit_predict_ph
    base = L.AR_COLS + L.OFF_COLS
    out[COL["B0 naive"]] = tgt["naive"].round(2)
    out[COL["B1 自回归"]] = np.round(fp(tr, tgt, L.AR_COLS), 2)
    out[COL["B2 +官方"]] = np.round(fp(tr, tgt, base), 2)
    out[COL["B3 +天气"]] = np.round(fp(tr, tgt, base + L.WX_COLS), 2)
    sub = L.select_features(tr, base, fp, embargo_days)
    out[COL[L.NESTED_NAME]] = np.round(fp(tr, tgt, base + sub), 2)
    out["sel"] = "+".join(sub) if sub else "none"
    out["train_rows"] = len(tr)
    out["train_last"] = str(tr["t"].max())[:16]
    return out, "ok"


def make_signals(con, target_day, decide_at: str = L.DECIDE_AT_NEM) -> pd.DataFrame:
    gen = now_utc()
    tau = tau_for(target_day, decide_at)
    rows = []
    for site_id, region, *_ in L.SITES:
        try:
            d = L.build(con, region, site_id, None, decide_at, live=True)
            sig, why = signal_from_frame(d, target_day, decide_at)
        except Exception as e:                                  # noqa: BLE001
            sig, why = None, f"{type(e).__name__}: {e}"
        if sig is None:
            print(f"  {region}: 没出信号 —— {why}")
            rows.append(pd.DataFrame([{"target_day": str(pd.Timestamp(target_day).date()),
                                       "region": region, "status": why[:200]}]))
            continue
        sig.insert(1, "region", region)
        sig["status"] = "ok"
        rows.append(sig)
        up = int((sig[COL["B3 +天气"]] > sig["aemo"]).sum())
        flip = int((np.sign(sig[COL["B3 +天气"]] - sig["aemo"])
                    != np.sign(sig[COL["B2 +官方"]] - sig["aemo"])).sum())
        print(f"  {region}: 24 小时 ✓  天气版做多 {up} 小时 / 做空 {24 - up} 小时，"
              f"天气改了 {flip} 个小时的方向；选中的天气变量：{sig['sel'].iloc[0]}")
    out = pd.concat(rows, ignore_index=True)
    out["generated_at_utc"] = gen.strftime("%Y-%m-%d %H:%M:%S")
    out["decide_at_utc"] = tau.strftime("%Y-%m-%d %H:%M")
    out["late"] = bool(gen > tau)
    return out


def append_signals(sig: pd.DataFrame) -> None:
    """同一目标日已有正常信号就不覆盖——第一版就是正式版，事后不能改。"""
    os.makedirs(PAPER_DIR, exist_ok=True)
    if os.path.exists(SIGNALS):
        old = pd.read_csv(SIGNALS, dtype=str)
        have = set(zip(old.loc[old["status"] == "ok", "target_day"],
                       old.loc[old["status"] == "ok", "region"]))
        sig = sig[[(a, b) not in have for a, b in zip(sig["target_day"], sig["region"])]]
        # 之前失败的记录，这次成功了就替换掉
        okk = set(zip(sig.loc[sig["status"] == "ok", "target_day"], sig.loc[sig["status"] == "ok", "region"]))
        old = old[[(a, b) not in okk or s == "ok"
                   for a, b, s in zip(old["target_day"], old["region"], old["status"])]]
        sig = pd.concat([old, sig.astype(str)], ignore_index=True)
    sig.to_csv(SIGNALS, index=False)


# ------------------------------------------------------------------ 结算
def _prices(con, regions) -> dict[str, pd.Series]:
    out = {}
    for r in regions:
        mk = con.execute("""
            SELECT interval_time AS t, value FROM truth
            WHERE site_id = ? AND variable = 'rrp'
            QUALIFY row_number() OVER (PARTITION BY interval_time
                                       ORDER BY (quality='final') DESC, known_at DESC) = 1
        """, [r]).df()
        out[r] = mk.set_index(pd.to_datetime(mk["t"]))["value"].sort_index()
    return out


def settle(con, signals: pd.DataFrame | None = None, prices: dict | None = None) -> dict:
    """逐日逐区结算，全部从 signals.csv + 实际价重算（无状态，重跑结果一样）。"""
    if signals is None:
        if not os.path.exists(SIGNALS):
            return {"status": "empty"}
        signals = pd.read_csv(SIGNALS)
    s = signals[signals["status"] == "ok"].copy()
    if s.empty:
        return {"status": "empty", "signals": 0}
    for c in ["aemo"] + list(COL.values()):
        s[c] = pd.to_numeric(s[c], errors="coerce")
    s["t"] = pd.to_datetime(s["t_utc"])
    s["late"] = s["late"].astype(str).str.lower() == "true"
    px = prices if prices is not None else _prices(con, sorted(s["region"].unique()))
    s["y"] = [px[r].get(t, np.nan) for r, t in zip(s["region"], s["t"])]

    days = []
    for (day, reg), g in s.groupby(["target_day", "region"]):
        g = g.sort_values("t")
        rec = {"day": day, "region": reg, "late": bool(g["late"].any()),
               "settled": bool(g["y"].notna().sum() >= 24)}
        if rec["settled"]:
            y, a = g["y"].to_numpy(float), g["aemo"].to_numpy(float)
            rec["short"] = round(float((a - y).sum()), 2)
            for m, c in COL.items():
                p = g[c].to_numpy(float)
                if not np.isfinite(p).all():
                    continue
                key = c[2:]
                if m != "B0 naive":
                    rec[f"bias_{key}"] = round(float((np.sign(p - a) * (y - a)).sum()), 2)
                    rec[f"day_{key}"] = round(float(np.sign(p.mean() - a.mean())
                                                    * (y.mean() - a.mean()) * len(y)), 2)
                v, ci, di = L._best_cycle(p, L.BATT_HOURS, L.BATT_ETA, L.BATT_COST)
                rec[f"batt_{key}"] = 0.0 if ci is None else round(float(
                    L.BATT_POWER_MW * (L.BATT_ETA * y[di].sum() - y[ci].sum()
                                       - L.BATT_COST * L.BATT_HOURS)), 2)
            v, ci, di = L._best_cycle(y, L.BATT_HOURS, L.BATT_ETA, L.BATT_COST)
            rec["batt_perfect"] = 0.0 if ci is None else round(float(
                L.BATT_POWER_MW * (L.BATT_ETA * y[di].sum() - y[ci].sum()
                                   - L.BATT_COST * L.BATT_HOURS)), 2)
            # 能下单的产品（已扣交易成本），口径和回测里的 products.py 一样
            import products as PR
            srcs = {"aemo": a, "perfect": y}
            for m, c in COL.items():
                p = g[c].to_numpy(float)
                if np.isfinite(p).all():
                    srcs[c[2:]] = p
            for k, (_, _, fn) in PR.PRODUCTS.items():
                for name, p in srcs.items():
                    rec[f"{k}_{name}"] = round(float(fn(p, y, PR.COSTS)), 2)
        days.append(rec)
    return {"status": "ok", "days": days}


def _agg(days: list[dict], reg: str) -> dict:
    ds = [x for x in days if x["region"] == reg and x["settled"] and not x["late"]]
    out = {"n": len(ds), "first": ds[0]["day"] if ds else None, "last": ds[-1]["day"] if ds else None}
    if not ds:
        return out
    def tot(k):
        v = [x.get(k) for x in ds if x.get(k) is not None]
        return round(float(np.sum(v)), 2) if v else None
    for k in ("short", "bias_b2", "bias_b3", "bias_b3sel", "day_b2", "day_b3",
              "batt_b0", "batt_b1", "batt_b2", "batt_b3", "batt_perfect"):
        out[k] = tot(k)
    import products as PR
    out["products"] = {}
    for k in PR.PRODUCTS:
        out["products"][k] = {m: tot(f"{k}_{m}") for m in
                              ("aemo", "b0", "b1", "b2", "b3", "b3sel", "perfect")}
    g = np.array([x["bias_b3"] - x["bias_b2"] for x in ds if "bias_b3" in x and "bias_b2" in x])
    out["w_gain_total"] = round(float(g.sum()), 2) if len(g) else None
    out["w_gain_per_day"] = round(float(g.mean()), 2) if len(g) else None
    out["w_gain_win_days"] = round(float((g > 0).mean()), 3) if len(g) else None
    out["w_gain_cum"] = [round(float(v), 2) for v in np.cumsum(g)] if len(g) else []
    out["w_gain_dates"] = [x["day"] for x in ds if "bias_b3" in x and "bias_b2" in x]
    return out


def write_summary(con) -> dict:
    st = settle(con)
    sig = pd.read_csv(SIGNALS) if os.path.exists(SIGNALS) else pd.DataFrame()
    summ = {"generated_at": now_utc().strftime("%Y-%m-%d %H:%M"), "status": st.get("status"),
            "regions": {}, "latest": None}
    if st.get("status") == "ok":
        regs = sorted({x["region"] for x in st["days"]})
        summ["regions"] = {r: _agg(st["days"], r) for r in regs}
        summ["days"] = st["days"]
    if not sig.empty:
        ok = sig[sig["status"] == "ok"]
        if not ok.empty:
            last = ok["target_day"].max()
            cur = ok[ok["target_day"] == last]
            summ["latest"] = {"target_day": last,
                              "generated_at_utc": str(cur["generated_at_utc"].iloc[0]),
                              "late": str(cur["late"].iloc[0]).lower() == "true",
                              "regions": {}}
            for r, g in cur.groupby("region"):
                g = g.sort_values("hour_nem")
                summ["latest"]["regions"][r] = {
                    "hour": g["hour_nem"].astype(int).tolist(),
                    "aemo": pd.to_numeric(g["aemo"]).round(1).tolist(),
                    "b2": pd.to_numeric(g[COL["B2 +官方"]]).round(1).tolist(),
                    "b3": pd.to_numeric(g[COL["B3 +天气"]]).round(1).tolist(),
                    "sel": str(g["sel"].iloc[0])}
        bad = sig[sig["status"] != "ok"]
        summ["failures"] = bad[["target_day", "region", "status"]].tail(10).to_dict("records")
        summ["start"] = str(sig["target_day"].min())
    os.makedirs(PAPER_DIR, exist_ok=True)
    with open(SUMMARY, "w") as f:
        json.dump(summ, f, ensure_ascii=False, indent=1)
    return summ


# ------------------------------------------------------------------ 每日流程
def _run(cmd: list[str], timeout: int = 900) -> bool:
    print("  $ " + " ".join(cmd), flush=True)
    try:
        r = subprocess.run(cmd, cwd=ROOT, timeout=timeout)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        print("    超时")
        return False


def refresh_data() -> None:
    """补数据。顺序有讲究：天气和预调度先抓，确保它们的 known_at 落在决策时刻之前。"""
    import ingest
    import aemo
    t0 = time.time()
    try:
        ingest.live(models=list(L.WX_MODELS), past_days=1, forecast_days=3)
    except Exception as e:                                       # noqa: BLE001
        print(f"  天气预报抓取失败：{e}")
    try:
        aemo.ingest_predispatch_current(4, PD_LEAD)
    except Exception as e:                                       # noqa: BLE001
        print(f"  预调度抓取失败：{e}")
    today = (now_utc() + NEM).normalize()
    try:
        aemo.ingest_daily_archive(str((today - pd.Timedelta(days=5)).date()),
                                  str((today - pd.Timedelta(days=1)).date()))
    except Exception as e:                                       # noqa: BLE001
        print(f"  日归档电价失败：{e}")
    try:
        aemo.ingest_current(aemo.CUR_DISPATCH, 16, "nemweb_current")
    except Exception as e:                                       # noqa: BLE001
        print(f"  当日电价失败：{e}")
    print(f"  补数据用时 {time.time() - t0:.0f}s")


def render_and_push(msg: str) -> None:
    py = sys.executable
    if os.path.exists(os.path.join(ROOT, "docs", "ladder.json")):
        _run([py, "l1/desk.py", "--in", "docs/ladder.json", "--out", "docs/index.html"])
    _run(["git", "add", "docs/paper", "docs/index.html", "docs/method.html"])
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode == 0:
        print("  没有新改动")
        return
    _run(["git", "commit", "-q", "-m", msg])
    if not _run(["git", "push", "-q", "origin", "HEAD"], timeout=180):
        print("  推送失败（多半是这个环境里没有 GitHub 登录）；提交已在本地，下次手动 bash publish.sh 会一起推上去")


def in_window(now: pd.Timestamp | None = None) -> bool:
    n = (now or now_utc()) + NEM
    hm = n.strftime("%H:%M")
    return WINDOW[0] <= hm <= WINDOW[1]


def have_signal(target_day: str) -> bool:
    if not os.path.exists(SIGNALS):
        return False
    s = pd.read_csv(SIGNALS, dtype=str)
    ok = s[(s["target_day"] == target_day) & (s["status"] == "ok")]
    return ok["region"].nunique() >= len(L.SITES)


def cron() -> int:
    now = now_utc()
    target = ((now + NEM).normalize() + pd.Timedelta(days=1))
    tday = str(target.date())
    if not in_window(now):
        return 0
    if have_signal(tday):
        return 0
    # 加锁：一次运行要几分钟，下一个 10 分钟的触发不能叠上来
    try:
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode()); os.close(fd)
    except FileExistsError:
        if time.time() - os.path.getmtime(LOCK) < 3600:
            print(f"{now} 上一次还在跑，跳过")
            return 0
        os.remove(LOCK)
        return cron()
    try:
        print(f"\n===== {now:%Y-%m-%d %H:%M} UTC · 给 {tday} 出信号 =====", flush=True)
        refresh_data()
        import store
        con = store.connect()
        sig = make_signals(con, target)
        append_signals(sig)
        write_summary(con)
        con.close()
        n_ok = int((sig["status"] == "ok").sum() // 24)
        render_and_push(f"Paper signal for {tday} ({n_ok} regions)")
    finally:
        os.remove(LOCK)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="前推纸面交易")
    ap.add_argument("--cron", action="store_true", help="定时任务入口：只在 13:05–13:29（NEM）之间干活")
    ap.add_argument("--signal", action="store_true", help="现在就给明天（或 --day）出信号")
    ap.add_argument("--day", default=None, help="目标 NEM 日 YYYY-MM-DD，默认明天")
    ap.add_argument("--force", action="store_true", help="不在窗口内也出（会标成 late，不计入成绩）")
    ap.add_argument("--refresh", action="store_true", help="补数据；和 --signal 一起用时先补再出信号")
    ap.add_argument("--settle", action="store_true", help="结算并写 summary.json")
    ap.add_argument("--push", action="store_true", help="完成后渲染交易台并 git 推送")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args(argv)

    if a.cron:
        return cron()
    import store
    if a.refresh and not a.signal:
        refresh_data()            # 只补数据：预调度、当日电价、最近几天的日归档、ECMWF 预报
        return 0
    if a.signal:
        target = pd.Timestamp(a.day) if a.day else (now_utc() + NEM).normalize() + pd.Timedelta(days=1)
        if not a.force and not in_window():
            print("现在不在 13:05–13:29（NEM）窗口内。要出也可以加 --force，但会标成 late、不计入成绩。")
            return 1
        if a.refresh:
            refresh_data()
        con = store.connect()
        sig = make_signals(con, target)       # 决策时刻之后才生成的会自动标成 late
        append_signals(sig)
        write_summary(con)
        con.close()
        if a.push:
            render_and_push(f"Paper signal for {target.date()}")
        return 0
    if a.settle or a.status:
        con = store.connect()
        s = write_summary(con)
        con.close()
        for r, v in (s.get("regions") or {}).items():
            print(f"  {r}: 结算 {v['n']} 天" + (f"，天气每天 {v['w_gain_per_day']:+.1f}" if v.get("w_gain_per_day") is not None else ""))
        if s.get("latest"):
            print(f"  最新信号：{s['latest']['target_day']}（生成于 {s['latest']['generated_at_utc']} UTC）")
        if a.push:
            render_and_push("Paper settle")
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

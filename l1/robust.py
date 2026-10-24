"""
稳健性检验 —— 天气在"赌 AEMO 预调度偏差"里多赚的那部分，是不是真的。
=====================================================================
只读 ladder.json（不重新拟合），回答四件事：

  1. 按月拆开：钱是均匀来的，还是一两个月撑起来的？去掉任何一个月还剩多少？
  2. 集中度：最好的 5 天占了总增量的多少？
  3. 多重比较：5 个区各检验一次，Holm 校正后还显著吗？
  4. 换口径：按天下注（一天押一次全天均价）还在不在；换训练窗长度还在不在。

用法：
    python l1/robust.py --in docs/ladder.json --alt 90=lad90.json 120=lad120.json --out docs/robust.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

W = "B3 +天气"
N = "B3 选特征"


def boot_p(x: np.ndarray, blk: int = 5, B: int = 4000, seed: int = 11) -> float | None:
    """移动块自助的双侧 p：把序列平移到均值 0 再抽，看抽到的均值有多常比实测更极端。"""
    n = len(x)
    if n < 3 * blk:
        return None
    x0 = x - x.mean()
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / blk))
    starts = rng.integers(0, n - blk + 1, size=(B, nb))
    idx = (starts[:, :, None] + np.arange(blk)[None, None, :]).reshape(B, -1)[:, :n]
    m = x0[idx].mean(1)
    return float((np.abs(m) >= abs(x.mean())).mean())


def holm(ps: dict[str, float | None]) -> dict[str, float | None]:
    items = sorted((p, k) for k, p in ps.items() if p is not None)
    m, out, run = len(items), {}, 0.0
    for i, (p, k) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = round(run, 4)
    for k, p in ps.items():
        out.setdefault(k, None)
    return out


def monthly(dates: list[str], d: np.ndarray) -> list[dict]:
    months = sorted({x[:7] for x in dates})
    tot = d.sum()
    out = []
    for mth in months:
        m = np.array([x[:7] == mth for x in dates])
        out.append({"month": mth, "days": int(m.sum()),
                    "per_day": round(float(d[m].mean()), 2),
                    "share": None if tot == 0 else round(float(d[m].sum() / tot), 3),
                    "pos": bool(d[m].sum() > 0)})
    return out


def leave_one_out(dates: list[str], d: np.ndarray) -> dict:
    months = sorted({x[:7] for x in dates})
    worst = None
    for mth in months:
        keep = np.array([x[:7] != mth for x in dates])
        v = float(d[keep].mean())
        if worst is None or v < worst[1]:
            worst = (mth, v)
    return {"month": worst[0], "per_day": round(worst[1], 2)}


def analyse_region(r: dict) -> dict | None:
    bt = r.get("bias") or {}
    if bt.get("status") != "ok":
        return None
    dl, ref = bt["daily"], bt["ref"]
    dates = dl["dates"]
    out = {"ref": ref, "days": len(dates)}
    for key, name in (("weather", W), ("nested", N)):
        if name not in dl or ref not in dl:
            continue
        d = np.asarray(dl[name], float) - np.asarray(dl[ref], float)
        top = np.sort(d)[::-1]
        tot = d.sum()
        out[key] = {
            "per_day": round(float(d.mean()), 2),
            "p": boot_p(d),
            "win_days": round(float((d > 0).mean()), 3),
            "top5_share": None if tot <= 0 else round(float(top[:5].sum() / tot), 3),
            "months": monthly(dates, d),
            "pos_months": sum(m["pos"] for m in monthly(dates, d)),
            "n_months": len(monthly(dates, d)),
            "leave_one_month_out": leave_one_out(dates, d),
        }
    bl = bt.get("baseload") or {}
    rows = {x["tier"]: x for x in bl.get("rows", [])}
    if W in rows and "gain" in rows[W]:
        d = np.asarray(bl["daily"][W], float) - np.asarray(bl["daily"][ref], float)
        out["baseload"] = {"per_day": rows[W]["gain"], "ci": rows[W]["gain_ci"],
                           "p": boot_p(d), "flips": rows[W].get("flips"),
                           "hit": rows[W].get("hit"), "ref_hit": (rows.get(ref) or {}).get("hit"),
                           "nested": None if N not in rows or "gain" not in rows[N]
                           else {"per_day": rows[N]["gain"], "ci": rows[N]["gain_ci"]}}
    return out


def _boot_ci(x: np.ndarray):
    import ladder as LD
    ci = LD._block_boot(x)
    return None if ci is None else [round(v, 2) for v in ci]


def window_row(r: dict, start: str | None = None) -> dict | None:
    """一个训练窗长度下的结果。start 给定时另算"共同样本段"（日期 ≥ start）：
    训练窗越短、样本外越早开始，直接比全段等于同时换了样本；共同段才是只换参数。"""
    bt = r.get("bias") or {}
    rows = {x["tier"]: x for x in bt.get("rows", [])}
    w = rows.get(W)
    if bt.get("status") != "ok" or not w or "gain" not in w:
        return None
    common = None
    dl = bt.get("daily") or {}
    if start and W in dl and bt["ref"] in dl:
        dates = np.array(dl["dates"])
        d = (np.asarray(dl[W], float) - np.asarray(dl[bt["ref"]], float))[dates >= start]
        if len(d) >= 30:
            common = {"days": int(len(d)), "gain": round(float(d.mean()), 2), "ci": _boot_ci(d)}
    n = rows.get(N) or {}
    ec = {x["tier"]: x for x in (r.get("economics") or {}).get("rows", [])}
    b3 = next((x for x in r.get("tiers", []) if x["tier"] == W), {})
    return {"days": bt["days"], "gain": w["gain"], "ci": w["gain_ci"], "common": common,
            "nested": n.get("gain"), "nested_ci": n.get("gain_ci"),
            "rmae_gain": b3.get("gain_rmae"), "gw_p": b3.get("gw_p"),
            "battery": (ec.get(W) or {}).get("gain"), "battery_ci": (ec.get(W) or {}).get("gain_ci")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="天气增量的稳健性检验（只读 ladder.json）")
    ap.add_argument("--in", dest="src", default="docs/ladder.json")
    ap.add_argument("--alt", nargs="*", default=[], metavar="训练天数=ladder.json")
    ap.add_argument("--out", default="docs/robust.json")
    a = ap.parse_args(argv)
    with open(a.src) as f:
        base = json.load(f)
    regs = base.get("regions", {})
    res = {"train_days": base.get("train_days"), "regions": {}, "windows": {}}
    for k, r in regs.items():
        x = analyse_region(r)
        if x:
            res["regions"][k] = x
    for key in ("weather", "nested"):
        adj = holm({k: v.get(key, {}).get("p") for k, v in res["regions"].items()})
        for k, v in res["regions"].items():
            if key in v:
                v[key]["p_holm"] = adj.get(k)
    adj = holm({k: (v.get("baseload") or {}).get("p") for k, v in res["regions"].items()})
    for k, v in res["regions"].items():
        if "baseload" in v:
            v["baseload"]["p_holm"] = adj.get(k)

    wins = {str(base.get("train_days")): base}
    for spec in a.alt:
        td, path = spec.split("=", 1)
        with open(path) as f:
            wins[td] = json.load(f)
    start = None
    for r in regs.values():
        dl = (r.get("bias") or {}).get("daily") or {}
        if dl.get("dates"):
            start = min(start or dl["dates"][0], dl["dates"][0])
    res["common_start"] = start
    for td, dd in sorted(wins.items(), key=lambda kv: int(kv[0])):
        res["windows"][td] = {k: window_row(r, start) for k, r in dd.get("regions", {}).items()}
    # 汇总成每区一张"检查单"，交易台直接用
    for k, v in res["regions"].items():
        w = v.get("weather")
        if not w:
            continue
        ws = [res["windows"][td].get(k) for td in res["windows"]]
        ws = [x["common"] for x in ws if x and x.get("common")]
        v["checks"] = {
            "holm": w["p_holm"] is not None and w["p_holm"] <= 0.05 and w["per_day"] > 0,
            "months": w["pos_months"] >= max(1, int(np.ceil(0.7 * w["n_months"]))),
            "leave_out": w["leave_one_month_out"]["per_day"] > 0,
            "windows": bool(ws) and all(x["gain"] > 0 for x in ws),
            "windows_sig": sum(1 for x in ws if x["ci"] and x["ci"][0] > 0),
            "windows_n": len(ws),
            "not_concentrated": w["top5_share"] is not None and w["top5_share"] < 0.5,
            "baseload": bool(v.get("baseload")) and v["baseload"]["ci"][0] > 0,
        }

    with open(a.out, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)

    # ---- 文字报告 ----
    print("赌 AEMO 偏差：天气（B3）相对", next(iter(res["regions"].values()))["ref"], "每天多赚多少")
    print(f"{'区':<6s}{'$/天':>8s}{'p':>8s}{'Holm p':>8s}{'赢的天':>8s}{'前5天占':>8s}"
          f"{'正的月':>8s}{'去掉最好的月':>16s}")
    for k, v in res["regions"].items():
        w = v.get("weather")
        if not w:
            continue
        lo = w["leave_one_month_out"]
        print(f"{k:<6s}{w['per_day']:>8.1f}{w['p']:>8.3f}{w['p_holm']:>8.3f}"
              f"{w['win_days']:>8.0%}{'—' if w['top5_share'] is None else format(w['top5_share'], '.0%'):>8s}"
              f"{w['pos_months']:>5d}/{w['n_months']:<2d}{lo['per_day']:>10.1f}（去 {lo['month']}）")
    for k, v in res["regions"].items():
        w = v.get("weather")
        if w:
            print(f"\n{k} 按月：" + "  ".join(f"{m['month'][2:]} {m['per_day']:+.0f}" for m in w["months"]))
    print("\n按天下注（一天押一次全天均价）")
    for k, v in res["regions"].items():
        b = v.get("baseload")
        if b:
            print(f"  {k:<6s}{b['per_day']:>+8.1f} [{b['ci'][0]:.1f}, {b['ci'][1]:.1f}]"
                  f"  p {b['p']:.3f}  Holm {b['p_holm']:.3f}  方向改了 {b['flips']} 天"
                  f"  命中 {b['ref_hit']:.0%}→{b['hit']:.0%}")
    print("\n换训练窗")
    for td, rows in res["windows"].items():
        print(f"  训练 {td} 天")
        for k, x in rows.items():
            if x:
                cm = x.get("common")
                cs = "" if not cm else f"  共同段 {cm['gain']:+.1f} [{cm['ci'][0]:.1f}, {cm['ci'][1]:.1f}]"
                print(f"    {k:<6s}样本外 {x['days']:>3d} 天{cs}  偏差交易 {x['gain']:>+7.1f} "
                      f"[{x['ci'][0]:.1f}, {x['ci'][1]:.1f}]  嵌套 {x['nested'] if x['nested'] is not None else '—'}"
                      f"  电池 {x['battery']:+.2f}  rMAE {x['rmae_gain']:+.4f} (p {x['gw_p']})")
    print(f"\n→ {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

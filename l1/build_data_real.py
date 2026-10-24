"""
用真实（或 fixture）L1 数据重跑整条评估管道，产出运行台的 data.json。
====================================================================
与 build_data.py 的唯一区别是**数据来源**：
  truth_series() / make_forecast()  →  realdata.load_backtest()
engine.py 里的 DM 检验、MCS、共形、pinball、CRPS、BESS DP、POT-GPD 一行没改。

    WQ_DATA_DIR=../data_fixture python build_data_real.py --out ../data_real.json
    WQ_DATA_DIR=../data        python build_data_real.py --out ../data.json

模型键保持与合成版一致（ecmwf/gfs/qra/...），这样网页不用改；
真实模型 id 到这些键的映射见 KEY_MAP。
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import engine as E  # noqa: E402
import price_link as PL  # noqa: E402
import realdata as R  # noqa: E402
import store  # noqa: E402
from config import SITES  # noqa: E402

VARS = R.VARS
VAR_META = {"ghi": {"label": "地面总辐照 GHI", "unit": "W/m²", "short": "GHI"},
            "wind100": {"label": "100m 风速", "unit": "m/s", "short": "WS100"},
            "temp": {"label": "2m 气温", "unit": "°C", "short": "T2M"}}

# 真实模型 id → 网页使用的稳定键
REGION_CN = {"NSW1": "新南威尔士", "QLD1": "昆士兰", "VIC1": "维多利亚",
             "SA1": "南澳", "TAS1": "塔斯马尼亚"}

KEY_MAP = {"persist": "persist", "clim": "clim", "gfs_global": "gfs",
           "icon_global": "icon", "gem_global": "gem", "ecmwf_ifs025": "ecmwf",
           "mm_mean": "mm", "qra": "qra"}
SHORT = {"persist": "PERS", "clim": "CLIM", "gfs": "GFS", "icon": "ICON",
         "gem": "GEM", "ecmwf": "IFS", "mm": "MM", "qra": "QRA"}
ORDER = ["persist", "clim", "gem", "icon", "gfs", "ecmwf", "mm", "qra"]
TAUS = [5, 10, 25, 50, 75, 90, 95]
NOMINALS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95]


def _ens(p: R.Panel, var: str) -> np.ndarray:
    """多模式集合，成员按模型天然跨变量耦合——比合成版更真。

    成员名册取面板实际用到的模型，不是写死的 R.NWP：只回填了一家时
    集合退化成单列，分位区间宽度为 0，共形 λ 会把它撑开——数值上仍然成立，
    只是"多模式离散度"这个信息量没有了。calibration.n_members 会如实写 1。
    """
    ms = p.cov.get("models_used") or R.NWP
    return np.column_stack([p.preds[m][var] for m in ms])


def _clip(var, a):
    if var == "ghi":
        return np.clip(a, 0.0, None)
    if var == "wind100":
        return np.clip(a, 0.05, None)
    return a


def build(con, args) -> dict:
    t0 = time.time()
    OUT = {"regions": [], "leaderboard": {}, "calibration": {}, "dm": {},
           "lead": {}, "regimes": {}, "event": {}, "decision": {}, "vre_norm": {}}
    panels = {}

    # ============================================================ 装配
    for s in SITES:
        p = R.fill_persist_head(R.load_backtest(
            con, s[0], lead_day=args.lead_day, strict=args.strict,
            final_only=not args.allow_provisional))
        panels[p.region] = p
        print(f"[{time.time()-t0:5.1f}s] {p.region:5s} {len(p):>6d} 小时 "
              f"({len(p)/24:.0f} 天)  {p.cov['span'][0][:10]} → {p.cov['span'][1][:10]}"
              f"  缺口 {p.cov['gaps_h']}h")

    # 自动识别数据来源：fixture 的内容是合成的，必须和真数据区分开，否则页面会说谎
    srcs = set(con.execute("SELECT DISTINCT source FROM truth").df()["source"].tolist())
    is_fixture = any(str(x).startswith("fixture") for x in srcs)
    print(f"[{time.time()-t0:5.1f}s] truth 来源: {', '.join(sorted(map(str, srcs)))}"
          f"  →  {'FIXTURE（内容合成）' if is_fixture else '真实数据'}")

    days = min(len(p) for p in panels.values()) // 24
    dec_days = min(args.decision_days, days - 30)
    half_note = "第 1 段标定 / 第 2 段检验"

    for region, p in panels.items():
        n = len(p)
        half = n // 2
        cal_s, tst_s = slice(0, half), slice(half, n)
        OUT["leaderboard"][region] = {}
        OUT["calibration"][region] = {}
        OUT["dm"][region] = {}
        OUT["lead"][region] = {}
        OUT["regimes"][region] = {}

        for var in VARS:
            truth = p.truth[var]
            ens = _ens(p, var)                       # (n, 4)
            med = np.median(ens, axis=1)
            halfw = (ens.max(axis=1) - ens.min(axis=1)) / 2.0

            # ---------- L4 分裂共形 ----------
            cov = []
            for nominal in NOMINALS:
                a = (100 - nominal) / 2.0
                lo = np.percentile(ens, a, axis=1); hi = np.percentile(ens, 100 - a, axis=1)
                hw = (hi - lo) / 2.0
                raw = float(np.mean((truth[tst_s] >= lo[tst_s]) & (truth[tst_s] <= hi[tst_s]))) * 100
                lam = E.conformal_lambda(truth[cal_s], med[cal_s], hw[cal_s], 1 - nominal / 100.0)
                clo, chi = med[tst_s] - lam * hw[tst_s], med[tst_s] + lam * hw[tst_s]
                con_ = float(np.mean((truth[tst_s] >= clo) & (truth[tst_s] <= chi))) * 100
                cov.append({"nominal": nominal, "raw": round(raw, 1),
                            "conformal": round(con_, 1), "lam": round(lam, 3)})
            lam90 = E.conformal_lambda(truth[cal_s], med[cal_s], halfw[cal_s], 0.10)
            ens_c = _clip(var, med[:, None] + lam90 * (ens - med[:, None]))

            pit_raw = np.mean(ens[tst_s] < truth[tst_s, None], axis=1)
            pit_con = np.mean(ens_c[tst_s] < truth[tst_s, None], axis=1)
            h_raw, _ = np.histogram(pit_raw, bins=10, range=(0, 1))
            h_con, _ = np.histogram(pit_con, bins=10, range=(0, 1))

            pb = [{"tau": t_,
                   "raw": round(E.pinball(truth[tst_s], np.percentile(ens[tst_s], t_, axis=1), t_ / 100), 4),
                   "conformal": round(E.pinball(truth[tst_s], np.percentile(ens_c[tst_s], t_, axis=1), t_ / 100), 4)}
                  for t_ in TAUS]

            track, W = [], 60 * 24
            for d in range(0, max(1, n // 24 - 60), 7):
                sl = slice(d * 24, d * 24 + W)
                if sl.stop > n:
                    break
                track.append({"d": d + 60,
                              "lam": round(E.conformal_lambda(truth[sl], med[sl], halfw[sl], 0.10), 3)})

            OUT["calibration"][region][var] = {
                "coverage": cov, "pit_raw": [int(v) for v in h_raw],
                "pit_con": [int(v) for v in h_con], "pit_n": int(len(pit_raw)),
                "lambda90": round(lam90, 3), "pinball": pb, "lambda_track": track,
                "n_members": int(ens.shape[1]), "split_note": half_note}

            # ---------- L5 排行榜 ----------
            # 基准取实际存在的第一家（通常是 ECMWF）。硬编码 ecmwf_ifs025 会在
            # 只回填了别家时 KeyError，而排行榜本来就该只列真有的模型。
            used = p.cov.get("models_used") or ["ecmwf_ifs025"]
            ref_m = used[0]
            base = p.preds[ref_m][var]
            mae_b = E.mae(base, truth)
            rows, Lmat, keys = [], [], []
            order = ["persist", "clim", "gem_global", "icon_global", "gfs_global",
                     "ecmwf_ifs025", "mm_mean", "qra"]
            for mid in [m for m in order if m in p.preds]:
                k = KEY_MAP[mid]
                y = p.preds[mid][var]
                m_ = E.mae(y, truth)
                nm, fam, prob = R.MODEL_META[mid]
                row = {"key": k, "name": nm, "family": fam,
                       "mae": round(m_, 3), "rmse": round(E.rmse(y, truth), 3),
                       "smape": round(E.smape(y, truth), 2),
                       "skill": round(1 - m_ / mae_b, 3),
                       "tau": round(E.kendall_tau(y, truth), 3),
                       "crps": None, "cov90": None}
                if k == "mm":
                    row["crps"] = round(E.crps_ensemble(truth[tst_s], ens[tst_s]), 3)
                    lo = np.percentile(ens[tst_s], 5, axis=1); hi = np.percentile(ens[tst_s], 95, axis=1)
                    row["cov90"] = round(float(np.mean((truth[tst_s] >= lo) & (truth[tst_s] <= hi))) * 100, 1)
                if k == "qra":
                    row["crps"] = round(E.crps_ensemble(truth[tst_s], ens_c[tst_s]), 3)
                    lo = np.percentile(ens_c[tst_s], 5, axis=1); hi = np.percentile(ens_c[tst_s], 95, axis=1)
                    row["cov90"] = round(float(np.mean((truth[tst_s] >= lo) & (truth[tst_s] <= hi))) * 100, 1)
                rows.append(row); Lmat.append(np.abs(y - truth)); keys.append(k)
            OUT["leaderboard"][region][var] = rows

            # ---------- L5 DM + MCS ----------
            Lm = np.vstack(Lmat).T
            M = len(keys)
            stat = np.zeros((M, M)); pval = np.ones((M, M))
            for i in range(M):
                for j in range(i + 1, M):
                    s_, pv = E.dm_test(Lm[:, i], Lm[:, j])
                    stat[i, j], stat[j, i], pval[i, j], pval[j, i] = s_, -s_, pv, pv
            mcs_w = []
            for wd in sorted({90, 365, days}):
                if wd * 24 > n:
                    continue
                sub = Lm[n - wd * 24:]
                m_ = E.model_confidence_set(sub, keys, np.random.default_rng(7), alpha=0.10, B=300)
                m_["days"] = wd; m_["n"] = int(sub.shape[0])
                mcs_w.append(m_)
            OUT["dm"][region][var] = {
                "keys": keys,
                "stat": [[round(float(v), 2) for v in r] for r in stat],
                "p": [[round(float(v), 4) for v in r] for r in pval],
                "mcs_windows": mcs_w, "n": int(n)}

            # ---------- L1 时效衰减：由存储里实际有的时效桶算 ----------
            curve = []
            for ld in range(1, args.max_lead_day + 1):
                try:
                    q = R.fill_persist_head(R.load_backtest(
                        con, p.site_id, lead_day=ld, strict=args.strict,
                        final_only=not args.allow_provisional))
                except Exception:
                    continue
                yt = q.truth[var]
                qm = (q.cov.get("models_used") or [ref_m])[0]
                curve.append({"lead": int(np.median(q.lead_h[qm])),
                              "ecmwf": round(E.mae(q.preds[qm][var], yt), 3),
                              "qra": round(E.mae(q.preds["qra"][var], yt), 3)})
            OUT["lead"][region][var] = {
                "curve": curve or [{"lead": int(np.median(p.lead_h[ref_m])),
                                    "ecmwf": round(mae_b, 3),
                                    "qra": round(E.mae(p.preds["qra"][var], truth), 3)}],
                "persist": round(E.mae(p.preds["persist"][var], truth), 3),
                "analysis": 0.0, "unit": VAR_META[var]["unit"]}

            # ---------- L5 区制切片（用晴空指数 kt，不用云量）----------
            kt = p.kt()
            temp, wind = p.truth["temp"], p.truth["wind100"]
            thot, tcold = np.percentile(temp, 97), np.percentile(temp, 3)
            day = p.cosz > 0.15
            regimes = {
                "夜间 / 无辐照": p.cosz <= 0.02,
                "晴空日间 kt>0.75": day & (kt > 0.75),
                "多云日间 kt 0.35–0.75": day & (kt > 0.35) & (kt <= 0.75),
                "阴雨 kt≤0.35": day & (kt <= 0.35),
                f"高温 ≥{thot:.0f}°C": temp >= thot,
                f"低温 ≤{tcold:.0f}°C": temp <= tcold,
                "低风 <4 m/s": wind < 4.0,
                "大风 >15 m/s": wind > 15.0,
            }
            rg = []
            for nm_, msk in regimes.items():
                msk = np.asarray(msk) & np.isfinite(truth)
                if msk.sum() < 60:
                    continue
                me = E.mae(base[msk], truth[msk]); mq = E.mae(p.preds["qra"][var][msk], truth[msk])
                rg.append({"regime": nm_, "n": int(msk.sum()),
                           "share": round(float(msk.mean()) * 100, 1),
                           "mae_ecmwf": round(me, 3), "mae_qra": round(mq, 3),
                           "skill": round(1 - mq / max(me, 1e-9), 3)})
            OUT["regimes"][region][var] = rg

        # ================= L3-C 极端事件头 =================
        vre_t = p.vre(p.truth["ghi"], p.truth["wind100"], p.truth["temp"])
        norm_by_h = np.array([vre_t[p.hod == h].mean() if (p.hod == h).any() else 0.0
                              for h in range(24)])
        OUT["vre_norm"][region] = [round(float(v), 4) for v in norm_by_h]
        norm = norm_by_h[p.hod]
        ratio_t = vre_t / np.maximum(norm, 1e-6)
        label = ratio_t < 0.5

        mem = np.array([p.vre(p.preds[m]["ghi"], p.preds[m]["wind100"], p.preds[m]["temp"])
                        for m in R.NWP])                    # (4, n) 天然耦合
        prob = np.mean((mem / np.maximum(norm, 1e-6)) < 0.5, axis=0)

        OUT.setdefault("band_ref", {})[region] = round(float(np.median(
            np.percentile(mem, 95, axis=0) - np.percentile(mem, 5, axis=0))), 5)

        pr = []
        for thr in np.arange(0.05, 0.96, 0.05):
            pred = prob >= thr
            tp = int(np.sum(pred & label)); fp = int(np.sum(pred & ~label)); fn = int(np.sum(~pred & label))
            prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
            pr.append({"thr": round(float(thr), 2), "precision": round(prec, 3),
                       "recall": round(rec, 3),
                       "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 3),
                       "n_pred": int(pred.sum())})
        op = max(pr, key=lambda r: r["f1"])
        nv = np.concatenate([label[:24], label[:-24]])
        tp = int(np.sum(nv & label)); fp = int(np.sum(nv & ~label)); fn = int(np.sum(~nv & label))
        npre, nrec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        base_rate = float(label.mean())
        rel = []
        for i in range(10):
            m_ = (prob >= i / 10) & (prob < (i + 1) / 10 + (1e-9 if i == 9 else 0))
            if m_.sum() >= 20:
                rel.append({"bin": i, "pred": round(float(prob[m_].mean()), 3),
                            "obs": round(float(label[m_].mean()), 3), "n": int(m_.sum())})
        deficit = np.clip(norm - vre_t, 0, None)
        pos = deficit[deficit > 1e-4]
        u = float(np.percentile(pos, 90)) if len(pos) else 0.0
        exc = pos[pos > u] - u if len(pos) else np.array([])
        gpd = E.fit_gpd(exc) if len(exc) else None
        rl = []
        if gpd:
            rate = len(exc) / len(p)
            for pd_ in [7, 30, 90, 365]:
                rl.append({"days": pd_,
                           "level": round(E.gpd_return_level(u, gpd["xi"], gpd["sigma"],
                                                             rate, pd_ * 24) * 100, 1)})
        OUT["event"][region] = {
            "base_rate": round(base_rate, 4), "pr": pr, "op": op,
            "naive": {"precision": round(npre, 3), "recall": round(nrec, 3),
                      "f1": round(2 * npre * nrec / max(npre + nrec, 1e-9), 3),
                      "name": "Persistence (D-1)"},
            "clim_f1": round(2 * base_rate / (base_rate + 1), 3),
            "reliability": rel, "n": int(len(p)),
            "gpd": ({"xi": round(gpd["xi"], 3), "sigma": round(gpd["sigma"] * 100, 2),
                     "n_exc": gpd["n_exc"], "u": round(u * 100, 1), "rl": rl} if gpd else None)}

        # ================= L6 决策层 =================
        d0 = len(p) - dec_days * 24
        sl = slice(max(0, d0), len(p))
        hod_d = p.hod[sl]
        E.VRE_SHARE[region] = p.vre_share
        E.MIX[region] = (p.solar_w, p.wind_w)
        net_real = E.net_load(region, p.truth["ghi"][sl], p.truth["wind100"][sl],
                              p.truth["temp"][sl], hod_d)
        cal = E.price_calib(net_real)
        p_proxy = E.price_from_net(net_real, cal)

        # ---- 真实 RRP 优先；缺了才退回代理，并明确标出来 ----
        price_source = "proxy"
        p_real = p_proxy
        if args.price_source in ("aemo", "auto"):
            rrp = PL.load_rrp(con, region)
            if len(rrp):
                want = pd.DataFrame({"interval_time": pd.to_datetime(p.t[sl])})
                j = want.merge(rrp, on="interval_time", how="left")
                cover = float(j["rrp"].notna().mean())
                if cover > 0.9:
                    # 少量缺口用代理价补，避免 DP 里出现 NaN
                    p_real = np.where(j["rrp"].notna(), j["rrp"].to_numpy(float), p_proxy)
                    price_source = f"aemo({cover*100:.0f}%)"
                elif args.price_source == "aemo":
                    print(f"    ⚠ {region} RRP 覆盖率仅 {cover*100:.0f}%，退回代理价")
            elif args.price_source == "aemo":
                print(f"    ⚠ {region} 没有 RRP 数据，退回代理价（先跑 aemo.py）")
        nd = len(p_real) // 24

        def revenue(gh, wd, tp_):
            # 排程用代理价（它是天气→价格的可微映射），结算永远用 p_real
            pp = E.price_from_net(E.net_load(region, gh, wd, tp_, hod_d), cal)
            return sum(E.bess_dispatch(pp[d * 24:(d + 1) * 24], p_real[d * 24:(d + 1) * 24])
                       for d in range(nd)), pp

        rev_perf = sum(E.bess_dispatch(p_real[d * 24:(d + 1) * 24], p_real[d * 24:(d + 1) * 24])
                       for d in range(nd))
        vc = []
        for mid in [m for m in ["persist", "gem_global", "icon_global", "gfs_global",
                                "ecmwf_ifs025", "mm_mean", "qra"] if m in p.preds]:
            rev, pp = revenue(p.preds[mid]["ghi"][sl], p.preds[mid]["wind100"][sl],
                              p.preds[mid]["temp"][sl])
            vc.append({"key": KEY_MAP[mid], "name": R.MODEL_META[mid][0],
                       "capture": round(rev / max(rev_perf, 1e-9), 4),
                       "revenue": round(rev, 1),
                       "tau": round(E.kendall_tau(pp, p_real), 3),
                       "mae_price": round(E.mae(pp, p_real), 2)})
        vc.append({"key": "perfect", "name": "完美预见（上界）", "capture": 1.0,
                   "revenue": round(rev_perf, 1), "tau": 1.0, "mae_price": 0.0})

        tau_curve = []
        for lam in [0.0, 0.15, 0.3, 0.45, 0.6, 0.8, 1.0, 1.3, 1.7, 2.2, 3.0, 4.0]:
            gh = np.clip(p.truth["ghi"][sl] + lam * (p.preds["qra"]["ghi"][sl] - p.truth["ghi"][sl]), 0, None)
            wd = np.clip(p.truth["wind100"][sl] + lam * (p.preds["qra"]["wind100"][sl] - p.truth["wind100"][sl]), 0.05, None)
            tp_ = p.truth["temp"][sl] + lam * (p.preds["qra"]["temp"][sl] - p.truth["temp"][sl])
            rev, pp = revenue(gh, wd, tp_)
            tau_curve.append({"lam": lam, "tau": round(E.kendall_tau(pp, p_real), 3),
                              "capture": round(rev / max(rev_perf, 1e-9), 4)})
        tau_curve.sort(key=lambda r: r["tau"])

        OUT["decision"][region] = {
            "days": nd, "value_capture": vc, "tau_curve": tau_curve,
            "price_stats": {"mean": round(float(p_real.mean()), 1),
                            "median": round(float(np.median(p_real)), 1),
                            "p95": round(float(np.percentile(p_real, 95)), 1),
                            "p99": round(float(np.percentile(p_real, 99)), 1),
                            "max": round(float(p_real.max()), 1),
                            "neg_share": round(float(np.mean(p_real < 0)) * 100, 2)},
            "bess": {"power_mw": E.P_MW, "energy_mwh": E.E_MWH, "eta": E.ETA,
                     "soc_levels": E.SOC_N},
            "price_source": price_source}

        # ---- 分水岭：缺口 ↔ RRP ----
        try:
            pl = PL.analyse_panel(con, p, args.spike)
        except Exception as e:
            pl = {"region": region, "status": "error", "hint": f"{type(e).__name__}: {e}"}
        for k in ("deciles",):
            pl.pop(k, None)
        OUT.setdefault("price_link", {})[region] = pl
        print(f"[{time.time()-t0:5.1f}s] {region} 全部面板完成")

    # ================= 运行窗口 =================
    win_ok = True
    win_h = 999
    for s in SITES:
        w = R.load_window(con, s[0], past_h=24, future_h=48)
        win_ok &= w["ok"]; win_h = min(win_h, w["horizon_h"])
        region = w["region"]
        p = panels[region]
        rd = {"code": region, "name": REGION_CN.get(region, region), "site": w["name"],
              "lat": w["lat"], "lon": w["lon"], "note": "",
              "hours": list(range(-24, 49)), "series": {}}
        for var in VARS:
            ss = w["series"][var]
            med = np.nan_to_num(ss["median"], nan=0.0)
            lo = np.nan_to_num(ss["lo"], nan=0.0); hi = np.nan_to_num(ss["hi"], nan=0.0)
            lam = OUT["calibration"][region][var]["lambda90"]
            hw = (hi - lo) / 2.0
            q5, q95 = med - lam * hw, med + lam * hw
            q25, q75 = med - 0.45 * lam * hw, med + 0.45 * lam * hw
            for a in (q5, q25, q75, q95):
                a[:] = _clip(var, a)
            rd["series"][var] = {
                "obs": [round(float(v), 2) for v in np.nan_to_num(ss["obs"], nan=0.0)],
                "qra": [round(float(v), 2) for v in med],
                "ecmwf": [round(float(v), 2) for v in np.nan_to_num(ss["ecmwf"], nan=0.0)],
                "gfs": [round(float(v), 2) for v in np.nan_to_num(ss["gfs"], nan=0.0)],
                "q5": [round(float(v), 2) for v in q5], "q25": [round(float(v), 2) for v in q25],
                "q75": [round(float(v), 2) for v in q75], "q95": [round(float(v), 2) for v in q95]}
        OUT["regions"].append(rd)
    as_of = pd.Timestamp(w["as_of"]) + pd.Timedelta(hours=10)
    if not win_ok:
        print(f"  ⚠ 运行窗口只有 {win_h}h 未来预报（不足 48h）。"
              f"回测面板不受影响；要补满窗口请跑：python ingest.py --live")

    # ================= meta / layers =================
    OUT["meta"] = {
        "as_of_nem": as_of.strftime("%Y-%m-%dT%H:%M:%S+10:00"),
        "tz": "NEM 时间 = UTC+10（全年固定，无夏令时）",
        "past_h": 24, "future_h": 48, "vars": VAR_META,
        "models": [{"key": KEY_MAP[m], "short": SHORT[KEY_MAP[m]],
                    "name": R.MODEL_META[m][0], "family": R.MODEL_META[m][1],
                    "prob": R.MODEL_META[m][2]} for m in R.MODEL_META],
        "backtest_days": days, "decision_days": dec_days,
        "data_mode": "fixture" if is_fixture else "real",
        "truth_sources": sorted(map(str, srcs)),
        "spike_threshold": args.spike,
        "window_ok": bool(win_ok), "window_horizon_h": int(win_h),
        "lead_day": args.lead_day,
        "coverage": {r: panels[r].cov for r in panels}}
    OUT["layers"] = [
        {"id": "L0", "name": "数据源", "status": "ok", "metric": f"{len(R.NWP)} 模式",
         "detail": "Open-Meteo：ECMWF-IFS / GFS / ICON / GEM + ERA5 再分析"},
        {"id": "L1", "name": "采集与时点存储", "status": "ok",
         "metric": f"{days} 天 · {sum(len(x) for x in panels.values()):,} 行",
         "detail": "Parquet 只追加；每条带 interval_time / known_at / ingested_at；NEM = UTC+10"},
        {"id": "L2", "name": "特征层", "status": "ok", "metric": f"D+{args.lead_day} 固定时效",
         "detail": "唯一入口 get_features(as_of, horizons)，只取 known_at ≤ t；各模型同一样本"},
        {"id": "L3", "name": "模型层", "status": "warn", "metric": "NWP + 集成",
         "detail": "点预测与分布头已就位；残差学习（LightGBM）与极端事件分类器待训练"},
        {"id": "L4", "name": "集成与校准", "status": "ok", "metric": "分裂共形",
         "detail": "多模式中位数 + 分裂共形 + 分位数单调化 + 物理边界截断"},
        {"id": "L5", "name": "评估回测", "status": "ok", "metric": f"{days}d · DM+MCS",
         "detail": "固定时效滚动评估；研究口径 + 交易口径并行；DM 检验与 MCS；按 kt 区制切片"},
        {"id": "L6", "name": "决策层", "status": "ok", "metric": f"{len(panels)} 区",
         "detail": "极端事件告警 · BESS DP 调度 · 价值捕获率；电价仍为净负荷代理"},
    ]
    print(f"[{time.time()-t0:5.1f}s] 完成")
    return OUT


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="用 L1 真实数据重建运行台数据集")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), "data_real.json"))
    ap.add_argument("--lead-day", type=int, default=1, help="回测取的预报时效（天）")
    ap.add_argument("--max-lead-day", type=int, default=5, help="时效衰减曲线最多画到几天")
    ap.add_argument("--decision-days", type=int, default=240)
    ap.add_argument("--strict", action="store_true", help="只用 exact/request 的 known_at")
    ap.add_argument("--allow-provisional", action="store_true", help="允许用临时真值")
    ap.add_argument("--price-source", choices=["auto", "aemo", "proxy"], default="auto",
                    help="L6 结算价来源：auto=有 RRP 就用，aemo=强制，proxy=只用代理")
    ap.add_argument("--spike", type=float, default=300.0, help="尖峰阈值 $/MWh")
    ap.add_argument("--coverage-only", action="store_true")
    a = ap.parse_args(argv)

    con = store.connect()
    try:
        if a.coverage_only:
            print(R.coverage_report(con, a.lead_day).to_string(index=False))
            return 0
        out = build(con, a)
    finally:
        con.close()

    with open(a.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    print(f"→ {a.out}  {os.path.getsize(a.out)/1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
造一份**格式完全正确、内容是合成的** L1 存储，用来端到端验证迁移代码。
=====================================================================
为什么需要它：写迁移脚本的环境访问不了 Open-Meteo，而"真数据装配"这段代码
最容易出错的地方（分位、时效分桶、样本对齐、缺口处理）跟数据真假无关。
这个 fixture 把 engine.py 的合成序列按 previous-runs 回填**一模一样的行格式**
写进 Parquet，于是 realdata.py / build_data_real.py 的每一行都能在离线跑通。

    python make_fixture.py --days 400 --out ./data_fixture
    WQ_DATA_DIR=./data_fixture python build_data_real.py

真数据到位后，把 WQ_DATA_DIR 指回真的 data/ 即可，代码一行不用改。
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import engine as E  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=400)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), "data_fixture"))
    ap.add_argument("--lead-days", type=int, default=3)
    ap.add_argument("--no-price", action="store_true", help="不生成合成 RRP")
    a = ap.parse_args(argv)

    os.environ["WQ_DATA_DIR"] = os.path.abspath(a.out)
    import store  # noqa: E402  必须在设置 WQ_DATA_DIR 之后导入
    from config import SITES, VARIABLES  # noqa: E402

    # 合成模型的误差尺度，跟真实 NWP 的相对强弱大致一致
    MODELS = {"ecmwf_ifs025": (1.00, 0.03), "gfs_global": (1.22, 0.05),
              "icon_global": (1.15, 0.02), "gem_global": (1.30, 0.04)}
    DELAY_H = 6

    n = a.days * 24
    end = pd.Timestamp.now("UTC").tz_localize(None).floor("h") - pd.Timedelta(days=6)
    t_utc = pd.date_range(end - pd.Timedelta(hours=n - 1), end, freq="h")
    run_id = store.new_run_id()
    print(f"fixture → {os.environ['WQ_DATA_DIR']}\n区间 {t_utc[0]} .. {t_utc[-1]}  ({a.days} 天)")

    tot_t = tot_f = 0
    for s in SITES:
        site_id, region, name, lat, lon = s[0], s[1], s[2], s[3], s[4]
        # engine.REGIONS 的元组结构：(code, cn, site, lat, lon, t_aug, trange, tamp, wmean, note)
        eref = next((r for r in E.REGIONS if r[0] == region), E.REGIONS[0])
        reg = (region, name, name, lat, lon, eref[5], eref[6], eref[7], eref[8], "")
        rng = np.random.default_rng(abs(hash(site_id)) % (2 ** 31))
        nem0 = (t_utc[0] + pd.Timedelta(hours=10))
        tr = E.truth_series(reg, int(nem0.dayofyear), int(nem0.hour), n, rng)

        # ---- 真值：ERA5，5 天延迟 ----
        tr_rows = []
        for v in VARIABLES:
            tr_rows.append(pd.DataFrame({
                "site_id": site_id, "variable": v, "interval_time": t_utc,
                "known_at": t_utc + pd.Timedelta(days=5), "value": tr[v],
                "quality": "final", "source": "era5", "ingested_at": store.utcnow()}))
        tdf = pd.concat(tr_rows, ignore_index=True)
        tot_t += store.append("truth", tdf, run_id)

        # ---- 预报：按 previous-runs 的行格式，逐时效一条 ----
        f_rows = []
        for model, (factor, bias) in MODELS.items():
            for lead in range(1, a.lead_days + 1):
                lead_arr = np.full(n, lead * 24.0)
                for v in VARIABLES:
                    pot = tr["potential"] if v == "ghi" else None
                    y = E.make_forecast(tr[v], v, lead_arr, factor, bias, rng, potential=pot)
                    f_rows.append(pd.DataFrame({
                        "site_id": site_id, "model": model, "variable": v,
                        "interval_time": t_utc,
                        "known_at": t_utc - pd.Timedelta(hours=lead * 24 - DELAY_H),
                        "value": y, "known_at_method": "approx",
                        "source": "previous_runs", "ingested_at": store.utcnow()}))
        fdf = pd.concat(f_rows, ignore_index=True)
        tot_f += store.append("forecast", fdf, run_id)

        # ---- 合成市场数据：site_id 用区代码，与气象站点区分 ----
        np_ = 0
        if not a.no_price:
            hod = ((t_utc.hour + 10) % 24).to_numpy()
            E.MIX[region] = (s[5], s[6]); E.VRE_SHARE[region] = s[7]
            net = E.net_load(region, tr["ghi"], tr["wind100"], tr["temp"], hod)
            cal = E.price_calib(net)
            rng2 = np.random.default_rng(abs(hash(site_id + "px")) % (2 ** 31))
            rrp = np.clip(E.price_from_net(net, cal) * np.exp(rng2.normal(0, .28, n)),
                          E.FLOOR, E.MPC)
            # AEMO 预调度：对净负荷极端段**反应不足**（缺口影响打 6 折）——
            # 这样残差里还留着缺口信息，price_link 的第 ③ 层才有东西可测
            net_d = cal["q50"] + 0.6 * (net - cal["q50"])
            aemo = np.clip(E.price_from_net(net_d, cal) * np.exp(rng2.normal(0, .18, n)),
                           E.FLOOR, E.MPC)
            prows = []
            for var, val in (("rrp", rrp), ("rrp_max", rrp * 1.35), ("demand", 7000 + 2500 * net)):
                prows.append(pd.DataFrame({
                    "site_id": region, "variable": var, "interval_time": t_utc,
                    "known_at": t_utc + pd.Timedelta(minutes=70), "value": val,
                    "quality": "final", "source": "fixture_market",
                    "ingested_at": store.utcnow()}))
            np_ += store.append("truth", pd.concat(prows, ignore_index=True), run_id)
            # 两版预调度：
            #   早版——决策时刻（前一天 13:30 NEM）前半小时发出，这是 ladder 该用的那一版；
            #   晚版——目标前 2 小时发出，数值**就是实际电价**。这是一个前视陷阱：
            #   哪天代码不小心用到了决策时刻之后的信息，B2 会突然变得完美，一眼就能看出来。
            import realdata as RD
            tau = pd.to_datetime(RD.decision_time(pd.Series(t_utc)))
            early = pd.DataFrame({
                "site_id": region, "model": "aemo_predispatch", "variable": "rrp",
                "interval_time": t_utc,
                "known_at": pd.Series(tau).to_numpy() - pd.Timedelta(minutes=30),
                "value": aemo, "known_at_method": "exact", "source": "fixture_market",
                "ingested_at": store.utcnow()})
            late = early.copy()
            late["known_at"] = t_utc - pd.Timedelta(hours=2)
            late["value"] = rrp
            adf = pd.concat([early, late], ignore_index=True)
            tot_f += store.append("forecast", adf, run_id)
            tot_t += np_
        print(f"  {site_id:10s} 真值 {len(tdf):>7d} · 预报 {len(fdf):>7d} · 市场 {np_:>6d}")

    print(f"\n写入：truth {tot_t} 行 · forecast {tot_f} 行")
    print("下一步：  WQ_DATA_DIR=%s python build_data_real.py" % os.environ["WQ_DATA_DIR"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

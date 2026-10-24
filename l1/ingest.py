"""
L1 采集 —— 回填 + 增量，写同一张只追加的表。
============================================
用法
  python ingest.py --selftest                                   离线自检（不联网）
  python ingest.py --status                                     看表里有什么
  python ingest.py --backfill-truth  --start 2024-03-14 --end 2026-08-20
  python ingest.py --backfill-forecast --method previous-runs --start 2024-03-14 --end 2026-08-20
  python ingest.py --backfill-forecast --method single-runs   --start 2026-06-01 --end 2026-08-20 --runs 00,12
  python ingest.py --live

回填和增量是同一张表的两端，可以并行跑：先开 --live 写今天的，同时慢慢回填历史，
两边在中间自然接上（重复键会被反连接剔掉）。

known_at 的三种来源，语义写死在这里，别的地方不许再解释一遍：
  exact    Single Runs 的 run= 就是初始化时刻，加模型发布延迟
  request  实时抓取用请求时刻；它 ≥ 真实可见时刻，只会更严格，不会泄漏
  approx   Previous Runs 推出来的保守上界：interval_time − N×24h + 发布延迟
"""
from __future__ import annotations
import argparse
import os
import sys
import traceback
from datetime import datetime, timedelta, timezone

import pandas as pd

import openmeteo as om
import store
from config import (LIVE_MODELS, MODELS, SITES, TRUTH_SOURCES,
                    VARIABLES, CHUNK_DAYS)

# 模型发布延迟：初始化时刻之后多久，这一版预报才真的能用
PUBLISH_DELAY_H = {"ecmwf_ifs025": 6, "gfs_global": 4, "icon_global": 4, "gem_global": 5}
DEFAULT_DELAY_H = 6


def _delay(model: str) -> timedelta:
    return timedelta(hours=PUBLISH_DELAY_H.get(model, DEFAULT_DELAY_H))


def _chunks(start: str, end: str, days: int):
    a, b = pd.Timestamp(start), pd.Timestamp(end)
    while a <= b:
        c = min(a + pd.Timedelta(days=days - 1), b)
        yield a.date().isoformat(), c.date().isoformat()
        a = c + pd.Timedelta(days=1)


# ================================================================== 真值
def backfill_truth(start: str, end: str, source: str = "era5") -> tuple[int, int]:
    cfg = TRUTH_SOURCES[source]
    lag = pd.Timedelta(days=cfg["lag_days"])
    fetch = om.fetch_archive if source == "era5" else om.fetch_ifs_analysis
    seen = written = 0
    run_id = store.new_run_id()
    for a, b in _chunks(start, end, CHUNK_DAYS):
        df = fetch(SITES, a, b)
        if df.empty:
            continue
        df["known_at"] = df["interval_time"] + lag        # 真值也有可见时刻
        df["quality"] = cfg["quality"]
        df["source"] = source
        df["ingested_at"] = store.utcnow()
        seen += len(df)
        written += store.append("truth", df, run_id)
        print(f"  truth[{source}] {a}..{b}  +{len(df):>7d} 行")
    return seen, written


# ================================================================== 预报：Single Runs（精确）
def backfill_single_runs(start: str, end: str, model: str, run_hours: list[int],
                         assume_yes: bool = False) -> tuple[int, int]:
    cfg = MODELS[model]
    lo = pd.Timestamp(cfg["archive_from"])
    a0 = max(pd.Timestamp(start), lo)
    if a0 > pd.Timestamp(end):
        print(f"  {model} 存档从 {cfg['archive_from']} 起，跳过")
        return 0, 0
    days = (pd.Timestamp(end) - a0).days + 1
    n_calls = days * len(run_hours)
    print(f"  {model}: {days} 天 × {len(run_hours)} 个 run = {n_calls} 次调用"
          f"（每次覆盖全部 {len(SITES)} 个站点）")
    if n_calls > 400 and not assume_yes:
        print("  调用数较多，加 --yes 确认，或缩短区间 / 减少 --runs")
        return 0, 0

    seen = written = 0
    run_id = store.new_run_id()
    d = a0
    while d <= pd.Timestamp(end):
        for h in run_hours:
            run_iso = f"{d.date().isoformat()}T{h:02d}:00"
            try:
                df = om.fetch_single_run(SITES, run_iso, model)
            except om.OpenMeteoError as e:
                print(f"    跳过 {run_iso}: {str(e)[:120]}")
                continue
            if df.empty:
                continue
            df["known_at"] = pd.Timestamp(run_iso) + _delay(model)
            df["known_at_method"] = "exact"
            df["source"] = "single_runs"
            df["ingested_at"] = store.utcnow()
            seen += len(df)
            written += store.append("forecast", df, run_id)
        if d.day == 1 or d == a0:
            print(f"  ... {d.date()}  累计写入 {written} 行")
        d += pd.Timedelta(days=1)
    return seen, written


# ================================================================== 预报：Previous Runs（保守近似）
def backfill_previous_runs(start: str, end: str, model: str,
                           max_day: int = 5) -> tuple[int, int]:
    seen = written = 0
    run_id = store.new_run_id()
    lo = pd.Timestamp(MODELS[model]["archive_from"])
    a0 = max(pd.Timestamp(start), lo).date().isoformat()
    for a, b in _chunks(a0, end, CHUNK_DAYS):
        # 一个窗口失败就丢掉四个月太浪费——多半只是响应太大而超时，
        # 所以对半切了再试，最小切到 7 天才认输。
        df = None
        queue = [(a, b)]
        while queue:
            qa, qb = queue.pop(0)
            try:
                d = om.fetch_previous_runs(SITES, qa, qb, model, max_day=max_day)
            except om.OpenMeteoError as e:
                span = (pd.Timestamp(qb) - pd.Timestamp(qa)).days
                if span > 7:
                    mid = (pd.Timestamp(qa) + pd.Timedelta(days=span // 2)).date().isoformat()
                    nxt = (pd.Timestamp(mid) + pd.Timedelta(days=1)).date().isoformat()
                    print(f"    {qa}..{qb} 失败，切成两半重试（{span}→{span//2} 天）")
                    queue = [(qa, mid), (nxt, qb)] + queue
                else:
                    print(f"    放弃 {qa}..{qb}:\n      {e}")
                continue
            df = d if df is None else pd.concat([df, d], ignore_index=True)
        if df is None or df.empty:
            continue
        # 保守上界：发报时刻落在 [T−(N+1)·24h, T−N·24h]，取最晚端 + 发布延迟。
        # N=0 时 known_at > interval_time，也就是"目标时刻之前不可见"——正确，
        # day0 对历史时刻本质是分析场，不该当特征用。
        df["known_at"] = (df["interval_time"]
                          - pd.to_timedelta(df["lead_days"] * 24, unit="h")
                          + _delay(model))
        df["known_at_method"] = "approx"
        df["source"] = "previous_runs"
        df["ingested_at"] = store.utcnow()
        df = df.drop(columns=["lead_days"])
        seen += len(df)
        written += store.append("forecast", df, run_id)
        print(f"  prev[{model}] {a}..{b}  +{len(df):>7d} 行")
    return seen, written


# ================================================================== 增量
def live(models=None, past_days: int = 2, forecast_days: int = 7) -> tuple[int, int]:
    models = models or LIVE_MODELS
    run_id = store.new_run_id()
    now = store.utcnow()
    df = om.fetch_forecast(SITES, models, past_days=past_days, forecast_days=forecast_days)
    if df.empty:
        return 0, 0
    df["known_at"] = now                    # 保守：请求时刻 ≥ 真实可见时刻
    df["known_at_method"] = "request"
    df["source"] = "forecast"
    df["ingested_at"] = now
    w = store.append("forecast", df, run_id)
    print(f"  live {len(models)} 模型 × {len(SITES)} 站点：见 {len(df)} 行，新写 {w} 行")

    # 顺带把最近的临时真值也拿了（ERA5 要等 5 天，先用 IFS 分析场顶着）
    t0 = (now - pd.Timedelta(days=past_days)).date().isoformat()
    t1 = now.date().isoformat()
    try:
        tdf = om.fetch_ifs_analysis(SITES, t0, t1)
        if not tdf.empty:
            tdf["known_at"] = tdf["interval_time"] + pd.Timedelta(hours=6)
            tdf["quality"] = "provisional"
            tdf["source"] = "ifs_analysis"
            tdf["ingested_at"] = now
            tw = store.append("truth", tdf, run_id)
            print(f"  live 临时真值：见 {len(tdf)} 行，新写 {tw} 行")
            w += tw
    except om.OpenMeteoError as e:
        print(f"  临时真值抓取失败（不致命）: {str(e)[:120]}")
    return len(df), w


# ================================================================== 离线自检
def selftest() -> int:
    """不联网。造一批带已知 known_at 的行，验证：
       ① 只追加语义 ② 重复写幂等 ③ as-of 读取不泄漏 ④ NEM 时间视图正确"""
    # 自检必须写进**临时目录**。以前它往真库里塞 63 行 source='selftest' 的
    # 假数据，跑一次污染一次——NSW1_CWO 的行数比别的站多出来就是这么来的。
    # 回测样本里混进人造行，是那种永远查不出来的错。
    import tempfile
    tmp = tempfile.mkdtemp(prefix="wq-selftest-")
    os.environ["WQ_DATA_DIR"] = tmp
    import importlib
    import config as _cfg
    _cfg.DATA_DIR = tmp
    importlib.reload(store)
    import features
    importlib.reload(features)

    print(f"离线自检 —— 不联网（临时库 {tmp}）\n" + "=" * 46)
    run_id = store.new_run_id()
    base = pd.Timestamp("2026-06-01 00:00")
    rows = []
    # 三个连续的模型运行，各覆盖同一批目标时刻
    for k, run in enumerate([base, base + pd.Timedelta(hours=12), base + pd.Timedelta(hours=24)]):
        known = run + pd.Timedelta(hours=6)
        for h in range(0, 72, 3):
            tgt = run + pd.Timedelta(hours=h)
            for var in VARIABLES:
                rows.append({"site_id": "NSW1_CWO", "model": "ecmwf_ifs025", "variable": var,
                             "interval_time": tgt, "known_at": known,
                             "value": float(100 + k * 10 + h), "known_at_method": "exact",
                             "source": "selftest", "ingested_at": store.utcnow()})
    df = pd.DataFrame(rows)
    n1 = store.append("forecast", df, run_id)
    n2 = store.append("forecast", df, store.new_run_id())      # 重复写
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' — ' + extra) if extra else ''}")

    check("首次写入落盘", n1 == len(df), f"{n1}/{len(df)} 行")
    check("重复写入幂等（只追加不覆盖）", n2 == 0, f"第二次写了 {n2} 行")

    truth = df[df.model == "ecmwf_ifs025"].drop_duplicates(
        subset=["variable", "interval_time"])[["site_id", "variable", "interval_time", "value"]].copy()
    truth["known_at"] = truth["interval_time"] + pd.Timedelta(days=5)
    truth["quality"] = "final"; truth["source"] = "selftest"
    truth["ingested_at"] = store.utcnow()
    store.append("truth", truth, run_id)

    con = store.connect()
    try:
        # ③ as-of：站在第二个 run 已发布、第三个还没发布的时刻
        as_of = base + pd.Timedelta(hours=20)          # run2(12:00) 已于 18:00 可见；run3 要 06-02 06:00
        feats = features.get_features(con, as_of, horizons=range(1, 25), sites=["NSW1_CWO"])
        check("as-of 读取返回了行", len(feats) > 0, f"{len(feats)} 行")

        raw = con.execute("""
            SELECT max(known_at) mk FROM forecast WHERE known_at <= ?
        """, [as_of]).fetchone()[0]
        check("可见集合的 known_at 全部 ≤ as_of", raw is not None and pd.Timestamp(raw) <= as_of,
              f"max(known_at)={raw}")

        leaked = con.execute("""
            SELECT count(*) FROM forecast WHERE known_at > ? AND known_at <= ?
        """, [as_of, as_of + pd.Timedelta(days=2)]).fetchone()[0]
        check("确实存在未来才可见的行（说明过滤不是空操作）", leaked > 0, f"{leaked} 行被正确挡住")

        picked = con.execute("""
            SELECT DISTINCT known_at FROM forecast
            WHERE known_at <= ? AND source='selftest' ORDER BY known_at DESC LIMIT 1
        """, [as_of]).fetchone()[0]
        check("取到的是当时最新的一版", pd.Timestamp(picked) == base + pd.Timedelta(hours=18),
              str(picked))

        # ④ NEM 时间视图
        d = con.execute("""
            SELECT date_diff('hour', min(f.interval_time), min(n.interval_time))
            FROM forecast f, forecast_nem n
        """).fetchone()[0]
        check("NEM 时间视图 = UTC + 10h", d == 10, f"差 {d} 小时")

        # 泄漏断言本身要能拦住坏输入
        try:
            features.assert_no_leak(pd.DataFrame({"known_at": [as_of + pd.Timedelta(hours=1)]}), as_of)
            check("泄漏断言会抛异常", False)
        except AssertionError:
            check("泄漏断言会抛异常", True)
    finally:
        con.close()

    print("=" * 46)
    print("自检通过 ✓" if ok else "自检失败 ✗")
    return 0 if ok else 1


# ================================================================== CLI
# ================================================================== 端点探测
def probe() -> int:
    """每个端点打一发**最小**请求（1 站 1 变量 1 天）。

    大窗口失败时你分不清是"网络不通"还是"响应太大超时"——最小请求能通，
    就说明是后者，切小窗口即可；最小请求也不通，才是网络或端点的问题。
    """
    from config import (EP_ARCHIVE, EP_FORECAST, EP_HIST_FORECAST,
                        EP_PREVIOUS_RUNS, EP_SINGLE_RUNS)
    s = SITES[0]
    day = (datetime.now(timezone.utc) - timedelta(days=10)).date().isoformat()
    cases = [
        ("forecast", EP_FORECAST,
         {"latitude": s[3], "longitude": s[4], "hourly": "temperature_2m",
          "forecast_days": 1, "timezone": "UTC"}),
        ("archive(ERA5)", EP_ARCHIVE,
         {"latitude": s[3], "longitude": s[4], "hourly": "temperature_2m",
          "start_date": day, "end_date": day, "timezone": "UTC"}),
        ("historical_forecast", EP_HIST_FORECAST,
         {"latitude": s[3], "longitude": s[4], "hourly": "temperature_2m",
          "start_date": day, "end_date": day, "timezone": "UTC"}),
        ("single_runs", EP_SINGLE_RUNS,
         {"latitude": s[3], "longitude": s[4], "hourly": "temperature_2m",
          "models": "ecmwf_ifs025", "start_date": day, "end_date": day, "timezone": "UTC"}),
        ("previous_runs", EP_PREVIOUS_RUNS,
         {"latitude": s[3], "longitude": s[4],
          "hourly": "temperature_2m_previous_day1", "models": "ecmwf_ifs025",
          "start_date": day, "end_date": day, "timezone": "UTC"}),
        ("previous_runs[gfs]", EP_PREVIOUS_RUNS,
         {"latitude": s[3], "longitude": s[4],
          "hourly": "temperature_2m_previous_day1", "models": "gfs_global",
          "start_date": day, "end_date": day, "timezone": "UTC"}),
    ]
    bad = 0
    # 探测要快：网络不通时别让用户等六轮退避
    om.MAX_RETRIES, om.HTTP_TIMEOUT = 2, 20
    print("Open-Meteo 端点探测 —— 每个端点 1 站 1 变量 1 天\n" + "=" * 60)
    for name, url, params in cases:
        try:
            j = om._get(url, params)
            h = (j[0].get("hourly") or {}) if j else {}
            n = len(h.get("time", []))
            vals = [k for k in h if k != "time"]
            print(f"  [OK]   {name:22s} {n:>3d} 个时刻，列 {vals}")
        except Exception as e:
            bad += 1
            print(f"  [FAIL] {name:22s} {e}")
    print("=" * 60)
    print("全部通过 ✓" if not bad else f"{bad} 个端点不可用 —— 上面的错误类型就是病因")
    return 0 if not bad else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="WeatherQuant NEM — L1 采集与时点存储")
    ap.add_argument("--probe", action="store_true", help="每个端点打一发最小请求，打印完整错误")
    ap.add_argument("--selftest", action="store_true", help="离线自检，不联网")
    ap.add_argument("--status", action="store_true", help="打印表状态")
    ap.add_argument("--live", action="store_true", help="增量：抓当前预报 + 临时真值")
    ap.add_argument("--backfill-truth", action="store_true")
    ap.add_argument("--backfill-forecast", action="store_true")
    ap.add_argument("--method", choices=["single-runs", "previous-runs"], default="previous-runs")
    ap.add_argument("--model", default="ecmwf_ifs025", choices=list(MODELS))
    ap.add_argument("--source", default="era5", choices=list(TRUTH_SOURCES))
    ap.add_argument("--start", default="2024-03-14")
    ap.add_argument("--end", default=(datetime.now(timezone.utc) - timedelta(days=6)).date().isoformat())
    ap.add_argument("--runs", default="00,12", help="single-runs 模式下要抓的 UTC 初始化时刻")
    ap.add_argument("--max-day", type=int, default=5, help="previous-runs 的最大时效天数")
    ap.add_argument("--yes", action="store_true", help="确认大批量调用")
    a = ap.parse_args(argv)

    if a.probe:
        return probe()
    if a.selftest:
        return selftest()
    if a.status:
        import json
        print(json.dumps(store.status(), ensure_ascii=False, indent=2, default=str))
        return 0

    started, run_id = store.utcnow(), store.new_run_id()
    mode = "live" if a.live else ("backfill-truth" if a.backfill_truth else "backfill-forecast")
    target = a.source if a.backfill_truth else (a.model if not a.live else ",".join(LIVE_MODELS))
    try:
        if a.live:
            seen, wrote = live()
        elif a.backfill_truth:
            seen, wrote = backfill_truth(a.start, a.end, a.source)
        elif a.backfill_forecast:
            if a.method == "single-runs":
                hours = [int(x) for x in a.runs.split(",") if x.strip() != ""]
                seen, wrote = backfill_single_runs(a.start, a.end, a.model, hours, a.yes)
            else:
                seen, wrote = backfill_previous_runs(a.start, a.end, a.model, a.max_day)
        else:
            ap.print_help()
            return 2
        store.log_run(run_id, started, mode, target, om.CALLS[0], seen, wrote, "ok")
        print(f"\n{mode}[{target}] 完成：HTTP {om.CALLS[0]} 次，见 {seen} 行，新写 {wrote} 行")
        return 0
    except Exception as e:
        store.log_run(run_id, started, mode, target, om.CALLS[0], 0, 0, "error",
                      f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

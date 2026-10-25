"""
精简数据仓 —— 给云上的每日任务（GitHub Actions）用。
=================================================
完整数据仓在你的 Mac 上（~/aus-weather-quant/data），每日任务跑在 GitHub 的机器上，拿不到它。
这里把每日任务**真正用得到的**那部分导出成一个小仓库，放进代码仓库的 data_live/：

  · truth：五个区的 rrp / rrp_max（全部历史，期货校准要用），五个站点的气象真值（全部历史，天气常态要用）
  · forecast：ecmwf_ifs025 预报、AEMO 预调度 —— 只保留最近 KEEP_DAYS 天的目标小时（训练窗 180 + 禁运 3 + 余量）

表结构和完整仓库一模一样（store.COLS），所以 ladder / paper / futures 不用改，只要把 WQ_DATA_DIR 指过去。
每日任务往这里追加新数据；--compact 把小文件合并成每表一个文件。

    python l1/livestore.py --export data_live           # 在有完整数据仓的机器上跑一次
    python l1/livestore.py --compact data_live          # 偶尔合并一次
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil

import duckdb
import pandas as pd

KEEP_DAYS = 330          # 训练窗 180 + 禁运 3 + 天气滚动分位 60 + 余量；少了会让天气特征和完整仓不一致
FC_MODELS = ("ecmwf_ifs025", "aemo_predispatch")


def _read(src: str, table: str, where: str = "") -> pd.DataFrame:
    pat = os.path.join(src, table, "**", "*.parquet")
    if not glob.glob(pat, recursive=True):
        return pd.DataFrame()
    con = duckdb.connect(":memory:")
    try:
        return con.execute(f"SELECT * FROM read_parquet('{pat}', union_by_name=true) {where}").df()
    finally:
        con.close()


def _write(df: pd.DataFrame, dest: str, table: str, name: str = "compact.parquet") -> int:
    import store
    d = os.path.join(dest, table)
    os.makedirs(d, exist_ok=True)
    df = df[store.COLS[table]].drop_duplicates(subset=store.KEYS[table], keep="last")
    for c in ("interval_time", "known_at", "ingested_at", "started_at", "finished_at"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c]).astype("datetime64[us]")
    df.to_parquet(os.path.join(d, name), index=False, compression="zstd")
    return len(df)


def export(src: str, dest: str, keep_days: int = KEEP_DAYS) -> dict:
    from config import SITES
    regions = sorted({s[1] for s in SITES})
    sites = [s[0] for s in SITES]
    since = (pd.Timestamp.now("UTC").tz_localize(None) - pd.Timedelta(days=keep_days)).strftime("%Y-%m-%d")
    q = lambda xs: ", ".join(f"'{x}'" for x in xs)          # noqa: E731
    tr = _read(src, "truth",
               f"WHERE (site_id IN ({q(regions)}) AND variable IN ('rrp','rrp_max')) "
               f"OR site_id IN ({q(sites)})")
    fc = _read(src, "forecast",
               f"WHERE model IN ({q(FC_MODELS)}) AND interval_time >= TIMESTAMP '{since}'")
    if os.path.exists(dest):
        for t in ("truth", "forecast", "ingest_log"):
            shutil.rmtree(os.path.join(dest, t), ignore_errors=True)
    n = {"truth": _write(tr, dest, "truth"), "forecast": _write(fc, dest, "forecast")}
    os.makedirs(os.path.join(dest, "ingest_log"), exist_ok=True)
    with open(os.path.join(dest, "README.md"), "w") as f:
        f.write("精简数据仓（每日任务用）。由 l1/livestore.py 导出，表结构同完整数据仓。\n"
                "数据来源：AEMO NEMWeb（电价、预调度）、Open-Meteo（ECMWF 预报、ERA5，CC BY 4.0）。\n")
    return n


def sync(src: str, dest: str, keep_days: int = KEEP_DAYS) -> dict:
    """把完整数据仓里**新补的**东西并进 data_live，不丢云上任务已经写进去的数据。
    用途：在 Mac 上回填了预调度里的供需基本面之后，把它们带给云上的每日任务。"""
    from config import SITES
    regions = sorted({x[1] for x in SITES})
    sites = [x[0] for x in SITES]
    since = (pd.Timestamp.now("UTC").tz_localize(None) - pd.Timedelta(days=keep_days)).strftime("%Y-%m-%d")
    q = lambda xs: ", ".join(f"'{x}'" for x in xs)          # noqa: E731
    out = {}
    for t, where in (("truth", f"WHERE (site_id IN ({q(regions)}) AND variable IN ('rrp','rrp_max')) "
                               f"OR site_id IN ({q(sites)})"),
                     ("forecast", f"WHERE model IN ({q(FC_MODELS)}) AND interval_time >= TIMESTAMP '{since}'")):
        a, b = _read(src, t, where), _read(dest, t)
        both = pd.concat([x for x in (b, a) if len(x)], ignore_index=True)
        if both.empty:
            continue
        old = glob.glob(os.path.join(dest, t, "**", "*.parquet"), recursive=True)
        out[t] = _write(both, dest, t, name="compact.new.parquet")
        for f in old:
            if not f.endswith("compact.new.parquet"):
                os.remove(f)
        os.replace(os.path.join(dest, t, "compact.new.parquet"), os.path.join(dest, t, "compact.parquet"))
    return out


def compact(dest: str, keep_days: int = KEEP_DAYS) -> dict:
    """把每日追加的小文件合并成每表一个文件；预报只留最近 keep_days 天。"""
    since = pd.Timestamp.now("UTC").tz_localize(None) - pd.Timedelta(days=keep_days)
    out = {}
    for t in ("truth", "forecast"):
        df = _read(dest, t)
        if df.empty:
            continue
        if t == "forecast":
            df = df[pd.to_datetime(df["interval_time"]) >= since]
        old = glob.glob(os.path.join(dest, t, "**", "*.parquet"), recursive=True)
        out[t] = _write(df, dest, t, name="compact.new.parquet")
        for f in old:
            if not f.endswith("compact.new.parquet"):
                os.remove(f)
        os.replace(os.path.join(dest, t, "compact.new.parquet"), os.path.join(dest, t, "compact.parquet"))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="精简数据仓")
    ap.add_argument("--export", metavar="DEST")
    ap.add_argument("--compact", metavar="DEST")
    ap.add_argument("--sync", metavar="DEST", help="把 --src（完整数据仓）里新补的数据并进 DEST，保留 DEST 里已有的")
    ap.add_argument("--src", default=os.environ.get("WQ_DATA_DIR"))
    a = ap.parse_args(argv)
    if a.export:
        n = export(a.src, a.export)
        size = sum(os.path.getsize(f) for f in glob.glob(os.path.join(a.export, "**", "*"), recursive=True)
                   if os.path.isfile(f))
        print(f"→ {a.export}：truth {n['truth']:,} 行、forecast {n['forecast']:,} 行，共 {size / 1e6:.1f} MB")
        return 0
    if a.sync:
        n = sync(a.src, a.sync)
        print(f"→ {a.sync} 合并完成：{n}")
        return 0
    if a.compact:
        n = compact(a.compact)
        print(f"→ {a.compact} 合并完成：{n}")
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
L1 存储层 —— Parquet 为准，DuckDB 当查询引擎。
==============================================
骨架的铁律是"只追加不覆盖"。这里把它做成**文件级**的事实而不是约定：
  · 每次采集写一个**新文件** data/<table>/dt=YYYY-MM-DD/part-<run_id>.parquet
  · 已存在的文件永远不被改写、不被删除
  · 重复键在写之前用反连接剔掉（同一条事实只落一次）
所以"覆盖"在物理上就不可能发生，回滚只需要删掉某个 run_id 的那些文件。
"""
from __future__ import annotations
import glob
import os
import uuid
from datetime import datetime, timezone

import duckdb
import pandas as pd

from config import DATA_DIR

TABLES = ("forecast", "truth", "ingest_log")

# ------------------------------------------------------------------ 列契约
FORECAST_COLS = ["site_id", "model", "variable", "interval_time", "known_at",
                 "value", "known_at_method", "source", "ingested_at"]
TRUTH_COLS = ["site_id", "variable", "interval_time", "known_at",
              "value", "quality", "source", "ingested_at"]
LOG_COLS = ["run_id", "started_at", "finished_at", "mode", "target",
            "http_calls", "rows_seen", "rows_written", "status", "message"]

KEYS = {
    "forecast": ["site_id", "model", "variable", "interval_time", "known_at"],
    "truth":    ["site_id", "variable", "interval_time", "source"],
    "ingest_log": ["run_id"],
}
COLS = {"forecast": FORECAST_COLS, "truth": TRUTH_COLS, "ingest_log": LOG_COLS}


def utcnow() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None).floor("s")


def new_run_id() -> str:
    return utcnow().strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]


def table_dir(table: str) -> str:
    return os.path.join(DATA_DIR, table)


def table_glob(table: str) -> str:
    return os.path.join(table_dir(table), "**", "*.parquet")


def has_data(table: str) -> bool:
    return bool(glob.glob(table_glob(table), recursive=True))


# ------------------------------------------------------------------ 连接
def connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """打开内存 DuckDB，并把每张表注册成 Parquet 之上的视图。
    没有数据时注册一张同构的空表，让下游 SQL 不用写分支。"""
    con = duckdb.connect(":memory:")
    for t in TABLES:
        os.makedirs(table_dir(t), exist_ok=True)
        if has_data(t):
            con.execute(
                f"CREATE OR REPLACE VIEW {t} AS "
                f"SELECT * FROM read_parquet('{table_glob(t)}', union_by_name=true)"
            )
        else:
            empty = _empty_frame(t)
            con.register(f"_empty_{t}", empty)
            con.execute(f"CREATE OR REPLACE VIEW {t} AS SELECT * FROM _empty_{t}")
    # NEM 时间视图：只在读的时候转，磁盘上永远是 UTC
    con.execute("""
        CREATE OR REPLACE VIEW forecast_nem AS
        SELECT * REPLACE (interval_time + INTERVAL 10 HOUR AS interval_time,
                          known_at      + INTERVAL 10 HOUR AS known_at)
        FROM forecast
    """)
    con.execute("""
        CREATE OR REPLACE VIEW truth_nem AS
        SELECT * REPLACE (interval_time + INTERVAL 10 HOUR AS interval_time,
                          known_at      + INTERVAL 10 HOUR AS known_at)
        FROM truth
    """)
    return con


def _empty_frame(table: str) -> pd.DataFrame:
    dt = "datetime64[us]"
    spec = {
        "forecast": {"site_id": "object", "model": "object", "variable": "object",
                     "interval_time": dt, "known_at": dt, "value": "float64",
                     "known_at_method": "object", "source": "object", "ingested_at": dt},
        "truth": {"site_id": "object", "variable": "object", "interval_time": dt,
                  "known_at": dt, "value": "float64", "quality": "object",
                  "source": "object", "ingested_at": dt},
        "ingest_log": {"run_id": "object", "started_at": dt, "finished_at": dt,
                       "mode": "object", "target": "object", "http_calls": "int64",
                       "rows_seen": "int64", "rows_written": "int64",
                       "status": "object", "message": "object"},
    }[table]
    return pd.DataFrame({k: pd.Series(dtype=v) for k, v in spec.items()})


# ------------------------------------------------------------------ 写入
def append(table: str, df: pd.DataFrame, run_id: str, con=None) -> int:
    """把 df 追加进表。返回真正落盘的行数（去重后）。
    绝不改写既有文件——重复键在写之前就被剔除。"""
    if df is None or len(df) == 0:
        return 0
    cols, keys = COLS[table], KEYS[table]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{table} 缺列: {missing}")
    df = df[cols].copy()
    for c in ("interval_time", "known_at", "ingested_at", "started_at", "finished_at"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c]).astype("datetime64[us]")

    # 表内自去重
    df = df.drop_duplicates(subset=keys, keep="last")

    own = con is None
    con = con or connect()
    try:
        if has_data(table):
            con.register("_incoming", df)
            on = " AND ".join(f"t.{k} IS NOT DISTINCT FROM i.{k}" for k in keys)
            df = con.execute(f"""
                SELECT i.* FROM _incoming i
                WHERE NOT EXISTS (SELECT 1 FROM {table} t WHERE {on})
            """).df()
            con.unregister("_incoming")
        if len(df) == 0:
            return 0

        part_col = "interval_time" if "interval_time" in df.columns else "started_at"
        written = 0
        for day, chunk in df.groupby(pd.to_datetime(df[part_col]).dt.date):
            d = os.path.join(table_dir(table), f"dt={day.isoformat()}")
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, f"part-{run_id}.parquet")
            if os.path.exists(path):        # 同一 run 内多次写同一分区
                path = os.path.join(d, f"part-{run_id}-{uuid.uuid4().hex[:4]}.parquet")
            chunk.reset_index(drop=True).to_parquet(path, index=False)
            written += len(chunk)
        return written
    finally:
        if own:
            con.close()


def log_run(run_id, started_at, mode, target, http_calls, rows_seen,
            rows_written, status, message=""):
    row = pd.DataFrame([{
        "run_id": run_id, "started_at": started_at, "finished_at": utcnow(),
        "mode": mode, "target": target, "http_calls": int(http_calls),
        "rows_seen": int(rows_seen), "rows_written": int(rows_written),
        "status": status, "message": (message or "")[:2000],
    }])
    append("ingest_log", row, run_id)


# ------------------------------------------------------------------ 巡检
def status() -> dict:
    con = connect()
    try:
        out = {}
        for t in ("forecast", "truth"):
            r = con.execute(f"""
                SELECT count(*) n, min(interval_time) t0, max(interval_time) t1,
                       count(DISTINCT site_id) sites, count(DISTINCT variable) vars
                FROM {t}
            """).fetchone()
            out[t] = {"rows": r[0], "from": str(r[1]), "to": str(r[2]),
                      "sites": r[3], "variables": r[4]}
        if out["forecast"]["rows"]:
            out["forecast"]["by_method"] = con.execute("""
                SELECT known_at_method, model, count(*) n FROM forecast
                GROUP BY 1,2 ORDER BY n DESC
            """).df().to_dict("records")
        if out["truth"]["rows"]:
            out["truth"]["by_quality"] = con.execute("""
                SELECT quality, source, count(*) n FROM truth GROUP BY 1,2 ORDER BY n DESC
            """).df().to_dict("records")
        out["files"] = {t: len(glob.glob(table_glob(t), recursive=True)) for t in TABLES}
        out["last_runs"] = (con.execute("""
            SELECT run_id, mode, target, status, rows_written, started_at
            FROM ingest_log ORDER BY started_at DESC LIMIT 8
        """).df().to_dict("records") if has_data("ingest_log") else [])
        return out
    finally:
        con.close()

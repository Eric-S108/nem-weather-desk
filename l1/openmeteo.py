"""
L1 取数层 —— Open-Meteo 客户端。
=================================
只负责"把 JSON 变成规整的长表"，不碰任何 known_at 语义（那是 ingest.py 的事）。
四个端点：
  forecast        实时预报
  single_runs     按初始化时刻取整个模型运行 —— known_at 精确的唯一来源
  previous_runs   固定时效偏移的长历史 —— 便宜，但 known_at 只能给保守上界
  archive         ERA5 再分析真值
"""
from __future__ import annotations
import random
import time
from typing import Iterable

import pandas as pd
import requests

from config import (EP_ARCHIVE, EP_FORECAST, EP_HIST_FORECAST, EP_PREVIOUS_RUNS,
                    EP_SINGLE_RUNS, HTTP_TIMEOUT, MAX_RETRIES, BACKOFF_BASE,
                    OM_TO_VAR, PACE_SECONDS, USER_AGENT, VARIABLES)

_last_call = [0.0]
CALLS = [0]


class OpenMeteoError(RuntimeError):
    pass


def _pace():
    dt = time.monotonic() - _last_call[0]
    if dt < PACE_SECONDS:
        time.sleep(PACE_SECONDS - dt)
    _last_call[0] = time.monotonic()


def _get(url: str, params: dict) -> list[dict]:
    """带退避的 GET。永远返回 list（单点也包成单元素列表）。"""
    last = None
    for attempt in range(MAX_RETRIES):
        _pace()
        try:
            CALLS[0] += 1
            r = requests.get(url, params=params, timeout=HTTP_TIMEOUT,
                             headers={"User-Agent": USER_AGENT})
        except requests.RequestException as e:
            # 带上异常类型：ConnectTimeout / ReadTimeout / SSLError / NameResolutionError
            # 是完全不同的病，笼统一句"连接失败"没法对症下药
            root = e
            while getattr(root, "__cause__", None) is not None:
                root = root.__cause__
            last = f"{type(e).__name__}: {type(root).__name__}: {str(root)[:200]}"
            time.sleep(min(BACKOFF_BASE ** attempt + random.random(), 20.0))
            continue

        if r.status_code == 200:
            j = r.json()
            return j if isinstance(j, list) else [j]

        if r.status_code == 429 or r.status_code >= 500:
            wait = float(r.headers.get("Retry-After") or 0) or (BACKOFF_BASE ** attempt)
            last = OpenMeteoError(f"HTTP {r.status_code}: {r.text[:200]}")
            time.sleep(wait + random.random())
            continue

        # 4xx（参数错、超出存档范围）不重试
        raise OpenMeteoError(f"HTTP {r.status_code} {url}\nparams={params}\n{r.text[:400]}")

    raise OpenMeteoError(f"重试 {MAX_RETRIES} 次仍失败: {url}\n  末次错误: {last}\n"
                         f"  hourly 列数={len(str(params.get('hourly','')).split(','))} "
                         f"区间={params.get('start_date')}..{params.get('end_date')}")


def _split_key(key: str) -> tuple[str | None, str]:
    """'temperature_2m_ecmwf_ifs025' -> ('temp', 'ecmwf_ifs025')
       'shortwave_radiation_previous_day3' -> ('ghi', 'previous_day3')
       'wind_speed_100m' -> ('wind100', '')"""
    for om, var in OM_TO_VAR.items():
        if key == om:
            return var, ""
        if key.startswith(om + "_"):
            return var, key[len(om) + 1:]
    return None, ""


def _to_long(payloads: list[dict], site_ids: list[str]) -> pd.DataFrame:
    """把 Open-Meteo 的 hourly 字典摊成长表 [site_id, variable, suffix, interval_time, value]。"""
    if len(payloads) != len(site_ids):
        raise OpenMeteoError(f"返回 {len(payloads)} 个位置，期望 {len(site_ids)} 个")
    frames = []
    for sid, p in zip(site_ids, payloads):
        h = p.get("hourly")
        if not h or "time" not in h:
            continue
        t = pd.to_datetime(pd.Series(h["time"])).astype("datetime64[us]")
        for key, vals in h.items():
            if key == "time":
                continue
            var, suffix = _split_key(key)
            if var is None or vals is None:
                continue
            frames.append(pd.DataFrame({
                "site_id": sid, "variable": var, "suffix": suffix,
                "interval_time": t, "value": pd.to_numeric(pd.Series(vals), errors="coerce"),
            }))
    if not frames:
        return pd.DataFrame(columns=["site_id", "variable", "suffix", "interval_time", "value"])
    out = pd.concat(frames, ignore_index=True)
    return out.dropna(subset=["value"])


def _coords(sites) -> dict:
    return {"latitude": ",".join(f"{s[3]:.4f}" for s in sites),
            "longitude": ",".join(f"{s[4]:.4f}" for s in sites)}


def _hourly(vars_: Iterable[str] | None = None) -> str:
    vars_ = list(vars_ or VARIABLES.keys())
    return ",".join(VARIABLES[v] for v in vars_)


# ------------------------------------------------------------------ 四个端点
def fetch_single_run(sites, run_iso: str, model: str, variables=None) -> pd.DataFrame:
    """按初始化时刻取一次模型运行。run_iso 形如 '2026-06-01T00:00'（UTC）。"""
    p = {**_coords(sites), "hourly": _hourly(variables), "models": model,
         "run": run_iso, "timezone": "UTC"}
    df = _to_long(_get(EP_SINGLE_RUNS, p), [s[0] for s in sites])
    df["model"] = model
    return df.drop(columns=["suffix"], errors="ignore")


def fetch_previous_runs(sites, start: str, end: str, model: str,
                        max_day: int = 5, variables=None) -> pd.DataFrame:
    """固定时效偏移的历史。返回带 lead_days 列（0..max_day）。"""
    variables = list(variables or VARIABLES.keys())
    cols = []
    for v in variables:
        om = VARIABLES[v]
        cols.append(om)
        cols += [f"{om}_previous_day{d}" for d in range(1, max_day + 1)]
    p = {**_coords(sites), "hourly": ",".join(cols), "models": model,
         "start_date": start, "end_date": end, "timezone": "UTC"}
    df = _to_long(_get(EP_PREVIOUS_RUNS, p), [s[0] for s in sites])
    if df.empty:
        return df
    df["lead_days"] = (df["suffix"].str.extract(r"previous_day(\d+)")[0]
                       .fillna("0").astype(int))
    df["model"] = model
    return df.drop(columns=["suffix"])


def fetch_forecast(sites, models, past_days: int = 2, forecast_days: int = 7,
                   variables=None) -> pd.DataFrame:
    """实时预报（含最近 past_days 的回溯段）。"""
    p = {**_coords(sites), "hourly": _hourly(variables),
         "models": ",".join(models), "past_days": past_days,
         "forecast_days": forecast_days, "timezone": "UTC"}
    df = _to_long(_get(EP_FORECAST, p), [s[0] for s in sites])
    if df.empty:
        return df
    # 单模型时 Open-Meteo 不加后缀
    df["model"] = df["suffix"].where(df["suffix"] != "", models[0])
    return df.drop(columns=["suffix"])


def fetch_archive(sites, start: str, end: str, variables=None) -> pd.DataFrame:
    """ERA5 再分析真值。"""
    p = {**_coords(sites), "hourly": _hourly(variables),
         "start_date": start, "end_date": end, "timezone": "UTC"}
    df = _to_long(_get(EP_ARCHIVE, p), [s[0] for s in sites])
    return df.drop(columns=["suffix"], errors="ignore")


def fetch_ifs_analysis(sites, start: str, end: str, variables=None) -> pd.DataFrame:
    """ECMWF IFS 9km 存档（无延迟），当 ERA5 落地前的临时真值。"""
    p = {**_coords(sites), "hourly": _hourly(variables), "models": "ecmwf_ifs025",
         "start_date": start, "end_date": end, "timezone": "UTC"}
    df = _to_long(_get(EP_HIST_FORECAST, p), [s[0] for s in sites])
    return df.drop(columns=["suffix"], errors="ignore")

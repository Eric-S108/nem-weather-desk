"""
L1 配置 —— 站点、变量、模型、端点。
=====================================
时区约定：**磁盘上一律存 UTC（naive datetime64）**。
NEM 时间 = UTC + 10h，全年固定、无夏令时，所以两者一一对应、无歧义。
读的时候用 features.to_nem() / store 里的 *_nem 视图转过去，写的时候永远不转。
"""
from __future__ import annotations
import os
from datetime import timedelta

# ------------------------------------------------------------------ 路径
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("WQ_DATA_DIR", os.path.join(os.path.dirname(ROOT), "data"))

# ------------------------------------------------------------------ 时间
NEM_OFFSET = timedelta(hours=10)          # NEM 时间 = UTC + 10h，固定
NEM_TZ_NOTE = "NEM 时间 = UTC+10，全年固定，无夏令时"

# ------------------------------------------------------------------ 站点（各州 REZ 中心）
SITES = [
    # site_id, NEM 区, 名称, lat, lon, 光权重, 风权重, VRE 铭牌/峰值需求
    ("NSW1_CWO",  "NSW1", "Central-West Orana REZ · Dubbo", -32.25, 148.60, 0.62, 0.38, 0.48),
    ("QLD1_DD",   "QLD1", "Darling Downs REZ · Dalby",      -27.18, 151.27, 0.72, 0.28, 0.42),
    ("VIC1_MR",   "VIC1", "Murray River REZ · Kerang",      -35.74, 143.92, 0.38, 0.62, 0.56),
    ("SA1_MN",    "SA1",  "Mid North REZ · Port Augusta",   -32.49, 137.77, 0.34, 0.66, 0.88),
    ("TAS1_NW",   "TAS1", "North-West REZ · Waratah",       -41.44, 145.53, 0.10, 0.90, 0.62),
]
SITE_BY_ID = {s[0]: s for s in SITES}

# ------------------------------------------------------------------ 变量
# 内部名 -> Open-Meteo 变量名
VARIABLES = {
    "ghi":     "shortwave_radiation",
    "wind100": "wind_speed_100m",
    "temp":    "temperature_2m",
}
OM_TO_VAR = {v: k for k, v in VARIABLES.items()}

# ------------------------------------------------------------------ 模型
# Single Runs / Historical Forecast 里的模型标识。
# ECMWF IFS HRES 9km 的存档最长（2024-03-14 起），是唯一值得做长回填的那个。
MODELS = {
    "ecmwf_ifs025": {"runs_utc": [0, 6, 12, 18], "archive_from": "2024-03-14",
                     "label": "ECMWF IFS HRES 9km"},
    "gfs_global":   {"runs_utc": [0, 6, 12, 18], "archive_from": "2026-04-02",
                     "label": "NOAA GFS"},
    "icon_global":  {"runs_utc": [0, 6, 12, 18], "archive_from": "2026-04-02",
                     "label": "DWD ICON"},
    "gem_global":   {"runs_utc": [0, 12],        "archive_from": "2026-04-02",
                     "label": "CMC GEM"},
}
LIVE_MODELS = ["ecmwf_ifs025", "gfs_global", "icon_global", "gem_global"]

# ------------------------------------------------------------------ 端点
EP_FORECAST      = "https://api.open-meteo.com/v1/forecast"
EP_SINGLE_RUNS   = "https://single-runs-api.open-meteo.com/v1/forecast"
EP_PREVIOUS_RUNS = "https://previous-runs-api.open-meteo.com/v1/forecast"
EP_ARCHIVE       = "https://archive-api.open-meteo.com/v1/archive"          # ERA5，延迟 ~5 天
EP_HIST_FORECAST = "https://historical-forecast-api.open-meteo.com/v1/forecast"

# ------------------------------------------------------------------ known_at 来源等级
#   exact   —— Single Runs API 的 run= 参数，就是模型初始化时刻，精确
#   request —— 实时抓取，用请求时刻当 known_at。它 ≥ 真实发布时刻，
#              所以只会让 as-of 过滤更严格，**永远不会泄漏**
#   approx  —— Previous Runs API 推出来的保守上界（见 ingest.py）
#              便宜但不精确，strict 模式下默认排除
KNOWN_AT_METHODS = ("exact", "request", "approx")
STRICT_METHODS = ("exact", "request")

# ------------------------------------------------------------------ 真值
#   final       —— ERA5 再分析，1940 至今，延迟约 5 天，最终口径
#   provisional —— ECMWF IFS 9km 分析场，无延迟，用来先算着，等 ERA5 落地再对账
TRUTH_SOURCES = {
    "era5":          {"quality": "final",       "lag_days": 5,   "endpoint": EP_ARCHIVE},
    "ifs_analysis":  {"quality": "provisional", "lag_days": 0.25, "endpoint": EP_HIST_FORECAST},
}

# ------------------------------------------------------------------ HTTP
HTTP_TIMEOUT = 60
MAX_RETRIES = 6
BACKOFF_BASE = 1.7          # 指数退避基数（秒）
PACE_SECONDS = float(os.environ.get("WQ_PACE_SECONDS", "0.35"))   # 每次调用之间的最小间隔
CHUNK_DAYS = 120            # 长区间回填的分块大小
USER_AGENT = "WeatherQuant-NEM-L1/1.0 (personal research)"

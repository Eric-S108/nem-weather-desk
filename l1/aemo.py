"""
L1 · AEMO NEMWeb 接入 —— 真实电价与需求，写进同一张时点存储表。
================================================================
骨架第 2 条：**AEMO 官方预测才是基准线**，真正的 alpha 是预测 AEMO 预测的偏差。
所以这里同时拿三样：
  实绩  DISPATCH/PRICE (RRP)、DISPATCH/REGIONSUM (需求、半调度出力)  → truth 表
  官方预测  PREDISPATCH / P5MIN                                      → forecast 表

schema 一行没改。约定：**市场数据的 site_id 直接用 NEM 区代码**（NSW1 / QLD1 …），
和气象站点 id（NSW1_CWO …）天然区分开。

三个容易搞错、这里写死的地方
--------------------------
1. **SETTLEMENTDATE 是区间结束时刻，且是 NEM 时间（UTC+10）**。
   标着 00:05 的那条覆盖 00:00–00:05。所以归到 UTC 小时是
   `(SETTLEMENTDATE − 5min − 10h).floor('h')`，少减那 5 分钟会整体错一格。
2. **必须过滤 INTERVENTION = 0**。干预运行的价格不是市场出清价，
   混进去回测就废了（骨架里 APC 区制那条的同类问题）。
3. **known_at 不等于 interval_time**。调度价在区间结束后约 1 分钟发布，
   这里保守取 +10 分钟；预调度取运行时刻（PREDISPATCH_RUN_DATETIME / SEQNO 反推），
   **不是 LASTCHANGED**——那是记录修改时间，与预测何时发出无关。

用法
----
    python aemo.py --selftest                         离线：解析 + 时区 + 聚合
    python aemo.py --probe                            探测 NEMWeb 可达性与文件名
    python aemo.py --current --hours 24                Reports/Current 增量（2 天窗口）
    python aemo.py --archive --start 2025-01 --end 2026-07   MMSDM 月度归档回填
    python aemo.py --predispatch                       官方预测（Current，约 2 天）
    python aemo.py --inspect-predispatch               看一眼最新一份预调度，不写库
    python aemo.py --predispatch-archive --start 2025-10-25 --end 2026-10-23 --lead-lo 24 --lead-hi 30
                                                       官方预测历史（日归档，约 13 个月，断点续跑）
"""
from __future__ import annotations
import argparse
import hashlib
import io
import os
import re
import sys
import time
import zipfile
from typing import Iterable

import pandas as pd
import requests

import store
from config import HTTP_TIMEOUT, MAX_RETRIES, BACKOFF_BASE, PACE_SECONDS

# ------------------------------------------------------------------ 端点
NEMWEB = "https://nemweb.com.au"
CUR_DISPATCH = f"{NEMWEB}/REPORTS/CURRENT/DispatchIS_Reports/"
CUR_PREDISP = f"{NEMWEB}/REPORTS/CURRENT/PredispatchIS_Reports/"
CUR_P5MIN = f"{NEMWEB}/REPORTS/CURRENT/P5_Reports/"
# 滚动归档：每天一个 zip，保留约 13 个月，格式与 Current 完全一致
ARC_DISPATCH = f"{NEMWEB}/Reports/ARCHIVE/DispatchIS_Reports/"
ARC_PREDISP = f"{NEMWEB}/Reports/ARCHIVE/PredispatchIS_Reports/"
MMSDM = "https://nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM"
# 月度归档的文件名模板。AEMO 实际用的是 PUBLIC_ARCHIVE#TABLE#FILE01#YYYYMM010000.zip，
# 文件名里带 '#'——在 URL 里必须编码成 %23，否则会被当成锚点，请求到的是目录页。
# 历史上还出现过 PUBLIC_DVD_ 前缀，所以保留两套模板 + 目录发现三重兜底。
MMSDM_TPL = ("{base}/{y}/MMSDM_{y}_{m:02d}/MMSDM_Historical_Data_SQLLoader/DATA/"
             "PUBLIC_ARCHIVE%23{table}%23FILE01%23{y}{m:02d}010000.zip")
MMSDM_TPL_ALT = ("{base}/{y}/MMSDM_{y}_{m:02d}/MMSDM_Historical_Data_SQLLoader/DATA/"
                 "PUBLIC_DVD_{table}_{y}{m:02d}010000.zip")
MMSDM_DIR = "{base}/{y}/MMSDM_{y}_{m:02d}/MMSDM_Historical_Data_SQLLoader/DATA/"

REGIONS = ["NSW1", "QLD1", "SA1", "TAS1", "VIC1"]
NEM_OFFSET_H = 10                      # NEM 时间 = UTC+10，全年固定
DISPATCH_MIN = 5                       # 调度区间长度
PUBLISH_DELAY_MIN = 10                 # 保守：区间结束后多久才算"可见"
MIN_INTERVALS_PER_HOUR = 10            # 一小时 12 个区间，少于这个数不聚合
# 预调度只保留这个时效带（小时）。回测用 D+1，留 6~48h 足够，
# 全量存下来一年几百万行，绝大部分永远用不到。
LEAD_KEEP_H = (float(os.environ.get("WQ_LEAD_LO", 6)),
               float(os.environ.get("WQ_LEAD_HI", 48)))
CALLS = [0]
_last = [0.0]


class AemoError(RuntimeError):
    pass


def _pace():
    dt = time.monotonic() - _last[0]
    if dt < PACE_SECONDS:
        time.sleep(PACE_SECONDS - dt)
    _last[0] = time.monotonic()


# ------------------------------------------------------------------ HTTP 层
# nemweb 的目录页很小、秒回，但几十 MB 的 zip 经常在传输中途被切断，报
#   SSL: UNEXPECTED_EOF_WHILE_READING
# 服务端没发 TLS close_notify 就断了（也可能是中间设备对长连接下手）。
# 一次性 requests.get 遇到这个只能整包重来，几十 MB 反复重下必然失败。
# 所以下载走三件事：① 流式分块写盘 ② 断了用 HTTP Range 断点续传
# ③ 落地缓存，重跑不重下。浏览器 UA 也补上——nemweb 对陌生 UA 偶尔挑剔。
BROWSER_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
HEADERS = {"User-Agent": os.environ.get("WQ_UA", BROWSER_UA),
           "Accept": "*/*", "Accept-Encoding": "identity", "Connection": "close"}
CACHE_DIR = os.environ.get(
    "WQ_HTTP_CACHE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".httpcache"))
CHUNK = 1 << 18                        # 256 KiB
MAX_BACKOFF = 20.0                     # 退避封顶，别让用户等到想按 Ctrl-C
_SESSION = None


def _session():
    global _SESSION
    if _SESSION is None:
        s = requests.Session()
        s.headers.update(HEADERS)
        # 关掉 urllib3 自己的重试：我们要自己控制 Range 续传，交给它会整包重来
        ad = requests.adapters.HTTPAdapter(max_retries=0, pool_maxsize=4)
        s.mount("https://", ad); s.mount("http://", ad)
        _SESSION = s
    return _SESSION


def _cache_path(url: str) -> str:
    h = hashlib.sha1(url.encode()).hexdigest()[:16]
    name = url.rstrip("/").split("/")[-1].replace("%23", "#")[:80] or "index"
    return os.path.join(CACHE_DIR, f"{h}-{name}")


def _get(url: str, binary: bool = False):
    """文本走内存；二进制走带断点续传的落地缓存。"""
    if not binary:
        last = None
        for i in range(MAX_RETRIES):
            _pace()
            try:
                CALLS[0] += 1
                r = _session().get(url, timeout=HTTP_TIMEOUT)
            except requests.RequestException as e:
                last = e
                time.sleep(min(BACKOFF_BASE ** i, MAX_BACKOFF)); continue
            if r.status_code == 200:
                return r.text
            if r.status_code == 404:
                raise AemoError(f"404 {url}")
            last = AemoError(f"HTTP {r.status_code} {url}")
            time.sleep(min(BACKOFF_BASE ** i, MAX_BACKOFF))
        raise AemoError(f"重试 {MAX_RETRIES} 次仍失败: {url} :: {last}")
    return download(url)


def download(url: str, quiet: bool = True, cache: bool = True) -> bytes:
    """分块下载 + Range 续传 + 磁盘缓存。返回完整字节。

    cache=False 时下完即删——Current 目录每 5 分钟一个新文件名，缓存会无限膨胀，
    而且那些文件本来就小、重下不痛。归档回填才需要留断点。
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _cache_path(url)
    done = path + ".done"
    if os.path.exists(done):
        with open(done, "rb") as fh:
            data = fh.read()
        if not cache:
            os.remove(done)
        return data

    total = None
    last = None
    for i in range(MAX_RETRIES):
        have = os.path.getsize(path) if os.path.exists(path) else 0
        hdr = {"Range": f"bytes={have}-"} if have else {}
        _pace()
        try:
            CALLS[0] += 1
            r = _session().get(url, timeout=HTTP_TIMEOUT, stream=True, headers=hdr)
            if r.status_code == 404:
                raise AemoError(f"404 {url}")
            if have and r.status_code == 200:
                # 服务端不认 Range，只能从头来；先把半截文件丢掉
                os.remove(path); have = 0
            elif have and r.status_code != 206:
                raise AemoError(f"HTTP {r.status_code}（续传被拒） {url}")
            elif not have and r.status_code != 200:
                raise AemoError(f"HTTP {r.status_code} {url}")
            cl = r.headers.get("Content-Length")
            if cl is not None:
                total = have + int(cl)
            with open(path, "ab") as fh:
                for chunk in r.iter_content(CHUNK):
                    if chunk:
                        fh.write(chunk)
            got = os.path.getsize(path)
            if total is not None and got < total:
                raise AemoError(f"只收到 {got}/{total} 字节")
            os.replace(path, done)
            with open(done, "rb") as fh:
                data = fh.read()
            if not cache:
                os.remove(done)
            return data
        except AemoError as e:
            if str(e).startswith("404"):
                raise
            last = e
        except (requests.RequestException, OSError) as e:
            last = e
        got = os.path.getsize(path) if os.path.exists(path) else 0
        if not quiet:
            print(f"      续传 {url.split('/')[-1]} 已有 {got/1e6:.1f} MB：{str(last)[:70]}")
        time.sleep(min(BACKOFF_BASE ** i, MAX_BACKOFF))
    raise AemoError(f"重试 {MAX_RETRIES} 次仍失败: {url} :: {last}")


def list_zips(dir_url: str) -> list[str]:
    """NEMWeb 目录是 IIS 列表页，正则抓 .zip 链接即可。"""
    html = _get(dir_url)
    out, seen = [], set()
    for m in re.finditer(r'href="([^"]+?\.zip)"', html, re.I):
        href = m.group(1)
        name = href.split("/")[-1]
        if name in seen:
            continue
        seen.add(name)
        out.append(href if href.startswith("http") else NEMWEB + "/" + href.lstrip("/"))
    return sorted(out)


# ================================================================== AEMO CSV 解析
def parse_mms_csv(text: str) -> dict[tuple[str, str], pd.DataFrame]:
    """解析 AEMO 的 C/I/D 三段式 CSV。

    I 行给出某张表的列名，其后的 D 行是数据，直到下一个 I 行。
    返回 {(报告, 子表): DataFrame}，例如 ("DISPATCH","PRICE")。
    这个函数是纯字符串处理，可以完全离线测试——事实上 --selftest 就是测它。
    """
    tables: dict[tuple[str, str], list] = {}
    cols: dict[tuple[str, str], list] = {}
    for raw in text.splitlines():
        if not raw or raw[0] not in "CID":
            continue
        f = next(csv_rows(raw))
        tag = f[0]
        if tag == "C":
            continue
        if len(f) < 5:
            continue
        key = (f[1].upper(), f[2].upper())
        if tag == "I":
            cols[key] = [c.strip() for c in f[4:]]
            tables.setdefault(key, [])
        elif tag == "D" and key in cols:
            tables[key].append(f[4:])
    out = {}
    for key, rows in tables.items():
        if not rows:
            continue
        c = cols[key]
        w = len(c)
        norm = [(r + [None] * w)[:w] for r in rows]
        out[key] = pd.DataFrame(norm, columns=c)
    return out


def csv_rows(line: str):
    """极简 CSV 分词：只需处理双引号包裹与逗号，AEMO 不用转义引号。"""
    out, buf, q = [], [], False
    for ch in line:
        if ch == '"':
            q = not q
        elif ch == "," and not q:
            out.append("".join(buf)); buf = []
        else:
            buf.append(ch)
    out.append("".join(buf))
    yield out


def unzip_all(blob: bytes) -> str:
    """NEMWeb 的 zip 有时嵌套一层（zip 里还是 zip）。全部展开成一段文本。"""
    texts = []
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for n in z.namelist():
            data = z.read(n)
            if n.lower().endswith(".zip"):
                texts.append(unzip_all(data))
            else:
                texts.append(data.decode("utf-8", errors="replace"))
    return "\n".join(texts)


# ================================================================== 时间与聚合
def settle_to_utc_hour(sd: pd.Series) -> pd.Series:
    """SETTLEMENTDATE（NEM 时间、区间**结束**时刻）→ 该区间所属的 UTC 整点。"""
    t = pd.to_datetime(sd, format="%Y/%m/%d %H:%M:%S", errors="coerce")
    bad = t.isna()
    if bad.any():
        t = t.where(~bad, pd.to_datetime(sd[bad], errors="coerce"))
    start_nem = t - pd.Timedelta(minutes=DISPATCH_MIN)      # 转成区间起点
    return (start_nem - pd.Timedelta(hours=NEM_OFFSET_H)).dt.floor("h")


def to_hourly_truth(df: pd.DataFrame, value_col: str, out_vars: dict,
                    source: str) -> pd.DataFrame:
    """5 分钟表 → 小时真值行。out_vars: {变量名: 聚合方式}。"""
    d = df.copy()
    # df.get(col, "0") 在列缺失时返回的是 **Python 字符串**，后面 .astype 会炸。
    # 干预区间在 DISPATCHPRICE 里有两行，结算用的是 INTERVENTION=0 那条。
    if "INTERVENTION" in d.columns:
        d = d[d["INTERVENTION"].astype(str).str.strip().isin(["0", "0.0", ""])]
    if d.empty:
        return pd.DataFrame()
    d = d[d["REGIONID"].isin(REGIONS)]
    d["hour"] = settle_to_utc_hour(d["SETTLEMENTDATE"])
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=["hour", value_col])
    # 去重：同一 (区, 5 分钟区间) 只该有一条。归档文件之间有重叠，
    # 不去重的话重叠区间会被算两次，均值本身没错，但计数校验会失真。
    if "SETTLEMENTDATE" in d.columns:
        d = d.drop_duplicates(subset=["REGIONID", "SETTLEMENTDATE"], keep="last")
    # 一小时应有 12 个 5 分钟区间。只收到两三个就求均值，得到的是一个
    # 看起来正常、其实代表不了这一小时的数——比缺失更危险，因为它不会报错。
    cnt = d.groupby(["REGIONID", "hour"])[value_col].size().rename("n_iv")
    full = cnt[cnt >= MIN_INTERVALS_PER_HOUR].reset_index()[["REGIONID", "hour"]]
    if full.empty:
        return pd.DataFrame()
    d = d.merge(full, on=["REGIONID", "hour"], how="inner")
    frames = []
    for var, how in out_vars.items():
        g = d.groupby(["REGIONID", "hour"])[value_col].agg(how).reset_index()
        g.columns = ["site_id", "interval_time", "value"]
        g["variable"] = var
        frames.append(g)
    o = pd.concat(frames, ignore_index=True)
    o["known_at"] = (o["interval_time"] + pd.Timedelta(hours=1)
                     + pd.Timedelta(minutes=PUBLISH_DELAY_MIN))
    o["quality"] = "final"
    o["source"] = source
    o["ingested_at"] = store.utcnow()
    return o


# ================================================================== 从一堆文本抽表
def extract(texts: Iterable[str]) -> dict[tuple[str, str], pd.DataFrame]:
    acc: dict[tuple[str, str], list] = {}
    for t in texts:
        for k, df in parse_mms_csv(t).items():
            acc.setdefault(k, []).append(df)
    return {k: pd.concat(v, ignore_index=True) for k, v in acc.items()}


def rows_from_tables(tabs: dict, source: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """把解析出的表转成 (truth 行, forecast 行)。"""
    truth = []
    price = tabs.get(("DISPATCH", "PRICE"))
    if price is not None and "RRP" in price.columns:
        truth.append(to_hourly_truth(price, "RRP",
                                     {"rrp": "mean", "rrp_max": "max"}, source))
    rs = tabs.get(("DISPATCH", "REGIONSUM"))
    if rs is not None:
        if "TOTALDEMAND" in rs.columns:
            truth.append(to_hourly_truth(rs, "TOTALDEMAND", {"demand": "mean"}, source))
        for col, var in (("SS_SOLAR_UIGF", "solar_uigf"), ("SS_WIND_UIGF", "wind_uigf")):
            if col in rs.columns:
                truth.append(to_hourly_truth(rs, col, {var: "mean"}, source))
    t = pd.concat([x for x in truth if x is not None and len(x)], ignore_index=True) \
        if any(len(x) for x in truth if x is not None) else pd.DataFrame()

    # ---- 官方预测不在这里处理 ----
    # 这个函数吃的是**拼接后**的表：几十个文件的行混在一起，已经分不清哪一行
    # 来自哪一次预调度运行，只能从 PREDISPATCHSEQNO 去猜运行时刻——之前猜错了
    # 两次（LASTCHANGED、PP×30 分钟），第三次的自标定也在 Current 上得到
    # "lead 0.5~0.5h、写进 0 行"。预调度现在走 parse_predispatch()，**逐个文件**
    # 解析，运行时刻直接读文件头 C 行里的发布时间。见下方"官方预测"一节。
    fc: list = []
    f = pd.concat(fc, ignore_index=True) if fc else pd.DataFrame()
    return t, f


# ================================================================== 官方预测（预调度）
PREDISP_MODEL = "aemo_predispatch"
PREDISP_TABLE = ("PREDISPATCH", "REGION_PRICES")
_TS_RE = re.compile(r"(?<!\d)(\d{14}|\d{12})(?!\d)")


def unzip_named(blob: bytes, prefix: str = "") -> list[tuple[str, str]]:
    """和 unzip_all 一样展开嵌套 zip，但**保留每个文件的名字**。

    预调度必须逐文件处理：一个文件 = 一次运行，运行时刻要从文件本身读。
    """
    out = []
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for n in z.namelist():
            data = z.read(n)
            if n.lower().endswith(".zip"):
                out.extend(unzip_named(data, prefix + n + "/"))
            else:
                out.append((prefix + n, data.decode("utf-8", errors="replace")))
    return out


def _nem_to_utc(ts: pd.Timestamp) -> pd.Timestamp:
    return ts - pd.Timedelta(hours=NEM_OFFSET_H)


def publish_time(text: str, name: str = "") -> pd.Timestamp | None:
    """这一份预调度**什么时候发出来的**（UTC）。

    首选文件头 C 行：  C,NEMP.WORLD,PREDISPATCHIS,AEMO,PUBLIC,2026/08/28,12:01:06,...
    第 6、7 个字段就是文件生成时刻（NEM 时间）。这是 AEMO 自己写的，
    不依赖任何列名约定。退而用文件名里最后一个 12/14 位时间戳
    （PUBLIC_PREDISPATCHIS_202608281230_20260828120106.zip 的后一个）。
    两者都有时取较晚的那个——宁可晚，不可早，早了就是前视。
    """
    cands = []
    for line in text.splitlines()[:5]:
        if line.startswith("C,"):
            f = next(csv_rows(line))
            if len(f) >= 7:
                t = pd.to_datetime(f"{f[5].strip()} {f[6].strip()}",
                                   format="%Y/%m/%d %H:%M:%S", errors="coerce")
                if pd.notna(t):
                    cands.append(_nem_to_utc(t))
            break
    base = name.replace("\\", "/").split("/")[-1]
    stamps = _TS_RE.findall(base)
    if stamps:
        st = stamps[-1]
        fmt = "%Y%m%d%H%M%S" if len(st) == 14 else "%Y%m%d%H%M"
        t = pd.to_datetime(st, format=fmt, errors="coerce")
        if pd.notna(t):
            cands.append(_nem_to_utc(t))
    return max(cands) if cands else None


def _only_region_prices(text: str) -> str:
    """预调度文件里有十几张表（约束、联络线、机组……），我们只要分区电价。
    先按行筛掉其余的再解析，单个文件快一个数量级。"""
    key = "," + ",".join(PREDISP_TABLE) + ","
    return "\n".join(l for l in text.splitlines() if l.startswith("C,") or key in l)


def parse_predispatch(text: str, name: str = "",
                      lead_band: tuple[float, float] | None = None) -> pd.DataFrame:
    """一份预调度文件 → 逐小时的预测行（site_id, interval_time, known_at, value, method）。

      · 只要 INTERVENTION = 0（干预运行不是市场口径）
      · DATETIME 是半小时时段的**结束**时刻、NEM 时间；两个半小时都在才凑成一小时，
        只有半截的小时丢掉（和实绩 RRP 的"少于 10 个 5 分钟区间不聚合"同一口径）
      · known_at = 文件发布时刻（publish_time）。读不到时退回"本次运行第一个时段的
        结束时刻"，这是个**上界**——晚于真实发布，所以不会前视，标成 approx
      · 只留 lead_band 内的时效（默认 LEAD_KEEP_H）
    """
    lo, hi = lead_band or LEAD_KEEP_H
    tabs = parse_mms_csv(_only_region_prices(text))
    df = tabs.get(PREDISP_TABLE)
    if df is None or "RRP" not in df.columns or "DATETIME" not in df.columns:
        return pd.DataFrame()
    d = df[df["REGIONID"].isin(REGIONS)].copy()
    if "INTERVENTION" in d.columns:
        d = d[d["INTERVENTION"].astype(str).str.strip().isin(["0", "0.0", ""])]
    if d.empty:
        return pd.DataFrame()
    d["value"] = pd.to_numeric(d["RRP"], errors="coerce")
    d["interval_time"] = settle_to_utc_hour(d["DATETIME"])
    d = d.dropna(subset=["value", "interval_time"])

    ka = publish_time(text, name)
    method = "exact"
    if ka is None:
        end = pd.to_datetime(d["DATETIME"], format="%Y/%m/%d %H:%M:%S", errors="coerce")
        ka = _nem_to_utc(end.min())
        method = "approx"

    g = (d.groupby(["REGIONID", "interval_time"])["value"]
           .agg(["mean", "count"]).reset_index())
    g = g[g["count"] >= 2]
    g["known_at"] = ka
    lead = (g["interval_time"] - g["known_at"]).dt.total_seconds() / 3600.0
    g = g[(lead >= lo) & (lead <= hi)]
    return pd.DataFrame({"site_id": g["REGIONID"].to_numpy(),
                         "interval_time": g["interval_time"].to_numpy(),
                         "known_at": g["known_at"].to_numpy(),
                         "value": g["mean"].to_numpy(),
                         "method": method})


def _predispatch_frame(parts: list[pd.DataFrame], source: str) -> pd.DataFrame:
    parts = [x for x in parts if x is not None and len(x)]
    if not parts:
        return pd.DataFrame()
    f = pd.concat(parts, ignore_index=True)
    return pd.DataFrame({"site_id": f["site_id"], "model": PREDISP_MODEL,
                         "variable": "rrp", "interval_time": f["interval_time"],
                         "known_at": f["known_at"], "value": f["value"],
                         "known_at_method": f["method"], "source": source,
                         "ingested_at": store.utcnow()})


def _write_predispatch(parts: list[pd.DataFrame], source: str) -> int:
    f = _predispatch_frame(parts, source)
    return store.append("forecast", f, store.new_run_id()) if len(f) else 0


def ingest_predispatch_current(hours: int, lead_band=None) -> tuple[int, int]:
    """Reports/Current 里最近 hours 小时的预调度（每 30 分钟一个文件，约保留 2 天）。
    只够做增量；历史要用 --predispatch-archive。"""
    urls = list_zips(CUR_PREDISP)
    if not urls:
        raise AemoError(f"目录里没找到 zip：{CUR_PREDISP}")
    take = urls[-max(1, hours * 2):]
    print(f"  {CUR_PREDISP}\n  共 {len(urls)} 个文件，取最近 {len(take)} 个")
    parts = []
    for u in take:
        try:
            for name, text in unzip_named(download(u, cache=False)):
                parts.append(parse_predispatch(text, name or u, lead_band))
        except AemoError as e:
            print(f"    跳过 {u.split('/')[-1]}: {str(e)[:80]}")
    n = _write_predispatch(parts, "nemweb_predispatch")
    print(f"  预调度：{len(take)} 次运行 → 写入 {n} 行")
    return 0, n


def _done_path() -> str:
    from config import DATA_DIR
    d = os.path.join(DATA_DIR, "_state")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "predispatch_archive_done.txt")


def ingest_predispatch_archive(start: str, end: str, lead_band=None) -> tuple[int, int]:
    """Reports/ARCHIVE/PredispatchIS_Reports/ —— 每天一个 zip（里面是当天 48 次运行），
    滚动保留约 13 个月。这是拿到**多时效预调度历史**的路径：
    Current 只有 2 天；MMSDM 月度 DATA 目录里的 PREDISPATCHPRICE 每个时段只留最后一版，
    拿不到"提前 24 小时"的那一版。

    断点续跑：每做完一天记进 data/_state/predispatch_archive_done.txt，
    重跑同一条命令会跳过已完成的日子。日文件很大，下完即删、不进缓存。
    """
    a, b = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    lo, hi = lead_band or LEAD_KEEP_H
    avail: dict[str, str] = {}
    for u in list_zips(ARC_PREDISP):
        m = re.search(r"(\d{8})", u.split("/")[-1])
        if m:
            avail.setdefault(m.group(1), u)
    done_file = _done_path()
    done = set(open(done_file).read().split()) if os.path.exists(done_file) else set()
    days = pd.date_range(a, b, freq="D")
    print(f"  归档目录 {len(avail)} 个日文件；请求 {a.date()}..{b.date()}（{len(days)} 天），"
          f"已完成 {sum(1 for d in days if d.strftime('%Y%m%d') in done)} 天；"
          f"时效带 {lo:g}–{hi:g}h")
    total, miss = 0, []
    for i, d in enumerate(days):
        key = d.strftime("%Y%m%d")
        if key in done:
            continue
        u = avail.get(key)
        if not u:
            miss.append(str(d.date())); continue
        try:
            files = unzip_named(download(u, quiet=False, cache=False))
        except AemoError as e:
            print(f"    [{i+1}/{len(days)}] {d.date()} ✗ {str(e)[:70]}（重跑会从这天继续）")
            continue
        parts = [parse_predispatch(t, n, (lo, hi)) for n, t in files]
        n = _write_predispatch(parts, "nemweb_predispatch_archive")
        total += n
        leads = [x for x in parts if len(x)]
        lr = ""
        if leads:
            f = pd.concat(leads)
            lh = (f["interval_time"] - f["known_at"]).dt.total_seconds() / 3600
            lr = f"，时效 {lh.min():.1f}–{lh.max():.1f}h"
        print(f"    [{i+1}/{len(days)}] {d.date()} ✓ {len(files)} 次运行 → {n} 行{lr}")
        with open(done_file, "a") as fh:
            fh.write(key + "\n")
    if miss:
        print(f"  归档里没有的日期 {len(miss)} 天，例如 {miss[:4]}（只滚动保留约 13 个月）")
    print(f"  预调度归档：共写入 {total} 行")
    return 0, total


def inspect_predispatch() -> int:
    """取 Current 里最新的一份预调度，打印解析结果，不写库。
    跑大回填之前先用它确认：发布时刻读到了、时效带里有行。"""
    urls = list_zips(CUR_PREDISP)
    if not urls:
        print("Current 目录没有预调度文件"); return 1
    u = urls[-1]
    files = unzip_named(download(u, cache=False))
    print(f"\n{u.split('/')[-1]}  （{len(files)} 个文件）\n" + "=" * 64)
    for name, text in files:
        tabs = parse_mms_csv(_only_region_prices(text))
        df = tabs.get(PREDISP_TABLE)
        print(f"  {name}")
        print(f"    发布时刻（UTC）：{publish_time(text, name)}")
        if df is None:
            print("    ⚠ 没有 PREDISPATCH/REGION_PRICES 表"); continue
        print(f"    REGION_PRICES {len(df)} 行；DATETIME 不同取值 "
              f"{df['DATETIME'].nunique()}；SEQNO 不同取值 "
              f"{df['PREDISPATCHSEQNO'].nunique() if 'PREDISPATCHSEQNO' in df else '—'}")
        full = parse_predispatch(text, name, (0, 1e9))
        if len(full):
            lh = (full["interval_time"] - full["known_at"]).dt.total_seconds() / 3600
            print(f"    逐小时 {len(full)} 行，时效 {lh.min():.1f}–{lh.max():.1f}h；"
                  f"其中 ≥24h 的 {int((lh >= 24).sum())} 行")
        else:
            print("    ⚠ 解析后为空")
    return 0


# ================================================================== 采集入口
def ingest_current(dir_url: str, hours: int, source: str) -> tuple[int, int]:
    urls = list_zips(dir_url)
    if not urls:
        raise AemoError(f"目录里没找到 zip：{dir_url}")
    take = urls[-max(1, hours * 60 // DISPATCH_MIN):]
    print(f"  {dir_url}\n  共 {len(urls)} 个文件，取最近 {len(take)} 个")
    texts = []
    for i, u in enumerate(take):
        try:
            texts.append(unzip_all(download(u, cache=False)))
        except AemoError as e:
            print(f"    跳过 {u.split('/')[-1]}: {str(e)[:80]}")
        if (i + 1) % 50 == 0:
            print(f"    ... {i+1}/{len(take)}")
    return _write(texts, source)


def ingest_archive(start: str, end: str, tables=("DISPATCHPRICE",)) -> tuple[int, int]:
    """MMSDM 月度归档。默认只拉 DISPATCHPRICE——它一个月才几 MB，
    而 DISPATCHREGIONSUM 一个月上百 MB，网络不稳时基本下不完。需要需求侧
    数据再显式 --tables DISPATCHPRICE,DISPATCHREGIONSUM。"""
    a, b = pd.Period(start, "M"), pd.Period(end, "M")
    texts = []
    p = a
    while p <= b:
        for tab in tables:
            got = False
            for tpl in (MMSDM_TPL, MMSDM_TPL_ALT):
                url = tpl.format(base=MMSDM, y=p.year, m=p.month, table=tab)
                try:
                    texts.append(unzip_all(_get(url, binary=True)))
                    print(f"  {p} {tab} ✓")
                    got = True
                    break
                except AemoError:
                    continue
            if not got:
                url2 = _discover(p.year, p.month, tab)
                if url2:
                    texts.append(unzip_all(_get(url2, binary=True)))
                    print(f"  {p} {tab} ✓（目录发现 {url2.split('/')[-1]}）")
                else:
                    print(f"  {p} {tab} ✗ 未找到")
        p += 1
    return _write(texts, "mmsdm")


def ingest_daily_archive(start: str, end: str) -> tuple[int, int]:
    """Reports/ARCHIVE/DispatchIS_Reports/ —— 每天一个 zip，滚动保留约 13 个月。

    比 MMSDM 月度归档小得多，而且和 Reports/Current 是同一套格式（同一个解析器），
    所以拿它回填最近一年是最省事的路径。更早的历史才需要走 --archive。
    """
    a, b = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    avail = {u.split("/")[-1]: u for u in list_zips(ARC_DISPATCH)}
    days = pd.date_range(a, b, freq="D")
    print(f"  归档目录共 {len(avail)} 个日文件，请求区间 {a.date()} .. {b.date()}（{len(days)} 天）")
    texts, miss, fail = [], [], []
    nt = nf = 0
    for i, d in enumerate(days):
        name = f"PUBLIC_DISPATCHIS_{d:%Y%m%d}.zip"
        u = avail.get(name)
        if not u:
            miss.append(str(d.date())); continue
        try:
            texts.append(unzip_all(download(u, quiet=False)))
            print(f"    [{i+1}/{len(days)}] {name} ✓")
        except AemoError as e:
            fail.append(str(d.date()))
            print(f"    [{i+1}/{len(days)}] {name} ✗ {str(e)[:70]}")
        # 分批落盘：181 天的解压文本全堆内存会吃掉几个 G，而且中途挂掉就白跑
        if len(texts) >= 5:
            a_, b_ = _write(texts, "nemweb_archive"); nt += a_; nf += b_; texts = []
    if texts:
        a_, b_ = _write(texts, "nemweb_archive"); nt += a_; nf += b_
    if miss:
        print(f"  归档里没有的日期 {len(miss)} 天，例如 {miss[:5]}"
              f"（归档只滚动保留约 13 个月，更早的要用 --archive）")
    if fail:
        print(f"  下载失败 {len(fail)} 天：{fail[:8]}。已下载的分片留在缓存里，"
              f"直接重跑同一条命令会断点续传，不会重下。")
    return nt, nf


def _discover(y: int, m: int, table: str) -> str | None:
    """模板对不上时，去目录页找**任何**含该表名的 zip。

    目录页里的 href 可能带未编码的 '#'（AEMO 的文件名里真的有），直接拿去请求会被
    截断成锚点，所以这里在拼 URL 前把路径段里的 '#' 编码掉。
    """
    base = MMSDM_DIR.format(base=MMSDM, y=y, m=m)
    try:
        html = _get(base)
    except AemoError:
        return None
    # 只认整段表名，避免 DISPATCHPRICE 误配到 PREDISPATCHPRICE 之类
    pat = re.compile(rf'href="([^"]*(?:#|_){re.escape(table)}(?:#|_)[^"]*\.zip)"', re.I)
    mm = pat.search(html)
    if not mm:
        pat = re.compile(rf'href="([^"]*{re.escape(table)}[^"]*\.zip)"', re.I)
        mm = pat.search(html)
    if not mm:
        return None
    href = mm.group(1).replace("&amp;", "&")
    if not href.startswith("http"):
        href = (base + href.lstrip("/")) if "/" not in href else NEMWEB + "/" + href.lstrip("/")
    scheme, rest = href.split("://", 1)
    host, _, path = rest.partition("/")
    return f"{scheme}://{host}/{path.replace('#', '%23')}"


def _write(texts: list[str], source: str) -> tuple[int, int]:
    if not texts:
        return 0, 0
    texts = [t for t in texts if t]
    tabs = extract(texts)
    print("  解析到的表：", ", ".join(f"{a}/{b}({len(d)})" for (a, b), d in sorted(tabs.items()))[:400])
    t, f = rows_from_tables(tabs, source)
    rid = store.new_run_id()
    nt = store.append("truth", t, rid) if len(t) else 0
    nf = store.append("forecast", f, rid) if len(f) else 0
    return nt, nf


def inspect_table(table: str, start: str, end: str) -> int:
    """把归档文件解析出来看一眼，不写库。

    为什么需要：预调度接了两次都写 0 行，两次都是我**猜**列的语义猜错了。
    与其再猜第三次，不如把文件里到底有什么打印出来。
    """
    p0 = pd.Period(start, "M")
    txt = None
    for tpl in (MMSDM_TPL, MMSDM_TPL_ALT):
        try:
            txt = unzip_all(_get(tpl.format(base=MMSDM, y=p0.year, m=p0.month,
                                            table=table), binary=True))
            break
        except AemoError:
            continue
    if txt is None:
        u = _discover(p0.year, p0.month, table)
        if not u:
            print(f"找不到 {table} {p0}"); return 1
        txt = unzip_all(download(u))

    tabs = parse_mms_csv(txt)
    print(f"\n{table} · {p0}\n" + "=" * 64)
    for (rep, sub), df in sorted(tabs.items()):
        print(f"\n[{rep}/{sub}]  {len(df)} 行 × {len(df.columns)} 列")
        print("  列：", ", ".join(df.columns[:20]))
        for c in ("PREDISPATCHSEQNO", "RUNNO", "REGIONID", "PERIODID",
                  "DATETIME", "SETTLEMENTDATE", "LASTCHANGED", "INTERVENTION"):
            if c in df.columns:
                u_ = df[c].astype(str).nunique()
                print(f"  {c:20s} 不同取值 {u_:>8d}   例：{df[c].astype(str).iloc[0]!r}")
        # 关键诊断：一次运行覆盖几个目标时刻？只有 1 个，就说明这份归档
        # 每个区间只留了一版预测，拿不到"提前 24 小时"的那一版。
        if "PREDISPATCHSEQNO" in df.columns and "DATETIME" in df.columns:
            g = df.groupby("PREDISPATCHSEQNO")["DATETIME"].nunique()
            print(f"  每个 SEQNO 覆盖的不同 DATETIME：中位数 {g.median():.0f}，"
                  f"最大 {g.max():.0f}，共 {len(g)} 个 SEQNO")
            if g.median() <= 1:
                print("  ⚠ 每次运行只对应一个目标时刻 —— 这份归档每个区间只保留了一版，")
                print("    没有多时效的预测轨迹，做不了 D+1 的同时效对比。")
    return 0


# ================================================================== 探测
def probe() -> int:
    ok = True
    for name, url in (("Dispatch Current", CUR_DISPATCH),
                      ("Predispatch Current", CUR_PREDISP),
                      ("P5MIN Current", CUR_P5MIN)):
        try:
            z = list_zips(url)
            print(f"  [OK]   {name:20s} {len(z)} 个 zip，最新 {z[-1].split('/')[-1] if z else '—'}")
        except Exception as e:
            ok = False
            print(f"  [FAIL] {name:20s} {str(e)[:110]}")
    for name, url in (("Dispatch ARCHIVE", ARC_DISPATCH),
                      ("Predispatch ARCHIVE", ARC_PREDISP)):
        try:
            z = list_zips(url)
            print(f"  [OK]   {name:20s} {len(z)} 天，{z[0].split('/')[-1]} .. {z[-1].split('/')[-1]}")
        except Exception as e:
            print(f"  [WARN] {name:20s} {str(e)[:110]}")

    p = pd.Period(pd.Timestamp.now("UTC").tz_localize(None), "M") - 2
    hit = None
    for tpl in (MMSDM_TPL, MMSDM_TPL_ALT):
        url = tpl.format(base=MMSDM, y=p.year, m=p.month, table="DISPATCHPRICE")
        try:
            _get(url, binary=True)
            hit = url
            break
        except Exception:
            continue
    if hit:
        print(f"  [OK]   MMSDM 模板           {hit.split('/')[-1]}")
    else:
        print("  [WARN] MMSDM 两套模板都对不上，试目录发现…")
        u2 = _discover(p.year, p.month, "DISPATCHPRICE")
        print(f"  {'[OK]  ' if u2 else '[FAIL]'} 目录发现            {u2 or '未找到'}")
        # MMSDM 只是"更早历史"的通道；ARCHIVE 能用就不算致命
        if not u2:
            print("  → MMSDM 不可用不影响 --daily-archive（最近约 13 个月）")
    return 0 if ok else 1


# ================================================================== 离线自检
SAMPLE = """C,NEMP.WORLD,DISPATCHIS,AEMO,PUBLIC,2026/08/28,00:00:05,,,
I,DISPATCH,PRICE,5,SETTLEMENTDATE,RUNNO,REGIONID,DISPATCHINTERVAL,INTERVENTION,RRP,EEP,APCFLAG
{PRICE}
I,DISPATCH,REGIONSUM,7,SETTLEMENTDATE,RUNNO,REGIONID,DISPATCHINTERVAL,INTERVENTION,TOTALDEMAND,SS_SOLAR_UIGF,SS_WIND_UIGF
{RSUM}
C,END OF REPORT,9
"""


def _sample() -> str:
    """造两小时、两个区的 5 分钟数据；其中掺入 INTERVENTION=1 的行必须被丢掉。"""
    pr, rs = [], []
    base = pd.Timestamp("2026-08-28 10:05:00")           # NEM 时间，区间结束
    for i in range(24):                                   # 24 × 5min = 2 小时
        sd = (base + pd.Timedelta(minutes=5 * i)).strftime("%Y/%m/%d %H:%M:%S")
        for j, reg in enumerate(("NSW1", "SA1")):
            rrp = 50 + i + 100 * j
            pr.append(f'D,DISPATCH,PRICE,5,"{sd}",1,{reg},{i+1},0,{rrp},0,0')
            rs.append(f'D,DISPATCH,REGIONSUM,7,"{sd}",1,{reg},{i+1},0,{7000+10*i},{300+j},{900-j}')
        # 干预运行：价格离谱，必须被过滤掉
        pr.append(f'D,DISPATCH,PRICE,5,"{sd}",1,NSW1,{i+1},1,99999,0,0')
    return SAMPLE.format(PRICE="\n".join(pr), RSUM="\n".join(rs))


def selftest() -> int:
    print("AEMO 离线自检 —— 不联网\n" + "=" * 52)
    ok = True

    def chk(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' — ' + extra) if extra else ''}")

    tabs = parse_mms_csv(_sample())
    chk("解析出 DISPATCH/PRICE 与 REGIONSUM",
        ("DISPATCH", "PRICE") in tabs and ("DISPATCH", "REGIONSUM") in tabs,
        ", ".join(f"{a}/{b}" for a, b in tabs))
    price = tabs[("DISPATCH", "PRICE")]
    chk("D 行数正确（含 24 条干预行）", len(price) == 24 * 3, f"{len(price)} 行")

    t, f = rows_from_tables(tabs, "selftest")
    chk("干预行被过滤", (pd.to_numeric(t.value) < 9999).all())

    rrp = t[(t.variable == "rrp") & (t.site_id == "NSW1")].sort_values("interval_time")
    chk("聚合成 2 个整点", len(rrp) == 2, f"{len(rrp)} 行")

    # NEM 10:05 结束 = 10:00 起点 = UTC 00:00
    first = pd.Timestamp(rrp.interval_time.iloc[0])
    chk("时区与区间端点：NEM 10:05 → UTC 00:00 那一格",
        first == pd.Timestamp("2026-08-28 00:00:00"), str(first))

    chk("第一小时 RRP 均值 = 50..61 的平均 55.5",
        abs(float(rrp.value.iloc[0]) - 55.5) < 1e-6, str(rrp.value.iloc[0]))
    mx = t[(t.variable == "rrp_max") & (t.site_id == "NSW1")].sort_values("interval_time")
    chk("第一小时 RRP 峰值 = 61", abs(float(mx.value.iloc[0]) - 61) < 1e-6, str(mx.value.iloc[0]))

    # 残缺小时必须被丢掉：只留 3 个区间的那一小时不该产出任何行。
    # 半小时的数据求个均值、看起来一切正常，是最难发现的一类脏数据。
    part = price[price.SETTLEMENTDATE.str.contains("2026/08/28 10:")].head(3).copy()
    tp = to_hourly_truth(part, "RRP", {"rrp": "mean"}, "selftest")
    chk("不足 10 个区间的小时被丢弃", len(tp) == 0, f"{len(tp)} 行")
    # 重复行不该让残缺小时"凑满"
    dup = pd.concat([part] * 4, ignore_index=True)
    td = to_hourly_truth(dup, "RRP", {"rrp": "mean"}, "selftest")
    chk("重复行不能凑满一小时", len(td) == 0, f"{len(td)} 行")

    chk("known_at 晚于区间起点（不泄漏）",
        (pd.to_datetime(t.known_at) > pd.to_datetime(t.interval_time)).all())
    chk("known_at = 整点 + 1h10min",
        (pd.to_datetime(t.known_at) - pd.to_datetime(t.interval_time)
         == pd.Timedelta(minutes=70)).all())

    chk("SA1 价格比 NSW1 高 100",
        abs(float(t[(t.variable == "rrp") & (t.site_id == "SA1")].value.iloc[0]) - 155.5) < 1e-6)
    chk("需求与半调度出力都落表",
        set(t.variable) >= {"rrp", "rrp_max", "demand", "solar_uigf", "wind_uigf"},
        ", ".join(sorted(set(t.variable))))

    # 嵌套 zip
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as z:
        z.writestr("PUBLIC_DISPATCHIS_1.CSV", _sample())
    outer = io.BytesIO()
    with zipfile.ZipFile(outer, "w") as z:
        z.writestr("inner.zip", inner.getvalue())
    chk("嵌套 zip 能展开", "DISPATCH,PRICE" in unzip_all(outer.getvalue()))

    print("=" * 52)
    print("自检通过 ✓" if ok else "自检失败 ✗")
    return 0 if ok else 1


# ================================================================== CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="AEMO NEMWeb → L1 时点存储")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--current", action="store_true", help="Reports/Current 增量（2 天窗口）")
    ap.add_argument("--predispatch", action="store_true",
                    help="官方预测（Current，约 2 天）→ forecast 表")
    ap.add_argument("--predispatch-archive", action="store_true",
                    help="官方预测历史（日归档，约 13 个月）→ forecast 表；--start/--end 用 YYYY-MM-DD")
    ap.add_argument("--inspect-predispatch", action="store_true",
                    help="解析最新一份预调度并打印，不写库")
    ap.add_argument("--lead-lo", type=float, default=None, help="预调度只留 ≥ 这个时效（小时）")
    ap.add_argument("--lead-hi", type=float, default=None, help="预调度只留 ≤ 这个时效（小时）")
    ap.add_argument("--archive", action="store_true", help="MMSDM 月度归档回填（更早历史）")
    ap.add_argument("--daily-archive", action="store_true",
                    help="Reports/ARCHIVE 日归档回填（最近约 13 个月，推荐先用这个）")
    ap.add_argument("--hours", type=int, default=6)
    ap.add_argument("--start", default="2025-01", help="--archive 用 YYYY-MM；--daily-archive 用 YYYY-MM-DD")
    ap.add_argument("--end", default="2026-06", help="同上")
    ap.add_argument("--tables", default="DISPATCHPRICE",
                    help="--archive 要拉的 MMSDM 表，逗号分隔")
    ap.add_argument("--clear-cache", action="store_true", help="清掉 HTTP 断点缓存后退出")
    ap.add_argument("--inspect", metavar="TABLE",
                    help="只解析、不写库：打印某张 MMSDM 表的真实结构（列名/行数/分组）")
    a = ap.parse_args(argv)

    if a.selftest:
        return selftest()
    if a.probe:
        return probe()
    if a.inspect:
        return inspect_table(a.inspect, a.start, a.end)
    if a.inspect_predispatch:
        return inspect_predispatch()
    band = None
    if a.lead_lo is not None or a.lead_hi is not None:
        band = (a.lead_lo if a.lead_lo is not None else LEAD_KEEP_H[0],
                a.lead_hi if a.lead_hi is not None else LEAD_KEEP_H[1])
    if a.clear_cache:
        import shutil
        shutil.rmtree(CACHE_DIR, ignore_errors=True)
        print(f"已清空 {CACHE_DIR}")
        return 0

    started, rid = store.utcnow(), store.new_run_id()
    try:
        if a.current:
            nt, nf = ingest_current(CUR_DISPATCH, a.hours, "nemweb_current")
        elif a.predispatch:
            nt, nf = ingest_predispatch_current(a.hours, band)
        elif a.predispatch_archive:
            nt, nf = ingest_predispatch_archive(a.start, a.end, band)
        elif a.daily_archive:
            nt, nf = ingest_daily_archive(a.start, a.end)
        elif a.archive:
            nt, nf = ingest_archive(a.start, a.end,
                                    tuple(x.strip() for x in a.tables.split(",") if x.strip()))
        else:
            ap.print_help(); return 2
        store.log_run(rid, started, "aemo", "nemweb", CALLS[0], nt + nf, nt + nf, "ok")
        print(f"\n完成：HTTP {CALLS[0]} 次，truth 新写 {nt} 行，forecast 新写 {nf} 行")
        return 0
    except Exception as e:
        store.log_run(rid, started, "aemo", "nemweb", CALLS[0], 0, 0, "error", f"{type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

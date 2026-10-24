"""
L2 特征层 —— 唯一入口 get_features(as_of, horizons)。
====================================================
骨架第 L2 条：训练与推断共用同一个函数，**只取 known_at ≤ as_of**。
这里把那条边界做成三重保险：
  ① SQL 里 WHERE known_at <= as_of
  ② 同一 (站点, 模型, 变量, 目标时刻) 只保留当时最新的一版（QUALIFY）
  ③ 返回前 assert_no_leak() 再验一遍，越界就抛异常而不是悄悄返回

默认 strict=True，只用 known_at_method ∈ {exact, request} 的行。
previous-runs 回填出来的 approx 行是保守上界、够用来画时效-技能曲线，
但不该进训练集，所以默认排除。
"""
from __future__ import annotations
from typing import Iterable, Sequence

import pandas as pd

from config import NEM_OFFSET, STRICT_METHODS


def to_nem(ts):
    """UTC -> NEM 时间（UTC+10，固定，无夏令时）。"""
    return pd.to_datetime(ts) + NEM_OFFSET


def from_nem(ts):
    return pd.to_datetime(ts) - NEM_OFFSET


def assert_no_leak(df: pd.DataFrame, as_of) -> None:
    """任何一条 known_at > as_of 都是泄漏。宁可崩，不可静默。"""
    if df is None or len(df) == 0 or "known_at" not in df.columns:
        return
    as_of = pd.Timestamp(as_of)
    bad = pd.to_datetime(df["known_at"]) > as_of
    n = int(bad.sum())
    assert n == 0, (
        f"as-of 泄漏：{n} 行的 known_at > as_of={as_of}；"
        f"最晚的一条是 {pd.to_datetime(df.loc[bad, 'known_at']).max()}"
    )


def _in_list(vals) -> str:
    return ", ".join("'" + str(v).replace("'", "''") + "'" for v in vals)


def get_features(con, as_of, horizons: Iterable[int],
                 sites: Sequence[str] | None = None,
                 models: Sequence[str] | None = None,
                 variables: Sequence[str] | None = None,
                 strict: bool = True,
                 wide: bool = True) -> pd.DataFrame:
    """站在 as_of 这一刻，取对 as_of + h（h ∈ horizons，单位小时）的全部可见预报。

    as_of / 返回的时间列都是 **UTC**。要 NEM 时间用 to_nem() 转。
    wide=True 时返回宽表，列名 f_{model}_{variable}；wide=False 返回长表。
    """
    as_of = pd.Timestamp(as_of)
    horizons = sorted({int(h) for h in horizons})
    targets = [as_of + pd.Timedelta(hours=h) for h in horizons]

    where = ["known_at <= ?", f"interval_time IN ({', '.join(['?'] * len(targets))})"]
    params: list = [as_of, *targets]
    if strict:
        where.append(f"known_at_method IN ({_in_list(STRICT_METHODS)})")
    if sites:
        where.append(f"site_id IN ({_in_list(sites)})")
    if models:
        where.append(f"model IN ({_in_list(models)})")
    if variables:
        where.append(f"variable IN ({_in_list(variables)})")

    sql = f"""
        SELECT site_id, model, variable, interval_time, known_at, value,
               known_at_method,
               date_diff('hour', ?, interval_time) AS horizon_h,
               date_diff('hour', known_at, interval_time) AS lead_h
        FROM forecast
        WHERE {' AND '.join(where)}
        QUALIFY row_number() OVER (
            PARTITION BY site_id, model, variable, interval_time
            ORDER BY known_at DESC, ingested_at DESC
        ) = 1
        ORDER BY site_id, interval_time, model, variable
    """
    df = con.execute(sql, [as_of, *params]).df()

    assert_no_leak(df, as_of)                       # ③ 最后一道闸
    if not wide or df.empty:
        return df

    w = df.pivot_table(index=["site_id", "interval_time", "horizon_h"],
                       columns=["model", "variable"], values="value", aggfunc="last")
    w.columns = [f"f_{m}_{v}" for m, v in w.columns]
    meta = (df.groupby(["site_id", "interval_time", "horizon_h"])
              .agg(known_at=("known_at", "max"), lead_h=("lead_h", "min"),
                   n_sources=("value", "size")))
    out = meta.join(w).reset_index()
    assert_no_leak(out, as_of)
    return out


def get_labels(con, as_of, start=None, end=None,
               sites: Sequence[str] | None = None,
               variables: Sequence[str] | None = None,
               final_only: bool = False) -> pd.DataFrame:
    """取在 as_of 这一刻**已经落地**的真值。

    ERA5 有约 5 天延迟，所以训练时能看到的标签总是滞后于特征——
    这一点必须体现在回测里，否则最近几天的评分是假的。
    同一时刻若 final 与 provisional 都在，优先 final。
    """
    as_of = pd.Timestamp(as_of)
    where = ["known_at <= ?"]
    params: list = [as_of]
    if start is not None:
        where.append("interval_time >= ?"); params.append(pd.Timestamp(start))
    if end is not None:
        where.append("interval_time <= ?"); params.append(pd.Timestamp(end))
    if final_only:
        where.append("quality = 'final'")
    if sites:
        where.append(f"site_id IN ({_in_list(sites)})")
    if variables:
        where.append(f"variable IN ({_in_list(variables)})")

    df = con.execute(f"""
        SELECT site_id, variable, interval_time, known_at, value, quality, source
        FROM truth
        WHERE {' AND '.join(where)}
        QUALIFY row_number() OVER (
            PARTITION BY site_id, variable, interval_time
            ORDER BY (quality = 'final') DESC, known_at DESC
        ) = 1
        ORDER BY site_id, interval_time, variable
    """, params).df()
    assert_no_leak(df, as_of)
    return df


def make_training_frame(con, as_of, horizons: Iterable[int],
                        lookback_days: int = 365, **kw) -> pd.DataFrame:
    """按 as_of 拼一份训练集：对过去每个整点，取"当时可见"的预报 + 已落地的真值。

    这是回测正确性的关键——训练集必须逐时刻重建，不能一次性拉平。
    数据量大时按天分批调用即可（每批共用同一个 as_of）。
    """
    as_of = pd.Timestamp(as_of)
    feats = get_features(con, as_of, horizons, **kw)
    if feats.empty:
        return feats
    labels = get_labels(con, as_of,
                        start=feats["interval_time"].min(),
                        end=feats["interval_time"].max(),
                        sites=kw.get("sites"), variables=kw.get("variables"))
    if labels.empty:
        return feats
    lab = labels.pivot_table(index=["site_id", "interval_time"],
                             columns="variable", values="value", aggfunc="last")
    lab.columns = [f"y_{c}" for c in lab.columns]
    out = feats.merge(lab.reset_index(), on=["site_id", "interval_time"], how="left")
    assert_no_leak(out, as_of)
    return out

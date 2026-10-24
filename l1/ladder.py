"""
基准阶梯 —— 天气到底给交易带来多少**增量**。
=============================================
这是整套系统的重写，起因是 Lago & Weron 的 EPF 综述指出的一条铁律：

    电力价格预测里，**近期价格本身**是主导预测因子；外生变量（负荷预报、
    新能源出力预报）是它之上的增量。任何新特征都必须先打赢
    naive（P d−7,h）和自回归基准，否则报出来的相关系数毫无意义。

之前的做法——"净负荷缺口 vs RRP 的相关"——问的是"天气和电价相不相关"。
那个问题的答案永远是"相关"，因为天气驱动供给、供给驱动价格。
但它**不是交易问题**。交易问题是：

    我已经知道最近的价格，也知道 AEMO 公开发布的预调度，
    再加上天气，还能多赚多少？

所以改成四层阶梯，每层只比上一层多一组特征：

    B0  naive        P(d−7,h)                     ——  基线，rMAE 的分母
    B1  自回归       + 决策时刻已结算的最新价 / 最近两天同一钟点 / 一周前 / 24h 均价
    B2  官方         + AEMO 预调度（同时效）
    B3  天气         + GHI / 风速 / 气温预报

**天气的价值 = B3 相对 B2 的改善**。B2 拿不到数据时，退而求其次看 B3 − B1，
但必须标明"没有扣掉官方预测"，因为那部分市场已经定价。

评估口径也按综述来：
  · rMAE（相对 naive），不是 MAPE —— 电价会穿零，MAPE 没有定义
  · 同时报 MAE 和 RMSE —— 只报一个会掩盖尖峰上的表现
  · Giacomini-White 检验，不是 Diebold-Mariano —— 滚动重拟合下
    校准窗口在变，DM 的前提（预测方法固定）不成立
  · 统计指标之外必须有**经济评估**：综述原话是统计误差
    "may not necessarily reflect the economic value of reducing prediction errors"

    python ladder.py --region QLD1 --hours 10-16
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import realdata as R  # noqa: E402
import store  # noqa: E402
from config import SITES  # noqa: E402

# 决策时刻：NEM 日 D 的全部 24 小时，在 D−1 天的固定时刻（NEM）一次性定计划。
# 所有特征必须在那一刻已经可见。定义在 realdata.decision_time，这里只引用。
#
# 为什么不是"每个小时滚动提前 24 小时"：AEMO 预调度最远只看到下一个交易日结束
# （次日 04:00），而且大约 12:30 之后才开始覆盖下一天。所以目标落在 04:00–12:30
# 的小时，不存在提前 ≥24h 发出的预调度；加了官方预测这一层，这半天会被整段丢掉。
# 固定时刻口径也正是电池运营商排次日计划的方式。2026-10 改。
DECIDE_AT_NEM = R.DECIDE_AT_NEM

# 一个小时的结算价（12 个 5 分钟区间的均值）要等这一小时结束、再过发布延迟
# （aemo.py 保守取 10 分钟）才可见：起点为 h 的那一小时，h + 1h10m 才知道。
PRICE_PUBLISH = pd.Timedelta(hours=1, minutes=10)

TIERS = ["B0 naive", "B1 自回归", "B2 +官方", "B3 +天气"]
# p_last  决策时刻已结算的最新一小时（同一天的 24 个小时共用）
# p_d1    最近一个"同一钟点"且已结算的价格（钟点早于决策时刻用昨天，否则用前天）
# p_d2    再往前一天的同一钟点
# p_w1    一周前同一钟点
# p_mean24  截至 p_last 那一小时的 24 小时均价
AR_COLS = ["p_last", "p_d1", "p_d2", "p_w1", "p_mean24"]
WX_COLS = ["ghi", "wind", "temp"]
OFF_COLS = ["aemo"]

# NEM 市场硬边界。上限逐年指数化（2025-26 约 $17,500/MWh），这里取宽松常数
# 当保险丝；真正防外推的是 _fit_predict 里的训练域夹取。
PRICE_CAP = 20000.0
PRICE_FLOOR = -1000.0
NEM_OFFSET_H = 10          # NEM 时间 = UTC+10，全年固定，无夏令时


def official(con, region: str, decide_at: str = DECIDE_AT_NEM) -> pd.DataFrame:
    """AEMO 预调度：每个目标小时取**决策时刻之前**发出的最新一版。

    目标落在 NEM 日 D，只要 known_at ≤ D−1 天 decide_at 的版本，其中取最新。
    按 known_at 排序，不在浮点时效上做 QUALIFY 排序——那会触发 DuckDB 1.5.x 的
    优化器内部错误（Mac 上实测复现）。
    """
    tau = R.decision_sql("interval_time", decide_at)
    return con.execute(f"""
        WITH f AS (SELECT interval_time AS t, value, known_at, {tau} AS tau
                   FROM forecast
                   WHERE site_id = ? AND variable = 'rrp' AND model = 'aemo_predispatch')
        SELECT t, value, date_diff('second', known_at, t) / 3600.0 AS lh
        FROM f
        WHERE known_at <= tau AND known_at >= tau - INTERVAL 4 DAY
        QUALIFY row_number() OVER (PARTITION BY t ORDER BY known_at DESC) = 1
    """, [region]).df()


def ar_features(s: pd.Series, t, decide_at: str = DECIDE_AT_NEM) -> pd.DataFrame:
    """价格自回归特征，**每一个都在决策时刻已经结算公布**。

    s：逐小时 RRP，索引是小时起点（UTC）。t：目标小时起点（UTC）。
    全部按时间戳取，不按下标平移——样本有洞时下标平移会静默错位。
    """
    t = pd.Series(pd.to_datetime(t)).reset_index(drop=True)
    tau = pd.Series(pd.to_datetime(R.decision_time(t, decide_at)))
    last_h = (tau - PRICE_PUBLISH).dt.floor("h")      # 决策时刻已结算的最新一小时
    y1 = t - pd.Timedelta(hours=24)
    d1 = y1.where(y1 + PRICE_PUBLISH <= tau, t - pd.Timedelta(hours=48))
    d2 = d1 - pd.Timedelta(hours=24)
    s = s.sort_index()
    roll = s.rolling("24h").mean()
    return pd.DataFrame({
        "p_last": s.reindex(last_h).to_numpy(),
        "p_d1": s.reindex(d1).to_numpy(),
        "p_d2": s.reindex(d2).to_numpy(),
        "p_w1": s.reindex(t - pd.Timedelta(hours=168)).to_numpy(),
        "p_mean24": roll.reindex(last_h).to_numpy(),
    })


WX_MODELS = ["ecmwf_ifs025"]     # 天气只用 ECMWF：历史回填只有它齐全，实时也只抓它，两边口径一致


def build(con, region: str, site_id: str, hours=None,
          decide_at: str = DECIDE_AT_NEM, live: bool = False) -> pd.DataFrame:
    """装配阶梯用的特征表。目标落在 NEM 日 D，**每一列在 D−1 天 decide_at 都已可见**：

      价格滞后   决策时刻前已结算的小时（结算价要过 1h10m 才发布）
      天气预报   known_at ≤ 决策时刻的版本里最新的一版
      官方预测   known_at ≤ 决策时刻的预调度里最新的一版
      naive     t − 168h
    """
    # live=True：实时出信号用。目标小时（明天）还没有气象真值，不能要求真值齐全；
    # 其余口径和回测完全一样——同一个决策时刻、同一套 as-of 规则、同一组特征。
    p = R.load_backtest(con, site_id, decide_at=decide_at, models=WX_MODELS,
                        require_truth=not live)
    if not live:
        p = R.fill_persist_head(p)
    q = p.preds["qra"]
    d = pd.DataFrame({"t": pd.to_datetime(p.t), "hod": p.hod,
                      "ghi_raw": q["ghi"], "wind_raw": q["wind100"],
                      "temp_raw": q["temp"]})

    mk = con.execute("""
        SELECT interval_time AS t, variable, value FROM truth
        WHERE site_id = ? AND variable IN ('rrp','rrp_max')
        QUALIFY row_number() OVER (PARTITION BY variable, interval_time
                                   ORDER BY (quality='final') DESC, known_at DESC) = 1
    """, [region]).df()
    if mk.empty:
        raise ValueError(f"{region}: 没有 rrp。先跑 aemo.py --archive。")
    mw = mk.pivot_table(index="t", columns="variable", values="value",
                        aggfunc="last").reset_index()
    d = d.merge(mw, on="t", how="left").sort_values("t").reset_index(drop=True)

    # 价格序列直接从 truth 取全量，不受气象面板覆盖范围的限制
    s = mw.set_index("t")["rrp"].sort_index()
    feats = ar_features(s, d["t"], decide_at)
    for c in feats.columns:
        d[c] = feats[c].to_numpy()

    # ---- 官方预测：决策时刻之前发出的最新预调度 ----
    fc = official(con, region, decide_at)
    d["aemo"] = (d.merge(fc[["t", "value"]].rename(columns={"value": "_a"}),
                         on="t", how="left")["_a"].to_numpy()
                 if len(fc) else np.nan)

    # ---- 天气：滚动分位，as-of 干净 ----
    import l3_signal as SG
    for k in WX_COLS:
        d[k] = SG.trailing_pct(d[f"{k}_raw"].to_numpy(float), d["hod"].to_numpy(), days=60)

    # naive 基线：一周前同一时刻
    d["naive"] = s.reindex(d["t"] - pd.Timedelta(hours=168)).to_numpy()

    if hours:
        d = d[(d["hod"] >= hours[0]) & (d["hod"] < hours[1])]
    return d.reset_index(drop=True)


def vst(y: np.ndarray, scale: float) -> np.ndarray:
    """方差稳定变换。综述 4.2 节：电价重尾且穿零，直接在原始尺度上做
    最小二乘，几个尖峰就能主导整条回归线。asinh 在零附近近似线性、
    尾部近似对数，且对负价有定义（log 没有）。"""
    return np.arcsinh(y / scale)


def ivst(z: np.ndarray, scale: float) -> np.ndarray:
    return np.sinh(z) * scale


def _fit_predict(tr: pd.DataFrame, te: pd.DataFrame, cols: list[str],
                 ridge: float = 1e-3) -> np.ndarray:
    """标准化 + 岭回归 + asinh 变换后拟合，预测再变回原尺度。

    岭系数很小（1e-3，作用在标准化后的设计阵上），目的不是收缩而是
    数值兜底：价格滞后彼此高度相关，纯 OLS 在某些窗口会病态。
    """
    ytr = tr["rrp"].to_numpy(float)
    # 变换尺度取训练集绝对价格的中位数，稳健且只用训练期信息
    scale = float(np.median(np.abs(ytr)))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    X = tr[cols].to_numpy(float)
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    Xs = (X - mu) / sd
    A = np.column_stack([Xs, np.ones(len(Xs))])
    z = vst(ytr, scale)
    P = A.T @ A + ridge * len(A) * np.eye(A.shape[1])
    P[-1, -1] -= ridge * len(A)          # 截距不惩罚
    beta = np.linalg.solve(P, A.T @ z)
    Xe = (te[cols].to_numpy(float) - mu) / sd
    # 输入端也不外推：每个特征夹回它在训练期见过的范围。
    #
    # 2026-10 真实数据上踩到：SA1 的 07:00 模型训练期在冬天，那个钟点天还黑，
    # GHI 分位几乎不动；九月中日出提前到 07:00 之前，GHI 突然出现训练期没见过的值，
    # 标准化后是个极大的数，预测一路顶到输出端的上限——也就是训练期里那一次
    # $16,506 的尖峰。结果连续好几天 07:00 都预测 $16,506，实际是 $-39 到 $59，
    # B3 的 rMAE 从 0.65 被拖到 0.97。输出端早就夹了，输入端漏了。
    Xe = np.clip(Xe, Xs.min(0), Xs.max(0))
    ze = Xe @ beta[:-1] + beta[-1]
    # 不外推：把变换空间里的预测夹回训练期**见过的范围**，一点都不放宽。
    #
    # 这里踩过一次：原本两端各放宽 20% 的 z 跨度。但 sinh 是指数的，
    # 训练窗里只要有一次触顶尖峰（NSW1 2024-11 的 $10,550），z 跨度就到 8.5，
    # 20% 的 pad 等于把上限乘 e^1.7 ≈ 5.5 倍——实测吐出 $46,636 的预测，
    # 而当时实际价是 $94。最坏 1% 的行吃掉了 66% 的总误差，B1 的 rMAE 因此到 1.49。
    # 现在的规矩：**没见过的价位，一个字都不说**。
    lo, hi = float(np.min(z)), float(np.max(z))
    p_ = ivst(np.clip(ze, lo, hi), scale)
    # 再加一道市场硬边界兜底。NEM 的市场价格上限逐年指数化（2025-26 约 $17,500），
    # 这里取一个宽松常数，只当保险丝用——真正起作用的是上面的训练域夹取。
    return np.clip(p_, PRICE_FLOOR, PRICE_CAP)


def _fit_predict_ph(tr: pd.DataFrame, te: pd.DataFrame, cols: list[str],
                    min_rows: int = 60) -> np.ndarray:
    """**每个 hour-of-day 各拟合一个模型**。综述里 day-ahead EPF 的标准做法。

    为什么必须这样：一个跨 24 小时的共享模型，日内形状全靠特征去撑，
    而 MAE 只奖励"贴近水平"，于是回归把日内曲线压平——NSW1 实测振幅比
    只有 0.09，也就是预测的日内起伏只有实际的 9%。水平贴得很准，
    但 argmax/argmin 挑出来的充放电时刻基本是噪声，套利收益就输给了 naive。

    分小时之后，日内形状由 24 个各自的截距天然给出，不需要模型去"学"；
    而且太阳能信号不会再被涂抹到夜间——GHI 在夜里恒为 0，
    它的滚动分位在那些小时是退化的，共享模型里等于往夜间注入噪声，
    偏偏夜间正是充电时刻所在。
    """
    out = np.full(len(te), np.nan)
    tr_h, te_h = tr["hod"].to_numpy(), te["hod"].to_numpy()
    for h in np.unique(te_h):
        m_te = te_h == h
        m_tr = tr_h == h
        src = tr[m_tr] if m_tr.sum() >= min_rows else tr   # 样本不够就退回全局
        out[m_te] = _fit_predict(src, te[m_te], cols)
    return out


NESTED_NAME = "B3 选特征"
NESTED_VAL_DAYS = 28        # 训练窗最后 28 天当验证段
NESTED_TOL = 0.005          # 验证 MAE 在最好的 0.5% 以内就取更少的特征


def _subsets(cols: list[str]) -> list[list[str]]:
    out = [[]]
    for c in cols:
        out += [x + [c] for x in out]
    return sorted(out, key=len)


def select_features(tr: pd.DataFrame, base: list[str], fp, embargo_days: int = 3,
                    val_days: int = NESTED_VAL_DAYS, tol: float = NESTED_TOL) -> list[str]:
    """**只用训练窗**挑天气特征：训练窗前段拟合、最后 val_days 天验证，
    在 base 之上枚举天气变量的全部子集（含空集），取验证 MAE 最小的；
    几个子集差距在 tol 以内时取变量更少的那个（多一个变量要用看得见的改善换）。

    这是在修"特征取舍"那张表的毛病：那张表是在测试集上比出来的，
    拿它去决定留哪几个变量，等于偷看了答案。这里每个滚动窗口各自挑一次，
    挑的时候测试段完全不可见。
    """
    t = tr["t"]
    cut = t.max() - pd.Timedelta(days=val_days)
    fit_m = (t < cut - pd.Timedelta(days=embargo_days)).to_numpy()
    val_m = (t >= cut).to_numpy()
    if fit_m.sum() < 200 or val_m.sum() < 48:
        return list(WX_COLS)
    yv = tr["rrp"].to_numpy(float)[val_m]
    scores = []
    for sub in _subsets(list(WX_COLS)):
        pv = fp(tr[fit_m], tr[val_m], base + sub)
        scores.append((float(np.mean(np.abs(pv - yv))), sub))
    best = min(sc for sc, _ in scores)
    ok = [(len(sub), sc, sub) for sc, sub in scores if sc <= best * (1 + tol)]
    return min(ok, key=lambda x: (x[0], x[1]))[2]


def run(d: pd.DataFrame, train_days=180, step_days=7, embargo_days=3,
        variants=True, per_hour=True, nested=True) -> dict:
    """滚动重拟合跑完四层，每层都在**同一批**测试行上评估。

    variants=True 时额外跑"只留一个天气变量"的几组，用来回答
    "哪几个特征真的有用、哪些该删"——这是精简的依据，不是阶梯的一环。
    """
    # 官方预测覆盖率：在其他特征都齐的行里，有多少行拿得到决策时刻前的预调度。
    # 固定时刻口径下应接近 100%；明显偏低说明决策时刻早于 AEMO 发布次日预调度。
    # 两个数要分开看：
    #   in_range —— 预调度有数据的那段日期里，每天的小时覆盖了多少。低于 100% 说明
    #               决策时刻早于 AEMO 发布次日预调度（实测：12:30 只有 17%，13:00 起 100%）。
    #   span     —— 预调度历史从哪天到哪天。比价格历史短时，B2/B3 只能在这段里评估。
    base_ok = d[AR_COLS + WX_COLS + ["rrp", "naive"]].notna().all(axis=1)
    off_cov = float(d.loc[base_ok, "aemo"].notna().mean()) if base_ok.any() else 0.0
    has_a = d["aemo"].notna()
    off_span, off_in = None, None
    if has_a.any():
        a_, b_ = d.loc[has_a, "t"].min(), d.loc[has_a, "t"].max()
        off_span = [str(a_), str(b_)]
        w_ = base_ok & (d["t"] >= a_ + pd.Timedelta(days=1)) & (d["t"] <= b_ - pd.Timedelta(days=1))
        off_in = float(d.loc[w_, "aemo"].notna().mean()) if w_.any() else None
    have_off = d["aemo"].notna().sum() > 200
    base = AR_COLS + OFF_COLS if have_off else AR_COLS
    tiers = {"B0 naive": None, "B1 自回归": AR_COLS,
             "B2 +官方": (AR_COLS + OFF_COLS) if have_off else None,
             "B3 +天气": base + WX_COLS}
    var_cols: dict[str, list[str]] = {}
    if variants:
        for k in WX_COLS:
            var_cols[f"+{k} 单独"] = base + [k]
        for k in WX_COLS:
            rest = [c for c in WX_COLS if c != k]
            if rest:
                var_cols[f"去掉 {k}"] = base + rest
    need = AR_COLS + WX_COLS + ["rrp", "naive"] + (OFF_COLS if have_off else [])
    d = d.dropna(subset=need).sort_values("t").reset_index(drop=True)
    if d.empty:
        return {"status": "no_rows"}

    day = (d["t"] - d["t"].iloc[0]).dt.days.to_numpy()
    span = int(day.max()) + 1
    if span < train_days + embargo_days + step_days:
        return {"status": "too_short", "span_days": span,
                "need": train_days + embargo_days + step_days}

    allm = {**{k: v for k, v in tiers.items()}, **var_cols}
    pred = {k: np.full(len(d), np.nan) for k in allm}
    pred["B0 naive"] = d["naive"].to_numpy(float)
    if nested:
        pred[NESTED_NAME] = np.full(len(d), np.nan)
    picks: list[list[str]] = []
    n_fit = 0
    for d0 in range(train_days + embargo_days, span, step_days):
        tr_m = (day >= d0 - embargo_days - train_days) & (day < d0 - embargo_days)
        te_m = (day >= d0) & (day < d0 + step_days)
        if tr_m.sum() < 200 or te_m.sum() < 12:
            continue
        n_fit += 1
        for name, cols in allm.items():
            if cols is None:
                continue
            fp = _fit_predict_ph if per_hour else _fit_predict
            pred[name][te_m] = fp(d[tr_m], d[te_m], cols)
        if nested:
            fp = _fit_predict_ph if per_hour else _fit_predict
            sub = select_features(d[tr_m], base, fp, embargo_days)
            picks.append(sub)
            pred[NESTED_NAME][te_m] = fp(d[tr_m], d[te_m], base + sub)

    # 只在**每一层都有预测**的行上比较，否则比的不是同一批样本
    ok = np.ones(len(d), bool)
    for name, cols in allm.items():
        if cols is not None or name == "B0 naive":
            ok &= np.isfinite(pred[name])
    if nested:
        ok &= np.isfinite(pred[NESTED_NAME])
    y = d["rrp"].to_numpy(float)[ok]
    if ok.sum() < 200:
        return {"status": "too_few_test", "n": int(ok.sum())}

    err = {k: np.abs(v[ok] - y) for k, v in pred.items() if np.isfinite(v[ok]).all()}
    mae0 = float(np.mean(err["B0 naive"]))
    med0 = float(np.median(err["B0 naive"])) or 1.0
    out = {"status": "ok", "n_test": int(ok.sum()), "n_refit": n_fit,
           "per_hour": bool(per_hour),
           "official_coverage": round(off_cov, 4),
           "official_span": off_span,
           "official_cov_in_range": None if off_in is None else round(off_in, 4),
           "span": [str(d["t"][ok].min()), str(d["t"][ok].max())],
           "has_official": bool(have_off), "tiers": [], "variants": []}
    prev = prev_row = None
    for name in TIERS:
        if name not in err:
            out["tiers"].append({"tier": name, "skipped": "缺 AEMO 预调度"})
            continue
        e = err[name]
        row = {"tier": name,
               "mae": round(float(np.mean(e)), 2),
               "rmae": round(float(np.mean(e)) / mae0, 4),
               "rmdae": round(float(np.median(e)) / med0, 4),
               "rmse": round(float(np.sqrt(np.mean((pred[name][ok] - y) ** 2))), 2),
               "rho": round(float(stats.spearmanr(pred[name][ok], y)[0]), 3)}
        if prev is not None:
            gw = giacomini_white(err[prev], e)
            row["vs_prev"] = prev
            # 用上一个**真正跑成的**层做对照，不能拿 tiers[-1]——
            # B2 缺数据被跳过时，tiers[-1] 是那条占位记录，没有 rmae。
            row["gain_rmae"] = round(prev_row["rmae"] - row["rmae"], 4)
            row["gw_p"] = None if gw is None else round(gw, 4)
        out["tiers"].append(row)
        prev, prev_row = name, row

    # ---- 特征取舍：每个变体都和 B2（没有官方数据时是 B1）比 ----
    ref = "B2 +官方" if have_off else "B1 自回归"
    if ref in err:
        for name in var_cols:
            if name not in err:
                continue
            e = err[name]
            gw = giacomini_white(err[ref], e)
            out["variants"].append({
                "name": name, "vs": ref,
                "rmae": round(float(np.mean(e)) / mae0, 4),
                "gain_rmae": round(float(np.mean(err[ref]) - np.mean(e)) / mae0, 4),
                "gw_p": None if gw is None else round(gw, 4)})
        out["variants"].sort(key=lambda r: -r["gain_rmae"])

    # ---- 嵌套选特征：和 B3 用同一个对照（B2，没有官方数据时 B1）----
    if nested and NESTED_NAME in err and ref in err:
        e = err[NESTED_NAME]
        gw = giacomini_white(err[ref], e)
        nw = max(len(picks), 1)
        freq = {c: round(sum(c in p_ for p_ in picks) / nw, 3) for c in WX_COLS}
        combos: dict[str, int] = {}
        for p_ in picks:
            k = "+".join(p_) if p_ else "不用天气"
            combos[k] = combos.get(k, 0) + 1
        out["nested"] = {
            "name": NESTED_NAME, "vs": ref, "val_days": NESTED_VAL_DAYS, "tol": NESTED_TOL,
            "rmae": round(float(np.mean(e)) / mae0, 4),
            "gain_rmae": round(float(np.mean(err[ref]) - np.mean(e)) / mae0, 4),
            "gw_p": None if gw is None else round(gw, 4),
            "windows": len(picks), "freq": freq,
            "combos": dict(sorted(combos.items(), key=lambda kv: -kv[1]))}

    out["_pred"] = {k: v[ok] for k, v in pred.items()}
    out["_y"] = y
    out["_t"] = d["t"].to_numpy()[ok]
    out["_aemo"] = d["aemo"].to_numpy(float)[ok] if have_off else None
    return out


def giacomini_white(e1: np.ndarray, e2: np.ndarray, lag: int | None = None):
    """条件预测能力检验（GW）。

    综述明确指出：滚动重拟合下校准窗口是变的，Diebold-Mariano 的前提
    （预测方法固定、只比预测本身）不成立，该用 GW。这里用它的
    无条件形式 + Newey-West 稳健方差——损失差分序列自相关很强，
    不做 HAC 修正会把 p 值压到荒谬的小。
    """
    dl = e1 - e2
    n = len(dl)
    if n < 50:
        return None
    m = float(np.mean(dl))
    dm = dl - m
    lag = lag or int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    g0 = float(np.dot(dm, dm)) / n
    var = g0
    for k in range(1, lag + 1):
        gk = float(np.dot(dm[:-k], dm[k:])) / n
        var += 2.0 * (1.0 - k / (lag + 1.0)) * gk
    if var <= 0:
        return None
    t = m / np.sqrt(var / n)
    return float(2 * (1 - stats.norm.cdf(abs(t))))


def diagnose(res: dict, top: int = 6) -> dict:
    """误差解剖：某一层输给基线时，是**整体偏**还是**几天炸**？

    这两种病因完全不同，处理方式也完全不同：
      · 整体偏（中位误差就大、bias 明显）—— 模型或特征错了
      · 几天炸（中位数正常、最坏 1% 吃掉大半 MAE）—— 尖峰时段的外推问题

    MAE 是个均值，把这两种情况混成一个数。所以每层都拆开报。
    """
    if res.get("status") != "ok" or "_pred" not in res:
        return {"status": "unavailable"}
    y, t = res["_y"], pd.to_datetime(res["_t"])
    out = {"status": "ok", "tiers": []}
    for name in TIERS:
        p = res["_pred"].get(name)
        if p is None or not np.isfinite(p).all():
            continue
        e = np.abs(p - y)
        tot = float(e.sum())
        k = max(1, int(round(0.01 * len(e))))
        worst = np.argsort(-e)[:k]
        out["tiers"].append({
            "tier": name,
            "mae": round(float(e.mean()), 2),
            "med_ae": round(float(np.median(e)), 2),
            "p90_ae": round(float(np.percentile(e, 90)), 2),
            "p99_ae": round(float(np.percentile(e, 99)), 2),
            "bias": round(float(np.mean(p - y)), 2),
            "top1_share": round(float(e[worst].sum() / tot), 3) if tot > 0 else None,
            "worst": [{"t": str(t[i] + pd.Timedelta(hours=NEM_OFFSET_H))[:16],
                       "actual": round(float(y[i]), 1),
                       "pred": round(float(p[i]), 1)}
                      for i in np.argsort(-e)[:top]]})
    return out


def _block_boot(x: np.ndarray, blk: int = 5, B: int = 2000, seed: int = 7):
    """按天序列的移动块自助，返回均值的 95% 区间。"""
    n = len(x)
    if n < 3 * blk:
        return None
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / blk))
    starts = rng.integers(0, n - blk + 1, size=(B, nb))
    idx = (starts[:, :, None] + np.arange(blk)[None, None, :]).reshape(B, -1)[:, :n]
    m = x[idx].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


# 储能规格。和退役前的 l4_value 保持同一口径，结果才可比。
BATT_POWER_MW = 1.0
BATT_HOURS = 2           # 1 MW / 2 MWh：每天充 2 小时、放 2 小时
BATT_ETA = 0.85          # 往返效率
BATT_COST = 5.0          # 吞吐成本 $/MWh（按放电量计），覆盖衰减


def _best_cycle(p: np.ndarray, k: int, eta: float, cost: float):
    """在一天按时间排好的价格序列上，找一次**先充后放**的最优安排。

    充 k 小时、放 k 小时，所有充电小时都早于所有放电小时（一天一个循环，
    真实电池做不到先放后充再放——起始 SOC 是空的）。枚举分割点 s：
    充电从 [0, s) 里挑最便宜的 k 个，放电从 [s, n) 里挑最贵的 k 个。

    返回 (价值, 充电下标, 放电下标)。价值 ≤ 0 就不交易——扣掉效率和
    吞吐成本还不赚钱的日子，真实操作里也不会动。
    """
    n = len(p)
    best = (0.0, None, None)
    if n < 2 * k:
        return best
    for s in range(k, n - k + 1):
        ci = np.argsort(p[:s], kind="stable")[:k]
        di = s + np.argsort(-p[s:], kind="stable")[:k]
        v = eta * float(p[di].sum()) - float(p[ci].sum()) - cost * k
        if v > best[0]:
            best = (v, ci, di)
    return best


def economics(res: dict, eta: float = BATT_ETA, hours: int = BATT_HOURS,
              cost: float = BATT_COST) -> dict:
    """经济评估。综述原话：统计误差"may not necessarily reflect the economic
    value of reducing prediction errors"，所以必须补一个真实用途的案例。

    用一块 1 MW / 2 MWh 的电池做日内套利，这是天气预测最直接的变现方式：

        每个 NEM 交易日，**只用预测价**定下充放电计划（先充后放，各 2 小时），
        预测收益 ≤ 0 就当天不动；然后按**实际价**结算。

    完美预知用同一个优化器、喂实际价——它是同一约束下的收益上限，
    所以捕获率 = 该层收益 / 完美预知 不会超过 1。

    这里合并了原来的 l4_value.py。那个模块拿 l3_signal 的**水平**信号去排
    **日内**计划，实测捕获率 −262.5%：水平信号只回答"今天贵不贵"，
    回答不了"今天哪个小时最便宜"。现在计划直接来自分小时的价格预测。

    日收益序列近似独立（隔夜断开），用块自助给区间，不用 t 检验。
    """
    if res.get("status") != "ok" or "_pred" not in res:
        return {"status": "unavailable"}
    y, t = res["_y"], pd.to_datetime(res["_t"])
    # 按 NEM 交易日分组（UTC+10），不能按 UTC 日——那会在 NEM 10:00 切一刀，
    # 把午间低谷和前一晚的高峰分进同一天。
    day = (pd.Series(t) + pd.Timedelta(hours=NEM_OFFSET_H)).dt.floor("D").to_numpy()
    udays, inv = np.unique(day, return_inverse=True)
    order = np.argsort(np.asarray(t), kind="stable")
    groups = [(udays[i], order[inv[order] == i]) for i in range(len(udays))]
    groups = [(dd, g) for dd, g in groups if len(g) >= 2 * hours]
    gdays = [str(pd.Timestamp(dd).date()) for dd, _ in groups]
    groups = [g for _, g in groups]
    if len(groups) < 30:
        return {"status": "too_few_days", "days": len(groups)}

    def realised(p: np.ndarray) -> np.ndarray:
        out = []
        for g in groups:
            v, ci, di = _best_cycle(p[g], hours, eta, cost)
            if ci is None:
                out.append(0.0)
                continue
            yy = y[g]
            out.append(BATT_POWER_MW * (eta * float(yy[di].sum())
                                        - float(yy[ci].sum()) - cost * hours))
        return np.asarray(out, float)

    def within_day(p: np.ndarray) -> tuple[float, float]:
        """套利只吃**日内排序**，和价格水平无关。所以单独量两件事：

          rank_rho  —— 每天算一次 pred 与 actual 的秩相关，再取均值
          spread    —— 日内预测振幅 / 实际振幅 的中位数

        一个把日内曲线压平的模型（预测几乎是条直线），MAE 可以很漂亮，
        但挑出来的充放电时刻基本是噪声。MAE 看不见这件事，这两个数看得见。
        """
        rs, sp = [], []
        for g in groups:
            pi, yi = p[g], y[g]
            if len(np.unique(pi)) > 1 and len(np.unique(yi)) > 1:
                rs.append(stats.spearmanr(pi, yi)[0])
            ry = float(yi.max() - yi.min())
            if ry > 0:
                sp.append(float(pi.max() - pi.min()) / ry)
        return (float(np.nanmean(rs)) if rs else float("nan"),
                float(np.median(sp)) if sp else float("nan"))

    perfect = realised(y)
    pf = float(perfect.mean())

    rows, series = [], {}
    for name in TIERS:
        if name not in res["_pred"] or not np.isfinite(res["_pred"][name]).all():
            continue
        g = realised(res["_pred"][name])
        rr, sp = within_day(res["_pred"][name])
        series[name] = g
        rows.append({"tier": name,
                     "daily": round(float(g.mean()), 2),
                     "ci": [round(v, 2) for v in (_block_boot(g) or [float("nan")] * 2)],
                     "capture": round(float(g.mean()) / pf, 3) if pf > 0 else None,
                     "rank_rho": None if not np.isfinite(rr) else round(rr, 3),
                     "spread": None if not np.isfinite(sp) else round(sp, 3),
                     "trade_days": round(float((g != 0).mean()), 3),
                     "win_days": round(float((g > 0).mean()), 3)})
    base = rows[0]["tier"] if rows else None
    for i, r_ in enumerate(rows):
        if i == 0:
            continue
        prev = rows[i - 1]["tier"]
        dg = series[r_["tier"]] - series[prev]
        r_["vs_prev"] = prev
        r_["gain"] = round(float(dg.mean()), 2)
        r_["gain_ci"] = [round(v, 2) for v in (_block_boot(dg) or [float("nan")] * 2)]
        # 相对"一周前同期"（naive）——交易台用它挑"该用哪个模型"
        dn = series[r_["tier"]] - series[base]
        r_["vs_naive"] = round(float(dn.mean()), 2)
        r_["vs_naive_ci"] = [round(v, 2) for v in (_block_boot(dn) or [float("nan")] * 2)]
    # 该用哪个模型排电池：**不显著差于最好那个的、最简单的那个**。
    # 只挑均值最高的会把运气当本事；多一层特征必须用显著更高的收益换来。
    rec = None
    if rows:
        best = max(rows, key=lambda r_: r_["daily"])["tier"]
        for r_ in rows:
            dv = series[r_["tier"]] - series[best]
            ci = _block_boot(dv) if r_["tier"] != best else [0.0, 0.0]
            r_["vs_best_ci"] = None if ci is None else [round(v, 2) for v in ci]
            if rec is None and (r_["tier"] == best or (ci is not None and ci[1] >= 0)):
                rec = r_["tier"]
    # 嵌套选特征那一版：和 B3 同一个对照，单独记，不进"该用哪个模型"的挑选
    nest = None
    pn = res["_pred"].get(NESTED_NAME)
    if pn is not None and np.isfinite(pn).all():
        ref = "B2 +官方" if "B2 +官方" in series else "B1 自回归"
        g = realised(pn)
        series[NESTED_NAME] = g
        if ref in series:
            dg = g - series[ref]
            nest = {"tier": NESTED_NAME, "vs": ref,
                    "daily": round(float(g.mean()), 2),
                    "ci": [round(v, 2) for v in (_block_boot(g) or [float("nan")] * 2)],
                    "capture": round(float(g.mean()) / pf, 3) if pf > 0 else None,
                    "gain": round(float(dg.mean()), 2),
                    "gain_ci": [round(v, 2) for v in (_block_boot(dg) or [float("nan")] * 2)]}
    # 逐日收益序列：交易台画累计收益用
    daily = {"dates": gdays, "perfect": [round(float(v), 2) for v in perfect]}
    for k, v in series.items():
        daily[k] = [round(float(x), 2) for x in v]
    return {"status": "ok", "days": len(groups), "eta": eta, "daily": daily,
            "recommended": rec, "nested": nest,
            "battery": {"mw": BATT_POWER_MW, "hours": hours, "cost": cost},
            "hours_per_day": int(np.median([len(g) for g in groups])),
            "perfect_daily": round(pf, 2), "rows": rows}


BIAS_SHORT = "做空 AEMO"


def bias_trade(res: dict) -> dict:
    """第二种用法：**赌 AEMO 预调度的偏差**，按实际价结算。

    每个小时，决策时刻已经看得到 AEMO 对那个小时的预调度价 a。
    我们的预测 p 比 a 高就做多 1 MW、比 a 低就做空 1 MW，结算 = 方向 × (实际 − a)。
    也就是问：**我们的模型对 AEMO 的修正，方向对不对、值多少钱**。

    这不是一张能直接下的单——NEM 没有日前市场，AEMO 预调度不是成交价。
    现实里对应的是：拿这个方向去调整对冲头寸 / 报价，或者在 ASX 期货、
    场外互换上表达对次日均价的看法。所以它衡量的是**信号**值多少钱，
    和电池套利（只吃日内排序）互补：天气主要移动的是整天的水平，
    电池用不上，这里用得上。

    电价重尾：一个尖峰小时就能决定一整季的盈亏。所以除了均值和块自助区间，
    还报"方向命中率"（|实际 − a| < $1 的小时不计），它不受尖峰支配。
    """
    a = res.get("_aemo")
    if res.get("status") != "ok" or a is None or "_pred" not in res:
        return {"status": "unavailable"}
    y, t = res["_y"], pd.to_datetime(res["_t"])
    day = (pd.Series(t) + pd.Timedelta(hours=NEM_OFFSET_H)).dt.floor("D").to_numpy()
    udays, inv = np.unique(day, return_inverse=True)
    if len(udays) < 30:
        return {"status": "too_few_days", "days": int(len(udays))}
    move = y - a
    real = np.abs(move) >= 1.0
    names = [n for n in ("B1 自回归", "B2 +官方", "B3 +天气", NESTED_NAME)
             if n in res["_pred"] and np.isfinite(res["_pred"][n]).all()]
    rows, series = [], {}
    # 基线：不看任何模型，每个小时都做空预调度。预调度本身系统性偏高
    # （它按报价排序、不含之后的重新报价，尖峰常常不兑现），光这一条就能"赚"很多——
    # 那是预调度的口径偏差，不是我们的信息。模型的价值要和这条基线比。
    for n in [BIAS_SHORT] + names:
        sgn = (np.full(len(y), -1.0) if n == BIAS_SHORT
               else np.sign(res["_pred"][n] - a))
        pnl = sgn * move
        g = np.bincount(inv, weights=pnl, minlength=len(udays))
        series[n] = g
        hit = float((np.sign(move) == sgn)[real & (sgn != 0)].mean()) if (real & (sgn != 0)).any() else None
        rows.append({"tier": n, "daily": round(float(g.mean()), 2),
                     "ci": [round(v, 2) for v in (_block_boot(g) or [float("nan")] * 2)],
                     "per_mwh": round(float(pnl.mean()), 2),
                     "median_day": round(float(np.median(g)), 2),
                     "hit": None if hit is None else round(hit, 4),
                     "win_days": round(float((g > 0).mean()), 3)})
    ref = "B2 +官方" if "B2 +官方" in series else "B1 自回归"
    for r_ in rows:
        if r_["tier"] != BIAS_SHORT:
            ds = series[r_["tier"]] - series[BIAS_SHORT]
            r_["vs_short"] = round(float(ds.mean()), 2)
            r_["vs_short_ci"] = [round(v, 2) for v in (_block_boot(ds) or [float("nan")] * 2)]
        if r_["tier"] in ("B3 +天气", NESTED_NAME) and ref in series:
            dg = series[r_["tier"]] - series[ref]
            r_["vs"] = ref
            r_["gain"] = round(float(dg.mean()), 2)
            r_["gain_ci"] = [round(v, 2) for v in (_block_boot(dg) or [float("nan")] * 2)]
            # 去掉尖峰日再看一次：最好和最差各 2% 的日子拿掉，结论还在不在
            k = max(1, int(0.02 * len(dg)))
            core = np.sort(dg)[k:-k]
            r_["gain_trim"] = round(float(core.mean()), 2)
    gdays = [str(pd.Timestamp(x).date()) for x in udays]
    daily = {"dates": gdays}
    for k_, v in series.items():
        daily[k_] = [round(float(x), 2) for x in v]
    return {"status": "ok", "days": int(len(udays)), "hours": int(len(y)),
            "ref": ref, "mw": BATT_POWER_MW, "rows": rows, "daily": daily,
            "baseload": _bias_baseload(res, inv, len(udays), names, ref)}


def _bias_baseload(res: dict, inv: np.ndarray, nd: int, names: list[str], ref: str) -> dict:
    """同一个问题换成**按天**下注：一天只押一次方向——明天全天均价比预调度的全天均价高还是低，
    1 MW 全天（每个可交易小时 1 MWh），按实际全天均价结算。

    这更接近能真正做的东西（日基荷互换、调整次日对冲量），也更难被个别小时的尖峰撑起来：
    小时级的命中可能只是"某几个小时方向对"，日级要的是整天的水平判断对。
    """
    y, a = res["_y"], res["_aemo"]
    cnt = np.bincount(inv, minlength=nd).astype(float)
    ybar = np.bincount(inv, weights=y, minlength=nd) / cnt
    abar = np.bincount(inv, weights=a, minlength=nd) / cnt
    series, rows = {}, []
    for n in [BIAS_SHORT] + names:
        if n == BIAS_SHORT:
            sgn = -np.ones(nd)
        else:
            pbar = np.bincount(inv, weights=res["_pred"][n], minlength=nd) / cnt
            sgn = np.sign(pbar - abar)
        g = sgn * (ybar - abar) * cnt * BATT_POWER_MW
        series[n] = g
        hit = (np.sign(ybar - abar) == sgn)[sgn != 0]
        rows.append({"tier": n, "daily": round(float(g.mean()), 2),
                     "ci": [round(v, 2) for v in (_block_boot(g) or [float("nan")] * 2)],
                     "hit": round(float(hit.mean()), 4) if len(hit) else None})
    for r_ in rows:
        if r_["tier"] in ("B3 +天气", NESTED_NAME) and ref in series:
            dg = series[r_["tier"]] - series[ref]
            r_["vs"] = ref
            r_["gain"] = round(float(dg.mean()), 2)
            r_["gain_ci"] = [round(v, 2) for v in (_block_boot(dg) or [float("nan")] * 2)]
            r_["flips"] = int((np.sign(series[r_["tier"]]) != np.sign(series[ref])).sum())
    out = {"rows": rows, "daily": {k: [round(float(x), 2) for x in v] for k, v in series.items()}}
    return out


def _meta(a) -> dict:
    return {"generated_at": pd.Timestamp.now("UTC").isoformat(),
            "hours": a.hours, "decide_at": a.decide_at,
            "is_fixture": "fixture" in os.environ.get("WQ_DATA_DIR", "").lower(),
            "train_days": a.train_days, "step_days": a.step_days,
            "embargo_days": a.embargo_days, "per_hour": not a.shared,
            "battery": {"mw": BATT_POWER_MW, "hours": BATT_HOURS,
                        "eta": BATT_ETA, "cost": BATT_COST},
            "regions": {}}


def _merge(a) -> int:
    out = _meta(a)
    order = [s_[1] for s_ in SITES]
    got = {}
    for f in a.merge:
        with open(f) as fh:
            r = json.load(fh)
        got[r.get("region") or os.path.basename(f).split(".")[0]] = r
    for k in sorted(got, key=lambda k: order.index(k) if k in order else 99):
        out["regions"][k] = got[k]
    with open(a.out or "ladder.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=str)
    print(f"→ {a.out or 'ladder.json'}（{'、'.join(out['regions'])}）")
    return 0


def _all_regions(a) -> int:
    """跑全部区域，汇总成一个 JSON 给交易台用。

    某一区跑挂了不中断——把失败原因原样记下来，页面上照实显示，
    比整个流程停在第一个错误上有用。
    """
    hrs = None if a.hours == "all" else tuple(int(x) for x in a.hours.split("-"))
    out = _meta(a)
    con = store.connect()
    try:
        for site_id, region, *_ in SITES:
            print(f"\n─── {region} " + "─" * 60)
            try:
                d = build(con, region, site_id, hrs, a.decide_at)
                res = run(d, a.train_days, a.step_days, a.embargo_days,
                          per_hour=not a.shared)
                ec = economics(res) if res.get("status") == "ok" else {"status": "skip"}
                dg = diagnose(res) if res.get("status") == "ok" else {"status": "skip"}
                pub = {k: v for k, v in res.items() if not k.startswith("_")}
                bt = bias_trade(res) if res.get("status") == "ok" else {"status": "skip"}
                out["regions"][region] = {**pub, "economics": ec, "diag": dg, "bias": bt}
                if res.get("status") == "ok":
                    sp = res.get("official_span")
                    if sp:
                        print(f"  预调度 {str(sp[0])[:10]} → {str(sp[1])[:10]}，"
                              f"区间内覆盖 {res.get('official_cov_in_range') or 0:.1%}")
                    else:
                        print("  预调度：无")
                    for r_ in res["tiers"]:
                        if r_.get("skipped"):
                            print(f"  {r_['tier']:<12s} —— {r_['skipped']}")
                        else:
                            print(f"  {r_['tier']:<12s} rMAE {r_['rmae']:.3f}"
                                  f"  MAE {r_['mae']:>8.2f}")
                else:
                    print(f"  {res}")
            except Exception as e:                      # noqa: BLE001
                print(f"  跑不动：{type(e).__name__}: {e}")
                out["regions"][region] = {"status": "error",
                                          "error": f"{type(e).__name__}: {e}"}
    finally:
        con.close()
    path = a.out or "ladder.json"
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=str)
    ok = sum(1 for v in out["regions"].values() if v.get("status") == "ok")
    print(f"\n→ {path}   {ok}/{len(out['regions'])} 个区跑通")
    print(f"  接着跑：python desk.py --in {path} --out 交易台.html")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="基准阶梯：天气相对价格自回归与官方预测的增量")
    ap.add_argument("--region", default="QLD1",
                    help="区域代码，或 all 跑全部四区并汇总到一个 JSON")
    ap.add_argument("--hours", default="10-16")
    ap.add_argument("--decide-at", default=DECIDE_AT_NEM,
                    help="决策时刻（NEM，前一天），默认 13:30。NEM 日 D 的 24 小时在 D−1 天这个时刻一次性定计划")
    ap.add_argument("--train-days", type=int, default=180)
    ap.add_argument("--step-days", type=int, default=7)
    ap.add_argument("--embargo-days", type=int, default=3)
    ap.add_argument("--out", default=None)
    ap.add_argument("--merge", nargs="+", default=None, metavar="REGION.json",
                    help="把几次单区运行的 JSON 合成交易台用的一个文件（配合 --out）。"
                         "单区分开跑时用，比如机器有单次运行时长限制。")
    ap.add_argument("--shared", action="store_true",
                    help="改用一个跨 24 小时的共享模型。默认是分小时拟合（综述的标准做法）；"
                         "共享模型会把日内曲线压平，只留作对照。")
    a = ap.parse_args(argv)

    if a.merge:
        return _merge(a)
    if a.region.lower() == "all":
        return _all_regions(a)

    site = next((s[0] for s in SITES if s[1] == a.region), None)
    if site is None:
        print(f"未知区 {a.region}"); return 2
    hrs = None if a.hours == "all" else tuple(int(x) for x in a.hours.split("-"))

    con = store.connect()
    try:
        d = build(con, a.region, site, hrs, a.decide_at)
        res = run(d, a.train_days, a.step_days, a.embargo_days,
                  per_hour=not a.shared)
    finally:
        con.close()

    print("\n" + "=" * 76)
    print(f"{a.region} · 时段 {a.hours} · 基准阶梯（全部样本外，滚动重拟合）")
    print("=" * 76)
    if res.get("status") != "ok":
        print(f"  {res}")
        return 1
    print(f"  测试 {res['n_test']} 小时   {res['span'][0][:10]} → {res['span'][1][:10]}"
          + ("   · 分小时拟合（24 个模型）" if res.get("per_hour")
             else "   · 单一共享模型（对照用）"))
    print(f"  决策时刻：前一天 {a.decide_at}（NEM）")
    if res["has_official"]:
        sp = res.get("official_span") or ["?", "?"]
        cin = res.get("official_cov_in_range")
        print(f"  AEMO 预调度历史 {str(sp[0])[:10]} → {str(sp[1])[:10]}，"
              f"区间内小时覆盖 {cin:.1%}" if cin is not None else "")
        if cin is not None and cin < 0.95:
            print("  ⚠ 区间内覆盖不到 95%：决策时刻早于 AEMO 发布覆盖次日的预调度，")
            print("    试试 --decide-at 13:30 或更晚（实测 12:30 只覆盖 17%）。")
        if res.get("official_coverage", 1) < 0.95:
            print("  注意：预调度历史比价格历史短，B2/B3 只在上面这段日期里评估。")
    if not res["has_official"]:
        print("  ⚠ 没有 AEMO 预调度数据：B2 跳过，B3 的增量**没有扣掉官方预测**，")
        print("    那部分市场已经定价，不能当作 alpha。")
    print(f"\n  {'层':<12s}{'MAE':>9s}{'rMAE':>8s}{'rMdAE':>8s}{'RMSE':>10s}"
          f"{'ρ':>8s}{'相对上层':>10s}{'GW p':>9s}")
    print("  " + "-" * 74)
    for r_ in res["tiers"]:
        if r_.get("skipped"):
            print(f"  {r_['tier']:<12s}  —— {r_['skipped']}")
            continue
        g = f"{r_['gain_rmae']:+.4f}" if "gain_rmae" in r_ else "    —"
        p_ = f"{r_['gw_p']:.4f}" if r_.get("gw_p") is not None else "    —"
        print(f"  {r_['tier']:<12s}{r_['mae']:>9.2f}{r_['rmae']:>8.3f}"
              f"{r_['rmdae']:>8.3f}{r_['rmse']:>10.2f}{r_['rho']:>8.3f}"
              f"{g:>10s}{p_:>9s}")
    print("  " + "-" * 74)
    print("  rMAE = 相对 naive(P d−7,h) 的 MAE 比值，<1 才算有用。")
    print("  rMdAE = 同样口径的**中位**误差比。尖峰主导时 MAE 讲的是尖峰，")
    print("  rMdAE 才讲'平常一小时'。两个背离就看下面的误差解剖。")
    print("  相对上层 = rMAE 的改善量；GW p = Giacomini-White 检验（滚动重拟合下")
    print("  校准窗口在变，DM 的前提不成立，该用 GW）。")

    if res.get("variants"):
        print(f"\n  特征取舍（每组都和 {res['variants'][0]['vs']} 比，正数才值得留）")
        print("  " + "-" * 74)
        for v in res["variants"]:
            p_ = f"{v['gw_p']:.3f}" if v.get("gw_p") is not None else "  —"
            gp = v.get("gw_p")
            mark = "✓" if v["gain_rmae"] > 0 and gp is not None and gp <= 0.10 else " "
            print(f"  {mark} {v['name']:<14s}rMAE {v['rmae']:>7.3f}"
                  f"   增量 {v['gain_rmae']:>+8.4f}   GW p {p_:>6s}")
        print("  " + "-" * 74)
        print("  ✓ = 有改善且 GW p≤0.10。没有 ✓ 的特征应当删掉：多一个自由度，")
        print("  少一分稳定，还骗人。")

    ec = economics(res)
    if ec.get("status") == "ok":
        hpd = ec.get("hours_per_day", 0)
        bt = ec["battery"]
        print(f"\n  经济评估 · {bt['mw']:.0f} MW / {bt['mw'] * bt['hours']:.0f} MWh 电池，"
              f"先充后放，η={ec['eta']}，吞吐成本 ${bt['cost']:.0f}/MWh")
        print(f"  {ec['days']} 个 NEM 交易日，每日 {hpd} 个可选小时")
        print("  " + "-" * 74)
        print(f"  {'层':<12s}{'日均 $/天':>13s}{'95% 区间':>19s}"
              f"{'捕获率':>8s}{'日内秩ρ':>9s}{'振幅比':>8s}{'相对上层':>10s}")
        for r_ in ec["rows"]:
            ci = f"[{r_['ci'][0]:.1f}, {r_['ci'][1]:.1f}]"
            cap = f"{r_['capture']:.1%}" if r_["capture"] is not None else "  —"
            g = (f"{r_['gain']:+.2f}" if "gain" in r_ else "   —")
            rr = "—" if r_.get("rank_rho") is None else f"{r_['rank_rho']:.3f}"
            sp = "—" if r_.get("spread") is None else f"{r_['spread']:.2f}"
            print(f"  {r_['tier']:<12s}{r_['daily']:>13.2f}{ci:>19s}{cap:>8s}"
                  f"{rr:>9s}{sp:>8s}{g:>10s}")
        print(f"  {'完美预知':<12s}{ec['perfect_daily']:>13.2f}"
              f"{'（收益上限）':>20s}{'100.0%':>9s}")
        print("  " + "-" * 74)
        print("  计划只用预测价定，按实际价结算；预测收益 ≤ 0 的日子不交易。")
        print("  决策只用到预测的**日内排序**，不用绝对水平。所以另外量两件事：")
        print("  日内秩ρ = 每天 pred 与 actual 的秩相关的均值；")
        print("  振幅比 = 日内预测振幅 / 实际振幅的中位数。")
        print("  一个把日内曲线压平的模型（振幅比接近 0），MAE 可以很漂亮，")
        print("  但挑出来的充放电时刻基本是噪声——这就是 MAE 赢了、钱却输了的机制。")
        best = max(ec["rows"], key=lambda r_: r_["daily"])
        b0 = ec["rows"][0]
        if best["tier"] != b0["tier"] and b0["daily"] > 0:
            pass
        elif best["tier"] == b0["tier"] and len(ec["rows"]) > 1:
            print(f"\n  ⚠ 赚钱最多的是 {b0['tier']}——统计上更好的层反而更少。")
            print("    看'日内秩ρ'和'振幅比'两列：多半是回归把日内曲线压平了。")
            print("    MAE 奖励'贴近水平'，套利要的是'排对顺序'，两个目标不一样。")
        if hpd and hpd < 12:
            print(f"\n  ⚠ 每天只有 {hpd} 个可选小时。储能真正的套利是"
                  "'午间充、晚峰放'，")
            print("    跨越的正是这个窗口之外的时段。要看真实经济价值，"
                  "请用 --hours all 再跑一遍；")
            print("    这里的窗口版只说明'在这个时段内部'能捕多少。")
        b3 = next((r_ for r_ in ec["rows"] if r_["tier"] == "B3 +天气"), None)
        if b3 and "gain" in b3:
            lo, hi = b3["gain_ci"]
            if lo > 0:
                print(f"\n  → 天气让每天多赚 ${b3['gain']:.2f}，"
                      f"95% 区间 [{lo:.2f}, {hi:.2f}] 不含 0。这是可交易的。")
            else:
                print(f"\n  → 天气的日均增益 ${b3['gain']:.2f}/天，"
                      f"95% 区间 [{lo:.2f}, {hi:.2f}] 跨 0——不能当作优势。")
    else:
        print(f"\n  经济评估：{ec.get('status')}")

    nd = res.get("nested")
    if nd:
        fq = "  ".join(f"{k} {v:.0%}" for k, v in nd["freq"].items())
        p_ = "—" if nd["gw_p"] is None else f"{nd['gw_p']:.4f}"
        print(f"\n  嵌套选特征（每个窗口只用训练段挑，最后 {nd['val_days']} 天验证）"
              f"：{nd['windows']} 个窗口")
        print(f"    被选中的比例：{fq}")
        print(f"    相对 {nd['vs']}：rMAE 增量 {nd['gain_rmae']:+.4f}   GW p {p_}")
        en = (ec or {}).get("nested") if ec.get("status") == "ok" else None
        if en:
            print(f"    电池日收益增量 {en['gain']:+.2f} $/天 "
                  f"[{en['gain_ci'][0]:.2f}, {en['gain_ci'][1]:.2f}]")

    bt_ = bias_trade(res)
    if bt_.get("status") == "ok":
        print(f"\n  赌 AEMO 预调度偏差（1 MW，方向 = sign(预测 − AEMO)，按实际价结算）"
              f"：{bt_['days']} 天")
        print(f"  {'层':<12s}{'日均 $/天':>12s}{'95% 区间':>22s}{'命中率':>8s}"
              f"{'相对' + bt_['ref']:>16s}")
        for r_ in bt_["rows"]:
            ci = f"[{r_['ci'][0]:.0f}, {r_['ci'][1]:.0f}]"
            hit = "—" if r_["hit"] is None else f"{r_['hit']:.1%}"
            g = ("" if "gain" not in r_ else
                 f"{r_['gain']:+.1f} [{r_['gain_ci'][0]:.0f}, {r_['gain_ci'][1]:.0f}]")
            print(f"  {r_['tier']:<12s}{r_['daily']:>12.1f}{ci:>22s}{hit:>8s}  {g}")

    dg = diagnose(res)
    if dg.get("status") == "ok":
        print("\n  误差解剖（MAE 是均值，会把'整体偏'和'几天炸'混成一个数）")
        print("  " + "-" * 74)
        print(f"  {'层':<12s}{'MAE':>9s}{'中位':>9s}{'p90':>9s}{'p99':>10s}"
              f"{'偏差':>10s}{'最坏1%占比':>12s}")
        for r_ in dg["tiers"]:
            sh = "—" if r_["top1_share"] is None else f"{r_['top1_share']:.1%}"
            print(f"  {r_['tier']:<12s}{r_['mae']:>9.2f}{r_['med_ae']:>9.2f}"
                  f"{r_['p90_ae']:>9.2f}{r_['p99_ae']:>10.2f}{r_['bias']:>10.2f}{sh:>12s}")
        print("  " + "-" * 74)
        bad = [r_ for r_ in dg["tiers"]
               if r_["top1_share"] is not None and r_["top1_share"] > 0.35]
        for r_ in bad:
            print(f"  ⚠ {r_['tier']}：最坏 1% 的行吃掉了 {r_['top1_share']:.0%} 的总误差。")
            print(f"    这不是'模型不好'，是几个时刻炸了。最坏的几行：")
            for w in r_["worst"][:4]:
                print(f"      {w['t']}  实际 {w['actual']:>9.1f}  预测 {w['pred']:>9.1f}")

    wx = next((r_ for r_ in res["tiers"] if r_["tier"] == "B3 +天气"), None)
    if wx and "gain_rmae" in wx:
        base = wx["vs_prev"]
        if wx["gain_rmae"] <= 0:
            print(f"\n  → 天气在 {base} 之上**没有改善**（{wx['gain_rmae']:+.4f}）。"
                  f"这条路线到此为止。")
        elif wx.get("gw_p") is not None and wx["gw_p"] > 0.05:
            print(f"\n  → 天气改善 {wx['gain_rmae']:+.4f}，但 GW p={wx['gw_p']:.3f} "
                  f"不显著——和运气分不开。")
        else:
            print(f"\n  → 天气在 {base} 之上改善 rMAE {wx['gain_rmae']:+.4f}"
                  f"（GW p={wx['gw_p']:.4f}）。这才是天气的交易价值。")
    print()

    if a.out:
        pub = {k: v for k, v in res.items() if not k.startswith("_")}
        with open(a.out, "w") as f:
            json.dump({"region": a.region, "hours": a.hours,
                       "generated_at": pd.Timestamp.now("UTC").isoformat(),
                       **pub, "economics": ec, "diag": dg,
                       "bias": bias_trade(res)}, f,
                      ensure_ascii=False, indent=1, default=str)
        print(f"→ {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

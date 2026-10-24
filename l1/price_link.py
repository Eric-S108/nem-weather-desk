"""
L6 分水岭：净负荷缺口 ↔ 真实 RRP 的关系到底有多强。
====================================================
整个系统能不能赚钱，取决于这一个问题。骨架第 2 条说"净负荷是电价第一驱动"——
那是别人的结论，必须在**你自己的样本**上验一遍。

三个层次，一层比一层严：
  ① 真值缺口 vs RRP    —— 物理关系存不存在。不存在，后面都不用谈。
  ② 预报缺口 vs RRP    —— 你实际能拿到的信号还剩多少。这才是可交易的那个数。
  ③ 预报缺口 vs (RRP − AEMO 预调度) —— **相对官方预测的增量**。
     骨架第 2 条的原话：真正的 alpha 是预测 AEMO 预测的偏差。
     ③ 显著不为零，你才有别人没有的东西；否则只是把公开信息重算了一遍。

另外按时段切片：这类信号常常只在晚高峰成立，全天平均会把它稀释掉。

    python price_link.py                       全部区
    python price_link.py --region SA1 --spike 300
    python price_link.py --out ../price_link.json
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
import engine as E  # noqa: E402
import realdata as R  # noqa: E402
import store  # noqa: E402
from config import SITES  # noqa: E402

# ERA5 真值的可见延迟（天）。任何用真值构造的基准都必须让开这段时间，
# 否则就是"决策时还看不到的东西"混进了特征。
TRUTH_LAG_D = 6

BOOT_B = 400          # 移动块自助次数

HOUR_BUCKETS = {
    "夜间 00–06": lambda h: (h >= 0) & (h < 6),
    "早高峰 06–10": lambda h: (h >= 6) & (h < 10),
    "日间 10–16": lambda h: (h >= 10) & (h < 16),
    "晚高峰 16–21": lambda h: (h >= 16) & (h < 21),
    "夜前 21–24": lambda h: (h >= 21) & (h < 24),
}


def _rate(flag, finite) -> float:
    """在**有观测**的行上算比例（百分数）。

    `spk > spike` 是布尔数组：NaN 比较后变 False，nanmean 已经没有 NaN 可跳，
    于是缺失的小时被静静算作"没有尖峰"，基准率被系统性压低。必须先掩码。
    """
    n = int(np.sum(finite))
    return float(np.sum(np.asarray(flag) & finite)) / n * 100.0 if n else float("nan")


def _corr(x, y):
    """Spearman 秩相关。电价厚尾，Pearson 会被几个尖峰绑架。"""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 50:
        return None, None, int(m.sum())
    r, p = stats.spearmanr(x[m], y[m])
    return float(r), float(p), int(m.sum())


def _clim_trailing(vre, t, hod, days=30, min_obs=8, lag_days=TRUTH_LAG_D):
    """同一整点、过去 days 天的滚动均值，且**整体后移 lag_days 天**。

    为什么要后移：这个基准是用 ERA5 真值算的，而 ERA5 有约 5 天延迟。
    要在 t−24h 决策，那一刻能拿到的真值只到 t−6 天左右。不后移的话，
    基准里就掺进了"决策时还看不到的真值"——一个安静的前视泄漏，
    它不会报错，只会让 ρ 偏高，然后在实盘上还回去。

    ① 和 ② 共用同一个基准：两者只应差在"用真值 VRE 还是预报 VRE"，
    其余一切必须相同，否则 ① 不再是 ② 的上限。
    """
    hod = np.asarray(hod)
    v = np.asarray(vre, float)
    out = np.full(len(v), np.nan)
    for h in range(24):
        idx = np.where(hod == h)[0]
        if len(idx) == 0:
            continue
        ser = pd.Series(v[idx])                      # 每个整点每天一条，行数即天数
        sh = ser.shift(lag_days)
        # 不再用 expanding(min_periods=1) 兜底：那会拿 1~7 个观测冒充"30 天气候态"，
        # 每个整点各有 7 行受影响，而它们正好落进十分位的极端档。
        # 窗口没攒够时正确答案是 NaN，下游的 isfinite 掩码会把它们丢掉。
        out[idx] = sh.rolling(days, min_periods=min_obs).mean().to_numpy()
    return out


def _corr_within(x, y, groups, boot=0, block=24, rng=None):
    """分层秩相关：先在每个组内把 x、y 转成组内秩，再整体求相关。

    组取 (整点 × 年月)。不分层的话，ρ 会被两个混杂吞掉：
    日循环（夜里没光、需求也低）和季节（冬天风光普遍低、电价另有走势）。
    这两个是**已知且公开**的东西，交易上不值钱；值钱的是"同一个 6 月的下午 3 点，
    今天比往常差多少"。分层之后剩下的才是那部分。

    boot>0 时给出移动块自助的 95% 区间。电价和天气都是强自相关的，
    普通 p 值把 N 个小时当成 N 个独立观测，会把置信度吹得离谱；
    按 block 小时整段重采样才尊重这个自相关。
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    g = np.asarray(groups)
    m = np.isfinite(x) & np.isfinite(y)
    out = {"rho": None, "p": None, "n": int(m.sum()), "ci": None}
    if m.sum() < 200:
        return out
    df = pd.DataFrame({"x": x[m], "y": y[m], "g": g[m]})
    df = df[df.groupby("g")["x"].transform("size") >= 10]
    if len(df) < 200:
        out["n"] = int(len(df))
        return out
    # 组内秩归一到 [0,1]，让不同大小的组可比
    for c in ("x", "y"):
        df[c + "r"] = df.groupby("g")[c].rank(pct=True)
        df[c + "r"] -= df.groupby("g")[c + "r"].transform("mean")
    xr = df["xr"].to_numpy(); yr = df["yr"].to_numpy()
    r, pv = stats.pearsonr(xr, yr)
    out.update(rho=float(r), p=float(pv), n=int(len(df)))
    if boot:
        out["ci"] = _boot_ci(xr, yr, block=block, B=boot, rng=rng)
    return out


def _boot_ci(xr, yr, block=24, B=400, rng=None, alpha=0.05):
    """移动块自助的相关系数区间。块长默认 24 小时，覆盖一个完整日循环。"""
    rng = rng or np.random.default_rng(12345)
    n = len(xr)
    if n < block * 10:
        return None
    nb = int(np.ceil(n / block))
    starts_hi = n - block
    vals = []
    for _ in range(B):
        st = rng.integers(0, starts_hi + 1, size=nb)
        idx = (st[:, None] + np.arange(block)[None, :]).ravel()[:n]
        a, b = xr[idx], yr[idx]
        sa, sb = a.std(), b.std()
        if sa < 1e-12 or sb < 1e-12:
            continue
        vals.append(float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb)))
    if len(vals) < B // 2:
        return None
    lo, hi = np.percentile(vals, [alpha / 2 * 100, (1 - alpha / 2) * 100])
    return [round(float(lo), 3), round(float(hi), 3)]


def _lag_scan(deficit, rrp, groups, lags=range(-12, 13), floor=None):
    """把电价整体前后平移若干小时，看相关在哪个位移上最强。

    这是**时区与端点对齐的体检**：缺口和电价是同一时刻的物理量，峰值就该落在 0。
    如果峰值落在 ±10 小时附近，几乎可以断定 NEM(UTC+10) 的偏移用反了；
    落在 ±1，多半是"区间起点/终点"记反了。
    一个数就能把一整类静默错误暴露出来，比反复读时区代码可靠。
    """
    d = np.asarray(deficit, float); r = np.asarray(rrp, float); g = np.asarray(groups)
    out = []
    for k in lags:
        rs = np.roll(r, k).astype(float)
        gs = np.roll(g, k)
        # 卷绕处的样本要丢掉，否则把序列首尾接在了一起
        if k > 0:
            rs[:k] = np.nan
        elif k < 0:
            rs[k:] = np.nan
        res = _corr_within(d, rs, gs)
        if res["rho"] is not None:
            out.append((int(k), round(res["rho"], 4)))
    if not out:
        return None
    best = max(out, key=lambda x: abs(x[1]))
    r0 = dict(out).get(0)
    # 关系本来就弱时，这条曲线是平的，argmax 纯属噪声——不设门槛就会**每个区都报警**，
    # 把人送去查一个根本不存在的时区 bug。所以用自助区间的半宽当噪声尺度：
    # 峰值没有明显超过 lag=0，就老实说"曲线是平的，测不出来"。
    # 噪声尺度：曾经试过用"各 lag 的 ρ 的标准差"，那是**错的**——相邻 lag 高度相关，
    # 曲线很光滑，标准差远小于单个 ρ 的抽样误差，结果是把每个区都判成错位。
    # 正确的量是单个 ρ 的抽样波动，也就是移动块自助的半宽。
    ci = _boot_ci(*_ranks_for_boot(d, r, g), block=24, B=200) if len(d) > 500 else None
    noise = max(0.03, (ci[1] - ci[0]) / 2.0) if ci else 0.05
    # 多重比较修正：峰值是**在 25 个 lag 里挑出来的最大值**，不是一次独立抽样。
    # 拿单次抽样的波动当门槛，最大值几乎必然越线——这个体检已经误报三次了。
    # 25 个（高度相关的）候选，用 sqrt(2·ln(K)) 放大门槛，是极值统计的标准做法。
    k = max(2, len(out))
    noise *= float(np.sqrt(2.0 * np.log(k)))
    gain = abs(best[1]) - abs(r0 or 0.0)

    # 更重要的一道闸：整条曲线本来就在噪声里时，argmax 落在哪儿毫无意义。
    # floor 由调用方给（拟合的空对照地板）。|ρ| 全程不过地板 → 拒绝下判断，
    # 而不是硬报一个"时间轴接错了"，把人送去查根本不存在的 bug。
    if floor is not None and abs(best[1]) < floor:
        verdict = "too_weak"
    elif best[0] == 0:
        verdict = "ok"
    else:
        verdict = "flat" if gain < noise else "misaligned"
    return {"best_lag_h": best[0], "best_rho": best[1], "rho_at_0": r0,
            "gain": round(float(gain), 4), "noise": round(float(noise), 4),
            "floor": None if floor is None else round(float(floor), 4),
            "verdict": verdict, "curve": out}


def _ranks_for_boot(x, y, g):
    """给 _boot_ci 用的组内去均值秩。

    必须套用和 _corr_within 一样的组大小过滤，否则区间和它要标定的 ρ
    算在两批不同的样本上。
    """
    m = np.isfinite(x) & np.isfinite(y)
    df = pd.DataFrame({"x": x[m], "y": y[m], "g": g[m]})
    df = df[df.groupby("g")["x"].transform("size") >= 10]
    for c in ("x", "y"):
        df[c + "r"] = df.groupby("g")[c].rank(pct=True)
        df[c + "r"] -= df.groupby("g")[c + "r"].transform("mean")
    return df["xr"].to_numpy(), df["yr"].to_numpy()


def acf_time(x, max_lag=None) -> float:
    """积分自相关时间：ACF 累加到首次转负为止。块长按它取，不靠猜。"""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 50:
        return 1.0
    x = x - x.mean()
    v = float(np.dot(x, x))
    if v <= 0:
        return 1.0
    max_lag = max_lag or min(n // 4, 2000)
    tau = 1.0
    for k in range(1, max_lag):
        r = float(np.dot(x[:-k], x[k:])) / v
        if r <= 0:
            break
        tau += 2.0 * r
    return max(1.0, tau)


def drivers_decomp(cols: dict, rrp, grp, hod, half_split=True, block=None):
    """把复合信号拆开：每个原始气象量单独和电价比，再拟合一个最优组合。

    为什么要拆：`净负荷 = 需求(气温,时刻) − 光权重×光 − 风权重×风` 里的三个权重
    （solar_w / wind_w / VRE_SHARE）全是我手填的估计值。权重填错，即使每个分量的
    方向都对，合成出来的复合信号照样可以是反的。分量表能直接看出是哪一项在拖后腿。

    拟合部分给的是**这套特征的诚实上限**：在组内秩空间上用前半段拟合、后半段打分。
    样本外 ρ 就是"就算权重让数据自己挑，最多也就这么多"。
    比手填权重的结果高很多，说明该重估权重；两者都低，就是特征本身不够。
    """
    out = {"per_driver": {}, "fit": None, "fit_null": None}
    keys = list(cols)
    R_ = {}
    m = np.isfinite(rrp)
    for k in keys:
        m &= np.isfinite(cols[k])
    if m.sum() < 500:
        return out

    df = pd.DataFrame({k: np.asarray(cols[k], float)[m] for k in keys})
    df["y"] = np.asarray(rrp, float)[m]
    df["g"] = np.asarray(grp)[m]
    df["_pos"] = np.where(m)[0]          # 记住原始行号，拟合值要贴回去
    df = df[df.groupby("g")["y"].transform("size") >= 10]
    if len(df) < 500:
        return out
    for c in keys + ["y"]:
        r = df.groupby("g")[c].rank(pct=True)
        R_[c] = (r - r.groupby(df["g"]).transform("mean")).to_numpy()

    for k in keys:
        rr, pv = stats.pearsonr(R_[k], R_["y"])
        out["per_driver"][k] = {"rho": round(float(rr), 3), "p": float(pv)}

    if half_split and len(df) > 1000:
        X = np.column_stack([R_[k] for k in keys])
        y = R_["y"]
        # 块长写死 24*7 行只在"整点全留"时才等于一周。drivers_day 只留 6 小时/天，
        # 168 行就是 28 天——CV 的禁运带被撑成 4 周，空对照又几乎打乱不动。
        # 这个坑在 l3_signal 里踩过一次（地板 == ρ），这里必须同样按数据量出来。
        blk = block or int(np.clip(round(3 * max(acf_time(y), acf_time(X[:, 0]))),
                                   24, max(24, len(y) // 12)))
        out["block_rows"] = int(blk)
        out["fit"] = _fit_eval(X, y, keys, block=blk)
        # 折外预测贴回原始行号，供十分位 / 尖峰提升使用。
        # 十分位若用手填口径，量的是一个已知失效的信号——报出来会误导。
        pr = out["fit"].pop("_preds", None)
        if pr is not None:
            full = np.full(len(rrp), np.nan)
            full[df["_pos"].to_numpy()] = pr
            out["fit_pred"] = full
        # 空对照必须**多抽几次**。单抽一次得到的是噪声的一个样本，不是噪声的尺度——
        # 拿它当地板，可能偏小（放行假信号）也可能偏大（毙掉真信号）。
        # 这里抽 20 次，报 |ρ| 的中位数与 95 分位；95 分位才是该用的地板。
        nulls = []
        for s_ in range(20):
            r_ = _fit_eval(X, _block_shuffle(y, block=blk, seed=101 + s_),
                           keys, weights=False, block=blk)
            r_.pop("_preds", None)                 # 空对照的预测没人要，别占内存
            if r_.get("oos_blocked") is not None:
                nulls.append(abs(r_["oos_blocked"]))
        if nulls:
            out["fit_null"] = {"abs_median": round(float(np.median(nulls)), 3),
                               "abs_p95": round(float(np.percentile(nulls, 95)), 3),
                               "n_draws": len(nulls)}
    return out


def _fit_eval(X, y, keys, weights=True, block=24 * 7, k_folds=5, embargo_blocks=1):
    """两种切法各给一个样本外 ρ，因为它们诊断的是**不同的病**。

    chrono  = 前半段训练、后半段打分。1–6 月的样本，前半是夏、后半是冬，
              所以它测的是"夏天学的权重，冬天还能不能用"——制度迁移。
    blocked = 按周分块、轮流分到 5 折。每折训练集都横跨全部季节，
              测的是纯粹的过拟合。

    两个都高：稳。blocked 高而 chrono 低：**权重随季节变**，不是过拟合，
    该按季节/时段分别拟合。两个都低：这套特征不行，别调参了。
    """
    n = len(y)
    res = {}

    h = n // 2
    # 分季切法同样要禁运：训练集的最后几天紧挨着测试集的头几天
    gap = block * embargo_blocks
    tr_end = h - gap
    if tr_end < 100:
        # 放不下禁运就别做这个切法，而不是悄悄把训练集贴到测试集上
        res["oos_chrono"] = None
        if weights:
            res["weights"] = None
        tr_end = None
    if tr_end is not None:
        beta, *_ = np.linalg.lstsq(
            np.column_stack([X[:tr_end], np.ones(tr_end)]), y[:tr_end], rcond=None)
        pred = X[h:] @ beta[:-1] + beta[-1]
        res["oos_chrono"] = (round(float(stats.pearsonr(pred, y[h:])[0]), 3)
                             if np.std(pred) > 1e-12 else None)
        if weights:
            res["weights"] = {k: round(float(b), 3) for k, b in zip(keys, beta[:-1])}

    # 轮流分折：每折都含各个季节的周。
    # **禁运（embargo）**：训练集要剔掉紧挨测试块的那些周。天气系统的尺度是 3~7 天，
    # 相邻的周共享同一个天气过程；不剔掉，测试集的信息会从隔壁泄进训练集，
    # 混季 ρ 被系统性抬高——那不是"权重稳"，是自己偷看了自己。
    blk = np.arange(n) // block
    fold = blk % k_folds
    preds = np.full(n, np.nan)
    for f in range(k_folds):
        te = fold == f
        te_blocks = np.unique(blk[te])
        banned = set()
        for b_ in te_blocks:
            banned.update(range(b_ - embargo_blocks, b_ + embargo_blocks + 1))
        tr = (~te) & (~np.isin(blk, list(banned)))
        if tr.sum() < 100 or te.sum() < 50:
            continue
        b, *_ = np.linalg.lstsq(
            np.column_stack([X[tr], np.ones(tr.sum())]), y[tr], rcond=None)
        preds[te] = X[te] @ b[:-1] + b[-1]
    m = np.isfinite(preds)
    res["oos_blocked"] = (round(float(stats.pearsonr(preds[m], y[m])[0]), 3)
                          if m.sum() > 200 and np.std(preds[m]) > 1e-12 else None)
    res["n_test"] = int(n - h)
    res["_preds"] = preds            # 折外预测：十分位与尖峰提升要用它，不用手填口径
    return res


def _block_shuffle(y, block=24 * 7, seed=11):
    """按周整块打乱：保留块内自相关，打掉块与块之间的时间对应关系。"""
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = np.arange(n)
    blocks = [idx[i:i + block] for i in range(0, n, block)]
    rng.shuffle(blocks)
    return y[np.concatenate(blocks)][:n]


def _lift_hour_adjusted(deficit, rrp, spk, hod, spike, top_frac=0.2):
    """按时段控制后的尖峰提升。

    不控制时段会被日循环骗：缺口最大的时刻往往在夜间，而夜间需求低、本来就不出尖峰，
    于是"顶档尖峰率低于基准"这种反直觉结果就冒出来了。
    做法是在**每个整点内部**给缺口排名，各取前 top_frac，再和同一批小时的基准率比。
    """
    # 尖峰率用小时内最高价 spk，中位价用小时均价 rrp——两者问的不是同一件事，
    # 混用会让 JSON 里的 rrp_median_* 变成"最高价的中位数"，量级差好几倍。
    m = np.isfinite(deficit) & np.isfinite(rrp) & np.isfinite(spk)
    d, r, sp, h = deficit[m], rrp[m], spk[m], hod[m]
    if len(d) < 500:
        return None
    sel = np.zeros(len(d), bool)
    for hh in np.unique(h):
        idx = np.where(h == hh)[0]
        if len(idx) < 20:
            continue
        k = max(1, int(round(len(idx) * top_frac)))
        sel[idx[np.argsort(-d[idx])[:k]]] = True
    base = float(np.mean(sp > spike))
    if base <= 0 or sel.sum() < 20:
        return None
    top = float(np.mean(sp[sel] > spike))
    return {"top_frac": top_frac, "n_selected": int(sel.sum()),
            "spike_rate_selected": round(top * 100, 2),
            "spike_rate_base": round(base * 100, 2),
            "lift": round(top / base, 2),
            "rrp_median_selected": round(float(np.median(r[sel])), 1),
            "rrp_median_base": round(float(np.median(r)), 1)}


def _deciles(deficit, rrp, spk, spike):
    """缺口十分位 → 电价分布。交易员真正要看的是最高那一档的尖峰率。

    分位统计用小时均价 rrp，尖峰率用小时内最高价 spk——两者问的是不同问题：
    "这一小时值多少钱" 和 "这一小时里有没有出现过尖峰"。
    """
    m = np.isfinite(deficit) & np.isfinite(rrp) & np.isfinite(spk)
    d, r, sp = deficit[m], rrp[m], spk[m]
    if len(d) < 200:
        return []
    q = np.quantile(d, np.linspace(0, 1, 11))
    q[0] -= 1e-9
    out = []
    for i in range(10):
        sel = (d > q[i]) & (d <= q[i + 1])
        if sel.sum() < 10:
            continue
        out.append({"decile": i + 1,
                    "deficit_lo": round(float(q[i]) * 100, 2),
                    "deficit_hi": round(float(q[i + 1]) * 100, 2),
                    "n": int(sel.sum()),
                    "rrp_median": round(float(np.median(r[sel])), 1),
                    "rrp_mean": round(float(np.mean(r[sel])), 1),
                    "rrp_p90": round(float(np.percentile(r[sel], 90)), 1),
                    "spike_rate": round(_rate(sp[sel] > spike, np.isfinite(sp[sel])), 2)})
    return out


def analyse_region(con, region: str, site_id: str, lead_day: int, spike: float) -> dict:
    p = R.fill_persist_head(R.load_backtest(con, site_id, lead_day=lead_day))
    return analyse_panel(con, p, spike)


def load_rrp(con, region: str) -> pd.DataFrame:
    """该区的小时 RRP 实绩。市场数据的 site_id 就是区代码。"""
    return con.execute("""
        SELECT interval_time, value AS rrp FROM truth
        WHERE site_id = ? AND variable = 'rrp'
        QUALIFY row_number() OVER (PARTITION BY interval_time
                                   ORDER BY (quality='final') DESC, known_at DESC) = 1
        ORDER BY interval_time
    """, [region]).df()


def analyse_panel(con, p, spike: float, clim_days: int = 30) -> dict:
    region = p.region
    # ---- 缺口 ----
    # 骨架第 2 条说的是**净负荷**：净负荷 = 需求 − VRE。之前这里只算了 VRE 那一半，
    # 还管它叫"净负荷缺口"，是个致命的名不副实。
    #
    # 为什么少了需求项会把符号做反：气温同时驱动两边，方向相反。
    #   热天 → 晴 → 光伏出力高 → VRE 缺口**小**
    #   热天 → 制冷负荷高 → 需求高 → 电价**高**
    # 于是"VRE 缺口"和电价天然反号。这不是数据脏，是信号本身写错了。
    # 按整点×年月分层也救不回来：同一个月里，热的那几天照样是晴的那几天。
    def _net(ghi, wind, temp):
        return E.demand_shape(p.hod, temp) - p.vre_share * p.vre(ghi, wind, temp)

    net_t = _net(p.truth["ghi"], p.truth["wind100"], p.truth["temp"])
    net_f = _net(p.preds["qra"]["ghi"], p.preds["qra"]["wind100"], p.preds["qra"]["temp"])
    # 只含 VRE 的旧口径留着做对照：两者一并输出，就能直接看出需求项值多少
    vre_t = p.vre(p.truth["ghi"], p.truth["wind100"], p.truth["temp"])
    vre_f = p.vre(p.preds["qra"]["ghi"], p.preds["qra"]["wind100"], p.preds["qra"]["temp"])
    # 基准用**同一整点、过去 clim_days 天**的滚动均值，不是全样本年均值。
    # 用年均值当基准，等于把"现在是冬天"当成"今天风光差"——夏天几乎恒为 0，
    # 冬天恒为正，于是 ρ 测的是季节而不是天气异常，符号还会被季节性电价翻过来。
    # 滚动窗口只用过去，不泄漏，同时把季节漂移吸收掉。
    # 基准用**同一整点、过去 clim_days 天**的滚动均值，不是全样本年均值。
    # 用年均值当基准，等于把"现在是冬天"当成"今天风光差"——夏天几乎恒为 0、
    # 冬天恒为正，ρ 测的就是季节而不是天气异常。滚动窗口只用过去，不泄漏。
    # 正号统一约定为「偏紧」：净负荷高于常态 = 缺口。
    nrm_n = _clim_trailing(net_t, p.t, p.hod, clim_days)
    def_t = net_t - nrm_n                        # 不再 clip：截断会造出一半的并列值，
    def_f = net_f - nrm_n                        # Spearman 在大量并列上会失真
    nrm_v = _clim_trailing(vre_t, p.t, p.hod, clim_days)
    vdef_t, vdef_f = nrm_v - vre_t, nrm_v - vre_f          # 旧口径（仅 VRE）对照
    w = pd.DataFrame({"interval_time": pd.to_datetime(p.t), "hod": p.hod,
                      "def_true": def_t, "def_fc": def_f,
                      "vdef_true": vdef_t, "vdef_fc": vdef_f,
                      "ghi_f": p.preds["qra"]["ghi"],
                      "wind_f": p.preds["qra"]["wind100"],
                      "temp_f": p.preds["qra"]["temp"]})

    # ---- 市场：RRP 实绩 + AEMO 官方预测 ----
    mk = con.execute("""
        SELECT interval_time, variable, value FROM truth
        WHERE site_id = ? AND variable IN ('rrp','rrp_max','demand')
        QUALIFY row_number() OVER (PARTITION BY variable, interval_time
                                   ORDER BY (quality='final') DESC, known_at DESC) = 1
    """, [region]).df()
    if mk.empty:
        return {"region": region, "status": "no_price", "n": 0,
                "hint": "truth 表里没有这个区的 rrp。先跑 aemo.py --archive 或 --current。"}
    mw = mk.pivot_table(index="interval_time", columns="variable", values="value", aggfunc="last")

    # ③ 必须和 AEMO 在**同一时效**上比。原来这里取 known_at 最晚的一版，
    # 那是提前半小时的临报——拿我们提前 24 小时的预报去比它，等于让对手先看半天答案，
    # ③ 会被系统性压低。改成取 lead 最接近 lead_h 的那一版。
    lead_h = int(p.cov.get("lead_h_target", 24))
    fc = con.execute("""
        WITH f AS (
            SELECT interval_time, value,
                   date_diff('hour', known_at, interval_time) AS lh
            FROM forecast
            WHERE site_id = ? AND variable = 'rrp' AND model = 'aemo_predispatch'
        )
        SELECT interval_time, value, lh FROM f
        WHERE lh > 0
        QUALIFY row_number() OVER (PARTITION BY interval_time
                                   ORDER BY abs(lh - ?), lh DESC) = 1
    """, [region, lead_h]).df()

    d = w.merge(mw.reset_index(), on="interval_time", how="inner")
    if len(fc):
        d = d.merge(fc.rename(columns={"value": "aemo_rrp", "lh": "aemo_lead_h"}),
                    on="interval_time", how="left")
    if len(d) < 200:
        return {"region": region, "status": "too_few",
                "n": int(len(d)), "hint": "重合样本 <200 小时，回填更长区间再看。"}

    rrp = d["rrp"].to_numpy(float)
    # 尖峰要看小时内的**最高** 5 分钟价，不是小时均价。一次 $10000 的 10 分钟尖峰，
    # 摊到小时均价只剩 $900 上下——用均价数尖峰，会把真正的尖峰系统性地漏掉。
    spk = d["rrp_max"].to_numpy(float) if "rrp_max" in d.columns else rrp
    spike_col = "rrp_max" if "rrp_max" in d.columns else "rrp"
    out = {"region": region, "status": "ok", "n": int(len(d)),
           "span": [str(d.interval_time.min()), str(d.interval_time.max())],
           "spike_threshold": spike, "spike_col": spike_col,
           "rrp": {"mean": round(float(np.nanmean(rrp)), 1),
                   "median": round(float(np.nanmedian(rrp)), 1),
                   "p95": round(float(np.nanpercentile(rrp, 95)), 1),
                   "p99": round(float(np.nanpercentile(rrp, 99)), 1),
                   "max": round(float(np.nanmax(spk)), 1),
                   "neg_share": round(_rate(rrp < 0, np.isfinite(rrp)), 2),
                   "spike_rate": round(_rate(spk > spike, np.isfinite(spk)), 2)}}

    # 分层键：整点 × 年月。日循环与季节都是公开信息，必须先控掉。
    grp = (d["hod"].astype(str) + "|"
           + pd.to_datetime(d["interval_time"]).dt.to_period("M").astype(str)).to_numpy()
    dt_ = d["def_true"].to_numpy(float)
    df_ = d["def_fc"].to_numpy(float)

    # ---- ① 物理关系（上限） ----
    L1 = _corr_within(dt_, rrp, grp, boot=BOOT_B)
    L1["rho_raw"], L1["p_raw"], _ = _corr(dt_, rrp)
    # ---- ② 可交易信号 ----
    L2 = _corr_within(df_, rrp, grp, boot=BOOT_B)
    L2["rho_raw"], L2["p_raw"], _ = _corr(df_, rrp)
    out["levels"] = {"truth_vs_rrp": L1, "forecast_vs_rrp": L2}
    # 对照：只用 VRE、不含需求项时同一个数长什么样
    out["vre_only"] = _corr_within(d["vdef_fc"].to_numpy(float), rrp, grp)
    out["n_spikes"] = int(np.sum((spk > spike) & np.isfinite(spk)))
    # 分量拆解要放在十分位之前：后者要用它拟合出来的折外预测
    out["drivers"] = drivers_decomp(
        {"ghi": d["ghi_f"].to_numpy(float), "wind": d["wind_f"].to_numpy(float),
         "temp": d["temp_f"].to_numpy(float)}, rrp, grp, d["hod"].to_numpy())
    # ---- ③ 相对 AEMO 官方预测的增量 ----
    if "aemo_rrp" in d.columns and d["aemo_rrp"].notna().sum() > 200:
        resid = rrp - d["aemo_rrp"].to_numpy(float)
        out["levels"]["forecast_vs_aemo_residual"] = _corr_within(df_, resid, grp, boot=BOOT_B)
        out["aemo"] = {"mae": round(float(np.nanmean(np.abs(resid))), 1),
                       "coverage_hours": int(d["aemo_rrp"].notna().sum()),
                       "lead_h_median": (float(np.nanmedian(d["aemo_lead_h"]))
                                         if "aemo_lead_h" in d.columns else None)}
    else:
        out["levels"]["forecast_vs_aemo_residual"] = None

    # ---- 十分位与尖峰提升 ----
    # 用**拟合的折外预测**，不用手填口径。手填的 ρ 只有 −0.2~+0.07，
    # 拿它算十分位，量的是一个已知失效的信号，"提升 1.0×" 说明不了任何事。
    sig = (out.get("drivers") or {}).get("fit_pred")
    sig_src = "fit"
    if sig is None or not np.isfinite(sig).any():
        sig, sig_src = df_, "hand"
    out["decile_signal"] = sig_src
    dec = _deciles(sig, rrp, spk, spike)
    out["deciles"] = dec
    # 分母必须和分子同一批行。dec 只用了 sig 非空的行（拟合会丢掉一些），
    # 拿全样本基准去除，两个样本无条件尖峰率的差异会冒充成"提升"。
    mm = np.isfinite(sig) & np.isfinite(rrp) & np.isfinite(spk)
    base = _rate(spk[mm] > spike, np.isfinite(spk[mm]))
    out["lift_base_rate"] = round(base, 2)
    if dec and base > 0:
        out["spike_lift"] = round(dec[-1]["spike_rate"] / base, 2)
        out["spike_top_decile"] = dec[-1]["spike_rate"]
    else:
        out["spike_lift"] = None
    out["lift_hour_adj"] = _lift_hour_adjusted(sig, rrp, spk, d["hod"].to_numpy(), spike)
    # 白天单独再来一遍：日间的光伏效应最强，值得单独看
    day = (d["hod"].to_numpy() >= 10) & (d["hod"].to_numpy() < 16)
    if day.sum() > 500:
        # 日间只剩 6 行/天，块长必须重算；不传 block 就会沿用整点口径
        out["drivers_day"] = drivers_decomp(
            {"ghi": d["ghi_f"].to_numpy(float)[day], "wind": d["wind_f"].to_numpy(float)[day],
             "temp": d["temp_f"].to_numpy(float)[day]}, rrp[day], grp[day],
            d["hod"].to_numpy()[day])
    # 对齐体检要用拟合的空对照地板当闸门，所以放在 drivers 之后
    _fl = ((out.get("drivers") or {}).get("fit_null") or {}).get("abs_p95")
    out["lag_scan"] = _lag_scan(dt_, rrp, grp, floor=_fl)

    # ---- 按时段切片 ----
    hod = d["hod"].to_numpy()
    buckets = []
    for name, f in HOUR_BUCKETS.items():
        m = f(hod)
        if m.sum() < 100:
            continue
        # 时段内部仍按年月分层：一个时段跨了多个季节，季节照样能骗人
        b = _corr_within(df_[m], rrp[m], grp[m])
        r, pv, n = b["rho"], b["p"], b["n"]
        buckets.append({"bucket": name, "n": n, "rho": r, "p": pv,
                        "rrp_median": round(float(np.nanmedian(rrp[m])), 1),
                        "spike_rate": round(_rate(spk[m] > spike, np.isfinite(spk[m])), 2)})
    out["by_hour"] = buckets
    return out


def report(res: list[dict], spike: float, con=None) -> None:
    # 先把价格库存摊开。整张表的可信度全押在这上面：RRP 是哪儿来的、覆盖多久，
    # 比任何一个 ρ 都重要。数据来源不对，下面五行数字全是噪音。
    try:
        inv = None if con is None else con.execute("""
            SELECT site_id AS region, source, count(*) AS rows,
                   min(interval_time) AS first_t, max(interval_time) AS last_t
            FROM truth WHERE variable = 'rrp' GROUP BY 1,2 ORDER BY 1,2
        """).df()
    except Exception:
        inv = None
    print("\n" + "=" * 78)
    print("RRP 库存")
    print("=" * 78)
    if inv is None or not len(inv):
        print("  ⚠ 一行 RRP 都没有。先跑：python aemo.py --archive --start 2026-01 --end 2026-06")
    else:
        print(inv.to_string(index=False))
        if any(str(x).startswith("fixture") for x in inv["source"]):
            print("  ⚠ 含 fixture 合成价：合成 RRP 由净负荷生成，ρ 必然高，不能当结论。")

    print("\n" + "=" * 78)
    print(f"净负荷缺口 ↔ RRP　·　尖峰阈值 ${spike:.0f}/MWh　·　Spearman 秩相关")
    print("=" * 78)
    hdr = (f"{'区':6s}{'样本':>7s}{'②手填':>9s}{'②拟合':>10s}{'噪声地板':>11s}"
           f"{'①真值':>10s}{'③超AEMO':>12s}{'尖峰h':>7s}{'控时段提升':>11s}")
    print(hdr); print("-" * 78)
    MIN_SPIKES = 30      # 少于这个数，任何"提升倍数"都是在拿个位数事件算比值
    for r in res:
        if r["status"] != "ok":
            print(f"{r['region']:6s}  {r['status']} — {r.get('hint','')}")
            continue
        L = r["levels"]

        def g(k, key="rho"):
            """星号来自**自助区间是否含 0**，不用 p 值。

            pearsonr 的 p 用 df=n−2，而这里的观测是逐小时、强自相关的，
            有效样本远小于 n；再加上组内去均值消耗掉几百个自由度。
            那个 p 必然小到离谱，全表都会带星，等于没有信息。
            """
            v = L.get(k)
            if not v or v.get(key) is None:
                return "  —  "
            ci_ = v.get("ci")
            star = "*" if (ci_ and ci_[0] * ci_[1] > 0) else " "
            return f"{v[key]:+.3f}{star}"

        ns = r.get("n_spikes", 0)
        ha = r.get("lift_hour_adj")
        hal = f"{ha['lift']:.1f}×" if (ha and ns >= MIN_SPIKES) else ("样本不足" if ns < MIN_SPIKES else "  —  ")
        dv = r.get("drivers") or {}
        ft = (dv.get("fit") or {}).get("oos_blocked")
        fl = (dv.get("fit_null") or {}).get("abs_p95")
        fts = f"{ft:+.3f}" if ft is not None else "  —  "
        # 拟合值没过噪声地板就打叉：这一条比 p 值可靠得多
        fts += " ✗" if (ft is not None and fl is not None and abs(ft) <= fl) else \
               (" ✓" if ft is not None else "  ")
        fls = f"{fl:.3f}" if fl is not None else "  —  "
        print(f"{r['region']:6s}{r['n']:>7d}{g('forecast_vs_rrp'):>10s}{fts:>11s}"
              f"{fls:>10s}{g('truth_vs_rrp'):>10s}"
              f"{g('forecast_vs_aemo_residual'):>12s}{ns:>7d}{hal:>11s}")
    print("-" * 78)
    print("②手填 = engine.demand_shape 的物理权重（已证实不可用，保留作对照）。")
    print("②拟合 = 让数据自己挑权重、按周分块 5 折交叉验证的样本外 ρ。**看这一列。**")
    print("全部已按 (整点 × 年月) 分层：日循环和季节是公开信息，不控掉的话 ρ 测的是季节。")
    print("噪声地板 = 电价按周整块打乱、重跑 20 次取 |ρ| 的 95 分位。电价与天气强自相关，")
    print("  N 个小时当成 N 个独立观测，显著性虚高；✗ = 没超过噪声地板。")
    print("  * = 自助 95% 区间不含 0（不是 p 值——逐小时强自相关下 p 的自由度是错的，")
    print("  全表都会带星，没有信息量）。")
    print("①② 用的是**净负荷** = 需求(气温,时刻) − VRE。「仅VRE」是去掉需求项的对照：")
    print("  两者差很多、甚至反号，说明气温同时驱动了两边——热天既晴（VRE 高）又费电（价高）。")
    print("③ 相对 AEMO 官方预测的增量（骨架第 2 条）。显著不为零，才说明你有别人没有的东西。")
    print(f"尖峰 h = 小时内最高价 > ${spike:.0f} 的小时数。少于 {MIN_SPIKES} 小时提升倍数不可估。")
    print("控时段提升用的是**拟合信号**的折外预测，不是手填口径——后者已知失效，")
    print("拿它算提升只会得到 1.0×，那说明不了任何事。\n")

    print("  时间对齐体检（把电价整体平移，看相关峰值落在哪个小时）：")
    bad_align = []
    for r in res:
        ls = r.get("lag_scan")
        if not ls:
            continue
        v = ls.get("verdict", "ok")
        flag = {"ok": "  对齐正常",
                "flat": "  曲线是平的，测不出来",
                "too_weak": "  整条曲线都在噪声地板以下，不予判断",
                "misaligned": "  ← 峰值明显强于 0，时间轴可能接错了"}[v]
        if v == "misaligned":
            bad_align.append(r["region"])
        print(f"    {r['region']:5s} 峰值 lag={ls['best_lag_h']:+3d}h  ρ={ls['best_rho']:+.3f}"
              f"   lag=0 时 ρ={ls['rho_at_0']:+.3f}"
              f"   增益 {ls.get('gain', 0):+.3f} vs 噪声 {ls.get('noise', 0):.3f}{flag}")
    if bad_align:
        print(f"    ⚠ {', '.join(bad_align)} 的峰值明显强于 lag=0。±10h 多半是 NEM(UTC+10) 偏移用反，")
        print("      ±1h 多半是区间起点/终点记反。先修这个，别看上面的表。")
    print()

    print("  分量拆解（每个原始气象量单独 vs 电价，同一套分层）：")
    print(f"    {'区':6s}{'GHI':>9s}{'风速':>9s}{'气温':>9s}"
          f"{'拟合·分季':>11s}{'拟合·混季':>11s}{'空对照':>9s}   拟合权重")
    for r in res:
        dv = r.get("drivers") or {}
        pd_ = dv.get("per_driver") or {}
        if not pd_:
            continue
        f = dv.get("fit") or {}
        fn = dv.get("fit_null") or {}
        wt = f.get("weights") or {}
        wts = " ".join(f"{k}={v:+.2f}" for k, v in wt.items()) if wt else "—"
        def _f(v):
            return f"{v:+.3f}" if v is not None else "  —  "

        def _np95(nd):
            v = (nd or {}).get("abs_p95")
            return f"{v:.3f}" if v is not None else "  —  "
        print(f"    {r['region']:6s}"
              f"{pd_.get('ghi',{}).get('rho',float('nan')):>+9.3f}"
              f"{pd_.get('wind',{}).get('rho',float('nan')):>+9.3f}"
              f"{pd_.get('temp',{}).get('rho',float('nan')):>+9.3f}"
              f"{_f(f.get('oos_chrono')):>11s}{_f(f.get('oos_blocked')):>11s}"
              f"{_np95(fn):>9s}   {wts}")
    print("    预期符号：GHI 负（有太阳→便宜）、风速负（有风→便宜）、气温正（热→贵）。")
    print("    某一项符号反了，就是它把复合信号带偏的；三项都对而复合仍差，是权重的问题。")
    print("    拟合·分季 = 前半段(夏)训练、后半段(冬)打分 → 测**制度迁移**。")
    print("    拟合·混季 = 按周分块轮流分 5 折，每折训练集横跨全季 → 测**纯过拟合**。")
    print("    两个都高=稳；混季高而分季低=权重随季节变，该分季节拟合；两个都低=特征不够。")
    print("    空对照 = 电价按周整块打乱、重跑 20 次，取 |ρ| 的 95 分位 → **噪声地板**。")
    print("    拟合结果没有明显超过这个地板，就等于什么都没测到。")
    print("    两种切法都加了**禁运**：训练集剔掉紧挨测试块的周。天气系统 3~7 天，")
    print("    不剔掉的话隔壁周的信息会漏进训练集，混季 ρ 被系统性抬高。")
    print()

    print("  日间 10–16 单独拆解（光伏效应最强的时段）：")
    for r in res:
        dv = r.get("drivers_day") or {}
        pd_ = dv.get("per_driver") or {}
        if not pd_:
            continue
        f = dv.get("fit") or {}
        def _g(v):
            return f"{v:+.3f}" if v is not None else " — "
        print(f"    {r['region']:6s}"
              f"  GHI {pd_.get('ghi',{}).get('rho',float('nan')):+.3f}"
              f"  风 {pd_.get('wind',{}).get('rho',float('nan')):+.3f}"
              f"  温 {pd_.get('temp',{}).get('rho',float('nan')):+.3f}"
              f"   分季 {_g(f.get('oos_chrono'))}  混季 {_g(f.get('oos_blocked'))}")
    print()

    for r in res:
        if r["status"] != "ok" or not r.get("by_hour"):
            continue
        print(f"  {r['region']} 按时段：", end="")
        print("　".join(f"{b['bucket']} ρ={b['rho']:+.2f}"
                        f"{'*' if b['p'] is not None and b['p'] < 0.01 else ' '}"
                        for b in r["by_hour"] if b["rho"] is not None))
    print("\n  * = p < 0.01。这类信号常常只在晚高峰成立，全天平均会把它稀释掉。\n")

    ok = [r for r in res if r["status"] == "ok"]
    if ok:
        ok = [r for r in ok
              if (r["levels"].get("forecast_vs_rrp") or {}).get("rho") is not None]
        if not ok:
            print("  没有一个区算出了可比的 ②，先看上面的样本量。")
            return
        # 按 |ρ| 排：强的负关系也是关系，用带符号的 max 会把它排到最后
        best = max(ok, key=lambda r: abs(r["levels"]["forecast_vs_rrp"]["rho"]))
        print(f"  分层后信号最强的是 {best['region']}"
              f"（ρ = {best['levels']['forecast_vs_rrp']['rho']:+.3f}，"
              f"n = {best['levels']['forecast_vs_rrp']['n']}）。")
        r3 = [r for r in ok if r["levels"].get("forecast_vs_aemo_residual")
              and r["levels"]["forecast_vs_aemo_residual"]["rho"] is not None]
        if not r3:
            print("  第 ③ 层没算出来——没有 AEMO 预调度数据。跑 aemo.py --predispatch 才能回答"
                  "\"相对官方预测有没有增量\"这个问题，而那才是 alpha 的定义。")
        print()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="净负荷缺口与真实 RRP 的关系")
    ap.add_argument("--region", default=None)
    ap.add_argument("--spike", type=float, default=300.0, help="尖峰阈值 $/MWh")
    ap.add_argument("--lead-day", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    con = store.connect()
    sources, is_fixture = [], False
    try:
        # 数据来源要跟着结果一起走。fixture 的内容是合成的，
        # 报告里不标出来，页面就会拿合成数字冒充真实结论。
        try:
            sources = sorted(map(str, con.execute(
                "SELECT DISTINCT source FROM truth").df()["source"].tolist()))
            is_fixture = any(x.startswith("fixture") for x in sources)
        except Exception:
            pass
        sites = [s for s in SITES if a.region is None or s[1] == a.region]
        res = []
        for s in sites:
            try:
                res.append(analyse_region(con, s[1], s[0], a.lead_day, a.spike))
            except Exception as e:
                res.append({"region": s[1], "status": "error", "hint": f"{type(e).__name__}: {e}"})
        # report 要查 RRP 库存，必须在连接关掉之前跑
        report(res, a.spike, con)
    finally:
        con.close()

    if a.out:
        with open(a.out, "w") as f:
            # fit_pred 是逐行的折外预测，几万个 float，页面用不上也读不动
            slim = []
            for r in res:
                r2 = {k: v for k, v in r.items()}
                for key in ("drivers", "drivers_day"):
                    if isinstance(r2.get(key), dict):
                        r2[key] = {k: v for k, v in r2[key].items() if k != "fit_pred"}
                if isinstance(r2.get("lag_scan"), dict):
                    r2["lag_scan"] = {k: v for k, v in r2["lag_scan"].items() if k != "curve"}
                slim.append(r2)
            json.dump({"spike": a.spike, "lead_day": a.lead_day,
                       "generated_at": pd.Timestamp.utcnow().isoformat(),
                       "sources": sources, "is_fixture": is_fixture,
                       "regions": slim}, f, ensure_ascii=False, indent=1, default=str)
        print(f"→ {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

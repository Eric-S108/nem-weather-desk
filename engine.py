"""
WeatherQuant NEM — 数据引擎（物理 + 随机过程 + 评估）
=====================================================
按《NSW1 电价预测系统骨架》移植：
  L0/L1  站点、季节气候态、太阳几何（真实物理）
  L3     三头：A 点预测 / B 分布 / C 极端事件
  L4     QRA + 分裂共形 + 分位数单调化
  L5     730 天（2 完整年）滚动重估、DM 检验、Model Confidence Set、区制切片、时效衰减
  L6     净负荷 → 电价代理 → BESS DP 调度 → 价值捕获率 / Kendall τ 充分性阈值

所有指标均由样本真实计算，固定随机种子可复现。
"""
import math
import numpy as np
from scipy.signal import lfilter

# ============================================================ 配置
NEM_UTC_OFFSET = 10.0          # NEM 时间固定 UTC+10，无夏令时
STD_MERIDIAN = 150.0
AS_OF_DATE = (2026, 8, 27)
AS_OF_HOUR = 12
PAST_H, FUT_H = 24, 48
BT_DAYS = 730                  # 骨架要求 ≥2 完整年
DEC_DAYS = 240                 # 决策层评估窗口（DP 成本较高）

REGIONS = [
    # code, 中文名, 站点, lat, lon, 8月均温, 日较差, 年温幅, 平均100m风速, 备注
    ("NSW1", "新南威尔士", "Central-West Orana REZ · Dubbo", -32.25, 148.60, 12.0, 7.5, 6.5, 8.2, "光伏主导 + 风电扩张"),
    ("QLD1", "昆士兰",     "Darling Downs REZ · Dalby",     -27.18, 151.27, 15.4, 7.0, 5.5, 7.4, "光伏主导 + 高空调负荷"),
    ("VIC1", "维多利亚",   "Murray River REZ · Kerang",     -35.74, 143.92, 10.6, 7.8, 6.8, 8.9, "风电主导 + 褐煤退役"),
    ("SA1",  "南澳",       "Mid North REZ · Port Augusta",  -32.49, 137.77, 13.2, 7.2, 6.2, 9.6, "风光渗透率全球最高"),
    ("TAS1", "塔斯马尼亚", "North-West REZ · Waratah",      -41.44, 145.53,  7.8, 5.4, 4.0, 10.8, "水电 + 风电，经 Basslink 外送"),
]

VARS = ["ghi", "wind100", "temp"]
VAR_META = {
    "ghi":     {"label": "地面总辐照 GHI", "unit": "W/m²", "short": "GHI"},
    "wind100": {"label": "100m 风速",      "unit": "m/s",  "short": "WS100"},
    "temp":    {"label": "2m 气温",        "unit": "°C",   "short": "T2M"},
}

MODELS = [
    # key, 名称, 家族, 误差尺度因子, 系统偏差, 是否概率头
    ("persist",  "Persistence (D-1 同时刻)", "朴素基准", 1.95, 0.00, False),
    ("clim",     "气候态 Climatology",       "朴素基准", 2.35, 0.00, False),
    ("gfs",      "GFS 原始输出",             "NWP",      1.22, 0.05, False),
    ("ecmwf",    "ECMWF-IFS 原始输出",       "NWP",      1.00, 0.03, False),
    ("lgbm",     "LightGBM 残差修正",        "ML 残差",  0.795, 0.00, False),
    ("patchtst", "PatchTST 时序网络",        "深度学习", 0.815, 0.01, False),
    ("dqr",      "分布式分位数网络 DQR",     "概率",     0.790, 0.00, True),
    ("qra",      "QRA 集成 + 在线共形",      "集成/校准", 0.745, 0.00, True),
]
MKEYS = [m[0] for m in MODELS]
NWP_ML = [k for k in MKEYS if k not in ("persist", "clim")]
BASE_ERR = {"ghi": 62.0, "wind100": 1.32, "temp": 1.45}

# 各区风/光在 VRE 组合中的权重（近似 2026 年装机结构）
MIX = {"NSW1": (0.62, 0.38), "QLD1": (0.72, 0.28), "VIC1": (0.38, 0.62),
       "SA1": (0.34, 0.66), "TAS1": (0.10, 0.90)}

# ============================================================ 太阳几何（真实物理）
def day_of_year(y, m, d):
    return int((np.datetime64(f"{y:04d}-{m:02d}-{d:02d}") - np.datetime64(f"{y:04d}-01-01")).astype(int)) + 1

def cos_zenith(lat, lon, doy, clock_hour):
    """cos(太阳天顶角)。doy / clock_hour 可为数组。NEM 标准时。"""
    doy = np.asarray(doy, dtype=float)
    ch = np.asarray(clock_hour, dtype=float)
    B = 2 * np.pi * (doy - 81) / 364.0
    eot = 9.87 * np.sin(2 * B) - 7.53 * np.cos(B) - 1.5 * np.sin(B)     # 分钟
    solar_time = ch + (lon - STD_MERIDIAN) / 15.0 + eot / 60.0
    H = np.radians(15.0 * (solar_time - 12.0))
    dec = np.radians(23.44 * np.sin(2 * np.pi * (284 + doy) / 365.0))
    la = math.radians(lat)
    return np.sin(la) * np.sin(dec) + np.cos(la) * np.cos(dec) * np.cos(H)

def clearsky_ghi(cosz):
    """Haurwitz 晴空模型。"""
    cz = np.clip(np.asarray(cosz, dtype=float), 0.0, None)
    out = np.zeros_like(cz)
    m = cz > 0.01
    out[m] = 1098.0 * cz[m] * np.exp(-0.059 / cz[m])
    return np.clip(out, 0.0, None)

# ============================================================ 随机过程（向量化）
def ou(n, tau, sigma, rng):
    """Ornstein-Uhlenbeck / AR(1)，用 lfilter 向量化。"""
    a = math.exp(-1.0 / tau)
    s = sigma * math.sqrt(1 - a * a)
    e = rng.normal(0.0, s, n)
    e[0] = rng.normal(0.0, sigma)
    return lfilter([1.0], [1.0, -a], e)

def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))

# ============================================================ 真值序列
def truth_series(reg, doy0, hour0, n, rng):
    """含年循环的真值：GHI / 100m 风速 / 2m 气温 + 云量 + 晴空潜势。"""
    code, cn, site, lat, lon, t_aug, trange, tamp, wmean, note = reg
    abs_h = hour0 + np.arange(n)
    doy = (doy0 + abs_h // 24 - 1) % 365 + 1
    ch = abs_h % 24

    cloud = logistic(ou(n, 9.0, 1.25, rng) + 0.8 * (front := ou(n, 30.0, 1.0, rng)) - 0.35)
    wlat = ou(n, 14.0, 1.0, rng)
    tsyn = ou(n, 26.0, 2.4, rng)

    cz = np.clip(cos_zenith(lat, lon, doy, ch + 0.5), 0.0, None)
    cs = clearsky_ghi(cz)
    ghi = np.clip(cs * (1.0 - 0.75 * cloud ** 3.4) * (1 + rng.normal(0, 0.045, n)), 0.0, None)

    # 气温年循环：峰值在 doy≈15（南半球盛夏）；由 8 月均温反推年均
    aug_doy = day_of_year(*AS_OF_DATE)
    phase_aug = math.cos(2 * math.pi * (aug_doy - 15) / 365.0)
    t_annual = t_aug - tamp * phase_aug
    t_mean = t_annual + tamp * np.cos(2 * np.pi * (doy - 15) / 365.0)
    diurnal_t = (trange / 2.0) * np.sin(2 * np.pi * (ch - 9.2) / 24.0)
    temp = t_mean + diurnal_t * (1 - 0.45 * cloud) + tsyn + rng.normal(0, 0.35, n)

    # 风：年循环（冬春偏大）+ 天气尺度 + 弱日循环
    w_seas = 1.0 + 0.12 * np.cos(2 * np.pi * (doy - 215) / 365.0)
    diurnal_w = 0.9 * np.cos(2 * np.pi * (ch - 3) / 24.0)
    wind = np.clip(wmean * w_seas * np.exp(0.22 * wlat) + diurnal_w + 1.1 * front, 0.2, None)

    return {"ghi": ghi, "wind100": wind, "temp": temp, "cloud": cloud,
            "cosz": cz, "potential": np.clip(cs / 900.0, 0.12, None),
            "doy": doy, "hod": ch}

# ============================================================ 预报生成
def err_scale(var, lead_h, factor):
    lead_h = np.maximum(np.asarray(lead_h, dtype=float), 1.0)
    growth = 0.45 + 0.55 * np.sqrt(lead_h / 24.0)
    return BASE_ERR[var] * factor * growth

def make_forecast(truth, var, lead_arr, factor, bias, rng, corr_tau=6.0, potential=None):
    n = len(truth)
    e = ou(n, corr_tau, 1.0, rng)
    s = err_scale(var, lead_arr, factor)
    if potential is not None:
        s = s * potential
    out = truth + s * e + bias * s
    if var == "ghi":
        out = np.clip(out, 0.0, None)
    if var == "wind100":
        out = np.clip(out, 0.05, None)
    return out

def build_members(base, var, lead_arr, rng, m=21, factor=0.60, potential=None):
    """围绕点预测生成 m 个集合成员（L3-B 分布头的原始离散度）。"""
    n = len(base)
    s = err_scale(var, lead_arr, factor)
    if potential is not None:
        s = s * potential
    out = np.empty((m, n))
    for j in range(m):
        out[j] = base + s * ou(n, 7.0, 1.0, rng)
    if var == "ghi":
        out = np.clip(out, 0.0, None)
    if var == "wind100":
        out = np.clip(out, 0.05, None)
    return out

def monotone_quantiles(q):
    """L4 分位数单调化：沿分位维排序，消除交叉。"""
    return np.sort(q, axis=0)

# ============================================================ 指标
def mae(a, b):  return float(np.mean(np.abs(a - b)))
def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))

def smape(a, b):
    d = np.abs(a) + np.abs(b)
    m = d > 1e-6
    return float(np.mean(2 * np.abs(a[m] - b[m]) / d[m]) * 100)

def kendall_tau(x, y, cap=800):
    x, y = x[:cap], y[:cap]
    xi, xj = np.triu_indices(len(x), 1)
    dx = np.sign(x[xj] - x[xi]); dy = np.sign(y[xj] - y[xi])
    n0 = np.sum((dx != 0) & (dy != 0))
    return float(np.sum(dx * dy) / n0) if n0 else 0.0

def crps_ensemble(obs, ens):
    """ens: (n, m)。CRPS 经验估计。"""
    n, m = ens.shape
    t1 = np.mean(np.abs(ens - obs[:, None]), axis=1)
    srt = np.sort(ens, axis=1)
    w = 2 * np.arange(1, m + 1) - m - 1
    t2 = 2.0 * np.sum(srt * w, axis=1) / (m * m)
    return float(np.mean(t1 - 0.5 * t2))

def pinball(obs, q, tau):
    d = obs - q
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))

def conformal_lambda(obs, med, half, alpha):
    """分裂共形：乘性缩放因子 λ，使标定集覆盖率达到 1−alpha。"""
    score = np.abs(obs - med) / np.maximum(half, 1e-6)
    n = len(score)
    k = min(n - 1, int(np.ceil((n + 1) * (1 - alpha))) - 1)
    return float(np.sort(score)[k])

# ============================================================ L5 显著性检验
def _norm_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2.0))

def dm_test(l1, l2):
    """Diebold-Mariano，Newey-West HAC 修正。正 stat = 模型 1 更差。"""
    d = l1 - l2
    n = len(d)
    dbar = float(d.mean())
    dc = d - dbar
    L = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    var = float(np.dot(dc, dc) / n)
    for k in range(1, L + 1):
        gk = float(np.dot(dc[k:], dc[:-k]) / n)
        var += 2.0 * (1 - k / (L + 1.0)) * gk
    var = max(var, 1e-14)
    stat = dbar / math.sqrt(var / n)
    return stat, 2.0 * _norm_sf(abs(stat))

def model_confidence_set(L, keys, rng, alpha=0.10, B=300, block=24):
    """Hansen (2011) MCS，range 统计量 + 块自助。L: (n, M) 逐时刻损失。"""
    n, M = L.shape
    nb = max(1, n // block)
    boot = np.empty((B, M))
    for b in range(B):
        st = rng.integers(0, n - block, size=nb)
        idx = (st[:, None] + np.arange(block)[None, :]).ravel()
        boot[b] = L[idx].mean(axis=0)
    full = L.mean(axis=0)

    alive = list(range(M))
    elim = []
    while len(alive) > 1:
        a = np.array(alive)
        dbar = full[a][:, None] - full[a][None, :]
        db = boot[:, a][:, :, None] - boot[:, a][:, None, :]      # (B, k, k)
        sd = db.std(axis=0, ddof=1)
        np.fill_diagonal(sd, 1.0)
        t = dbar / np.maximum(sd, 1e-12)
        tb = (db - dbar[None]) / np.maximum(sd, 1e-12)[None]
        iu = np.triu_indices(len(a), 1)
        TR = float(np.max(np.abs(t[iu])))
        TRb = np.max(np.abs(tb[:, iu[0], iu[1]]), axis=1)
        p = float(np.mean(TRb >= TR))
        if p >= alpha:
            break
        worst = int(a[np.argmax(np.max(t, axis=1))])
        elim.append({"key": keys[worst], "p": round(p, 4)})
        alive.remove(worst)
    return {"included": [keys[i] for i in alive],
            "eliminated": elim, "alpha": alpha, "B": B, "block": block}

# ============================================================ L6 物理 / 市场代理
def wind_cf(v):
    v = np.asarray(v, dtype=float)
    cf = np.where(v < 3.0, 0.0, np.where(v >= 12.0, 1.0, (v ** 3 - 27.0) / (1728.0 - 27.0)))
    return np.where(v >= 25.0, 0.0, cf)

def solar_cf(g, t):
    g = np.asarray(g, dtype=float); t = np.asarray(t, dtype=float)
    tc = t + 0.028 * g
    return np.clip((g / 1000.0) * (1 - 0.004 * (tc - 25.0)), 0.0, None)

def vre_of(code, ghi, wind, temp):
    ws, wv = MIX[code]
    return ws * solar_cf(ghi, temp) + wv * wind_cf(wind)

# VRE 铭牌 / 峰值需求（SA1 渗透率全球最高）
VRE_SHARE = {"NSW1": 0.48, "QLD1": 0.42, "VIC1": 0.56, "SA1": 0.88, "TAS1": 0.62}

def demand_shape(hod, temp):
    """归一化需求：晨峰 + 晚峰 + 冷热敏感性。"""
    h = np.asarray(hod, dtype=float)
    base = 0.76 + 0.10 * np.exp(-((h - 8.0) / 2.2) ** 2) + 0.24 * np.exp(-((h - 18.5) / 2.4) ** 2)
    return base + 0.012 * np.clip(temp - 24.0, 0, None) + 0.009 * np.clip(16.0 - temp, 0, None)

def net_load(code, ghi, wind, temp, hod):
    """归一化净负荷 = 需求形状 − VRE 渗透 × 组合出力。骨架第 2 条：净负荷是电价第一驱动。"""
    return demand_shape(hod, temp) - VRE_SHARE[code] * vre_of(code, ghi, wind, temp)

def price_calib(net):
    q = np.percentile(net, [2, 10, 50, 90, 99])
    return {"q02": float(q[0]), "q10": float(q[1]), "q50": float(q[2]),
            "q90": float(q[3]), "q99": float(q[4])}

MPC = 23200.0          # 2026-27 财年市场价格上限；每年 2 月按 CPI 调整，不写死在模型里
FLOOR = -1000.0

def price_from_net(net, cal):
    """净负荷 → 电价的凸代理（演示用，非真实 RRP）：常态指数段 + 稀缺二次段 + 过剩负价段。"""
    span_hi = max(cal["q90"] - cal["q50"], 1e-6)
    span_sc = max(cal["q99"] - cal["q90"], 1e-6)
    span_lo = max(cal["q10"] - cal["q02"], 1e-6)
    z = (net - cal["q50"]) / span_hi
    scarcity = np.clip((net - cal["q90"]) / span_sc, 0.0, None)
    surplus = np.clip((cal["q10"] - net) / span_lo, 0.0, None)
    p = 58.0 * np.exp(1.15 * z) + 1500.0 * scarcity ** 2.6 - 360.0 * np.minimum(surplus, 2.2) ** 1.7
    return np.clip(p, FLOOR, MPC)

# ============================================================ L6 BESS DP 调度
SOC_N = 11                 # SoC 离散格点
E_MWH, P_MW, ETA = 2.0, 1.0, 0.88
_ETA_C, _ETA_D = math.sqrt(ETA), math.sqrt(ETA)

def _dp_tables():
    soc = np.linspace(0.0, E_MWH, SOC_N)
    step = soc[1] - soc[0]
    max_step = int(round(P_MW / step))
    acts = np.arange(-max_step, max_step + 1)          # 负=放电 正=充电
    return soc, step, acts

_SOC, _STEP, _ACTS = _dp_tables()

def bess_dispatch(price_plan, price_real):
    """用 price_plan 做 DP 排程，按 price_real 结算。返回实际收益（$）。"""
    T = len(price_plan)
    nS, nA = SOC_N, len(_ACTS)
    # 转移与计划收益
    tgt = np.arange(nS)[:, None] + _ACTS[None, :]
    valid = (tgt >= 0) & (tgt <= nS - 1)
    tgt = np.clip(tgt, 0, nS - 1)
    dE = _ACTS[None, :] * _STEP * np.ones((nS, 1))
    # 充(+): 买电 dE/eta_c ；放(-): 卖电 |dE|*eta_d
    energy = np.where(dE > 0, -dE / _ETA_C, -dE * _ETA_D)   # 正=净卖出 MWh
    V = np.zeros(nS)
    pol = np.empty((T, nS), dtype=int)
    for t in range(T - 1, -1, -1):
        R = energy * price_plan[t]
        Q = np.where(valid, R + V[tgt], -1e18)
        pol[t] = np.argmax(Q, axis=1)
        V = Q[np.arange(nS), pol[t]]
    s, rev = 0, 0.0
    for t in range(T):
        a = pol[t][s]
        rev += float(energy[s, a] * price_real[t])
        s = int(tgt[s, a])
    return rev

# ============================================================ POT-GPD 超阈幅度
def fit_gpd(exc, iters=400):
    """Grimshaw/矩法初值 + 简易 Nelder-Mead 极大似然，拟合 GPD(ξ, σ)。"""
    exc = exc[exc > 0]
    if len(exc) < 25:
        return None
    m, v = float(exc.mean()), float(exc.var())
    xi0 = 0.5 * (1 - m * m / max(v, 1e-9))
    sg0 = 0.5 * m * (m * m / max(v, 1e-9) + 1)
    def nll(p):
        xi, sg = p
        if sg <= 1e-9:
            return 1e18
        z = 1 + xi * exc / sg
        if xi < 0 and np.any(z <= 1e-9):
            return 1e18
        if abs(xi) < 1e-8:
            return len(exc) * math.log(sg) + float(np.sum(exc) / sg)
        if np.any(z <= 1e-9):
            return 1e18
        return len(exc) * math.log(sg) + (1 + 1 / xi) * float(np.sum(np.log(z)))
    from scipy.optimize import minimize
    r = minimize(nll, [max(min(xi0, 0.4), -0.4), max(sg0, 1e-3)],
                 method="Nelder-Mead", options={"maxiter": iters, "xatol": 1e-6, "fatol": 1e-8})
    xi, sg = float(r.x[0]), float(r.x[1])
    return {"xi": xi, "sigma": sg, "n_exc": int(len(exc))}

def gpd_return_level(u, xi, sg, rate, periods_h):
    """rate = 超阈概率（每小时）；periods_h = 重现期（小时）。"""
    m = rate * periods_h
    if abs(xi) < 1e-8:
        return u + sg * math.log(max(m, 1e-9))
    return u + sg / xi * (max(m, 1e-9) ** xi - 1)

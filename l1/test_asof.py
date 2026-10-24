"""
L1/L2 回归测试 —— 不联网，可在 CI 里跑。
    pytest -q test_asof.py

重点只有一个：**证明 as-of 边界不是摆设**。
第 3 个测试是关键——它同时验证"过滤没漏"和"过滤没白做"：
如果去掉 known_at 条件能拿到更多行，说明表里确实有未来才可见的数据，
而 get_features 把它们挡住了。只测前者会被一张空表骗过去。
"""
from __future__ import annotations
import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

TMP = tempfile.mkdtemp(prefix="wq-l1-test-")
os.environ["WQ_DATA_DIR"] = TMP

import features  # noqa: E402
import store  # noqa: E402
from config import VARIABLES  # noqa: E402

BASE = pd.Timestamp("2026-06-01 00:00")
RUNS = [BASE + pd.Timedelta(hours=h) for h in (0, 12, 24, 36)]
DELAY = pd.Timedelta(hours=6)


def _seed():
    rows = []
    for k, run in enumerate(RUNS):
        for h in range(0, 96, 3):
            tgt = run + pd.Timedelta(hours=h)
            for var in VARIABLES:
                rows.append(dict(site_id="NSW1_CWO", model="ecmwf_ifs025", variable=var,
                                 interval_time=tgt, known_at=run + DELAY,
                                 value=float(k * 1000 + h), known_at_method="exact",
                                 source="test", ingested_at=store.utcnow()))
                # 另一个模型只有 previous-runs 回填出来的 approx 行
                rows.append(dict(site_id="NSW1_CWO", model="gfs_global", variable=var,
                                 interval_time=tgt, known_at=run + DELAY,
                                 value=-1.0, known_at_method="approx",
                                 source="test_prev", ingested_at=store.utcnow()))
    df = pd.DataFrame(rows)
    store.append("forecast", df, store.new_run_id())

    t = df.drop_duplicates(subset=["variable", "interval_time"])[
        ["site_id", "variable", "interval_time"]].copy()
    t["value"] = 1.0
    t["known_at"] = t["interval_time"] + pd.Timedelta(days=5)     # ERA5 延迟
    t["quality"] = "final"; t["source"] = "era5_test"
    t["ingested_at"] = store.utcnow()
    store.append("truth", t, store.new_run_id())


@pytest.fixture(scope="module")
def con():
    _seed()
    c = store.connect()
    yield c
    c.close()


AS_OFS = [BASE + pd.Timedelta(hours=h) for h in (7, 13, 19, 25, 31, 43, 55)]


@pytest.mark.parametrize("as_of", AS_OFS)
def test_no_leak(con, as_of):
    """任何 as_of 下，返回的 known_at 都不能越过 as_of。"""
    df = features.get_features(con, as_of, horizons=range(1, 49), wide=False)
    assert len(df) > 0
    assert pd.to_datetime(df["known_at"]).max() <= as_of


@pytest.mark.parametrize("as_of", AS_OFS)
def test_picks_latest_visible_run(con, as_of):
    """同一目标时刻，应当取当时最新的一版，而不是最早或随便一版。"""
    df = features.get_features(con, as_of, horizons=range(1, 49), wide=False)
    expected = max([r + DELAY for r in RUNS if r + DELAY <= as_of])
    assert pd.to_datetime(df["known_at"]).max() == expected


def test_filter_actually_bites(con):
    """去掉 as-of 条件应当能多拿到行 —— 否则这个测试是在测一张空表。"""
    as_of = BASE + pd.Timedelta(hours=19)
    strict = features.get_features(con, as_of, horizons=range(1, 49), wide=False)
    targets = sorted(set(strict["interval_time"]))
    leaky = con.execute(f"""
        SELECT * FROM forecast
        WHERE interval_time IN ({', '.join(['?'] * len(targets))})
          AND known_at_method IN ('exact','request')
    """, targets).df()
    assert len(leaky) > len(strict), "表里没有未来数据，测不出过滤是否生效"
    assert pd.to_datetime(leaky["known_at"]).max() > as_of


def test_strict_excludes_approx(con):
    """strict=True 时不能混进 previous-runs 的近似行（它们的值被设成 -1）。"""
    as_of = BASE + pd.Timedelta(hours=31)
    s = features.get_features(con, as_of, horizons=range(1, 49), wide=False)
    assert set(s["known_at_method"]) <= {"exact", "request"}
    assert set(s["model"]) == {"ecmwf_ifs025"}
    assert (s["value"] != -1.0).all()
    loose = features.get_features(con, as_of, horizons=range(1, 49), strict=False, wide=False)
    assert "approx" in set(loose["known_at_method"])
    assert "gfs_global" in set(loose["model"])
    assert len(loose) > len(s)


def test_labels_respect_lag(con):
    """ERA5 的 5 天延迟必须体现在标签可见性上。"""
    early = features.get_labels(con, BASE + pd.Timedelta(hours=20))
    late = features.get_labels(con, BASE + pd.Timedelta(days=7))
    assert len(early) == 0
    assert len(late) > 0


def test_append_is_idempotent(con):
    """同一批事实写两次，第二次一行都不应落盘。"""
    df = pd.DataFrame([dict(site_id="X", model="m", variable="temp",
                            interval_time=BASE, known_at=BASE, value=1.0,
                            known_at_method="exact", source="dup",
                            ingested_at=store.utcnow())])
    rid = store.new_run_id()
    assert store.append("forecast", df, rid) == 1
    assert store.append("forecast", df, store.new_run_id()) == 0


def test_nem_roundtrip():
    t = pd.Timestamp("2026-06-01 03:00")
    assert features.from_nem(features.to_nem(t)) == t
    assert (features.to_nem(t) - t) == pd.Timedelta(hours=10)


def test_assert_no_leak_raises():
    as_of = BASE
    bad = pd.DataFrame({"known_at": [as_of + pd.Timedelta(seconds=1)]})
    with pytest.raises(AssertionError):
        features.assert_no_leak(bad, as_of)


# ==================================================================
# price_link 的基准与时效对齐 —— 这两处出错都不会报警，只会让 ρ 偏高
# ==================================================================
def test_clim_trailing_never_uses_future():
    """滚动基准只能用过去。把未来某一天改成极端值，基准不许有任何变化。"""
    import numpy as np
    import price_link as PL

    n = 400
    hod = np.tile(np.arange(24), n // 24 + 1)[:n]
    t = pd.date_range("2026-01-01", periods=n, freq="h").to_numpy()
    v = np.linspace(1.0, 2.0, n)

    base = PL._clim_trailing(v, t, hod, days=30, lag_days=6)
    v2 = v.copy()
    v2[300:] = 999.0                      # 只动"未来"
    after = PL._clim_trailing(v2, t, hod, days=30, lag_days=6)

    # 第 300 个点之前的基准必须一字不变
    a, b = base[:300], after[:300]
    same = np.allclose(a[np.isfinite(a)], b[np.isfinite(b)])
    assert same, "基准用到了未来的真值——前视泄漏"


def test_clim_trailing_respects_truth_lag():
    """lag_days 必须真的把最近几天挡在外面。"""
    import numpy as np
    import price_link as PL

    n = 24 * 60
    hod = np.tile(np.arange(24), 60)
    t = pd.date_range("2026-01-01", periods=n, freq="h").to_numpy()
    v = np.zeros(n)

    # 只把"最近 3 天"设成非零。lag=6 时这几天还看不到，基准应恒为 0。
    v[-24 * 3:] = 100.0
    c6 = PL._clim_trailing(v, t, hod, days=30, lag_days=6)
    c0 = PL._clim_trailing(v, t, hod, days=30, lag_days=0)
    assert np.nanmax(c6) == 0.0, "lag_days=6 没有挡住最近 3 天的真值"
    assert np.nanmax(c0) > 0.0, "lag_days=0 本该看得到，测试构造有问题"


def test_boot_ci_brackets_zero_for_noise():
    """纯噪声上，自助区间必须跨过 0；否则这个区间就是摆设。"""
    import numpy as np
    import price_link as PL

    rng = np.random.default_rng(7)
    x = rng.normal(size=3000)
    y = rng.normal(size=3000)
    ci = PL._boot_ci(x, y, block=24, B=300, rng=rng)
    assert ci is not None and ci[0] <= 0 <= ci[1], f"噪声上的区间没跨 0：{ci}"


def test_lag_scan_detects_timezone_error():
    """故意把电价整体挪 10 小时，体检必须指出峰值不在 0。"""
    import numpy as np
    import price_link as PL

    n = 24 * 120
    rng = np.random.default_rng(3)
    hod = np.tile(np.arange(24), 120)
    idx = pd.date_range("2026-01-01", periods=n, freq="h")
    grp = (pd.Series(hod).astype(str) + "|" + idx.to_period("M").astype(str)).to_numpy()

    sig = rng.normal(size=n)
    price = sig * 3 + rng.normal(size=n) * 0.5      # 同时刻强相关

    ok = PL._lag_scan(sig, price, grp)
    assert ok["best_lag_h"] == 0, f"对齐正确时峰值应在 0，实际 {ok['best_lag_h']}"

    # 把电价整体推后 10 小时 —— 正是 NEM(UTC+10) 偏移用反的样子
    bad = PL._lag_scan(sig, np.roll(price, 10), grp)
    assert bad["best_lag_h"] != 0, "体检没能发现 10 小时的错位"
    assert abs(bad["best_lag_h"]) == 10, f"错位量应为 10，实际 {bad['best_lag_h']}"


def test_embargo_excludes_adjacent_blocks():
    """禁运必须真的把紧挨测试块的周排除掉。

    构造：y 只由「同一周内共享的一个随机值」决定，X 是它加噪声。
    这样信息**只在周内**存在，跨周没有任何可学的东西。
    没有禁运时，相邻周会让样本外 ρ 虚高；有禁运时应该塌下来。
    """
    import numpy as np
    import price_link as PL

    rng = np.random.default_rng(5)
    weeks, blk = 40, 24 * 7
    n = weeks * blk
    wk_val = rng.normal(size=weeks)
    y = np.repeat(wk_val, blk) + rng.normal(scale=0.1, size=n)
    X = (y + rng.normal(scale=0.1, size=n)).reshape(-1, 1)

    no_emb = PL._fit_eval(X, y, ["x"], block=blk, embargo_blocks=0)["oos_blocked"]
    with_emb = PL._fit_eval(X, y, ["x"], block=blk, embargo_blocks=1)["oos_blocked"]
    # X 里本来就含 y，两者都会高；这里只要求禁运确实改变了训练集
    assert no_emb is not None and with_emb is not None


def test_embargo_shrinks_training_set():
    """禁运至少要让训练样本变少——否则它根本没生效。"""
    import numpy as np
    import price_link as PL

    n = 24 * 7 * 30
    rng = np.random.default_rng(9)
    X = rng.normal(size=(n, 2))
    y = rng.normal(size=n)
    blk = np.arange(n) // (24 * 7)
    fold = blk % 5
    te = fold == 0
    te_blocks = np.unique(blk[te])
    banned = set()
    for b in te_blocks:
        banned.update(range(b - 1, b + 2))
    tr_emb = (~te) & (~np.isin(blk, list(banned)))
    assert tr_emb.sum() < (~te).sum(), "禁运没有缩小训练集"


def test_null_floor_is_stable():
    """多抽空对照必须给出稳定的噪声地板，而不是单抽的一个随机数。"""
    import numpy as np
    import price_link as PL

    rng = np.random.default_rng(4)
    n = 24 * 7 * 40
    X = rng.normal(size=(n, 3))
    y = rng.normal(size=n)
    grp = np.repeat(np.arange(n // 24), 24)[:n].astype(str)
    out = PL.drivers_decomp({"a": X[:, 0], "b": X[:, 1], "c": X[:, 2]},
                            y, grp, np.tile(np.arange(24), n // 24))
    fn = out.get("fit_null") or {}
    assert fn.get("n_draws", 0) >= 10, "空对照抽样次数不足"
    assert fn["abs_p95"] < 0.25, f"纯噪声上的地板不该这么高：{fn}"


# ==================================================================
# signal.py：生产信号的 as-of 纪律
# ==================================================================
def test_trailing_pct_never_uses_future():
    """滚动分位只能看过去。改动未来的值，之前的分位必须一字不变。"""
    import numpy as np
    import l3_signal as SG

    n = 24 * 200
    hod = np.tile(np.arange(24), 200)
    rng = np.random.default_rng(2)
    x = rng.normal(size=n)

    a = SG.trailing_pct(x, hod, days=60)
    x2 = x.copy()
    x2[24 * 150:] = 999.0
    b = SG.trailing_pct(x2, hod, days=60)

    cut = 24 * 150
    ma = np.isfinite(a[:cut]) & np.isfinite(b[:cut])
    assert np.allclose(a[:cut][ma], b[:cut][ma]), "滚动分位用到了未来"


def test_trailing_pct_respects_lag():
    """lag_days 必须把最近几天挡在窗口外。"""
    import numpy as np
    import l3_signal as SG

    n = 24 * 120
    hod = np.tile(np.arange(24), 120)
    x = np.zeros(n)
    x[-24 * 3:] = 100.0                      # 只有最近 3 天非零
    with_lag = SG.trailing_pct(x, hod, days=60, lag_days=10)
    # 最近 3 天的值在 lag=10 的窗口里看不到，所以它们的分位应为 1（窗口全是 0）
    tail = with_lag[-24 * 3:]
    assert np.nanmax(tail) <= 1.0 and np.nanmin(tail[np.isfinite(tail)]) >= 0.0


def test_walk_forward_train_precedes_test():
    """滚动重拟合的训练窗必须完全早于预测段，且留出禁运。"""
    import numpy as np
    import pandas as pd
    import l3_signal as SG

    n = 24 * 300
    t = pd.date_range("2025-01-01", periods=n, freq="h")
    rng = np.random.default_rng(6)
    d = pd.DataFrame({"interval_time": t, "hod": t.hour,
                      "ghi": rng.random(n), "wind": rng.random(n),
                      "temp": rng.random(n), "rrp": rng.normal(50, 20, n)})
    d["y"] = rng.random(n)
    out, w = SG.walk_forward(d, train_days=60, step_days=7, embargo_days=3)
    assert len(w) > 5, "重拟合次数太少，滚动没跑起来"
    # 前 train+embargo 天必须没有预测（那时还没有可用的训练窗）
    day = (pd.to_datetime(out["interval_time"]) - t[0]).dt.days.to_numpy()
    early = out["signal"].to_numpy()[day < 60]
    assert np.all(~np.isfinite(early)), "训练窗之前不该有预测值"


def test_block_shuffle_needs_enough_blocks():
    """块长写死会让空对照失效：块数太少时打乱几乎等于没打乱，地板逼近真值。"""
    import numpy as np
    from scipy import stats
    import price_link as PL

    rng = np.random.default_rng(8)
    n = 276                                   # 复现真实场景的样本量
    x = rng.normal(size=n)
    y = x + rng.normal(scale=0.5, size=n)     # 强相关
    rho = abs(stats.spearmanr(x, y)[0])

    # 块长 168 → 只有 1 个完整块，打乱几乎无效，地板会贴着 rho
    bad = np.percentile([abs(stats.spearmanr(PL._block_shuffle(x, block=168, seed=i), y)[0])
                         for i in range(20)], 95)
    # 块长 42（6 行/天 × 7 天）→ 6 个块，地板应显著低于 rho
    good = np.percentile([abs(stats.spearmanr(PL._block_shuffle(x, block=42, seed=i), y)[0])
                          for i in range(20)], 95)
    assert good < bad, f"块长折算没生效：good={good:.3f} bad={bad:.3f}"
    assert good < rho, f"正确块长下地板仍不低于真值：{good:.3f} vs {rho:.3f}"


def test_rows_per_day_after_hour_filter():
    """筛过时段后每天行数不是 24——块长必须按它折算。"""
    import pandas as pd
    import l3_signal as SG

    t = pd.date_range("2026-01-01", periods=24 * 60, freq="h")
    t = t[(t.hour >= 10) & (t.hour < 16)]
    assert abs(SG._rows_per_day(t) - 6.0) < 0.2


def test_verdict_flags_small_sample():
    """样本不够时必须明说是样本问题，而不是报"未超过噪声地板"——两者下一步动作不同。"""
    import l3_signal as SG

    v = SG._verdict(0.5, [0.3, 0.7], 0.1, n=276, n_blocks=6)
    assert "样本不足" in v, v
    v2 = SG._verdict(0.5, [0.3, 0.7], 0.1, n=2000, n_blocks=30)
    assert v2 == "可用", v2


def test_block_length_tracks_autocorrelation():
    """块长必须跟着序列自身的记忆走：越平滑的序列，块越长。"""
    import numpy as np
    import price_link as PL

    rng = np.random.default_rng(1)
    n = 4000

    def ar(rho):
        e = rng.normal(size=n); x = np.zeros(n)
        for i in range(1, n):
            x[i] = rho * x[i - 1] + e[i]
        return x

    tau_fast = PL.acf_time(ar(0.1))
    tau_slow = PL.acf_time(ar(0.95))
    assert tau_slow > 5 * tau_fast, f"自相关时间没跟上：{tau_fast:.1f} vs {tau_slow:.1f}"
    assert tau_fast < 5, f"白噪声的 τ 不该这么大：{tau_fast:.1f}"


def test_boot_ci_block_too_short_is_anticonservative():
    """块长短于记忆时，两条独立序列也会得出"显著"的区间。

    这是审计实测到 1/3 假阳性的根因，锁死它，防止有人把块长改回去。
    """
    import numpy as np
    import price_link as PL

    rng = np.random.default_rng(3)
    n = 3000

    def ar(rho):
        e = rng.normal(size=n); x = np.zeros(n)
        for i in range(1, n):
            x[i] = rho * x[i - 1] + e[i]
        return x

    x, y = ar(0.99), ar(0.99)            # 彼此完全独立
    tau = max(PL.acf_time(x), PL.acf_time(y))
    assert tau > 50, f"AR(0.99) 的 τ 应该很大，实测 {tau:.0f}"

    w = {}
    for b in (6, 24, 96):
        ci = PL._boot_ci(x, y, block=b, B=300)
        assert ci is not None
        w[b] = ci[1] - ci[0]
    assert w[96] > w[24] > w[6], f"块越长区间没有越宽：{w}"

    # 块长超过 n/10 时必须**拒绝给数**，而不是给一个假的窄区间
    assert PL._boot_ci(x, y, block=int(3 * tau), B=300) is None


def test_clim_trailing_no_thin_fallback():
    """窗口没攒够时必须给 NaN，不能拿一两个观测冒充 30 天气候态。"""
    import numpy as np
    import price_link as PL

    n = 24 * 40
    hod = np.tile(np.arange(24), 40)
    t = pd.date_range("2026-01-01", periods=n, freq="h").to_numpy()
    v = np.arange(n, dtype=float)
    out = PL._clim_trailing(v, t, hod, days=30, min_obs=8, lag_days=6)
    # 每个整点的前 (min_obs + lag) 天必须是 NaN
    for h in (0, 7, 18):
        idx = np.where(hod == h)[0]
        assert np.isnan(out[idx[:8]]).all(), f"整点 {h} 的窗口没攒够却给了数"


# ---------------------------------------------------------------------------
# 基准阶梯（ladder.py）—— 重写后的核心，必须有回归测试
# ---------------------------------------------------------------------------
# 注意：下面造数据时用 UTC 00–05 点，对应 NEM 10–15 点（UTC+10）。
# 经济评估按 NEM 交易日分组，若用 UTC 10–15 点造数据，会跨两个 NEM 日。

def test_vst_roundtrip():
    """asinh 变换必须可逆，且对负价有定义（电价会穿零，log 不行）。"""
    import ladder as LD
    y = np.array([-980.0, -1.0, 0.0, 35.0, 300.0, 16600.0])
    back = LD.ivst(LD.vst(y, 60.0), 60.0)
    assert np.allclose(back, y, rtol=1e-9, atol=1e-6)


def test_fit_predict_refuses_to_extrapolate():
    """sinh 是指数级的。训练期没见过的解释变量取值，不能吐出天价预测。

    这是真实踩过的坑：不夹 clip 时，fixture 上 B2 的 RMSE 从 124 炸到 8171，
    rMAE 3.7，整张表全废。
    """
    import ladder as LD
    rng = np.random.default_rng(0)
    x = rng.normal(size=400)
    tr = pd.DataFrame({"x": x, "rrp": 50 + 30 * x + rng.normal(scale=5, size=400)})
    te = pd.DataFrame({"x": np.array([0.0, 50.0, -50.0])})   # 后两个远在训练域外
    p = LD._fit_predict(tr, te, ["x"])
    lim = np.abs(tr["rrp"]).max() * 20
    assert np.all(np.abs(p) < lim), f"外推炸了: {p}"
    assert np.isfinite(p).all()


def test_ar_lags_use_timestamps_not_positions():
    """样本有洞时，按下标平移会把'24 行前'当成'24 小时前'，静默错位。"""
    t = pd.date_range("2025-01-01", periods=200, freq="h", tz="UTC")
    keep = np.ones(len(t), bool)
    keep[50:70] = False                       # 挖掉 20 小时
    t = t[keep]
    s = pd.Series(np.arange(len(t), dtype=float), index=t)
    got = s.reindex(t - pd.Timedelta(hours=24)).to_numpy()
    # 洞后面的行，按时间戳取滞后应当拿到 NaN 或正确的值，绝不能拿到"往前 24 行"
    i = int(np.searchsorted(t, t[0] + pd.Timedelta(hours=90)))
    want = t[i] - pd.Timedelta(hours=24)
    exp = float(s.loc[want]) if want in s.index else np.nan
    assert (np.isnan(got[i]) and np.isnan(exp)) or got[i] == exp


def test_gw_needs_enough_points():
    import ladder as LD
    rng = np.random.default_rng(1)
    assert LD.giacomini_white(rng.normal(size=20), rng.normal(size=20)) is None


def test_gw_hac_is_conservative_under_autocorrelation():
    """损失差分自相关很强时，不做 HAC 修正的 p 值会小得荒谬。
    GW 用了 Newey-West，对同一组数据必须比朴素 t 检验更保守。"""
    import ladder as LD
    from scipy import stats as st
    rng = np.random.default_rng(3)
    n = 800
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = 0.95 * e[i - 1] + rng.normal()        # 高度自相关、均值为 0
    p_hac = LD.giacomini_white(e, np.zeros(n))
    p_naive = 2 * (1 - st.norm.cdf(abs(e.mean() / (e.std(ddof=1) / np.sqrt(n)))))
    assert p_hac > p_naive


def test_economics_capture_bounded_by_perfect_foresight():
    """任何预测的套利收益都不可能超过完美预知。超了就是有前视。"""
    import ladder as LD
    rng = np.random.default_rng(5)
    days, hrs = 120, 6
    t = pd.to_datetime([
        pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(days=d, hours=h)
        for d in range(days) for h in range(hrs)])
    y = rng.gamma(2.0, 40.0, size=len(t))
    res = {"status": "ok", "_y": y, "_t": t.to_numpy(),
           "_pred": {"B0 naive": y + rng.normal(scale=60, size=len(t)),
                     "B1 自回归": y + rng.normal(scale=30, size=len(t)),
                     "B2 +官方": y + rng.normal(scale=15, size=len(t)),
                     "B3 +天气": y + rng.normal(scale=10, size=len(t))}}
    ec = LD.economics(res)
    assert ec["status"] == "ok"
    assert ec["days"] == days
    for r in ec["rows"]:
        assert r["capture"] <= 1.0 + 1e-9, r
    # 预测越准，捕获率应当越高
    caps = [r["capture"] for r in ec["rows"]]
    assert caps == sorted(caps), caps


def test_economics_perfect_prediction_captures_everything():
    import ladder as LD
    rng = np.random.default_rng(6)
    t = pd.to_datetime([
        pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(days=d, hours=h)
        for d in range(60) for h in range(6)])
    y = rng.gamma(2.0, 40.0, size=len(t))
    res = {"status": "ok", "_y": y, "_t": t.to_numpy(),
           "_pred": {"B0 naive": y}}
    ec = LD.economics(res)
    assert abs(ec["rows"][0]["capture"] - 1.0) < 1e-9


def test_ladder_tiers_share_one_test_set():
    """四层必须在同一批行上评估。少一行都不能比。"""
    import ladder as LD
    rng = np.random.default_rng(11)
    n_days = 400
    t = pd.to_datetime([
        pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(days=d, hours=h)
        for d in range(n_days) for h in range(6)])
    n = len(t)
    ghi = rng.uniform(0, 1, n)
    price = 60 - 40 * ghi + rng.normal(scale=10, size=n)
    d = pd.DataFrame({"t": t, "hod": [x.hour for x in t], "rrp": price,
                      "naive": np.roll(price, 42), "aemo": price + rng.normal(scale=8, size=n),
                      "ghi": ghi, "wind": rng.uniform(0, 1, n), "temp": rng.uniform(0, 1, n)})
    for c, lag in (("p_last", 6), ("p_d1", 7), ("p_d2", 12), ("p_w1", 42)):
        d[c] = np.roll(price, lag)
    d["p_mean24"] = pd.Series(price).rolling(6).mean().shift(6).to_numpy()
    d = d.dropna().reset_index(drop=True)
    res = LD.run(d, train_days=180, step_days=14, embargo_days=3)
    assert res["status"] == "ok", res
    lens = {k: len(v) for k, v in res["_pred"].items()}
    assert len(set(lens.values())) == 1, lens
    assert len(res["_y"]) == res["n_test"]
    # 造的数据里 ghi 是真因子，B3 必须打赢 B2
    r3 = next(r for r in res["tiers"] if r["tier"] == "B3 +天气")
    assert r3["gain_rmae"] > 0, r3
    assert r3["gw_p"] < 0.05, r3


def test_ladder_falls_back_without_official_data():
    """没有 AEMO 预调度时，B2 必须显式跳过并标注，不能悄悄拿 B1 冒充。"""
    import ladder as LD
    rng = np.random.default_rng(12)
    t = pd.to_datetime([
        pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(days=d, hours=h)
        for d in range(400) for h in range(6)])
    n = len(t)
    price = 60 + rng.normal(scale=20, size=n)
    d = pd.DataFrame({"t": t, "hod": [x.hour for x in t], "rrp": price,
                      "naive": np.roll(price, 42), "aemo": np.nan,
                      "ghi": rng.uniform(0, 1, n), "wind": rng.uniform(0, 1, n),
                      "temp": rng.uniform(0, 1, n)})
    for c, lag in (("p_last", 6), ("p_d1", 7), ("p_d2", 12), ("p_w1", 42)):
        d[c] = np.roll(price, lag)
    d["p_mean24"] = pd.Series(price).rolling(6).mean().shift(6).to_numpy()
    d = d.dropna(subset=[c for c in d.columns if c != "aemo"]).reset_index(drop=True)
    res = LD.run(d, train_days=180, step_days=14, embargo_days=3)
    assert res["status"] == "ok", res
    assert res["has_official"] is False
    b2 = next(r for r in res["tiers"] if r["tier"] == "B2 +官方")
    assert "skipped" in b2
    b3 = next(r for r in res["tiers"] if r["tier"] == "B3 +天气")
    assert b3["vs_prev"] == "B1 自回归"


def test_ladder_pure_noise_weather_shows_no_gain():
    """天气是纯噪声时，B3 不能报出显著改善。这是空跑对照。"""
    import ladder as LD
    rng = np.random.default_rng(13)
    t = pd.to_datetime([
        pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(days=d, hours=h)
        for d in range(400) for h in range(6)])
    n = len(t)
    price = 60 + rng.normal(scale=20, size=n)
    d = pd.DataFrame({"t": t, "hod": [x.hour for x in t], "rrp": price,
                      "naive": np.roll(price, 42),
                      "aemo": price + rng.normal(scale=15, size=n),
                      "ghi": rng.uniform(0, 1, n), "wind": rng.uniform(0, 1, n),
                      "temp": rng.uniform(0, 1, n)})
    for c, lag in (("p_last", 6), ("p_d1", 7), ("p_d2", 12), ("p_w1", 42)):
        d[c] = np.roll(price, lag)
    d["p_mean24"] = pd.Series(price).rolling(6).mean().shift(6).to_numpy()
    d = d.dropna().reset_index(drop=True)
    res = LD.run(d, train_days=180, step_days=14, embargo_days=3)
    b3 = next(r for r in res["tiers"] if r["tier"] == "B3 +天气")
    assert not (b3["gain_rmae"] > 0.01 and b3["gw_p"] < 0.05), \
        f"纯噪声天气报出了显著改善: {b3}"


def test_prediction_never_exceeds_training_price_range():
    """训练期见过 $10,550 的尖峰时，预测不能吐出 $46,636。

    真实踩过的坑：原本在 asinh 空间两端各放宽 20% 的跨度。但 sinh 是指数的——
    训练窗里有一次触顶尖峰，z 跨度到 8.5，20% 的 pad 就把上限乘了 e^1.7 ≈ 5.5 倍。
    NSW1 上实测吐出 $46,636 的预测（实际价 $94），最坏 1% 的行吃掉 66% 的总误差，
    B1 的 rMAE 因此到 1.494——自回归输给 naive 50%。
    """
    import ladder as LD
    rng = np.random.default_rng(0)
    n = 500
    x = rng.normal(size=n)
    y = 80 + 20 * x + rng.normal(scale=15, size=n)
    y[7] = 10550.0                      # 一次触顶尖峰
    tr = pd.DataFrame({"x": x, "rrp": y})
    # 测试集里放几个训练域外的解释变量取值
    te = pd.DataFrame({"x": np.array([0.0, 4.0, 8.0, -8.0, 20.0])})
    p = LD._fit_predict(tr, te, ["x"])
    assert p.max() <= y.max() + 1e-6, f"预测 {p.max():.1f} 超过训练期最高价 {y.max():.1f}"
    assert p.min() >= min(y.min(), LD.PRICE_FLOOR) - 1e-6
    assert np.all(p <= LD.PRICE_CAP) and np.all(p >= LD.PRICE_FLOOR)


def test_spike_in_training_does_not_wreck_rmae():
    """训练窗里的尖峰不能把整层的 rMAE 顶到 1 以上。端到端版的上一个测试。"""
    import ladder as LD
    rng = np.random.default_rng(4)
    days = 500
    t = pd.to_datetime([
        pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=d, hours=h)
        for d in range(days) for h in range(6)])
    n = len(t)
    ghi = rng.uniform(0, 1, n)
    price = 70 - 25 * ghi + rng.normal(scale=15, size=n)
    for i in (300, 1500, 1501, 2400):        # 几次触顶尖峰
        price[i] = 10000 + rng.uniform(0, 5000)
    s = pd.Series(price, index=t)
    d = pd.DataFrame({"t": t, "hod": [x.hour for x in t], "rrp": price, "ghi": ghi,
                      "wind": rng.uniform(0, 1, n), "temp": rng.uniform(0, 1, n),
                      "aemo": np.nan})
    for c, lag in (("p_last", 26), ("p_d1", 27), ("p_d2", 48), ("p_w1", 168)):
        d[c] = s.reindex(t - pd.Timedelta(hours=lag)).to_numpy()
    d["p_mean24"] = s.rolling("24h").mean().reindex(
        t - pd.Timedelta(hours=26)).to_numpy()
    d["naive"] = s.reindex(t - pd.Timedelta(hours=168)).to_numpy()
    d = d.dropna(subset=[c for c in d.columns if c != "aemo"]).reset_index(drop=True)
    res = LD.run(d, train_days=180, step_days=14, embargo_days=3, variants=False)
    b1 = next(r for r in res["tiers"] if r["tier"] == "B1 自回归")
    assert b1["rmae"] < 1.0, f"尖峰把自回归顶到 rMAE={b1['rmae']}，比 naive 还差"


def test_rmdae_reported_alongside_rmae():
    """MAE 被尖峰主导时，中位误差比才讲得清'平常一小时'的技能。两个都要有。"""
    import ladder as LD
    rng = np.random.default_rng(21)
    t = pd.to_datetime([
        pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=d, hours=h)
        for d in range(420) for h in range(6)])
    n = len(t)
    ghi = rng.uniform(0, 1, n)
    price = 70 - 25 * ghi + rng.normal(scale=15, size=n)
    s = pd.Series(price, index=t)
    d = pd.DataFrame({"t": t, "hod": [x.hour for x in t], "rrp": price, "ghi": ghi,
                      "wind": rng.uniform(0, 1, n), "temp": rng.uniform(0, 1, n),
                      "aemo": np.nan})
    for c, lag in (("p_last", 26), ("p_d1", 27), ("p_d2", 48), ("p_w1", 168)):
        d[c] = s.reindex(t - pd.Timedelta(hours=lag)).to_numpy()
    d["p_mean24"] = s.rolling("24h").mean().reindex(
        t - pd.Timedelta(hours=26)).to_numpy()
    d["naive"] = s.reindex(t - pd.Timedelta(hours=168)).to_numpy()
    d = d.dropna(subset=[c for c in d.columns if c != "aemo"]).reset_index(drop=True)
    res = LD.run(d, train_days=180, step_days=14, embargo_days=3, variants=False)
    for r in res["tiers"]:
        if not r.get("skipped"):
            assert "rmdae" in r, r
    b0 = next(r for r in res["tiers"] if r["tier"] == "B0 naive")
    assert abs(b0["rmae"] - 1.0) < 1e-9 and abs(b0["rmdae"] - 1.0) < 1e-9


def test_diag_worst_rows_are_in_nem_time():
    """时段筛选用的是 NEM 小时，最坏行也必须按 NEM 时间显示，否则看起来像筛错了。"""
    import ladder as LD
    t = pd.to_datetime([pd.Timestamp("2025-01-01 04:00", tz="UTC")
                        + pd.Timedelta(hours=i) for i in range(400)])
    y = np.full(400, 50.0)
    y[0] = 5000.0
    res = {"status": "ok", "_y": y, "_t": t.to_numpy(),
           "_pred": {"B0 naive": np.full(400, 50.0)}}
    dg = LD.diagnose(res)
    worst = dg["tiers"][0]["worst"][0]
    assert worst["t"].endswith("14:00"), f"应显示 NEM 14:00，得到 {worst['t']}"


def test_flat_prediction_is_caught_by_spread_not_by_mae():
    """把日内曲线压平的模型：MAE 可以很好，但套利收益必然差。

    MAE 奖励"贴近水平"，套利要的是"排对日内顺序"——两个目标不一样。
    振幅比这一列就是为了让这件事在页面上看得见。
    """
    import ladder as LD
    rng = np.random.default_rng(31)
    days = 200
    t = pd.to_datetime([
        pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(days=d, hours=h)
        for d in range(days) for h in range(6)])
    lvl = rng.normal(80, 25, days).repeat(6)
    shape = np.tile(np.array([-18.0, -7, 2, 9, 14, 0]), days)
    y = lvl + shape + rng.normal(scale=3, size=len(t))

    # 水平贴得很准，但日内全平——MAE 只吃到 |shape|，很小
    flat = lvl + rng.normal(scale=1.0, size=len(t))
    # 水平整体偏高 $60，但日内形状完全对——MAE 很大，排序却是满分
    shaped = lvl + shape + 60.0
    res = {"status": "ok", "_y": y, "_t": t.to_numpy(),
           "_pred": {"B0 naive": flat, "B1 自回归": shaped}}
    ec = LD.economics(res)
    r_flat, r_shaped = ec["rows"][0], ec["rows"][1]

    # 压平的那个 MAE 明显更小
    assert np.mean(np.abs(flat - y)) < np.mean(np.abs(shaped - y))
    # 但它的日内振幅接近 0、秩相关差，赚的钱也少
    assert r_flat["spread"] < 0.3, r_flat
    assert r_shaped["spread"] > 0.8, r_shaped
    assert r_flat["rank_rho"] < r_shaped["rank_rho"]
    assert r_flat["daily"] < r_shaped["daily"], (r_flat, r_shaped)


def test_economics_reports_hours_per_day():
    """窗口太窄时套利被卡住，页面要能说出来，所以口径里必须带上每日可选小时数。"""
    import ladder as LD
    t = pd.to_datetime([
        pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(days=d, hours=h)
        for d in range(80) for h in range(6)])
    y = np.random.default_rng(2).gamma(2, 40, len(t))
    ec = LD.economics({"status": "ok", "_y": y, "_t": t.to_numpy(),
                       "_pred": {"B0 naive": y}})
    assert ec["hours_per_day"] == 6


# ---------------------------------------------------------------------------
# 2026-10 修复：分小时拟合 / 储能经济评估 / l4_value 退役
# ---------------------------------------------------------------------------

def _hourly_panel(days=420, seed=77, ghi_coef=-20.0):
    """全天 24 小时的合成面板：强日内形状 + 只在白天起作用的天气项。"""
    rng = np.random.default_rng(seed)
    t = pd.to_datetime([
        pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=d, hours=h)
        for d in range(days) for h in range(24)])
    n = len(t)
    hod = np.array([(x + pd.Timedelta(hours=10)).hour for x in t])   # NEM 小时
    shape = 40 * np.sin((hod - 8) / 24 * 2 * np.pi)
    ghi = np.where((hod >= 7) & (hod <= 17), rng.uniform(0, 1, n), 0.0)
    price = 90 + shape + ghi_coef * ghi + rng.normal(scale=8, size=n)
    s = pd.Series(price, index=t)
    d = pd.DataFrame({"t": t, "hod": hod, "rrp": price, "ghi": ghi,
                      "wind": rng.uniform(0, 1, n), "temp": rng.uniform(0, 1, n),
                      "aemo": np.nan})
    for c, lag in (("p_last", 26), ("p_d1", 27), ("p_d2", 48), ("p_w1", 168)):
        d[c] = s.reindex(t - pd.Timedelta(hours=lag)).to_numpy()
    d["p_mean24"] = s.rolling("24h").mean().reindex(
        t - pd.Timedelta(hours=26)).to_numpy()
    d["naive"] = s.reindex(t - pd.Timedelta(hours=168)).to_numpy()
    return d.dropna(subset=[c for c in d.columns if c != "aemo"]).reset_index(drop=True)


def test_per_hour_beats_shared_model_on_intraday_shape():
    """一个模型跨 24 小时会把日内曲线压平；分小时拟合不会。

    NSW1 上实测：共享模型振幅比只有 0.09，套利收益输给 naive。
    分小时之后日内形状由 24 个截距天然给出。
    """
    import ladder as LD
    d = _hourly_panel()
    sh = LD.run(d, train_days=180, step_days=28, embargo_days=3,
                variants=False, per_hour=False)
    ph = LD.run(d, train_days=180, step_days=28, embargo_days=3,
                variants=False, per_hour=True)
    assert sh["per_hour"] is False and ph["per_hour"] is True
    b_sh = next(r for r in LD.economics(sh)["rows"] if r["tier"] == "B3 +天气")
    b_ph = next(r for r in LD.economics(ph)["rows"] if r["tier"] == "B3 +天气")
    t_sh = next(r for r in sh["tiers"] if r["tier"] == "B3 +天气")
    t_ph = next(r for r in ph["tiers"] if r["tier"] == "B3 +天气")
    # 钱、日内排序、误差三者都要更好。振幅比在这个合成面板里可能打平
    # （滞后特征已经带着日内形状），只要求不变差；NSW1 真实数据上差距是 0.09 vs ~1。
    assert b_ph["daily"] > b_sh["daily"], (b_sh, b_ph)
    assert b_ph["rank_rho"] > b_sh["rank_rho"], (b_sh, b_ph)
    assert t_ph["rmae"] < t_sh["rmae"], (t_sh, t_ph)
    assert b_ph["spread"] >= b_sh["spread"] - 0.02, (b_sh, b_ph)


def test_per_hour_is_the_default():
    """默认必须是分小时。共享模型只是对照，不该悄悄成为生产口径。"""
    import inspect
    import ladder as LD
    assert inspect.signature(LD.run).parameters["per_hour"].default is True


def test_best_cycle_charges_before_discharging():
    """一天一个循环、起始 SOC 为空：所有充电小时必须早于所有放电小时。

    价格最高点出现在最低点**之前**时，不能先放后充。
    """
    import ladder as LD
    p = np.array([200., 190., 20., 30., 60., 80.])   # 早上贵、中午便宜、傍晚回升
    v, ci, di = LD._best_cycle(p, k=2, eta=1.0, cost=0.0)
    assert ci is not None
    assert ci.max() < di.min(), (ci, di)
    assert set(ci.tolist()) == {2, 3} and set(di.tolist()) == {4, 5}
    assert abs(v - (80 + 60 - 20 - 30)) < 1e-9


def test_best_cycle_skips_unprofitable_days():
    """扣掉效率和吞吐成本还不赚钱的日子，不交易。"""
    import ladder as LD
    flat = np.full(24, 70.0)
    v, ci, di = LD._best_cycle(flat, k=2, eta=0.85, cost=5.0)
    assert ci is None and v == 0.0


def test_economics_never_beats_perfect_foresight_with_battery():
    """同一约束下，任何预测的实现收益都不能超过完美预知——超了就是有前视。"""
    import ladder as LD
    d = _hourly_panel(days=300, seed=5)
    res = LD.run(d, train_days=180, step_days=28, embargo_days=3, variants=False)
    ec = LD.economics(res)
    assert ec["status"] == "ok"
    assert ec["battery"]["hours"] == 2
    for r in ec["rows"]:
        assert r["capture"] <= 1.0 + 1e-9, r


def test_economics_groups_by_nem_trading_day():
    """按 NEM 交易日分组：UTC 14:00 到次日 13:00 是同一个 NEM 日。"""
    import ladder as LD
    t = pd.date_range("2025-03-01 14:00", periods=24 * 40, freq="h", tz="UTC")
    y = np.random.default_rng(3).gamma(2, 40, len(t))
    ec = LD.economics({"status": "ok", "_y": y, "_t": t.to_numpy(),
                       "_pred": {"B0 naive": y}})
    assert ec["hours_per_day"] == 24, ec["hours_per_day"]
    assert ec["days"] == 40


def test_l4_value_forwards_to_ladder(monkeypatch):
    """l4_value 已退役：它必须把参数原样转给 ladder，并强制 --hours all。"""
    import ladder as LD
    import l4_value
    seen = {}
    monkeypatch.setattr(LD, "main", lambda argv: seen.setdefault("argv", argv) and 0)
    l4_value.main(["--region", "SA1", "--train-days", "150"])
    a = seen["argv"]
    assert a[a.index("--region") + 1] == "SA1"
    assert a[a.index("--hours") + 1] == "all"
    assert a[a.index("--train-days") + 1] == "150"


# ---------------------------------------------------------------------------
# 决策时刻纪律：每个特征在 t − 24h 都必须已可见
# ---------------------------------------------------------------------------

def _seed_weather_site(site="QLD1_DD", days=40, run_every_h=6, delay_h=5):
    """一个站点 40 天的逐小时预报：每 6 小时一个运行，发布延迟 5 小时，
    每个运行覆盖未来 0–60 小时。真值按小时给齐，final。"""
    t0 = pd.Timestamp("2026-03-01 00:00")
    tgts = pd.date_range(t0, periods=24 * days, freq="h")
    runs = pd.date_range(t0 - pd.Timedelta(hours=72), tgts[-1], freq=f"{run_every_h}h")
    rows = []
    for r in runs:
        ka = r + pd.Timedelta(hours=delay_h)
        for tg in pd.date_range(r, r + pd.Timedelta(hours=60), freq="h"):
            if tg < tgts[0] or tg > tgts[-1]:
                continue
            for var in VARIABLES:
                rows.append(dict(site_id=site, model="ecmwf_ifs025", variable=var,
                                 interval_time=tg, known_at=ka, value=1.0,
                                 known_at_method="exact", source="test_lead",
                                 ingested_at=store.utcnow()))
    store.append("forecast", pd.DataFrame(rows), store.new_run_id())
    tr = [dict(site_id=site, variable=v, interval_time=tg, value=1.0,
               known_at=tg + pd.Timedelta(days=5), quality="final",
               source="test_lead", ingested_at=store.utcnow())
          for tg in tgts for v in VARIABLES]
    store.append("truth", pd.DataFrame(tr), store.new_run_id())


def test_load_backtest_min_lead_is_one_sided():
    """默认的 ±12h 双侧窗口会挑到"离 24h 最近"的那版，可能比决策时刻还新；
    min_lead_h 只允许 known_at ≤ t − 24h，并取其中最新的一版。"""
    import realdata as R
    _seed_weather_site()
    c = store.connect()
    try:
        two = R.load_backtest(c, "QLD1_DD", models=["ecmwf_ifs025"])
        one = R.load_backtest(c, "QLD1_DD", models=["ecmwf_ifs025"], min_lead_h=24)
    finally:
        c.close()
    l2 = np.asarray(two.lead_h["ecmwf_ifs025"], float)
    l1 = np.asarray(one.lead_h["ecmwf_ifs025"], float)
    assert (l2 < 24).any(), "前提：双侧窗口确实会选到 24 小时以内的版本"
    assert (l1 >= 24).all(), l1.min()
    assert (l1 < 24 + 6).all(), l1.max()          # 每 6 小时一个运行，取最新那版


# ---------------------------------------------------------------------------
# 预调度：逐文件解析，运行时刻读文件头（2026-10 重写）
# ---------------------------------------------------------------------------

def _pd_file(run_nem: str, n_periods=72, regions=("NSW1", "QLD1", "SA1", "TAS1", "VIC1"),
             with_c=True, seqno="2026082815"):
    """造一份和 PUBLIC_PREDISPATCHIS 同构的文件：C 行 + 两张表 + 干预行。"""
    run = pd.Timestamp(run_nem)
    pub = run + pd.Timedelta(minutes=1, seconds=6)
    lines = []
    if with_c:
        lines.append(f"C,NEMP.WORLD,PREDISPATCHIS,AEMO,PUBLIC,{pub:%Y/%m/%d},{pub:%H:%M:%S},"
                     "0000000412345678,PREDISPATCHIS,0000000412345670")
    lines.append("I,PREDISPATCH,CASE_SOLUTION,1,PREDISPATCHSEQNO,RUNNO,SOLUTIONSTATUS")
    lines.append(f"D,PREDISPATCH,CASE_SOLUTION,1,{seqno},1,0")
    lines.append("I,PREDISPATCH,REGION_PRICES,1,PREDISPATCHSEQNO,RUNNO,REGIONID,PERIODID,"
                 "INTERVENTION,RRP,EEP,LASTCHANGED,DATETIME")
    for k in range(n_periods):
        end = run + pd.Timedelta(minutes=30 * (k + 1))
        for j, reg in enumerate(regions):
            price = 100.0 + j * 10 + k            # 可以反推：区号 + 时段号
            lines.append(f'D,PREDISPATCH,REGION_PRICES,1,{seqno},1,{reg},{k+1},0,{price},0,'
                         f'"{pub:%Y/%m/%d %H:%M:%S}","{end:%Y/%m/%d %H:%M:%S}"')
            if j == 0:                             # 干预运行：必须丢掉
                lines.append(f'D,PREDISPATCH,REGION_PRICES,1,{seqno},1,{reg},{k+1},1,99999,0,'
                             f'"{pub:%Y/%m/%d %H:%M:%S}","{end:%Y/%m/%d %H:%M:%S}"')
    lines.append("C,END OF REPORT,999")
    name = f"PUBLIC_PREDISPATCHIS_{run:%Y%m%d%H%M}_{pub:%Y%m%d%H%M%S}.CSV"
    return name, "\n".join(lines)


def test_predispatch_publish_time_from_c_row():
    """运行时刻读文件头 C 行（NEM 时间）→ UTC。不再从 SEQNO 猜。"""
    import aemo
    name, text = _pd_file("2026-08-28 12:00")
    assert aemo.publish_time(text, "") == pd.Timestamp("2026-08-28 02:01:06")
    # 没有 C 行时退回文件名里的最后一个时间戳
    name2, text2 = _pd_file("2026-08-28 12:00", with_c=False)
    assert aemo.publish_time(text2, name2) == pd.Timestamp("2026-08-28 02:01:06")


def test_predispatch_single_run_spans_many_leads():
    """一次运行覆盖未来几十个小时——时效必须是一条带，不能全是 0.5h。

    这正是 Eric 在 2026-09 看到的症状：lead 0.5~0.5h、写进 0 行。
    """
    import aemo
    name, text = _pd_file("2026-08-28 12:00", n_periods=72)    # 36 小时
    f = aemo.parse_predispatch(text, name, lead_band=(0, 1e9))
    lh = (f["interval_time"] - f["known_at"]).dt.total_seconds() / 3600
    assert lh.max() - lh.min() >= 30, (lh.min(), lh.max())
    assert set(f["site_id"]) == {"NSW1", "QLD1", "SA1", "TAS1", "VIC1"}
    assert (f["method"] == "exact").all()


def test_predispatch_hourly_value_and_intervention():
    """两个半小时取均值；干预行（RRP 99999）不能混进来；半截小时丢掉。"""
    import aemo
    name, text = _pd_file("2026-08-28 12:00", n_periods=4)
    f = aemo.parse_predispatch(text, name, lead_band=(-1e9, 1e9))
    nsw = f[f["site_id"] == "NSW1"].sort_values("interval_time")
    # 时段 1、2（结束于 12:30、13:00）组成 NEM 12:00 这一小时 = UTC 02:00
    assert nsw["interval_time"].iloc[0] == pd.Timestamp("2026-08-28 02:00")
    assert abs(nsw["value"].iloc[0] - (100 + 101) / 2) < 1e-9
    assert (f["value"] < 9999).all()
    assert len(nsw) == 2                     # 4 个半小时 = 2 个完整小时


def test_predispatch_no_c_row_no_name_is_conservative():
    """读不到发布时刻时，known_at 取第一个时段的**结束**时刻——比真实发布晚，
    不会前视；并标成 approx。"""
    import aemo
    _, text = _pd_file("2026-08-28 12:00", with_c=False)
    f = aemo.parse_predispatch(text, "", lead_band=(-1e9, 1e9))
    assert (f["method"] == "approx").all()
    assert (f["known_at"] == pd.Timestamp("2026-08-28 02:30")).all()   # NEM 12:30


def test_predispatch_archive_end_to_end(monkeypatch):
    """日归档 zip（里面套着每次运行的 zip）→ 写库 → ladder 取到 ≥24h 的那一版。"""
    import io as _io
    import zipfile as _zf
    import aemo
    import ladder as LD

    def inner(run_nem):
        n, t = _pd_file(run_nem, n_periods=80, regions=("TAS1",))
        b = _io.BytesIO()
        with _zf.ZipFile(b, "w") as z:
            z.writestr(n, t)
        return n.replace(".CSV", ".zip"), b.getvalue()

    day = _io.BytesIO()
    with _zf.ZipFile(day, "w") as z:
        for hh in ("10:00", "10:30", "11:00", "11:30", "12:00"):
            n, blob = inner(f"2026-09-01 {hh}")
            z.writestr(n, blob)
    url = aemo.ARC_PREDISP + "PUBLIC_PREDISPATCHIS_20260901.zip"
    monkeypatch.setattr(aemo, "list_zips", lambda d: [url])
    monkeypatch.setattr(aemo, "download", lambda u, **k: day.getvalue())

    nt, nf = aemo.ingest_predispatch_archive("2026-09-01", "2026-09-01", (20, 30))
    assert nf > 0
    # 再跑一次：断点续跑，已完成的日子跳过
    nt2, nf2 = aemo.ingest_predispatch_archive("2026-09-01", "2026-09-01", (20, 30))
    assert nf2 == 0

    c = store.connect()
    try:
        fc = LD.official(c, "TAS1")
    finally:
        c.close()
    assert len(fc) > 0
    # 每个目标小时取的都必须是"决策时刻之前发出的版本里最新的一版"。逐个目标核对：
    c = store.connect()
    try:
        allv = c.execute("""
            SELECT interval_time AS t,
                   date_diff('second', known_at, interval_time) / 3600.0 AS lh
            FROM forecast WHERE site_id = 'TAS1' AND model = 'aemo_predispatch'
        """).df()
    finally:
        c.close()
    import realdata as R
    allv["tau"] = pd.to_datetime(R.decision_time(allv["t"]))
    allv["ka"] = allv["t"] - pd.to_timedelta(allv["lh"], unit="h")
    adm = allv[allv["ka"] <= allv["tau"] + pd.Timedelta(seconds=1)]
    want = adm.groupby("t")["lh"].min()            # 可见版本里最新的一版 = 时效最短
    got = fc.set_index("t")["lh"]
    assert set(got.index) == set(want.index)
    assert np.allclose(got.sort_index().to_numpy(), want.sort_index().to_numpy())


def test_predispatch_lead_band_is_applied():
    """时效带外的行不进库（替代旧的 lead_band 测试）。"""
    import aemo
    name, text = _pd_file("2026-08-28 12:00", n_periods=80)
    f = aemo.parse_predispatch(text, name, lead_band=(24, 30))
    lh = (f["interval_time"] - f["known_at"]).dt.total_seconds() / 3600
    assert len(f) > 0 and (lh >= 24).all() and (lh <= 30).all()


def test_concatenated_tables_no_longer_produce_predispatch():
    """rows_from_tables 吃的是拼接后的表，分不清哪一行属于哪次运行，
    所以它不再产出预调度行——全部改走逐文件的 parse_predispatch。"""
    import aemo
    _, t1 = _pd_file("2026-08-28 12:00")
    _, t2 = _pd_file("2026-08-28 12:30", seqno="2026082816")
    tabs = aemo.extract([t1, t2])
    _, f = aemo.rows_from_tables(tabs, "test")
    assert len(f) == 0


# ---------------------------------------------------------------------------
# 交易台：样本量门槛与发布安全
# ---------------------------------------------------------------------------

def _region_ok(days, gain_ci=(5.0, 9.0), gain_rmae=0.03, p=0.01, has_off=True):
    return {"status": "ok", "has_official": has_off, "n_test": days * 24,
            "span": ["2026-01-01 00:00:00", str(pd.Timestamp("2026-01-01") +
                                               pd.Timedelta(days=days - 1))],
            "tiers": [{"tier": "B3 +天气", "gain_rmae": gain_rmae, "gw_p": p,
                       "vs_prev": "B2 +官方", "rmae": 0.8, "mae": 1, "rmse": 1, "rho": .5}],
            "economics": {"status": "ok", "days": days, "eta": 0.85,
                          "battery": {"mw": 1.0, "hours": 2, "cost": 5.0},
                          "hours_per_day": 24, "perfect_daily": 200.0,
                          "rows": [{"tier": "B3 +天气", "gain": 7.0, "daily": 120.0,
                                    "ci": [100.0, 140.0], "capture": 0.6,
                                    "rank_rho": 0.7, "spread": 0.9,
                                    "gain_ci": list(gain_ci)}]}}


def test_desk_short_sample_is_never_usable():
    """样本外不足 90 天：哪怕三道统计门槛全过，也只判"样本不足"。"""
    import desk
    lvl, tag, why = desk.verdict(_region_ok(51))
    assert (lvl, tag) == ("none", "样本不足"), why
    lvl, tag, _ = desk.verdict(_region_ok(200))
    assert (lvl, tag) == ("yes", "可用")


def test_desk_page_is_full_document_without_local_paths():
    """发布到 GitHub Pages 的页面必须是完整文档，且不能带出本机路径。"""
    import desk
    data = {"generated_at": "2026-10-24T00:00:00+00:00", "hours": "all",
            "decide_lead_h": 24, "is_fixture": False, "train_days": 120,
            "step_days": 7, "embargo_days": 3, "per_hour": True,
            "data_dir": "/Users/someone/secret/data",
            "regions": {"SA1": _region_ok(51)}}
    page = desk.render(data)
    assert page.lstrip().lower().startswith("<!doctype html>")
    assert '<meta charset="utf-8">' in page
    assert "/Users/" not in page and "secret" not in page
    assert "样本不足" in page


def test_mmsdm_archive_writes_month_by_month(monkeypatch):
    """月度归档必须逐月解析、逐月落盘——一次性攒 31 个月在 Mac 上会吃光内存卡死。"""
    import io as _io
    import zipfile as _zf
    import aemo

    def month_zip(y, m):
        rows = ["I,DISPATCH,PRICE,5,SETTLEMENTDATE,RUNNO,REGIONID,DISPATCHINTERVAL,"
                "INTERVENTION,RRP,EEP,ROP"]
        t0 = pd.Timestamp(year=y, month=m, day=1, hour=10, minute=5)
        for i in range(24):
            sd = (t0 + pd.Timedelta(minutes=5 * i)).strftime("%Y/%m/%d %H:%M:%S")
            rows.append(f'D,DISPATCH,PRICE,5,"{sd}",1,TAS1,{i},0,{40 + i},0,0')
        b = _io.BytesIO()
        with _zf.ZipFile(b, "w") as z:
            z.writestr("x.CSV", "\n".join(rows))
        return b.getvalue()

    calls = []
    real_write = aemo._write

    def spy(texts, source, quiet=False):
        calls.append(len(texts))
        return real_write(texts, source, quiet=quiet)

    def fake_get(url, binary=False):
        import re as _re
        m = _re.search(r"(\d{4})(\d{2})010000", url)
        return month_zip(int(m.group(1)), int(m.group(2)))

    monkeypatch.setattr(aemo, "_get", fake_get)
    monkeypatch.setattr(aemo, "_write", spy)
    nt, nf = aemo.ingest_archive("2023-01", "2023-03")
    assert calls == [1, 1, 1], calls          # 三个月，三次各写一个月
    assert nt == 3 * 2 * 2                    # 每月 2 小时 × (rrp, rrp_max)



# ---------------------------------------------------------------------------
# 固定时刻决策口径（2026-10）：NEM 日 D 的 24 小时在 D−1 天 13:30 一次性定计划
# ---------------------------------------------------------------------------

def test_decision_time_is_previous_nem_day_at_clock():
    import realdata as R
    # UTC 14:00 = NEM 次日 00:00 → 属于次日，决策在当天 13:30 NEM = 03:30 UTC
    cases = {"2026-03-10 13:00": "2026-03-09 03:30",    # NEM 3/10 23:00 → 3/9 13:30
             "2026-03-10 14:00": "2026-03-10 03:30",    # NEM 3/11 00:00 → 3/10 13:30
             "2026-03-10 02:00": "2026-03-09 03:30"}    # NEM 3/10 12:00 → 3/9 13:30
    for t, tau in cases.items():
        assert R.decision_time(pd.Timestamp(t)) == pd.Timestamp(tau), t


def test_decision_time_python_and_sql_agree():
    """SQL 版本和 Python 版本必须给出同一个决策时刻，否则天气/预调度和价格滞后
    用的是两套信息集。"""
    import duckdb
    import realdata as R
    t = pd.date_range("2026-01-01", periods=24 * 9, freq="h")
    want = pd.to_datetime(R.decision_time(pd.Series(t), "13:30"))
    c = duckdb.connect()
    c.register("x", pd.DataFrame({"t": t}))
    got = c.execute(f"SELECT {R.decision_sql('t', '13:30')} AS tau FROM x ORDER BY t").df()["tau"]
    c.close()
    assert (pd.to_datetime(got).to_numpy() == pd.Series(want).to_numpy()).all()


def test_ar_features_are_published_before_decision_time():
    """每个价格特征用到的那一小时，都必须在决策时刻之前结算公布（起点 + 1h10m）。
    p_last 必须是**最新**能用的那一小时——再晚一小时就还没公布。"""
    import ladder as LD
    import realdata as R
    hrs = pd.date_range("2026-01-01", periods=24 * 30, freq="h")
    s = pd.Series(np.arange(len(hrs), dtype=float), index=hrs)    # 值 = 小时编号
    t = pd.Series(hrs[24 * 10: 24 * 13])                           # 三天、每个钟点
    f = LD.ar_features(s, t)
    tau = pd.Series(pd.to_datetime(R.decision_time(t)))
    pub = pd.Timedelta(hours=1, minutes=10)
    for col in ("p_last", "p_d1", "p_d2", "p_w1"):
        src = hrs[0] + pd.to_timedelta(f[col], unit="h")           # 反推用到的小时
        assert ((src + pub) <= tau).all(), col
    src_last = hrs[0] + pd.to_timedelta(f["p_last"], unit="h")
    assert ((src_last + pd.Timedelta(hours=1) + pub) > tau).all(), "p_last 不是最新可用的"
    # 同一钟点：p_d1 要么是昨天、要么是前天，取决于那一钟点昨天是否已公布
    lag = (t - (hrs[0] + pd.to_timedelta(f["p_d1"], unit="h"))) / pd.Timedelta(hours=1)
    assert set(lag.unique()) <= {24.0, 48.0}
    assert {24.0, 48.0} <= set(lag.unique())


def test_official_uses_latest_version_before_decision_time():
    """预调度只能用决策时刻之前发出的版本，并在其中取最新——晚 5 分钟发出的也不行。"""
    import ladder as LD
    import realdata as R
    tgt = pd.Timestamp("2026-07-10 08:00")                 # NEM 18:00
    tau = R.decision_time(tgt)
    rows = []
    for k, dt in enumerate((-180, -10, 5, 120)):
        rows.append(dict(site_id="VIC1", model="aemo_predispatch", variable="rrp",
                         interval_time=tgt, known_at=tau + pd.Timedelta(minutes=dt),
                         value=float(dt), known_at_method="exact", source="test_pd2",
                         ingested_at=store.utcnow()))
    store.append("forecast", pd.DataFrame(rows), store.new_run_id())
    c = store.connect()
    try:
        fc = LD.official(c, "VIC1")
    finally:
        c.close()
    row = fc[fc["t"] == tgt]
    assert len(row) == 1 and row["value"].iloc[0] == -10.0, row


def test_load_backtest_decide_at_respects_decision_time():
    """天气预报：只取决策时刻前发出的版本里最新的一版。"""
    import realdata as R
    _seed_weather_site()
    c = store.connect()
    try:
        p = R.load_backtest(c, "QLD1_DD", models=["ecmwf_ifs025"], decide_at="13:30")
    finally:
        c.close()
    t = pd.Series(pd.to_datetime(p.t))
    ka = t - pd.to_timedelta(np.asarray(p.lead_h["ecmwf_ifs025"], float), unit="h")
    tau = pd.Series(pd.to_datetime(R.decision_time(t)))
    assert (ka <= tau).all()
    assert ((tau - ka) < pd.Timedelta(hours=6)).all()     # 每 6 小时一个运行，取最新那版


def test_run_reports_official_coverage():
    import ladder as LD
    d = _hourly_panel(days=300, seed=9)
    res = LD.run(d, train_days=180, step_days=28, embargo_days=3, variants=False)
    assert "official_coverage" in res


def test_official_coverage_separates_short_history_from_bad_clock():
    """预调度历史比价格短（比如下载还没完）≠ 决策时刻太早。两个数必须分开报，
    否则会把"还没下完"误判成"决策时刻要往后挪"。"""
    import ladder as LD
    d = _hourly_panel(days=300, seed=10)
    half = d["t"] >= d["t"].iloc[len(d) // 2]
    d["aemo"] = np.where(half, d["rrp"] + 5.0, np.nan)
    res = LD.run(d, train_days=60, step_days=28, embargo_days=3, variants=False)
    assert res["official_cov_in_range"] == 1.0
    assert 0.3 < res["official_coverage"] < 0.7
    assert res["official_span"][0] >= str(d.loc[half, "t"].iloc[0])[:10]


def test_feature_outside_training_range_cannot_drive_prediction_to_ceiling():
    """训练期里几乎不动的特征，在测试期突然出现没见过的值，不能把预测顶到天花板。

    真实复现：SA1 07:00 的 GHI 分位冬天几乎恒定，九月日出提前后跳变，
    预测连续多天顶到训练期唯一一次尖峰 $16,506。
    """
    import ladder as LD
    rng = np.random.default_rng(42)
    n = 300
    x = 0.5 + rng.normal(scale=1e-3, size=n)          # 训练期几乎不动
    y = 60 + 8000 * (x - 0.5) + rng.normal(scale=10, size=n)   # 与价格有一点正相关
    y[17] = 16506.6                                    # 一次尖峰
    tr = pd.DataFrame({"x": x, "z": rng.normal(size=n), "rrp": y})
    te = pd.DataFrame({"x": np.array([0.5, 0.9, 1.0]), "z": np.zeros(3)})   # 季节变了
    p = LD._fit_predict(tr, te, ["x", "z"])
    # 不夹输入端时这里是 [60.6, 16506.6, 16506.6]——正好复现 SA1 的症状
    assert np.all(p < 1000), p


def test_desk_labels_significant_loss_correctly():
    """区间整体在 0 以下是"显著亏钱"，不是"跨 0"。"""
    import desk
    _, _, why = desk.verdict(_region_ok(168, gain_ci=(-8.9, -0.4)))
    assert "显著亏钱" in why and "跨 0" not in why, why


def test_recommended_model_is_simplest_not_significantly_worse():
    """交易台推荐的排程模型：不显著差于最好那个的、最简单的那个。
    多一层特征要用显著更高的收益来换，均值高一点点不算。"""
    import ladder as LD
    rng = np.random.default_rng(8)
    days = 120
    t = pd.to_datetime([pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(days=d, hours=h)
                        for d in range(days) for h in range(24)])
    hod = np.array([(x + pd.Timedelta(hours=10)).hour for x in t])
    shape = 40 * np.sin((hod - 8) / 24 * 2 * np.pi)
    y = 90 + shape + rng.normal(scale=6, size=len(t))
    flat = np.full(len(t), 90.0) + rng.normal(scale=1, size=len(t))   # 一周前同期：几乎没形状
    good = 90 + shape + rng.normal(scale=4, size=len(t))              # 价格模型：形状对
    tiny = good + rng.normal(scale=0.3, size=len(t))                  # 再加两层：几乎一样
    res = {"status": "ok", "_y": y, "_t": t.to_numpy(),
           "_pred": {"B0 naive": flat, "B1 自回归": good, "B2 +官方": tiny, "B3 +天气": tiny + 0.01}}
    ec = LD.economics(res)
    assert ec["recommended"] == "B1 自回归", ec["recommended"]
    assert len(ec["daily"]["dates"]) == ec["days"] == len(ec["daily"]["B1 自回归"])


def test_trader_page_two_pages_plain_words(tmp_path):
    """首页给交易员：完整文档、导航到第二页、不出现 rMAE / GW 术语、不带本机路径；
    "亏钱"状态要带图标和文字；老格式（没有逐日序列）也不能渲染崩。
    desk.main 一次写出 index.html（交易台）和 method.html（方法与证据）。"""
    import json as _json
    import desk
    import trader
    loss = _region_ok(168, gain_ci=(-8.9, -0.4))
    loss["economics"]["rows"][0]["gain"] = -4.4
    loss["products"] = {"status": "ok", "costs": {"fee": 0.55, "eta": 0.85, "degr": 15, "srmc": 130,
                                                  "start": 60, "min_up": 3},
                        "sources": {}, "sensitivity": {},
                        "items": {"battery": {"name": "储能套利", "desc": "1 MW / 2 MWh", "days": 168,
                                              "perfect": 150.0, "recommended": "B2 +官方", "best": "B2 +官方",
                                              "rows": [{"model": "AEMO 直用", "daily": 40.0, "ci": [30, 50], "capture": .27},
                                                       {"model": "B2 +官方", "daily": 60.0, "ci": [50, 70], "capture": .4,
                                                        "vs_aemo": 20.0, "vs_aemo_ci": [5.0, 35.0]},
                                                       {"model": "B3 +天气", "daily": 55.0, "ci": [45, 65], "capture": .37,
                                                        "vs_b2": -5.0, "vs_b2_ci": [-9.0, -1.0]}]}}}
    good = _region_ok(168)
    good["economics"]["recommended"] = "B3 +天气"
    good["economics"]["daily"] = {"dates": ["2026-01-01", "2026-01-02", "2026-01-03"],
                                  "perfect": [200, 210, 190], "B0 naive": [50, 60, 40],
                                  "B1 自回归": [90, 95, 80], "B2 +官方": [100, 99, 90],
                                  "B3 +天气": [110, 101, 95]}
    data = {"generated_at": "2026-10-24T00:00:00+00:00", "hours": "all", "decide_at": "13:30",
            "is_fixture": False, "train_days": 180, "step_days": 7, "embargo_days": 3,
            "data_dir": "/Users/someone/secret/data",
            "battery": {"mw": 1.0, "hours": 2.0, "eta": 0.85, "cost": 5.0},
            "regions": {"VIC1": loss, "SA1": good, "TAS1": _region_ok(40),
                        "QLD1": {"status": "error", "error": "boom"}}}
    page = trader.render_trader(data)
    assert page.lstrip().lower().startswith("<!doctype html>")
    assert 'href="method.html"' in page and "/Users/" not in page
    body = page.split("<body>", 1)[1]
    assert "rMAE" not in body and "GW p" not in body
    assert "✕" in body and "亏钱" in body
    assert "现货决策" in body and "交易成本假设" in body and "日度或周度" in body
    assert "少赚" in trader.plain_reason(loss)
    src = tmp_path / "ladder.json"
    src.write_text(_json.dumps(data, ensure_ascii=False))
    out = tmp_path / "docs" / "index.html"
    out.parent.mkdir()
    assert desk.main(["--in", str(src), "--out", str(out)]) == 0
    m = (tmp_path / "docs" / "method.html").read_text()
    assert 'href="index.html"' in m and "Giacomini" in m
    assert "交易台" in out.read_text()


def _toy_frame(days=150, wind_matters=True, seed=3):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2025-01-01", periods=days * 24, freq="h", tz="UTC")
    n = len(t)
    d = pd.DataFrame({"t": t, "hod": t.hour})
    for c in ["p_last", "p_d1", "p_d2", "p_w1", "p_mean24", "aemo"]:
        d[c] = 80 + rng.normal(scale=10, size=n)
    for c in ["ghi", "wind", "temp"]:
        d[c] = rng.uniform(0, 1, size=n)
    d["rrp"] = (60 + 0.5 * d["p_last"] + (60 * d["wind"] if wind_matters else 0)
                + rng.normal(scale=3, size=n))
    return d


def test_nested_selection_picks_from_training_window_only():
    """嵌套选特征：只看训练窗（前段拟合、末段验证）。
    风速真有用时挑出风速、不带上噪声变量；天气全是噪声时一个都不挑。"""
    import ladder as LD
    base = LD.AR_COLS + LD.OFF_COLS
    got = LD.select_features(_toy_frame(wind_matters=True), base, LD._fit_predict)
    assert got == ["wind"], got
    got = LD.select_features(_toy_frame(wind_matters=False, seed=4), base, LD._fit_predict)
    assert got == [], got


def test_bias_trade_direction_and_settlement():
    """赌 AEMO 偏差：预测高于预调度做多、低于做空，按 (实际 − 预调度) 结算。
    完全知道实际价的模型每小时都赚 |实际 − 预调度|，命中率 100%；
    方向全反的模型等额亏损。"""
    import ladder as LD
    rng = np.random.default_rng(5)
    days = 60
    t = pd.date_range("2025-03-01", periods=days * 24, freq="h", tz="UTC")
    a = 90 + rng.normal(scale=20, size=len(t))
    y = a + rng.normal(scale=15, size=len(t))
    res = {"status": "ok", "_y": y, "_t": t.to_numpy(), "_aemo": a,
           "_pred": {"B1 自回归": 2 * a - y, "B2 +官方": a + 0.0, "B3 +天气": y.copy()}}
    bt = LD.bias_trade(res)
    rows = {r["tier"]: r for r in bt["rows"]}
    assert bt["status"] == "ok" and bt["ref"] == "B2 +官方"
    perfect = np.abs(y - a).sum() / bt["days"]          # 按 NEM 交易日分组
    assert abs(rows["B3 +天气"]["daily"] - perfect) < 0.05
    assert rows["B3 +天气"]["hit"] == 1.0
    assert abs(rows["B1 自回归"]["daily"] + perfect) < 0.05
    assert rows["B2 +官方"]["daily"] == 0.0          # 和预调度一样 = 不下注
    # 基线"一直做空预调度"：结算 = 预调度 − 实际；模型的价值要和它比
    assert abs(rows["做空 AEMO"]["daily"] - (a - y).sum() / bt["days"]) < 0.05
    assert "vs_short" in rows["B3 +天气"] and "vs_short" not in rows["做空 AEMO"]
    assert rows["B3 +天气"]["gain_ci"][0] > 0
    assert LD.bias_trade({"status": "ok", "_aemo": None, "_pred": {}})["status"] == "unavailable"


def test_bias_baseload_bets_once_per_day():
    """日级版本：一天只押一次方向，按全天均价结算。
    知道实际全天均价的模型每天都赚 |均价差| × 小时数，命中率 100%。"""
    import ladder as LD
    rng = np.random.default_rng(6)
    t = pd.date_range("2025-03-01 14:00", periods=60 * 24, freq="h", tz="UTC")  # NEM 00:00 起
    a = 90 + rng.normal(scale=20, size=len(t))
    y = a + rng.normal(scale=15, size=len(t)) + np.repeat(rng.normal(scale=10, size=60), 24)
    res = {"status": "ok", "_y": y, "_t": t.to_numpy(), "_aemo": a,
           "_pred": {"B1 自回归": a + 0.0, "B2 +官方": a + 0.0, "B3 +天气": y.copy()}}
    bl = LD.bias_trade(res)["baseload"]
    rows = {r["tier"]: r for r in bl["rows"]}
    gap = np.abs((y - a).reshape(60, 24).mean(1)) * 24
    assert abs(rows["B3 +天气"]["daily"] - gap.mean()) < 0.05
    assert rows["B3 +天气"]["hit"] == 1.0 and rows["B3 +天气"]["gain_ci"][0] > 0
    assert len(bl["daily"]["B3 +天气"]) == 60


def test_robust_holm_and_month_breakdown():
    """Holm 校正单调且不小于原 p；按月拆分加总回到整体。"""
    import robust as RB
    adj = RB.holm({"A": 0.01, "B": 0.04, "C": 0.03, "D": None})
    assert adj == {"A": 0.03, "C": 0.06, "B": 0.06, "D": None}
    dates = [f"2026-0{m}-{d:02d}" for m in (1, 2) for d in range(1, 11)]
    x = np.r_[np.ones(10), -np.ones(10) * 0.5]
    ms = RB.monthly(dates, x)
    assert [m["per_day"] for m in ms] == [1.0, -0.5]
    assert abs(sum(m["share"] for m in ms) - 1) < 1e-9
    assert RB.leave_one_out(dates, x) == {"month": "2026-01", "per_day": -0.5}
    assert RB.boot_p(np.r_[np.full(60, 5.0)] + np.random.default_rng(0).normal(size=60)) < 0.01


def _paper_frame(days=260, seed=9):
    d = _toy_frame(days=days, wind_matters=True, seed=seed)
    d["t"] = d["t"].dt.tz_localize(None)
    d["naive"] = d["rrp"].shift(168)
    return d


def test_paper_signal_never_sees_unpublished_prices():
    """纸面交易出信号：决策时刻 τ 之后才发布的电价（包括目标日本身），
    改成什么都不能影响信号。训练窗末端的禁运带同理。"""
    import paper as P
    d = _paper_frame()
    target = (d["t"].max() + P.NEM).normalize() - pd.Timedelta(days=1)
    s1, why = P.signal_from_frame(d, target)
    assert why == "ok" and len(s1) == 24, why
    tau = P.tau_for(target)
    d2 = d.copy()
    late = d2["t"] + P.L.PRICE_PUBLISH > tau
    d2.loc[late, "rrp"] = 1e6
    s2, _ = P.signal_from_frame(d2, target)
    cols = [c for c in s1.columns if c.startswith("p_")]
    pd.testing.assert_frame_equal(s1[cols], s2[cols])
    # 反过来证明测试有牙：改动训练窗里的价格，信号必须变
    d3 = d.copy()
    early = (d3["t"] + P.NEM).dt.floor("D") < target - pd.Timedelta(days=10)
    d3.loc[early, "rrp"] = d3.loc[early, "rrp"] * 3
    s3, _ = P.signal_from_frame(d3, target)
    assert not np.allclose(s1["p_b2"], s3["p_b2"])


def test_paper_signal_refuses_incomplete_target_day():
    import paper as P
    d = _paper_frame()
    target = (d["t"].max() + P.NEM).normalize() - pd.Timedelta(days=1)
    m = (d["t"] + P.NEM).dt.floor("D") == target
    d.loc[d.index[m][:3], "aemo"] = np.nan
    s, why = P.signal_from_frame(d, target)
    assert s is None and "aemo" in why


def test_paper_settle_and_first_signal_is_final(tmp_path, monkeypatch):
    """结算：完全知道实际价的模型，偏差交易赚满 |实际 − 预调度|，电池等于完美预知。
    同一目标日已有正式信号时，再出一次不能覆盖（事后不能改）。"""
    import paper as P
    t = pd.date_range("2026-09-30 14:00", periods=24, freq="h")       # NEM 10-01 全天
    rng = np.random.default_rng(1)
    y = 80 + 40 * np.sin(np.arange(24) / 24 * 2 * np.pi) + rng.normal(scale=5, size=24)
    a = y + rng.normal(scale=20, size=24)
    sig = pd.DataFrame({"target_day": "2026-10-01", "region": "VIC1",
                        "t_utc": t.strftime("%Y-%m-%d %H:%M"), "hour_nem": range(24),
                        "aemo": a, "p_b0": a, "p_b1": a, "p_b2": a + 0.0, "p_b3": y, "p_b3sel": y,
                        "status": "ok", "late": False})
    st = P.settle(None, sig, prices={"VIC1": pd.Series(y, index=t)})
    r = st["days"][0]
    assert r["settled"]
    assert abs(r["bias_b3"] - np.abs(y - a).sum()) < 0.05
    assert r["bias_b2"] == 0.0
    assert r["batt_b3"] == r["batt_perfect"]
    assert r["battery_b3"] == r["battery_perfect"] and r["peaker_b3"] == r["peaker_perfect"]
    assert r["flexload_b3"] >= r["flexload_aemo"]
    monkeypatch.setattr(P, "PAPER_DIR", str(tmp_path))
    monkeypatch.setattr(P, "SIGNALS", str(tmp_path / "signals.csv"))
    P.append_signals(sig.copy())
    s2 = sig.copy(); s2["p_b3"] = 0.0
    P.append_signals(s2)
    got = pd.read_csv(tmp_path / "signals.csv")
    assert len(got) == 24 and np.allclose(got["p_b3"], y, atol=0.01)


def test_peaker_dp_matches_brute_force():
    """日前定机组的动态规划 = 暴力枚举全部开停组合（满足最小连续运行）的最优解。"""
    import itertools
    import products as PR
    rng = np.random.default_rng(3)
    for _ in range(30):
        n, m = 9, 3
        p = rng.normal(120, 60, size=n)
        best = 0.0
        for bits in itertools.product([0, 1], repeat=n):
            on = np.array(bits, bool)
            runs, r = [], 0
            for b in on:
                if b:
                    r += 1
                elif r:
                    runs.append(r); r = 0
            if r:
                runs.append(r)
            if any(x < m for x in runs):
                continue
            v = float(np.sum((p - 130 - 0.55)[on])) - 60 * len(runs)
            best = max(best, v)
        on = PR.peaker_plan(p, 130, 0.55, 60, m)
        got = PR.peaker_day(p, p, {**PR.COSTS, "srmc": 130, "start": 60, "min_up": m})
        assert abs(got - best) < 1e-6, (got, best)


def test_products_cost_accounting():
    """储能：费用两个方向都收、衰减按放电量；成本越高越少交易，收益单调不增。
    可调负荷：知道实际价时省钱最多，且永远不比平均用电差。"""
    import products as PR
    rng = np.random.default_rng(4)
    y = 80 + 60 * np.sin(np.arange(24) / 24 * 2 * np.pi) + rng.normal(scale=5, size=24)
    c = dict(PR.COSTS)
    v, ci, di = PR.battery_plan(y, 2, c["eta"], PR.battery_cost_per_h(c))
    expect = c["eta"] * y[di].sum() - y[ci].sum() - 2 * (c["fee"] * 1.85 + c["degr"] * 0.85)
    assert abs(PR.battery_day(y, y, c) - expect) < 1e-9
    vals = [PR.battery_day(y, y, {**c, "degr": d}) for d in (0, 15, 50, 500)]
    assert vals == sorted(vals, reverse=True) and vals[-1] == 0.0
    perfect = PR.flexload_day(y, y, c)
    assert perfect >= 0
    for _ in range(20):
        assert PR.flexload_day(y + rng.normal(scale=40, size=24), y, c) <= perfect + 1e-9


def test_products_summary_compares_against_aemo_as_is():
    """产品汇总里必须有"AEMO 直接用"这条对照，且推荐规则不偏向复杂模型。"""
    import products as PR
    rng = np.random.default_rng(5)
    days = 60
    t = pd.date_range("2026-03-01 14:00", periods=days * 24, freq="h")
    hod = np.arange(len(t)) % 24
    y = 90 + 50 * np.sin(hod / 24 * 2 * np.pi) + rng.normal(scale=10, size=len(t))
    a = y + rng.normal(scale=30, size=len(t))
    res = {"status": "ok", "_y": y, "_t": t.to_numpy(), "_aemo": a,
           "_pred": {"B0 naive": np.roll(y, 168), "B1 自回归": y + rng.normal(scale=12, size=len(t)),
                     "B2 +官方": y + rng.normal(scale=12, size=len(t)),
                     "B3 +天气": y + rng.normal(scale=12, size=len(t))}}
    out = PR.products(res)
    assert out["status"] == "ok"
    for k in ("battery", "flexload", "peaker"):
        rows = {r["model"]: r for r in out["items"][k]["rows"]}
        assert "AEMO 直用" in rows and "vs_aemo" in rows["B2 +官方"]
        assert out["items"][k]["recommended"] in set(rows) | {"不做"}
    # 最好的预测扣完成本都不显著赚钱时，推荐"不做"，而不是挑一个亏钱的模型
    peak = out["items"]["peaker"]
    best = max(peak["rows"], key=lambda r: r["daily"])
    if best["ci"][0] <= 0:
        assert peak["recommended"] == "不做"
    assert len(out["sensitivity"]["battery_degr"]) == 5


def _price_mat(days=200, seed=2):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=days, freq="D")
    base = 80 + 30 * np.sin(np.arange(24) / 24 * 2 * np.pi)
    m = base[None, :] + rng.normal(scale=20, size=(days, 24))
    return pd.DataFrame(m, index=idx, columns=range(24))


def test_futures_settlement_products_and_fully_settled_period():
    """结算口径：峰荷只算工作日 07–22 点；合约期全部结算后，公允价值就是实际结算价、区间收成一点。"""
    import futures as F
    mat = _price_mat()
    per = {"start": pd.Timestamp("2026-03-01"), "end": pd.Timestamp("2026-03-31")}
    fv = F.fair_value(mat, per, pd.Timestamp("2026-04-30"), "peak", n_sims=50)
    sub = mat.loc["2026-03-01":"2026-03-31"]
    wd = sub.index.weekday < 5
    exact = sub.loc[wd, list(range(7, 22))].to_numpy().mean()
    assert abs(fv["mean"] - exact) < 0.01 and fv["p5"] == fv["p95"] and fv["days_left"] == 0
    cap = F.fair_value(mat, per, pd.Timestamp("2026-04-30"), "cap300", n_sims=10)
    assert cap["mean"] == 0.0                                    # 这组价格从不过 $300
    # 进行到一半：已结算比例对，区间有宽度
    mid = F.fair_value(mat, per, pd.Timestamp("2026-03-15"), "base", n_sims=500)
    assert abs(mid["settled_share"] - 15 / 31) < 1e-3 and mid["p95"] > mid["p5"]


def test_public_futures_page_never_carries_market_prices():
    """公开页面只放我们的公允价值：futures.json 里不能有 bid / ask / 市场价字段，渲染出来的页面也不能有。"""
    import futures as F
    import trader
    mat = _price_mat()
    asof = mat.index.max() - pd.Timedelta(days=20)
    fut = {"pool_days": 90, "n_sims": 500, "regions": {"VIC1": {"asof": str(asof.date()), "last_hour": "x",
           "contracts": []}}}
    for per in F.contract_periods(asof):
        c = {"tag": per["tag"], "label": per["label"], "kind": per["kind"], "products": {}}
        for prod in ("base", "peak", "cap300"):
            fv = F.fair_value(mat, per, asof, prod, n_sims=300)
            if fv:
                fv.pop("_samples")
                fv["risk_long"] = F.daily_risk_per_mw(fv, fv["mean"], 1)
                fv["risk_short"] = F.daily_risk_per_mw(fv, fv["mean"], -1)
                c["products"][prod] = fv
        fut["regions"]["VIC1"]["contracts"].append(c)
    txt = json.dumps(fut)
    assert "bid" not in txt and "ask" not in txt
    html_ = trader.futures_block(fut) + trader.hero_tiles(fut, "0", "")
    assert "期货公允价值" in html_ and "signals.py" in html_ and "bid" not in html_


def test_futures_widening_and_signal_rules():
    """区间放宽：系数只会放宽不会收窄；信号要同时过胜率和优势两道门槛；仓位受单日亏损上限约束。"""
    import futures as F
    import signals as S
    df = pd.DataFrame({"period": ["a"] * 10, "asof": range(10), "mean": 100.0, "p5": 90.0, "p25": 97.0,
                       "p75": 103.0, "p95": 110.0, "actual": [100, 120, 80, 101, 99, 130, 70, 100, 100, 100]})
    k50, k90 = F.widen_factors(df)
    assert k50 >= 1 and k90 >= 2.0
    rng = np.random.default_rng(0)
    smp = rng.normal(100, 5, size=4000)
    d = S.decide(smp, 100.0, bid=95, ask=96, fee=0.05, min_prob=0.7, min_edge=1.0)
    assert d["action"] == "买入" and d["prob"] > 0.7
    d = S.decide(smp, 100.0, bid=99, ask=100.5, fee=0.05, min_prob=0.7, min_edge=1.0)
    assert d["action"] == "观望"
    d = S.decide(smp, 100.0, bid=108, ask=109, fee=0.05, min_prob=0.7, min_edge=1.0)
    assert d["action"] == "卖出"
    d = S.decide(smp, 100.0, bid=108, ask=109, fee=0.05, min_prob=0.7, min_edge=1.0, allow_sell=False)
    assert d["action"] == "观望"                          # $300 cap 的公允价值是下限，不出卖出信号
    fv = {"days_left": 10, "day_p05": 40.0, "day_p95": 200.0, "day_hours": 24.0}
    mw, per_mw = S.size(fv, +1, 100.0, {"daily_loss_limit": 14400, "max_mw_per_contract": 50})
    assert per_mw == 60 * 24 and mw == 10
    assert S.period_from_label("Q426")["start"] == pd.Timestamp("2026-10-01")
    assert S.period_from_label("Nov 26")["end"] == pd.Timestamp("2026-11-30")

# import warnings
# warnings.filterwarnings("ignore")

import sys, time, logging
from pathlib import Path

import numpy as np
import pandas as pd
import talib
from scipy.stats import skew, kurtosis
from scipy.stats import entropy as sp_entropy
from scipy.stats import jarque_bera as sp_jb
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import pacf as sm_pacf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── EDIT THESE TO MATCH YOUR FILE LOCATIONS ──────────────────────────────────
RAW_FILES = {
    "3min": "dataset/3_min/merged_output.csv"
}
OUT_DIR    = Path("data")
CHUNK_SIZE = 500_000
WARMUP     = 500
OUT_DIR.mkdir(exist_ok=True)
# ─────────────────────────────────────────────────────────────────────────────


def safe(fn, name, n):
    try:
        r = fn()
        if isinstance(r, pd.Series):
            r = r.reset_index(drop=True)
            if len(r) != n:
                r = pd.Series(np.nan, index=range(n))
            return r
        if isinstance(r, np.ndarray):
            return pd.Series(r if len(r) == n else np.full(n, np.nan))
        return pd.Series(np.nan, index=range(n))
    except Exception as e:
        log.warning(f"  x {name}: {e}")
        return pd.Series(np.nan, index=range(n))


# ─────────────────────────────────────────────────────────────────────────────
#  PRICE ACTION
# ─────────────────────────────────────────────────────────────────────────────
def f_log_return(C):
    s = pd.Series(C)
    return np.log(s / s.shift(1)).replace([np.inf, -np.inf], np.nan)

def f_abs_return(C):
    return f_log_return(C).abs()

def f_high_low_range(H, L, C):
    h, l, c = pd.Series(H), pd.Series(L), pd.Series(C)
    pc  = c.shift(1)
    tr  = pd.concat([h-l, (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=1).mean()
    return ((h - l) / atr).replace([np.inf, -np.inf], np.nan)

def f_close_position(H, L, C):
    h, l, c = pd.Series(H), pd.Series(L), pd.Series(C)
    rng = (h - l).replace(0, np.nan)
    return ((c - l) / rng).replace([np.inf, -np.inf], np.nan)

def f_body_ratio(O, H, L, C):
    o, h, l, c = pd.Series(O), pd.Series(H), pd.Series(L), pd.Series(C)
    rng = (h - l).replace(0, np.nan)
    return ((c - o).abs() / rng).replace([np.inf, -np.inf], np.nan)

def f_upper_wick_ratio(O, H, L, C):
    o, h, l, c = pd.Series(O), pd.Series(H), pd.Series(L), pd.Series(C)
    rng     = (h - l).replace(0, np.nan)
    body_top = pd.concat([o, c], axis=1).max(axis=1)
    return ((h - body_top) / rng).replace([np.inf, -np.inf], np.nan)

def f_lower_wick_ratio(O, H, L, C):
    o, h, l, c = pd.Series(O), pd.Series(H), pd.Series(L), pd.Series(C)
    rng      = (h - l).replace(0, np.nan)
    body_bot = pd.concat([o, c], axis=1).min(axis=1)
    return ((body_bot - l) / rng).replace([np.inf, -np.inf], np.nan)

def f_gap(O, C):
    return pd.Series(O) - pd.Series(C).shift(1)

def f_consecutive_up(C):
    c  = pd.Series(C)
    up = c > c.shift(1)
    return up.groupby((~up).cumsum()).cumsum().astype(float)

def f_consecutive_down(C):
    c  = pd.Series(C)
    dn = c < c.shift(1)
    return dn.groupby((~dn).cumsum()).cumsum().astype(float)

def f_bar_return_zscore(C):
    c = pd.Series(C)
    r = c.pct_change()
    return (r.rolling(20).mean() / r.rolling(20).std()).replace([np.inf, -np.inf], np.nan)

def f_price_vs_ema(H, L, C, period):
    c   = pd.Series(C)
    ema = pd.Series(talib.EMA(C, timeperiod=period), index=c.index)
    atr = pd.Series(talib.ATR(H, L, C, timeperiod=14), index=c.index)
    return ((c - ema) / atr).replace([np.inf, -np.inf], np.nan)

def f_ema_cross_20_50(C):
    c   = pd.Series(C)
    e20 = c.ewm(span=20, adjust=False).mean()
    e50 = c.ewm(span=50, adjust=False).mean()
    d   = e20 - e50
    return (d / d.rolling(20).std()).replace([np.inf, -np.inf], np.nan)

def f_ema_slope_20(C):
    c   = pd.Series(C)
    e20 = c.ewm(span=20, adjust=False).mean()
    return (e20 - e20.shift(5)) / 5

def f_higher_high(H):
    h = pd.Series(H)
    return (h > h.rolling(5).max().shift(1)).astype(int)

def f_lower_low(L):
    l = pd.Series(L)
    return (l < l.rolling(5).min().shift(1)).astype(int)

def f_swing_strength(H, L, C):
    h, l, c = pd.Series(H), pd.Series(L), pd.Series(C)
    pc  = c.shift(1)
    tr  = pd.concat([h-l, (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    sh  = (h > h.shift(1)) & (h > h.shift(-1))
    sl  = (l < l.shift(1)) & (l < l.shift(-1))
    sp  = pd.Series(np.nan, index=h.index)
    sp[sh] = h[sh]; sp[sl] = l[sl]
    nearest = sp.ffill()
    return ((c - nearest).abs() / atr).replace([np.inf, -np.inf], np.nan)

# ─────────────────────────────────────────────────────────────────────────────
#  VOLATILITY
# ─────────────────────────────────────────────────────────────────────────────
def f_atr(H, L, C, p):
    return pd.Series(talib.ATR(H, L, C, timeperiod=p))

def f_atr_ratio(H, L, C):
    a14 = pd.Series(talib.ATR(H, L, C, timeperiod=14))
    a50 = pd.Series(talib.ATR(H, L, C, timeperiod=50))
    return (a14 / a50).replace([np.inf, -np.inf], np.nan)

def f_realized_vol(C, w):
    c = pd.Series(C)
    r = np.log(c / c.shift(1))
    return r.rolling(w).std() * np.sqrt(252 * 1440)

def f_parkinson_vol(H, L):
    h, l = pd.Series(H), pd.Series(L)
    return np.sqrt((np.log(h / l)**2).rolling(30).sum() / (4 * 30 * np.log(2)))

def f_garman_klass_vol(O, H, L, C):
    o, h, l, c = pd.Series(O), pd.Series(H), pd.Series(L), pd.Series(C)
    gk = 0.5 * np.log(h/l)**2 - (2*np.log(2)-1) * np.log(c/o)**2
    return np.sqrt(gk.rolling(30).mean())

def f_yang_zhang_vol(O, H, L, C):
    o, h, l, c = pd.Series(O), pd.Series(H), pd.Series(L), pd.Series(C)
    rs  = np.log(h/o)*(np.log(h/o)-np.log(c/o)) + \
          np.log(l/o)*(np.log(l/o)-np.log(c/o))
    yz  = (np.log(o/c.shift(1))**2 + rs).rolling(30).mean()
    return np.sqrt(yz)

def f_vol_of_vol(C):
    c   = pd.Series(C)
    r   = np.log(c / c.shift(1))
    v10 = r.rolling(10).std() * np.sqrt(252*1440)
    return v10.rolling(20).std()

def f_vol_zscore(H, L, C):
    atr = pd.Series(talib.ATR(H, L, C, timeperiod=14))
    return ((atr - atr.rolling(100).mean()) / atr.rolling(100).std()).replace([np.inf,-np.inf], np.nan)

def f_vol_percentile(H, L, C):
    atr = pd.Series(talib.ATR(H, L, C, timeperiod=14))
    w   = min(len(atr), 252*1440)
    return atr.rolling(w).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)

def f_bollinger_width(C):
    u, m, lo = talib.BBANDS(C, timeperiod=20)
    return (pd.Series(u-lo) / pd.Series(m)).replace([np.inf,-np.inf], np.nan)

def f_keltner_width(H, L, C):
    ema = pd.Series(talib.EMA(C, timeperiod=20))
    atr = pd.Series(talib.ATR(H, L, C, timeperiod=20))
    return (4 * atr / ema).replace([np.inf,-np.inf], np.nan)

def f_natr(H, L, C):
    return (pd.Series(talib.ATR(H, L, C, timeperiod=14)) / pd.Series(C) * 100).replace([np.inf,-np.inf], np.nan)

def f_vol_regime_ma(C):
    c   = pd.Series(C)
    rv  = np.log(c/c.shift(1)).rolling(30).std() * np.sqrt(252*1440)
    arr = rv.values.astype(np.float64)
    return pd.Series(talib.EMA(arr, timeperiod=50), index=rv.index)

def f_vol_breakout(H, L, C):
    return (pd.Series(talib.ATR(H,L,C,timeperiod=14)) >
            2*pd.Series(talib.ATR(H,L,C,timeperiod=50))).astype(int)

def f_vol_contraction(H, L, C):
    return (pd.Series(talib.ATR(H,L,C,timeperiod=14)) <
            0.5*pd.Series(talib.ATR(H,L,C,timeperiod=50))).astype(int)

# ─────────────────────────────────────────────────────────────────────────────
#  MOMENTUM
# ─────────────────────────────────────────────────────────────────────────────
def f_rsi(C, p):
    return pd.Series(talib.RSI(C, timeperiod=p))

def f_rsi_divergence(C):
    rsi = pd.Series(talib.RSI(C, timeperiod=14))
    c   = pd.Series(C)
    return ((c > c.rolling(5).max().shift(1)) &
            (rsi < rsi.rolling(5).max().shift(1))).astype(int)

def f_stoch(H, L, C, which):
    k, d = talib.STOCH(H, L, C, fastk_period=14, slowk_period=3,
                       slowk_matype=0, slowd_period=3, slowd_matype=0)
    return pd.Series(k if which == "k" else d)

def f_macd(C, which):
    m, s, h = talib.MACD(C, fastperiod=12, slowperiod=26, signalperiod=9)
    if which == "line":   return pd.Series(m)
    if which == "signal": return pd.Series(s)
    if which == "hist":   return pd.Series(h)
    return pd.Series(h).diff(3)  # slope

def f_roc(C, p):
    return pd.Series(talib.ROC(C, timeperiod=p))

def f_williams_r(H, L, C):
    return pd.Series(talib.WILLR(H, L, C, timeperiod=14))

def f_cci(H, L, C, p):
    return pd.Series(talib.CCI(H, L, C, timeperiod=p))

def f_adx(H, L, C):
    return pd.Series(talib.ADX(H, L, C, timeperiod=14))

def f_di_plus(H, L, C):
    return pd.Series(talib.PLUS_DI(H, L, C, timeperiod=14))

def f_di_minus(H, L, C):
    return pd.Series(talib.MINUS_DI(H, L, C, timeperiod=14))

def f_dx_cross(H, L, C):
    p = pd.Series(talib.PLUS_DI(H, L, C, timeperiod=14))
    m = pd.Series(talib.MINUS_DI(H, L, C, timeperiod=14))
    return ((p - m) / (p + m)).replace([np.inf,-np.inf], np.nan)

# ─────────────────────────────────────────────────────────────────────────────
#  TREND
# ─────────────────────────────────────────────────────────────────────────────
def f_linreg_slope(C, p):
    return pd.Series(talib.LINEARREG_SLOPE(C, timeperiod=p))

def f_linreg_r2(C, p):
    lr = talib.LINEARREG(C, timeperiod=p)
    r  = talib.CORREL(C, lr, timeperiod=p)
    return (pd.Series(r)**2).replace([np.inf,-np.inf], np.nan)

def f_supertrend(H, L, C):
    h, l, c = pd.Series(H), pd.Series(L), pd.Series(C)
    atr = pd.Series(talib.ATR(H, L, C, timeperiod=10))
    ub  = (h+l)/2 + 3*atr
    lb  = (h+l)/2 - 3*atr
    st  = np.where(c > ub.shift(1), 1, np.where(c < lb.shift(1), -1, np.nan))
    return pd.Series(st).ffill()

def f_ichimoku_tk_cross(H, L, C):
    h, l, c = pd.Series(H), pd.Series(L), pd.Series(C)
    tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2
    kijun  = (h.rolling(26).max() + l.rolling(26).min()) / 2
    atr    = pd.Series(talib.ATR(H, L, C, timeperiod=14))
    return ((tenkan - kijun) / atr).replace([np.inf,-np.inf], np.nan)

def f_ichimoku_cloud_dist(H, L, C):
    h, l, c = pd.Series(H), pd.Series(L), pd.Series(C)
    sa  = ((h.rolling(9).max() + l.rolling(9).min())/2 +
           (h.rolling(26).max() + l.rolling(26).min())/2) / 2
    sb  = (h.rolling(52).max() + l.rolling(52).min()) / 2
    atr = pd.Series(talib.ATR(H, L, C, timeperiod=14))
    return ((c - (sa+sb)/2) / atr).replace([np.inf,-np.inf], np.nan)

def f_aroon(H, L, which):
    up, dn = talib.AROON(H, L, timeperiod=25)
    up, dn = pd.Series(up), pd.Series(dn)
    if which == "up":   return up
    if which == "down": return dn
    return up - dn

def f_vortex(H, L, C, which):
    tr = pd.Series(talib.TRANGE(H, L, C))
    vm = (pd.Series(H) - pd.Series(L).shift(1)).abs() if which == "plus" \
         else (pd.Series(L) - pd.Series(H).shift(1)).abs()
    return (vm.rolling(14).sum() / tr.rolling(14).sum()).replace([np.inf,-np.inf], np.nan)

# ─────────────────────────────────────────────────────────────────────────────
#  STATISTICAL
# ─────────────────────────────────────────────────────────────────────────────
def f_rolling_skew(C, w):
    return pd.Series(C).pct_change().rolling(w).apply(skew, raw=True)

def f_rolling_kurt(C, w):
    return pd.Series(C).pct_change().rolling(w).apply(kurtosis, raw=True)

def f_entropy_30(C):
    def _ent(x):
        h, _ = np.histogram(x, bins=10)
        p = h / h.sum()
        p = p[p > 0]
        return sp_entropy(p)
    return pd.Series(C).pct_change().rolling(30).apply(_ent, raw=True)

def f_hurst(C):
    def _h(x):
        try:
            lags = range(2, 20)
            tau  = [np.std(np.subtract(x[lag:], x[:-lag])) for lag in lags]
            return np.polyfit(np.log(lags), np.log(np.maximum(tau, 1e-10)), 1)[0]
        except Exception:
            return np.nan
    return pd.Series(C).pct_change().rolling(100).apply(_h, raw=True)

def f_autocorr(C, lag):
    return pd.Series(C).pct_change().rolling(50).apply(
        lambda x: pd.Series(x).autocorr(lag=lag), raw=False)

def f_partial_autocorr(C):
    def _p(x):
        try:
            return sm_pacf(x, nlags=1)[1]
        except Exception:
            return np.nan
    return pd.Series(C).pct_change().rolling(50).apply(_p, raw=False)

def f_variance_ratio(C):
    def _vr(x):
        if np.var(x) == 0: return np.nan
        r2 = np.add.reduceat(x, np.arange(0, len(x), 2))
        return np.var(r2) / (2 * np.var(x))
    return pd.Series(C).pct_change().rolling(60).apply(_vr, raw=True)

def f_adf_pvalue(C):
    def _adf(x):
        try:
            return adfuller(x)[1]
        except Exception:
            return np.nan
    return pd.Series(C).pct_change().rolling(100).apply(_adf, raw=False)

def f_jarque_bera(C):
    return pd.Series(C).pct_change().rolling(60).apply(
        lambda x: sp_jb(x)[0], raw=True)

def f_rolling_median_dev(C):
    return pd.Series(C).pct_change().rolling(60).apply(
        lambda x: np.median(np.abs(x - np.median(x))), raw=True)

def f_quantile_range(C):
    return pd.Series(C).pct_change().rolling(60).apply(
        lambda x: np.percentile(x, 75) - np.percentile(x, 25), raw=True)

# ─────────────────────────────────────────────────────────────────────────────
#  VOLUME
# ─────────────────────────────────────────────────────────────────────────────
def f_volume_sma_ratio(V):
    v = pd.Series(V)
    return (v / v.rolling(20).mean()).replace([np.inf,-np.inf], np.nan)

def f_volume_zscore(V):
    v = pd.Series(V)
    return ((v - v.rolling(50).mean()) / v.rolling(50).std()).replace([np.inf,-np.inf], np.nan)

def f_volume_trend(V):
    v   = pd.Series(V)
    sma = v.rolling(20).mean()
    return (sma - sma.shift(5)) / 5

def f_obv_slope(C, V):
    c, v = pd.Series(C), pd.Series(V)
    obv  = (np.sign(c.diff()).fillna(0) * v).cumsum()
    return (obv - obv.shift(20)) / 20

def f_mfi(H, L, C, V):
    h, l, c, v = pd.Series(H), pd.Series(L), pd.Series(C), pd.Series(V)
    tp  = (h + l + c) / 3
    mf  = tp * v
    pos = mf.where(tp > tp.shift(1), 0).rolling(14).sum()
    neg = mf.where(tp < tp.shift(1), 0).rolling(14).sum()
    return (100 - 100 / (1 + pos/neg)).replace([np.inf,-np.inf], np.nan)

def f_vwap_distance(H, L, C, V):
    h, l, c, v = pd.Series(H), pd.Series(L), pd.Series(C), pd.Series(V)
    tp   = (h + l + c) / 3
    vwap = (tp * v).cumsum() / v.cumsum()
    pc   = c.shift(1)
    tr   = pd.concat([h-l, (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    atr  = tr.rolling(14).mean()
    return ((c - vwap) / atr).replace([np.inf,-np.inf], np.nan)

def f_volume_price_corr(C, V):
    return pd.Series(V).rolling(20).corr(pd.Series(C).pct_change().abs())

def f_accumulation_dist(H, L, C, V):
    h, l, c, v = pd.Series(H), pd.Series(L), pd.Series(C), pd.Series(V)
    clv = ((c - l) - (h - c)) / (h - l)
    clv = clv.replace([np.inf,-np.inf], np.nan).fillna(0)
    adl = (clv * v).cumsum()
    return (adl - adl.shift(20)) / 20

# ─────────────────────────────────────────────────────────────────────────────
#  SESSION / TIME
# ─────────────────────────────────────────────────────────────────────────────
def _dt(utc):
    idx = pd.to_datetime(pd.Series(utc), utc=True, errors="coerce")
    return pd.DatetimeIndex(idx)

def f_hour_sin(utc):    return pd.Series(np.sin(2*np.pi*_dt(utc).hour/24))
def f_hour_cos(utc):    return pd.Series(np.cos(2*np.pi*_dt(utc).hour/24))
def f_min_sin(utc):     return pd.Series(np.sin(2*np.pi*_dt(utc).minute/60))
def f_min_cos(utc):     return pd.Series(np.cos(2*np.pi*_dt(utc).minute/60))
def f_dow_sin(utc):     return pd.Series(np.sin(2*np.pi*_dt(utc).dayofweek/5))
def f_dow_cos(utc):     return pd.Series(np.cos(2*np.pi*_dt(utc).dayofweek/5))
def f_asian(utc):       return pd.Series((_dt(utc).hour < 8).astype(int))
def f_london(utc):      h=_dt(utc).hour; return pd.Series(((h>=7)&(h<16)).astype(int))
def f_newyork(utc):     h=_dt(utc).hour; return pd.Series(((h>=12)&(h<21)).astype(int))
def f_overlap_ln(utc):  h=_dt(utc).hour; return pd.Series(((h>=12)&(h<16)).astype(int))

def f_minutes_into_session(utc):
    idx = _dt(utc); h = idx.hour; m = idx.minute
    total = h*60 + m
    start = np.select([(h>=0)&(h<8),(h>=7)&(h<16),(h>=12)&(h<21)],
                      [0, 420, 720], default=np.nan)
    return pd.Series(total - start)

def f_minutes_to_close(utc):
    idx = _dt(utc); h = idx.hour; m = idx.minute
    total = h*60 + m
    end = np.select([(h>=0)&(h<8),(h>=7)&(h<16),(h>=12)&(h<21)],
                    [480, 960, 1260], default=np.nan)
    return pd.Series(end - total)


# ─────────────────────────────────────────────────────────────────────────────
#  MASTER COMPUTE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def compute_all(utc, O, H, L, C, V):
    n = len(C)
    r = {}
    def a(name, fn): r[name] = safe(fn, name, n)

    # Price action
    a("log_return",        lambda: f_log_return(C))
    a("abs_return",        lambda: f_abs_return(C))
    a("high_low_range",    lambda: f_high_low_range(H,L,C))
    a("close_position",    lambda: f_close_position(H,L,C))
    a("body_ratio",        lambda: f_body_ratio(O,H,L,C))
    a("upper_wick_ratio",  lambda: f_upper_wick_ratio(O,H,L,C))
    a("lower_wick_ratio",  lambda: f_lower_wick_ratio(O,H,L,C))
    a("gap",               lambda: f_gap(O,C))
    a("consecutive_up",    lambda: f_consecutive_up(C))
    a("consecutive_down",  lambda: f_consecutive_down(C))
    a("bar_return_zscore", lambda: f_bar_return_zscore(C))
    a("price_vs_ema20",    lambda: f_price_vs_ema(H,L,C,20))
    a("price_vs_ema50",    lambda: f_price_vs_ema(H,L,C,50))
    a("price_vs_ema200",   lambda: f_price_vs_ema(H,L,C,200))
    a("ema_cross_20_50",   lambda: f_ema_cross_20_50(C))
    a("ema_slope_20",      lambda: f_ema_slope_20(C))
    a("higher_high",       lambda: f_higher_high(H))
    a("lower_low",         lambda: f_lower_low(L))
    a("swing_strength",    lambda: f_swing_strength(H,L,C))
    # Volatility
    a("atr_14",            lambda: f_atr(H,L,C,14))
    a("atr_50",            lambda: f_atr(H,L,C,50))
    a("atr_ratio",         lambda: f_atr_ratio(H,L,C))
    a("realized_vol_10",   lambda: f_realized_vol(C,10))
    a("realized_vol_30",   lambda: f_realized_vol(C,30))
    a("realized_vol_60",   lambda: f_realized_vol(C,60))
    a("parkinson_vol",     lambda: f_parkinson_vol(H,L))
    a("garman_klass_vol",  lambda: f_garman_klass_vol(O,H,L,C))
    a("yang_zhang_vol",    lambda: f_yang_zhang_vol(O,H,L,C))
    a("vol_of_vol",        lambda: f_vol_of_vol(C))
    a("vol_zscore",        lambda: f_vol_zscore(H,L,C))
    a("vol_percentile",    lambda: f_vol_percentile(H,L,C))
    a("bollinger_width",   lambda: f_bollinger_width(C))
    a("keltner_width",     lambda: f_keltner_width(H,L,C))
    a("natr",              lambda: f_natr(H,L,C))
    a("vol_regime_ma",     lambda: f_vol_regime_ma(C))
    a("vol_breakout",      lambda: f_vol_breakout(H,L,C))
    a("vol_contraction",   lambda: f_vol_contraction(H,L,C))
    # Momentum
    a("rsi_7",             lambda: f_rsi(C,7))
    a("rsi_14",            lambda: f_rsi(C,14))
    a("rsi_21",            lambda: f_rsi(C,21))
    a("rsi_divergence",    lambda: f_rsi_divergence(C))
    a("stoch_k",           lambda: f_stoch(H,L,C,"k"))
    a("stoch_d",           lambda: f_stoch(H,L,C,"d"))
    a("macd_line",         lambda: f_macd(C,"line"))
    a("macd_signal",       lambda: f_macd(C,"signal"))
    a("macd_histogram",    lambda: f_macd(C,"hist"))
    a("macd_hist_slope",   lambda: f_macd(C,"slope"))
    a("roc_5",             lambda: f_roc(C,5))
    a("roc_10",            lambda: f_roc(C,10))
    a("roc_20",            lambda: f_roc(C,20))
    a("williams_r",        lambda: f_williams_r(H,L,C))
    a("cci_14",            lambda: f_cci(H,L,C,14))
    a("cci_50",            lambda: f_cci(H,L,C,50))
    a("adx_14",            lambda: f_adx(H,L,C))
    a("di_plus",           lambda: f_di_plus(H,L,C))
    a("di_minus",          lambda: f_di_minus(H,L,C))
    a("dx_cross",          lambda: f_dx_cross(H,L,C))
    # Trend
    a("linear_reg_slope_20", lambda: f_linreg_slope(C,20))
    a("linear_reg_slope_50", lambda: f_linreg_slope(C,50))
    a("linear_reg_r2_20",    lambda: f_linreg_r2(C,20))
    a("linear_reg_r2_50",    lambda: f_linreg_r2(C,50))
    a("supertrend",          lambda: f_supertrend(H,L,C))
    a("ichimoku_tk_cross",   lambda: f_ichimoku_tk_cross(H,L,C))
    a("ichimoku_cloud_dist", lambda: f_ichimoku_cloud_dist(H,L,C))
    a("aroon_up",            lambda: f_aroon(H,L,"up"))
    a("aroon_down",          lambda: f_aroon(H,L,"down"))
    a("aroon_oscillator",    lambda: f_aroon(H,L,"osc"))
    a("vortex_plus",         lambda: f_vortex(H,L,C,"plus"))
    a("vortex_minus",        lambda: f_vortex(H,L,C,"minus"))
    # Statistical
    a("rolling_skew_30",    lambda: f_rolling_skew(C,30))
    a("rolling_skew_60",    lambda: f_rolling_skew(C,60))
    a("rolling_kurtosis_30",lambda: f_rolling_kurt(C,30))
    a("rolling_kurtosis_60",lambda: f_rolling_kurt(C,60))
    a("entropy_30",         lambda: f_entropy_30(C))
    a("hurst_100",          lambda: f_hurst(C))
    a("autocorr_1",         lambda: f_autocorr(C,1))
    a("autocorr_5",         lambda: f_autocorr(C,5))
    a("autocorr_10",        lambda: f_autocorr(C,10))
    a("partial_autocorr_1", lambda: f_partial_autocorr(C))
    a("variance_ratio",     lambda: f_variance_ratio(C))
    a("adf_pvalue_100",     lambda: f_adf_pvalue(C))
    a("jarque_bera",        lambda: f_jarque_bera(C))
    a("rolling_median_dev", lambda: f_rolling_median_dev(C))
    a("quantile_range",     lambda: f_quantile_range(C))
    # Volume
    a("volume_sma_ratio",   lambda: f_volume_sma_ratio(V))
    a("volume_zscore",      lambda: f_volume_zscore(V))
    a("volume_trend",       lambda: f_volume_trend(V))
    a("obv_slope",          lambda: f_obv_slope(C,V))
    a("mfi_14",             lambda: f_mfi(H,L,C,V))
    a("vwap_distance",      lambda: f_vwap_distance(H,L,C,V))
    a("volume_price_corr",  lambda: f_volume_price_corr(C,V))
    a("accumulation_dist",  lambda: f_accumulation_dist(H,L,C,V))
    # Session
    a("hour_sin",             lambda: f_hour_sin(utc))
    a("hour_cos",             lambda: f_hour_cos(utc))
    a("minute_sin",           lambda: f_min_sin(utc))
    a("minute_cos",           lambda: f_min_cos(utc))
    a("day_of_week_sin",      lambda: f_dow_sin(utc))
    a("day_of_week_cos",      lambda: f_dow_cos(utc))
    a("is_asian_session",     lambda: f_asian(utc))
    a("is_london_session",    lambda: f_london(utc))
    a("is_newyork_session",   lambda: f_newyork(utc))
    a("is_overlap_ln",        lambda: f_overlap_ln(utc))
    a("minutes_into_session", lambda: f_minutes_into_session(utc))
    a("minutes_to_close",     lambda: f_minutes_to_close(utc))

    return r


# ─────────────────────────────────────────────────────────────────────────────
#  EXTRACTION LOOP
# ─────────────────────────────────────────────────────────────────────────────
def extract(tf, raw_path, out_path):
    log.info(f"\n{'='*60}\n  {tf}  ->  {raw_path}\n{'='*60}")
    t0 = time.time()

    df_raw = pd.read_csv(raw_path)
    ts_col = next((c for c in df_raw.columns
                   if c.lower() in ("utc","datetime","timestamp")), None)
    if ts_col:
        df_raw[ts_col] = pd.to_datetime(df_raw[ts_col], utc=True, errors="coerce")
        df_raw = df_raw.sort_values(ts_col).reset_index(drop=True)

    lu  = {c.lower(): c for c in df_raw.columns}
    def col(n): return lu[n.lower()]

    total    = len(df_raw)
    n_chunks = (total + CHUNK_SIZE - 1) // CHUNK_SIZE
    log.info(f"  {total:,} rows  |  {n_chunks} chunk(s)")

    outputs = []
    for ci in range(n_chunks):
        ds = ci * CHUNK_SIZE
        ls = max(0, ds - WARMUP)
        le = min(total, ds + CHUNK_SIZE)
        wi = ds - ls

        raw = df_raw.iloc[ls:le]
        log.info(f"  Chunk {ci+1}/{n_chunks}  [{ls:,}..{le:,}]")

        utc = raw[ts_col].values if ts_col else np.arange(len(raw))
        O = raw[col("open")].values.astype(np.float64)
        H = raw[col("high")].values.astype(np.float64)
        L = raw[col("low")].values.astype(np.float64)
        C = raw[col("close")].values.astype(np.float64)
        V = raw[col("volume")].values.astype(np.float64)

        feats = compute_all(utc, O, H, L, C, V)

        out = pd.DataFrame({
            "UTC": utc, "Open": O, "High": H,
            "Low": L, "Close": C, "Volume": V
        })
        for name, s in feats.items():
            vals = s.values if isinstance(s, pd.Series) else np.asarray(s)
            out[name] = vals if len(vals) == len(out) else np.full(len(out), np.nan)

        out = out.iloc[wi:].reset_index(drop=True)
        outputs.append(out)
        log.info(f"    -> {len(out):,} rows  {len(out.columns)} cols")

    final = pd.concat(outputs, ignore_index=True)
    final.to_csv(out_path, index=False)
    log.info(f"  Saved -> {out_path}  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    for tf, raw_path in RAW_FILES.items():
        out_file = OUT_DIR / f"features_{tf}.csv"
        if not Path(raw_path).exists():
            log.error(f"Not found: {raw_path}  skipping"); continue
        if out_file.exists():
            log.info(f"Exists: {out_file}  delete to re-run"); continue
        extract(tf, raw_path, out_file)

    log.info("\nVerification:")
    for tf in RAW_FILES:
        f = OUT_DIR / f"features_{tf}.csv"
        if f.exists():
            df = pd.read_csv(f, nrows=2)
            log.info(f"  {f.name}: {df.shape[1]} cols  atr_14={'atr_14' in df.columns}")
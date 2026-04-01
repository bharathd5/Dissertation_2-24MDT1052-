import numpy as np
import pandas as pd
import talib as ta

from scipy.stats import skew, kurtosis, entropy, jarque_bera
from statsmodels.tsa.stattools import adfuller, pacf

import talib as ta

import numpy as np
import pandas as pd
import talib as ta
# ==============================
# Linear Regression Slope (20)
# ==============================


import numpy as np
import pandas as pd
import talib



# =========================
# Vol Regime MA
# =========================
def compute_vol_regime_ma(df: pd.DataFrame) -> pd.Series:
    close = df['close']
    log_returns = np.log(close / close.shift(1))
    factor = np.sqrt(252 * 1440)

    realized_vol_30 = log_returns.rolling(30).std() * factor
    ema = talib.EMA(realized_vol_30, timeperiod=50)

    ema.name = "vol_regime_ma"
    return ema

# =========================
# Vol Percentile
# =========================
def compute_vol_percentile(df: pd.DataFrame) -> pd.Series:
    atr14 = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)

    window = 252 * 1440

    percentile = atr14.rolling(window).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1]
    )

    percentile.name = "vol_percentile"
    return percentile







def compute_lower_wick_ratio(df: pd.DataFrame) -> pd.Series:
    lower_wick_ratio = (df[["open", "close"]].min(axis=1)) / (df["high"] - df["low"])
    lower_wick_ratio = lower_wick_ratio.replace([np.inf, -np.inf], np.nan)
    lower_wick_ratio.name = "lower_wick_ratio"
    return lower_wick_ratio


def compute_gap(df: pd.DataFrame) -> pd.Series:
    gap = df["open"] - df["close"].shift(1)

    gap.name = "gap"
    return gap


def compute_consecutive_up(df: pd.DataFrame) -> pd.Series:
    up = df["close"] > df["close"].shift(1)
    consecutive_up = up.groupby((~up).cumsum()).cumsum()
    consecutive_up.name = "consecutive_up"
    return consecutive_up


def compute_consecutive_down(df: pd.DataFrame) -> pd.Series:
    down = df["close"] < df["close"].shift(1)
    consecutive_down = down.groupby((~down).cumsum()).cumsum()
    consecutive_down.name = "consecutive_down"
    return consecutive_down

def compute_bar_return_2score(df: pd.DataFrame) -> pd.Series:

    close = df["close"]

    returns = close.pct_change()

    rolling_mean = returns.rolling(20).mean()
    rolling_std = returns.rolling(20).std()

    bar_return_2score = rolling_mean / rolling_std

    bar_return_2score = bar_return_2score.replace([np.inf, -np.inf], np.nan)

    bar_return_2score.name = "bar_return_2score"
    return bar_return_2score

def compute_price_vs_ema20(df: pd.DataFrame) -> pd.Series:

    close = df["close"]
    high = df["high"]
    low = df["low"]
    ema20 = ta.EMA(close, timeperiod=20)
    atr14 = ta.ATR(high, low, close, timeperiod=14)
    price_vs_ema20 = (close - ema20) / atr14
    price_vs_ema20 = price_vs_ema20.replace([np.inf, -np.inf], np.nan)
    price_vs_ema20.name = "price_vs_ema20"
    return price_vs_ema20

def compute_price_vs_ema50(df: pd.DataFrame) -> pd.Series:

    close = df["close"]
    high = df["high"]
    low = df["low"]
    ema50 = ta.EMA(close, timeperiod=50)
    atr14 = ta.ATR(high, low, close, timeperiod=14)
    price_vs_ema50 = (close - ema50) / atr14
    price_vs_ema50 = price_vs_ema50.replace([np.inf, -np.inf], np.nan)
    price_vs_ema50.name = "price_vs_ema50"
    return price_vs_ema50


def compute_price_vs_ema200(df: pd.DataFrame) -> pd.Series:

    close = df["close"]
    high = df["high"]
    low = df["low"]
    ema200 = ta.EMA(close, timeperiod=200)
    atr14 = ta.ATR(high, low, close, timeperiod=14)
    price_vs_ema200 = (close - ema200) / atr14
    price_vs_ema200 = price_vs_ema200.replace([np.inf, -np.inf], np.nan)
    price_vs_ema200.name = "price_vs_ema200"
    return price_vs_ema200

def compute_log_return(df: pd.DataFrame) -> pd.Series:
    log_return = np.log(df["close"] / df["close"].shift(1))

    log_return = log_return.replace([np.inf, -np.inf], np.nan)
    log_return.name = "log_return"

    return log_return


def compute_abs_return(df: pd.DataFrame) -> pd.Series:
    log_return = np.log(df["close"] / df["close"].shift(1))

    log_return = log_return.replace([np.inf, -np.inf], np.nan)

    abs_return = np.abs(log_return)
    abs_return.name = "abs_return"

    return abs_return


def compute_high_low_range(df: pd.DataFrame) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_close = close.shift(1)

    # True Range (TR)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    # ATR(14)
    atr = tr.rolling(window=14, min_periods=1).mean()

    high_low_range = (high - low) / atr
    high_low_range = high_low_range.replace([np.inf, -np.inf], np.nan)

    high_low_range.name = "high_low_range"
    return high_low_range


def compute_close_position(df):
    df.columns = [col.lower() for col in df.columns]
    price_range = df['high'] - df['low']
    price_range = price_range.replace(0, np.nan)
    series = (df['close'] - df['low']) / price_range
    series.name = "close_position"
    return series


def compute_body_ratio(df):
    df.columns = [col.lower() for col in df.columns]
    price_range = df['high'] - df['low']
    price_range = price_range.replace(0, np.nan)
    series = (df['close'] - df['open']).abs() / price_range
    series.name = "body_ratio"
    return series


def compute_upper_wick_ratio(df):
    df.columns = [col.lower() for col in df.columns]
    price_range = df['high'] - df['low']
    price_range = price_range.replace(0, np.nan)
    series = (df['high'] - df[['open', 'close']].max(axis=1)) / price_range
    series.name = "upper_wick_ratio"
    return series








# =========================
# ATR(14)
# =========================
def compute_atr_14(df: pd.DataFrame) -> pd.Series:
    high = df[[c for c in df.columns if c.lower() == "high"][0]]
    low = df[[c for c in df.columns if c.lower() == "low"][0]]
    close = df[[c for c in df.columns if c.lower() == "close"][0]]

    atr = talib.ATR(high, low, close, timeperiod=14)
    atr.name = "atr_14"
    return atr


# =========================
# ATR(50)
# =========================
def compute_atr_50(df: pd.DataFrame) -> pd.Series:
    high = df[[c for c in df.columns if c.lower() == "high"][0]]
    low = df[[c for c in df.columns if c.lower() == "low"][0]]
    close = df[[c for c in df.columns if c.lower() == "close"][0]]

    atr = talib.ATR(high, low, close, timeperiod=50)
    atr.name = "atr_50"
    return atr


# =========================
# ATR Ratio
# =========================
def compute_atr_ratio(df: pd.DataFrame) -> pd.Series:
    high = df[[c for c in df.columns if c.lower() == "high"][0]]
    low = df[[c for c in df.columns if c.lower() == "low"][0]]
    close = df[[c for c in df.columns if c.lower() == "close"][0]]

    atr14 = talib.ATR(high, low, close, timeperiod=14)
    atr50 = talib.ATR(high, low, close, timeperiod=50)

    ratio = (atr14 / atr50).replace([np.inf, -np.inf], np.nan)
    ratio.name = "atr_ratio"
    return ratio


# =========================
# Realized Vol (10)
# =========================
def compute_realized_vol_10(df: pd.DataFrame) -> pd.Series:
    close = df[[c for c in df.columns if c.lower() == "close"][0]]

    log_ret = np.log(close / close.shift(1))
    vol = log_ret.rolling(10).std() * np.sqrt(252 * 1440)

    vol.name = "realized_vol_10"
    return vol


# =========================
# Realized Vol (30)
# =========================
def compute_realized_vol_30(df: pd.DataFrame) -> pd.Series:
    close = df[[c for c in df.columns if c.lower() == "close"][0]]

    log_ret = np.log(close / close.shift(1))
    vol = log_ret.rolling(30).std() * np.sqrt(252 * 1440)

    vol.name = "realized_vol_30"
    return vol


# =========================
# Realized Vol (60)
# =========================
def compute_realized_vol_60(df: pd.DataFrame) -> pd.Series:
    close = df[[c for c in df.columns if c.lower() == "close"][0]]

    log_ret = np.log(close / close.shift(1))
    vol = log_ret.rolling(60).std() * np.sqrt(252 * 1440)

    vol.name = "realized_vol_60"
    return vol


# =========================
# Parkinson Volatility
# =========================
def compute_parkinson_vol(df: pd.DataFrame) -> pd.Series:
    high = df[[c for c in df.columns if c.lower() == "high"][0]]
    low = df[[c for c in df.columns if c.lower() == "low"][0]]

    hl_log = np.log(high / low) ** 2
    vol = np.sqrt(hl_log.rolling(30).sum() / (4 * 30 * np.log(2)))

    vol.name = "parkinson_vol"
    return vol


# =========================
# Garman-Klass Volatility
# =========================
def compute_garman_klass_vol(df: pd.DataFrame) -> pd.Series:
    high = df[[c for c in df.columns if c.lower() == "high"][0]]
    low = df[[c for c in df.columns if c.lower() == "low"][0]]
    open_ = df[[c for c in df.columns if c.lower() == "open"][0]]
    close = df[[c for c in df.columns if c.lower() == "close"][0]]

    log_hl = np.log(high / low)
    log_co = np.log(close / open_)

    gk = 0.5 * log_hl**2 - (2*np.log(2) - 1) * log_co**2
    vol = np.sqrt(gk.rolling(30).mean())

    vol.name = "garman_klass_vol"
    return vol


# =========================
# Yang-Zhang Volatility
# =========================
def compute_yang_zhang_vol(df: pd.DataFrame) -> pd.Series:
    high = df[[c for c in df.columns if c.lower() == "high"][0]]
    low = df[[c for c in df.columns if c.lower() == "low"][0]]
    open_ = df[[c for c in df.columns if c.lower() == "open"][0]]
    close = df[[c for c in df.columns if c.lower() == "close"][0]]

    log_ho = np.log(high / open_)
    log_lo = np.log(low / open_)
    log_co = np.log(close / open_)
    log_oc = np.log(open_ / close.shift(1))

    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    yz = (log_oc**2 + rs).rolling(30).mean()

    vol = np.sqrt(yz)
    vol.name = "yang_zhang_vol"
    return vol


# =========================
# Vol of Vol
# =========================
def compute_vol_of_vol(df: pd.DataFrame) -> pd.Series:
    close = df[[c for c in df.columns if c.lower() == "close"][0]]

    log_ret = np.log(close / close.shift(1))
    vol10 = log_ret.rolling(10).std() * np.sqrt(252 * 1440)

    vol_of_vol = vol10.rolling(20).std()
    vol_of_vol.name = "vol_of_vol"

    return vol_of_vol


# =========================
# Vol Z-Score
# =========================
def compute_vol_zscore(df: pd.DataFrame) -> pd.Series:
    high = df[[c for c in df.columns if c.lower() == "high"][0]]
    low = df[[c for c in df.columns if c.lower() == "low"][0]]
    close = df[[c for c in df.columns if c.lower() == "close"][0]]

    atr = talib.ATR(high, low, close, timeperiod=14)

    z = (atr - atr.rolling(100).mean()) / atr.rolling(100).std()
    z = z.replace([np.inf, -np.inf], np.nan)

    z.name = "vol_zscore"
    return z


# =========================
# Bollinger Width
# =========================
def compute_bollinger_width(df: pd.DataFrame) -> pd.Series:
    close = df[[c for c in df.columns if c.lower() == "close"][0]]

    upper, mid, lower = talib.BBANDS(close, timeperiod=20)

    width = (upper - lower) / mid
    width = width.replace([np.inf, -np.inf], np.nan)

    width.name = "bollinger_width"
    return width


# =========================
# Keltner Width
# =========================
def compute_keltner_width(df: pd.DataFrame) -> pd.Series:
    high = df[[c for c in df.columns if c.lower() == "high"][0]]
    low = df[[c for c in df.columns if c.lower() == "low"][0]]
    close = df[[c for c in df.columns if c.lower() == "close"][0]]

    ema = talib.EMA(close, timeperiod=20)
    atr = talib.ATR(high, low, close, timeperiod=20)

    width = (ema + 2*atr - (ema - 2*atr)) / ema
    width = width.replace([np.inf, -np.inf], np.nan)

    width.name = "keltner_width"
    return width


# =========================
# NATR
# =========================
def compute_natr(df: pd.DataFrame) -> pd.Series:
    high = df[[c for c in df.columns if c.lower() == "high"][0]]
    low = df[[c for c in df.columns if c.lower() == "low"][0]]
    close = df[[c for c in df.columns if c.lower() == "close"][0]]

    atr = talib.ATR(high, low, close, timeperiod=14)

    natr = (atr / close) * 100
    natr = natr.replace([np.inf, -np.inf], np.nan)

    natr.name = "natr"
    return natr


# =========================
# Vol Breakout
# =========================
def compute_vol_breakout(df: pd.DataFrame) -> pd.Series:
    high = df[[c for c in df.columns if c.lower() == "high"][0]]
    low = df[[c for c in df.columns if c.lower() == "low"][0]]
    close = df[[c for c in df.columns if c.lower() == "close"][0]]

    atr14 = talib.ATR(high, low, close, timeperiod=14)
    atr50 = talib.ATR(high, low, close, timeperiod=50)

    breakout = (atr14 > 2 * atr50).astype(int)
    breakout.name = "vol_breakout"

    return breakout


# =========================
# Vol Contraction
# =========================
def compute_vol_contraction(df: pd.DataFrame) -> pd.Series:
    high = df[[c for c in df.columns if c.lower() == "high"][0]]
    low = df[[c for c in df.columns if c.lower() == "low"][0]]
    close = df[[c for c in df.columns if c.lower() == "close"][0]]

    atr14 = talib.ATR(high, low, close, timeperiod=14)
    atr50 = talib.ATR(high, low, close, timeperiod=50)

    contraction = (atr14 < 0.5 * atr50).astype(int)
    contraction.name = "vol_contraction"

    return contraction



def compute_linear_reg_slope_20(df: pd.DataFrame) -> pd.Series:

    close = df["close"]
    slope = ta.LINEARREG_SLOPE(close, timeperiod=20)
    slope.name = "linear_reg_slope_20"
    return slope
# ==============================
# Linear Regression Slope (50)
# ==============================
def compute_linear_reg_slope_50(df: pd.DataFrame) -> pd.Series:

    close = df["close"]
    slope = ta.LINEARREG_SLOPE(close, timeperiod=50)
    slope.name = "linear_reg_slope_50"
    return slope

# ==============================
# Linear Regression R² (20)
# ==============================
def compute_linear_reg_r2_20(df: pd.DataFrame) -> pd.Series:

    close = df["close"]
    r = ta.CORREL(close, ta.LINEARREG(close, timeperiod=20), timeperiod=20)
    r2 = r ** 2
    r2.name = "linear_reg_r2_20"
    return r2
# ==============================
# Linear Regression R² (50)
# ==============================
import numpy as np
import pandas as pd
import talib as ta

# ==============================
# Linear Regression Slope (20)
# ==============================
def compute_linear_reg_slope_20(df: pd.DataFrame) -> pd.Series:

    close = df[[c for c in df.columns if c.lower() == "close"][0]]

    slope = ta.LINEARREG_SLOPE(close, timeperiod=20)
    slope.name = "linear_reg_slope_20"
    return slope


# ==============================
# Linear Regression Slope (50)
# ==============================
def compute_linear_reg_slope_50(df: pd.DataFrame) -> pd.Series:

    close = df[[c for c in df.columns if c.lower() == "close"][0]]

    slope = ta.LINEARREG_SLOPE(close, timeperiod=50)
    slope.name = "linear_reg_slope_50"
    return slope


# ==============================
# Linear Regression R² (20)
# ==============================
def compute_linear_reg_r2_20(df: pd.DataFrame) -> pd.Series:

    close = df[[c for c in df.columns if c.lower() == "close"][0]]

    lr = ta.LINEARREG(close, timeperiod=20)
    r = ta.CORREL(close, lr, timeperiod=20)
    r2 = (r ** 2).replace([np.inf, -np.inf], np.nan)

    r2.name = "linear_reg_r2_20"
    return r2


# ==============================
# Linear Regression R² (50)
# ==============================
def compute_linear_reg_r2_50(df: pd.DataFrame) -> pd.Series:

    close = df[[c for c in df.columns if c.lower() == "close"][0]]

    lr = ta.LINEARREG(close, timeperiod=50)
    r = ta.CORREL(close, lr, timeperiod=50)
    r2 = (r ** 2).replace([np.inf, -np.inf], np.nan)

    r2.name = "linear_reg_r2_50"
    return r2


# ==============================
# Supertrend (Binary Direction)
# ==============================
def compute_supertrend(df: pd.DataFrame) -> pd.Series:

    high = df[[c for c in df.columns if c.lower() == "high"][0]]
    low = df[[c for c in df.columns if c.lower() == "low"][0]]
    close = df[[c for c in df.columns if c.lower() == "close"][0]]

    atr = ta.ATR(high, low, close, timeperiod=10)

    hl2 = (high + low) / 2
    upperband = hl2 + (3 * atr)
    lowerband = hl2 - (3 * atr)

    st = np.where(close > upperband.shift(1), 1,
         np.where(close < lowerband.shift(1), -1, np.nan))

    supertrend = pd.Series(st, index=df.index).ffill()
    supertrend.name = "supertrend"

    return supertrend


# ==============================
# Ichimoku TK Cross (normalized)
# ==============================
def compute_ichimoku_tk_cross(df: pd.DataFrame) -> pd.Series:

    high = df[[c for c in df.columns if c.lower() == "high"][0]]
    low = df[[c for c in df.columns if c.lower() == "low"][0]]
    close = df[[c for c in df.columns if c.lower() == "close"][0]]

    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2

    atr = ta.ATR(high, low, close, timeperiod=14)

    tk_cross = (tenkan - kijun) / atr
    tk_cross = tk_cross.replace([np.inf, -np.inf], np.nan)

    tk_cross.name = "ichimoku_tk_cross"
    return tk_cross


# ==============================
# Ichimoku Cloud Distance
# ==============================
def compute_ichimoku_cloud_dist(df: pd.DataFrame) -> pd.Series:

    high = df[[c for c in df.columns if c.lower() == "high"][0]]
    low = df[[c for c in df.columns if c.lower() == "low"][0]]
    close = df[[c for c in df.columns if c.lower() == "close"][0]]

    span_a = ((high.rolling(9).max() + low.rolling(9).min()) / 2 +
              (high.rolling(26).max() + low.rolling(26).min()) / 2) / 2

    span_b = (high.rolling(52).max() + low.rolling(52).min()) / 2

    cloud_mid = (span_a + span_b) / 2

    atr = ta.ATR(high, low, close, timeperiod=14)

    cloud_dist = (close - cloud_mid) / atr
    cloud_dist = cloud_dist.replace([np.inf, -np.inf], np.nan)

    cloud_dist.name = "ichimoku_cloud_dist"
    return cloud_dist


# ==============================
# Aroon Indicators
# ==============================
def compute_aroon_up(df: pd.DataFrame) -> pd.Series:

    high = df[[c for c in df.columns if c.lower() == "high"][0]]
    low = df[[c for c in df.columns if c.lower() == "low"][0]]

    aroon_up, _ = ta.AROON(high, low, timeperiod=25)
    aroon_up.name = "aroon_up"

    return aroon_up


def compute_aroon_down(df: pd.DataFrame) -> pd.Series:

    high = df[[c for c in df.columns if c.lower() == "high"][0]]
    low = df[[c for c in df.columns if c.lower() == "low"][0]]

    _, aroon_down = ta.AROON(high, low, timeperiod=25)
    aroon_down.name = "aroon_down"

    return aroon_down


def compute_aroon_oscillator(df: pd.DataFrame) -> pd.Series:

    high = df[[c for c in df.columns if c.lower() == "high"][0]]
    low = df[[c for c in df.columns if c.lower() == "low"][0]]

    aroon_up, aroon_down = ta.AROON(high, low, timeperiod=25)

    osc = aroon_up - aroon_down
    osc.name = "aroon_oscillator"

    return osc


# ==============================
# Vortex Indicator
# ==============================
def compute_vortex_plus(df: pd.DataFrame) -> pd.Series:

    high = df[[c for c in df.columns if c.lower() == "high"][0]]
    low = df[[c for c in df.columns if c.lower() == "low"][0]]
    close = df[[c for c in df.columns if c.lower() == "close"][0]]

    tr = ta.TRANGE(high, low, close)

    vm_plus = abs(high - low.shift(1))

    vortex_plus = vm_plus.rolling(14).sum() / tr.rolling(14).sum()
    vortex_plus = vortex_plus.replace([np.inf, -np.inf], np.nan)

    vortex_plus.name = "vortex_plus"
    return vortex_plus


def compute_vortex_minus(df: pd.DataFrame) -> pd.Series:

    high = df[[c for c in df.columns if c.lower() == "high"][0]]
    low = df[[c for c in df.columns if c.lower() == "low"][0]]
    close = df[[c for c in df.columns if c.lower() == "close"][0]]

    tr = ta.TRANGE(high, low, close)

    vm_minus = abs(low - high.shift(1))

    vortex_minus = vm_minus.rolling(14).sum() / tr.rolling(14).sum()
    vortex_minus = vortex_minus.replace([np.inf, -np.inf], np.nan)

    vortex_minus.name = "vortex_minus"
    return vortex_minus







def volume_sma_ratio(df: pd.DataFrame) -> pd.Series:
    sma20 = df["volume"].rolling(window=20).mean()
    ratio = df["volume"] / sma20

    ratio.name = "volume_sma_ratio"
    return ratio


def volume_zscore(df: pd.DataFrame) -> pd.Series:
    mean = df["volume"].rolling(window=50).mean()
    std = df["volume"].rolling(window=50).std()

    zscore = (df["volume"] - mean) / std
    zscore = zscore.replace([np.inf, -np.inf], np.nan)

    zscore.name = "volume_zscore"
    return zscore


def volume_trend(df: pd.DataFrame) -> pd.Series:
    sma20 = df["volume"].rolling(window=20).mean()
    slope = (sma20 - sma20.shift(5)) / 5

    slope.name = "volume_trend"
    return slope


def obv_slope(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff()).fillna(0)
    obv = (direction * df["volume"]).cumsum()

    slope = (obv - obv.shift(20)) / 20

    slope.name = "obv_slope"
    return slope


def mfi_14(df: pd.DataFrame) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    mf = tp * df["volume"]

    positive_mf = mf.where(tp > tp.shift(1), 0)
    negative_mf = mf.where(tp < tp.shift(1), 0)

    pos_sum = positive_mf.rolling(14).sum()
    neg_sum = negative_mf.rolling(14).sum()

    mfi = 100 - (100 / (1 + (pos_sum / neg_sum)))
    mfi = mfi.replace([np.inf, -np.inf], np.nan)

    mfi.name = "mfi_14"
    return mfi


def vwap_distance(df: pd.DataFrame) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3

    cumulative_vol = df["volume"].cumsum()
    cumulative_vp = (tp * df["volume"]).cumsum()

    vwap = cumulative_vp / cumulative_vol

    # ATR (14)
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean()

    distance = (df["close"] - vwap) / atr
    distance = distance.replace([np.inf, -np.inf], np.nan)

    distance.name = "vwap_distance"
    return distance


def volume_price_corr(df: pd.DataFrame) -> pd.Series:
    returns = df["close"].pct_change().abs()

    corr = df["volume"].rolling(20).corr(returns)

    corr.name = "volume_price_corr"
    return corr


def accumulation_dist(df: pd.DataFrame) -> pd.Series:
    clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"])
    clv = clv.replace([np.inf, -np.inf], np.nan).fillna(0)

    adl = (clv * df["volume"]).cumsum()

    slope = (adl - adl.shift(20)) / 20

    slope.name = "accumulation_dist"
    return slope



def hour_sin(df: pd.DataFrame) -> pd.Series:
    if not isinstance(df.index, pd.DatetimeIndex):
        for col in ["UTC", "Time", "Datetime", "Date"]:
            if col in df.columns:
                idx = pd.to_datetime(df[col], utc=True)
                break
        else:
            idx = pd.to_datetime(df.index, utc=True)
    else:
        idx = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")

    if isinstance(idx, pd.Series):
        idx = pd.DatetimeIndex(idx)

    hour = idx.hour
    result = np.sin(2 * np.pi * hour / 24)

    return pd.Series(result, index=df.index, name="hour_sin")


def minute_sin(df: pd.DataFrame) -> pd.Series:
    if not isinstance(df.index, pd.DatetimeIndex):
        for col in ["UTC", "Time", "Datetime", "Date"]:
            if col in df.columns:
                idx = pd.to_datetime(df[col], utc=True)
                break
        else:
            idx = pd.to_datetime(df.index, utc=True)
    else:
        idx = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")

    if isinstance(idx, pd.Series):
        idx = pd.DatetimeIndex(idx)

    minute = idx.minute
    result = np.sin(2 * np.pi * minute / 60)

    return pd.Series(result, index=df.index, name="minute_sin")

def is_london_session(df: pd.DataFrame) -> pd.Series:
    if not isinstance(df.index, pd.DatetimeIndex):
        for col in ["UTC", "Time", "Datetime", "Date"]:
            if col in df.columns:
                idx = pd.to_datetime(df[col], utc=True)
                break
        else:
            idx = pd.to_datetime(df.index, utc=True)
    else:
        idx = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")

    if isinstance(idx, pd.Series):
        idx = pd.DatetimeIndex(idx)

    hour = idx.hour
    result = ((hour >= 7) & (hour < 16)).astype(int)

    return pd.Series(result, index=df.index, name="is_london_session")

def minutes_into_session(df: pd.DataFrame) -> pd.Series:
    if not isinstance(df.index, pd.DatetimeIndex):
        for col in ["UTC", "Time", "Datetime", "Date"]:
            if col in df.columns:
                idx = pd.to_datetime(df[col], utc=True)
                break
        else:
            idx = pd.to_datetime(df.index, utc=True)
    else:
        idx = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")

    if isinstance(idx, pd.Series):
        idx = pd.DatetimeIndex(idx)

    hour = idx.hour
    minute = idx.minute
    total_minutes = hour * 60 + minute

    session_start = np.select(
        [
            (hour >= 0) & (hour < 8),
            (hour >= 7) & (hour < 16),
            (hour >= 12) & (hour < 21),
        ],
        [0, 7*60, 12*60],
        default=np.nan
    )

    result = total_minutes - session_start
    return pd.Series(result, index=df.index, name="minutes_into_session")


def hour_cos(df: pd.DataFrame) -> pd.Series:
    if not isinstance(df.index, pd.DatetimeIndex):
        for col in ["UTC", "Time", "Datetime", "Date"]:
            if col in df.columns:
                idx = pd.to_datetime(df[col], utc=True)
                break
        else:
            idx = pd.to_datetime(df.index, utc=True)
    else:
        idx = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")

    if isinstance(idx, pd.Series):
        idx = pd.DatetimeIndex(idx)

    hour = idx.hour
    result = np.cos(2 * np.pi * hour / 24)

    return pd.Series(result, index=df.index, name="hour_cos")

def minute_cos(df: pd.DataFrame) -> pd.Series:
    if not isinstance(df.index, pd.DatetimeIndex):
        for col in ["UTC", "Time", "Datetime", "Date"]:
            if col in df.columns:
                idx = pd.to_datetime(df[col], utc=True)
                break
        else:
            idx = pd.to_datetime(df.index, utc=True)
    else:
        idx = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")

    if isinstance(idx, pd.Series):
        idx = pd.DatetimeIndex(idx)

    minute = idx.minute
    result = np.cos(2 * np.pi * minute / 60)

    return pd.Series(result, index=df.index, name="minute_cos")


def day_of_week_sin(df: pd.DataFrame) -> pd.Series:
    if not isinstance(df.index, pd.DatetimeIndex):
        for col in ["UTC", "Time", "Datetime", "Date"]:
            if col in df.columns:
                idx = pd.to_datetime(df[col], utc=True)
                break
        else:
            idx = pd.to_datetime(df.index, utc=True)
    else:
        idx = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")

    if isinstance(idx, pd.Series):
        idx = pd.DatetimeIndex(idx)

    day = idx.dayofweek
    result = np.sin(2 * np.pi * day / 5)

    return pd.Series(result, index=df.index, name="day_of_week_sin")

def day_of_week_cos(df: pd.DataFrame) -> pd.Series:
    if not isinstance(df.index, pd.DatetimeIndex):
        for col in ["UTC", "Time", "Datetime", "Date"]:
            if col in df.columns:
                idx = pd.to_datetime(df[col], utc=True)
                break
        else:
            idx = pd.to_datetime(df.index, utc=True)
    else:
        idx = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")

    if isinstance(idx, pd.Series):
        idx = pd.DatetimeIndex(idx)

    day = idx.dayofweek
    result = np.cos(2 * np.pi * day / 5)

    return pd.Series(result, index=df.index, name="day_of_week_cos")

def is_asian_session(df: pd.DataFrame) -> pd.Series:
    if not isinstance(df.index, pd.DatetimeIndex):
        for col in ["UTC", "Time", "Datetime", "Date"]:
            if col in df.columns:
                idx = pd.to_datetime(df[col], utc=True)
                break
        else:
            idx = pd.to_datetime(df.index, utc=True)
    else:
        idx = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")

    if isinstance(idx, pd.Series):
        idx = pd.DatetimeIndex(idx)

    hour = idx.hour
    result = ((hour >= 0) & (hour < 8)).astype(int)

    return pd.Series(result, index=df.index, name="is_asian_session")


def is_newyork_session(df: pd.DataFrame) -> pd.Series:
    if not isinstance(df.index, pd.DatetimeIndex):
        for col in ["UTC", "Time", "Datetime", "Date"]:
            if col in df.columns:
                idx = pd.to_datetime(df[col], utc=True)
                break
        else:
            idx = pd.to_datetime(df.index, utc=True)
    else:
        idx = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")

    if isinstance(idx, pd.Series):
        idx = pd.DatetimeIndex(idx)

    hour = idx.hour
    result = ((hour >= 12) & (hour < 21)).astype(int)

    return pd.Series(result, index=df.index, name="is_newyork_session")


def is_overlap_ln(df: pd.DataFrame) -> pd.Series:
    if not isinstance(df.index, pd.DatetimeIndex):
        for col in ["UTC", "Time", "Datetime", "Date"]:
            if col in df.columns:
                idx = pd.to_datetime(df[col], utc=True)
                break
        else:
            idx = pd.to_datetime(df.index, utc=True)
    else:
        idx = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")

    if isinstance(idx, pd.Series):
        idx = pd.DatetimeIndex(idx)

    hour = idx.hour
    result = ((hour >= 12) & (hour < 16)).astype(int)

    return pd.Series(result, index=df.index, name="is_overlap_ln")

def minutes_to_close(df: pd.DataFrame) -> pd.Series:
    if not isinstance(df.index, pd.DatetimeIndex):
        for col in ["UTC", "Time", "Datetime", "Date"]:
            if col in df.columns:
                idx = pd.to_datetime(df[col], utc=True)
                break
        else:
            idx = pd.to_datetime(df.index, utc=True)
    else:
        idx = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")

    if isinstance(idx, pd.Series):
        idx = pd.DatetimeIndex(idx)

    hour = idx.hour
    minute = idx.minute
    total_minutes = hour * 60 + minute

    session_end = np.select(
        [
            (hour >= 0) & (hour < 8),
            (hour >= 7) & (hour < 16),
            (hour >= 12) & (hour < 21),
        ],
        [8*60, 16*60, 21*60],
        default=np.nan
    )

    result = session_end - total_minutes
    return pd.Series(result, index=df.index, name="minutes_to_close")








def ema_cross_20_50(df: pd.DataFrame) -> pd.Series:
    ema20 = df["close"].ewm(span=20, adjust=False).mean()
    ema50 = df["close"].ewm(span=50, adjust=False).mean()

    ema_cross = ema20 - ema50

    # Normalize using rolling std (z-score style)
    normalized = ema_cross / ema_cross.rolling(window=20).std()

    normalized = normalized.replace([np.inf, -np.inf], np.nan)
    normalized.name = "ema_cross_20_50"
    return normalized


def ema_slope_20(df: pd.DataFrame) -> pd.Series:
    ema20 = df["close"].ewm(span=20, adjust=False).mean()

    # slope = change over 5 bars
    slope = (ema20 - ema20.shift(5)) / 5

    slope.name = "ema_slope_20"
    return slope


def higher_high(df: pd.DataFrame) -> pd.Series:
    prev_max = df["high"].rolling(window=5).max().shift(1)

    higher_high = (df["high"] > prev_max).astype(int)

    higher_high.name = "higher_high"
    return higher_high


def lower_low(df: pd.DataFrame) -> pd.Series:
    prev_min = df["low"].rolling(window=5).min().shift(1)

    lower_low = (df["low"] < prev_min).astype(int)

    lower_low.name = "lower_low"
    return lower_low


def swing_strength(df: pd.DataFrame) -> pd.Series:
    # --- ATR (14) ---
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean()

    # --- Simple swing detection (local extrema) ---
    swing_high = (df["high"] > df["high"].shift(1)) & (df["high"] > df["high"].shift(-1))
    swing_low = (df["low"] < df["low"].shift(1)) & (df["low"] < df["low"].shift(-1))

    swing_points = pd.Series(np.nan, index=df.index)
    swing_points[swing_high] = df["high"]
    swing_points[swing_low] = df["low"]

    # forward fill nearest swing
    nearest_swing = swing_points.ffill()

    # --- distance normalized by ATR ---
    distance = (df["close"] - nearest_swing).abs()
    swing_strength = distance / atr

    swing_strength = swing_strength.replace([np.inf, -np.inf], np.nan)
    swing_strength.name = "swing_strength"
    return swing_strength
# 1. Rolling Skew
def rolling_skew_30(df):
    df.columns = [col.lower() for col in df.columns]
    r = df['close'].pct_change()
    s = r.rolling(30).apply(skew, raw=True)
    s.name = "rolling_skew_30"
    return s


def rolling_skew_60(df):
    df.columns = [col.lower() for col in df.columns]
    r = df['close'].pct_change()
    s = r.rolling(60).apply(skew, raw=True)
    s.name = "rolling_skew_60"
    return s


# 2. Rolling Kurtosis
def rolling_kurtosis_30(df):
    df.columns = [col.lower() for col in df.columns]
    r = df['close'].pct_change()
    s = r.rolling(30).apply(kurtosis, raw=True)
    s.name = "rolling_kurtosis_30"
    return s


def rolling_kurtosis_60(df):
    df.columns = [col.lower() for col in df.columns]
    r = df['close'].pct_change()
    s = r.rolling(60).apply(kurtosis, raw=True)
    s.name = "rolling_kurtosis_60"
    return s


# 3. Entropy
def entropy_30(df):
    df.columns = [col.lower() for col in df.columns]
    r = df['close'].pct_change()

    s = r.rolling(30).apply(
        lambda x: entropy((np.histogram(x, bins=10)[0] / np.sum(np.histogram(x, bins=10)[0]))[
            (np.histogram(x, bins=10)[0] / np.sum(np.histogram(x, bins=10)[0])) > 0
        ]),
        raw=True
    )
    s.name = "entropy_30"
    return s


# 4. Hurst Exponent
def hurst_100(df):
    df.columns = [col.lower() for col in df.columns]
    r = df['close'].pct_change()

    s = r.rolling(100).apply(
        lambda x: np.polyfit(
            np.log(range(2, 20)),
            np.log([np.std(np.subtract(x[lag:], x[:-lag])) for lag in range(2, 20)]),
            1
        )[0],
        raw=False
    )
    s.name = "hurst_100"
    return s


# 5. Autocorrelation
def autocorr_1(df):
    df.columns = [col.lower() for col in df.columns]
    r = df['close'].pct_change()
    s = r.rolling(50).apply(lambda x: pd.Series(x).autocorr(lag=1), raw=False)
    s.name = "autocorr_1"
    return s


def autocorr_5(df):
    df.columns = [col.lower() for col in df.columns]
    r = df['close'].pct_change()
    s = r.rolling(50).apply(lambda x: pd.Series(x).autocorr(lag=5), raw=False)
    s.name = "autocorr_5"
    return s


def autocorr_10(df):
    df.columns = [col.lower() for col in df.columns]
    r = df['close'].pct_change()
    s = r.rolling(50).apply(lambda x: pd.Series(x).autocorr(lag=10), raw=False)
    s.name = "autocorr_10"
    return s


# 6. Partial Autocorrelation
def partial_autocorr_1(df):
    df.columns = [col.lower() for col in df.columns]
    r = df['close'].pct_change()
    s = r.rolling(50).apply(lambda x: pacf(x, nlags=1)[1], raw=False)
    s.name = "partial_autocorr_1"
    return s


# 7. Variance Ratio
def variance_ratio(df):
    df.columns = [col.lower() for col in df.columns]
    r = df['close'].pct_change()

    s = r.rolling(60).apply(
        lambda x: (np.var(np.add.reduceat(x, np.arange(0, len(x), 2))) /
                   (2 * np.var(x))) if np.var(x) != 0 else np.nan,
        raw=True
    )
    s.name = "variance_ratio"
    return s


# 8. ADF p-value
def adf_pvalue_100(df):
    df.columns = [col.lower() for col in df.columns]
    r = df['close'].pct_change()

    s = r.rolling(100).apply(
        lambda x: adfuller(x)[1] if len(x.dropna()) > 0 else np.nan,
        raw=False
    )
    s.name = "adf_pvalue_100"
    return s


# 9. Jarque-Bera
def jarque_bera_60(df):
    df.columns = [col.lower() for col in df.columns]
    r = df['close'].pct_change()
    s = r.rolling(60).apply(lambda x: jarque_bera(x)[0], raw=True)
    s.name = "jarque_bera"
    return s


# 10. Median Absolute Deviation
def rolling_median_dev(df):
    df.columns = [col.lower() for col in df.columns]
    r = df['close'].pct_change()

    s = r.rolling(60).apply(
        lambda x: np.median(np.abs(x - np.median(x))),
        raw=True
    )
    s.name = "rolling_median_dev"
    return s


# 11. Quantile Range
def quantile_range(df):
    df.columns = [col.lower() for col in df.columns]
    r = df['close'].pct_change()

    s = r.rolling(60).apply(
        lambda x: np.percentile(x, 75) - np.percentile(x, 25),
        raw=True
    )
    s.name = "quantile_range"
    return s







# ---------- RSI ----------
def rsi_7(df):
    rsi = ta.RSI(df["Close"], timeperiod=7)
    rsi.name = "rsi_7"
    return rsi


def rsi_14(df):
    rsi = ta.RSI(df["Close"], timeperiod=14)
    rsi.name = "rsi_14"
    return rsi


def rsi_21(df):
    rsi = ta.RSI(df["Close"], timeperiod=21)
    rsi.name = "rsi_21"
    return rsi


def rsi_divergence(df):
    rsi = ta.RSI(df["Close"], timeperiod=14)

    price_high = df["Close"] > df["Close"].rolling(5).max().shift(1)
    rsi_not_high = rsi < pd.Series(rsi).rolling(5).max().shift(1)

    out = (price_high & rsi_not_high).astype(int)
    out.name = "rsi_divergence"
    return out


# ---------- STOCH ----------
def stoch_k(df):
    k, _ = ta.STOCH(
        df["High"], df["Low"], df["Close"],
        fastk_period=14, slowk_period=3, slowk_matype=0,
        slowd_period=3, slowd_matype=0
    )
    k.name = "stoch_k"
    return k


def stoch_d(df):
    _, d = ta.STOCH(
        df["High"], df["Low"], df["Close"],
        fastk_period=14, slowk_period=3, slowk_matype=0,
        slowd_period=3, slowd_matype=0
    )
    d.name = "stoch_d"
    return d


# ---------- MACD ----------
def macd_line(df):
    macd, _, _ = ta.MACD(df["Close"], fastperiod=12, slowperiod=26, signalperiod=9)
    macd.name = "macd_line"
    return macd


def macd_signal(df):
    _, signal, _ = ta.MACD(df["Close"], fastperiod=12, slowperiod=26, signalperiod=9)
    signal.name = "macd_signal"
    return signal


def macd_histogram(df):
    _, _, hist = ta.MACD(df["Close"], fastperiod=12, slowperiod=26, signalperiod=9)
    hist.name = "macd_histogram"
    return hist


def macd_hist_slope(df):
    _, _, hist = ta.MACD(df["Close"], fastperiod=12, slowperiod=26, signalperiod=9)
    slope = pd.Series(hist).diff(3)
    slope.name = "macd_hist_slope"
    return slope


# ---------- ROC ----------
def roc_5(df):
    roc = ta.ROC(df["Close"], timeperiod=5)
    roc.name = "roc_5"
    return roc


def roc_10(df):
    roc = ta.ROC(df["Close"], timeperiod=10)
    roc.name = "roc_10"
    return roc


def roc_20(df):
    roc = ta.ROC(df["Close"], timeperiod=20)
    roc.name = "roc_20"
    return roc


# ---------- Williams %R ----------
def williams_r(df):
    wr = ta.WILLR(df["High"], df["Low"], df["Close"], timeperiod=14)
    wr.name = "williams_r"
    return wr


# ---------- CCI ----------
def cci_14(df):
    cci = ta.CCI(df["High"], df["Low"], df["Close"], timeperiod=14)
    cci.name = "cci_14"
    return cci


def cci_50(df):
    cci = ta.CCI(df["High"], df["Low"], df["Close"], timeperiod=50)
    cci.name = "cci_50"
    return cci


# ---------- ADX / DI ----------
def adx_14(df):
    adx = ta.ADX(df["High"], df["Low"], df["Close"], timeperiod=14)
    adx.name = "adx_14"
    return adx


def di_plus(df):
    di = ta.PLUS_DI(df["High"], df["Low"], df["Close"], timeperiod=14)
    di.name = "di_plus"
    return di


def di_minus(df):
    di = ta.MINUS_DI(df["High"], df["Low"], df["Close"], timeperiod=14)
    di.name = "di_minus"
    return di


def dx_cross(df):
    plus = ta.PLUS_DI(df["High"], df["Low"], df["Close"], timeperiod=14)
    minus = ta.MINUS_DI(df["High"], df["Low"], df["Close"], timeperiod=14)

    dx = (plus - minus) / (plus + minus)
    dx = dx.replace([np.inf, -np.inf], np.nan)

    dx.name = "dx_cross"
    return dx


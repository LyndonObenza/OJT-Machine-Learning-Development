from tvDatafeed import TvDatafeed, Interval
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from ta.trend import PSARIndicator, SMAIndicator, EMAIndicator, ADXIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange


feature_columns = [
    'hl_range_pct', 'close_pct_change', 'volume_ratio', 'trend', 'ma_200_percent',
    'rsi', 'bb_position', 'rsi_slope', 'rsi_divergence', 'macd_histogram',
    'macd_hist_slope', 'adx', 'adx_slope', 'upper_wick_atr', 'lower_wick_atr',
    'bars_since_swing_high', 'bars_since_swing_low', 'close_minus_ema',
    'close_minus_psar', 'return_5', 'volatility_10', 'upper_wick_ratio',
    'lower_wick_ratio', 'body_ratio', 'ATR_percent', 'bb_width',
    'psar_distance_pct', 'fractal_swing_phase',
]

# Forward PSAR bullish/bearish state at t+1 … t+4
# NaN rows (ambiguous: |close - psar| < 0.3%) are dropped via dropna()
target_columns = ['target_t1', 'target_t2', 'target_t3', 'target_t4']

PSAR_THRESHOLD_PCT = 0.003   # 0.3% — label only when price clearly above/below PSAR


def _bars_since_extreme(series: pd.Series, window: int = 20, mode: str = 'max') -> pd.Series:
    arr = series.values
    result = np.full(len(arr), np.nan)
    fn = np.argmax if mode == 'max' else np.argmin
    for i in range(window - 1, len(arr)):
        result[i] = window - 1 - fn(arr[i - window + 1:i + 1])
    return pd.Series(result, index=series.index)


def _fractal_swing_phase(high: pd.Series, low: pd.Series, n: int = 2) -> pd.Series:
    """
    Williams Fractal pivot detection (n bars each side).
    Returns a series: 1 = currently in upswing, 0 = downswing, NaN = not yet determined.
    A pivot high flips phase to 0 (top reached, downswing starts).
    A pivot low  flips phase to 1 (bottom reached, upswing starts).
    """
    h = high.values
    l = low.values
    phase = np.full(len(h), np.nan)
    current = np.nan
    for i in range(n, len(h) - n):
        window_h = h[i - n:i + n + 1]
        window_l = l[i - n:i + n + 1]
        if h[i] == window_h.max():   # pivot high → downswing
            current = 0.0
        elif l[i] == window_l.min(): # pivot low  → upswing
            current = 1.0
        phase[i] = current
    # forward-fill so every bar has a phase label
    result = pd.Series(phase, index=high.index)
    result = result.ffill()
    return result


def get_tradingview_data(symbol: str, exchange: str, tv_interval: Interval):
    data = None
    max_trial = 10

    for trial in range(max_trial):
        if data is not None:
            break
        try:
            print(f"Trial {trial + 1} for {symbol}...")
            tv = TvDatafeed()
            data = tv.get_hist(
                symbol=symbol,
                exchange=exchange,
                interval=tv_interval,
                n_bars=10000,
            )

            data['date'] = data.index
            data = data.rename(columns={
                'open': 'open_price', 'high': 'high_price',
                'low': 'low_price',   'close': 'close_price',
                'volume': 'volume',
            })
            data = data[['date', 'open_price', 'high_price',
                         'low_price', 'close_price', 'volume']]
            data['date'] = pd.to_datetime(data['date'])

            hi  = data['high_price']
            lo  = data['low_price']
            cl  = data['close_price']
            op  = data['open_price']
            vol = data['volume']

            candle_range = hi - lo
            upper_wick   = hi - np.maximum(op, cl)
            lower_wick   = np.minimum(op, cl) - lo
            body         = (cl - op).abs()
            safe_range   = candle_range.replace(0, np.nan)

            # ATR
            atr = AverageTrueRange(hi, lo, cl, window=14).average_true_range()

            # PSAR + trend features
            psar = PSARIndicator(hi, lo, cl).psar()
            data['PSAR']             = psar
            data['trend']            = (cl > psar).astype(int)
            data['psar_distance_pct'] = (cl - psar) / cl * 100  # + = bullish, - = bearish

            # MA200
            ma_200 = SMAIndicator(cl, window=200).sma_indicator()
            data['ma_200_percent'] = (cl - ma_200) / ma_200

            # EMA20
            ema_20 = EMAIndicator(cl, window=20).ema_indicator()

            # RSI
            rsi = RSIIndicator(cl, window=14).rsi()
            data['rsi']            = rsi
            data['rsi_slope']      = rsi.diff(3)
            data['rsi_divergence'] = rsi.diff(5) - cl.pct_change(5) * 100

            # Bollinger Bands
            bb = BollingerBands(cl, window=20, window_dev=2)
            bb_upper  = bb.bollinger_hband()
            bb_lower  = bb.bollinger_lband()
            bb_middle = bb.bollinger_mavg()
            data['bb_position'] = (cl - bb_lower) / (bb_upper - bb_lower)
            data['bb_width']    = (bb_upper - bb_lower) / bb_middle

            # MACD
            macd_ind  = MACD(cl)
            macd_hist = macd_ind.macd_diff()
            data['macd_histogram']  = macd_hist
            data['macd_hist_slope'] = macd_hist.diff(3)

            # ADX
            adx = ADXIndicator(hi, lo, cl, window=14).adx()
            data['adx']       = adx
            data['adx_slope'] = adx.diff(3)

            # Candle structure
            data['upper_wick_atr']   = upper_wick / atr
            data['lower_wick_atr']   = lower_wick / atr
            data['upper_wick_ratio'] = upper_wick / safe_range
            data['lower_wick_ratio'] = lower_wick / safe_range
            data['body_ratio']       = body / safe_range

            # Range / ATR percent
            data['hl_range_pct'] = candle_range / cl * 100
            data['ATR_percent']  = atr / cl * 100

            # Volume
            data['volume_ratio'] = vol / vol.rolling(20).mean()

            # Returns / volatility
            data['close_pct_change'] = cl.pct_change()
            data['return_5']         = cl.pct_change(5)
            data['volatility_10']    = cl.pct_change().rolling(10).std()

            # Distance features
            data['close_minus_ema']  = cl - ema_20
            data['close_minus_psar'] = cl - psar

            # Bars since rolling swing high / low (20-bar window)
            data['bars_since_swing_high'] = _bars_since_extreme(hi, window=20, mode='max')
            data['bars_since_swing_low']  = _bars_since_extreme(lo, window=20, mode='min')

            # Williams Fractal swing phase (n=2 → 5-bar pivot)
            data['fractal_swing_phase'] = _fractal_swing_phase(hi, lo, n=2)

            # ── Multi-horizon targets: forward PSAR bullish state ─────────────────
            # Label 1 = close[t+h] clearly above psar[t+h]  (bullish PSAR)
            # Label 0 = close[t+h] clearly below psar[t+h]  (bearish PSAR)
            # NaN     = within threshold → ambiguous, removed by dropna()
            for h in range(1, 5):
                fwd_cl   = cl.shift(-h)
                fwd_psar = psar.shift(-h)
                dist     = (fwd_cl - fwd_psar) / fwd_cl  # signed relative distance
                t = pd.Series(np.nan, index=data.index)
                t[dist >  PSAR_THRESHOLD_PCT] = 1.0
                t[dist < -PSAR_THRESHOLD_PCT] = 0.0
                data[f'target_t{h}'] = t

            data = data.dropna().reset_index(drop=True)

        except Exception as e:
            print(e)
            continue

    return data


# ── Fetch tickers ─────────────────────────────────────────────────────────────
aapl_df  = get_tradingview_data("AAPL",  "NASDAQ", Interval.in_daily)
nvda_df  = get_tradingview_data("NVDA",  "NASDAQ", Interval.in_daily)
tsla_df  = get_tradingview_data("TSLA",  "NASDAQ", Interval.in_daily)
msft_df  = get_tradingview_data("MSFT",  "NASDAQ", Interval.in_daily)
amd_df   = get_tradingview_data("AMD",   "NASDAQ", Interval.in_daily)
googl_df = get_tradingview_data("GOOGL", "NASDAQ", Interval.in_daily)
avgo_df  = get_tradingview_data("AVGO",  "NASDAQ", Interval.in_daily)
meta_df  = get_tradingview_data("META",  "NASDAQ", Interval.in_daily)
orcl_df  = get_tradingview_data("ORCL",  "NYSE",   Interval.in_daily)
ibm_df   = get_tradingview_data("IBM",   "NYSE",   Interval.in_daily)


# ── Assign tickers and rename date → datetime ─────────────────────────────────
ticker_frames = {
    "AAPL": aapl_df,
    "NVDA": nvda_df,
    "TSLA": tsla_df,
    "MSFT": msft_df,
    "AMD":  amd_df,
    "GOOGL": googl_df,
    "AVGO": avgo_df,
    "META": meta_df,
    "ORCL": orcl_df,
    "IBM":  ibm_df,
}

for ticker, frame in ticker_frames.items():
    frame["ticker"] = ticker
    if "date" in frame.columns and "datetime" not in frame.columns:
        frame.rename(columns={"date": "datetime"}, inplace=True)

# ── Save individual CSVs ──────────────────────────────────────────────────────
for ticker, frame in ticker_frames.items():
    path = f"old_ipynb/{ticker.lower()}_daily.csv"
    frame.to_csv(path, index=False)
    print(f"Saved: {path}  ({len(frame):,} rows)")

# ── Combine and save master CSV ───────────────────────────────────────────────
df = pd.concat(ticker_frames.values(), ignore_index=True)
df = df.sort_values(["ticker", "datetime"]).reset_index(drop=True)
df.to_csv("all_tickers_daily_old.csv", index=False)

print(f"\nTotal rows : {len(df):,}")
print(df["ticker"].value_counts())
print("\nSaved: all_tickers_daily_old.csv")

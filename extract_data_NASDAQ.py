from tvDatafeed import TvDatafeed, Interval
import pandas as pd
import numpy as np
import warnings
import os
warnings.filterwarnings("ignore", category=FutureWarning)

from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, ROCIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, MFIIndicator, ChaikinMoneyFlowIndicator


def compute_indicators(data: pd.DataFrame) -> pd.DataFrame:
    close  = data['close_price']
    high   = data['high_price']
    low    = data['low_price']
    open_  = data['open_price']
    volume = data['volume']

    # --- Trend ---
    data['EMA_10']        = EMAIndicator(close, window=10).ema_indicator()
    data['EMA_20']        = EMAIndicator(close, window=20).ema_indicator()
    data['EMA_50']        = EMAIndicator(close, window=50).ema_indicator()
    data['EMA_200']       = EMAIndicator(close, window=200).ema_indicator()
    data['EMA_ratio']     = close / data['EMA_20']
    data['EMA_50_ratio']  = close / data['EMA_50']
    data['EMA_200_ratio'] = close / data['EMA_200']
    macd                  = MACD(close, window_slow=26, window_fast=12, window_sign=9)
    data['MACD_hist']     = macd.macd_diff()

    # --- Momentum ---
    data['RSI_14'] = RSIIndicator(close, window=14).rsi()
    data['ROC_5']  = ROCIndicator(close, window=5).roc()
    data['ROC_10'] = ROCIndicator(close, window=10).roc()
    data['ROC_20'] = ROCIndicator(close, window=20).roc()

    # --- Directional movement (trend strength + direction) ---
    adx              = ADXIndicator(high, low, close, window=14)
    data['ADX_14']   = adx.adx()
    data['DI_plus']  = adx.adx_pos()
    data['DI_minus'] = adx.adx_neg()

    # --- Volatility ---
    data['ATR_14']       = AverageTrueRange(high, low, close, window=14).average_true_range()
    data['ATR_ratio']    = data['ATR_14'] / data['ATR_14'].rolling(20).mean()
    data['realized_vol'] = close.pct_change().rolling(10).std() * 100
    bb                   = BollingerBands(close, window=20, window_dev=2)
    data['BB_pct']       = bb.bollinger_pband()
    data['BB_width']     = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()

    # --- Volume ---
    data['OBV']           = OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    data['OBV_momentum']  = data['OBV'].pct_change(5) * 100
    data['volume_SMA_20'] = volume.rolling(window=20).mean()
    data['volume_ratio']  = volume / data['volume_SMA_20']
    data['volume_surge']  = volume / volume.rolling(5).mean()
    data['MFI_14']        = MFIIndicator(high, low, close, volume, window=14).money_flow_index()
    data['CMF_20']        = ChaikinMoneyFlowIndicator(high, low, close, volume, window=20).chaikin_money_flow()

    # --- Price action ---
    prev_close = close.shift(1)
    data['overnight_gap']      = (open_ / prev_close - 1) * 100
    data['intraday_direction'] = (close - open_) / open_ * 100
    atr_safe = data['ATR_14'].replace(0, np.nan)
    data['upper_shadow'] = (high - np.maximum(open_.values, close.values)) / atr_safe
    data['lower_shadow'] = (np.minimum(open_.values, close.values) - low) / atr_safe

    # --- 52-week range percentile (min_periods=50 to avoid excessive NaN early) ---
    low_252  = close.rolling(252, min_periods=50).min()
    high_252 = close.rolling(252, min_periods=50).max()
    data['price_percentile_52w'] = (close - low_252) / (high_252 - low_252 + 1e-8)

    # --- Calendar (sin/cos so Mon=0 and Fri=4 encode cyclically) ---
    dow = data['date'].dt.dayofweek
    data['dow_sin'] = np.sin(2 * np.pi * dow / 5)
    data['dow_cos'] = np.cos(2 * np.pi * dow / 5)

    # --- Regime ---
    data['above_ema10'] = (close > data['EMA_10']).astype(float)
    data['above_ema20'] = (close > data['EMA_20']).astype(float)
    data['roc5_pos']    = (data['ROC_5']  > 0).astype(float)
    data['roc20_pos']   = (data['ROC_20'] > 0).astype(float)
    daily_up            = (close.diff(1) > 0).astype(int)
    streak_id           = (daily_up != daily_up.shift()).cumsum()
    data['up_streak']   = daily_up.groupby(streak_id).cumcount() + 1
    data['up_streak']   = data['up_streak'] * daily_up

    # --- Targets ---
    data['price_pct_change']  = close.pct_change(1).shift(-1) * 100
    data['volume_pct_change'] = volume.pct_change(1).shift(-1) * 100

    # --- Lag features ---
    returns = close.pct_change(1) * 100
    for lag in range(1, 6):
        data[f'close_lag_{lag}']   = close.shift(lag)
        data[f'returns_lag_{lag}'] = returns.shift(lag)
        data[f'volume_lag_{lag}']  = volume.shift(lag)

    return data


def fetch_market_returns(symbol: str, exchange: str, col_name: str):
    """Fetch daily close for a market index, return lag-1 return Series indexed by normalized date."""
    try:
        tv = TvDatafeed()
    except Exception as e:
        print(f"[ERROR] TvDatafeed init for {symbol}: {e}")
        return None

    for trial in range(5):
        try:
            raw = tv.get_hist(symbol=symbol, exchange=exchange,
                              interval=Interval.in_daily, n_bars=10000)
            if raw is not None and not raw.empty:
                s = raw['close'].pct_change(1) * 100
                s.index = pd.to_datetime(raw.index).normalize()
                s = s.shift(1)   # lag-1: yesterday's market return
                s.name = col_name
                return s
        except Exception as e:
            print(f"  [ERROR] {symbol} trial {trial+1}: {e}")

    print(f"[FAILED] {symbol}")
    return None


def get_tradingview_data(symbol: str, exchange: str, tv_interval: Interval):
    data      = None
    max_trial = 10

    try:
        tv = TvDatafeed()
    except Exception as e:
        print(f"[ERROR] Could not initialise TvDatafeed for {symbol}: {e}")
        return None

    for trial in range(max_trial):
        if data is not None:
            break
        try:
            print(f"Trial {trial + 1} for {symbol} ({exchange})...")
            raw = tv.get_hist(
                symbol=symbol,
                exchange=exchange,
                interval=tv_interval,
                n_bars=10000,
            )

            if raw is None or raw.empty:
                print(f"  [WARN] No data returned for {symbol} on trial {trial + 1}.")
                continue

            raw['date'] = raw.index
            raw = raw.rename(columns={
                'open':   'open_price',
                'high':   'high_price',
                'low':    'low_price',
                'close':  'close_price',
                'volume': 'volume',
            })
            raw = raw[['date', 'open_price', 'high_price',
                       'low_price', 'close_price', 'volume']]
            raw['date'] = pd.to_datetime(raw['date'])

            data = compute_indicators(raw)
            data = data.dropna().reset_index(drop=True)

        except Exception as e:
            print(f"  [ERROR] {e}")
            continue

    if data is None:
        print(f"[FAILED] Could not retrieve data for {symbol} on {exchange} after {max_trial} trials.")

    return data


# ── Fetch market index returns first (shared across all tickers) ───────────────
print("Fetching QQQ returns...")
qqq_returns = fetch_market_returns("QQQ", "NASDAQ", "QQQ_return_lag1")

print("Fetching SPY returns...")
spy_returns = fetch_market_returns("SPY", "AMEX", "SPY_return_lag1")


# ── Per-ticker exchange map ────────────────────────────────────────────────────
TICKER_EXCHANGE = {
    "NVDA":  "NASDAQ",
    "GOOGL": "NASDAQ",
    "AAPL":  "NASDAQ",
    "GOOG":  "NASDAQ",
    "MSFT":  "NASDAQ",
    "AVGO":  "NASDAQ",
    "TSLA":  "NASDAQ",
    "META":  "NASDAQ",
    "AMD":  "NASDAQ",
    "ASML":  "NASDAQ",
    "LRCX":  "NASDAQ",
    "QQQ": "NASDAQ",
    "SPY":"AMEX",
    
}

# ── Fetch tickers ──────────────────────────────────────────────────────────────
ticker_frames = {}
for ticker, exchange in TICKER_EXCHANGE.items():
    result = get_tradingview_data(ticker, exchange, Interval.in_daily)
    if result is None:
        print(f"[SKIP] {ticker} — no data, skipping.")
        continue
    ticker_frames[ticker] = result

# ── Assign ticker and rename date → datetime ───────────────────────────────────
for ticker, frame in ticker_frames.items():
    frame["ticker"] = ticker
    if "date" in frame.columns and "datetime" not in frame.columns:
        frame.rename(columns={"date": "datetime"}, inplace=True)

# ── Merge QQQ / SPY lag-1 returns ─────────────────────────────────────────────
for ticker, frame in ticker_frames.items():
    date_idx = pd.to_datetime(frame["datetime"]).dt.normalize()

    if qqq_returns is not None:
        qqq_df = qqq_returns.reset_index()
        qqq_df.columns = ["datetime", "QQQ_return_lag1"]
        frame["datetime_norm"] = date_idx
        frame = frame.merge(qqq_df.rename(columns={"datetime": "datetime_norm"}),
                            on="datetime_norm", how="left")
        frame.drop(columns=["datetime_norm"], inplace=True)
    else:
        frame["QQQ_return_lag1"] = np.nan

    if spy_returns is not None:
        spy_df = spy_returns.reset_index()
        spy_df.columns = ["datetime", "SPY_return_lag1"]
        frame["datetime_norm"] = date_idx
        frame = frame.merge(spy_df.rename(columns={"datetime": "datetime_norm"}),
                            on="datetime_norm", how="left")
        frame.drop(columns=["datetime_norm"], inplace=True)
    else:
        frame["SPY_return_lag1"] = np.nan

    # Forward-fill any gaps from holiday misalignment
    frame[["QQQ_return_lag1", "SPY_return_lag1"]] = (
        frame[["QQQ_return_lag1", "SPY_return_lag1"]].ffill()
    )
    ticker_frames[ticker] = frame


# ── Final column order (58 features + targets) ────────────────────────────────
feature_cols = [
    'datetime', 'ticker',
    # OHLCV
    'open_price', 'high_price', 'low_price', 'close_price', 'volume',
    # Trend
    'EMA_10', 'EMA_20', 'EMA_ratio', 'EMA_50_ratio', 'EMA_200_ratio', 'MACD_hist',
    # Momentum
    'RSI_14', 'ROC_5', 'ROC_10', 'ROC_20',
    # Directional movement
    'ADX_14', 'DI_plus', 'DI_minus',
    # Volatility
    'ATR_14', 'ATR_ratio', 'BB_pct', 'BB_width', 'realized_vol',
    # Volume
    'OBV', 'OBV_momentum', 'volume_ratio', 'volume_surge', 'MFI_14', 'CMF_20',
    # Price action
    'overnight_gap', 'intraday_direction', 'upper_shadow', 'lower_shadow',
    # Range / calendar
    'price_percentile_52w', 'dow_sin', 'dow_cos',
    # Market context
    'QQQ_return_lag1', 'SPY_return_lag1',
    # Lags
    'close_lag_1', 'close_lag_2', 'close_lag_3', 'close_lag_4', 'close_lag_5',
    'returns_lag_1', 'returns_lag_2', 'returns_lag_3', 'returns_lag_4', 'returns_lag_5',
    'volume_lag_1', 'volume_lag_2', 'volume_lag_3', 'volume_lag_4', 'volume_lag_5',
    # Regime
    'above_ema10', 'above_ema20', 'roc5_pos', 'roc20_pos', 'up_streak',
    # Targets
    'price_pct_change', 'volume_pct_change',
]

os.makedirs("NASDAQ_DATA", exist_ok=True)

for ticker, frame in ticker_frames.items():
    cols = [c for c in feature_cols if c in frame.columns]
    ticker_frames[ticker] = frame[cols]
    path = f"NASDAQ_DATA/{ticker}_daily.csv"
    frame[cols].to_csv(path, index=False)
    print(f"Saved: {path}  ({len(frame):,} rows)")

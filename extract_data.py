from tvDatafeed import TvDatafeed, Interval
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator, ROCIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, MFIIndicator


def compute_indicators(data: pd.DataFrame) -> pd.DataFrame:
    close  = data['close_price']
    high   = data['high_price']
    low    = data['low_price']
    volume = data['volume']

    # --- Trend ---
    data['EMA_10']     = EMAIndicator(close, window=10).ema_indicator()
    macd               = MACD(close, window_slow=26, window_fast=12, window_sign=9)
    data['MACD_hist']  = macd.macd_diff()

    # --- Momentum ---
    data['RSI_14']     = RSIIndicator(close, window=14).rsi()
    data['ROC_5']      = ROCIndicator(close, window=5).roc()

    # --- Volatility ---
    data['ATR_14']     = AverageTrueRange(high, low, close, window=14).average_true_range()
    bb                 = BollingerBands(close, window=20, window_dev=2)
    data['BB_pct']     = bb.bollinger_pband()           # (close - lower) / (upper - lower)

    # --- Volume ---
    data['OBV']          = OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    data['volume_SMA_20'] = close.rolling(window=20).mean()   # reuse series for ratio
    data['volume_ratio'] = volume / volume.rolling(window=20).mean()
    data['MFI_14']       = MFIIndicator(high, low, close, volume, window=14).money_flow_index()

    # --- Targets ---
    data['price_pct_change']  = close.pct_change(1).shift(-1) * 100   # T+1
    data['volume_pct_change'] = volume.pct_change(1).shift(-1) * 100  # T+1

    return data


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
                'open':   'open_price',
                'high':   'high_price',
                'low':    'low_price',
                'close':  'close_price',
                'volume': 'volume'
            })
            data = data[['date', 'open_price', 'high_price',
                         'low_price', 'close_price', 'volume']]
            data['date'] = pd.to_datetime(data['date'])

            # compute all 14 features + targets
            data = compute_indicators(data)

            # drop NaNs from indicator warmup periods (ATR=14, BB=20, MFI=14)
            data = data.dropna().reset_index(drop=True)

        except Exception as e:
            print(e)
            continue

    return data

ticker_frames = {}

tadawul_tickers = ["2110", "1080"]

for ticker in tadawul_tickers:
    ticker_frames[ticker] = get_tradingview_data(ticker, "TADAWUL", Interval.in_daily)


# ── Assign ticker and rename date → datetime ──────────────────────────────────
for ticker, frame in ticker_frames.items():
    frame["ticker"] = ticker
    if "date" in frame.columns and "datetime" not in frame.columns:
        frame.rename(columns={"date": "datetime"}, inplace=True)


# ── Final column order ────────────────────────────────────────────────────────
feature_cols = [
    'datetime', 'ticker',
    # OHLCV
    'open_price', 'high_price', 'low_price', 'close_price', 'volume',
    # Trend
    'EMA_10', 'MACD_hist',
    # Momentum
    'RSI_14', 'ROC_5',
    # Volatility
    'ATR_14', 'BB_pct',
    # Volume
    'OBV', 'volume_ratio', 'MFI_14',
    # Targets
    'price_pct_change', 'volume_pct_change'
]

for ticker, frame in ticker_frames.items():
    ticker_frames[ticker] = frame[feature_cols]


# ── Save individual CSVs ──────────────────────────────────────────────────────
for ticker, frame in ticker_frames.items():
    path = f"{ticker}_daily.csv"
    frame.to_csv(path, index=False)
    print(f"Saved: {path}  ({len(frame):,} rows)")
    
# tasi_df = get_tradingview_data("TASI", "TADAWUL", Interval.in_daily)
# tasi_df["ticker"] = "TASI"
# tasi_df.rename(columns={"date": "datetime"}, inplace=True)
# tasi_df.to_csv("tasi_daily.csv", index=False)
# print(f"Saved: tasi_daily.csv  ({len(tasi_df):,} rows)")

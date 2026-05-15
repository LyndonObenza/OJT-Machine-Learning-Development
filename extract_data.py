
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
    data['EMA_20']     = EMAIndicator(close, window=20).ema_indicator()        
    data['EMA_ratio']  = close / data['EMA_20']                                
    macd               = MACD(close, window_slow=26, window_fast=12, window_sign=9)
    data['MACD_hist']  = macd.macd_diff()

    # --- Momentum ---
    data['RSI_14']     = RSIIndicator(close, window=14).rsi()
    data['ROC_5']      = ROCIndicator(close, window=5).roc()
    data['ROC_10']     = ROCIndicator(close, window=10).roc()                  
    data['ROC_20']     = ROCIndicator(close, window=20).roc()                  

    # --- Volatility ---
    data['ATR_14']      = AverageTrueRange(high, low, close, window=14).average_true_range()
    data['ATR_ratio']   = data['ATR_14'] / data['ATR_14'].rolling(20).mean()  
    data['realized_vol'] = close.pct_change().rolling(10).std() * 100         
    bb                  = BollingerBands(close, window=20, window_dev=2)
    data['BB_pct']      = bb.bollinger_pband()

    # --- Volume ---
    data['OBV']          = OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    data['OBV_momentum'] = data['OBV'].pct_change(5) * 100                
    data['volume_SMA_20'] = volume.rolling(window=20).mean()
    data['volume_ratio'] = volume / data['volume_SMA_20']
    data['volume_surge'] = volume / volume.rolling(5).mean()                 
    data['MFI_14']       = MFIIndicator(high, low, close, volume, window=14).money_flow_index()
    
    # --- Regime features ---  ← ADD HERE, before Targets
    data['above_ema10'] = (close > data['EMA_10']).astype(float)
    data['above_ema20'] = (close > data['EMA_20']).astype(float)
    data['roc5_pos']    = (data['ROC_5']  > 0).astype(float)
    data['roc20_pos']   = (data['ROC_20'] > 0).astype(float)
    
    # Consecutive up-day streak
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
        data[f'close_lag_{lag}'] = close.shift(lag)
        data[f'returns_lag_{lag}'] = returns.shift(lag)
        data[f'volume_lag_{lag}'] = volume.shift(lag)

    return data


def get_tradingview_data(symbol: str, exchange: str, tv_interval: Interval):
    data      = None
    max_trial = 10

    for trial in range(max_trial):
        if data is not None:
            break
        try:
            print(f"Trial {trial + 1} for {symbol}...")
            tv   = TvDatafeed()
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

            data = compute_indicators(data)
            data = data.dropna().reset_index(drop=True)

        except Exception as e:
            print(e)
            continue

    return data

#  problem with retrieving  "8270"
# ── Fetch tickers ─────────────────────────────────────────────────────────────
ticker_frames   = {}
tadawul_tickers = [
"8300", 
    "1010", "8100", 
"7030", "7010", "2050", "2060",
"1180", "8200", "1050", "1030", "1150", "3080", "3090",
"4003", "2020", "1140", "7020", "2370", "3030", "1020",
"2150", "4260", "3020", "4001", "1320", "4007", "3040",
"8280", "6070", "3010", "2270", "3003", "1120", "8170",
"1214", "8010", "3002", "3050", "8030", "4250", "3005",
"2230", "4280", "4020", "4004", "8210", "2280", "1212",
"4002", "8012", "4150", "8160", "8070", "3004", "2080",
"3060", "1302", "8020", "4300", "4090", "6001", "6004",
"7040", "2310", "4006", "2300", "8230", "4200", "4100",
"2070", "2190", "2250", "1211", "8180", "2290", "2240",
"6020", "1301", "4180", "8240", "4210", "8040", "4130",
"8310", "4290", "4050", "4040", "1210", "4080", "2010",
"4170", "4240"
                    ]

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
    'open_price', 'high_price', 'low_price', 'close_price', 'volume',
    'EMA_10', 'EMA_20', 'EMA_ratio', 'MACD_hist',
    'RSI_14', 'ROC_5', 'ROC_10', 'ROC_20',
    'ATR_14', 'ATR_ratio', 'BB_pct', 'realized_vol',
    'OBV', 'OBV_momentum', 'volume_ratio', 'volume_surge', 'MFI_14',
    'close_lag_1', 'close_lag_2', 'close_lag_3', 'close_lag_4', 'close_lag_5',
    'returns_lag_1', 'returns_lag_2', 'returns_lag_3', 'returns_lag_4', 'returns_lag_5',
    'volume_lag_1', 'volume_lag_2', 'volume_lag_3', 'volume_lag_4', 'volume_lag_5',
    'above_ema10', 'above_ema20', 'roc5_pos', 'roc20_pos', 'up_streak','price_pct_change', 'volume_pct_change'
]

for ticker, frame in ticker_frames.items():
    ticker_frames[ticker] = frame[feature_cols]

# ── Save individual CSVs ──────────────────────────────────────────────────────
for ticker, frame in ticker_frames.items():
    path = f"{ticker}_daily.csv"
    frame.to_csv(path, index=False)
    print(f"Saved: {path}  ({len(frame):,} rows)")
    
tasi_df = get_tradingview_data("TASI", "TADAWUL", Interval.in_daily)
tasi_df["ticker"] = "TASI"
tasi_df.rename(columns={"date": "datetime"}, inplace=True)
tasi_df.to_csv("tasi_daily.csv", index=False)
print(f"Saved: tasi_daily.csv  ({len(tasi_df):,} rows)")

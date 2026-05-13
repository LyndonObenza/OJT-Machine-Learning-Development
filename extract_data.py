
# THIS IS FOR THE DATA NEEDED in the 100+ LSTM models


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

    # --- Targets ---
    data['price_pct_change']  = close.pct_change(1).shift(-1) * 100
    data['volume_pct_change'] = volume.pct_change(1).shift(-1) * 100

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
    #                "4030", "3092", "4348", "4190", "2222", 
#                    "7030", "7010", "2050", "2060", "4321", "8100", "2120", "6010", "8250", 
#                    "8300", "2200", "1060", "1010", "1321", "1180", "2285", "8200", "1050", "1030", 
#                    "4081", "2286", "1150", "2320", "3080", "4340", "3090", "4342", "2283", "2284", 
#                    "1323", "4009", "4003", "4334", "2020", "1140", "4084", "7020", "2370", "3030", 
#                    "4261", "4012", "1020", "1830", "2150", "4260", "3020", "4333", "4001", "1835", "4165", 
#                    "1320", "4007", "3040", "8280", "4347", "2382", "2223", "4163", "6070", "4262", "4083", 
#                    "3010", "4164", "2270", "3003", "1834", "1120", "1303", "8170", "1214", "8010", "2081", 
#                    "4162", "7204", "7202", "2281", "3002", "3050", "1833", "4143", "8030", "4250", "3005", 
#                    "2230", "4280", "4020", "4004", "8210", "4142", "4263", "2280", "1212", "4002", "8012", 
#                    "4150", "8160", "1831", "8070", "4051", "1322", "3004", "1182", "6017", "2080", "4161", 
#                    "4031", "3060", "4071", "1302", "4322", "8020", "4264", "4016", "4072", "4300", "4090", 
#                    "6001", "6004", "7040", "4191", "4332", "2381", "6014", "4017", "4015", "4324", "6015", 
#                    "2310", "4336", "4006", "2300", "4338", "8230", "4082", "4144", "4200", "4014", "1111", 
#                    "4291", "4339", "4013", "2282", "4325", "1304", "7203", "4018", "2084", "7200", "4100", 
#                    "2070", "2287", "4331", "2190", "2250", "1211", "8313", "4193", "8180", "1183", "2290", 
#                    "2240", "6020", "4192", "4345", "1301", "8270", "4180", "8240", "4210", "8040", "4130", 
#                    "8310", "4290", "4344", "6016", "4050", "2082", "4040", "1210", "2083", "4080", "6013", 
#                    "2010", "4170", "3007", "4240"
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
    # OHLCV
    'open_price', 'high_price', 'low_price', 'close_price', 'volume',
    # Trend
    'EMA_10', 'EMA_20', 'EMA_ratio', 'MACD_hist',                  # EMA_20, EMA_ratio added
    # Momentum
    'RSI_14', 'ROC_5', 'ROC_10', 'ROC_20',                         # ROC_10, ROC_20 added
    # Volatility
    'ATR_14', 'ATR_ratio', 'BB_pct', 'realized_vol',               # ATR_ratio, realized_vol added
    # Volume
    'OBV', 'OBV_momentum', 'volume_ratio', 'volume_surge', 'MFI_14', # OBV_momentum, volume_surge added
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

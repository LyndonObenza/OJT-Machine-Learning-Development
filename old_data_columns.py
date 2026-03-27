from tvDatafeed import TvDatafeed, Interval
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from ta.trend import PSARIndicator, SMAIndicator, EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

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
                'volume': 'volume'
            })
            data = data[['date', 'open_price', 'high_price',
                         'low_price', 'close_price', 'volume']]
            data['date']         = pd.to_datetime(data['date'])
            data['PSAR']         = PSARIndicator(data['high_price'],
                                                  data['low_price'],
                                                  data['close_price']).psar()
            data['trend']        = (data['close_price'] > data['PSAR']).astype(int)
            data['ma_200']       = SMAIndicator(data['close_price'],
                                                 window=200).sma_indicator()
            data['ma_200_percent'] = (data['close_price'] - data['ma_200']) / data['ma_200']
            data = data.dropna().reset_index(drop=True)

        except Exception as e:
            print(e)
            continue

    return data


# ── Fetch all 10 tickers ──────────────────────────────────────────────────────
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
    "AAPL": aapl_df,  "NVDA": nvda_df,  "TSLA": tsla_df,
    "MSFT": msft_df,  "AMD":  amd_df,   "GOOGL": googl_df,
    "AVGO": avgo_df,  "META": meta_df,  "ORCL":  orcl_df,
    "IBM":  ibm_df,
}

for ticker, frame in ticker_frames.items():
    frame["ticker"] = ticker
    if "date" in frame.columns and "datetime" not in frame.columns:
        frame.rename(columns={"date": "datetime"}, inplace=True)

# ── Save individual CSVs ───────────────────────────────────────────────────────
for ticker, frame in ticker_frames.items():
    path = f"{ticker.lower()}_daily.csv"
    frame.to_csv(path, index=False)
    print(f"Saved: {path}  ({len(frame):,} rows)")

# ── Combine and save master CSV ───────────────────────────────────────────────
df = pd.concat(ticker_frames.values(), ignore_index=True)
df = df.sort_values(["ticker", "datetime"]).reset_index(drop=True)
df.to_csv("all_tickers_daily.csv", index=False)

print(f"\nTotal rows : {len(df):,}")
print(df["ticker"].value_counts())
print("\nSaved: all_tickers_daily.csv")
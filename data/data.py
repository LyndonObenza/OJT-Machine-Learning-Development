from tvDatafeed import TvDatafeed, Interval
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from ta.trend import PSARIndicator, SMAIndicator, EMAIndicator, ADXIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange


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
                symbol   = symbol,
                exchange = exchange,
                interval = tv_interval,
                n_bars   = 10000,
            )

            data['date'] = data.index
            data = data.rename(columns={
                'open':  'open_price',
                'high':  'high_price',
                'low':   'low_price',
                'close': 'close_price',
                'volume':'volume'
            })
            data = data[['date', 'open_price', 'high_price',
                          'low_price', 'close_price', 'volume']]
            data['date'] = pd.to_datetime(data['date'])

            # ══════════════════════════════════════════════════════════════════
            # STATIONARY PRICE FEATURES
            # Replaces raw open/high/low/close with normalised versions
            # ══════════════════════════════════════════════════════════════════
            data['hl_range_pct'] = np.where(
                data['close_price'] > 0,
                (data['high_price'] - data['low_price']) / data['close_price'],
                0
            )
            data['close_pct_change'] = data['close_price'].pct_change()

            # ══════════════════════════════════════════════════════════════════
            # VOLUME — normalised ratio (replaces raw volume)
            # ══════════════════════════════════════════════════════════════════
            data['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()

            # ══════════════════════════════════════════════════════════════════
            # CANDLE STRUCTURE
            # ══════════════════════════════════════════════════════════════════
            candle_range = data['high_price'] - data['low_price']
            upper_wick   = data['high_price'] - data[['open_price', 'close_price']].max(axis=1)
            lower_wick   = data[['open_price', 'close_price']].min(axis=1) - data['low_price']
            body         = abs(data['close_price'] - data['open_price'])

            data['upper_wick_ratio'] = np.where(candle_range > 0, upper_wick / candle_range, 0)
            data['lower_wick_ratio'] = np.where(candle_range > 0, lower_wick / candle_range, 0)
            data['body_ratio']       = np.where(candle_range > 0, body / candle_range, 0)

            # ══════════════════════════════════════════════════════════════════
            # ATR — needed early for wick_atr features below
            # ══════════════════════════════════════════════════════════════════
            atr_indicator = AverageTrueRange(
                high   = data['high_price'],
                low    = data['low_price'],
                close  = data['close_price'],
                window = 14
            )
            data['ATR']         = atr_indicator.average_true_range()
            data['ATR_percent'] = np.where(
                data['close_price'] > 0,
                data['ATR'] / data['close_price'],
                0
            )

            # ATR-normalised wicks (absolute significance, not just candle-relative)
            data['upper_wick_atr'] = np.where(data['ATR'] > 0, upper_wick / data['ATR'], 0)
            data['lower_wick_atr'] = np.where(data['ATR'] > 0, lower_wick / data['ATR'], 0)

            data = data.drop(['ATR'], axis=1)

            # ══════════════════════════════════════════════════════════════════
            # PSAR + TREND FLAG
            # ══════════════════════════════════════════════════════════════════
            data['PSAR']  = PSARIndicator(
                data['high_price'], data['low_price'], data['close_price']
            ).psar()
            data['trend'] = (data['close_price'] > data['PSAR']).astype(int)

            data['close_minus_psar'] = np.where(
                data['PSAR'] > 0,
                (data['close_price'] - data['PSAR']) / data['PSAR'],
                0
            )

            # Drop raw PSAR level — close_minus_psar is the stationary version
            data = data.drop(['PSAR'], axis=1)

            # ══════════════════════════════════════════════════════════════════
            # SMA 200
            # ══════════════════════════════════════════════════════════════════
            data['ma_200']         = SMAIndicator(data['close_price'], window=200).sma_indicator()
            data['ma_200_percent'] = np.where(
                data['ma_200'] > 0,
                (data['close_price'] - data['ma_200']) / data['ma_200'],
                0
            )
            # Drop raw MA level
            data = data.drop(['ma_200'], axis=1)

            # ══════════════════════════════════════════════════════════════════
            # RSI + SLOPE + DIVERGENCE
            # ══════════════════════════════════════════════════════════════════
            data['rsi']           = RSIIndicator(close=data['close_price'], window=14).rsi()
            data['rsi_slope']     = data['rsi'] - data['rsi'].shift(3)
            data['rsi_divergence'] = (
                data['close_pct_change'] - (data['rsi'] - data['rsi'].shift(5)) / 100
            )

            # ══════════════════════════════════════════════════════════════════
            # BOLLINGER BANDS
            # ══════════════════════════════════════════════════════════════════
            bb = BollingerBands(close=data['close_price'], window=20, window_dev=2)
            bb_upper = bb.bollinger_hband()
            bb_lower = bb.bollinger_lband()

            data['bb_position'] = np.where(
                (bb_upper - bb_lower) > 0,
                (data['close_price'] - bb_lower) / (bb_upper - bb_lower),
                0.5
            )
            data['bb_width'] = np.where(
                data['close_price'] > 0,
                (bb_upper - bb_lower) / data['close_price'],
                0
            )

            # ══════════════════════════════════════════════════════════════════
            # MACD HISTOGRAM + SLOPE
            # ══════════════════════════════════════════════════════════════════
            macd_indicator    = MACD(close=data['close_price'], window_slow=26, window_fast=12, window_sign=9)
            macd_line         = macd_indicator.macd()
            signal_line       = macd_indicator.macd_signal()
            macd_hist_raw     = macd_line - signal_line

            # Normalise by close price for cross-ticker comparability
            data['macd_histogram']  = np.where(
                data['close_price'] > 0,
                macd_hist_raw / data['close_price'],
                0
            )
            data['macd_hist_slope'] = data['macd_histogram'] - data['macd_histogram'].shift(3)

            # ══════════════════════════════════════════════════════════════════
            # ADX + SLOPE
            # ══════════════════════════════════════════════════════════════════
            adx_indicator = ADXIndicator(
                high   = data['high_price'],
                low    = data['low_price'],
                close  = data['close_price'],
                window = 14
            )
            data['adx']       = adx_indicator.adx()
            data['adx_slope'] = data['adx'] - data['adx'].shift(5)

            # ══════════════════════════════════════════════════════════════════
            # EMA 50 DISTANCE
            # ══════════════════════════════════════════════════════════════════
            ema_50 = EMAIndicator(close=data['close_price'], window=50).ema_indicator()
            data['close_minus_ema'] = np.where(
                ema_50 > 0,
                (data['close_price'] - ema_50) / ema_50,
                0
            )

            # ══════════════════════════════════════════════════════════════════
            # MOMENTUM & VOLATILITY
            # ══════════════════════════════════════════════════════════════════
            data['return_5']      = data['close_price'].pct_change(5)
            data['volatility_10'] = data['close_price'].pct_change().rolling(10).std()

            # ══════════════════════════════════════════════════════════════════
            # BARS SINCE SWING HIGH / LOW
            # ══════════════════════════════════════════════════════════════════
            def compute_bars_since_swing(df, lookback=20):
                highs, lows = [], []
                high_vals = df['high_price'].values
                low_vals  = df['low_price'].values
                for i in range(len(df)):
                    start = max(0, i - lookback)
                    window_h = high_vals[start:i + 1]
                    window_l = low_vals[start:i + 1]
                    highs.append(len(window_h) - 1 - int(np.argmax(window_h)))
                    lows.append(len(window_l) - 1 - int(np.argmin(window_l)))
                return pd.Series(highs, index=df.index), pd.Series(lows, index=df.index)

            data['bars_since_swing_high'], data['bars_since_swing_low'] = \
                compute_bars_since_swing(data, lookback=20)

            # ══════════════════════════════════════════════════════════════════
            # RENAME & FINALISE
            # ══════════════════════════════════════════════════════════════════
            data = data.rename(columns={
                'open_price':  'open',
                'high_price':  'high',
                'low_price':   'low',
                'close_price': 'close',
                'date':        'datetime'
            })

            # Keep only the final feature columns + metadata
            keep_cols = [
                'datetime', 'open', 'high', 'low', 'close', 'volume',
                # Stationary price & volume
                'hl_range_pct', 'close_pct_change', 'volume_ratio',
                # Trend
                'trend', 'ma_200_percent', 'close_minus_psar', 'close_minus_ema',
                # RSI family
                'rsi', 'rsi_slope', 'rsi_divergence',
                # MACD family
                'macd_histogram', 'macd_hist_slope',
                # ADX family
                'adx', 'adx_slope',
                # Bollinger
                'bb_position', 'bb_width',
                # Candle structure
                'upper_wick_ratio', 'lower_wick_ratio', 'body_ratio',
                'upper_wick_atr', 'lower_wick_atr',
                # Swing age
                'bars_since_swing_high', 'bars_since_swing_low',
                # Volatility & momentum
                'return_5', 'volatility_10', 'ATR_percent',
            ]
            data = data[keep_cols]
            data = data.dropna().reset_index(drop=True)

            print(f"  {symbol}: {len(data)} rows, {len(data.columns)} columns")

        except Exception as e:
            print(f"  Error: {e}")
            data = None
            continue

    return data


# ══════════════════════════════════════════════════════════════════════════════
# Fetch Data
# ══════════════════════════════════════════════════════════════════════════════
tickers = [
    ("AAPL", "NASDAQ"),
    ("NVDA", "NASDAQ"),
    ("TSLA", "NASDAQ"),
    ("MSFT","NASDAQ"),
    ("AMD","NASDAQ"),
    ("GOOGL","NASDAQ"),
    ("AVGO","NASDAQ"),
    ("META","NASDAQ"),
    ("ORCL","NYSE"),
    ("IBM","NYSE")
    
]

frames = {}
for symbol, exchange in tickers:
    print(f"\nFetching {symbol}...")
    df = get_tradingview_data(symbol, exchange, Interval.in_daily)
    if df is not None:
        df['ticker'] = symbol
        frames[symbol] = df
        df.to_csv(f"{symbol.lower()}_daily.csv", index=False)
        print(f"  Saved {symbol.lower()}_daily.csv")

# ── Combine and save master CSV ───────────────────────────────────────────────
combined = pd.concat(frames.values(), ignore_index=True)
combined = combined.sort_values(['ticker', 'datetime']).reset_index(drop=True)
combined.to_csv("all_tickers_daily.csv", index=False)

# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("DATA EXTRACTION COMPLETE")
print("=" * 70)
print(f"Total rows : {len(combined)}")
print(f"Tickers    : {combined['ticker'].value_counts().to_dict()}")
print(f"\nFeature columns (26 model inputs):")

feature_columns = [
    'hl_range_pct', 'close_pct_change', 'volume_ratio',
    'trend', 'ma_200_percent', 'close_minus_psar', 'close_minus_ema',
    'rsi', 'rsi_slope', 'rsi_divergence',
    'macd_histogram', 'macd_hist_slope',
    'adx', 'adx_slope',
    'bb_position', 'bb_width',
    'upper_wick_ratio', 'lower_wick_ratio', 'body_ratio',
    'upper_wick_atr', 'lower_wick_atr',
    'bars_since_swing_high', 'bars_since_swing_low',
    'return_5', 'volatility_10', 'ATR_percent',
]
for i, col in enumerate(feature_columns, 1):
    print(f"  {i:2d}. {col}")

print(f"\nTotal model input features: {len(feature_columns)}")
print("\nSaved files:")
for symbol, _ in tickers:
    print(f"  - {symbol.lower()}_daily.csv")
print("  - all_tickers_daily.csv")
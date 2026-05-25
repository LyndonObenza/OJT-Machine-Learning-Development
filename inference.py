"""
Inference script — BiLSTM Attention ONNX models
Runs on the last 10% (test split) of each symbol's daily CSV.

File layout expected (relative to inference.py):
  {symbol}_daily.csv
  models/{symbol}/{symbol}_feature_scaler.pkl
  models/{symbol}/{symbol}_target_scaler.pkl
  models/{symbol}/{symbol}_bilstm.onnx

Output:
  models/{symbol}/{symbol}_inference_result.csv
"""

import os
import pickle
import numpy as np
import pandas as pd
import onnxruntime as ort


SYMBOLS = ["8300", "1010", "8100", "2050", "2060", "1180", "3080", "3090"]

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

SEQ_LEN_DEFAULT = 20
SEQ_LEN = {
    "8300": SEQ_LEN_DEFAULT,
    "1010": SEQ_LEN_DEFAULT,
    "8100": SEQ_LEN_DEFAULT,
    "2050": SEQ_LEN_DEFAULT,
    "2060": SEQ_LEN_DEFAULT,
    "1180": SEQ_LEN_DEFAULT,
    "3080": SEQ_LEN_DEFAULT,
    "3090": SEQ_LEN_DEFAULT,
    "2370": SEQ_LEN_DEFAULT,
}

FEATURE_COLS = [
    # OHLCV (5)
    'open_price', 'high_price', 'low_price', 'close_price', 'volume',
    # Trend (4)
    'EMA_10', 'EMA_20', 'EMA_ratio', 'MACD_hist',
    # Momentum (4)
    'RSI_14', 'ROC_5', 'ROC_10', 'ROC_20',
    # Volatility (4)
    'ATR_14', 'ATR_ratio', 'BB_pct', 'realized_vol',
    # Volume (5)
    'OBV', 'OBV_momentum', 'volume_ratio', 'volume_surge', 'MFI_14',
    # Lag features (15)
    'close_lag_1', 'close_lag_2', 'close_lag_3', 'close_lag_4', 'close_lag_5',
    'returns_lag_1', 'returns_lag_2', 'returns_lag_3', 'returns_lag_4', 'returns_lag_5',
    'volume_lag_1', 'volume_lag_2', 'volume_lag_3', 'volume_lag_4', 'volume_lag_5',
    # Regime / persistence (5)
    'above_ema10', 'above_ema20', 'roc5_pos', 'roc20_pos', 'up_streak',
]  # 42 total


# ── Feature engineering ───────────────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close  = df["close_price"]
    high   = df["high_price"]
    low    = df["low_price"]
    volume = df["volume"]

    # ── Trend ─────────────────────────────────────────────────────────────────
    df["EMA_10"]    = close.ewm(span=10, adjust=False).mean()
    df["EMA_20"]    = close.ewm(span=20, adjust=False).mean()
    df["EMA_ratio"] = df["EMA_10"] / df["EMA_20"]

    ema12           = close.ewm(span=12, adjust=False).mean()
    ema26           = close.ewm(span=26, adjust=False).mean()
    macd_line       = ema12 - ema26
    macd_signal     = macd_line.ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = macd_line - macd_signal

    # ── Momentum ──────────────────────────────────────────────────────────────
    delta     = close.diff()
    gain      = delta.clip(lower=0).rolling(14).mean()
    loss      = (-delta.clip(upper=0)).rolling(14).mean()
    rs        = gain / loss.replace(0, np.nan)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    df["ROC_5"]  = close.pct_change(5)  * 100
    df["ROC_10"] = close.pct_change(10) * 100
    df["ROC_20"] = close.pct_change(20) * 100

    # ── Volatility ────────────────────────────────────────────────────────────
    tr              = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    df["ATR_14"]    = tr.rolling(14).mean()
    df["ATR_ratio"] = df["ATR_14"] / close

    rolling_std     = close.rolling(20).std()
    sma20           = close.rolling(20).mean()
    bb_upper        = sma20 + 2 * rolling_std
    bb_lower        = sma20 - 2 * rolling_std
    df["BB_pct"]    = (close - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)

    df["realized_vol"] = close.pct_change().rolling(20).std() * np.sqrt(252)

    # ── Volume ────────────────────────────────────────────────────────────────
    direction       = np.sign(close.diff()).fillna(0)
    df["OBV"]       = (direction * volume).cumsum()
    df["OBV_momentum"] = df["OBV"].diff(5)
    df["volume_ratio"] = volume / volume.rolling(20).mean()
    df["volume_surge"] = (volume > volume.rolling(20).mean() * 1.5).astype(float)

    # MFI
    typical_price   = (high + low + close) / 3
    raw_money_flow  = typical_price * volume
    pos_flow        = raw_money_flow.where(typical_price > typical_price.shift(), 0).rolling(14).sum()
    neg_flow        = raw_money_flow.where(typical_price < typical_price.shift(), 0).rolling(14).sum()
    mfi_ratio       = pos_flow / neg_flow.replace(0, np.nan)
    df["MFI_14"]    = 100 - (100 / (1 + mfi_ratio))

    # ── Lag features ──────────────────────────────────────────────────────────
    for lag in range(1, 6):
        df[f"close_lag_{lag}"]   = close.shift(lag)
        df[f"returns_lag_{lag}"] = close.pct_change(1).shift(lag) * 100
        df[f"volume_lag_{lag}"]  = volume.shift(lag)

    # ── Regime / persistence ──────────────────────────────────────────────────
    df["above_ema10"] = (close > df["EMA_10"]).astype(float)
    df["above_ema20"] = (close > df["EMA_20"]).astype(float)
    df["roc5_pos"]    = (df["ROC_5"]  > 0).astype(float)
    df["roc20_pos"]   = (df["ROC_20"] > 0).astype(float)

    daily_up          = close.diff(1).gt(0).astype(int)
    streak_id         = (daily_up != daily_up.shift()).cumsum()
    df["up_streak"]   = (daily_up.groupby(streak_id).cumcount() + 1) * daily_up

    # ── Targets ───────────────────────────────────────────────────────────────
    df["price_pct_change"]      = close.pct_change(1) * 100
    df["volume_log"]            = np.log1p(volume)
    df["volume_pct_change_log"] = df["volume_log"].diff(1).shift(-1)

    df = df.dropna().reset_index(drop=True)
    return df


# ── Sequence builder ──────────────────────────────────────────────────────────
def create_sequences(X_scaled: np.ndarray, seq_len: int) -> np.ndarray:
    seqs = []
    for i in range(len(X_scaled) - seq_len):
        seqs.append(X_scaled[i : i + seq_len])
    return np.array(seqs, dtype=np.float32)


# ── Main inference loop ───────────────────────────────────────────────────────
any_success = False

for symbol in SYMBOLS:
    model_dir = os.path.join(DATA_DIR, "models", symbol)
    csv_path  = os.path.join(DATA_DIR, f"{symbol}_daily.csv")
    feat_pkl  = os.path.join(model_dir, f"{symbol}_feature_scaler.pkl")
    tgt_pkl   = os.path.join(model_dir, f"{symbol}_target_scaler.pkl")
    onnx_path = os.path.join(model_dir, f"{symbol}_bilstm.onnx")

    missing = [p for p in [csv_path, feat_pkl, tgt_pkl, onnx_path] if not os.path.exists(p)]
    if missing:
        print(f"[{symbol}] Skipping — missing files:")
        for p in missing:
            print(f"    {p}")
        continue

    print(f"\n{'─'*55}")
    print(f"  Inference — {symbol}")
    print(f"{'─'*55}")

    # ── Load & engineer features ──────────────────────────────────────────────
    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    df = build_features(df)

    # ── Test split: last 10% ──────────────────────────────────────────────────
    n       = len(df)
    test_df = df.iloc[int(n * 0.90):].reset_index(drop=True)
    print(f"  Test rows  : {len(test_df)}")
    print(f"  Date range : {test_df['datetime'].min().date()} → {test_df['datetime'].max().date()}")

    # ── Load scalers ──────────────────────────────────────────────────────────
    with open(feat_pkl, "rb") as f:
        feature_scaler = pickle.load(f)
    with open(tgt_pkl, "rb") as f:
        target_scaler  = pickle.load(f)

    # ── Scale — pass DataFrame so column names match ──────────────────────────
    feat_cols_present = [c for c in FEATURE_COLS if c in test_df.columns]
    missing_cols      = [c for c in FEATURE_COLS if c not in test_df.columns]
    if missing_cols:
        print(f"  ⚠ Missing cols (will be skipped): {missing_cols}")

    X_test = feature_scaler.transform(test_df[feat_cols_present])  # DataFrame, keeps names

    # ── Build sequences ───────────────────────────────────────────────────────
    seq_len   = SEQ_LEN[symbol]
    X_seq     = create_sequences(X_test, seq_len)
    pred_rows = test_df.iloc[seq_len:].reset_index(drop=True)
    print(f"  Sequences  : {len(X_seq)}")

    # ── ONNX inference ────────────────────────────────────────────────────────
    sess        = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name  = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    raw_preds  = sess.run([output_name], {input_name: X_seq})[0]
    preds_orig = target_scaler.inverse_transform(raw_preds)

    # ── Build result table ────────────────────────────────────────────────────
    close_vals = pred_rows["close_price"].values
    pred_pct   = preds_orig[:, 0]

    result = pd.DataFrame({
        "Symbol"              : symbol,
        "Date"                : pred_rows["datetime"].dt.date.values,
        "Open"                : pred_rows["open_price"].values,
        "High"                : pred_rows["high_price"].values,
        "Low"                 : pred_rows["low_price"].values,
        "Close (Actual Price)": close_vals,
        "Actual Price pct"    : pred_rows["price_pct_change"].values,
        "Predicted Price pct" : pred_pct,
        "Predicted Price"     : close_vals * (1 + pred_pct / 100),
    })

    # ── Sanity stats ──────────────────────────────────────────────────────────
    actual    = result["Actual Price pct"].values
    predicted = result["Predicted Price pct"].values
    dir_acc   = np.mean(np.sign(actual) == np.sign(predicted))
    print(f"  Direction accuracy : {dir_acc:.2%}")
    print(f"  Pred range         : [{predicted.min():.4f}, {predicted.max():.4f}]")
    print(f"  Actual range       : [{actual.min():.4f},  {actual.max():.4f}]")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = os.path.join(model_dir, f"{symbol}_inference_result.csv")
    result.to_csv(out_path, index=False)
    print(f"  ✓ Saved → {out_path}")
    any_success = True

if not any_success:
    print("\nNo results — check your file paths and symbol list.")
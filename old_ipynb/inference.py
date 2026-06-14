import numpy as np
import pandas as pd
import joblib
import onnxruntime as ort

STOCKS      = "nvda"
WINDOW_SIZE = 30
THRESHOLD   = 0.5
START_DATE  = pd.Timestamp("2019-01-01")
OLD = "old_ipynb"

features = [
    'hl_range_pct', 'close_pct_change', 'volume_ratio', 'trend', 'ma_200_percent',
    'rsi', 'bb_position', 'rsi_slope', 'rsi_divergence', 'macd_histogram',
    'macd_hist_slope', 'adx', 'adx_slope', 'upper_wick_atr', 'lower_wick_atr',
    'bars_since_swing_high', 'bars_since_swing_low', 'close_minus_ema',
    'close_minus_psar', 'return_5', 'volatility_10', 'upper_wick_ratio',
    'lower_wick_ratio', 'body_ratio', 'ATR_percent', 'bb_width',
    'psar_distance_pct', 'fractal_swing_phase',
]
target_cols = ['target_t1', 'target_t2', 'target_t3', 'target_t4']

# ── Load ──────────────────────────────────────────────────────────────────────
scalers = joblib.load(f"{OLD}/{STOCKS}/{STOCKS}_per_ticker.pkl")
sc      = scalers[STOCKS.upper()]

df = pd.read_csv(f"old_ipynb/{STOCKS}_daily.csv", parse_dates=["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)
df = df.dropna(subset=features)

sess = ort.InferenceSession(f"{OLD}/{STOCKS}/{STOCKS}_stock_trend_transformer.onnx")

# ── Precompute scaled features for full series ────────────────────────────────
feat_scaled = np.clip(sc.transform(df[features].values), -5.0, 5.0).astype(np.float32)

# ── Iterate from 2019 onwards ─────────────────────────────────────────────────
results = []

for i in range(WINDOW_SIZE, len(df)):
    date = df["datetime"].iloc[i]
    if date < START_DATE:
        continue

    x      = feat_scaled[i - WINDOW_SIZE:i][np.newaxis]          # (1, 30, 28)
    logits = sess.run(["output"], {"input": x})[0]                # (1, 4)
    probs  = 1 / (1 + np.exp(-logits[0]))                         # (4,)
    preds  = (probs > THRESHOLD).astype(int)

    row = {
        "datetime":   date,
        "close_price": df["close_price"].iloc[i],
        "prob_t1": probs[0], "prob_t2": probs[1],
        "prob_t3": probs[2], "prob_t4": probs[3],
        "pred_t1": preds[0], "pred_t2": preds[1],
        "pred_t3": preds[2], "pred_t4": preds[3],
    }
    # Attach actuals if available (NaN for last 4 rows)
    for col in target_cols:
        row[f"actual_{col}"] = df[col].iloc[i] if col in df.columns else np.nan

    results.append(row)

results_df = pd.DataFrame(results)

# ── Accuracy per horizon (exclude NaN actuals) ────────────────────────────────
print(f"Total inference rows : {len(results_df):,}")
print(f"Date range           : {results_df['datetime'].min().date()} → {results_df['datetime'].max().date()}")
print()
print(f"{'Horizon':<8} {'Accuracy':>10} {'Pred↑%':>8} {'True↑%':>8} {'Rows':>8}")
print("─" * 46)
for h in range(1, 5):
    pred_col   = f"pred_t{h}"
    actual_col = f"actual_target_t{h}"
    sub        = results_df.dropna(subset=[actual_col])
    if len(sub) == 0:
        continue
    acc        = (sub[pred_col] == sub[actual_col].astype(int)).mean()
    pred_up    = sub[pred_col].mean() * 100
    true_up    = sub[actual_col].mean() * 100
    print(f"  t+{h:<5}  {acc:>10.4f} {pred_up:>7.1f}% {true_up:>7.1f}% {len(sub):>8,}")

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = f"{OLD}/{STOCKS}/{STOCKS}_inference_2019_2026.csv"
results_df.to_csv(out_path, index=False)
print(f"\nSaved → {out_path}")
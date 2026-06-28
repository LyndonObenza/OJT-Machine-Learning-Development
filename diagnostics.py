"""
diagnostic.py — standalone feature diagnostics for the per-symbol BiLSTM
pipeline (Step 1 baseline: leak fixed, per-symbol training).

Runs INDEPENDENTLY of the training notebook. Rebuilds the same val split
from your raw CSVs (same date cuts, embargo, scaling-fit logic as your
training notebook) and loads your saved per-symbol model + scaler
checkpoints from disk. Does not need the notebook kernel running.

TWO CHECKS
----------
1. CORRELATION / REDUNDANCY   — model-free. Flags near-duplicate feature
   pairs (|corr| > threshold) on pooled scaled train data.
2. PERMUTATION IMPORTANCE     — model-aware. For each symbol's trained
   model, shuffles one feature at a time across validation sequences and
   measures the AUC / balanced-accuracy drop. Near-zero drop = the model
   isn't using that feature = noise candidate.

BEFORE RUNNING — fix the CONFIG block below
--------------------------------------------
- DATA_DIR / CSV columns: must match your NASDAQ_DATA/{symbol}_daily.csv
  files exactly. EMA_10, EMA_20, MACD_hist, ROC_5, ROC_10, ROC_20, RSI_14,
  EMA_ratio, ATR_ratio, BB_pct, realized_vol, OBV_momentum, volume_ratio,
  volume_surge, MFI_14 are assumed to already exist as columns in the CSV
  (pre-computed by your data pipeline) — this script does NOT recompute
  them, since their exact formulas live in your data-prep step, not the
  training notebook. If any of these are missing from your CSVs, this
  script will raise a clear KeyError rather than silently fabricating
  values.
- MODEL_DIR_TEMPLATE / MODEL_FILE_TEMPLATE / PARAMS_FILE_TEMPLATE /
  SCALER_FILE_TEMPLATE: must match the paths your training notebook
  actually saved to. Defaults below match the save pattern visible in
  your training notebook (models/{symbol}/{symbol}_feature_scaler.pkl
  etc.) — confirm against your own checkpoint cell and adjust if it
  saves the trained model weights under a different name.
- StockPctChangeBiLSTMAttention: copy your exact model class definition
  into model_def.py next to this script (see import below), so this
  script can instantiate it and load your saved state_dict.

USAGE
-----
    python diagnostic.py
    python diagnostic.py --symbols AAPL TSLA NVDA
    python diagnostic.py --skip-permutation     # correlation check only
    python diagnostic.py --skip-correlation      # permutation check only
"""

import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

# ──────────────────────────────────────────────────────────────────────────
# CONFIG — adjust these to match your actual file layout
# ──────────────────────────────────────────────────────────────────────────
DATA_DIR = "NASDAQ_DATA"
SYMBOLS_DEFAULT = ["GOOGL", "AVGO", "MSFT", "META", "NVDA", "AAPL", "TSLA"]

MODEL_DIR_TEMPLATE     = "models/{symbol}"
MODEL_FILE_TEMPLATE    = "{symbol}_best_bilstm.pt"        # <-- confirm vs your save cell
PARAMS_FILE_TEMPLATE   = "{symbol}_best_params.json"       # <-- confirm vs your save cell
FEATURE_SCALER_FILE    = "{symbol}_feature_scaler.pkl"     # matches notebook save pattern
TARGET_SCALER_FILE     = "{symbol}_target_scaler.pkl"      # matches notebook save pattern

SEQ_LEN_DEFAULT = 30
N_PERMUTATIONS = 5
CORR_THRESHOLD = 0.85
OUTLIER_SIGMA = 3
VAL_SPLIT_FRACTION = 0.85   # fraction of pre-2019 data used as train (rest = val)
EMBARGO_ROWS = 5             # must match the 5-day-ahead target horizon
TEST_CUTOFF_DATE = "2019-01-01"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURE_COLS = [
    "ret_1", "gap", "intraday_range", "body",
    "EMA_ratio", "macd_norm",
    "RSI_14", "ROC_5", "ROC_10", "ROC_20",
    "ATR_ratio", "BB_pct", "realized_vol",
    "OBV_momentum", "volume_ratio", "volume_surge", "MFI_14",
    "returns_lag_1", "returns_lag_2", "returns_lag_3", "returns_lag_4", "returns_lag_5",
    "above_ema10", "above_ema20", "roc5_pos", "roc20_pos", "up_streak",
]
TARGET_COL = "price_pct_change_5d_vn"

# Columns this script computes itself (mirrors training notebook exactly).
# Everything else in FEATURE_COLS (EMA_ratio, RSI_14, ATR_ratio, BB_pct,
# realized_vol, OBV_momentum, volume_ratio, volume_surge, MFI_14, EMA_10,
# EMA_20, MACD_hist, ROC_5, ROC_10, ROC_20) must already exist in the CSV.
COMPUTED_HERE = [
    "returns_lag_1", "returns_lag_2", "returns_lag_3", "returns_lag_4", "returns_lag_5",
    "above_ema10", "above_ema20", "roc5_pos", "roc20_pos", "up_streak",
    "ret_1", "gap", "intraday_range", "body", "macd_norm",
]


# ──────────────────────────────────────────────────────────────────────────
# DATA PREP — mirrors the training notebook's logic exactly (no leakage)
# ──────────────────────────────────────────────────────────────────────────
def build_symbol_splits(symbol, data_dir=DATA_DIR):
    """
    Returns dict with train/val_X_scaled, train/val_y_scaled, and the
    fitted feature_scaler, OR loads pre-fit scalers from disk if present
    (preferred — guarantees identical scaling to what the model trained on).
    """
    filepath = os.path.join(data_dir, f"{symbol}_daily.csv")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing {filepath} — check DATA_DIR.")

    df = pd.read_csv(filepath, parse_dates=["datetime"])

    missing_precomputed = [c for c in FEATURE_COLS if c not in COMPUTED_HERE and c not in df.columns]
    if missing_precomputed:
        raise KeyError(
            f"[{symbol}] CSV is missing pre-computed columns: {missing_precomputed}. "
            f"This script does not recompute these — they must already exist in "
            f"{filepath}, produced by your data pipeline."
        )

    # ── Lag features ──
    for lag in range(1, 6):
        df[f"returns_lag_{lag}"] = df["close_price"].pct_change(1).shift(lag)

    # ── Regime / persistence features ──
    df["above_ema10"] = (df["close_price"] > df["EMA_10"]).astype(float)
    df["above_ema20"] = (df["close_price"] > df["EMA_20"]).astype(float)
    df["roc5_pos"]    = (df["ROC_5"] > 0).astype(float)
    df["roc20_pos"]   = (df["ROC_20"] > 0).astype(float)

    daily_up  = df["close_price"].diff(1).gt(0).astype(int)
    streak_id = (daily_up != daily_up.shift()).cumsum()
    df["up_streak"] = (daily_up.groupby(streak_id).cumcount() + 1) * daily_up

    # ── Stationary price-action features ──
    df["ret_1"]          = df["close_price"].pct_change(1)
    df["gap"]            = df["open_price"] / df["close_price"].shift(1) - 1
    df["intraday_range"] = (df["high_price"] - df["low_price"]) / df["close_price"]
    df["body"]           = (df["close_price"] - df["open_price"]) / df["open_price"]
    df["macd_norm"]      = df["MACD_hist"] / df["close_price"]

    # ── Target: 5d-ahead return, vol-normalized ──
    df["price_pct_change_5d"] = (df["close_price"].shift(-5) / df["close_price"] - 1) * 100
    _daily_ret = df["close_price"].pct_change()
    df["vol_denom_5d"] = (_daily_ret.rolling(20).std() * np.sqrt(5) * 100).clip(lower=0.5)
    df["price_pct_change_5d_vn"] = df["price_pct_change_5d"] / df["vol_denom_5d"]

    df = df.dropna().reset_index(drop=True)

    # ── 3-way split with embargo (no leakage) ──
    pre  = df[df["datetime"] < TEST_CUTOFF_DATE].reset_index(drop=True)
    cut  = int(len(pre) * VAL_SPLIT_FRACTION)
    train_df = pre.iloc[:cut - EMBARGO_ROWS]
    val_df   = pre.iloc[cut:-EMBARGO_ROWS] if EMBARGO_ROWS > 0 else pre.iloc[cut:]

    # ── Outlier clipping on TRAIN ONLY ──
    mean, std = train_df[TARGET_COL].mean(), train_df[TARGET_COL].std()
    train_df = train_df[train_df[TARGET_COL].between(mean - OUTLIER_SIGMA * std,
                                                        mean + OUTLIER_SIGMA * std)]

    train_X_raw = train_df[FEATURE_COLS].values.astype(np.float32)
    val_X_raw   = val_df[FEATURE_COLS].values.astype(np.float32)
    train_y_raw = train_df[[TARGET_COL]].values.astype(np.float32)
    val_y_raw   = val_df[[TARGET_COL]].values.astype(np.float32)

    # ── Prefer loading the EXACT scaler the model trained with, if saved ──
    scaler_path = os.path.join(MODEL_DIR_TEMPLATE.format(symbol=symbol),
                                FEATURE_SCALER_FILE.format(symbol=symbol))
    target_scaler_path = os.path.join(MODEL_DIR_TEMPLATE.format(symbol=symbol),
                                       TARGET_SCALER_FILE.format(symbol=symbol))

    if os.path.exists(scaler_path) and os.path.exists(target_scaler_path):
        with open(scaler_path, "rb") as f:
            feature_scaler = pickle.load(f)
        with open(target_scaler_path, "rb") as f:
            target_scaler = pickle.load(f)
        print(f"  [{symbol}] loaded saved scalers from {scaler_path}")
    else:
        print(f"  [{symbol}] WARNING — no saved scaler found at {scaler_path}; "
              f"fitting a NEW scaler on train data. This may not exactly match "
              f"what the model was trained with.")
        feature_scaler = StandardScaler().fit(train_X_raw)
        target_scaler = StandardScaler().fit(train_y_raw)

    train_X_scaled = feature_scaler.transform(train_X_raw)
    val_X_scaled   = feature_scaler.transform(val_X_raw)
    train_y_scaled = target_scaler.transform(train_y_raw)
    val_y_scaled   = target_scaler.transform(val_y_raw)

    return {
        "train_X_scaled": train_X_scaled, "train_y_scaled": train_y_scaled,
        "val_X_scaled": val_X_scaled, "val_y_scaled": val_y_scaled,
    }


def load_all_symbols(symbols, data_dir=DATA_DIR):
    prepared = {}
    for symbol in symbols:
        try:
            prepared[symbol] = build_symbol_splits(symbol, data_dir)
        except (FileNotFoundError, KeyError) as e:
            print(f"  [{symbol}] SKIPPED — {e}")
    return prepared


# ──────────────────────────────────────────────────────────────────────────
# PART 1: CORRELATION / REDUNDANCY CHECK
# ──────────────────────────────────────────────────────────────────────────
def correlation_redundancy_report(prepared, feature_cols=FEATURE_COLS, threshold=CORR_THRESHOLD,
                                   out_png="feature_correlation_heatmap.png"):
    all_X = [data["train_X_scaled"] for data in prepared.values()]
    if not all_X:
        print("No data loaded — skipping correlation check.")
        return None, []

    X = np.vstack(all_X)
    df = pd.DataFrame(X, columns=feature_cols)
    corr = df.corr()

    pairs = []
    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            r = corr.iloc[i, j]
            if abs(r) >= threshold:
                pairs.append((feature_cols[i], feature_cols[j], round(float(r), 3)))
    pairs.sort(key=lambda x: -abs(x[2]))

    print(f"\n{'='*70}")
    print(f"CORRELATION REDUNDANCY REPORT  (|r| >= {threshold})")
    print(f"{'='*70}")
    if not pairs:
        print("No feature pairs exceed the threshold. No obvious redundancy.")
    else:
        print(f"{'Feature A':<22} {'Feature B':<22} {'corr':>7}")
        print("-" * 53)
        for a, b, r in pairs:
            print(f"{a:<22} {b:<22} {r:>7.3f}")
        print(f"\n{len(pairs)} redundant pair(s) found. Consider dropping ONE feature "
              f"from each pair.")

    plt.figure(figsize=(14, 11))
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.3, cbar_kws={"shrink": 0.8})
    plt.title("Feature Correlation Matrix (pooled train, post-scaling)", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"\nSaved heatmap -> {out_png}")

    return corr, pairs


# ──────────────────────────────────────────────────────────────────────────
# PART 2: PERMUTATION IMPORTANCE
# ──────────────────────────────────────────────────────────────────────────
def _make_seqs(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i: i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs), np.array(ys)


def _predict(model, X, batch_size=128):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i + batch_size], dtype=torch.float32).to(device)
            preds.append(model(xb).cpu().numpy())
    return np.vstack(preds)


def _score(y_true, y_pred):
    true_dir = (y_true[:, 0] > 0).astype(int)
    pred_dir = (y_pred[:, 0] > 0).astype(int)
    try:
        auc = roc_auc_score(true_dir, y_pred[:, 0])
    except ValueError:
        auc = float("nan")
    bal_acc = balanced_accuracy_score(true_dir, pred_dir)
    return auc, bal_acc


def permutation_importance_for_symbol(symbol, model, val_X_seq, val_y_seq,
                                       feature_cols=FEATURE_COLS, n_repeats=N_PERMUTATIONS,
                                       seed=42):
    rng = np.random.RandomState(seed)
    baseline_pred = _predict(model, val_X_seq)
    base_auc, base_bal = _score(val_y_seq, baseline_pred)

    results = []
    for f_idx, fname in enumerate(feature_cols):
        auc_drops, bal_drops = [], []
        for _ in range(n_repeats):
            X_perm = val_X_seq.copy()
            perm_order = rng.permutation(len(X_perm))
            X_perm[:, :, f_idx] = val_X_seq[perm_order, :, f_idx]
            perm_pred = _predict(model, X_perm)
            auc, bal = _score(val_y_seq, perm_pred)
            auc_drops.append(base_auc - auc)
            bal_drops.append(base_bal - bal)

        results.append({
            "symbol": symbol, "feature": fname,
            "baseline_auc": round(base_auc, 4),
            "auc_drop_mean": round(float(np.mean(auc_drops)), 4),
            "auc_drop_std": round(float(np.std(auc_drops)), 4),
            "bal_acc_drop_mean": round(float(np.mean(bal_drops)), 4),
        })

    return pd.DataFrame(results).sort_values("auc_drop_mean", ascending=False)


def run_permutation_importance_all_symbols(prepared, model_class, symbols,
                                            feature_cols=FEATURE_COLS,
                                            seq_len_default=SEQ_LEN_DEFAULT,
                                            out_png="permutation_importance.png"):
    all_results = []

    for symbol in symbols:
        if symbol not in prepared:
            continue

        model_dir = MODEL_DIR_TEMPLATE.format(symbol=symbol)
        params_path = os.path.join(model_dir, PARAMS_FILE_TEMPLATE.format(symbol=symbol))
        model_path = os.path.join(model_dir, MODEL_FILE_TEMPLATE.format(symbol=symbol))

        if not os.path.exists(model_path):
            print(f"  [{symbol}] SKIPPED — model not found at {model_path}")
            continue

        if os.path.exists(params_path):
            with open(params_path) as f:
                pj = json.load(f)
            best = pj.get("best_params", pj)
            this_seq_len = best.get("seq_len", seq_len_default)
            hidden_size = best.get("hidden_size", 64)
            num_layers = best.get("num_layers", 2)
            dropout = best.get("dropout", 0.2)
        else:
            print(f"  [{symbol}] no params json at {params_path}, using defaults")
            this_seq_len, hidden_size, num_layers, dropout = seq_len_default, 64, 2, 0.2

        data = prepared[symbol]
        val_X_seq, val_y_seq = _make_seqs(data["val_X_scaled"], data["val_y_scaled"], this_seq_len)
        if len(val_X_seq) == 0:
            print(f"  [{symbol}] SKIPPED — no val sequences at seq_len={this_seq_len}")
            continue

        model = model_class(input_size=len(feature_cols), hidden_size=hidden_size,
                             num_layers=num_layers, dropout=dropout).to(device)
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.eval()

        print(f"  [{symbol}] running permutation importance "
              f"({len(val_X_seq)} val sequences, seq_len={this_seq_len})...")
        df_sym = permutation_importance_for_symbol(symbol, model, val_X_seq, val_y_seq, feature_cols)
        all_results.append(df_sym)

    if not all_results:
        print("\nNo models found — check MODEL_DIR_TEMPLATE / MODEL_FILE_TEMPLATE at top of script.")
        return None, None

    combined = pd.concat(all_results, ignore_index=True)

    agg = (combined.groupby("feature")["auc_drop_mean"]
           .agg(["mean", "std", "min", "max"])
           .sort_values("mean", ascending=False).round(4))
    agg.columns = ["avg_auc_drop", "std_across_symbols", "min_drop", "max_drop"]

    print(f"\n{'='*70}")
    print("AGGREGATE PERMUTATION IMPORTANCE (averaged across symbols)")
    print("Higher avg_auc_drop = more important. Near-zero/negative = noise candidate.")
    print(f"{'='*70}")
    print(agg.to_string())

    noise_candidates = agg[agg["avg_auc_drop"] <= 0.001].index.tolist()
    if noise_candidates:
        print(f"\nLikely NOISE candidates (avg AUC drop <= 0.001 when shuffled):")
        for f in noise_candidates:
            print(f"  - {f}")
    else:
        print("\nNo features show near-zero importance across the board.")

    plt.figure(figsize=(10, 8))
    agg_sorted = agg.sort_values("avg_auc_drop")
    colors = ["#d62728" if v <= 0.001 else "#1f77b4" for v in agg_sorted["avg_auc_drop"]]
    plt.barh(agg_sorted.index, agg_sorted["avg_auc_drop"], xerr=agg_sorted["std_across_symbols"],
              color=colors, capsize=3)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Avg AUC drop when shuffled (higher = more important)")
    plt.title("Permutation Importance — averaged across symbols\n(red = near-zero/negative = noise candidate)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"\nSaved chart -> {out_png}")

    combined.to_csv("permutation_importance_by_symbol.csv", index=False)
    agg.to_csv("permutation_importance_aggregate.csv")
    print("Saved CSVs -> permutation_importance_by_symbol.csv, permutation_importance_aggregate.csv")

    return combined, agg


# ──────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Standalone feature diagnostics")
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS_DEFAULT)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--skip-correlation", action="store_true")
    parser.add_argument("--skip-permutation", action="store_true")
    args = parser.parse_args()

    print(f"Loading data for symbols: {args.symbols}")
    prepared = load_all_symbols(args.symbols, args.data_dir)
    if not prepared:
        print("No symbols loaded successfully. Check DATA_DIR / CSV files. Exiting.")
        return

    if not args.skip_correlation:
        correlation_redundancy_report(prepared)

    if not args.skip_permutation:
        try:
            from model_def import StockPctChangeBiLSTMAttention
        except ImportError:
            print("\n" + "="*70)
            print("PERMUTATION IMPORTANCE SKIPPED — model_def.py not found.")
            print("Create model_def.py in this same folder containing your exact")
            print("StockPctChangeBiLSTMAttention class definition (copy-paste from")
            print("your training notebook), then re-run.")
            print("="*70)
            return
        run_permutation_importance_all_symbols(prepared, StockPctChangeBiLSTMAttention, args.symbols)


if __name__ == "__main__":
    main()
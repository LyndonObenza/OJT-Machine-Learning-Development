"""
backtest_5D_NASDAQ_mktctx.py — 5-day directional backtest for the MARKET-CONTEXT
NASDAQ BiLSTM models (trainingLSTM_NASDAQ_5D_v2_1.ipynb).

Identical to backtest_5D_NASDAQ.py EXCEPT the model also consumes 6 QQQ/SPY
market-context features (33 inputs total instead of 27). Use this script for the
v2_1 models; use backtest_5D_NASDAQ.py for the 27-feature baseline models.

    • 27 stationary single-name features + 6 market-context features:
          qqq_ret_1, qqq_mom_5, qqq_mom_20, spy_ret_1, spy_mom_20, rel_strength
      All trailing (pct_change) → value at date d uses only data ≤ d (no look-ahead).
      rel_strength = symbol ret_1 − qqq_ret_1 (excess daily return vs QQQ).
      Market context built once from QQQ_daily.csv / SPY_daily.csv, merged per
      symbol on datetime (left join).
    • Target is the 5-day move VOLATILITY-NORMALIZED (price_pct_change_5d_vn);
      model emits a z-score → de-normalize to percent via vol_denom_5d at entry.
    • Strategy B: trade when |pred 5d move| > THRESHOLD; LONG / SHORT; hold 5 days,
      non-overlapping per symbol. Per-symbol account starts at 750,000 (compounds).

Per-symbol seq_len read from the tuned <SYM>_best_params.json. Scalers per symbol
(<SYM>_feature_scaler.pkl, expecting 33 features). Window: strictly after 2018-12-31.

Usage:
    python backtest_5D_NASDAQ_mktctx.py
python backtest_5D_NASDAQ_mktctx.py --symbols AAPL NVDA --threshold 4.0
    
"""

import os
import sys
import json
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
import onnxruntime as ort

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DATA_DIR      = os.path.dirname(os.path.abspath(__file__))
THRESHOLD_PCT = 4.0          # |pred 5d move| must exceed this (percent) to trade
HOLD_DAYS     = 5            # trading days held per position
TEST_AFTER    = "2018-12-31" # backtest bars are strictly after this date


# 33 features — exact order the v2_1 feature_scaler was fit on (27 base + 6 market).
FEATURE_COLS = [
    # stationary price action
    "ret_1", "gap", "intraday_range", "body",
    # trend (stationary)
    "EMA_ratio", "macd_norm",
    # momentum
    "RSI_14", "ROC_5", "ROC_10", "ROC_20",
    # volatility (ratios, not levels)
    "ATR_ratio", "BB_pct", "realized_vol",
    # volume (stationary)
    "OBV_momentum", "volume_ratio", "volume_surge", "MFI_14",
    # lagged returns
    "returns_lag_1", "returns_lag_2", "returns_lag_3", "returns_lag_4", "returns_lag_5",
    # regime / persistence
    "above_ema10", "above_ema20", "roc5_pos", "roc20_pos", "up_streak",
    # market context (QQQ / SPY) — stationary, trailing, no look-ahead
    "qqq_ret_1", "qqq_mom_5", "qqq_mom_20", "spy_ret_1", "spy_mom_20", "rel_strength",
]


def load_market():
    """QQQ/SPY trailing market-context features, built once (no look-ahead)."""
    def _close(path, col):
        m = pd.read_csv(os.path.join(DATA_DIR, path), parse_dates=["datetime"])
        return (m[["datetime", "close_price"]].sort_values("datetime")
                 .rename(columns={"close_price": col}))
    q = _close("QQQ_daily.csv", "_qc")
    s = _close("SPY_daily.csv", "_sc")
    m = q.merge(s, on="datetime", how="outer").sort_values("datetime").reset_index(drop=True)
    m["qqq_ret_1"]  = m["_qc"].pct_change(1)
    m["qqq_mom_5"]  = m["_qc"].pct_change(5)
    m["qqq_mom_20"] = m["_qc"].pct_change(20)
    m["spy_ret_1"]  = m["_sc"].pct_change(1)
    m["spy_mom_20"] = m["_sc"].pct_change(20)
    return m[["datetime", "qqq_ret_1", "qqq_mom_5", "qqq_mom_20", "spy_ret_1", "spy_mom_20"]]


# ── Feature prep — exact replica of the v2_1 training cell ───────────────────────
def prep_features(df, market):
    df = df.copy()
    df = df.merge(market, on="datetime", how="left")   # trailing market context

    # lagged returns (no *100)
    for lag in range(1, 6):
        df[f"returns_lag_{lag}"] = df["close_price"].pct_change(1).shift(lag)

    # regime / persistence
    df["above_ema10"] = (df["close_price"] > df["EMA_10"]).astype(float)
    df["above_ema20"] = (df["close_price"] > df["EMA_20"]).astype(float)
    df["roc5_pos"]    = (df["ROC_5"]  > 0).astype(float)
    df["roc20_pos"]   = (df["ROC_20"] > 0).astype(float)
    daily_up  = df["close_price"].diff(1).gt(0).astype(int)
    streak_id = (daily_up != daily_up.shift()).cumsum()
    df["up_streak"] = (daily_up.groupby(streak_id).cumcount() + 1) * daily_up

    # stationary price-action (regime-transferable)
    df["ret_1"]          = df["close_price"].pct_change(1)
    df["gap"]            = df["open_price"] / df["close_price"].shift(1) - 1
    df["intraday_range"] = (df["high_price"] - df["low_price"]) / df["close_price"]
    df["body"]           = (df["close_price"] - df["open_price"]) / df["open_price"]
    df["macd_norm"]      = df["MACD_hist"] / df["close_price"]
    df["rel_strength"]   = df["ret_1"] - df["qqq_ret_1"]    # excess return vs QQQ (trailing)

    # 5-day forward close %-change + trailing vol denominator used to de-normalize
    # the model's volatility-scaled prediction back into a percent move.
    df["price_pct_change_5d"] = (df["close_price"].shift(-5) / df["close_price"] - 1) * 100
    _daily_ret = df["close_price"].pct_change()
    df["vol_denom_5d"] = (_daily_ret.rolling(20).std() * np.sqrt(5) * 100).clip(lower=0.5)

    return df.dropna().reset_index(drop=True)


def load_seq_len(model_dir, symbol):
    """seq_len comes from the tuned <SYM>_best_params.json saved during training."""
    bp_path = os.path.join(model_dir, f"{symbol}_best_params.json")
    with open(bp_path, "r") as f:
        params = json.load(f)
    return int(params["best_params"]["seq_len"])


def discover_symbols(models_dir):
    out = []
    if not os.path.isdir(models_dir):
        return out
    for name in sorted(os.listdir(models_dir)):
        d = os.path.join(models_dir, name)
        if (os.path.isdir(d)
                and os.path.exists(os.path.join(d, f"{name}_bilstm.onnx"))
                and os.path.exists(os.path.join(d, f"{name}_feature_scaler.pkl"))
                and os.path.exists(os.path.join(d, f"{name}_target_scaler.pkl"))
                and os.path.exists(os.path.join(d, f"{name}_best_params.json"))
                and os.path.exists(os.path.join(DATA_DIR, f"{name}_daily.csv"))):
            out.append(name)
    return out


METRIC_KEYS = ["Sharpe", "Volatility %", "Max Drawdown %", "Alpha %",
               "Information Ratio", "Upside Capture %", "Downside Capture %"]


def perf_metrics(r_strat, r_bench, periods=252):
    """Risk/benchmark metrics from daily strat returns vs a buy&hold benchmark.
    r_strat / r_bench are decimal daily returns (0 on flat days for strat)."""
    r_strat = np.asarray(r_strat, dtype=float)
    r_bench = np.asarray(r_bench, dtype=float)
    out = {k: np.nan for k in METRIC_KEYS}
    if len(r_strat) < 2:
        return out

    sd = r_strat.std(ddof=1)
    out["Volatility %"] = round(sd * np.sqrt(periods) * 100, 4)
    if sd > 0:
        out["Sharpe"] = round(r_strat.mean() / sd * np.sqrt(periods), 4)

    eq   = np.cumprod(1 + r_strat)
    peak = np.maximum.accumulate(eq)
    out["Max Drawdown %"] = round((eq / peak - 1).min() * 100, 4)

    var_b = r_bench.var(ddof=1)
    if var_b > 0:
        beta  = np.cov(r_strat, r_bench, ddof=1)[0, 1] / var_b
        alpha = r_strat.mean() - beta * r_bench.mean()      # daily intercept (rf=0)
        out["Alpha %"] = round(alpha * periods * 100, 4)    # annualized

    active = r_strat - r_bench
    te = active.std(ddof=1)
    if te > 0:
        out["Information Ratio"] = round(active.mean() / te * np.sqrt(periods), 4)

    up, dn = r_bench > 0, r_bench < 0
    if up.any() and r_bench[up].mean() != 0:
        out["Upside Capture %"]   = round(r_strat[up].mean() / r_bench[up].mean() * 100, 2)
    if dn.any() and r_bench[dn].mean() != 0:
        out["Downside Capture %"] = round(r_strat[dn].mean() / r_bench[dn].mean() * 100, 2)
    return out


# ── Backtest one symbol ────────────────────────────────────────────────────────
def backtest_symbol(symbol, models_dir, market, start_pp, threshold, hold, long_only=False):
    d         = os.path.join(models_dir, symbol)
    csv_path  = os.path.join(DATA_DIR, f"{symbol}_daily.csv")
    seq_len   = load_seq_len(d, symbol)

    feature_scaler = pickle.load(open(f"{d}/{symbol}_feature_scaler.pkl", "rb"))
    target_scaler  = pickle.load(open(f"{d}/{symbol}_target_scaler.pkl", "rb"))
    sess = ort.InferenceSession(f"{d}/{symbol}_bilstm.onnx", providers=["CPUExecutionProvider"])
    in_name, out_name = sess.get_inputs()[0].name, sess.get_outputs()[0].name

    full = prep_features(pd.read_csv(csv_path, parse_dates=["datetime"]), market)

    # Need seq_len bars of context before the first tradable (post-2018) bar, so
    # scale the whole frame, then restrict trading to bars after TEST_AFTER.
    X_all     = feature_scaler.transform(full[FEATURE_COLS].values).astype(np.float32)
    vol_denom = full["vol_denom_5d"].values.astype(np.float32)   # de-normalize preds
    after     = pd.Timestamp(TEST_AFTER)

    pp     = float(start_pp)
    trades = []
    daily  = []
    trade_spans = []                 # (entry_idx, +1 long / -1 short) for daily MTM
    auc_scores, auc_labels = [], []
    next_free = seq_len - 1

    for i in range(seq_len - 1, len(full) - hold):
        entry_day = full.iloc[i]["datetime"]
        if entry_day.normalize() <= after:        # date-only; csv datetimes carry a time
            continue

        window = X_all[i - seq_len + 1 : i + 1][np.newaxis, :, :]
        raw    = sess.run([out_name], {in_name: window})[0]      # (1, 1)
        # Model emits a volatility-normalized z-score → de-normalize to percent.
        pred_vn  = float(target_scaler.inverse_transform(raw)[0, 0])
        pred_pct = pred_vn * float(vol_denom[i])

        entry   = float(full.iloc[i]["close_price"])
        exit_px = float(full.iloc[i + hold]["close_price"])
        exit_day = full.iloc[i + hold]["datetime"]
        actual_move = (exit_px - entry) / entry * 100

        auc_scores.append(pred_pct)
        auc_labels.append(int(actual_move > 0))

        # Raw signal from the prediction, before the in-position lockout.
        if pred_pct > threshold:
            signal = "LONG"
        elif pred_pct < -threshold and not long_only:   # skip shorts in long-only mode
            signal = "SHORT"
        else:
            signal = "NONE"
        in_position = i < next_free

        traded, direction, ret_pct, win = False, "", None, None
        if signal != "NONE" and not in_position:
            traded    = True
            direction = signal
            ret_pct   = actual_move if direction == "LONG" else -actual_move
            win       = ret_pct > 0
            pnl_cash  = pp * (ret_pct / 100)
            pp       += pnl_cash
            next_free = i + hold
            trade_spans.append((i, 1 if direction == "LONG" else -1))

            trades.append({
                "Symbol"       : symbol,
                "Entry Date"   : entry_day.date(),
                "Exit Date"    : exit_day.date(),
                "Direction"    : direction,
                "Pred 5D %"    : round(pred_pct, 4),
                "Entry"        : round(entry, 4),
                "Exit"         : round(exit_px, 4),
                "Actual Move %": round(actual_move, 4),
                "Win"          : win,
                "Return %"     : round(ret_pct, 4),
                "P&L Cash"     : round(pnl_cash, 2),
                "PP After"     : round(pp, 2),
            })

        # Why no trade? (for supervisor visibility)
        if traded:
            status = "TRADED"
        elif in_position:
            status = "IN POSITION"          # signal ignored, prior trade still open
        elif signal == "NONE":
            status = "NO SIGNAL"            # |pred| below threshold
        else:
            status = "NO SIGNAL"

        daily.append({
            "Symbol"       : symbol,
            "Date"         : entry_day.date(),
            "Close"        : round(entry, 4),
            "Pred 5D %"    : round(pred_pct, 4),
            "Actual 5D %"  : round(actual_move, 4),
            "Signal"       : signal,                                  # LONG/SHORT/NONE
            "Status"       : status,                                  # TRADED/IN POSITION/NO SIGNAL
            "Direction"    : direction,                               # filled only if traded
            "Return %"     : round(ret_pct, 4) if traded else "",
            "Win"          : win if traded else "",
            "PP After"     : round(pp, 2),
        })

    from sklearn.metrics import roc_auc_score
    labels = np.array(auc_labels)
    auc = (round(roc_auc_score(labels, np.array(auc_scores)), 4)
           if len(labels) and 0 < labels.sum() < len(labels) else np.nan)

    # ── Daily strat vs buy&hold benchmark (post-2018) for risk/benchmark metrics ──
    close    = full["close_price"].values.astype(float)
    dret     = np.zeros(len(full))
    dret[1:] = close[1:] / close[:-1] - 1
    pos      = np.zeros(len(full))
    for ti, sgn in trade_spans:
        pos[ti + 1 : ti + hold + 1] = sgn        # return realized days entry+1..exit
    mask    = (full["datetime"].dt.normalize() > after).values
    r_strat = pos[mask] * dret[mask]             # 0 on flat (cash) days
    r_bench = dret[mask]                          # always-invested buy&hold
    metrics = perf_metrics(r_strat, r_bench)

    tdf  = pd.DataFrame(trades)
    n    = len(tdf)
    wins = int(tdf["Win"].sum()) if n else 0
    summary = {
        "Symbol"          : symbol,
        "Seq Len"         : seq_len,
        "Trades"          : n,
        "Wins"            : wins,
        "Losses"          : n - wins,
        "Hit Ratio %"     : round(100 * wins / n, 2) if n else 0.0,
        "AUC"             : auc,
        "Avg Ret/Trade %" : round(tdf["Return %"].mean(), 4) if n else 0.0,
        "Sharpe"             : metrics["Sharpe"],
        "Volatility %"       : metrics["Volatility %"],
        "Max Drawdown %"     : metrics["Max Drawdown %"],
        "Alpha %"            : metrics["Alpha %"],
        "Information Ratio"  : metrics["Information Ratio"],
        "Upside Capture %"   : metrics["Upside Capture %"],
        "Downside Capture %" : metrics["Downside Capture %"],
        "Start PP"        : round(start_pp, 2),
        "End PP"          : round(pp, 2),
        "Net P&L"         : round(pp - start_pp, 2),
        "Return %"        : round(100 * (pp - start_pp) / start_pp, 2),
    }
    return tdf, summary, pd.DataFrame(daily)


# ── Per-year performance breakdown ──────────────────────────────────────────────
def _year_stats(g, year, symbol=None):
    n    = len(g)
    wins = int(g["Win"].sum())
    row  = {} if symbol is None else {"Symbol": symbol}
    row.update({
        "Year"            : year,
        "Trades"          : n,
        "Wins"            : wins,
        "Losses"          : n - wins,
        "Win Rate %"      : round(100 * wins / n, 2) if n else 0.0,
        "Avg Ret/Trade %" : round(g["Return %"].mean(), 4) if n else 0.0,
        "Total Return %"  : round(g["Return %"].sum(), 4),
        "Net P&L"         : round(g["P&L Cash"].sum(), 2),
    })
    return row


def build_yearly(trades_all):
    """Yearly breakdown, all symbols pooled (+ TOTAL row)."""
    if trades_all.empty:
        return None
    ty = trades_all.copy()
    ty["Year"] = pd.to_datetime(ty["Entry Date"]).dt.year
    rows = [_year_stats(g, int(yr)) for yr, g in ty.groupby("Year")]
    rows.append(_year_stats(ty, "TOTAL"))
    return pd.DataFrame(rows)


def build_yearly_by_symbol(trades_all):
    """Per-symbol × per-year breakdown, with an ALL-years row per symbol."""
    if trades_all.empty:
        return None
    ty = trades_all.copy()
    ty["Year"] = pd.to_datetime(ty["Entry Date"]).dt.year
    rows = []
    for sym, gs in ty.groupby("Symbol"):
        for yr, g in gs.groupby("Year"):
            rows.append(_year_stats(g, int(yr), symbol=sym))
        rows.append(_year_stats(gs, "ALL", symbol=sym))
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description="5-day directional BiLSTM backtest — market-context (Strategy B)")
    p.add_argument("--symbols",   nargs="*", default=None)
    p.add_argument("--start-pp",  type=float, default=750000.0)
    p.add_argument("--threshold", type=float, default=THRESHOLD_PCT)
    p.add_argument("--hold",      type=int,   default=HOLD_DAYS)
    p.add_argument("--models-subdir", default="models_5D")
    p.add_argument("--long-only", action="store_true", help="skip SHORT signals (long-only)")
    args = p.parse_args()

    models_dir = os.path.join(DATA_DIR, args.models_subdir)
    symbols    = args.symbols or discover_symbols(models_dir)
    if not symbols:
        print(f"✗ No 5D models with matching *_daily.csv found in {models_dir}")
        return

    try:
        market = load_market()
    except Exception as e:
        print(f"✗ Could not build market context (QQQ_daily.csv / SPY_daily.csv): {e}")
        return

    print(f"\n{'═'*60}")
    print(f"  NASDAQ BiLSTM (QQQ/SPY market-context) — 5-Day Backtest (Strategy B)")
    print(f"  Window     : after {TEST_AFTER} → end of csv")
    print(f"  Threshold   : |pred 5d move| > {args.threshold}%")
    print(f"  Direction   : {'LONG-only' if args.long_only else 'LONG + SHORT'}")
    print(f"  Hold        : {args.hold} trading days")
    print(f"  Start PP    : {args.start_pp:,.0f} per symbol")
    print(f"  Symbols     : {len(symbols)} — {', '.join(symbols)}")
    print(f"{'═'*60}\n")

    all_trades, all_daily, summaries, failed = [], [], [], []
    for symbol in symbols:
        try:
            tdf, summary, ddf = backtest_symbol(
                symbol, models_dir, market, args.start_pp, args.threshold, args.hold,
                long_only=args.long_only)
        except Exception as e:
            print(f"  [{symbol}] error: {e}")
            failed.append(symbol)
            continue
        summaries.append(summary)
        if not tdf.empty:
            all_trades.append(tdf)
        if not ddf.empty:
            all_daily.append(ddf)
        print(f"  [{symbol}] seq={summary['Seq Len']}  {summary['Trades']} trades  "
              f"HitRatio {summary['Hit Ratio %']}%  AUC {summary['AUC']}  "
              f"AvgRet {summary['Avg Ret/Trade %']}%  "
              f"End PP {summary['End PP']:,.0f}  ({summary['Return %']:+.2f}%)")

    out_dir = os.path.join(DATA_DIR, "backtest_5D")
    os.makedirs(out_dir, exist_ok=True)
    tag = "_".join(symbols) if len(symbols) <= 6 else f"{len(symbols)}symbols"

    summ_out = None
    if summaries:
        sdf = pd.DataFrame(summaries)
        total = {
            "Symbol"          : "TOTAL",
            "Seq Len"         : "",
            "Trades"          : int(sdf["Trades"].sum()),
            "Wins"            : int(sdf["Wins"].sum()),
            "Losses"          : int(sdf["Losses"].sum()),
            "Hit Ratio %"     : round(100 * sdf["Wins"].sum() / max(sdf["Trades"].sum(), 1), 2),
            "AUC"             : round(sdf["AUC"].mean(), 4),
            "Avg Ret/Trade %" : round(sdf["Avg Ret/Trade %"].mean(), 4),
            "Sharpe"             : round(sdf["Sharpe"].mean(), 4),
            "Volatility %"       : round(sdf["Volatility %"].mean(), 4),
            "Max Drawdown %"     : round(sdf["Max Drawdown %"].mean(), 4),
            "Alpha %"            : round(sdf["Alpha %"].mean(), 4),
            "Information Ratio"  : round(sdf["Information Ratio"].mean(), 4),
            "Upside Capture %"   : round(sdf["Upside Capture %"].mean(), 2),
            "Downside Capture %" : round(sdf["Downside Capture %"].mean(), 2),
            "Start PP"        : round(sdf["Start PP"].sum(), 2),
            "End PP"          : round(sdf["End PP"].sum(), 2),
            "Net P&L"         : round(sdf["Net P&L"].sum(), 2),
            "Return %"        : round(100 * sdf["Net P&L"].sum() / max(sdf["Start PP"].sum(), 1), 2),
        }
        summ_out = pd.concat([sdf, pd.DataFrame([total])], ignore_index=True)

    out_path   = os.path.join(out_dir, f"backtest_5D_mktctx_{tag}.xlsx")
    trades_all = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    daily_all  = pd.concat(all_daily,  ignore_index=True) if all_daily  else pd.DataFrame()

    if summ_out is None and trades_all.empty and daily_all.empty:
        print(f"✗ Nothing to write — all symbols failed: {failed}")
        return

    yearly_out     = build_yearly(trades_all)
    yearly_sym_out = build_yearly_by_symbol(trades_all)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        if summ_out is not None:
            summ_out.to_excel(writer, sheet_name="Summary", index=False)
        if yearly_out is not None:
            yearly_out.to_excel(writer, sheet_name="Yearly", index=False)
        if yearly_sym_out is not None:
            yearly_sym_out.to_excel(writer, sheet_name="Yearly_BySymbol", index=False)
        if not trades_all.empty:
            trades_all.to_excel(writer, sheet_name="Trades", index=False)
        if not daily_all.empty:
            daily_all.to_excel(writer, sheet_name="Daily", index=False)
    print(f"\n✓ Saved → {out_path}\n")
    if summ_out is not None:
        print(summ_out.to_string(index=False))
    if yearly_out is not None:
        print(f"\n{'─'*60}\n  Yearly breakdown (by entry year, all symbols pooled)\n{'─'*60}")
        print(yearly_out.to_string(index=False))
    if yearly_sym_out is not None:
        print(f"\n{'─'*60}\n  Per-symbol × per-year breakdown\n{'─'*60}")
        print(yearly_sym_out.to_string(index=False))
    if failed:
        print(f"\n⚠ Failed symbols: {failed}")


if __name__ == "__main__":
    main()
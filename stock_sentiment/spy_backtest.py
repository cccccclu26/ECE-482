"""
SPY Benchmark Backtest - Compare ML model portfolio vs buy-and-hold SPY.

Strategy:
  - Monthly rebalancing
  - At each rebalance date, score all tickers using the ML model
  - Buy equal-weight top K stocks where P(up) is highest
  - Hold for ~21 trading days, then rebalance
  - Compare cumulative return vs SPY buy-and-hold

Usage:
  python spy_backtest.py
  python spy_backtest.py --start 2023-01-01 --end 2025-12-01 --top-k 3
"""
import argparse
import ssl
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

ssl._create_default_https_context = ssl._create_unverified_context

from ml_scorer import (
    generate_features_from_price_data,
    fetch_sentiment_for_date,
    load_model,
    FEATURE_COLS,
    SENTIMENT_FEATURE_COLS,
)

TICKERS = ["AAPL", "NVDA", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "JPM", "LLY"]


def fetch_all_price_data(tickers, start_date, end_date):
    """Fetch price data for all tickers with enough history for EMA warmup."""
    start_dt = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=250)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=10)

    data = {}
    for ticker in tickers:
        df = yf.Ticker(ticker).history(
            start=start_dt.strftime("%Y-%m-%d"),
            end=end_dt.strftime("%Y-%m-%d"),
        )
        df.index = df.index.tz_localize(None)
        data[ticker] = df
    return data


def get_ml_score(model, scaler, feature_cols, ticker, date, price_data):
    """
    Score a stock at a given date using the ML model.
    Uses only data available up to that date (no look-ahead bias).
    Returns P(up) probability 0-1.
    """
    hist = price_data[price_data.index <= date].copy()
    if len(hist) < 150:
        return 0.5

    features_df = generate_features_from_price_data(hist, forward_days=1)
    if len(features_df) == 0:
        return 0.5

    latest = features_df.iloc[-1]
    feat = {col: latest.get(col, 0) for col in FEATURE_COLS}

    # Add sentiment if model expects it
    if any(col in feature_cols for col in SENTIMENT_FEATURE_COLS):
        sent = fetch_sentiment_for_date(ticker, date)
        feat["sentiment_score"] = sent["sentiment_score"]
        feat["sentiment_confidence"] = sent["sentiment_confidence"]
        feat["news_count"] = sent["news_count"]

    X = np.array([[feat.get(col, 0) for col in feature_cols]])
    X_scaled = scaler.transform(X)
    prob_up = model.predict_proba(X_scaled)[0][1]
    return float(prob_up)


def score_to_weight(scores: dict, threshold: float = 0.52) -> dict:
    """
    Convert ML P(up) scores to portfolio weights.

    Only include stocks above threshold.
    Weight = (score - 0.5) i.e. excess confidence above neutral.
    Normalized so all weights sum to 1.
    """
    eligible = {t: s for t, s in scores.items() if s >= threshold}
    if not eligible:
        # Fallback: equal weight top 3
        top3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        eligible = dict(top3)

    excess = {t: s - 0.5 for t, s in eligible.items()}
    total = sum(excess.values())
    if total <= 0:
        n = len(eligible)
        return {t: 1/n for t in eligible}
    return {t: v / total for t, v in excess.items()}


def run_spy_backtest(start_date, end_date, top_k=3, rebalance_days=21, threshold=0.52):
    """
    Run the portfolio backtest and compare against SPY.

    Args:
        start_date: Backtest start "YYYY-MM-DD"
        end_date: Backtest end "YYYY-MM-DD"
        top_k: Max number of stocks to hold (unused — now score-weighted)
        rebalance_days: Trading days between rebalances (~21 = monthly)
        threshold: Min P(up) to be included in portfolio (default 0.52)
    """
    print(f"\n{'='*65}")
    print(f"SPY BENCHMARK BACKTEST")
    print(f"{'='*65}")
    print(f"Period:     {start_date} to {end_date}")
    print(f"Universe:   {', '.join(TICKERS)}")
    print(f"Strategy:   Score-weighted portfolio (threshold={threshold:.0%}), rebalance every {rebalance_days} days")
    print(f"Benchmark:  SPY buy-and-hold")

    # Load model
    model, scaler, metadata = load_model()
    feature_cols = metadata["feature_columns"]
    include_sentiment = metadata.get("include_sentiment", False)
    print(f"Model:      {metadata['n_samples']} samples, {metadata.get('test_accuracy', 'N/A')}% test accuracy")
    print(f"{'='*65}\n")

    # Fetch all price data
    print("Fetching price data for all tickers + SPY...")
    all_tickers = TICKERS + ["SPY"]
    price_data = fetch_all_price_data(all_tickers, start_date, end_date)

    # Get rebalance dates (actual trading days)
    spy_data = price_data["SPY"]
    trading_days = spy_data.index
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    valid_days = trading_days[(trading_days >= start_ts) & (trading_days <= end_ts)]

    rebalance_dates = []
    i = 0
    while i < len(valid_days):
        rebalance_dates.append(valid_days[i])
        i += rebalance_days

    print(f"Rebalance dates: {len(rebalance_dates)} ({rebalance_days}-day intervals)\n")

    # --- Run strategy ---
    portfolio_value = 10000.0
    spy_value = 10000.0
    portfolio_history = []
    trades_log = []

    spy_entry_price = spy_data[spy_data.index >= start_ts].iloc[0]["Close"]

    for r_idx, rebal_date in enumerate(rebalance_dates):
        # Determine hold period end
        if r_idx + 1 < len(rebalance_dates):
            next_rebal = rebalance_dates[r_idx + 1]
        else:
            # Last period: hold until end_date
            future = valid_days[valid_days > rebal_date]
            next_rebal = future[-1] if len(future) > 0 else rebal_date

        print(f"[{r_idx+1:02d}/{len(rebalance_dates)}] {rebal_date.strftime('%Y-%m-%d')} → {next_rebal.strftime('%Y-%m-%d')}")

        # Score all tickers at rebalance date
        scores = {}
        for ticker in TICKERS:
            p_up = get_ml_score(model, scaler, feature_cols, ticker, rebal_date, price_data[ticker])
            scores[ticker] = p_up

        # Convert scores to weights
        weights = score_to_weight(scores, threshold=threshold)
        selected = list(weights.keys())

        print(f"  Portfolio ({len(selected)} stocks):")
        for t in sorted(weights, key=weights.get, reverse=True):
            print(f"    {t}: P(up)={scores[t]:.0%}  weight={weights[t]:.1%}")

        # Calculate weighted portfolio return over hold period
        period_return = 0.0
        for ticker, weight in weights.items():
            ticker_data = price_data[ticker]
            future = ticker_data[ticker_data.index >= rebal_date]
            if len(future) < 2:
                continue
            entry_price = future.iloc[0]["Close"]
            exit_candidates = ticker_data[ticker_data.index >= next_rebal]
            if len(exit_candidates) == 0:
                continue
            exit_price = exit_candidates.iloc[0]["Close"]
            ret = (exit_price - entry_price) / entry_price
            period_return += weight * ret
        portfolio_value *= (1 + period_return)

        # SPY return over same period
        spy_future = spy_data[spy_data.index >= rebal_date]
        spy_next = spy_data[spy_data.index >= next_rebal]
        if len(spy_future) > 0 and len(spy_next) > 0:
            spy_period_return = (spy_next.iloc[0]["Close"] - spy_future.iloc[0]["Close"]) / spy_future.iloc[0]["Close"]
        else:
            spy_period_return = 0.0

        spy_value *= (1 + spy_period_return)

        print(f"  Portfolio: {period_return:+.2f}%  |  SPY: {spy_period_return:+.2f}%  |  Portfolio total: ${portfolio_value:,.0f}  SPY total: ${spy_value:,.0f}")

        portfolio_history.append({
            "date": rebal_date.strftime("%Y-%m-%d"),
            "selected": selected,
            "weights": {t: round(w, 4) for t, w in weights.items()},
            "scores": {t: round(scores[t], 4) for t in selected},
            "portfolio_return_pct": round(period_return * 100, 2),
            "spy_return_pct": round(spy_period_return * 100, 2),
            "portfolio_value": round(portfolio_value, 2),
            "spy_value": round(spy_value, 2),
        })

    # --- Final summary ---
    total_portfolio_return = (portfolio_value - 10000) / 10000 * 100
    total_spy_return = (spy_value - 10000) / 10000 * 100
    alpha = total_portfolio_return - total_spy_return

    n_months = len(rebalance_dates)
    years = n_months * rebalance_days / 252

    portfolio_cagr = ((portfolio_value / 10000) ** (1 / years) - 1) * 100 if years > 0 else 0
    spy_cagr = ((spy_value / 10000) ** (1 / years) - 1) * 100 if years > 0 else 0

    print(f"\n{'='*65}")
    print(f"FINAL RESULTS (${10000:,} initial investment)")
    print(f"{'='*65}")
    print(f"{'':30} {'Portfolio':>12}  {'SPY':>10}")
    print(f"{'Final Value':30} ${portfolio_value:>11,.0f}  ${spy_value:>9,.0f}")
    print(f"{'Total Return':30} {total_portfolio_return:>+11.1f}%  {total_spy_return:>+9.1f}%")
    print(f"{'CAGR':30} {portfolio_cagr:>+11.1f}%  {spy_cagr:>+9.1f}%")
    print(f"{'Alpha (vs SPY)':30} {alpha:>+11.1f}%")
    print(f"{'='*65}")

    if alpha > 0:
        print(f"\n✓ Portfolio BEAT SPY by {alpha:.1f}%")
    else:
        print(f"\n✗ Portfolio UNDERPERFORMED SPY by {abs(alpha):.1f}%")

    # Most selected stocks
    all_selected = [t for p in portfolio_history for t in p["selected"]]
    from collections import Counter
    freq = Counter(all_selected)
    print(f"\nMost selected stocks:")
    for ticker, count in freq.most_common():
        print(f"  {ticker}: selected {count}/{len(rebalance_dates)} periods ({count/len(rebalance_dates):.0%})")

    return portfolio_history


def main():
    parser = argparse.ArgumentParser(description="SPY Benchmark Backtest")
    parser.add_argument("--start", type=str, default="2023-01-01")
    parser.add_argument("--end", type=str, default="2025-12-01")
    parser.add_argument("--top-k", type=int, default=3, help="(unused) kept for compatibility")
    parser.add_argument("--rebalance-days", type=int, default=21, help="Trading days between rebalances (default: 21 = monthly)")
    parser.add_argument("--threshold", type=float, default=0.52, help="Min P(up) to include stock (default: 0.52)")
    args = parser.parse_args()

    run_spy_backtest(
        start_date=args.start,
        end_date=args.end,
        top_k=args.top_k,
        rebalance_days=args.rebalance_days,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()

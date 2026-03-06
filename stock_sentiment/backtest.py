"""
Backtesting Module - Evaluate the stock grading algorithm against historical data.

For each test date:
  1. Fetch news from that date's lookback window
  2. Run LLM sentiment analysis on those articles
  3. Calculate technical indicators as of that date
  4. Produce a combined score (sentiment 60% + technical 40%)
  5. Check actual price movement over the next N trading days
  6. Record whether the prediction was correct

Usage:
  python backtest.py --ticker AAPL --start 2023-03-01 --end 2026-02-01
  python backtest.py --ticker AAPL --start 2023-03-01 --end 2026-02-01 --interval 14
"""
import argparse
import json
import os
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import pandas as pd
import yfinance as yf

from news_fetcher import NewsFetcher
from sentiment_analyzer import SentimentAnalyzer
from technical_analysis import TechnicalAnalyzer
from combined_scorer import combine_scores
import config

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def get_trading_days(ticker: str, start: str, end: str) -> pd.DatetimeIndex:
    """Get actual trading days from yfinance history."""
    data = yf.Ticker(ticker).history(start=start, end=end)
    return data.index


def get_price_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch full price history for the backtest period (with extra buffer for forward returns)."""
    # Add 30-day buffer after end date for forward return calculation
    end_dt = datetime.strptime(end, "%Y-%m-%d") + timedelta(days=45)
    # Add 200-day buffer before start date for EMA calculation
    start_dt = datetime.strptime(start, "%Y-%m-%d") - timedelta(days=250)

    data = yf.Ticker(ticker).history(start=start_dt.strftime("%Y-%m-%d"), end=end_dt.strftime("%Y-%m-%d"))
    # Remove timezone info to avoid tz-naive vs tz-aware comparison issues
    data.index = data.index.tz_localize(None)
    return data


def calculate_technical_score_at_date(price_data: pd.DataFrame, target_date: pd.Timestamp) -> Dict:
    """
    Calculate technical indicators using only data available up to target_date.
    This prevents look-ahead bias.
    """
    # Only use data up to and including target_date
    historical = price_data[price_data.index <= target_date].copy()

    if len(historical) < 100:
        return {"technical_score": 50, "signal": "neutral", "ema": {}, "rsi": {}, "price": {}}

    analyzer = TechnicalAnalyzer()
    historical = analyzer.calculate_ema(historical)
    historical = analyzer.calculate_rsi(historical)

    ema_result = analyzer.calculate_ema_points(historical, period=30)
    rsi_result = analyzer.calculate_rsi_points(historical)

    technical_score = round(
        ema_result["normalized_score"] * 0.5 + rsi_result["normalized_score"] * 0.5, 2
    )

    if technical_score >= 60:
        signal = "bullish"
    elif technical_score <= 40:
        signal = "bearish"
    else:
        signal = "neutral"

    latest = historical.iloc[-1]

    return {
        "technical_score": technical_score,
        "signal": signal,
        "ema": {
            "score": ema_result["normalized_score"],
            "raw_points": ema_result["raw_points"],
        },
        "rsi": {
            "score": rsi_result["normalized_score"],
            "value": rsi_result["rsi_value"],
            "signal": rsi_result["signal"],
        },
        "price": {
            "current": round(float(latest["Close"]), 2),
        }
    }


def get_forward_return(price_data: pd.DataFrame, target_date: pd.Timestamp, days: int = 5) -> Optional[float]:
    """
    Calculate the actual forward return over the next N trading days after target_date.
    Returns percentage change, or None if data unavailable.
    """
    future = price_data[price_data.index > target_date]
    if len(future) < days:
        return None

    price_at_signal = price_data[price_data.index <= target_date].iloc[-1]["Close"]
    price_after = future.iloc[days - 1]["Close"]

    return round((price_after - price_at_signal) / price_at_signal * 100, 4)


class Backtester:
    """Run historical backtests on the stock grading algorithm."""

    def __init__(self, news_limit: int = 10, news_lookback: int = 7, forward_days: int = 5):
        """
        Args:
            news_limit: Number of news articles per test date
            news_lookback: Days of news to look back from each test date
            forward_days: Days ahead to measure actual price change
        """
        self.news_fetcher = NewsFetcher()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.news_limit = news_limit
        self.news_lookback = news_lookback
        self.forward_days = forward_days

    def run(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval_days: int = 7,
    ) -> pd.DataFrame:
        """
        Run backtest over a date range.

        Args:
            ticker: Stock ticker
            start_date: "YYYY-MM-DD" backtest start
            end_date: "YYYY-MM-DD" backtest end
            interval_days: Days between each test point (7=weekly, 30=monthly)

        Returns:
            DataFrame with all test results
        """
        print(f"\n{'='*70}")
        print(f"BACKTEST: {ticker}")
        print(f"Period: {start_date} to {end_date} (every {interval_days} days)")
        print(f"News: {self.news_limit} articles, {self.news_lookback}-day lookback")
        print(f"Forward return: {self.forward_days} trading days")
        print(f"{'='*70}")

        # Fetch all price data at once
        print(f"\nFetching price history...")
        price_data = get_price_data(ticker, start_date, end_date)
        print(f"  Got {len(price_data)} trading days of data")

        # Generate test dates
        trading_days = price_data.index
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)

        # Filter trading days within our range
        test_candidates = trading_days[(trading_days >= start_dt) & (trading_days <= end_dt)]

        # Sample at interval
        test_dates = []
        last_date = None
        for d in test_candidates:
            if last_date is None or (d - last_date).days >= interval_days:
                test_dates.append(d)
                last_date = d

        print(f"  Test points: {len(test_dates)}")
        print(f"\nRunning backtest...\n")

        results = []

        for i, test_date in enumerate(test_dates, 1):
            date_str = test_date.strftime("%Y-%m-%d")
            print(f"[{i}/{len(test_dates)}] {date_str}", end="")

            # 1. Fetch historical news
            news_end = test_date.strftime("%Y-%m-%d")
            news_start = (test_date - timedelta(days=self.news_lookback)).strftime("%Y-%m-%d")
            news_list = self.news_fetcher.get_news_by_date(
                ticker, news_start, news_end, limit=self.news_limit
            )

            # 2. Sentiment analysis
            if news_list:
                sentiment_results = self.sentiment_analyzer.analyze_news_batch(news_list)
                sentiment_agg = self.sentiment_analyzer.aggregate_sentiment(sentiment_results)
                sentiment_score = sentiment_agg["final_score"]
            else:
                sentiment_score = 50
                sentiment_agg = {"news_count": 0}

            # 3. Technical analysis (no look-ahead)
            tech_result = calculate_technical_score_at_date(price_data, test_date)
            technical_score = tech_result["technical_score"]

            # 4. Combined score
            combined = combine_scores(sentiment_score, technical_score)
            final_score = combined["final_score"]
            grade = combined["grade"]

            # 5. Actual forward return
            actual_return = get_forward_return(price_data, test_date, self.forward_days)

            # 6. Determine prediction vs actual
            predicted_direction = "up" if final_score >= 50 else "down"
            if actual_return is not None:
                actual_direction = "up" if actual_return >= 0 else "down"
                correct = predicted_direction == actual_direction
            else:
                actual_direction = None
                correct = None

            result = {
                "date": date_str,
                "final_score": final_score,
                "grade": grade,
                "sentiment_score": round(sentiment_score, 2),
                "technical_score": round(technical_score, 2),
                "news_count": sentiment_agg.get("news_count", 0),
                "predicted_direction": predicted_direction,
                "actual_return_pct": actual_return,
                "actual_direction": actual_direction,
                "correct": correct,
                "price": tech_result.get("price", {}).get("current"),
                "rsi": tech_result.get("rsi", {}).get("value"),
            }
            results.append(result)

            # Progress output
            correct_str = "HIT" if correct else ("MISS" if correct is not None else "N/A")
            ret_str = f"{actual_return:+.2f}%" if actual_return is not None else "N/A"
            print(f"  Score:{final_score:<6.1f} Pred:{predicted_direction:<5} Actual:{ret_str:<8} {correct_str}")

            # Small delay to respect API rate limits
            time.sleep(0.5)

        df = pd.DataFrame(results)
        self._print_summary(df, ticker)
        self._save_results(df, ticker, start_date, end_date)

        return df

    def _print_summary(self, df: pd.DataFrame, ticker: str):
        """Print backtest performance summary."""
        valid = df[df["correct"].notna()].copy()

        if len(valid) == 0:
            print("\nNo valid test points with forward returns.")
            return

        total = len(valid)
        correct = int(valid["correct"].sum())
        accuracy = correct / total * 100

        avg_return_correct = valid[valid["correct"] == True]["actual_return_pct"].mean()
        avg_return_wrong = valid[valid["correct"] == False]["actual_return_pct"].mean()

        print(f"\n{'='*70}")
        print(f"BACKTEST RESULTS: {ticker}")
        print(f"{'='*70}")
        print(f"Test Points:         {total}")
        print(f"Correct Predictions: {correct}")
        print(f"Directional Accuracy:{accuracy:.1f}%")
        print(f"{'='*70}")

        if pd.notna(avg_return_correct):
            print(f"Avg return (correct): {avg_return_correct:+.2f}%")
        if pd.notna(avg_return_wrong):
            print(f"Avg return (wrong):   {avg_return_wrong:+.2f}%")

        # Breakdown by grade
        print(f"\n--- Accuracy by Grade ---")
        for grade in df["grade"].unique():
            subset = valid[valid["grade"] == grade]
            if len(subset) > 0:
                g_correct = int(subset["correct"].sum())
                g_total = len(subset)
                g_acc = g_correct / g_total * 100
                print(f"  {grade:<20} {g_correct}/{g_total} ({g_acc:.0f}%)")

        # Score distribution
        print(f"\n--- Score Stats ---")
        print(f"  Mean:   {df['final_score'].mean():.1f}")
        print(f"  Median: {df['final_score'].median():.1f}")
        print(f"  Min:    {df['final_score'].min():.1f}")
        print(f"  Max:    {df['final_score'].max():.1f}")

    def _save_results(self, df: pd.DataFrame, ticker: str, start: str, end: str):
        """Save backtest results to files."""
        ensure_results_dir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"backtest_{ticker}_{start}_to_{end}_{timestamp}"

        csv_path = os.path.join(RESULTS_DIR, f"{prefix}.csv")
        df.to_csv(csv_path, index=False)

        json_path = os.path.join(RESULTS_DIR, f"{prefix}.json")
        summary = {
            "ticker": ticker,
            "period": f"{start} to {end}",
            "test_points": len(df),
            "forward_days": self.forward_days,
            "news_limit": self.news_limit,
            "news_lookback_days": self.news_lookback,
            "accuracy": None,
            "results": df.to_dict(orient="records"),
        }
        valid = df[df["correct"].notna()]
        if len(valid) > 0:
            summary["accuracy"] = round(int(valid["correct"].sum()) / len(valid) * 100, 2)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

        print(f"\n[Saved] CSV  -> {csv_path}")
        print(f"[Saved] JSON -> {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Backtest Stock Grading Algorithm")
    parser.add_argument("-t", "--ticker", type=str, default="AAPL", help="Stock ticker (default: AAPL)")
    parser.add_argument("--start", type=str, default="2023-03-01", help="Backtest start date YYYY-MM-DD (default: 2023-03-01)")
    parser.add_argument("--end", type=str, default="2026-02-01", help="Backtest end date YYYY-MM-DD (default: 2026-02-01)")
    parser.add_argument("--interval", type=int, default=7, help="Days between test points (default: 7 = weekly)")
    parser.add_argument("--news-limit", type=int, default=10, help="Articles per test date (default: 10)")
    parser.add_argument("--news-lookback", type=int, default=7, help="News lookback days per test date (default: 7)")
    parser.add_argument("--forward-days", type=int, default=5, help="Forward return measurement days (default: 5)")

    args = parser.parse_args()

    backtester = Backtester(
        news_limit=args.news_limit,
        news_lookback=args.news_lookback,
        forward_days=args.forward_days,
    )

    df = backtester.run(
        ticker=args.ticker.upper(),
        start_date=args.start,
        end_date=args.end,
        interval_days=args.interval,
    )


if __name__ == "__main__":
    main()

"""
ML Scorer - Logistic Regression model that learns scoring weights from historical data.

Replaces the rule-based combined_scorer.py with a data-driven approach:
  - Generates training data from historical price/technical indicators
  - Trains a Logistic Regression model to predict stock direction (up/down)
  - Outputs probability as a 0-100 score (replaces hardcoded 60/40 weights)

Usage:
  # Train model on historical data
  python ml_scorer.py --train --tickers AAPL NVDA MSFT --start 2023-01-01 --end 2025-12-01

  # Train and evaluate with test split
  python ml_scorer.py --train --tickers AAPL NVDA MSFT --start 2023-01-01 --end 2025-12-01 --eval

  # Show model details
  python ml_scorer.py --info
"""
import argparse
import os
import json
import pickle
import ssl
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Fix SSL for NLTK download on Windows
ssl._create_default_https_context = ssl._create_unverified_context
import nltk
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    _vader = SentimentIntensityAnalyzer()
except LookupError:
    nltk.download('vader_lexicon', quiet=True)
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    _vader = SentimentIntensityAnalyzer()

import config
from technical_analysis import TechnicalAnalyzer
from news_fetcher import NewsFetcher
from sentiment_analyzer import SentimentAnalyzer

# Paths
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "lr_scorer.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "lr_scaler.pkl")
META_PATH = os.path.join(MODEL_DIR, "lr_meta.json")


def get_per_stock_paths(ticker: str):
    """Get model paths for a per-stock model."""
    return (
        os.path.join(MODEL_DIR, f"{ticker}_lr.pkl"),
        os.path.join(MODEL_DIR, f"{ticker}_lr_scaler.pkl"),
        os.path.join(MODEL_DIR, f"{ticker}_lr_meta.json"),
    )


def save_per_stock_model(ticker: str, model, scaler, metadata: Dict):
    """Save a per-stock model to disk."""
    ensure_model_dir()
    model_path, scaler_path, meta_path = get_per_stock_paths(ticker)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)


def load_per_stock_model(ticker: str):
    """Load a per-stock model. Falls back to global model if not found."""
    model_path, scaler_path, meta_path = get_per_stock_paths(ticker)
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        with open(meta_path, "r") as f:
            metadata = json.load(f)
        return model, scaler, metadata
    # Fallback to global model
    return load_model()


def ensure_model_dir():
    os.makedirs(MODEL_DIR, exist_ok=True)


# ==================== Feature Engineering ====================

def generate_features_from_price_data(
    price_data: pd.DataFrame,
    forward_days: int = 5,
) -> pd.DataFrame:
    """
    Generate feature rows from historical price data.

    Features extracted per row:
      - ema100_uptrend (bool -> 0/1)
      - ema50_above_ema100 (bool -> 0/1)
      - ema25_above_ema100 (bool -> 0/1)
      - ema_raw_points (-4 to +4)
      - rsi_value (0-100)
      - price_vs_ema100 (% above/below EMA100)
      - price_vs_ema50 (% above/below EMA50)
      - price_vs_ema25 (% above/below EMA25)
      - volume_ratio (today's volume / 20-day avg volume)

    Label:
      - direction: 1 if price goes up over next N trading days, 0 otherwise

    Args:
        price_data: DataFrame with OHLCV data (must have >= 120 rows)
        forward_days: Days ahead to measure actual price change for labeling

    Returns:
        DataFrame with features and label columns
    """
    analyzer = TechnicalAnalyzer()

    data = price_data.copy()
    data = analyzer.calculate_ema(data)
    data = analyzer.calculate_rsi(data)

    # Volume moving average
    data["Volume_MA20"] = data["Volume"].rolling(window=20).mean()

    rows = []
    dates = data.index.tolist()

    # Need at least 100 days of history for EMA100 to stabilize
    start_idx = 100

    for i in range(start_idx, len(dates) - forward_days):
        current = data.iloc[i]
        future = data.iloc[i + forward_days]

        # Skip if any NaN in key columns
        if pd.isna(current["EMA100"]) or pd.isna(current["RSI"]):
            continue

        # EMA conditions (using 30-day lookback window)
        lookback = max(0, i - 30)
        window = data.iloc[lookback:i + 1]

        ema100_uptrend = int(current["EMA100"] > window["EMA100"].min())
        ema50_above = int(all(window["EMA50"].iloc[-min(30, len(window)):] > window["EMA100"].iloc[-min(30, len(window)):]))
        ema25_above = int(all(window["EMA25"].iloc[-min(30, len(window)):] > window["EMA100"].iloc[-min(30, len(window)):]))

        # EMA raw points (same logic as technical_analysis.py)
        points = 0
        trend = False
        if ema100_uptrend:
            points += 1
            trend = True
        else:
            points -= 1
        if ema50_above:
            points += 1
        elif not trend:
            points -= 1
        if ema25_above:
            points += 2
        elif not trend:
            points -= 2

        # Price relative to EMAs (percentage)
        price_vs_ema100 = (current["Close"] - current["EMA100"]) / current["EMA100"] * 100
        price_vs_ema50 = (current["Close"] - current["EMA50"]) / current["EMA50"] * 100
        price_vs_ema25 = (current["Close"] - current["EMA25"]) / current["EMA25"] * 100

        # Volume ratio
        vol_ratio = current["Volume"] / current["Volume_MA20"] if current["Volume_MA20"] > 0 else 1.0

        # Label: did price go up?
        forward_return = (future["Close"] - current["Close"]) / current["Close"] * 100
        direction = 1 if forward_return >= 0 else 0

        rows.append({
            "date": dates[i],
            "ema100_uptrend": ema100_uptrend,
            "ema50_above_ema100": ema50_above,
            "ema25_above_ema100": ema25_above,
            "ema_raw_points": points,
            "rsi_value": current["RSI"],
            "price_vs_ema100": round(price_vs_ema100, 4),
            "price_vs_ema50": round(price_vs_ema50, 4),
            "price_vs_ema25": round(price_vs_ema25, 4),
            "volume_ratio": round(vol_ratio, 4),
            "forward_return_pct": round(forward_return, 4),
            "direction": direction,
        })

    return pd.DataFrame(rows)


def generate_training_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    forward_days: int = 5,
    sample_every: int = 1,
) -> pd.DataFrame:
    """
    Generate training data from multiple tickers.

    Args:
        tickers: List of stock tickers
        start_date: Start date for historical data
        end_date: End date for historical data
        forward_days: Forward return measurement period

    Returns:
        Combined DataFrame with features and labels from all tickers
    """
    all_data = []

    for ticker in tickers:
        print(f"  Generating features for {ticker}...")
        try:
            # Fetch extra history before start_date for EMA warmup
            start_dt = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=200)
            data = yf.Ticker(ticker).history(
                start=start_dt.strftime("%Y-%m-%d"),
                end=end_date,
            )
            data.index = data.index.tz_localize(None)

            if len(data) < 150:
                print(f"    Skipped {ticker}: insufficient data ({len(data)} rows)")
                continue

            features = generate_features_from_price_data(data, forward_days)

            # Filter to requested date range
            features = features[features["date"] >= pd.Timestamp(start_date)]

            # Weekly/custom sampling to reduce data points (useful for LLM sentiment)
            if sample_every > 1:
                features = features.iloc[::sample_every].reset_index(drop=True)

            features["ticker"] = ticker

            print(f"    {len(features)} samples ({features['direction'].sum()} up, {len(features) - features['direction'].sum()} down)")
            all_data.append(features)

        except Exception as e:
            print(f"    Error for {ticker}: {e}")

    if not all_data:
        raise ValueError("No training data generated from any ticker")

    combined = pd.concat(all_data, ignore_index=True)
    print(f"\n  Total: {len(combined)} samples from {len(all_data)} tickers")
    return combined


# ==================== Sentiment Feature Generation ====================

def _vader_score(text: str) -> float:
    """Score a text with VADER, returns 0-100 scale (50=neutral)."""
    scores = _vader.polarity_scores(text)
    compound = scores["compound"]  # -1 to +1
    return round((compound + 1) / 2 * 100, 2)


def fetch_sentiment_for_date(ticker: str, date: pd.Timestamp) -> Dict:
    """
    Fetch news from Polygon.io for a 7-day window ending at date,
    score each article with VADER, return aggregated sentiment features.

    Returns dict with: sentiment_score, sentiment_confidence, news_count
    """
    end_dt = date
    start_dt = date - timedelta(days=7)

    url = f"{config.POLYGON_BASE_URL}/v2/reference/news"
    params = {
        "ticker": ticker,
        "published_utc.gte": start_dt.strftime("%Y-%m-%d"),
        "published_utc.lte": end_dt.strftime("%Y-%m-%d"),
        "limit": 10,
        "sort": "published_utc",
        "order": "desc",
        "apiKey": config.POLYGON_API_KEY,
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        articles = resp.json().get("results", [])
    except Exception:
        return {"sentiment_score": 50.0, "sentiment_confidence": 0.0, "news_count": 0}

    if not articles:
        return {"sentiment_score": 50.0, "sentiment_confidence": 0.0, "news_count": 0}

    scores = []
    for art in articles:
        title = art.get("title", "")
        desc = art.get("description", "")
        text = f"{title}. {desc}".strip()
        if text:
            scores.append(_vader_score(text))

    if not scores:
        return {"sentiment_score": 50.0, "sentiment_confidence": 0.0, "news_count": 0}

    avg_score = float(np.mean(scores))
    # Confidence: how far from neutral (50), normalized to 0-100
    confidence = float(abs(avg_score - 50) * 2)

    return {
        "sentiment_score": round(avg_score, 2),
        "sentiment_confidence": round(confidence, 2),
        "news_count": len(scores),
    }


def fetch_llm_sentiment_for_date(ticker: str, date: pd.Timestamp, news_limit: int = 5) -> Dict:
    """
    Fetch news from Polygon.io and score with real LLM ensemble (Claude + GPT-5).
    Same pipeline as main.py — used for high-quality training labels.

    Returns dict with: sentiment_score, sentiment_confidence, news_count
    """
    fetcher = NewsFetcher()
    analyzer = SentimentAnalyzer()

    end_str = date.strftime("%Y-%m-%d")
    start_str = (date - timedelta(days=7)).strftime("%Y-%m-%d")

    news_list = fetcher.get_news_by_date(ticker, start_str, end_str, limit=news_limit)
    if not news_list:
        return {"sentiment_score": 50.0, "sentiment_confidence": 0.0, "news_count": 0}

    results = analyzer.analyze_news_batch(news_list)
    agg = analyzer.aggregate_sentiment(results)

    return {
        "sentiment_score": float(agg["final_score"]),
        "sentiment_confidence": float(agg["avg_confidence"]),
        "news_count": int(agg["news_count"]),
    }


def add_sentiment_to_training_data(df: pd.DataFrame, ticker_col: str = "ticker") -> pd.DataFrame:
    """
    Add VADER-based sentiment columns to an existing feature DataFrame.
    Fetches Polygon.io news for each row's date+ticker.

    Args:
        df: DataFrame with 'date' and 'ticker' columns
        ticker_col: Column name containing ticker symbol

    Returns:
        DataFrame with added sentiment_score, sentiment_confidence, news_count columns
    """
    sentiment_scores = []
    total = len(df)

    # Group by ticker to reduce API call overhead
    for ticker, group in df.groupby(ticker_col):
        print(f"    Fetching VADER sentiment for {ticker} ({len(group)} samples)...")
        for idx, row in group.iterrows():
            sent = fetch_sentiment_for_date(ticker, row["date"])
            sentiment_scores.append({"_idx": idx, **sent})
            time.sleep(0.05)  # Gentle rate limiting

    sent_df = pd.DataFrame(sentiment_scores).set_index("_idx")
    result = df.copy()
    result["sentiment_score"] = sent_df["sentiment_score"]
    result["sentiment_confidence"] = sent_df["sentiment_confidence"]
    result["news_count"] = sent_df["news_count"]
    return result


def add_llm_sentiment_to_training_data(df: pd.DataFrame, ticker_col: str = "ticker", news_limit: int = 5) -> pd.DataFrame:
    """
    Add real LLM sentiment (Claude + GPT-5) to training data.
    Much slower than VADER but produces higher-quality sentiment features
    that match what the model sees at prediction time.

    Args:
        df: DataFrame with 'date' and 'ticker' columns
        ticker_col: Column name for ticker symbol
        news_limit: Articles per date (keep low to control API calls)

    Returns:
        DataFrame with sentiment_score, sentiment_confidence, news_count columns
    """
    sentiment_rows = []
    total = len(df)
    done = 0

    for ticker, group in df.groupby(ticker_col):
        print(f"\n  [{ticker}] {len(group)} dates to analyze...")
        for idx, row in group.iterrows():
            done += 1
            date_str = row["date"].strftime("%Y-%m-%d")
            print(f"    [{done}/{total}] {ticker} {date_str}", end=" ", flush=True)
            try:
                sent = fetch_llm_sentiment_for_date(ticker, row["date"], news_limit=news_limit)
                print(f"→ score={sent['sentiment_score']:.0f} conf={sent['sentiment_confidence']:.0f} n={sent['news_count']}")
            except Exception as e:
                print(f"→ failed ({e}), using default")
                sent = {"sentiment_score": 50.0, "sentiment_confidence": 0.0, "news_count": 0}
            sentiment_rows.append({"_idx": idx, **sent})
            time.sleep(1.0)  # Respect API rate limits

    sent_df = pd.DataFrame(sentiment_rows).set_index("_idx")
    result = df.copy()
    result["sentiment_score"] = sent_df["sentiment_score"]
    result["sentiment_confidence"] = sent_df["sentiment_confidence"]
    result["news_count"] = sent_df["news_count"]
    return result


# ==================== Feature columns ====================

FEATURE_COLS = [
    "ema100_uptrend",
    "ema50_above_ema100",
    "ema25_above_ema100",
    "ema_raw_points",
    "rsi_value",
    "price_vs_ema100",
    "price_vs_ema50",
    "price_vs_ema25",
    "volume_ratio",
]

# When sentiment is available, these are added
SENTIMENT_FEATURE_COLS = [
    "sentiment_score",
    "sentiment_confidence",
    "news_count",
]


# ==================== Model Training ====================

def train_model(
    train_data: pd.DataFrame,
    include_sentiment: bool = False,
) -> Tuple[LogisticRegression, StandardScaler, Dict]:
    """
    Train a Logistic Regression model on the provided data.

    Args:
        train_data: DataFrame with feature columns and 'direction' label
        include_sentiment: Whether to include sentiment features (if available)

    Returns:
        (model, scaler, metadata_dict)
    """
    feature_cols = FEATURE_COLS.copy()
    if include_sentiment:
        for col in SENTIMENT_FEATURE_COLS:
            if col in train_data.columns:
                feature_cols.append(col)

    X = train_data[feature_cols].values
    y = train_data["direction"].values

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train logistic regression
    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",  # Handle class imbalance
        random_state=42,
    )
    model.fit(X_scaled, y)

    # Feature importance (coefficients)
    coef_dict = {col: round(float(c), 4) for col, c in zip(feature_cols, model.coef_[0])}

    metadata = {
        "trained_at": datetime.now().isoformat(),
        "feature_columns": feature_cols,
        "n_samples": len(y),
        "n_up": int(y.sum()),
        "n_down": int(len(y) - y.sum()),
        "coefficients": coef_dict,
        "intercept": round(float(model.intercept_[0]), 4),
        "include_sentiment": include_sentiment,
    }

    return model, scaler, metadata


def evaluate_model(
    model: LogisticRegression,
    scaler: StandardScaler,
    test_data: pd.DataFrame,
    feature_cols: List[str],
) -> Dict:
    """Evaluate model on test data."""
    X = test_data[feature_cols].values
    y = test_data["direction"].values
    X_scaled = scaler.transform(X)

    y_pred = model.predict(X_scaled)
    accuracy = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, target_names=["down", "up"], output_dict=True)

    return {
        "accuracy": round(accuracy * 100, 2),
        "n_test_samples": len(y),
        "report": report,
    }


def save_model(model: LogisticRegression, scaler: StandardScaler, metadata: Dict):
    """Save trained model, scaler, and metadata to disk."""
    ensure_model_dir()

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  Model saved to {MODEL_PATH}")
    print(f"  Scaler saved to {SCALER_PATH}")
    print(f"  Metadata saved to {META_PATH}")


def load_model() -> Tuple[LogisticRegression, StandardScaler, Dict]:
    """Load trained model, scaler, and metadata from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. Run 'python ml_scorer.py --train' first."
        )

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(META_PATH, "r") as f:
        metadata = json.load(f)

    return model, scaler, metadata


# ==================== Prediction (replaces combined_scorer) ====================

class MLScorer:
    """
    ML-based scorer that replaces the rule-based combined_scorer.

    Uses a trained Logistic Regression model to predict stock direction
    and output a probability-based score (0-100).
    """

    def __init__(self):
        self.model, self.scaler, self.metadata = load_model()
        self.feature_cols = self.metadata["feature_columns"]

    def predict(
        self,
        sentiment_score: float = None,
        sentiment_confidence: float = None,
        news_count: int = None,
        technical_data: Dict = None,
        price_data: pd.DataFrame = None,
    ) -> Dict:
        """
        Predict stock direction and score.

        Can work with:
          - Technical features only (from price_data or technical_data dict)
          - Technical + sentiment features (when sentiment_score is provided)

        Args:
            sentiment_score: 0-100 LLM sentiment score (optional)
            sentiment_confidence: 0-100 confidence (optional)
            news_count: Number of news articles analyzed (optional)
            technical_data: Dict from TechnicalAnalyzer.analyze() (optional)
            price_data: Raw price DataFrame - will extract features (optional)

        Returns:
            Dict with ml_score (0-100), predicted_direction, probability, grade
        """
        features = {}

        # Extract technical features from technical_data dict
        if technical_data:
            ema = technical_data.get("ema", {})
            rsi = technical_data.get("rsi", {})
            price = technical_data.get("price", {})

            features["ema100_uptrend"] = int(ema.get("ema100_uptrend", False))
            features["ema50_above_ema100"] = int(ema.get("ema50_above_ema100", False))
            features["ema25_above_ema100"] = int(ema.get("ema25_above_ema100", False))
            features["ema_raw_points"] = ema.get("raw_points", 0)
            features["rsi_value"] = rsi.get("value", 50)

            current_price = price.get("current", 0)
            ema100 = price.get("ema100", current_price)
            ema50 = price.get("ema50", current_price)
            ema25 = price.get("ema25", current_price)

            features["price_vs_ema100"] = (current_price - ema100) / ema100 * 100 if ema100 else 0
            features["price_vs_ema50"] = (current_price - ema50) / ema50 * 100 if ema50 else 0
            features["price_vs_ema25"] = (current_price - ema25) / ema25 * 100 if ema25 else 0
            features["volume_ratio"] = 1.0  # Not available from analyze() output

        # Add sentiment features if model expects them
        if "sentiment_score" in self.feature_cols and sentiment_score is not None:
            features["sentiment_score"] = sentiment_score
        if "sentiment_confidence" in self.feature_cols and sentiment_confidence is not None:
            features["sentiment_confidence"] = sentiment_confidence
        if "news_count" in self.feature_cols and news_count is not None:
            features["news_count"] = news_count

        # Build feature vector in correct order
        X = np.array([[features.get(col, 0) for col in self.feature_cols]])
        X_scaled = self.scaler.transform(X)

        # Predict
        prob = self.model.predict_proba(X_scaled)[0]  # [P(down), P(up)]
        prob_up = prob[1]
        ml_score = round(prob_up * 100, 2)

        predicted_direction = "up" if prob_up >= 0.5 else "down"

        # Grade label (same thresholds as combined_scorer)
        if ml_score >= 80:
            grade = "STRONG BUY"
        elif ml_score >= 65:
            grade = "BUY"
        elif ml_score >= 55:
            grade = "SLIGHTLY BULLISH"
        elif ml_score >= 45:
            grade = "NEUTRAL"
        elif ml_score >= 35:
            grade = "SLIGHTLY BEARISH"
        elif ml_score >= 20:
            grade = "SELL"
        else:
            grade = "STRONG SELL"

        return {
            "ml_score": ml_score,
            "predicted_direction": predicted_direction,
            "probability_up": round(prob_up, 4),
            "probability_down": round(prob[0], 4),
            "grade": grade,
            "features_used": features,
            "model_type": "LogisticRegression",
        }


# ==================== CLI ====================

def print_model_info():
    """Print information about the trained model."""
    try:
        _, _, metadata = load_model()
    except FileNotFoundError:
        print("No trained model found. Run 'python ml_scorer.py --train' first.")
        return

    print(f"\n{'='*60}")
    print("ML Scorer Model Info")
    print(f"{'='*60}")
    print(f"Trained at:       {metadata['trained_at']}")
    print(f"Training samples: {metadata['n_samples']} ({metadata['n_up']} up, {metadata['n_down']} down)")
    print(f"Features:         {len(metadata['feature_columns'])}")
    print(f"Intercept:        {metadata['intercept']}")

    if "test_accuracy" in metadata:
        print(f"Test Accuracy:    {metadata['test_accuracy']}%")

    print(f"\n--- Feature Coefficients (Learned Weights) ---")
    coefs = metadata["coefficients"]
    sorted_coefs = sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True)
    for name, coef in sorted_coefs:
        direction = "+" if coef > 0 else ""
        bar = "#" * int(abs(coef) * 5)
        print(f"  {name:<25} {direction}{coef:<8} {bar}")

    print(f"\nThese coefficients show how much each feature contributes to")
    print(f"the 'up' prediction. Positive = bullish signal, Negative = bearish signal.")


def main():
    parser = argparse.ArgumentParser(description="ML Scorer - Train Logistic Regression model")
    parser.add_argument("--train", action="store_true", help="Train model on historical data")
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "INTC", "CRM"],
                        help="Tickers to use for training data")
    parser.add_argument("--start", type=str, default="2023-01-01", help="Training data start date")
    parser.add_argument("--end", type=str, default="2025-12-01", help="Training data end date")
    parser.add_argument("--forward-days", type=int, default=5, help="Forward return period in days")
    parser.add_argument("--eval", action="store_true", help="Evaluate with train/test split")
    parser.add_argument("--sentiment", action="store_true", help="Include VADER sentiment features from Polygon.io news")
    parser.add_argument("--llm-sentiment", action="store_true", help="Include real LLM (Claude+GPT-5) sentiment features — slower but higher quality")
    parser.add_argument("--sample-every", type=int, default=1, help="Sample every N rows (e.g. 5=weekly). Recommended with --llm-sentiment to reduce API calls")
    parser.add_argument("--per-stock", action="store_true", help="Train and save a separate model for each ticker")
    parser.add_argument("--info", action="store_true", help="Show trained model info")

    args = parser.parse_args()

    if args.info:
        print_model_info()
        return

    if args.train:
        print(f"\n{'='*60}")
        print("ML Scorer - Training Logistic Regression")
        print(f"{'='*60}")
        print(f"Tickers: {', '.join(args.tickers)}")
        print(f"Period:  {args.start} to {args.end}")
        print(f"Forward: {args.forward_days} days")
        print()

        use_llm = args.llm_sentiment
        sample_every = args.sample_every if use_llm and args.sample_every == 1 else args.sample_every
        # Default weekly sampling when using LLM sentiment (to control API cost)
        if use_llm and args.sample_every == 1:
            sample_every = 5
            print("  [Auto] --llm-sentiment detected: defaulting to --sample-every 5 (weekly) to reduce API calls")

        # Generate training data
        print("[1/3] Generating training data from historical prices...")
        data = generate_training_data(
            tickers=args.tickers,
            start_date=args.start,
            end_date=args.end,
            forward_days=args.forward_days,
            sample_every=sample_every,
        )

        if use_llm:
            print(f"\n[1b] Adding LLM sentiment (Claude+GPT-5) from Polygon.io news...")
            print(f"  {len(data)} data points × ~5 articles × 2 models ≈ {len(data)*10} API calls")
            data = add_llm_sentiment_to_training_data(data)
            print(f"\n  LLM sentiment added. Avg score: {data['sentiment_score'].mean():.1f}, "
                  f"Avg confidence: {data['sentiment_confidence'].mean():.1f}")
        elif args.sentiment:
            print("\n[1b] Adding VADER sentiment features from Polygon.io news...")
            data = add_sentiment_to_training_data(data)
            print(f"  Sentiment added. Avg score: {data['sentiment_score'].mean():.1f}, "
                  f"Avg news/day: {data['news_count'].mean():.1f}")

        if args.per_stock:
            print(f"\n[Per-Stock Mode] Training individual model for each ticker...")
            summary = []
            for ticker in args.tickers:
                print(f"\n{'─'*50}")
                print(f"Training: {ticker}")
                ticker_data = data[data["ticker"] == ticker].copy()
                if len(ticker_data) < 20:
                    print(f"  Skipped: insufficient data ({len(ticker_data)} samples)")
                    continue
                t_train, t_test = train_test_split(ticker_data, test_size=0.2, random_state=42, stratify=ticker_data["direction"])
                t_model, t_scaler, t_meta = train_model(t_train, include_sentiment=args.sentiment or use_llm)
                eval_r = evaluate_model(t_model, t_scaler, t_test, t_meta["feature_columns"])
                t_meta["test_accuracy"] = eval_r["accuracy"]
                t_meta["ticker"] = ticker
                # Retrain on full ticker data
                t_model, t_scaler, t_meta_full = train_model(ticker_data, include_sentiment=args.sentiment or use_llm)
                t_meta_full["test_accuracy"] = eval_r["accuracy"]
                t_meta_full["ticker"] = ticker
                save_per_stock_model(ticker, t_model, t_scaler, t_meta_full)
                sent_coef = t_meta_full["coefficients"].get("sentiment_score", 0)
                print(f"  Accuracy: {eval_r['accuracy']:.1f}%  |  sentiment_score weight: {sent_coef:+.3f}")
                summary.append({"ticker": ticker, "accuracy": eval_r["accuracy"], "sentiment_weight": sent_coef, "samples": len(ticker_data)})

            print(f"\n{'='*55}")
            print(f"PER-STOCK TRAINING SUMMARY")
            print(f"{'='*55}")
            print(f"  {'Ticker':<8} {'Samples':>8} {'Accuracy':>10} {'Sentiment Weight':>18}")
            print(f"  {'─'*46}")
            for s in summary:
                print(f"  {s['ticker']:<8} {s['samples']:>8} {s['accuracy']:>9.1f}%  {s['sentiment_weight']:>+17.3f}")
            avg_acc = sum(s['accuracy'] for s in summary) / len(summary)
            avg_sent = sum(s['sentiment_weight'] for s in summary) / len(summary)
            print(f"  {'─'*46}")
            print(f"  {'Average':<8} {'':>8} {avg_acc:>9.1f}%  {avg_sent:>+17.3f}")
            print(f"\nPer-stock models saved to: {MODEL_DIR}")
            return

        if args.eval:
            # Split into train/test
            print("\n[2/3] Training with 80/20 train/test split...")
            train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data["direction"])
            print(f"  Train: {len(train_data)} samples | Test: {len(test_data)} samples")

            model, scaler, metadata = train_model(train_data, include_sentiment=args.sentiment or use_llm)

            # Evaluate on test set
            print("\n[3/3] Evaluating on test set...")
            eval_result = evaluate_model(model, scaler, test_data, metadata["feature_columns"])
            metadata["test_accuracy"] = eval_result["accuracy"]
            metadata["test_samples"] = eval_result["n_test_samples"]

            print(f"\n  Test Accuracy: {eval_result['accuracy']:.1f}%")
            print(f"\n  Classification Report:")
            report = eval_result["report"]
            print(f"  {'':>12} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}")
            for label in ["down", "up"]:
                r = report[label]
                print(f"  {label:>12} {r['precision']:>10.2f} {r['recall']:>10.2f} {r['f1-score']:>10.2f} {r['support']:>10.0f}")

            # Now retrain on ALL data for the final saved model
            print("\n  Retraining on full dataset for final model...")
            model, scaler, metadata_full = train_model(data, include_sentiment=args.sentiment or use_llm)
            metadata_full["test_accuracy"] = eval_result["accuracy"]
            metadata_full["test_samples"] = eval_result["n_test_samples"]
            save_model(model, scaler, metadata_full)

        else:
            # Train on all data
            print("\n[2/3] Training on full dataset...")
            model, scaler, metadata = train_model(data, include_sentiment=args.sentiment or use_llm)

            print("\n[3/3] Saving model...")
            save_model(model, scaler, metadata)

        # Print model info
        print_model_info()

        print(f"\nDone! Model is ready to use.")
        print(f"To use in code:  from ml_scorer import MLScorer; scorer = MLScorer()")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

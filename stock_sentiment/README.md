# LLM-Based Stock Grading Algorithm

**ECE 482 Senior Design Project** — University of Miami, Spring 2026

A stock analysis system that combines dual-model LLM sentiment analysis (Claude 3.7 Sonnet + GPT-5), technical indicators (EMA/RSI), and per-stock Logistic Regression models to produce data-driven probability scores for portfolio allocation. Benchmarked against SPY buy-and-hold.

---

## Performance Summary

| Backtest Period | Sample Type | Portfolio | SPY | Alpha |
|----------------|-------------|-----------|-----|-------|
| 2019-01 to 2023-01 | Mixed (partial out-of-sample) | +347.1% | +70.4% | **+276.7%** |
| 2021-09 to 2026-03 | Mixed (in-sample + out-of-sample) | +156.3% | +43.6% | **+112.7%** |

- **Training period**: 2021-01 to 2024-12 (recorded in each model's `meta.json`)
- **Average directional accuracy**: 61.7% across 10 per-stock models
- **Prediction horizon**: 63 trading days (~3 months)

---

## Architecture

```
Input: 10 stock tickers
         |
         +--- [1] News Fetcher (Polygon.io API)
         |         Fetch recent financial news articles
         |
         +--- [2] Sentiment Analyzer (WaveSpeed AI)
         |         Claude 3.7 Sonnet -> score -> wait 1s
         |         GPT-5 -> score -> average both
         |         Cache results to sentiment_cache.json
         |
         +--- [3] Technical Analysis (Polygon.io / yfinance)
         |         EMA25, EMA50, EMA100 trend signals
         |         RSI(14) momentum
         |         Volume ratio (current / 20-day avg)
         |
         +--- [4] ML Scorer (Per-Stock Logistic Regression)
         |         12-feature vector -> P(up) probability
         |         Clip output to [0.40, 0.90]
         |
         +--- [5] Portfolio Engine
                   Score-weighted allocation
                   Defensive cash mode (P(up) < 52% excluded)
                   Monthly rebalance vs SPY benchmark
```

### 12-Feature Vector

| # | Feature | Source | Range |
|---|---------|--------|-------|
| 1 | EMA100 uptrend | Technical | {0, 1} |
| 2 | EMA50 > EMA100 | Technical | {0, 1} |
| 3 | EMA25 > EMA100 | Technical | {0, 1} |
| 4 | EMA alignment score | Technical | -4 to +4 |
| 5 | RSI(14) | Technical | 0-100 |
| 6 | Price vs EMA100 | Technical | % deviation |
| 7 | Price vs EMA50 | Technical | % deviation |
| 8 | Price vs EMA25 | Technical | % deviation |
| 9 | Volume ratio | Technical | ratio |
| 10 | Sentiment score | LLM | 0-100 |
| 11 | Sentiment confidence | LLM | 0-100 |
| 12 | News count | LLM | integer |

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/cccccclu26/ECE-482.git
cd ECE-482/stock_sentiment
pip install -r requirements.txt
```

### 2. Set up API keys

Create a `.env` file in `stock_sentiment/`:

```
POLYGON_API_KEY=your_polygon_key_here
WAVESPEED_API_KEY=your_wavespeed_key_here
```

| Key | Provider | Purpose |
|-----|----------|---------|
| Polygon.io | [polygon.io](https://polygon.io/) (free tier) | News articles & price data |
| WaveSpeed AI | [wavespeed.ai](https://wavespeed.ai/) | Claude 3.7 + GPT-5 inference |

### 3. Run analysis

```bash
# Analyze a single stock with ML scoring
python main.py -t AAPL --ml

# Analyze all 10 stocks
python main.py -a --ml

# Run portfolio backtest vs SPY
python spy_backtest.py --start 2021-09-01 --end 2026-03-01

# Train per-stock models (requires LLM API access)
python ml_scorer.py --train --llm-sentiment --per-stock \
    --tickers AAPL NVDA META JPM TSLA MSFT AMZN GOOGL AVGO LLY \
    --start 2021-01-01 --end 2024-12-31 --eval

# Show trained model info
python ml_scorer.py --info
```

---

## Stock Universe

10 per-stock Logistic Regression models, each independently trained:

| Ticker | Accuracy | Sentiment Weight | Key Finding |
|--------|----------|-----------------|-------------|
| AAPL | 58.3% | +0.003 | Primarily technical-driven |
| NVDA | 58.3% | +0.033 | Primarily technical-driven |
| META | 66.7% | **+0.667** | Positive news strongly predicts up |
| JPM | **75.0%** | **-0.554** | Contrarian: positive news predicts down |
| TSLA | 58.3% | +0.195 | Mixed signal |
| MSFT | 66.7% | +0.393 | Sentiment + technical |
| AMZN | 33.3% | -0.242 | Weak signal (below random) |
| GOOGL | 66.7% | +0.229 | Mixed signal |
| AVGO | 66.7% | -0.363 | Contrarian sentiment |
| LLY | 66.7% | -0.403 | Contrarian sentiment |
| **Avg** | **61.7%** | | Exceeds 60% target |

---

## Key Design Decisions

- **Per-stock models over global model**: A single global model achieved only 42.4% accuracy (below random). Stocks like META and JPM have opposite sentiment dynamics — mixing them destroyed the signal.
- **LLM for both training and prediction**: Using VADER for training but LLM for prediction creates a feature distribution mismatch. Both stages use the same Claude+GPT-5 ensemble.
- **63-day prediction horizon**: Aligned with long-term investor goals. Reduces noise compared to 5-day prediction.
- **Probability clipping [0.40, 0.90]**: Prevents LR saturation and ensures diversified portfolio weights.
- **Defensive cash mode**: Stocks with P(up) < 52% are excluded. When all stocks are weak, the portfolio goes to 100% cash.
- **Sentiment caching**: LLM responses stored locally by (ticker, date). Fuzzy matching (+-3 days) reduces redundant API calls across backtests.

---

## Project Structure

```
stock_sentiment/
├── main.py                 # Entry point: single/multi stock analysis
├── ml_scorer.py            # Per-stock LR model: train, predict, evaluate
├── spy_backtest.py         # Portfolio backtest engine vs SPY benchmark
├── sentiment_analyzer.py   # Dual-model LLM sentiment (Claude + GPT-5)
├── technical_analysis.py   # EMA25/50/100 + RSI(14) + volume ratio
├── news_fetcher.py         # Polygon.io news API client
├── combined_scorer.py      # Rule-based scorer (60% sent + 40% tech)
├── backtest.py             # Single-stock historical backtester
├── config.py               # API keys, stock list, LLM model config
├── requirements.txt        # Python dependencies
├── .env                    # API keys (not committed)
├── models/
│   ├── {TICKER}_lr.pkl          # Per-stock trained model (x10)
│   ├── {TICKER}_lr_scaler.pkl   # Per-stock StandardScaler (x10)
│   ├── {TICKER}_lr_meta.json    # Coefficients, accuracy, training range (x10)
│   └── sentiment_cache.json     # Cached LLM sentiment scores
└── results/                # Analysis output (JSON + CSV)
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | `pip install -r requirements.txt` |
| `POLYGON_API_KEY is required` | Create `.env` file with your keys |
| `No trained model found` | Run `python ml_scorer.py --train` first |
| `429 Too Many Requests` | WaveSpeed rate limit — system uses sequential calls with 1s delay |
| `401 Unauthorized` | Check WaveSpeed API key in `.env` |
| Slow backtest | Sentiment cache reduces repeat API calls; use `--start`/`--end` to limit range |

---

## Important Notes

- **Do NOT share your `.env` file** — it contains private API keys
- This is an **educational project**, not financial advice
- Past backtest performance does not guarantee future results
- In-sample results (2021-2024) should not be treated as performance claims

## Team

- **Zonglu Chen** — LLM Pipeline, ML Scoring & Backtesting
- **Jorge Garzon** — Technical Analysis & Data Pipeline
- **Alexander Pena** — System Integration & Documentation
- **Advisor**: Dr. Mingzhe Chen

*ECE 481/482 Senior Design — University of Miami, 2025-2026*

# LLM-Based Stock Sentiment Analysis

**ECE 482 Senior Design Project** — University of Miami

This tool analyzes recent financial news about tech stocks using a dual-model LLM ensemble (Claude 3.7 + GPT-5), combines it with technical analysis (EMA + RSI), and uses a trained Logistic Regression model to produce a data-driven stock score from 0 to 100.

---

## Quick Start

### Step 1: Clone the project

```bash
git clone https://github.com/cccccclu26/ECE-482.git
cd ECE-482/stock_sentiment
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Set up API keys

Create a `.env` file in the `stock_sentiment/` folder (copy from `.env.example`):

```
POLYGON_API_KEY=your_polygon_key_here
WAVESPEED_API_KEY=your_wavespeed_key_here
```

| Key | Where to get it | Purpose |
|-----|-----------------|---------|
| Polygon.io | [polygon.io](https://polygon.io/) — free tier | Fetch stock news & price data |
| WaveSpeed AI | [wavespeed.ai](https://wavespeed.ai/) | Run Claude 3.7 + GPT-5 inference |

### Step 4: Run it

```bash
# Standard rule-based mode
python main.py -t AAPL

# ML mode (uses trained Logistic Regression model)
python main.py -t AAPL --ml
```

---

## Usage Examples

```bash
# Analyze one stock
python main.py -t AAPL

# Analyze with ML scoring
python main.py -t NVDA --ml

# Analyze all 10 tech stocks
python main.py -a

# Run backtesting on AAPL (last 6 months)
python backtest.py -t AAPL

# Train the ML model (with sentiment features)
python ml_scorer.py --train --tickers AAPL NVDA MSFT --start 2023-01-01 --end 2025-12-01 --sentiment --eval

# Show trained model info & learned weights
python ml_scorer.py --info
```

### Command Options

| Command | Description |
|---------|-------------|
| `python main.py -t AAPL` | Analyze one stock |
| `python main.py -t AAPL --ml` | Analyze with ML-based scoring |
| `python main.py -a` | Analyze all 10 tech stocks |
| `python main.py -t AAPL -n 10` | Use 10 news articles |
| `python backtest.py -t AAPL` | Run historical backtesting |
| `python ml_scorer.py --train --sentiment` | Train ML model with sentiment |
| `python ml_scorer.py --info` | Show model coefficients |

---

## How It Works

```
Pick a stock (e.g., AAPL)
        |
        ├─── [1] Sentiment Analysis (LLM)
        │         Fetch 20 recent news articles (Polygon.io)
        │         Analyze each with Claude 3.7 + GPT-5 ensemble
        │         Output: sentiment score 0-100 per article
        │
        ├─── [2] Technical Analysis
        │         Fetch 6+ months of price data (yfinance)
        │         Calculate EMA25 / EMA50 / EMA100 trend signals
        │         Calculate RSI momentum signal
        │         Output: technical score 0-100
        │
        └─── [3] Scoring
                  Standard mode: 60% sentiment + 40% technical (rule-based)
                  ML mode: Logistic Regression trained on historical data
                  Output: final score 0-100 + grade
```

### Scoring Modes

**Standard mode** (`python main.py -t AAPL`):
- Sentiment score × 60% + Technical score × 40%
- Weights are manually defined

**ML mode** (`python main.py -t AAPL --ml`):
- Logistic Regression trained on 3 years of historical data
- 12 features: 9 technical (EMA trends, RSI, price ratios, volume) + 3 sentiment (score, confidence, article count)
- Output is probability of price going up over the next 5 trading days
- Replaces hardcoded weights with data-driven learned coefficients

---

## Understanding the Output

```
============================================================
  AAPL  |  FINAL SCORE: 20.01 / 100  |  SELL
============================================================

  Scoring Method: ML Logistic Regression
  Sentiment Input:  57.3
  Technical Input:  47.5
  P(Up): 20.01%  P(Down): 79.99%

  --- Sentiment (20 articles) ---
  Bullish: 5  Neutral: 14  Bearish: 1
  Avg Confidence: 80.4%

  --- Technical (EMA + RSI) ---
  EMA Score: 75.0  (raw: 2/4)
  EMA100 Uptrend: True
  EMA50 > EMA100: True
  EMA25 > EMA100: False
  RSI: 37.74  (bearish_momentum)

  --- Price ---
  Current: $254.28  EMA25: $260.99  EMA50: $262.92  EMA100: $262.16
```

### Grade Scale

| Score | Grade | Meaning |
|-------|-------|---------|
| 80–100 | STRONG BUY | Strong bullish signal |
| 65–79 | BUY | Bullish |
| 55–64 | SLIGHTLY BULLISH | Mild positive |
| 45–54 | NEUTRAL | No clear signal |
| 35–44 | SLIGHTLY BEARISH | Mild negative |
| 20–34 | SELL | Bearish |
| 0–19 | STRONG SELL | Strong bearish signal |

---

## ML Model Details

The Logistic Regression model (`ml_scorer.py`) is trained on historical price and news data:

- **Training data**: 3 years of daily OHLCV data + Polygon.io news for each ticker
- **Label**: Did the stock go up 5 trading days later? (binary classification)
- **Sentiment features**: Generated using VADER on historical Polygon.io news headlines (fast, free, reproducible for training — LLM scores used at prediction time)
- **Test accuracy**: ~60% on held-out data (AAPL 2023–2025)

To retrain the model:
```bash
python ml_scorer.py --train --tickers AAPL MSFT NVDA GOOGL --start 2023-01-01 --end 2025-12-01 --sentiment --eval
```

---

## Project Structure

```
stock_sentiment/
├── main.py                # Entry point — run analysis
├── ml_scorer.py           # ML model: train, evaluate, predict (Logistic Regression)
├── combined_scorer.py     # Rule-based scorer (60/40 weights)
├── backtest.py            # Historical backtesting module
├── sentiment_analyzer.py  # Dual-model LLM sentiment engine
├── technical_analysis.py  # EMA + RSI technical indicators
├── news_fetcher.py        # Polygon.io news fetching
├── config.py              # API keys, stock list, settings
├── requirements.txt       # Python dependencies
├── .env                   # Your API keys (never commit this!)
├── .env.example           # Template for .env
├── models/
│   └── lr_meta.json       # Trained model metadata & coefficients
└── results/               # Analysis output (JSON + CSV)
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `POLYGON_API_KEY is required` | Create `.env` file with your keys |
| `No trained model found` | Run `python ml_scorer.py --train` first |
| `429 Too Many Requests` | WaveSpeed API rate limit — reduce `-n` or wait a moment |
| Garbled text on Windows | Run with `python -X utf8 main.py ...` |

---

## Important Notes

- **Do NOT share your `.env` file** — it contains private API keys
- This is an **educational project**, not financial advice
- API calls may incur costs — monitor usage on Polygon.io and WaveSpeed AI

## Team

- Zonglu Chen — LLM Pipeline, ML Scoring & Backtesting
- Jorge Garzon — Technical Analysis & Backend
- Alexander Pena — Web Interface & User Experience
- Advisor: Dr. Mingzhe Chen

*ECE 481/482 Senior Design — University of Miami*

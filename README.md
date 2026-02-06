# LLM-Based Stock Sentiment Analysis

**ECE 482 Senior Design Project** — University of Miami

This tool reads recent financial news about tech stocks and uses AI to tell you whether the news is positive (bullish), negative (bearish), or neutral. It gives each stock a score from 0 to 100 so you can quickly see market sentiment at a glance.

---

## Quick Start (Get Running in 5 Minutes)

### Step 1: Download the project

Open a terminal (Command Prompt on Windows, Terminal on Mac) and run:

```bash
git clone https://github.com/cccccclu26/ECE-482.git
cd ECE-482/stock_sentiment
```

> **Don't have Git?** You can also click the green "Code" button on GitHub and select "Download ZIP", then unzip the folder.

### Step 2: Install Python packages

```bash
pip install -r requirements.txt
```

> **"pip not found"?** Try `python -m pip install -r requirements.txt` or `py -m pip install -r requirements.txt` on Windows.

### Step 3: Set up your API keys

You need two free API keys. Here's how to get them:

| Key | Where to get it | What it does |
|-----|-----------------|--------------|
| Polygon.io | [polygon.io](https://polygon.io/) - Sign up free | Fetches stock news |
| WaveSpeed AI | [wavespeed.ai](https://wavespeed.ai/) | Runs the AI analysis |

Once you have both keys, create a file called `.env` in the `stock_sentiment/` folder:

```
POLYGON_API_KEY=paste_your_polygon_key_here
WAVESPEED_API_KEY=paste_your_wavespeed_key_here
```

> **Tip:** There's already a file called `.env.example` in the folder. You can copy it, rename it to `.env`, and replace the placeholder text with your real keys.

### Step 4: Run it!

```bash
python main.py -t AAPL
```

That's it! You should see sentiment analysis results for Apple (AAPL) in your terminal.

---

## Usage Examples

```bash
# Analyze Apple stock
python main.py -t AAPL

# Analyze NVIDIA with 10 news articles
python main.py -t NVDA -n 10

# Analyze all 10 pre-configured tech stocks at once
python main.py -a

# Quick demo (analyzes AAPL, NVDA, MSFT)
python main.py
```

### Options

| Command | What it does |
|---------|--------------|
| `python main.py -t AAPL` | Analyze one stock (replace AAPL with any ticker) |
| `python main.py -a` | Analyze all 10 tech stocks |
| `python main.py -t AAPL -n 10` | Analyze with a specific number of articles |
| `python main.py` | Run a quick demo |

### Supported Stocks (default)

AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, AMD, INTC, CRM

You can add or remove stocks by editing the `TECH_STOCKS` list in `config.py`.

---

## Understanding the Output

When you run an analysis, you'll see something like this:

```
============================================================
Result: AAPL
============================================================
Final Score:    51.6 / 100
Sentiment:      NEUTRAL
Articles:       20
Lookback:       7 days
Avg Confidence: 68.2%
Bullish/Neutral/Bearish: 5/11/4
```

### What do the numbers mean?

| Score Range | Sentiment | Meaning |
|-------------|-----------|---------|
| 70 - 100 | BULLISH | News is mostly positive, stock may go up |
| 60 - 69 | Slightly Bullish | Leaning positive |
| 41 - 59 | NEUTRAL | Mixed or no strong signal |
| 31 - 40 | Slightly Bearish | Leaning negative |
| 0 - 30 | BEARISH | News is mostly negative, stock may go down |

### Where are results saved?

Every analysis is automatically saved to the `results/` folder:

- `results/AAPL_20260206_160659.json` — detailed results for a single stock
- `results/analysis_20260206_160659.json` — detailed results for batch analysis
- `results/summary_20260206_160659.csv` — summary table (open with Excel)

---

## How It Works (Simple Version)

```
You pick a stock (e.g., AAPL)
        |
        v
Fetch 20 recent news articles (last 7 days)
        |
        v
AI reads each article and scores it 0-100
        |
        v
Combine all scores into one final score
        |
        v
Show results + save to file
```

---

## Project Structure

```
stock_sentiment/
├── main.py                # Run this file to start
├── config.py              # Settings (change stock list, article count, etc.)
├── news_fetcher.py        # Gets news from Polygon.io
├── sentiment_analyzer.py  # AI sentiment analysis engine
├── requirements.txt       # Python packages needed
├── .env                   # Your API keys (keep this private!)
├── .env.example           # Template for .env
└── results/               # Analysis results saved here
```

## Settings You Can Change

Edit `config.py` to customize:

| Setting | Default | What it does |
|---------|---------|--------------|
| `DEFAULT_NEWS_LIMIT` | 20 | How many articles to analyze per stock |
| `NEWS_LOOKBACK_DAYS` | 7 | How far back to look for news (in days) |
| `MAX_CONCURRENT_LLM_CALLS` | 5 | How many articles to analyze at the same time (higher = faster but uses more API quota) |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `pip not found` | Use `python -m pip install -r requirements.txt` |
| `ModuleNotFoundError` | Make sure you ran `pip install -r requirements.txt` |
| `POLYGON_API_KEY is required` | Create the `.env` file with your API keys (see Step 3) |
| `WAVESPEED_API_KEY is required` | Same as above — make sure both keys are in `.env` |
| `No news found for XXXX` | The ticker might not have recent news, or check your Polygon API key |
| Garbled text on Windows | This is a known encoding issue — the analysis still works correctly |

---

## Important

- **Do NOT share your `.env` file** — it contains your private API keys
- API calls may cost money — check your usage on Polygon.io and WaveSpeed AI
- This is an **educational project**, not financial advice. Do not make investment decisions based solely on this tool.

## Team

- Zonglu Chen — LLM Pipeline & Evaluation
- Jorge Garzon — Backend & Data Infrastructure
- Alexander Pena — Web Interface & User Experience
- Advisor: Dr. Mingzhe Chen

*ECE 481/482 Senior Design — University of Miami*

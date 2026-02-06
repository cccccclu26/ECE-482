"""
Configuration file for stock sentiment analysis project
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
WAVESPEED_API_KEY = os.getenv("WAVESPEED_API_KEY")

# Polygon API config
POLYGON_BASE_URL = "https://api.polygon.io"

# News fetching config
DEFAULT_NEWS_LIMIT = 10  # Default number of news articles per stock
NEWS_LOOKBACK_DAYS = 3   # Fetch news from the past N days

# WaveSpeed LLM API config
WAVESPEED_API_URL = "https://api.wavespeed.ai/api/v3/wavespeed-ai/any-llm"
LLM_MODEL = "anthropic/claude-3.7-sonnet"

# Target stock list (tech sector)
TECH_STOCKS = [
    "AAPL",   # Apple
    "MSFT",   # Microsoft
    "GOOGL",  # Alphabet
    "AMZN",   # Amazon
    "NVDA",   # NVIDIA
    "META",   # Meta
    "TSLA",   # Tesla
    "AMD",    # AMD
    "INTC",   # Intel
    "CRM",    # Salesforce
]

# Sentiment analysis prompt template
SENTIMENT_PROMPT = """You are a professional financial analyst. Analyze the following news about {ticker} stock and provide a sentiment score.

News Title: {title}
News Summary: {description}
Published: {published_date}

Return ONLY valid JSON in this exact format (no other text, no markdown):
{{"sentiment": "bullish", "score": 75, "confidence": 80, "reason": "brief explanation"}}

Rules:
- sentiment: must be "bullish", "neutral", or "bearish"
- score: integer 0-100 (50=neutral, 100=extremely bullish, 0=extremely bearish)
- confidence: integer 0-100 (your confidence level)
- reason: brief explanation in under 30 words
- Return ONLY the JSON object, nothing else
"""

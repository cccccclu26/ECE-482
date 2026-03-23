"""
Technical Analysis Module - EMA and RSI based stock scoring
Based on original work by Jorge Garzon, refactored for integration.

Outputs a 0-100 technical score:
  - EMA indicators: 50% of technical score
  - RSI indicators: 50% of technical score
"""
import yfinance as yf
import pandas as pd
from typing import Dict


class TechnicalAnalyzer:
    """Analyze stock price data using technical indicators (EMA + RSI)"""

    def __init__(self, period: str = "6mo"):
        """
        Args:
            period: yfinance history period (must be >= 6mo for EMA100)
        """
        self.period = period

    def fetch_data(self, ticker: str) -> pd.DataFrame:
        """
        Fetch historical price data for a ticker.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")

        Returns:
            DataFrame with OHLCV data
        """
        stock = yf.Ticker(ticker)
        data = stock.history(period=self.period)

        if data.empty:
            raise ValueError(f"No price data found for {ticker}")

        return data

    # ==================== EMA ====================

    def calculate_ema(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate EMA 100, 50, 25 on Close prices"""
        data = data.copy()
        data["EMA100"] = data["Close"].ewm(span=100, adjust=False).mean().round(3)
        data["EMA50"] = data["Close"].ewm(span=50, adjust=False).mean().round(3)
        data["EMA25"] = data["Close"].ewm(span=25, adjust=False).mean().round(3)
        return data

    def _is_ema100_uptrend(self, data: pd.DataFrame, period: int) -> bool:
        """Check if EMA100 is in an uptrend over the past N days"""
        ema100 = data["EMA100"]
        return bool(ema100.iloc[-1] > ema100.iloc[-period:].min())

    def _is_ema50_above_ema100(self, data: pd.DataFrame, period: int) -> bool:
        """Check if EMA50 has been above EMA100 for the past N days"""
        return bool(all(data["EMA50"].iloc[-period:] > data["EMA100"].iloc[-period:]))

    def _is_ema25_above_ema100(self, data: pd.DataFrame, period: int) -> bool:
        """Check if EMA25 has been above EMA100 for the past N days"""
        return bool(all(data["EMA25"].iloc[-period:] > data["EMA100"].iloc[-period:]))

    def calculate_ema_points(self, data: pd.DataFrame, period: int = 30) -> Dict:
        """
        Calculate EMA-based points using original scoring logic.
        Raw points range: -4 to +4

        Returns:
            Dict with raw_points, details, and normalized score (0-100)
        """
        points = 0
        trend = False

        ema100_uptrend = self._is_ema100_uptrend(data, period)
        ema50_above = self._is_ema50_above_ema100(data, period)
        ema25_above = self._is_ema25_above_ema100(data, period)

        # EMA100 uptrend: +1 / -1
        if ema100_uptrend:
            points += 1
            trend = True
        else:
            points -= 1

        # EMA50 above EMA100: +1 / -1 (only penalize if no uptrend)
        if ema50_above:
            points += 1
        elif not trend:
            points -= 1

        # EMA25 above EMA100: +2 / -2 (double weight, penalize if no uptrend)
        if ema25_above:
            points += 2
        elif not trend:
            points -= 2

        # Normalize: -4..+4 -> 0..100
        normalized = round((points + 4) / 8 * 100, 2)

        return {
            "raw_points": points,
            "normalized_score": normalized,
            "ema100_uptrend": ema100_uptrend,
            "ema50_above_ema100": ema50_above,
            "ema25_above_ema100": ema25_above,
        }

    # ==================== RSI ====================

    def calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate RSI (Relative Strength Index).

        Args:
            data: DataFrame with Close prices
            period: RSI period (default 14 days)

        Returns:
            DataFrame with RSI column added
        """
        data = data.copy()
        delta = data["Close"].diff()

        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()

        rs = avg_gain / avg_loss
        data["RSI"] = (100 - (100 / (1 + rs))).round(2)

        return data

    def calculate_rsi_points(self, data: pd.DataFrame) -> Dict:
        """
        Calculate RSI-based score.

        Scoring logic:
          RSI > 70: overbought, slightly bearish signal      -> 30 points
          RSI 60-70: bullish momentum                         -> 80 points
          RSI 50-60: mildly bullish                           -> 65 points
          RSI 40-50: mildly bearish                           -> 35 points
          RSI 30-40: bearish momentum                         -> 20 points
          RSI < 30: oversold, potential reversal (bullish)    -> 70 points

        Returns:
            Dict with rsi_value, normalized_score, signal
        """
        current_rsi = data["RSI"].iloc[-1]

        if current_rsi > 70:
            score = 30
            signal = "overbought"
        elif current_rsi > 60:
            score = 80
            signal = "bullish_momentum"
        elif current_rsi > 50:
            score = 65
            signal = "mildly_bullish"
        elif current_rsi > 40:
            score = 35
            signal = "mildly_bearish"
        elif current_rsi > 30:
            score = 20
            signal = "bearish_momentum"
        else:
            score = 70
            signal = "oversold_reversal"

        return {
            "rsi_value": round(float(current_rsi), 2),
            "normalized_score": score,
            "signal": signal,
        }

    # ==================== Combined ====================

    def analyze(self, ticker: str, ema_period: int = 30) -> Dict:
        """
        Run full technical analysis on a ticker.

        Technical score = EMA score (50%) + RSI score (50%)
        Output range: 0-100

        Args:
            ticker: Stock ticker symbol
            ema_period: Lookback period for EMA conditions (days)

        Returns:
            Dict with technical_score, ema details, rsi details, price info
        """
        print(f"  Fetching price data for {ticker}...")
        data = self.fetch_data(ticker)

        # Calculate indicators
        data = self.calculate_ema(data)
        data = self.calculate_rsi(data)

        # Score each component
        ema_result = self.calculate_ema_points(data, ema_period)
        rsi_result = self.calculate_rsi_points(data)

        # Combined: 50% EMA + 50% RSI
        technical_score = round(
            ema_result["normalized_score"] * 0.5 + rsi_result["normalized_score"] * 0.5,
            2
        )

        # Determine label
        if technical_score >= 60:
            signal = "bullish"
        elif technical_score <= 40:
            signal = "bearish"
        else:
            signal = "neutral"

        # Current price info
        latest = data.iloc[-1]

        # Volume ratio: today's volume / 20-day average
        vol_ratio = 1.0
        if "Volume" in data.columns and len(data) >= 20:
            avg_vol = data["Volume"].iloc[-20:].mean()
            if avg_vol > 0:
                vol_ratio = round(float(latest["Volume"] / avg_vol), 4)

        return {
            "ticker": ticker,
            "technical_score": technical_score,
            "signal": signal,
            "ema": {
                "score": ema_result["normalized_score"],
                "raw_points": ema_result["raw_points"],
                "ema100_uptrend": ema_result["ema100_uptrend"],
                "ema50_above_ema100": ema_result["ema50_above_ema100"],
                "ema25_above_ema100": ema_result["ema25_above_ema100"],
            },
            "rsi": {
                "score": rsi_result["normalized_score"],
                "value": rsi_result["rsi_value"],
                "signal": rsi_result["signal"],
            },
            "price": {
                "current": round(float(latest["Close"]), 2),
                "ema25": round(float(latest["EMA25"]), 2),
                "ema50": round(float(latest["EMA50"]), 2),
                "ema100": round(float(latest["EMA100"]), 2),
            },
            "volume_ratio": vol_ratio,
        }


# Test code
if __name__ == "__main__":
    analyzer = TechnicalAnalyzer()

    print("=" * 60)
    print("Technical Analysis Test")
    print("=" * 60)

    result = analyzer.analyze("AAPL")

    print(f"\nTicker: {result['ticker']}")
    print(f"Technical Score: {result['technical_score']} / 100")
    print(f"Signal: {result['signal'].upper()}")
    print(f"\n--- EMA ---")
    print(f"  EMA Score: {result['ema']['score']}")
    print(f"  Raw Points: {result['ema']['raw_points']} (range: -4 to +4)")
    print(f"  EMA100 Uptrend: {result['ema']['ema100_uptrend']}")
    print(f"  EMA50 > EMA100: {result['ema']['ema50_above_ema100']}")
    print(f"  EMA25 > EMA100: {result['ema']['ema25_above_ema100']}")
    print(f"\n--- RSI ---")
    print(f"  RSI Score: {result['rsi']['score']}")
    print(f"  RSI Value: {result['rsi']['value']}")
    print(f"  RSI Signal: {result['rsi']['signal']}")
    print(f"\n--- Price ---")
    print(f"  Current: ${result['price']['current']}")
    print(f"  EMA25:   ${result['price']['ema25']}")
    print(f"  EMA50:   ${result['price']['ema50']}")
    print(f"  EMA100:  ${result['price']['ema100']}")

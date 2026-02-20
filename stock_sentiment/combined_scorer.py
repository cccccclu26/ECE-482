"""
Combined Scorer - Merges sentiment analysis and technical analysis into a final score.

Final Score (0-100) = Sentiment Score × 60% + Technical Score × 40%
"""
from typing import Dict


SENTIMENT_WEIGHT = 0.60
TECHNICAL_WEIGHT = 0.40


def combine_scores(sentiment_score: float, technical_score: float) -> Dict:
    """
    Combine sentiment and technical scores into a final score.

    Args:
        sentiment_score: 0-100 from LLM news analysis
        technical_score: 0-100 from EMA + RSI analysis

    Returns:
        Dict with final_score, grade, and breakdown
    """
    sentiment_contribution = round(sentiment_score * SENTIMENT_WEIGHT, 2)
    technical_contribution = round(technical_score * TECHNICAL_WEIGHT, 2)
    final_score = round(sentiment_contribution + technical_contribution, 2)

    # Clamp to 0-100
    final_score = max(0, min(100, final_score))

    # Grade label
    if final_score >= 80:
        grade = "STRONG BUY"
    elif final_score >= 65:
        grade = "BUY"
    elif final_score >= 55:
        grade = "SLIGHTLY BULLISH"
    elif final_score >= 45:
        grade = "NEUTRAL"
    elif final_score >= 35:
        grade = "SLIGHTLY BEARISH"
    elif final_score >= 20:
        grade = "SELL"
    else:
        grade = "STRONG SELL"

    return {
        "final_score": final_score,
        "grade": grade,
        "breakdown": {
            "sentiment_score": round(sentiment_score, 2),
            "sentiment_weight": SENTIMENT_WEIGHT,
            "sentiment_contribution": sentiment_contribution,
            "technical_score": round(technical_score, 2),
            "technical_weight": TECHNICAL_WEIGHT,
            "technical_contribution": technical_contribution,
        }
    }

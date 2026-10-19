"""Rule-based review parser for structured ecommerce comment analysis.

This optional module converts raw review text into stable JSON-like structures
for sentiment, topic tags, and positive/negative points. It does not call an
LLM, modify Agent flow, connect to databases, or depend on product tools.
"""

from __future__ import annotations

from typing import Any


TAG_KEYWORDS: dict[str, tuple[str, ...]] = {
    "物流": ("物流", "发货", "配送", "快递", "到货"),
    "包装": ("包装", "包裹", "破损", "盒子"),
    "价格": ("价格", "价钱", "贵", "便宜", "划算"),
    "产品体验": ("体验", "使用", "好用", "效果", "口感", "产品"),
    "服务": ("客服", "服务", "回复", "售后"),
}

POSITIVE_KEYWORDS = (
    "很快",
    "快",
    "很好",
    "不错",
    "满意",
    "好用",
    "划算",
    "便宜",
)

NEGATIVE_KEYWORDS = (
    "贵",
    "太贵",
    "慢",
    "不好",
    "差",
    "破损",
    "失望",
    "客服回复慢",
)


def parse_reviews(reviews: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Parse raw ecommerce reviews into structured JSON-compatible results."""
    if not reviews:
        return []

    return [_parse_single_review(review) for review in reviews if isinstance(review, dict)]


def _parse_single_review(review: dict[str, Any]) -> dict[str, Any]:
    """Parse one review record into the standard output schema."""
    review_id = review.get("review_id")
    text = str(review.get("text") or "").strip()

    return {
        "review_id": review_id,
        "sentiment": _classify_sentiment(text),
        "tags": _extract_tags(text),
        "analysis": {
            "positive_points": _extract_points(text, POSITIVE_KEYWORDS),
            "negative_points": _extract_points(text, NEGATIVE_KEYWORDS),
        },
    }


def _classify_sentiment(text: str) -> str:
    """Classify sentiment as positive, negative, or neutral."""
    if not text:
        return "neutral"

    positive_score = sum(1 for keyword in POSITIVE_KEYWORDS if keyword in text)
    negative_score = sum(1 for keyword in NEGATIVE_KEYWORDS if keyword in text)

    if positive_score > negative_score:
        return "positive"
    if negative_score > positive_score:
        return "negative"
    return "neutral"


def _extract_tags(text: str) -> list[str]:
    """Extract ecommerce topic tags from review text."""
    if not text:
        return []

    return [
        tag
        for tag, keywords in TAG_KEYWORDS.items()
        if any(keyword in text for keyword in keywords)
    ]


def _extract_points(text: str, keywords: tuple[str, ...]) -> list[str]:
    """Extract matched keywords as lightweight analysis points."""
    if not text:
        return []

    return [keyword for keyword in keywords if keyword in text]


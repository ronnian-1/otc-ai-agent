"""Tests for the optional ecommerce review parser module."""

from app.analysis.review_parser import parse_reviews


def test_positive_review_extracts_sentiment_and_tags() -> None:
    """Positive review should include sentiment and logistics/packaging tags."""
    result = parse_reviews(
        [{"review_id": 1, "text": "物流很快，包装很好，使用体验不错"}]
    )

    assert result[0]["sentiment"] == "positive"
    assert "物流" in result[0]["tags"]
    assert "包装" in result[0]["tags"]


def test_negative_review_extracts_price_and_service_tags() -> None:
    """Negative review should identify price and service concerns."""
    result = parse_reviews([{"review_id": 2, "text": "价格太贵，客服回复慢"}])

    assert result[0]["sentiment"] == "negative"
    assert "价格" in result[0]["tags"]
    assert "服务" in result[0]["tags"]


def test_batch_reviews_keep_result_length() -> None:
    """Batch parsing should return one result per valid review dict."""
    result = parse_reviews(
        [
            {"review_id": 1, "text": "物流很快"},
            {"review_id": 2, "text": "价格太贵"},
        ]
    )

    assert len(result) == 2


def test_empty_input_and_empty_text_do_not_crash() -> None:
    """Empty review lists and empty review text should be handled safely."""
    assert parse_reviews([]) == []

    result = parse_reviews([{"review_id": 3, "text": ""}])

    assert result[0]["sentiment"] == "neutral"
    assert result[0]["tags"] == []
    assert result[0]["analysis"]["positive_points"] == []
    assert result[0]["analysis"]["negative_points"] == []


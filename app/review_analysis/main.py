"""Standalone entry point placeholder for review analysis experiments."""

from app.review_analysis.analyzer import analyze_reviews


def main() -> None:
    """Run a minimal review analysis smoke check."""
    result = analyze_reviews("S001")
    print(result)


if __name__ == "__main__":
    main()


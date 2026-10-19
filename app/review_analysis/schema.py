"""Pydantic schemas for review analysis outputs.

The schema is intentionally minimal at this stage and will be expanded when
review analysis logic is implemented.
"""

from pydantic import BaseModel, Field


class ReviewSignal(BaseModel):
    """Structured signal extracted from a product review."""

    label: str = Field(description="Signal label, such as sentiment or risk type.")
    evidence: str = Field(description="Short evidence text from the source review.")


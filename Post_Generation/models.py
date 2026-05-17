from pydantic import BaseModel, Field, computed_field
from typing import List, Optional


class SynthesisOutput(BaseModel):
    """
    Structured output for the Synthesizer agent.
    Mirrors SearchNodeOutput from the original chains.py.
    """
    main_topic: str
    key_insights: List[str]
    technical_concepts: List[str]
    real_world_applications: Optional[List[str]] = None
    challenges_tradeoffs: Optional[List[str]] = None
    emerging_trends: Optional[List[str]] = None


class ReflectionOutput(BaseModel):
    """
    Structured output for the Reflector agent.
    Mirrors ReflectionOutput from the original chains.py.

    quality_score drives the should_continue logic:
        >= 70  → stop (approved)
        < 70   → loop back to generator
    """
    quality_score: int = Field(
        description=(
            "Overall quality score from 0 to 100. "
            "0–59: poor, needs major rework. "
            "60–79: acceptable but has clear gaps. "
            "80–100: strong, publish-ready."
        )
    )
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]

    @computed_field
    @property
    def approved(self) -> bool:
        """True when quality_score >= 80 — mirrors the original @property."""
        return self.quality_score >= 80

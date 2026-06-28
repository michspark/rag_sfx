"""Pydantic v2 output schemas for the three planners.

These models serve two jobs at once:
  1. They validate the LLM's JSON output (reject malformed or hallucinated keys).
  2. Their `model_json_schema()` IS the JSON schema a future guided-decoding backend
     (vLLM `guided_json`, Outlines, XGrammar) will constrain generation with.

Design rule that enforces "LLM plans, DSP places": there is NO float field anywhere.
Every temporal reference is an integer segment index (or list of them), so it is
structurally impossible to emit an absolute time in seconds.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _Strict(BaseModel):
    """Base for all output models: forbid unknown keys.

    `extra="forbid"` makes validation fail if the model invents a key like
    `start_time` or `timestamp` — a second layer of defense for the no-seconds rule.
    """

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------
# Keynote — scene-spanning ambience beds
# --------------------------------------------------------------------------
class KeynoteItem(_Strict):
    query: str = Field(
        ...,
        min_length=1,
        description="CLAP-ready ambience query, e.g. 'quiet outdoor winter wind, light snow, residential'",
    )
    environment_descriptors: list[str] = Field(
        default_factory=list,
        description="Corroborated setting facts, e.g. ['outdoor', 'winter', 'driveway', 'residential']",
    )
    start_segment: int = Field(..., ge=0, description="Inclusive first segment index the bed covers")
    end_segment: int = Field(..., ge=0, description="Inclusive last segment index the bed covers")
    rationale: str = Field(..., min_length=1, description="Why this bed, and which noisy labels were rejected")

    @model_validator(mode="after")
    def _check_span(self) -> "KeynoteItem":
        # A bed's end cannot come before its start.
        if self.end_segment < self.start_segment:
            raise ValueError("end_segment must be >= start_segment")
        return self


class KeynoteResult(_Strict):
    # Expect 1-3 beds for a single clip.
    beds: list[KeynoteItem] = Field(..., min_length=1, max_length=3)


# --------------------------------------------------------------------------
# Signal — foreground one-shot events tied to on-screen actions
# --------------------------------------------------------------------------
class SignalItem(_Strict):
    event_ref: str = Field(..., min_length=1, description="Short action label, e.g. 'shovel_scrape'")
    query: str = Field(..., min_length=1, description="CLAP-ready one-shot query for the sound")
    material_hint: str | None = Field(
        default=None,
        description="Surface/material interaction when it matters, e.g. 'metal blade on concrete'",
    )
    segment_hints: list[int] = Field(
        ...,
        min_length=1,
        description="All segment indices where this action occurs (anchors downstream onset detection)",
    )
    rationale: str = Field(..., min_length=1)

    @field_validator("segment_hints")
    @classmethod
    def _clean_hints(cls, value: list[int]) -> list[int]:
        # Reject negatives, then dedupe and sort so repeated instances merge cleanly.
        if any(index < 0 for index in value):
            raise ValueError("segment_hints must be >= 0")
        return sorted(set(value))


class SignalResult(_Strict):
    # May be empty in principle, though a normal action clip yields several.
    signals: list[SignalItem] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Archetypal — convention-based, non-literal designed sounds
# --------------------------------------------------------------------------
class ArchetypalItem(_Strict):
    query: str = Field(..., min_length=1)
    motivation: str = Field(..., min_length=1, description="Narrative/emotional justification")
    generative_fallback: bool = Field(
        ...,
        description="True if unlikely to exist in a real SFX library (must be synthesized)",
    )
    rationale: str = Field(..., min_length=1)


class ArchetypalResult(_Strict):
    # An EMPTY list is valid and often correct for mundane clips.
    cues: list[ArchetypalItem] = Field(default_factory=list, max_length=3)


# --------------------------------------------------------------------------
# Convenience: expose the JSON schema for guided-decoding backends
# --------------------------------------------------------------------------
def json_schema(model_cls: type[BaseModel]) -> dict:
    """Return the JSON schema dict for a result model (vLLM guided_json / Outlines)."""
    return model_cls.model_json_schema()

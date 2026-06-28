"""Parse the upstream perception record (preprocess_output.txt) into structured segments.

The record is a plain-text file produced by `preprocessing/preprocess.py`. Each ~2s
segment carries three independent perception signals: a Qwen3-VL caption, RAM++ tags,
and Places365 scene scores. This module turns that text into typed `Segment` objects
and renders a compact, *seconds-free* evidence block for the LLM planner prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# --------------------------------------------------------------------------
# Data containers
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Segment:
    """One ~2s segment of perception evidence."""

    index: int
    t_start: float                       # coarse reference ONLY — never put into prompts/output
    t_end: float                         # coarse reference ONLY
    vlm_caption: str                     # the [Qwen3-VL] line(s)
    ram_tags: list[str]                  # the [RAM++ tags] line, split on "|"
    places_indoor_outdoor: str           # "indoor" | "outdoor"
    places_scores: dict[str, float]      # ordered label -> probability


@dataclass(frozen=True)
class PerceptionRecord:
    """The whole parsed record for one video clip."""

    video_path: str
    duration_s: float
    segments: list[Segment]


# --------------------------------------------------------------------------
# Compiled patterns (tolerant of the format's double-spaces and minor drift)
# --------------------------------------------------------------------------
# "VIDEO: /path/to/file.mp4"
_VIDEO_LINE = re.compile(r"^VIDEO:\s*(.+)$")
# "DURATION: 10.04s  |  SEGMENTS: 5 x 2.0s"
_DURATION_LINE = re.compile(r"^DURATION:\s*([\d.]+)s")
# "========== SEGMENT 0  [0.0s ~ 2.0s] =========="
_SEGMENT_HEADER = re.compile(
    r"^=+\s*SEGMENT\s+(\d+)\s*\[\s*([\d.]+)s\s*~\s*([\d.]+)s\s*\]\s*=+$"
)
# "[Qwen3-VL] A person ..."
_VLM_LINE = re.compile(r"^\[Qwen3-VL\]\s*(.*)$")
# "[RAM++ tags] blanket | car | ..."
_RAM_LINE = re.compile(r"^\[RAM\+\+ tags\]\s*(.*)$")
# "[Places365] outdoor | driveway (0.449), ice_floe (0.188), ..."
_PLACES_LINE = re.compile(r"^\[Places365\]\s*(\w+)\s*\|\s*(.*)$")
# A single "driveway (0.449)" item inside the Places365 list
_PLACES_ITEM = re.compile(r"([A-Za-z0-9_./-]+)\s*\(([\d.]+)\)")


# --------------------------------------------------------------------------
# Internal builder — accumulates one segment's fields while scanning lines
# --------------------------------------------------------------------------
class _SegmentBuilder:
    """Mutable scratch object; frozen into a `Segment` once fully read."""

    def __init__(self, index: int, t_start: float, t_end: float) -> None:
        self.index = index
        self.t_start = t_start
        self.t_end = t_end
        self.caption_parts: list[str] = []   # caption can span multiple physical lines
        self.ram_tags: list[str] = []
        self.places_indoor_outdoor: str = ""
        self.places_scores: dict[str, float] = {}

    def finish(self) -> Segment:
        # Join any wrapped caption lines into one clean string.
        caption = " ".join(part.strip() for part in self.caption_parts).strip()
        if not caption:
            raise ValueError(f"Segment {self.index} has no [Qwen3-VL] caption")
        return Segment(
            index=self.index,
            t_start=self.t_start,
            t_end=self.t_end,
            vlm_caption=caption,
            ram_tags=self.ram_tags,
            places_indoor_outdoor=self.places_indoor_outdoor,
            places_scores=self.places_scores,
        )


def _parse_places(body: str) -> dict[str, float]:
    """Turn 'driveway (0.449), ice_floe (0.188)' into an ordered {label: prob} dict."""
    scores: dict[str, float] = {}
    for label, prob_text in _PLACES_ITEM.findall(body):
        scores[label] = float(prob_text)
    return scores


# --------------------------------------------------------------------------
# Public parsing API
# --------------------------------------------------------------------------
def parse_record(text: str) -> PerceptionRecord:
    """Parse the full perception record text into a `PerceptionRecord`.

    Uses a line-state machine so a [Qwen3-VL] caption may wrap across several
    physical lines: any non-marker line after the caption marker is treated as a
    continuation until the next [RAM++ tags] / [Places365] / segment header.
    """
    video_path: str = ""
    duration_s: float = 0.0
    builders: list[_SegmentBuilder] = []
    current: Optional[_SegmentBuilder] = None
    in_caption: bool = False   # True right after a [Qwen3-VL] marker, until another marker

    for raw_line in text.splitlines():
        line = raw_line.rstrip()

        # --- file-level header lines ---
        video_match = _VIDEO_LINE.match(line)
        if video_match:
            video_path = video_match.group(1).strip()
            continue
        duration_match = _DURATION_LINE.match(line)
        if duration_match:
            duration_s = float(duration_match.group(1))
            continue

        # --- new segment starts: flush the previous builder ---
        header_match = _SEGMENT_HEADER.match(line)
        if header_match:
            if current is not None:
                builders.append(current)
            current = _SegmentBuilder(
                index=int(header_match.group(1)),
                t_start=float(header_match.group(2)),
                t_end=float(header_match.group(3)),
            )
            in_caption = False
            continue

        # Lines below only make sense inside a segment.
        if current is None:
            continue

        # --- caption marker ---
        vlm_match = _VLM_LINE.match(line)
        if vlm_match:
            current.caption_parts.append(vlm_match.group(1))
            in_caption = True
            continue

        # --- RAM++ tags marker ---
        ram_match = _RAM_LINE.match(line)
        if ram_match:
            tags = [tag.strip() for tag in ram_match.group(1).split("|")]
            current.ram_tags = [tag for tag in tags if tag]   # drop empties
            in_caption = False
            continue

        # --- Places365 marker ---
        places_match = _PLACES_LINE.match(line)
        if places_match:
            current.places_indoor_outdoor = places_match.group(1).strip()
            current.places_scores = _parse_places(places_match.group(2))
            in_caption = False
            continue

        # --- caption continuation (wrapped line) ---
        if in_caption and line.strip():
            current.caption_parts.append(line)

    # Flush the final segment at EOF.
    if current is not None:
        builders.append(current)

    # Loud failure on format drift: no segments means our patterns went stale.
    if not builders:
        raise ValueError("No segments parsed — perception record format may have changed")

    segments = [builder.finish() for builder in builders]
    return PerceptionRecord(
        video_path=video_path,
        duration_s=duration_s,
        segments=segments,
    )


def parse_record_file(path: str) -> PerceptionRecord:
    """Read a perception record file from disk and parse it."""
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    return parse_record(text)


# --------------------------------------------------------------------------
# Prompt rendering — compact, stoplist-filtered, and SECONDS-FREE
# --------------------------------------------------------------------------
def segments_to_prompt_block(
    segments: list[Segment],
    *,
    stoplist: Optional[set[str]] = None,
    max_tags: int = 12,
    places_min_score: float = 0.05,
) -> str:
    """Render segments as evidence text for the planner prompt.

    Deliberately emits ONLY the integer segment index — never t_start/t_end — so the
    "LLM plans, DSP places" boundary holds at the prompt level. The optional stoplist
    removes known-noise labels from both RAM++ tags and Places365 scores before the
    model ever sees them.
    """
    stop = stoplist or set()
    blocks: list[str] = []

    for segment in segments:
        # Filter and cap RAM++ tags.
        kept_tags = [tag for tag in segment.ram_tags if tag not in stop][:max_tags]

        # Filter Places365 by stoplist and a minimum-confidence cut.
        kept_scenes = [
            f"{label} {score:.2f}"
            for label, score in segment.places_scores.items()
            if label not in stop and score >= places_min_score
        ]

        lines = [
            f"SEG {segment.index}:",
            f"  caption: {segment.vlm_caption}",
            f"  tags: {' | '.join(kept_tags)}",
            f"  scene: {segment.places_indoor_outdoor} ({', '.join(kept_scenes)})",
        ]
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)

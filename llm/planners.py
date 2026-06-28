"""LLM planning stage: perception record -> typed retrieval queries.

One shared Qwen3-14B is loaded once and injected into three planners that differ only
in their role prompt and output schema:
  - KeynotePlanner    -> ambience beds         (KeynoteResult)
  - SignalPlanner     -> foreground events     (SignalResult)
  - ArchetypalPlanner -> designed/non-literal  (ArchetypalResult)

Architectural rule: the LLM plans (what / why / which segment); downstream DSP places
(frame-precise timing). No planner output may contain an absolute time in seconds.
"""

from __future__ import annotations

import abc
import json
import re
from dataclasses import dataclass
from typing import Optional, Type

import torch
from pydantic import BaseModel, ValidationError
from transformers import AutoModelForCausalLM, AutoTokenizer

from parser import PerceptionRecord, Segment, parse_record_file, segments_to_prompt_block
from schemas import ArchetypalResult, KeynoteResult, SignalResult


# --------------------------------------------------------------------------
# Model + decoding constants
# --------------------------------------------------------------------------
DEFAULT_MODEL_ID: str = "Qwen/Qwen3-14B"
THINK_CLOSE_TOKEN_ID: int = 151668   # the "</think>" special token id for Qwen3

# Per-run noise pre-filter. The captions describe a paved residential winter scene, so
# icy-wilderness scene labels and a few RAM++ false positives are dropped before the LLM.
# The system prompt is the BACKSTOP for whatever residual noise slips through this list.
DEFAULT_STOPLIST: set[str] = {
    "ice_floe",
    "glacier",
    "iceberg",
    "ice_shelf",
    "igloo",
    "blanket",
    "snowstorm",
}


# --------------------------------------------------------------------------
# Shared model loader — load ONCE, inject into all three planners
# --------------------------------------------------------------------------
def load_planner_model(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    load_in_4bit: bool = True,
    device_map: str = "auto",
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load one Qwen3-14B (+ tokenizer) to be shared across all planners.

    bf16 14B is ~28GB and will not co-reside with the vision stack on a 24GB GPU, so
    `load_in_4bit=True` (nf4) is the default. Pass `load_in_4bit=False` for the bf16
    quality reference when memory allows.
    """
    quantization_config = None
    torch_dtype: Optional[torch.dtype] = torch.bfloat16

    if load_in_4bit:
        # Guarded import: bitsandbytes is an optional heavy dependency.
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as error:
            raise RuntimeError(
                "load_in_4bit=True needs bitsandbytes — `pip install bitsandbytes`"
            ) from error
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",                 # normal-float-4: better than plain int4
            bnb_4bit_compute_dtype=torch.bfloat16,     # matmuls run in bf16, weights stored in 4bit
            bnb_4bit_use_double_quant=True,            # quantize the quant constants too (extra saving)
        )
        torch_dtype = None   # the quantization config governs dtype now

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
        device_map=device_map,
    )
    model.eval()   # inference mode: disables dropout, etc.
    return model, tokenizer


# --------------------------------------------------------------------------
# Prompts: one shared evidence-handling system prompt + three role/task prompts
# --------------------------------------------------------------------------
SHARED_SYSTEM_PROMPT: str = """\
You are a sound-design planning assistant. You convert noisy multi-model video
perception into precise audio retrieval plans.

For each ~2 second segment you receive THREE independent perception sources:
  - caption: a vision-language description (most reliable, but can hallucinate detail),
  - tags: recognition tags (useful, but include false positives),
  - scene: scene-classifier guesses with confidence scores (often wrong on close-up or
    motion-heavy footage).

EVIDENCE RULES (follow strictly):
1. Trust facts CORROBORATED across MULTIPLE segments AND MULTIPLE sources. A detail the
   caption confirms in several segments is reliable.
2. REJECT isolated single-source labels that the captions contradict. A high confidence
   score is NOT correctness. Treat scene scores as weak hints only.
3. Scene labels describing a completely different environment than the captions (for
   example icy-wilderness labels like ice floe, glacier, iceberg, igloo when every
   caption describes a person clearing a paved residential surface) are NOISE. Do not let
   them influence the environment you infer.
4. Reference time ONLY by integer segment index. NEVER output seconds, timestamps, or any
   number followed by 's'. Temporal fields are segment indices only.
5. Do not invent events, drama, or atmosphere the evidence does not support. Under-planning
   is better than fabricating.

OUTPUT RULES:
- Output a SINGLE valid JSON object conforming exactly to the provided schema.
- No markdown, no code fences, no commentary, no extra keys. JSON only.
"""

KEYNOTE_TASK_PROMPT: str = """\
ROLE: Keynote (ambience bed) planner.

A keynote is the continuous background ambience that defines the scene's acoustic
environment (the "bed"). Produce 1 to 3 beds describing the corroborated environment.

Infer the environment ONLY from facts corroborated across segments and sources. If the
captions consistently describe an outdoor winter setting with snow on a driveway or
sidewalk, the bed is a quiet outdoor winter exterior (light cold wind, snow-damped
residential tone). Do NOT produce glacier, iceberg, or ice-floe ambiences — those scene
labels are unsupported noise.

For each bed provide:
  - query: a concrete CLAP-ready ambience retrieval query.
  - environment_descriptors: the corroborated scene descriptors.
  - start_segment / end_segment: the inclusive segment-index span the bed covers (use the
    full clip span when the environment is constant).
  - rationale: which corroborating evidence supports it, and which noisy labels you rejected.

Prefer ONE bed spanning the whole clip when the environment is constant; split only if it
genuinely changes. Return JSON matching the KeynoteResult schema.
"""

SIGNAL_TASK_PROMPT: str = """\
ROLE: Signal (discrete event) planner.

A signal is a foreground, time-localized sound caused by a visible action. Identify the
distinct physical actions in the clip and the sounds they make.

For snow-clearing footage the corroborated repeated action is a person shoveling snow, so
the expected signals are for example:
  - shovel scrape: the blade dragging across pavement/snow,
  - snow dump / impact: shoveled snow thrown to the side and landing,
  - footstep crunch: boots stepping on snow.
Only include signals the evidence supports.

MERGE repeated instances of the SAME action into ONE item whose segment_hints lists every
segment index where it occurs (a shovel scrape seen in segments 0-4 is ONE item with
segment_hints [0,1,2,3,4], not five items). Do not over-enumerate.

For each signal provide:
  - event_ref: short stable label (e.g. 'shovel_scrape').
  - query: a CLAP-ready one-shot retrieval query.
  - material_hint: the surface/material interaction when it matters (e.g. 'metal shovel
    blade on concrete and packed snow'); omit when material is irrelevant.
  - segment_hints: all segment indices where the action appears.
  - rationale: the corroborating evidence.

Return JSON matching the SignalResult schema.
"""

ARCHETYPAL_TASK_PROMPT: str = """\
ROLE: Archetypal (narrative / genre) planner.

Archetypal cues are non-diegetic, convention-based designed sounds that exist only to serve
dramatic or narrative intent (tension drones, risers, swells, emotional underscoring). They
are NOT the physical sounds of the scene.

Most ordinary, non-dramatic footage needs NONE of these. A person doing a routine winter
chore (clearing snow) has no narrative stakes and warrants no archetypal cues.

Do NOT manufacture drama, suspense, or emotion the footage does not contain. If there is no
genuine narrative justification, return an EMPTY list. An empty result is the correct and
expected answer for mundane footage.

Only if a cue is genuinely justified, provide for each:
  - query, motivation (narrative/emotional), generative_fallback (true if it must be
    synthesized rather than retrieved from a library), rationale.

Return JSON matching the ArchetypalResult schema (an empty cues list is valid).
"""


# --------------------------------------------------------------------------
# JSON extraction helper — the model is prompted (not constrained) to emit JSON,
# so be defensive about stray code fences or surrounding prose.
# --------------------------------------------------------------------------
_FENCE = re.compile(r"```(?:json)?", re.IGNORECASE)


def _extract_json(text: str) -> str:
    """Strip code fences and slice to the outermost {...} object."""
    cleaned = _FENCE.sub("", text).strip()
    # Defensive fallback: keep only the outermost JSON object.
    if not cleaned.startswith("{"):
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]
    return cleaned.strip()


# --------------------------------------------------------------------------
# BasePlanner — shared model, prompt assembly, generate + parse + validate loop
# --------------------------------------------------------------------------
class BasePlanner(abc.ABC):
    """Abstract planner. Subclasses set only `role_task_prompt` and `schema`."""

    role_task_prompt: str        # set by subclass
    schema: Type[BaseModel]      # set by subclass

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        *,
        stoplist: Optional[set[str]] = None,
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_retries: int = 1,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.stoplist = DEFAULT_STOPLIST if stoplist is None else stoplist
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.max_retries = max_retries

    def build_messages(
        self,
        segments: list[Segment],
        *,
        repair_note: Optional[str] = None,
    ) -> list[dict[str, str]]:
        """Assemble the chat messages: shared system prompt + role task + evidence + schema."""
        evidence_block = segments_to_prompt_block(segments, stoplist=self.stoplist)
        schema_json = json.dumps(self.schema.model_json_schema())

        user_content = (
            f"{self.role_task_prompt}\n"
            f"=== PERCEPTION EVIDENCE (segment indices only; no seconds) ===\n"
            f"{evidence_block}\n\n"
            f"=== OUTPUT JSON SCHEMA ===\n{schema_json}\n"
        )
        # On a retry, tell the model exactly what was wrong with its last attempt.
        if repair_note is not None:
            user_content += (
                f"\nYOUR PREVIOUS OUTPUT WAS INVALID: {repair_note}\n"
                f"Return corrected JSON only."
            )

        return [
            {"role": "system", "content": SHARED_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def _generate(self, messages: list[dict[str, str]]) -> str:
        """Run one constrained-path generation and return the decoded text."""
        # enable_thinking=False: free-form <think> blocks and JSON-only output don't mix.
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        model_inputs = self.tokenizer([prompt_text], return_tensors="pt").to(self.model.device)

        # Greedy by default (temperature 0.0) for reproducibility; sampling when asked.
        do_sample = self.temperature > 0.0
        generation_kwargs: dict = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            generation_kwargs["temperature"] = self.temperature
            generation_kwargs["top_p"] = self.top_p

        with torch.no_grad():   # no gradients needed at inference -> saves memory
            generated_ids = self.model.generate(**model_inputs, **generation_kwargs)

        # Keep only the newly generated tokens (drop the prompt portion).
        prompt_length = model_inputs.input_ids.shape[1]
        new_token_ids = generated_ids[0][prompt_length:].tolist()

        # Belt-and-suspenders: if a stray </think> appears, keep only what follows it.
        if THINK_CLOSE_TOKEN_ID in new_token_ids:
            close_index = len(new_token_ids) - new_token_ids[::-1].index(THINK_CLOSE_TOKEN_ID)
            new_token_ids = new_token_ids[close_index:]

        return self.tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()

    def plan(self, segments: list[Segment]) -> BaseModel:
        """Generate -> extract JSON -> validate against the schema, with one repair retry."""
        repair_note: Optional[str] = None
        last_error: Optional[Exception] = None

        for _ in range(self.max_retries + 1):
            raw_text = self._generate(self.build_messages(segments, repair_note=repair_note))
            json_text = _extract_json(raw_text)
            try:
                data = json.loads(json_text)
                return self.schema.model_validate(data)
            except (json.JSONDecodeError, ValidationError) as error:
                # Feed the error back so the retry can correct itself.
                last_error = error
                repair_note = str(error)[:500]

        raise ValueError(
            f"{type(self).__name__} failed to produce valid JSON: {last_error}"
        )


# --------------------------------------------------------------------------
# Three concrete planners — differ ONLY in role prompt + output schema
# --------------------------------------------------------------------------
class KeynotePlanner(BasePlanner):
    role_task_prompt = KEYNOTE_TASK_PROMPT
    schema = KeynoteResult


class SignalPlanner(BasePlanner):
    role_task_prompt = SIGNAL_TASK_PROMPT
    schema = SignalResult


class ArchetypalPlanner(BasePlanner):
    role_task_prompt = ARCHETYPAL_TASK_PROMPT
    schema = ArchetypalResult


# --------------------------------------------------------------------------
# run_all — run the three planners over one shared model
# --------------------------------------------------------------------------
@dataclass
class PlanningResult:
    keynote: KeynoteResult
    signal: SignalResult
    archetypal: ArchetypalResult


def run_all(
    record: PerceptionRecord,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    *,
    stoplist: Optional[set[str]] = None,
    **planner_kwargs,
) -> PlanningResult:
    """Run keynote, signal, and archetypal planners over the shared model + tokenizer."""
    segments = record.segments
    keynote = KeynotePlanner(model, tokenizer, stoplist=stoplist, **planner_kwargs).plan(segments)
    signal = SignalPlanner(model, tokenizer, stoplist=stoplist, **planner_kwargs).plan(segments)
    archetypal = ArchetypalPlanner(model, tokenizer, stoplist=stoplist, **planner_kwargs).plan(segments)
    return PlanningResult(keynote=keynote, signal=signal, archetypal=archetypal)


# --------------------------------------------------------------------------
# __main__ demo + runtime seconds-leak guard
# --------------------------------------------------------------------------
import os

_PREPROCESS_TXT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "preprocessing",
    "preprocess_output.txt",
)
_SECONDS_PATTERN = re.compile(r"\d+\.\d+s")   # catches "2.0s", "10.04s", etc.


def assert_no_seconds(payload: str) -> None:
    """Fail loudly if a seconds-like token leaked into serialized planner output."""
    hits = _SECONDS_PATTERN.findall(payload)
    if hits:
        raise AssertionError(f"Seconds-like tokens leaked into output: {hits}")


def main() -> None:
    record = parse_record_file(_PREPROCESS_TXT)
    model, tokenizer = load_planner_model(load_in_4bit=True)
    result = run_all(record, model, tokenizer)

    for name, planner_result in (
        ("keynote", result.keynote),
        ("signal", result.signal),
        ("archetypal", result.archetypal),
    ):
        dumped = planner_result.model_dump_json(indent=2)
        assert_no_seconds(dumped)   # runtime guard for the no-seconds rule
        print(f"=== {name} ===")
        print(dumped)
        print()


if __name__ == "__main__":
    main()

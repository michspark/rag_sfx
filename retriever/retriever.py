"""Signal retrieval over FSD50K-dev with CLAP + FAISS.

Consumes the LLM planner's output and, for now, retrieves only the `signal` events:
the top-k real SFX clips per event, matched by CLAP **audio** embedding similarity, with
full FSD50K metadata + license attached to every result.

Pipeline: planner signals -> compose text query -> CLAP text embedding -> FAISS cosine
search over precomputed clip audio embeddings -> attach metadata/license.

Design rules:
  - License travels with every asset (downstream stem recombination depends on it).
  - Filenames are opaque Freesound ids: matching is by audio embedding only, and every
    displayed description comes from metadata, never the filename.
  - Cosine via L2-normalized vectors + IndexFlatIP -> deterministic, reproducible order.
  - Factored so keynote/archetypal retrieval can reuse the same index later.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import Any, Optional

import faiss
import numpy as np
import pandas as pd
import torch


# --------------------------------------------------------------------------
# Paths and constants
# --------------------------------------------------------------------------
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))        # rag_sfx/retriever
FSK_DIR: str = "/home/sangheon/Desktop/FSK"
CLAP_EMB_DIR: str = os.path.join(FSK_DIR, "FSD50K.dev_clap")      # {id}.pt, each [512] f32
DEV_AUDIO_DIR: str = os.path.join(FSK_DIR, "FSD50K.dev_audio")    # {id}.wav (fallback source)
DEV_CSV: str = os.path.join(FSK_DIR, "FSD50K.ground_truth", "dev.csv")
META_JSON: str = os.path.join(FSK_DIR, "FSD50K.metadata", "dev_clips_info_FSD50K.json")

INDEX_DIR: str = os.path.join(BASE_DIR, "index")                  # persisted artifacts
EMB_NPY: str = os.path.join(INDEX_DIR, "clap_audio.npy")          # stacked (N,512) f32 normalized
IDS_JSON: str = os.path.join(INDEX_DIR, "ids.json")              # row -> clip id (aligned to npy+faiss)
FAISS_PATH: str = os.path.join(INDEX_DIR, "index.faiss")
META_TABLE: str = os.path.join(INDEX_DIR, "metadata.json")       # id -> resolved metadata

EMBED_DIM: int = 512
DEFAULT_K: int = 10


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize each row so that inner product equals cosine similarity.

    A zero-guard avoids division by zero. The FSD50K .pt embeddings are already
    normalized, so this is idempotent here, but we keep it so query embeddings and any
    freshly computed audio embeddings are normalized on the same footing.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0          # avoid 0/0 for any all-zero vector
    return (matrix / norms).astype(np.float32)


def resolve_license(url: str) -> str:
    """Map a Creative Commons license URL to a short tag. Never returns null/empty.

    Ordered most-specific-first so 'by-nc-sa' is not swallowed by the 'by' rule.
    Unknown schemes fall back to the raw URL (still non-null) rather than guessing.
    """
    lowered = (url or "").lower()
    if "by-nc-sa" in lowered:
        return "CC-BY-NC-SA"
    if "by-nc" in lowered:
        return "CC-BY-NC"
    if "by-sa" in lowered:
        return "CC-BY-SA"
    if "by" in lowered:
        return "CC-BY"
    if "publicdomain/zero" in lowered or "zero/1.0" in lowered:
        return "CC0"
    if "sampling+" in lowered:
        return "Sampling+"
    return url or "UNKNOWN"             # last resort: never null


# --------------------------------------------------------------------------
# CLAP wrapper (swappable). Wraps LAION-CLAP so the rest of the file only needs
# encode_text / encode_audio returning L2-normalized numpy arrays.
# --------------------------------------------------------------------------
class ClapEncoder:
    """Thin wrapper around laion_clap.CLAP_Module (630k-audioset-best checkpoint)."""

    def __init__(self) -> None:
        # Import here so the heavy CLAP/torch import only happens when retrieval runs.
        from laion_clap import CLAP_Module

        # enable_fusion=False + HTSAT-tiny + roberta matches the 630k-audioset-best ckpt
        # that produced the precomputed 512-dim audio embeddings.
        self.model = CLAP_Module(enable_fusion=False, amodel="HTSAT-tiny", tmodel="roberta")
        self.model.load_ckpt()         # uses the cached 630k-audioset-best checkpoint
        self.model.eval()

    def encode_text(self, texts: list[str]) -> np.ndarray:
        """Encode query strings -> (N, 512) L2-normalized float32."""
        with torch.no_grad():
            embeddings = self.model.get_text_embedding(texts, use_tensor=False)
        return _l2_normalize(np.asarray(embeddings, dtype=np.float32))

    def encode_audio(self, paths: list[str]) -> np.ndarray:
        """Encode wav files -> (N, 512) L2-normalized float32 (fallback path; unused here)."""
        with torch.no_grad():
            embeddings = self.model.get_audio_embedding_from_filelist(x=paths, use_tensor=False)
        return _l2_normalize(np.asarray(embeddings, dtype=np.float32))


# --------------------------------------------------------------------------
# Index build / load — the expensive stacking runs once, then is cached to disk
# --------------------------------------------------------------------------
def build_or_load_index() -> tuple[faiss.IndexFlatIP, list[str]]:
    """Return a FAISS cosine index over the corpus + the row-aligned list of clip ids.

    On first run: read every {id}.pt, stack + normalize, and persist the embeddings, the id
    list, and the FAISS index. On later runs: reload those artifacts (no recompute).
    """
    # Fast path: every artifact already exists -> just reload.
    if os.path.exists(EMB_NPY) and os.path.exists(IDS_JSON) and os.path.exists(FAISS_PATH):
        with open(IDS_JSON, "r", encoding="utf-8") as handle:
            ids = json.load(handle)
        index = faiss.read_index(FAISS_PATH)
        return index, ids

    os.makedirs(INDEX_DIR, exist_ok=True)

    # Sort ids so the row order (and therefore every reranking tie-break) is deterministic.
    pt_paths = sorted(glob.glob(os.path.join(CLAP_EMB_DIR, "*.pt")))
    if not pt_paths:
        raise FileNotFoundError(f"No .pt embeddings found in {CLAP_EMB_DIR}")

    ids: list[str] = []
    vectors: list[np.ndarray] = []
    for pt_path in pt_paths:
        clip_id = os.path.splitext(os.path.basename(pt_path))[0]   # "64760.pt" -> "64760"
        tensor = torch.load(pt_path, map_location="cpu")
        ids.append(clip_id)
        vectors.append(tensor.float().numpy())

    # Stack to (N, 512) and normalize so IndexFlatIP gives cosine similarity.
    embeddings = _l2_normalize(np.vstack(vectors).astype(np.float32))

    index = faiss.IndexFlatIP(EMBED_DIM)   # exact inner-product search (no approximation)
    index.add(embeddings)

    # Persist all three artifacts for instant, identical reloads.
    np.save(EMB_NPY, embeddings)
    with open(IDS_JSON, "w", encoding="utf-8") as handle:
        json.dump(ids, handle)
    faiss.write_index(index, FAISS_PATH)

    return index, ids


# --------------------------------------------------------------------------
# Metadata build / load — labels (csv) + description/tags/license (json) per id
# --------------------------------------------------------------------------
def build_or_load_metadata(ids: list[str]) -> dict[str, dict[str, Any]]:
    """Return id -> {labels, title, description, tags, license, license_url}.

    Joins AudioSet-style labels from dev.csv with Freesound info from the metadata JSON,
    resolving each license URL to a short tag. Persisted so later runs skip the join.
    """
    if os.path.exists(META_TABLE):
        with open(META_TABLE, "r", encoding="utf-8") as handle:
            return json.load(handle)

    os.makedirs(INDEX_DIR, exist_ok=True)

    # labels: read fname as string to match the .pt / json string ids exactly.
    dev_frame = pd.read_csv(DEV_CSV, dtype={"fname": str})
    labels_by_id: dict[str, str] = dict(zip(dev_frame["fname"], dev_frame["labels"]))

    with open(META_JSON, "r", encoding="utf-8") as handle:
        clips_info = json.load(handle)

    table: dict[str, dict[str, Any]] = {}
    for clip_id in ids:
        info = clips_info.get(clip_id, {})
        license_url = info.get("license", "")
        table[clip_id] = {
            "labels": labels_by_id.get(clip_id, ""),
            "title": info.get("title", ""),
            "description": info.get("description", ""),
            "tags": info.get("tags", []),
            "license_url": license_url,
            "license": resolve_license(license_url),
        }

    with open(META_TABLE, "w", encoding="utf-8") as handle:
        json.dump(table, handle)

    return table


# --------------------------------------------------------------------------
# Query composer — signal item -> CLAP text query
# --------------------------------------------------------------------------
def compose_query(signal: dict[str, Any]) -> str:
    """Build the CLAP query string for one signal item.

    Prefer an explicit free-text `query` when present. Otherwise humanize `event_ref`
    ('shovel_scrape' -> 'shovel scrape') and append `material_hint` so material fidelity
    is part of the query, e.g. 'shovel scrape, metal shovel blade on concrete'.
    """
    explicit_query = signal.get("query")
    if explicit_query:
        return explicit_query

    humanized = signal["event_ref"].replace("_", " ").strip()
    material_hint = signal.get("material_hint")
    if material_hint:
        return f"{humanized}, {material_hint}"
    return humanized


# --------------------------------------------------------------------------
# Result containers
# --------------------------------------------------------------------------
@dataclass
class Hit:
    clip_id: str
    score: float
    labels: str          # FSD50K AudioSet-style labels
    description: str     # from metadata, never the filename
    tags: list[str]
    license: str         # resolved short tag (never null)
    license_url: str
    audio_path: str


@dataclass
class SignalRetrieval:
    event_ref: str
    query: str
    material_hint: Optional[str]
    segment_hints: list[int]     # passed through for downstream onset anchoring
    retriever_type: str          # "signal"
    hits: list[Hit]


# --------------------------------------------------------------------------
# Generic search core (reused by future retrieve_keynote) + signal wrapper
# --------------------------------------------------------------------------
def _search(
    encoder: ClapEncoder,
    index: faiss.IndexFlatIP,
    ids: list[str],
    metadata: dict[str, dict[str, Any]],
    texts: list[str],
    k: int,
) -> list[list[Hit]]:
    """Encode texts and return, per text, its top-k Hits sorted by descending cosine."""
    query_embeddings = encoder.encode_text(texts)            # (Q, 512) normalized
    scores, neighbors = index.search(query_embeddings, k)    # IndexFlatIP -> cosine, desc order

    results: list[list[Hit]] = []
    for row_scores, row_neighbors in zip(scores, neighbors):
        hits: list[Hit] = []
        for score, row in zip(row_scores, row_neighbors):
            clip_id = ids[row]
            info = metadata[clip_id]
            hits.append(
                Hit(
                    clip_id=clip_id,
                    score=float(score),
                    labels=info["labels"],
                    description=info["description"],
                    tags=info["tags"],
                    license=info["license"],
                    license_url=info["license_url"],
                    audio_path=os.path.join(DEV_AUDIO_DIR, f"{clip_id}.wav"),
                )
            )
        results.append(hits)
    return results


def retrieve_signals(
    planner_output: dict[str, Any],
    k: int = DEFAULT_K,
    *,
    encoder: ClapEncoder,
    index: faiss.IndexFlatIP,
    ids: list[str],
    metadata: dict[str, dict[str, Any]],
) -> list[SignalRetrieval]:
    """Retrieve top-k SFX clips for each signal event in the planner output."""
    signals: list[dict[str, Any]] = planner_output.get("signals", [])
    if not signals:
        return []

    texts = [compose_query(signal) for signal in signals]
    per_query_hits = _search(encoder, index, ids, metadata, texts, k)

    retrievals: list[SignalRetrieval] = []
    for signal, query_text, hits in zip(signals, texts, per_query_hits):
        retrievals.append(
            SignalRetrieval(
                event_ref=signal["event_ref"],
                query=query_text,
                material_hint=signal.get("material_hint"),
                segment_hints=signal.get("segment_hints", []),
                retriever_type="signal",
                hits=hits,
            )
        )
    return retrievals


# --------------------------------------------------------------------------
# Demo / CLI
# --------------------------------------------------------------------------
# The 3-signal sample produced by the planner on the snow-shoveling clip.
SAMPLE_PLANNER_OUTPUT: dict[str, Any] = {
    "signals": [
        {
            "event_ref": "shovel_scrape",
            "material_hint": "metal shovel blade on concrete and packed snow",
            "segment_hints": [0, 1, 2, 3, 4],
        },
        {
            "event_ref": "snow_dump",
            "material_hint": "snow on concrete",
            "segment_hints": [0, 1, 2, 4],
        },
        {
            "event_ref": "footstep_crunch",
            "material_hint": "snow and ice",
            "segment_hints": [0, 2, 3, 4],
        },
    ]
}


def main() -> None:
    encoder = ClapEncoder()
    index, ids = build_or_load_index()
    metadata = build_or_load_metadata(ids)
    print(f"[index] {len(ids)} clips indexed (dim={EMBED_DIM})")

    retrievals = retrieve_signals(
        SAMPLE_PLANNER_OUTPUT, k=DEFAULT_K,
        encoder=encoder, index=index, ids=ids, metadata=metadata,
    )

    for retrieval in retrievals:
        print("\n" + "=" * 70)
        print(f"SIGNAL: {retrieval.event_ref}  (type={retrieval.retriever_type})")
        print(f"  query: {retrieval.query}")
        print(f"  segment_hints: {retrieval.segment_hints}")
        print("-" * 70)

        # Acceptance checks: exactly k hits, every license non-null.
        assert len(retrieval.hits) == DEFAULT_K, f"expected {DEFAULT_K} hits"
        for rank, hit in enumerate(retrieval.hits, start=1):
            assert hit.license, f"missing license for clip {hit.clip_id}"
            description_snippet = (hit.description or "").replace("\n", " ")[:60]
            print(
                f"  {rank:2d}. {hit.score:.3f}  id={hit.clip_id:>8}  "
                f"[{hit.license}]  labels={hit.labels[:40]}"
            )
            print(f"        desc: {description_snippet}")

    print("\n[ok] 3 signals x 10 hits, all licenses present")


if __name__ == "__main__":
    main()

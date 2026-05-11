"""
graph/builder.py — load parsed embeddings from disk, build analysis inputs.

Each document in parsed/ has:
  {stem}_sections.npy    float32 [N, 768]
  {stem}_meta.json       list of N section dicts

A directory of such files is loaded into an EmbeddingSet which stacks all
section embeddings and keeps a flat list of Section objects for labelling.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Section:
    stem: str  # filename stem (e.g. "0200.2a")
    doc_id: str  # "DOE O 200.2A" or stem if blank
    doc_type: str
    subject: str
    section_id: str
    heading: str
    text_preview: str
    obligations: int
    citations: int
    global_idx: int  # row in the stacked embedding matrix


@dataclass
class EmbeddingSet:
    sections: list[Section]
    embeddings: np.ndarray  # float32 [N, 768]

    @property
    def n(self) -> int:
        return len(self.sections)

    def label(self, idx: int, max_heading: int = 25) -> str:
        """Short display label for section idx."""
        s = self.sections[idx]
        doc = s.doc_id or s.stem
        doc = doc[:18]
        heading = s.heading[:max_heading] if s.heading else s.section_id
        return f"{doc} | {heading}"


def _load_one(npy_path: Path) -> tuple[np.ndarray, list[Section]]:
    stem = npy_path.stem.replace("_sections", "")
    meta_path = npy_path.parent / f"{stem}_meta.json"

    embeddings = np.load(str(npy_path)).astype(np.float32)

    if not meta_path.exists():
        # Fallback: create synthetic meta
        sections = [
            Section(
                stem=stem,
                doc_id=stem,
                doc_type="Order",
                subject=stem,
                section_id=str(i),
                heading=f"SEC-{i}",
                text_preview="",
                obligations=0,
                citations=0,
                global_idx=0,
            )
            for i in range(embeddings.shape[0])
        ]
    else:
        with open(meta_path, encoding="utf-8") as fh:
            raw = json.load(fh)
        sections = []
        for i, m in enumerate(raw):
            doc_id = m.get("doc_id") or stem
            sections.append(
                Section(
                    stem=stem,
                    doc_id=doc_id,
                    doc_type=m.get("doc_type", "Order"),
                    subject=m.get("subject", ""),
                    section_id=m.get("section_id", str(i)),
                    heading=m.get("heading", ""),
                    text_preview=m.get("text_preview", ""),
                    obligations=m.get("obligations", 0),
                    citations=m.get("citations", 0),
                    global_idx=0,  # filled in below
                )
            )

    return embeddings, sections


def load_embeddings(sources: list[Path]) -> EmbeddingSet:
    """
    Load embeddings from a list of *_sections.npy files (or directories).

    If a source is a directory, all *_sections.npy files within it are loaded.
    Sections are numbered consecutively across all documents.
    """
    npy_paths: list[Path] = []
    for src in sources:
        src = Path(src)
        if src.is_dir():
            npy_paths.extend(sorted(src.glob("*_sections.npy")))
        elif src.suffix == ".npy" and src.exists():
            npy_paths.append(src)
        else:
            raise FileNotFoundError(f"Not a .npy file or directory: {src}")

    if not npy_paths:
        raise ValueError(f"No *_sections.npy files found in {sources}")

    all_embeddings: list[np.ndarray] = []
    all_sections: list[Section] = []
    global_idx = 0

    for p in npy_paths:
        emb, secs = _load_one(p)
        for i, s in enumerate(secs):
            s.global_idx = global_idx + i
        all_embeddings.append(emb)
        all_sections.extend(secs)
        global_idx += len(secs)

    stacked = (
        np.vstack(all_embeddings) if all_embeddings else np.zeros((0, 768), np.float32)
    )
    return EmbeddingSet(sections=all_sections, embeddings=stacked)

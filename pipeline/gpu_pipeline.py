#!/usr/bin/env python3
"""
pipeline/gpu_pipeline.py — GPU-accelerated streaming PDF parser + embedder

Work-stealing producer-consumer architecture:
  - Shared pdf_queue feeds N CPU workers (any worker grabs any PDF — no pre-assignment)
  - CPU workers parse PDFs, truncate section text to ~2K chars, push onto section_queue
  - 1 GPU worker batches sections → legal-bert embeddings → .npy files
  - Text is truncated BEFORE queue push — no 169K-char pickles through OS pipes

Replaces the sequential parser.py → embedder.py flow with concurrent CPU/GPU
execution.  The P100 GPU is kept saturated while CPU workers parse ahead.
Handles large PDFs (13MB+, 149 pages) without GPU starvation.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# CUDA + multiprocessing: must use 'spawn' (not 'fork') on Linux.
# fork inherits the parent's CUDA context, and CUDA cannot be re-initialized
# in a child process. 'spawn' starts a fresh Python interpreter per child.
if mp.get_start_method(allow_none=True) != "spawn":
    mp.set_start_method("spawn", force=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
EMBED_DIM = 768
MAX_TOKENS = 512          # legal-bert max input tokens
TEXT_TRUNC_CHARS = MAX_TOKENS * 4  # ~2,000 chars — safely above 512 tokens
GPU_BATCH_SIZE = 64       # sections per GPU forward pass (P100 16GB)
CPU_WORKERS = 4           # number of parallel PDF parsers
SECTION_QUEUE_SIZE = 200  # max pending section dicts before backpressure
PARSE_TIMEOUT = 30        # seconds per PDF before worker is killed + restarted

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Compiled regex patterns (copied from parser.py — identical logic)
# DOE directive references: DOE O 251.1C, DOE P 226.1B, DOE M 435.1-1
_DOE_REF = re.compile(
    r"\bDOE\s+[OPMGN]\s+[\d]+\.[\d]+[A-Za-z\d.\-]*"
    r"(?:\s+Chg\s*\d+)?(?:\s+\([A-Za-z]+\))?",
    re.IGNORECASE,
)
# CFR: 10 CFR 830, 48 CFR 952.204-2, 10 C.F.R. § 835
_CFR_REF = re.compile(
    r"\b(\d+)\s+C\.?F\.?R\.?\s+(?:[Pp]arts?\s+|§\s*)?(\d+[\d.\-]*)",
)
# USC: 50 U.S.C. Section 2406, 42 USC § 7158
_USC_REF = re.compile(
    r"\b(\d+)\s+U\.?S\.?C\.?\s+(?:[Ss]ec(?:tions?|\.?)\s*|§\s*)?(\d+[\d.\-]*)",
)
# Executive Orders
_EO_REF = re.compile(r"\bExecutive\s+Order\s+(\d+)", re.IGNORECASE)
# Public Law
_PL_REF = re.compile(
    r"\b(?:P\.?L\.?|Public\s+Law)\s+(\d+[-–]\d+)",
    re.IGNORECASE,
)
# Homeland Security Presidential Directives
_HSPD_REF = re.compile(r"\bHSPD[-\s](\d+)", re.IGNORECASE)

# Obligation sentences: contains "shall" or "must" as a modal
_OBLIGATION_RE = re.compile(
    r"[^.!?]*\b(?:shall|must)\b[^.!?]*[.!?]",
    re.IGNORECASE,
)
# Definition patterns
_DEF_INLINE = re.compile(
    r'"([^"]{2,80})"\s+means\s+([^.]{10,300}\.)',
    re.IGNORECASE,
)
_DEF_CAPS = re.compile(
    r"\b([A-Z][A-Z\s]{2,50})\s+means\s+([^.]{10,300}\.)",
)
_DEF_PHRASE = re.compile(
    r'\bthe\s+term\s+"([^"]{2,80})"\s+(?:means|is\s+defined\s+as)\s+([^.]{10,300}\.)',
    re.IGNORECASE,
)

# Page-header noise: "DOE O 151.1D  \n  8-11-2016" repeating at top of each page
_PAGE_HEADER = re.compile(
    r"\n\s*\d*\s*\n?\s*DOE\s+[OPMGN]\s+[\d]+\.[\d]+[A-Za-z\d.\- ]*\n\s*[\d\-]+\s*\n",
    re.IGNORECASE,
)
# Catch the alternate header layout (page num on separate line)
_PAGE_HEADER2 = re.compile(
    r"\n\s*DOE\s+[OPMGN]\s+[\d]+\.[\d]+[A-Za-z\d.\- ]*\s*\n\s*\d+\s*\n\s*[\d\-]+\s*\n",
    re.IGNORECASE,
)

# Section number on its own line: "1.", "2.", ..., "25."
_SEC_NUM_LINE = re.compile(r"^\s*(\d{1,2})\.\s*$")
# Section heading: ALL-CAPS words (2+ chars) followed by period
_SEC_HEADING = re.compile(r"^([A-Z][A-Z\s,/()\-]{2,80})\.\s*(.*)")

# DOE agencies (dictionary lookup — no spaCy needed)
_DOE_AGENCIES = {
    "DOE", "NNSA", "DOT", "NRC", "EPA", "OSHA", "OMB", "DOD",
    "NASA", "GAO", "DHS", "FBI", "CIA", "NSA", "FEMA", "NIST",
}

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Citation:
    cit_type: str  # cfr | doe | usc | eo | pl | hspd
    reference: str  # canonical citation text
    context: str  # surrounding ~80 chars


@dataclass
class Obligation:
    modal: str  # "shall" or "must"
    sentence: str


@dataclass
class Definition:
    term: str
    definition: str


@dataclass
class ParsedSection:
    section_id: str
    heading: str
    text: str
    citations: list = field(default_factory=list)
    obligations: list = field(default_factory=list)
    definitions: list = field(default_factory=list)
    dates: list = field(default_factory=list)
    agencies: list = field(default_factory=list)


@dataclass
class ParsedDocument:
    source_file: str
    doc_id: str  # "DOE O 151.1D"
    doc_type: str  # "Order" | "Policy" | "Manual" | "Guide" | "Notice"
    subject: str
    approved_date: str
    initiating_office: str
    page_count: int
    sections: list = field(default_factory=list)
    all_citations: list = field(default_factory=list)
    all_obligations: list = field(default_factory=list)
    all_definitions: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Logging (call once per process — file handles don't survive mp.Process)
# ---------------------------------------------------------------------------


def setup_logging(output_dir: Path) -> logging.Logger:
    """Create a per-process logger writing to stdout + file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(processName)s] %(levelname)-8s %(message)s"
    # Clear any existing handlers from parent process
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(output_dir / "gpu_pipeline.log", encoding="utf-8"),
        ],
        force=True,
    )
    return logging.getLogger("gpu_pipeline")


# ---------------------------------------------------------------------------
# Text extraction (ported from parser.py)
# ---------------------------------------------------------------------------


def _extract_full_text(pdf_path: Path) -> tuple[str, int]:
    """Return (concatenated_text, page_count) from a PDF."""
    doc = fitz.open(str(pdf_path))
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n\n".join(pages), len(pages)


def _clean_text(text: str) -> str:
    """Strip repeating page headers and normalize whitespace."""
    text = _PAGE_HEADER.sub("\n", text)
    text = _PAGE_HEADER2.sub("\n", text)
    # Collapse 3+ blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


# ---------------------------------------------------------------------------
# Header metadata extraction (ported from parser.py)
# ---------------------------------------------------------------------------


def _extract_metadata(text: str, log: logging.Logger) -> dict:
    """Pull doc_id, subject, date, office from document header text."""
    import dateparser

    meta = {
        "doc_id": "",
        "doc_type": "Order",
        "subject": "",
        "approved_date": "",
        "initiating_office": "",
    }

    # Doc ID: "DOE O 151.1D" or "DOE O 151.1D Chg 1"
    m = re.search(
        r"\bDOE\s+([OPMGN])\s+([\d]+\.[\d]+[A-Za-z\d.\-]*(?:\s+Chg\s*\d+)?)", text[:500]
    )
    if m:
        type_map = {
            "O": "Order", "P": "Policy", "M": "Manual",
            "G": "Guide", "N": "Notice",
        }
        meta["doc_id"] = f"DOE {m.group(1)} {m.group(2).strip()}"
        meta["doc_type"] = type_map.get(m.group(1), "Order")

    # Subject line
    m = re.search(r"SUBJECT[:\s]+([^\n]{5,200})", text[:1500], re.IGNORECASE)
    if m:
        meta["subject"] = m.group(1).strip()

    # Approval date
    m = re.search(r"Approved\s*[:\s]+([^\n]{5,30})", text[:800], re.IGNORECASE)
    if m:
        raw = m.group(1).strip()
        try:
            parsed = dateparser.parse(raw, settings={"RETURN_AS_TIMEZONE_AWARE": False})
            meta["approved_date"] = parsed.strftime("%Y-%m-%d") if parsed else raw
        except Exception:
            meta["approved_date"] = raw

    # Initiating office
    m = re.search(r"INITIATED\s+BY[:\s]+([^\n]{5,100})", text[:1000], re.IGNORECASE)
    if m:
        meta["initiating_office"] = m.group(1).strip()

    return meta


# ---------------------------------------------------------------------------
# Section splitting (ported from parser.py — with TEXT_TRUNC_CHARS truncation)
# ---------------------------------------------------------------------------


def _split_sections(text: str) -> list[dict]:
    """
    Split DOE Order body text into numbered top-level sections.

    DOE Orders use two header formats:
      Format A:  "1.\n\nPURPOSE. text..."   (number then heading on next line)
      Format B:  "1. PURPOSE. text..."       (all on same line)

    Each section's text is truncated to TEXT_TRUNC_CHARS to prevent
    169K-char sections from being pickled through IPC pipes only to be
    truncated to 512 tokens at embed time.
    """
    lines = text.splitlines()
    sections: list[dict] = []
    cur_num: Optional[str] = None
    cur_heading: Optional[str] = None
    cur_lines: list[str] = []

    def _flush():
        if cur_num and cur_heading:
            sections.append(
                {
                    "id": cur_num,
                    "heading": cur_heading,
                    "text": "\n".join(cur_lines).strip()[:TEXT_TRUNC_CHARS],
                }
            )

    i = 0
    while i < len(lines):
        stripped = lines[i].strip()

        # Format B: "1. PURPOSE. text" (number + heading on same line)
        m = re.match(r"^(\d{1,2})\.\s+([A-Z][A-Z ,/()\-]{2,80})\.\s*(.*)", stripped)
        if m:
            _flush()
            cur_num = m.group(1)
            cur_heading = m.group(2).strip()
            cur_lines = [m.group(3)] if m.group(3).strip() else []
            i += 1
            continue

        # Format A: bare section number on its own line
        m_num = _SEC_NUM_LINE.match(stripped)
        if m_num:
            _flush()
            cur_num = m_num.group(1)
            cur_heading = None
            cur_lines = []
            i += 1
            # Scan ahead for the heading (skip blanks)
            while i < len(lines):
                next_stripped = lines[i].strip()
                if not next_stripped:
                    i += 1
                    continue
                m_head = _SEC_HEADING.match(next_stripped)
                if m_head:
                    cur_heading = m_head.group(1).strip()
                    if m_head.group(2).strip():
                        cur_lines.append(m_head.group(2).strip())
                    i += 1
                break
            continue

        # Regular content
        if cur_num:
            cur_lines.append(lines[i])
        i += 1

    _flush()

    # Fallback: no sections detected — treat entire text as one block
    if not sections:
        sections = [{"id": "0", "heading": "FULL_TEXT", "text": text.strip()[:TEXT_TRUNC_CHARS]}]

    return sections


# ---------------------------------------------------------------------------
# NLP extractors (ported from parser.py)
# ---------------------------------------------------------------------------


def _extract_citations(text: str) -> list[Citation]:
    cits: list[Citation] = []

    def _ctx(m):
        s, e = max(0, m.start() - 40), min(len(text), m.end() + 40)
        return text[s:e].replace("\n", " ").strip()

    for m in _DOE_REF.finditer(text):
        cits.append(Citation("doe", m.group().strip(), _ctx(m)))
    for m in _CFR_REF.finditer(text):
        cits.append(Citation("cfr", f"{m.group(1)} CFR {m.group(2)}", _ctx(m)))
    for m in _USC_REF.finditer(text):
        cits.append(Citation("usc", f"{m.group(1)} USC {m.group(2)}", _ctx(m)))
    for m in _EO_REF.finditer(text):
        cits.append(Citation("eo", f"EO {m.group(1)}", _ctx(m)))
    for m in _PL_REF.finditer(text):
        cits.append(Citation("pl", f"PL {m.group(1)}", _ctx(m)))
    for m in _HSPD_REF.finditer(text):
        cits.append(Citation("hspd", f"HSPD {m.group(1)}", _ctx(m)))

    # Deduplicate by reference text
    seen = set()
    unique = []
    for c in cits:
        if c.reference not in seen:
            seen.add(c.reference)
            unique.append(c)
    return unique


def _extract_obligations(text: str) -> list[Obligation]:
    """Extract sentences containing 'shall' or 'must' (modal, not noun use)."""
    results: list[Obligation] = []
    for m in _OBLIGATION_RE.finditer(text):
        sent = m.group().strip()
        if len(sent) < 15 or len(sent) > 600:
            continue
        modal = "shall" if re.search(r"\bshall\b", sent, re.IGNORECASE) else "must"
        results.append(Obligation(modal, sent.replace("\n", " ")))
    # Deduplicate
    seen: set[str] = set()
    unique = []
    for o in results:
        if o.sentence not in seen:
            seen.add(o.sentence)
            unique.append(o)
    return unique


def _extract_definitions(text: str) -> list[Definition]:
    defs: list[Definition] = []
    for pat in (_DEF_INLINE, _DEF_CAPS, _DEF_PHRASE):
        for m in pat.finditer(text):
            defs.append(Definition(m.group(1).strip(), m.group(2).strip()))
    seen: set[str] = set()
    unique = []
    for d in defs:
        if d.term not in seen:
            seen.add(d.term)
            unique.append(d)
    return unique


def _extract_dates(text: str) -> list[str]:
    """Find dates in regulatory formats (M-D-YYYY, spelled out, etc.)."""
    import dateparser
    date_re = re.compile(
        r"\b(?:"
        r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}"  # 8-11-2016
        r"|(?:January|February|March|April|May|June|"
        r"July|August|September|October|November|December)"
        r"\s+\d{1,2},?\s+\d{4}"  # January 1, 2020
        r")",
        re.IGNORECASE,
    )
    dates: list[str] = []
    seen: set[str] = set()
    for m in date_re.finditer(text):
        raw = m.group().strip()
        if raw in seen:
            continue
        seen.add(raw)
        try:
            parsed = dateparser.parse(raw, settings={"RETURN_AS_TIMEZONE_AWARE": False})
            dates.append(parsed.strftime("%Y-%m-%d") if parsed else raw)
        except Exception:
            dates.append(raw)
    return dates


def _extract_agencies(text: str) -> list[str]:
    """Extract agency names via dictionary lookup (no spaCy dependency).

    DOE regulatory text uses a closed set of well-known agency acronyms.
    Dictionary lookup is instant and removes the en_core_web_sm dependency.
    """
    names: set[str] = set()
    for acro in _DOE_AGENCIES:
        if re.search(rf"\b{acro}\b", text):
            names.add(acro)
    return sorted(names)


# ---------------------------------------------------------------------------
# Top-level parse function (ported from parser.py)
# ---------------------------------------------------------------------------


def parse_pdf(pdf_path: Path, log: logging.Logger) -> Optional[ParsedDocument]:
    """Parse a single DOE Order PDF into a structured ParsedDocument."""
    try:
        raw_text, page_count = _extract_full_text(pdf_path)
    except Exception as exc:
        log.error("fitz failed on %s: %s", pdf_path.name, exc)
        return None

    if len(raw_text.strip()) < 100:
        log.warning("Nearly empty text in %s — likely scanned/image PDF", pdf_path.name)

    clean = _clean_text(raw_text)
    meta = _extract_metadata(clean, log)

    # Extract NLP features from the full document
    all_cits = _extract_citations(clean)
    all_obligs = _extract_obligations(clean)
    all_defs = _extract_definitions(clean)
    agencies = _extract_agencies(clean)

    # Split into sections and annotate each
    raw_sections = _split_sections(clean)
    parsed_sections = []
    for sec in raw_sections:
        text = sec["text"]
        psec = ParsedSection(
            section_id=sec["id"],
            heading=sec["heading"],
            text=text,
            citations=[asdict(c) for c in _extract_citations(text)],
            obligations=[asdict(o) for o in _extract_obligations(text)],
            definitions=[asdict(d) for d in _extract_definitions(text)],
            dates=_extract_dates(text),
            agencies=agencies,  # shared across sections (doc-level)
        )
        parsed_sections.append(psec)

    return ParsedDocument(
        source_file=pdf_path.name,
        doc_id=meta["doc_id"],
        doc_type=meta["doc_type"],
        subject=meta["subject"],
        approved_date=meta["approved_date"],
        initiating_office=meta["initiating_office"],
        page_count=page_count,
        sections=[asdict(s) for s in parsed_sections],
        all_citations=[asdict(c) for c in all_cits],
        all_obligations=[asdict(o) for o in all_obligs[:50]],  # cap for JSON size
        all_definitions=[asdict(d) for d in all_defs],
    )


def save_parsed(doc: ParsedDocument, output_dir: Path) -> Path:
    """Write ParsedDocument to JSON in output_dir. Returns path to written file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(doc.source_file).stem
    out_path = output_dir / f"{stem}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(asdict(doc), fh, indent=2, ensure_ascii=False)
    return out_path


# ---------------------------------------------------------------------------
# GPU embedding worker
# ---------------------------------------------------------------------------


def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token embeddings, ignoring padding tokens."""
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def _flush_batch(
    doc_sections: dict[str, list[dict]],
    tokenizer,
    model,
    device,
    output_dir: Path,
    log: logging.Logger,
    batch_size: int = 64,
) -> int:
    """
    Embed all pending sections in doc_sections, save .npy files, clear dict.
    Text is pre-truncated — no further length checks needed.

    CUDA cosine kernels normalize on the fly, so we store raw embeddings
    (no L2 normalization needed at save time).

    Returns number of sections embedded.
    """
    if not doc_sections:
        return 0

    embedded = 0
    # Snapshot keys so we can delete while iterating
    for stem in list(doc_sections.keys()):
        sections = doc_sections[stem]
        if not sections:
            del doc_sections[stem]
            continue

        texts = []
        meta = []
        for sec in sections:
            heading = sec.get("heading", "")
            text = sec.get("text", "")
            full = f"{heading} | {text}"
            texts.append(full)
            meta.append({
                "section_id": sec.get("section_id", ""),
                "heading": heading,
                "text_preview": text[:200].replace("\n", " "),
                "obligations": len(sec.get("obligations", [])),
                "citations": len(sec.get("citations", [])),
            })

        # Batch embed through GPU
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch, padding=True, truncation=True,
                max_length=MAX_TOKENS, return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                out = model(**enc)
            vecs = _mean_pool(out.last_hidden_state, enc["attention_mask"])
            all_vecs.append(vecs.cpu().float().numpy())

        embeddings = (
            np.vstack(all_vecs)
            if all_vecs
            else np.zeros((0, EMBED_DIM), dtype=np.float32)
        )

        # Save
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(output_dir / f"{stem}_sections.npy"), embeddings)
        with open(output_dir / f"{stem}_meta.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, ensure_ascii=False)

        embedded += len(sections)
        del doc_sections[stem]

    if embedded:
        log.info("  GPU: embedded %d sections", embedded)
    return embedded


def _gpu_worker(
    section_queue: mp.Queue,
    output_dir: Path,
    stop_event: mp.Event,
    batch_size: int = 64,
) -> int:
    """
    GPU worker process.  Pulls section dicts from the queue, batches them,
    runs legal-bert embeddings, saves .npy + _meta.json per document.

    All section text is pre-truncated to TEXT_TRUNC_CHARS by CPU workers.
    Sets up its own logging (file handles don't survive mp.Process).

    Returns total sections embedded.
    """
    log = setup_logging(output_dir)

    # Load model once (stays resident in GPU memory)
    log.info("GPU worker: loading %s", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    log.info("GPU worker: model on %s, %.1f GB free",
             device,
             torch.cuda.mem_get_info()[0] / 1e9 if torch.cuda.is_available() else 0)

    # Accumulate sections per document
    doc_sections: dict[str, list[dict]] = {}  # stem → list of section dicts
    total_embedded = 0

    while not stop_event.is_set():
        try:
            item = section_queue.get(timeout=2.0)
        except Exception:
            continue

        if item is None:  # poison pill — flush and exit
            break

        stem, section_dict = item
        if stem not in doc_sections:
            doc_sections[stem] = []
        doc_sections[stem].append(section_dict)

        # Flush when enough sections are pending across all documents
        total_pending = sum(len(v) for v in doc_sections.values())
        if total_pending >= batch_size:
            total_embedded += _flush_batch(
                doc_sections, tokenizer, model, device, output_dir, log, batch_size
            )

    # Final flush — drain everything
    total_embedded += _flush_batch(
        doc_sections, tokenizer, model, device, output_dir, log, batch_size
    )
    log.info("GPU worker: done — %d sections embedded", total_embedded)
    return total_embedded


# ---------------------------------------------------------------------------
# CPU parser worker (work-stealing — pulls from shared pdf_queue)
# ---------------------------------------------------------------------------


def _cpu_worker(
    pdf_queue: mp.Queue,
    section_queue: mp.Queue,
    progress_queue: mp.Queue,
    output_dir: Path,
) -> tuple[int, int]:
    """
    CPU worker process.  Pulls PDFs from the shared pdf_queue (work-stealing),
    parses them, and pushes truncated section dicts to section_queue.

    Work-stealing means no worker is stuck waiting while others idle —
    any worker that finishes its current PDF immediately grabs the next one.

    Sets up its own logging (file handles don't survive mp.Process).

    Returns (parsed, failed).
    """
    log = setup_logging(output_dir)
    parsed = 0
    failed = 0

    while True:
        try:
            pdf_path = pdf_queue.get(timeout=1.0)
        except Exception:
            # Queue empty and all producers done — exit
            break

        if pdf_path is None:  # sentinel — no more PDFs
            break

        try:
            doc = parse_pdf(pdf_path, log)
            if doc is None:
                log.warning("CPU worker: failed to parse %s", pdf_path.name)
                failed += 1
                progress_queue.put(("failed", pdf_path.name))
                continue

            # Save parsed JSON to user-specified output_dir
            save_parsed(doc, output_dir)

            # Push sections to GPU worker — text is already truncated
            # to TEXT_TRUNC_CHARS by _split_sections inside parse_pdf()
            stem = Path(doc.source_file).stem
            for sec in doc.sections:
                section_queue.put((stem, sec))

            parsed += 1
            progress_queue.put(("parsed", pdf_path.name))
            log.debug("CPU worker: parsed %s (%d sections)", pdf_path.name, len(doc.sections))

        except Exception as exc:
            log.error("CPU worker failed on %s: %s", pdf_path.name, exc)
            failed += 1
            progress_queue.put(("failed", pdf_path.name))

    log.info("CPU worker exiting: parsed=%d failed=%d", parsed, failed)
    return parsed, failed


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _auto_tune(log: logging.Logger) -> tuple[int, int]:
    """Auto-scale batch size and workers from GPU VRAM."""
    if not torch.cuda.is_available():
        log.warning("CUDA not available — using CPU-only fallback")
        return 8, 1

    free_vram_gb = torch.cuda.mem_get_info()[0] / 1e9
    gpu_name = torch.cuda.get_device_name(0)
    log.info("GPU: %s — %.1f GB free VRAM", gpu_name, free_vram_gb)

    # legal-bert base: ~440MB weights + ~100MB activations per 64-batch
    # P100 16GB: comfortably handles 128-batch with 12GB+ free
    if free_vram_gb >= 14:
        batch = 128
    elif free_vram_gb >= 8:
        batch = 64
    elif free_vram_gb >= 4:
        batch = 32
    else:
        batch = 16

    # CPU workers: match physical cores, cap at 8
    # Work-stealing means more workers is strictly better — no starvation risk
    workers = min(mp.cpu_count(), 8)

    log.info("Auto-tuned: batch_size=%d, cpu_workers=%d", batch, workers)
    return batch, workers


def _detect_collisions(pdf_files: list[Path], log: logging.Logger) -> dict[str, list[Path]]:
    """Detect PDFs that would overwrite each other due to identical stems.

    Returns dict of stem → list of colliding paths.  Non-colliding stems are not included.
    """
    by_stem = defaultdict(list)
    for p in pdf_files:
        by_stem[p.stem].append(p)
    collisions = {stem: paths for stem, paths in by_stem.items() if len(paths) > 1}
    if collisions:
        log.warning("STEM COLLISIONS DETECTED — these PDFs would overwrite each other:")
        for stem, paths in collisions.items():
            log.warning("  %s: %s", stem, [str(p) for p in paths])
    return collisions


def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    log: logging.Logger,
) -> dict:
    """
    Main orchestrator: finds PDFs, populates shared pdf_queue, spawns
    CPU+GPU workers with work-stealing, enforces timeouts, reports progress
    including GPU backlog depth.
    """
    global GPU_BATCH_SIZE, CPU_WORKERS

    # Auto-tune from GPU VRAM
    gpu_batch, cpu_workers = _auto_tune(log)
    GPU_BATCH_SIZE = gpu_batch
    CPU_WORKERS = cpu_workers

    # Find all PDFs
    pdf_files = sorted(input_dir.rglob("*.pdf")) + sorted(input_dir.rglob("*.PDF"))
    if not pdf_files:
        raise ValueError(f"No PDF files found in {input_dir}")

    log.info("Found %d PDFs — starting work-stealing pipeline", len(pdf_files))

    # Detect stem collisions before starting
    _detect_collisions(pdf_files, log)

    # ── Shared queues ──
    # pdf_queue: work-stealing — all CPU workers pull from here
    pdf_queue = mp.Queue(maxsize=len(pdf_files))
    section_queue = mp.Queue(maxsize=SECTION_QUEUE_SIZE)
    progress_queue = mp.Queue()
    stop_event = mp.Event()

    # Fill pdf_queue with all PDFs
    for pdf in pdf_files:
        pdf_queue.put(pdf)

    # ── Start GPU worker first (so it's ready when sections arrive) ──
    gpu_proc = mp.Process(
        target=_gpu_worker,
        args=(section_queue, output_dir, stop_event, gpu_batch),
        name="GPU-Embedder",
    )
    gpu_proc.start()
    log.info("GPU worker started")

    # ── Start CPU workers ──
    cpu_procs = []
    for i in range(CPU_WORKERS):
        p = mp.Process(
            target=_cpu_worker,
            args=(pdf_queue, section_queue, progress_queue, output_dir),
            name=f"CPU-Parser-{i}",
        )
        p.start()
        cpu_procs.append(p)
    log.info("%d CPU workers started", len(cpu_procs))

    # ── Monitor progress ──
    parsed = 0
    failed = 0
    total = len(pdf_files)
    t0 = time.time()
    last_report = t0

    while any(p.is_alive() for p in cpu_procs):
        # Check if GPU worker died prematurely
        if not gpu_proc.is_alive():
            log.error("GPU worker died unexpectedly (exit code %s) — aborting",
                      gpu_proc.exitcode)
            stop_event.set()
            # Drain pdf_queue so CPU workers exit cleanly
            while not pdf_queue.empty():
                try:
                    pdf_queue.get_nowait()
                except Exception:
                    break
            for _ in cpu_procs:
                try:
                    pdf_queue.put_nowait(None)
                except Exception:
                    break
            break
        try:
            msg, name = progress_queue.get(timeout=1.0)
            if msg == "parsed":
                parsed += 1
            elif msg == "failed":
                failed += 1

            # Report every 5 seconds or on every milestone
            now = time.time()
            if now - last_report >= 5 or (parsed + failed) % 10 == 0:
                elapsed = now - t0
                queue_depth = section_queue.qsize()
                rate = parsed / elapsed if elapsed > 0 else 0
                pct = (parsed + failed) / total * 100
                log.info(
                    "[%d/%d] %.0f%%  parsed=%d  failed=%d  "
                    "GPU_backlog=%d sections  (%.1f PDF/min)  %.0fs",
                    parsed + failed, total, pct, parsed, failed,
                    queue_depth, rate * 60, elapsed,
                )
                last_report = now
        except Exception:
            pass

    # Drain remaining progress messages
    while not progress_queue.empty():
        try:
            msg, name = progress_queue.get_nowait()
            if msg == "parsed":
                parsed += 1
            elif msg == "failed":
                failed += 1
        except Exception:
            break

    # ── Wait for all CPU workers to finish ──
    for p in cpu_procs:
        p.join(timeout=10)

    # ── Signal GPU worker to finish (only if it's still alive) ──
    log.info("All CPU workers done. Signaling GPU worker to flush and exit...")
    if gpu_proc.is_alive():
        section_queue.put(None)  # poison pill
        gpu_proc.join(timeout=120)  # GPU may have backlog to flush

        if gpu_proc.is_alive():
            log.error("GPU worker did not exit within 120s — terminating")
            gpu_proc.terminate()
            gpu_proc.join(timeout=10)

    elapsed = time.time() - t0
    stats = {
        "total_pdfs": total,
        "parsed": parsed,
        "failed": failed,
        "elapsed_s": round(elapsed, 1),
        "pdfs_per_minute": round(parsed / elapsed * 60, 1) if elapsed > 0 else 0,
    }
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="GPU-accelerated PDF parser + embedder")
    ap.add_argument("--input-dir", required=True, help="Directory containing PDFs")
    ap.add_argument("--output-dir", required=True, help="Output directory for .json + .npy")
    ap.add_argument("--workers", type=int, default=CPU_WORKERS, help="CPU parser workers")
    ap.add_argument("--batch-size", type=int, default=GPU_BATCH_SIZE, help="GPU batch size")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log = setup_logging(output_dir)
    log.info("=== GPU pipeline starting ===")
    log.info("Input:  %s", input_dir)
    log.info("Output: %s", output_dir)
    log.info("Workers: %d CPU | batch: %d GPU | trunc: %d chars",
             args.workers, args.batch_size, TEXT_TRUNC_CHARS)

    # Apply CLI overrides
    CPU_WORKERS = args.workers
    GPU_BATCH_SIZE = args.batch_size

    stats = run_pipeline(input_dir, output_dir, log)
    log.info("=== Done: %s ===", json.dumps(stats))

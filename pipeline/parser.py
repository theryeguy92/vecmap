#!/usr/bin/env python3
"""
pipeline/parser.py — DOE Order PDF parser

Extracts structured regulatory content from DOE Order PDFs using:
  - PyMuPDF (fitz)  — PDF text extraction
  - spaCy           — NER for agency names
  - dateparser      — date normalization
  - regex           — citations, obligations, definitions

Note on LexNLP: lexnlp 2.3.0 requires lxml==4.9.1 and old pickled scikit-learn
models that are incompatible with Python 3.12 and scikit-learn >=1.0. The same
extraction categories (citations, obligations, definitions, dates, agencies) are
implemented here with patterns tuned specifically for DOE regulatory text.
"""

import multiprocessing as mp
import re
import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import spacy
import dateparser

PARSE_TIMEOUT = 30  # seconds per PDF before we kill the worker

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOC_DIR  = PROJECT_ROOT / "tests" / "sample_docs" / "doe_orders"
OUT_DIR  = PROJECT_ROOT / "tests" / "sample_docs" / "parsed"
LOG_FILE = OUT_DIR / "parser.log"

# ---------------------------------------------------------------------------
# Compiled regex patterns for DOE regulatory text
# ---------------------------------------------------------------------------

# DOE directive references: DOE O 251.1C, DOE P 226.1B, DOE M 435.1-1
_DOE_REF = re.compile(
    r'\bDOE\s+[OPMGN]\s+[\d]+\.[\d]+[A-Za-z\d.\-]*'
    r'(?:\s+Chg\s*\d+)?(?:\s+\([A-Za-z]+\))?',
    re.IGNORECASE,
)
# CFR: 10 CFR 830, 48 CFR 952.204-2, 10 C.F.R. § 835
_CFR_REF = re.compile(
    r'\b(\d+)\s+C\.?F\.?R\.?\s+(?:[Pp]arts?\s+|§\s*)?(\d+[\d.\-]*)',
)
# USC: 50 U.S.C. Section 2406, 42 USC § 7158
_USC_REF = re.compile(
    r'\b(\d+)\s+U\.?S\.?C\.?\s+(?:[Ss]ec(?:tions?|\.?)\s*|§\s*)?(\d+[\d.\-]*)',
)
# Executive Orders
_EO_REF = re.compile(r'\bExecutive\s+Order\s+(\d+)', re.IGNORECASE)
# Public Law
_PL_REF = re.compile(
    r'\b(?:P\.?L\.?|Public\s+Law)\s+(\d+[-–]\d+)', re.IGNORECASE,
)
# Homeland Security Presidential Directives
_HSPD_REF = re.compile(r'\bHSPD[-\s](\d+)', re.IGNORECASE)

# Obligation sentences: contains "shall" or "must" as a modal
_OBLIGATION_RE = re.compile(
    r'[^.!?]*\b(?:shall|must)\b[^.!?]*[.!?]', re.IGNORECASE,
)
# Definition patterns
_DEF_INLINE  = re.compile(
    r'"([^"]{2,80})"\s+means\s+([^.]{10,300}\.)',
    re.IGNORECASE,
)
_DEF_CAPS = re.compile(
    r'\b([A-Z][A-Z\s]{2,50})\s+means\s+([^.]{10,300}\.)',
)
_DEF_PHRASE = re.compile(
    r'\bthe\s+term\s+"([^"]{2,80})"\s+(?:means|is\s+defined\s+as)\s+([^.]{10,300}\.)',
    re.IGNORECASE,
)

# Page-header noise: "DOE O 151.1D  \n  8-11-2016" repeating at top of each page
_PAGE_HEADER = re.compile(
    r'\n\s*\d*\s*\n?\s*DOE\s+[OPMGN]\s+[\d]+\.[\d]+[A-Za-z\d.\- ]*\n\s*[\d\-]+\s*\n',
    re.IGNORECASE,
)
# Catch the alternate header layout (page num on separate line)
_PAGE_HEADER2 = re.compile(
    r'\n\s*DOE\s+[OPMGN]\s+[\d]+\.[\d]+[A-Za-z\d.\- ]*\s*\n\s*\d+\s*\n\s*[\d\-]+\s*\n',
    re.IGNORECASE,
)

# Section number on its own line: "1.", "2.", ..., "25."
_SEC_NUM_LINE = re.compile(r'^\s*(\d{1,2})\.\s*$')
# Section heading: ALL-CAPS words (2+ chars) followed by period
_SEC_HEADING  = re.compile(r'^([A-Z][A-Z\s,/()\-]{2,80})\.\s*(.*)')

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Citation:
    cit_type:  str   # cfr | doe | usc | eo | pl | hspd
    reference: str   # canonical citation text
    context:   str   # surrounding ~80 chars

@dataclass
class Obligation:
    modal:    str   # "shall" or "must"
    sentence: str

@dataclass
class Definition:
    term:       str
    definition: str

@dataclass
class ParsedSection:
    section_id: str
    heading:    str
    text:       str
    citations:  list = field(default_factory=list)
    obligations: list = field(default_factory=list)
    definitions: list = field(default_factory=list)
    dates:      list = field(default_factory=list)
    agencies:   list = field(default_factory=list)

@dataclass
class ParsedDocument:
    source_file:      str
    doc_id:           str   # "DOE O 151.1D"
    doc_type:         str   # "Order" | "Policy" | "Manual" | "Guide" | "Notice"
    subject:          str
    approved_date:    str
    initiating_office: str
    page_count:       int
    sections:         list = field(default_factory=list)
    all_citations:    list = field(default_factory=list)
    all_obligations:  list = field(default_factory=list)
    all_definitions:  list = field(default_factory=list)

# ---------------------------------------------------------------------------
# NLP setup (module-level to load once)
# ---------------------------------------------------------------------------

_nlp: Optional[spacy.Language] = None

def _get_nlp() -> spacy.Language:
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    return _nlp

# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def _extract_full_text(pdf_path: Path) -> tuple[str, int]:
    """Return (concatenated_text, page_count) from a PDF."""
    doc  = fitz.open(str(pdf_path))
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
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text

# ---------------------------------------------------------------------------
# Header metadata extraction
# ---------------------------------------------------------------------------

def _extract_metadata(text: str) -> dict:
    """Pull doc_id, subject, date, office from document header text."""
    meta = {
        "doc_id":           "",
        "doc_type":         "Order",
        "subject":          "",
        "approved_date":    "",
        "initiating_office": "",
    }

    # Doc ID: "DOE O 151.1D" or "DOE O 151.1D Chg 1"
    m = re.search(r'\bDOE\s+([OPMGN])\s+([\d]+\.[\d]+[A-Za-z\d.\-]*(?:\s+Chg\s*\d+)?)', text[:500])
    if m:
        type_map = {"O": "Order", "P": "Policy", "M": "Manual",
                    "G": "Guide",  "N": "Notice"}
        meta["doc_id"]   = f"DOE {m.group(1)} {m.group(2).strip()}"
        meta["doc_type"] = type_map.get(m.group(1), "Order")

    # Subject line
    m = re.search(r'SUBJECT[:\s]+([^\n]{5,200})', text[:1500], re.IGNORECASE)
    if m:
        meta["subject"] = m.group(1).strip()

    # Approval date
    m = re.search(r'Approved\s*[:\s]+([^\n]{5,30})', text[:800], re.IGNORECASE)
    if m:
        raw = m.group(1).strip()
        parsed = dateparser.parse(raw, settings={"RETURN_AS_TIMEZONE_AWARE": False})
        meta["approved_date"] = parsed.strftime("%Y-%m-%d") if parsed else raw

    # Initiating office
    m = re.search(r'INITIATED\s+BY[:\s]+([^\n]{5,100})', text[:1000], re.IGNORECASE)
    if m:
        meta["initiating_office"] = m.group(1).strip()

    return meta

# ---------------------------------------------------------------------------
# Section splitting
# ---------------------------------------------------------------------------

def _split_sections(text: str) -> list[dict]:
    """
    Split DOE Order body text into numbered top-level sections.

    DOE Orders use two header formats:
      Format A:  "1.\n\nPURPOSE. text..."   (number then heading on next line)
      Format B:  "1. PURPOSE. text..."       (all on same line)
    """
    lines = text.splitlines()
    sections: list[dict] = []
    cur_num: Optional[str] = None
    cur_heading: Optional[str] = None
    cur_lines: list[str] = []

    def _flush():
        if cur_num and cur_heading:
            sections.append({
                "id":      cur_num,
                "heading": cur_heading,
                "text":    "\n".join(cur_lines).strip(),
            })

    i = 0
    while i < len(lines):
        stripped = lines[i].strip()

        # Format B: "1. PURPOSE. text" (number + heading on same line)
        m = re.match(r'^(\d{1,2})\.\s+([A-Z][A-Z ,/()\-]{2,80})\.\s*(.*)', stripped)
        if m:
            _flush()
            cur_num     = m.group(1)
            cur_heading = m.group(2).strip()
            cur_lines   = [m.group(3)] if m.group(3).strip() else []
            i += 1
            continue

        # Format A: bare section number on its own line
        m_num = _SEC_NUM_LINE.match(stripped)
        if m_num:
            _flush()
            cur_num     = m_num.group(1)
            cur_heading = None
            cur_lines   = []
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
        sections = [{"id": "0", "heading": "FULL_TEXT", "text": text.strip()}]

    return sections

# ---------------------------------------------------------------------------
# NLP extractors
# ---------------------------------------------------------------------------

def _extract_citations(text: str) -> list[Citation]:
    cits: list[Citation] = []

    def _ctx(m):
        s, e = max(0, m.start()-40), min(len(text), m.end()+40)
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
        modal = "shall" if re.search(r'\bshall\b', sent, re.IGNORECASE) else "must"
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
    date_re = re.compile(
        r'\b(?:'
        r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}'               # 8-11-2016
        r'|(?:January|February|March|April|May|June|'
        r'July|August|September|October|November|December)'
        r'\s+\d{1,2},?\s+\d{4}'                          # January 1, 2020
        r')',
        re.IGNORECASE,
    )
    dates: list[str] = []
    seen:  set[str]  = set()
    for m in date_re.finditer(text):
        raw = m.group().strip()
        if raw in seen:
            continue
        seen.add(raw)
        parsed = dateparser.parse(raw, settings={"RETURN_AS_TIMEZONE_AWARE": False})
        dates.append(parsed.strftime("%Y-%m-%d") if parsed else raw)
    return dates


def _extract_agencies(text: str) -> list[str]:
    """Extract government agency names via spaCy NER + DOE acronym list."""
    DOE_AGENCIES = {
        "DOE", "NNSA", "DOT", "NRC", "EPA", "OSHA", "OMB", "DOD",
        "NASA", "GAO", "DHS", "FBI", "CIA", "NSA", "FEMA", "NIST",
    }
    nlp     = _get_nlp()
    # Truncate to 100k chars to keep spaCy fast
    doc     = nlp(text[:100_000])
    names:  set[str] = set()

    for ent in doc.ents:
        if ent.label_ in ("ORG", "GPE"):
            name = ent.text.strip()
            if 3 <= len(name) <= 80:
                names.add(name)

    # Always include known DOE acronyms found in text
    for acro in DOE_AGENCIES:
        if re.search(rf'\b{acro}\b', text):
            names.add(acro)

    return sorted(names)

# ---------------------------------------------------------------------------
# Top-level parse function
# ---------------------------------------------------------------------------

def parse_pdf(pdf_path: Path, log: logging.Logger) -> Optional[ParsedDocument]:
    """Parse a single DOE Order PDF into a structured ParsedDocument."""
    try:
        raw_text, page_count = _extract_full_text(pdf_path)
    except Exception as exc:
        log.error(f"fitz failed on {pdf_path.name}: {exc}")
        return None

    if len(raw_text.strip()) < 100:
        log.warning(f"Nearly empty text in {pdf_path.name} — likely scanned/image PDF")

    clean = _clean_text(raw_text)
    meta  = _extract_metadata(clean)

    # Extract NLP features from the full document
    all_cits  = _extract_citations(clean)
    all_obligs = _extract_obligations(clean)
    all_defs  = _extract_definitions(clean)
    agencies  = _extract_agencies(clean)

    # Split into sections and annotate each
    raw_sections = _split_sections(clean)
    parsed_sections = []
    for sec in raw_sections:
        text = sec["text"]
        psec = ParsedSection(
            section_id  = sec["id"],
            heading     = sec["heading"],
            text        = text,
            citations   = [asdict(c) for c in _extract_citations(text)],
            obligations = [asdict(o) for o in _extract_obligations(text)],
            definitions = [asdict(d) for d in _extract_definitions(text)],
            dates       = _extract_dates(text),
            agencies    = agencies,  # shared across sections (doc-level)
        )
        parsed_sections.append(psec)

    return ParsedDocument(
        source_file       = pdf_path.name,
        doc_id            = meta["doc_id"],
        doc_type          = meta["doc_type"],
        subject           = meta["subject"],
        approved_date     = meta["approved_date"],
        initiating_office = meta["initiating_office"],
        page_count        = page_count,
        sections          = [asdict(s) for s in parsed_sections],
        all_citations     = [asdict(c) for c in all_cits],
        all_obligations   = [asdict(o) for o in all_obligs[:50]],  # cap for JSON size
        all_definitions   = [asdict(d) for d in all_defs],
    )


def save_parsed(doc: ParsedDocument, out_dir: Path) -> Path:
    """Write ParsedDocument to JSON in out_dir."""
    stem     = Path(doc.source_file).stem
    out_path = out_dir / f"{stem}.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(asdict(doc), fh, indent=2, ensure_ascii=False)
    return out_path


# ---------------------------------------------------------------------------
# Per-file timeout via subprocess
# ---------------------------------------------------------------------------

def _parse_worker(pdf_path_str: str, result_queue: mp.Queue) -> None:
    """Worker process: parse one PDF and put the result on the queue."""
    # Silence duplicate log output — main process handles all user-visible logging
    logging.disable(logging.CRITICAL)
    try:
        doc = parse_pdf(Path(pdf_path_str), logging.getLogger("worker"))
        result_queue.put(doc)
    except Exception:
        result_queue.put(None)


def parse_with_timeout(
    pdf_path: Path,
    log: logging.Logger,
    timeout: int = PARSE_TIMEOUT,
) -> Optional[ParsedDocument]:
    """Run parse_pdf in a child process; kill it and return None if it hangs."""
    q: "mp.Queue[Optional[ParsedDocument]]" = mp.Queue()
    p = mp.Process(target=_parse_worker, args=(str(pdf_path), q), daemon=True)
    p.start()
    p.join(timeout)

    if p.is_alive():
        log.warning(f"TIMEOUT ({timeout}s) — killing {pdf_path.name}")
        p.terminate()
        p.join(5)
        if p.is_alive():
            p.kill()
            p.join()
        return None

    return q.get_nowait() if not q.empty() else None

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(doc: ParsedDocument) -> None:
    print(f"\n{'='*60}")
    print(f"File     : {doc.source_file}")
    print(f"Doc ID   : {doc.doc_id}  ({doc.doc_type})")
    print(f"Subject  : {doc.subject[:70]}")
    print(f"Approved : {doc.approved_date}")
    print(f"Office   : {doc.initiating_office[:60]}")
    print(f"Pages    : {doc.page_count}")
    print(f"Sections : {len(doc.sections)}")
    print(f"Citations: {len(doc.all_citations)}")
    print(f"Obligs   : {len(doc.all_obligations)}")
    print(f"Defs     : {len(doc.all_definitions)}")
    print(f"Agencies : {len(doc.sections[0]['agencies']) if doc.sections else 0}")

    print("\n-- Sections --")
    for s in doc.sections:
        print(f"  [{s['section_id']}] {s['heading']:<30}  "
              f"oblig={len(s['obligations'])}  cit={len(s['citations'])}")

    if doc.all_citations:
        print("\n-- Sample citations (first 5) --")
        for c in doc.all_citations[:5]:
            print(f"  [{c['cit_type']:6}] {c['reference']}")

    if doc.all_obligations:
        print("\n-- Sample obligations (first 3) --")
        for o in doc.all_obligations[:3]:
            print(f"  [{o['modal']:5}] {o['sentence'][:100]}...")

    if doc.all_definitions:
        print("\n-- Sample definitions (first 3) --")
        for d in doc.all_definitions[:3]:
            print(f"  {d['term']!r:30} => {d['definition'][:60]}...")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def setup_logging() -> logging.Logger:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)-8s %(message)s"
    logging.basicConfig(
        level=logging.INFO, format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
        ],
    )
    return logging.getLogger("parser")


def main(test_only: bool = False) -> None:
    log = setup_logging()
    log.info("=== regmap parser starting ===")
    log.info(f"Input : {DOC_DIR}")
    log.info(f"Output: {OUT_DIR}")

    pdfs = sorted(DOC_DIR.glob("*.pdf")) + sorted(DOC_DIR.glob("*.PDF"))
    if not pdfs:
        log.error(f"No PDFs found in {DOC_DIR}")
        sys.exit(1)

    if test_only:
        pdfs = pdfs[:5]
        log.info(f"TEST MODE: processing {len(pdfs)} PDFs")
    else:
        log.info(f"Processing {len(pdfs)} PDFs")

    ok = skip = fail = 0
    for i, pdf in enumerate(pdfs, 1):
        out_path = OUT_DIR / f"{pdf.stem}.json"
        if out_path.exists() and not test_only:
            skip += 1
            continue

        log.info(f"[{i}/{len(pdfs)}] {pdf.name}")
        doc = parse_with_timeout(pdf, log)
        if doc is None:
            fail += 1
            continue

        save_parsed(doc, OUT_DIR)
        ok += 1

        if test_only:
            print_summary(doc)

    log.info(f"=== Done: ok={ok}  skip={skip}  fail={fail} ===")


if __name__ == "__main__":
    test_mode = "--test" not in sys.argv[1:] or "--test" in sys.argv[1:]
    # Default to test mode when run directly; pass --all for full run
    full = "--all" in sys.argv[1:]
    main(test_only=not full)

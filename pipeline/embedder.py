#!/usr/bin/env python3
"""
pipeline/embedder.py — Section embeddings for regmap compliance mapping

Uses nlpaueb/legal-bert-base-uncased to embed each parsed section into a
768-dim float32 vector suitable for the CUDA cosine similarity kernels.

Output per document:
  {stem}_sections.npy   — float32 array [num_sections, 768]
  {stem}_meta.json      — section metadata (id, heading, text preview)

The numpy arrays are directly consumable by the CUDA cosine_warp kernel as:
  h_A[num_regs   × 768]  — regulation section embeddings
  h_B[num_procs  × 768]  — procedure section embeddings
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARSED_DIR = PROJECT_ROOT / "tests" / "sample_docs" / "parsed"
EMB_DIR = PARSED_DIR  # embeddings live alongside parsed JSON
LOG_FILE = EMB_DIR / "embedder.log"

MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
EMBED_DIM = 768
MAX_TOKENS = 512  # BERT limit
BATCH_SIZE = 8  # sections per GPU forward pass

# ---------------------------------------------------------------------------
# Model loading (once per process)
# ---------------------------------------------------------------------------

_tokenizer = None
_model = None
_device = None


def load_model() -> tuple:
    """Load legal-bert tokenizer and model, return (tokenizer, model, device)."""
    global _tokenizer, _model, _device

    if _model is not None:
        return _tokenizer, _model, _device

    log = logging.getLogger("embedder")
    log.info(f"Loading model: {MODEL_NAME}")

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _model = AutoModel.from_pretrained(MODEL_NAME)
    _model.eval()

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model.to(_device)
    log.info(f"Model on: {_device}  (embed_dim={EMBED_DIM})")

    return _tokenizer, _model, _device


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token embeddings, ignoring padding tokens."""
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def embed_texts(texts: list[str], tokenizer, model, device) -> np.ndarray:
    """
    Embed a list of text strings.
    Returns float32 ndarray of shape [len(texts), 768].
    Texts longer than MAX_TOKENS are truncated — acceptable for compliance
    mapping because requirements are typically stated in the first few sentences.
    """
    all_vecs: list[np.ndarray] = []

    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_TOKENS,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc)

        vecs = _mean_pool(out.last_hidden_state, enc["attention_mask"])
        all_vecs.append(vecs.cpu().float().numpy())

    return np.vstack(all_vecs)  # [N, 768]


# ---------------------------------------------------------------------------
# Per-document embedding
# ---------------------------------------------------------------------------


def embed_document(
    json_path: Path,
    tokenizer,
    model,
    device,
    log: logging.Logger,
) -> tuple[np.ndarray, list[dict]]:
    """
    Load a parsed JSON document, embed each section.
    Returns (embeddings [N, 768], metadata list).
    """
    with open(json_path, encoding="utf-8") as fh:
        doc = json.load(fh)

    sections = doc.get("sections", [])
    if not sections:
        log.warning(f"No sections in {json_path.name}")
        return np.zeros((0, EMBED_DIM), dtype=np.float32), []

    # Build text inputs: combine heading + section text
    texts = []
    meta = []
    for sec in sections:
        heading = sec.get("heading", "")
        text = sec.get("text", "")
        # Prepend the document subject and section heading for context
        full = f"{doc.get('subject', '')} | {heading} | {text}"
        texts.append(full)
        meta.append(
            {
                "doc_id": doc.get("doc_id", ""),
                "doc_type": doc.get("doc_type", ""),
                "subject": doc.get("subject", ""),
                "section_id": sec.get("section_id", ""),
                "heading": heading,
                "text_preview": text[:200].replace("\n", " "),
                "obligations": len(sec.get("obligations", [])),
                "citations": len(sec.get("citations", [])),
            }
        )

    embeddings = embed_texts(texts, tokenizer, model, device)
    log.info(
        f"  {json_path.name}: {len(sections)} sections → " f"shape {embeddings.shape}"
    )
    return embeddings, meta


def save_embeddings(stem: str, embeddings: np.ndarray, meta: list[dict]) -> None:
    """Save .npy array and metadata JSON alongside the parsed JSON."""
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    npy_path = EMB_DIR / f"{stem}_sections.npy"
    meta_path = EMB_DIR / f"{stem}_meta.json"

    np.save(str(npy_path), embeddings)
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Summary / diagnostics
# ---------------------------------------------------------------------------


def print_embedding_summary(
    stem: str, embeddings: np.ndarray, meta: list[dict]
) -> None:
    print(f"\n{'='*60}")
    print(f"Document : {stem}")
    print(f"Shape    : {embeddings.shape}  dtype={embeddings.dtype}")
    norms = np.linalg.norm(embeddings, axis=1)
    print(
        f"L2 norms : min={norms.min():.4f}  max={norms.max():.4f}  "
        f"mean={norms.mean():.4f}"
    )

    print("\n-- Section embeddings --")
    print(f"  {'ID':<5} {'Heading':<30} {'Norm':>8}  Preview")
    print(f"  {'-'*5} {'-'*30} {'-'*8}  {'-'*40}")
    for i, (m, vec) in enumerate(zip(meta, embeddings)):
        norm = float(np.linalg.norm(vec))
        preview = m["text_preview"][:40].replace("\n", " ")
        print(f"  {m['section_id']:<5} {m['heading']:<30} {norm:>8.4f}  {preview}")

    # Cosine similarity matrix (small — sections × sections)
    if len(embeddings) > 1:
        nrm = embeddings / (norms[:, None] + 1e-9)
        sim = nrm @ nrm.T
        np.fill_diagonal(sim, 0)
        idx = np.unravel_index(sim.argmax(), sim.shape)
        print(
            f"\nMost similar section pair: "
            f"[{meta[idx[0]]['heading']}] ↔ [{meta[idx[1]]['heading']}]  "
            f"sim={sim[idx]:.4f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def setup_logging() -> logging.Logger:
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)-8s %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
        ],
    )
    return logging.getLogger("embedder")


def main(test_only: bool = True) -> None:
    log = setup_logging()
    log.info("=== regmap embedder starting ===")
    log.info(f"Model  : {MODEL_NAME}")
    log.info(f"Input  : {PARSED_DIR}")
    log.info(f"Device : {'cuda' if torch.cuda.is_available() else 'cpu'}")

    tokenizer, model, device = load_model()

    json_files = sorted(PARSED_DIR.glob("*.json"))
    # Exclude log and meta files
    json_files = [
        f
        for f in json_files
        if not f.name.endswith("_meta.json")
        and f.name not in ("parser.log", "embedder.log")
        and not f.name.endswith(".log")
    ]

    if not json_files:
        log.error(f"No parsed JSON files found in {PARSED_DIR}")
        sys.exit(1)

    if test_only:
        json_files = json_files[:5]
        log.info(f"TEST MODE: embedding {len(json_files)} documents")
    else:
        log.info(f"Embedding {len(json_files)} documents")

    ok = skip = fail = 0
    for i, jf in enumerate(json_files, 1):
        stem = jf.stem
        npy_path = EMB_DIR / f"{stem}_sections.npy"

        if npy_path.exists() and not test_only:
            skip += 1
            continue

        log.info(f"[{i}/{len(json_files)}] {jf.name}")
        try:
            embeddings, meta = embed_document(jf, tokenizer, model, device, log)
        except Exception as exc:
            log.error(f"  FAIL {jf.name}: {exc}")
            fail += 1
            continue

        if embeddings.shape[0] == 0:
            fail += 1
            continue

        save_embeddings(stem, embeddings, meta)
        ok += 1

        if test_only:
            print_embedding_summary(stem, embeddings, meta)

    log.info(f"=== Done: ok={ok}  skip={skip}  fail={fail} ===")

    # Print stats on all .npy files produced
    npys = sorted(EMB_DIR.glob("*_sections.npy"))
    if npys:
        total_sections = sum(np.load(str(p)).shape[0] for p in npys)
        total_mb = sum(p.stat().st_size for p in npys) / 1_048_576
        log.info(
            f"Embedding files: {len(npys)}  "
            f"total sections: {total_sections}  "
            f"disk: {total_mb:.1f} MB"
        )


if __name__ == "__main__":
    full = "--all" in sys.argv[1:]
    main(test_only=not full)

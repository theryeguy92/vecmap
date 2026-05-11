#!/usr/bin/env python3
"""
DOE Order scraper for regmap compliance mapping pipeline.

Primary path:   Plone REST API at /++api++/@search  (no Selenium needed)
Fallback path:  Selenium + Chrome headless

Discovery notes (probed 2026-05-05):
  - Site is Plone 6 + Volto (React SSR), confirmed by window.env.apiPath
  - /++api++/ traverser exposes Plone REST API publicly
  - @search returns paginated JSON with portal_type filter
  - DOE Orders have 'order' (case-insensitive) in their URL slug,
    e.g. '0151-1-border-d', '0000.1-BOrder-a'
  - PDF download: {item_url}/@@download/file
  - Content-Disposition header provides the canonical filename
"""

import os
import re
import sys
import time
import logging
import urllib.parse
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# Selenium imports — only used in fallback path
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.common.by import By
    from webdriver_manager.chrome import ChromeDriverManager
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL    = "https://www.directives.doe.gov"
BROWSE_URL  = f"{BASE_URL}/directives-browse"
API_URL     = f"{BASE_URL}/++api++/@search"

USER_AGENT  = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36 "
    "regmap-scraper/1.0"
)

RATE_LIMIT_SECS   = 1.5   # between every download
API_PAUSE_SECS    = 0.75  # between API pagination calls
PAGE_SIZE         = 100   # Plone search batch size
REQUEST_TIMEOUT   = 60    # seconds

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR      = PROJECT_ROOT / "tests" / "sample_docs" / "doe_orders"
LOG_FILE     = OUT_DIR / "scraper.log"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging() -> logging.Logger:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)-8s %(message)s"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)
    return logging.getLogger("scraper")


# ---------------------------------------------------------------------------
# Primary path: Plone REST API
# ---------------------------------------------------------------------------

def probe_api(session: requests.Session, log: logging.Logger) -> bool:
    """Return True if the Plone REST API is reachable and returns items."""
    try:
        r = session.get(
            API_URL,
            params={"b_size": 1, "portal_type": "DOE.Directives.directive"},
            timeout=15,
        )
        if r.status_code == 200:
            data = r.json()
            if data.get("items_total", 0) > 0:
                log.info(
                    f"Plone REST API available — {data['items_total']} directives total"
                )
                return True
    except Exception as e:
        log.warning(f"API probe failed: {e}")
    return False


def fetch_all_directives(session: requests.Session, log: logging.Logger) -> list[dict]:
    """Page through @search and return all published directive records."""
    params = {
        "portal_type": "DOE.Directives.directive",
        "review_state": "published",
        "b_size": PAGE_SIZE,
        "b_start": 0,
    }
    items: list[dict] = []
    while True:
        r = session.get(API_URL, params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        batch = data.get("items", [])
        items.extend(batch)
        total = data.get("items_total", 0)
        log.info(f"  Fetched {len(items)}/{total} directives")
        if len(items) >= total or not batch:
            break
        params["b_start"] += PAGE_SIZE
        time.sleep(API_PAUSE_SECS)
    return items


def is_doe_order(item: dict) -> bool:
    """
    Identify DOE Orders by URL slug.

    URL slug patterns observed on the live site:
      Orders  : '0151-1-border-d', '0000.1-BOrder-a', '0473.5-border'
      Guides  : '0151.1-EGuide-2', '0414.1-1bguide'
      Notices : '0135.01-CNotice'
      Policies: '0203-1-apolicy', '0444.1-apolicy'
      Manuals : '5632-1-dmanual-c-chg1'
    The type word is always embedded in the slug; filtering 'order'
    (case-insensitive) and excluding 'certification' is sufficient.
    """
    slug = item.get("@id", "").split("/")[-1].lower()
    return "order" in slug and "certification" not in slug


def filename_from_response(resp: requests.Response, slug: str) -> str:
    """Extract canonical PDF filename from Content-Disposition or fall back to slug."""
    cd = resp.headers.get("content-disposition", "")
    # filename*=UTF-8''DOE%20O%20151.1D%20Chg%201%20MinChg.pdf
    name = None
    m = re.search(r"filename\*=UTF-8''(.+)", cd, re.IGNORECASE)
    if m:
        name = urllib.parse.unquote(m.group(1).strip())
    else:
        m = re.search(r'filename="?([^";]+)"?', cd, re.IGNORECASE)
        if m:
            name = m.group(1).strip()
    if not name:
        name = f"{slug}.pdf"
    # Always ensure a .pdf extension for PDF content
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    return name


def download_order(
    session: requests.Session,
    item: dict,
    out_dir: Path,
    log: logging.Logger,
) -> tuple[bool, str]:
    """
    Download one DOE Order PDF.
    Returns (success, reason_string).
    """
    item_url = item["@id"]
    slug     = item_url.split("/")[-1]
    dl_url   = f"{item_url}/@@download/file"

    # Stream the response first to get the real filename from headers
    try:
        resp = session.get(dl_url, timeout=REQUEST_TIMEOUT, stream=True)
    except requests.RequestException as exc:
        log.error(f"FAIL  {slug}: {exc}")
        return False, str(exc)

    if resp.status_code == 404:
        log.warning(f"MISS  {slug} — 404 (no PDF attached)")
        return False, "404"
    if resp.status_code != 200:
        log.warning(f"FAIL  {slug} — HTTP {resp.status_code}")
        return False, f"HTTP {resp.status_code}"

    content_type = resp.headers.get("content-type", "")
    if "pdf" not in content_type and "octet-stream" not in content_type:
        log.warning(f"SKIP  {slug} — unexpected content-type: {content_type}")
        resp.close()
        return False, f"content-type:{content_type}"

    filename = filename_from_response(resp, slug)
    # Sanitize: strip characters unsafe in filenames
    filename = re.sub(r'[<>:"/\\|?*]', "_", filename)
    out_path = out_dir / filename

    if out_path.exists():
        resp.close()
        log.info(f"SKIP  {filename} (already downloaded)")
        return True, "exists"

    size = 0
    try:
        with open(out_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=65_536):
                fh.write(chunk)
                size += len(chunk)
    except OSError as exc:
        log.error(f"FAIL  {filename}: write error — {exc}")
        out_path.unlink(missing_ok=True)
        return False, str(exc)

    log.info(f"OK    {filename} ({size // 1024} KB)")
    return True, f"{size} bytes"


# ---------------------------------------------------------------------------
# Fallback path: Selenium
# ---------------------------------------------------------------------------

def selenium_get_order_urls(log: logging.Logger) -> list[str]:
    """
    Render the directives-browse page with Chrome headless and scrape PDF links.
    Used only when the Plone REST API is unavailable.
    """
    if not SELENIUM_AVAILABLE:
        log.error("Selenium not installed — cannot use fallback")
        return []

    log.info("Selenium fallback: launching Chrome headless")
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument(f"--user-agent={USER_AGENT}")

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=opts,
    )
    pdf_urls: list[str] = []
    try:
        log.info(f"Loading {BROWSE_URL}")
        driver.get(BROWSE_URL)
        # Wait for the listing to render
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "a[href*='@@download']"))
        )
        html  = driver.page_source
        soup  = BeautifulSoup(html, "html.parser")
        links = soup.find_all("a", href=re.compile(r"@@download"))
        for link in links:
            href = link["href"]
            if not href.startswith("http"):
                href = BASE_URL + href
            # Keep only Order PDFs
            slug = href.split("/")[-2].lower() if "@@download" in href else ""
            if "order" in slug and "certification" not in slug:
                pdf_urls.append(href)
        log.info(f"Selenium found {len(pdf_urls)} Order PDF links")
    except Exception as exc:
        log.error(f"Selenium error: {exc}")
    finally:
        driver.quit()
    return pdf_urls


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log = setup_logging()
    log.info("=" * 60)
    log.info("regmap DOE Order scraper")
    log.info(f"Output directory: {OUT_DIR}")
    log.info("=" * 60)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})

    # ------------------------------------------------------------------ #
    # Step 1: Discover source — API first, Selenium fallback              #
    # ------------------------------------------------------------------ #
    log.info("Step 1: Checking for Plone REST API at /++api++/@search")

    use_api = probe_api(session, log)

    if use_api:
        # ---------------------------------------------------------------- #
        # Step 2 (API): Fetch all directive metadata, filter for Orders    #
        # ---------------------------------------------------------------- #
        log.info("Step 2: Fetching all published directives via API")
        all_items = fetch_all_directives(session, log)
        log.info(f"Total directives fetched: {len(all_items)}")

        orders = [item for item in all_items if is_doe_order(item)]
        log.info(f"DOE Orders identified:    {len(orders)}")

        # ---------------------------------------------------------------- #
        # Step 3 (API): Download each Order PDF                            #
        # ---------------------------------------------------------------- #
        log.info(f"Step 3: Downloading {len(orders)} PDFs to {OUT_DIR}")
        session.headers.update({"Accept": "*/*"})

        ok = skip = fail = 0
        for n, item in enumerate(orders, 1):
            success, reason = download_order(session, item, OUT_DIR, log)
            if reason == "exists":
                skip += 1
            elif success:
                ok += 1
            else:
                fail += 1
            if n % 20 == 0:
                log.info(
                    f"Progress {n}/{len(orders)} — "
                    f"ok={ok}  skip={skip}  fail={fail}"
                )
            time.sleep(RATE_LIMIT_SECS)

    else:
        # ---------------------------------------------------------------- #
        # Step 2/3 (Selenium): Render page and download                    #
        # ---------------------------------------------------------------- #
        log.info("Step 2: API unavailable — using Selenium fallback")
        pdf_urls = selenium_get_order_urls(log)
        if not pdf_urls:
            log.error("No PDFs found via Selenium either — aborting")
            sys.exit(1)

        log.info(f"Step 3: Downloading {len(pdf_urls)} PDFs via Selenium-discovered URLs")
        session.headers.update({"Accept": "*/*"})
        ok = skip = fail = 0
        for n, url in enumerate(pdf_urls, 1):
            slug    = url.split("/")[-2]
            item    = {"@id": url.replace("/@@download/file", "")}
            success, reason = download_order(session, item, OUT_DIR, log)
            if reason == "exists":
                skip += 1
            elif success:
                ok += 1
            else:
                fail += 1
            if n % 20 == 0:
                log.info(f"Progress {n}/{len(pdf_urls)} — ok={ok} skip={skip} fail={fail}")
            time.sleep(RATE_LIMIT_SECS)

    # ------------------------------------------------------------------ #
    # Final summary                                                        #
    # ------------------------------------------------------------------ #
    pdfs       = sorted(OUT_DIR.glob("*.pdf"))
    total_mb   = sum(p.stat().st_size for p in pdfs) / 1_048_576
    log.info("=" * 60)
    log.info(f"Downloaded : {ok}")
    log.info(f"Skipped    : {skip}  (already on disk)")
    log.info(f"Failed     : {fail}")
    log.info(f"Total PDFs : {len(pdfs)} files, {total_mb:.1f} MB")
    log.info(f"Log file   : {LOG_FILE}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()

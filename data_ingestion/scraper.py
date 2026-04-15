"""
scraper.py — Fetch university web pages via Jina Reader and BeautifulSoup.

Each scraped page is saved as a dict with keys:
    url, faculty, doc_type, text, last_updated
"""

import re
import time
from datetime import date
from typing import Optional

import requests
from bs4 import BeautifulSoup

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import SCRAPE_TARGETS, JINA_READER_BASE, UNIVERSITY_BASE_URL

# ── Jina Reader fetch ──────────────────────────────────────────────────────────

def fetch_via_jina(url: str, timeout: int = 30) -> Optional[str]:
    """Return clean markdown text from Jina Reader, or None on failure."""
    jina_url = f"{JINA_READER_BASE}{url}"
    try:
        resp = requests.get(jina_url, timeout=timeout, headers={"Accept": "text/plain"})
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        print(f"[scraper] Jina Reader failed for {url}: {e}")
        return None


def fetch_via_requests(url: str, timeout: int = 20) -> Optional[str]:
    """Fallback: fetch raw HTML and extract text with BeautifulSoup."""
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        # Remove nav/footer/script noise
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)
    except requests.RequestException as e:
        print(f"[scraper] Requests fallback failed for {url}: {e}")
        return None


# ── Faculty / doc_type inference ───────────────────────────────────────────────

FACULTY_KEYWORDS: dict[str, list[str]] = {
    "CS": ["информатик", "computer science", "программирован", "software"],
    "Engineering": ["инженер", "engineering", "техник"],
    "Economics": ["экономик", "economics", "финанс", "finance"],
    "Law": ["юридическ", "law", "право"],
    "Business": ["бизнес", "business", "менеджмент", "management"],
    "Education": ["педагогик", "education", "teaching"],
    "Design": ["дизайн", "design", "media"],
    "General": [],
}

DOC_TYPE_KEYWORDS: dict[str, list[str]] = {
    "admissions": ["поступлен", "admission", "вступительн", "приём"],
    "tuition": ["стоимост", "tuition", "оплат", "fee", "цена"],
    "syllabus": ["syllabus", "силлабус", "учебный план", "curriculum"],
    "program": ["программ", "специальност", "major", "факультет", "faculty"],
    "general": [],
}


def _infer(text_lower: str, keyword_map: dict[str, list[str]]) -> str:
    for label, keywords in keyword_map.items():
        if any(kw in text_lower for kw in keywords):
            return label
    return list(keyword_map.keys())[-1]  # last key = fallback


# ── ORT threshold extraction ───────────────────────────────────────────────────

ORT_PATTERN = re.compile(
    r"([\w\s\-]+?)\s*[:\-–]\s*(\d{3})\s*(?:баллов|points|балл)?",
    re.IGNORECASE,
)


def extract_ort_thresholds(text: str) -> dict[str, int]:
    """Parse ORT minimum score table patterns from page text."""
    thresholds: dict[str, int] = {}
    for match in ORT_PATTERN.finditer(text):
        program = match.group(1).strip()
        score = int(match.group(2))
        if 80 <= score <= 260:   # ORT scores are in this range
            thresholds[program] = score
    return thresholds


# ── Main scraper ───────────────────────────────────────────────────────────────

def scrape_all(
    targets: list[str] = SCRAPE_TARGETS,
    delay: float = 1.5,
) -> tuple[list[dict], dict[str, int]]:
    """
    Scrape all target URLs.

    Returns:
        pages: list of page dicts ready for chunking
        ort_thresholds: aggregated {program: min_score} dict
    """
    pages: list[dict] = []
    ort_thresholds: dict[str, int] = {}
    today = str(date.today())

    for url in targets:
        print(f"[scraper] Fetching: {url}")
        text = fetch_via_jina(url) or fetch_via_requests(url)
        if not text:
            print(f"[scraper] Skipping (no content): {url}")
            continue

        text_lower = text.lower()
        faculty = _infer(text_lower, FACULTY_KEYWORDS)
        doc_type = _infer(text_lower, DOC_TYPE_KEYWORDS)

        pages.append({
            "url": url,
            "faculty": faculty,
            "doc_type": doc_type,
            "text": text,
            "last_updated": today,
        })

        # Opportunistically extract ORT thresholds from any page
        found = extract_ort_thresholds(text)
        ort_thresholds.update(found)

        time.sleep(delay)

    print(f"[scraper] Done. {len(pages)} pages scraped.")
    return pages, ort_thresholds


if __name__ == "__main__":
    pages, ort = scrape_all()
    for p in pages:
        print(f"  {p['doc_type']:12s} | {p['faculty']:12s} | {p['url']}")
    print(f"\nORT thresholds found: {ort}")

"""
pdf_extractor.py — Extract text from local syllabus PDFs using PyPDFLoader.

Each PDF is returned as a list of page dicts compatible with the chunker.
"""

import os
import sys
from datetime import date
from pathlib import Path
from typing import Optional

import yaml
from langchain_community.document_loaders import PyPDFLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
with open(_cfg_path) as _f:
    _cfg = yaml.safe_load(_f)

_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
PDF_DIR = str(_PROJECT_ROOT / _cfg["data"]["raw_pdfs"].lstrip("./"))


def _infer_faculty_from_filename(filename: str) -> str:
    name = filename.lower()
    mapping = {
        "engineering": "Факультет инженерии и информатики",
        "informatics": "Факультет инженерии и информатики",
        "инженер": "Факультет инженерии и информатики",
        "cs": "Факультет инженерии и информатики",
        "computer": "Факультет инженерии и информатики",
        "econ": "Факультет экономики и управления",
        "economics": "Факультет экономики и управления",
        "экономик": "Факультет экономики и управления",
        "law": "Факультет экономики и управления",
        "медицин": "Медицинский факультет",
        "medicine": "Медицинский факультет",
        "medical": "Медицинский факультет",
        "social": "Факультет социальных наук",
        "социал": "Факультет социальных наук",
        "human": "Факультет гуманитарных наук",
        "гуманит": "Факультет гуманитарных наук",
        "lingv": "Факультет гуманитарных наук",
        "педагог": "Факультет гуманитарных наук",
    }
    for keyword, faculty in mapping.items():
        if keyword in name:
            return faculty
    return "General"


def extract_pdf(pdf_path: str, faculty: Optional[str] = None) -> list[dict]:
    """Load a single PDF and return one dict per page with metadata."""
    path = Path(pdf_path)
    if not path.exists():
        print(f"[pdf] File not found: {pdf_path}")
        return []

    inferred_faculty = faculty or _infer_faculty_from_filename(path.name)
    today = str(date.today())

    try:
        loader = PyPDFLoader(str(path))
        docs = loader.load()
    except Exception as e:
        print(f"[pdf] Failed to load {pdf_path}: {e}")
        return []

    pages = []
    for doc in docs:
        text = doc.page_content.strip()
        if len(text) < 50:
            continue
        pages.append({
            "text": text,
            "faculty": inferred_faculty,
            "doc_type": "syllabus",
            "source_file": path.name,
            "page": doc.metadata.get("page", 0),
            "last_updated": today,
        })

    print(f"[pdf] {path.name}: {len(pages)} pages extracted (faculty={inferred_faculty})")
    return pages


def extract_all_pdfs(pdf_dir: str = PDF_DIR) -> list[dict]:
    """Scan pdf_dir for all *.pdf files and extract them all."""
    pdf_dir_path = Path(pdf_dir)
    if not pdf_dir_path.exists():
        print(f"[pdf] PDF directory not found: {pdf_dir}. Skipping PDF extraction.")
        return []

    all_pages: list[dict] = []
    pdf_files = list(pdf_dir_path.glob("**/*.pdf"))

    if not pdf_files:
        print(f"[pdf] No PDFs found in {pdf_dir}.")
        return []

    for pdf_file in pdf_files:
        all_pages.extend(extract_pdf(str(pdf_file)))

    print(f"[pdf] Total pages extracted: {len(all_pages)}")
    return all_pages


if __name__ == "__main__":
    pages = extract_all_pdfs()
    for p in pages[:5]:
        print(f"  {p['source_file']} p.{p['page']} | {p['faculty']} | {len(p['text'])} chars")

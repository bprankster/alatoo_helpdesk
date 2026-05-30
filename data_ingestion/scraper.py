"""
scraper.py — DISABLED

The university website (alatoo.edu.kg) blocks automated scrapers.
Use local file ingestion instead:
  - PDFs:        data/raw/pdfs/
  - Manual text: data/raw/manual/

To re-ingest all content:
    python data_ingestion/embedder.py
"""


def scrape_all() -> tuple[list, dict]:
    print("[scraper] Web scraping disabled — site blocks bots.")
    print("[scraper] Add PDFs to data/raw/pdfs/ and text files to data/raw/manual/")
    return [], {}

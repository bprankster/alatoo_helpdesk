"""
refresh.py — Monthly automated data refresh script.

Re-runs the full ingestion pipeline, overwriting stale ChromaDB embeddings
identified by their source_url metadata key. Run via cron or a scheduler:

    # Example crontab (first day of every month at 03:00)
    0 3 1 * * cd /app && python data_ingestion/refresh.py >> /var/log/alatoo_refresh.log 2>&1
"""

import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data_ingestion.embedder import run_full_ingestion


if __name__ == "__main__":
    print(f"[refresh] Starting scheduled data refresh at {datetime.now().isoformat()}")
    run_full_ingestion()
    print(f"[refresh] Refresh complete at {datetime.now().isoformat()}")

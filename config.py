import os
from dotenv import load_dotenv

load_dotenv()

# ── Grok / xAI ────────────────────────────────────────────────────────────────
XAI_API_KEY: str = os.getenv("XAI_API_KEY", "")
GROK_MODEL: str = os.getenv("GROK_MODEL", "grok-3-mini")
GROK_BASE_URL: str = "https://api.x.ai/v1"
LLM_TEMPERATURE: float = 0.0

# ── Telegram ───────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
OFFICER_CHAT_ID: str = os.getenv("OFFICER_CHAT_ID", "")   # channel/chat that receives handoff alerts
TELEGRAM_WEBHOOK_URL: str = os.getenv("TELEGRAM_WEBHOOK_URL", "")  # e.g. https://yourdomain.com/telegram

# ── Embedding ─────────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = "BAAI/bge-m3"
EMBEDDING_DEVICE: str = os.getenv("EMBEDDING_DEVICE", "cpu")

# ── ChromaDB ──────────────────────────────────────────────────────────────────
CHROMA_PATH: str = os.path.join(os.path.dirname(__file__), "data", "chromadb")
CHROMA_COLLECTION: str = "university_docs"

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 50

# ── Voice / STT ───────────────────────────────────────────────────────────────
WHISPER_MODEL_SIZE: str = os.getenv("WHISPER_MODEL_SIZE", "medium")
WHISPER_DEVICE: str = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE: str = "int8"
WHISPER_CONTEXTUAL_PROMPT: str = (
    "Саламатсызбы, ОРТ, программист, факультет, информатика, "
    "экономика, юридический, стипендия, поступление, специальность"
)

# ── Data files ────────────────────────────────────────────────────────────────
DATA_DIR: str = os.path.join(os.path.dirname(__file__), "data")
RIASEC_MAPPING_FILE: str = os.path.join(DATA_DIR, "riasec_mapping.json")
ORT_THRESHOLDS_FILE: str = os.path.join(DATA_DIR, "ort_thresholds.json")

# ── Scraper ───────────────────────────────────────────────────────────────────
UNIVERSITY_BASE_URL: str = "https://www.alatoo.edu.kg"
SCRAPE_TARGETS: list[str] = [
    "https://www.alatoo.edu.kg/admissions/",
    "https://www.alatoo.edu.kg/faculties/",
    "https://www.alatoo.edu.kg/tuition/",
    "https://www.alatoo.edu.kg/programs/",
]
JINA_READER_BASE: str = "https://r.jina.ai/"

# ── Agent ─────────────────────────────────────────────────────────────────────
RETRIEVAL_TOP_K: int = 3                # Hit Rate @ 3
RIASEC_MAX_QUESTIONS: int = 5
SESSION_TTL_SECONDS: int = 3600         # 1 hour idle session expiry

# ── Guardrails ────────────────────────────────────────────────────────────────
INJECTION_KEYWORDS: list[str] = [
    "ignore previous", "ignore all", "jailbreak", "pretend you are",
    "act as", "system prompt", "forget your instructions", "you are now",
    "disregard", "override instructions",
]
DOMAIN_KEYWORDS: list[str] = [
    "поступление", "факультет", "специальность", "орт", "ort", "стипендия",
    "туплум", "syllabus", "программа", "обучение", "университет", "аlatoo",
    "ала-тоо", "карьера", "профессия", "вступительный", "проходной балл",
    "tuition", "fee", "admissions", "faculty", "program", "career",
    "мажор", "bachelor", "master", "бакалавр", "магистр",
]

# ── FastAPI ───────────────────────────────────────────────────────────────────
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))

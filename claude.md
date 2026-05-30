# Agentic RAG — University Admissions & Professional Orientation
## Ala-Too International University (МУА) — Bishkek, Kyrgyzstan
## Final Architecture v3 

---

## What This File Is

This is the authoritative project brief for Claude Code. Read it fully before
touching any file. It supersedes the original plan file entirely.

---

## Project Overview

A fully local Multimodal Agentic RAG system for МУА admissions and professional
orientation. No cloud LLMs. Everything runs on the university server.

Handles:
- Admissions queries (ORT scores, discounts, programs, documents required)
- Career guidance via adaptive RIASEC survey (LLM-generated questions)
- Multilingual voice input (Russian / Kyrgyz / English code-switching)
- Intelligent tool routing via fine-tuned KyrgyzBERT intent classifier
- TTS responses for the web kiosk
- Human handoff via Telegram notification

Targets: FastAPI web kiosk + Telegram bot (future: HIVA holographic kiosk)

---

## Server Hardware

```
GPU:  NVIDIA RTX 4080 — 16GB VRAM
RAM:  32GB system RAM
OS:   Ubuntu 24, CUDA 13.1, Driver 590.48.01
LLM:  Qwen3-14B via Ollama
      Binary:    ~/bin/ollama
      Models:    /dev/shm/ollama_models  (RAM disk — re-pull after reboot)
      Env:       OLLAMA_MODELS set in ~/.bashrc
      Verify:    curl http://localhost:11434/api/tags
```

VRAM budget at full load:
```
Qwen3-14B (Q4):          ~9.0 GB
BGE-m3 embeddings:       ~0.5 GB
Whisper medium:          ~1.5 GB
Kani-TTS-2 (Kyrgyz):     ~3.0 GB
─────────────────────────────────
Total:                   ~14.0 GB  ✅ fits with ~2GB headroom
```

---

## Tech Stack — Final Decisions (Locked)

| Component | Choice | Notes |
|---|---|---|
| LLM | Qwen3-14B via Ollama | Local only. NEVER cloud APIs |
| LangChain | ChatOllama from langchain_ollama | LangChain 0.3 syntax only |
| Dense embeddings | BAAI/bge-m3 | Via langchain_huggingface |
| Sparse retrieval | BM25 via rank_bm25 | Covers exact Kyrgyz term matching |
| Retrieval | EnsembleRetriever 30/70 BM25+dense | Replaces dense-only ChromaDB search |
| Ablation baseline | intfloat/multilingual-e5-large | Section 4 comparison only |
| Vector store | ChromaDB | Local, faculty metadata filtering |
| Intent classifier | KyrgyzBERT fine-tuned | 4-class, NLP course contribution |
| STT | nineninesix/kyrgyz-whisper-medium | Kyrgyz+Russian+English, no router needed |
| TTS | nineninesix/kani-tts-400m-ky | 3GB VRAM, Kyrgyz native |
| TTS toggle | Disabled for Telegram, enabled for web | Config flag |
| Backend | FastAPI | Session management |
| Frontend | Gradio | Web kiosk |
| Messaging | python-telegram-bot | Telegram webhook |
| Config | config.yaml | ALL hyperparams — zero hardcoding |
| Data ingestion | Local PDFs + manual text files | NO web scraper (site blocks bots) |

NEVER use Gemini API, Grok API, OpenAI API, or any cloud LLM.
ALWAYS use LangChain 0.3 import paths.
NEVER call langchain_openai for the main LLM (it exists in requirements for Grok — remove it).

---

## Existing Files (Do NOT recreate)

```
agent/__init__.py
agent/core.py              ← REPLACE: remove Grok, add ChatOllama + thinking mode
agent/guardrails.py        ← keep, add Kyrgyz injection keywords
agent/session.py           ← keep as-is
agent/tools/__init__.py
agent/tools/human_handoff.py      ← keep as-is
agent/tools/orientation_engine.py ← REPLACE: adaptive LLM-generated questions
agent/tools/ort_validator.py      ← REPLACE: use real ORT data below
agent/tools/program_comparator.py ← keep, update faculty aliases
api/__init__.py
api/chat_endpoint.py       ← UPDATE: add TTS response for web platform
api/main.py                ← UPDATE: add lifespan model loading
api/telegram_bot.py        ← keep as-is
config.py                  ← REPLACE ENTIRELY with config.yaml approach
data/ort_thresholds.json   ← REPLACE with real data below
data/riasec_mapping.json   ← REPLACE with real faculty mapping below
data_ingestion/__init__.py
data_ingestion/chunker.py  ← keep as-is
data_ingestion/embedder.py ← UPDATE: add BM25 support
data_ingestion/pdf_extractor.py ← keep as-is
data_ingestion/refresh.py  ← keep as-is
data_ingestion/scraper.py  ← DELETE or empty — site blocks scrapers
docker/Dockerfile
docker/docker-compose.yml  ← UPDATE: remove XAI_API_KEY, add Ollama
evaluation/__init__.py
evaluation/evaluate.py     ← UPDATE: add classifier ablation metric
evaluation/golden_dataset.json ← UPDATE: use real faculty names
requirements.txt           ← UPDATE: add new packages
ui/__init__.py
ui/kiosk.py                ← UPDATE: add TTS audio playback
voice/__init__.py
voice/stt.py               ← REPLACE: use kyrgyz-whisper-medium, remove router
```

New files to create:
```
classifier/
├── __init__.py
├── train.py               ← KyrgyzBERT fine-tuning
├── predict.py             ← Intent inference
├── dataset.py             ← Data loader
└── training_data.json     ← 200+ labeled examples scaffold

retrieval/
├── __init__.py
└── chroma_store.py        ← Hybrid BM25 + ChromaDB EnsembleRetriever

tts/
├── __init__.py
└── kani_tts.py            ← Kani-TTS-2 Kyrgyz wrapper

agent/
└── router.py              ← KyrgyzBERT → tool dispatch

data/raw/
├── pdfs/                  ← Place university PDFs here
└── manual/                ← .txt files with manually copied content
```

---

## Real University Data

### ORT Threshold (Official 2024-2025)

**Minimum ORT score: 110 points** (set by Ministry of Education KR)
- This applies to ALL faculties equally
- МУА has NO budget places — it is a private university
- Discounts replace budget seats (see discount table below)

Additional subject requirements:
- Факультет инженерии и информатики: Math ≥60 AND Physics ≥60
- Медицинский факультет (Лечебное дело): Biology ≥60 AND Chemistry ≥60
- All other faculties: no additional subjects required

English requirement for all programs: B1 Intermediate (AIU EPT internal test)

### ORT Discount Structure (Replace ort_thresholds.json)

```json
{
  "_note": "Official Ala-Too University ORT-based discount system 2024-2025. No budget places — private university.",
  "min_ort_threshold": 110,
  "discount_table": [
    {"score_range": "gold_certificate", "discount_percent": 100},
    {"score_range": "216-220", "discount_percent": 50},
    {"score_range": "211-215", "discount_percent": 45},
    {"score_range": "206-210", "discount_percent": 40},
    {"score_range": "201-205", "discount_percent": 35},
    {"score_range": "196-200", "discount_percent": 30},
    {"score_range": "191-195", "discount_percent": 25},
    {"score_range": "186-190", "discount_percent": 20},
    {"score_range": "181-185", "discount_percent": 15},
    {"score_range": "176-180", "discount_percent": 10},
    {"score_range": "171-175", "discount_percent": 5}
  ],
  "other_discounts": [
    "Выпускникам колледжа МУА: 10%",
    "Выпускникам IT&Business колледжа МУА с красным дипломом: 20%",
    "Выпускникам лицеев МОУ Сапат: 15%",
    "Сиротам (оба родителя): 40%",
    "Потеря одного родителя: 10%",
    "Семьи с 4+ детьми до 22 лет: 10%"
  ],
  "notes": [
    "Скидка прекращается при 4 академических задолженностях в семестре",
    "Студент может использовать только одну скидку",
    "Оплата принимается только в национальной валюте по курсу НБКР",
    "Первоначальный платеж: 30% от стоимости обучения",
    "Можно платить по 50% в начале каждого семестра",
    "ОРТ прошлых лет использовать НЕЛЬЗЯ",
    "Подача через портал МОН КР: 2020.edu.gov.kg/vuz"
  ],
  "additional_subject_requirements": {
    "Факультет инженерии и информатики": {"math": 60, "physics": 60},
    "Медицинский факультет": {"biology": 60, "chemistry": 60}
  }
}
```

### Real Faculty & Program Data (Replace riasec_mapping.json)

```json
{
  "faculties": {
    "Факультет инженерии и информатики": {
      "riasec_types": ["R", "I"],
      "additional_ort": "Математика ≥60 и Физика ≥60",
      "programs": [
        {
          "name": "Компьютерная инженерия",
          "degree": "Бакалавр",
          "duration": "4 года",
          "language": "Английский",
          "careers": ["Инженер-программист", "Архитектор ПО", "QA инженер", "Руководитель проекта"],
          "subjects": ["Разработка мобильных приложений", "Backend разработка", "Frontend разработка", "Архитектура и шаблоны проектирования"]
        },
        {
          "name": "Прикладная математика и информатика",
          "degree": "Бакалавр",
          "duration": "4 года",
          "language": "Английский",
          "careers": ["Разработчик образовательного ПО", "Аналитик данных", "Педагогический дизайнер"],
          "subjects": ["Теоретическая информатика", "Анализ и принятие решений", "Искусство преподавания"]
        },
        {
          "name": "Анализ данных и интеллектуальные системы",
          "degree": "Бакалавр",
          "duration": "4 года",
          "language": "Английский",
          "careers": ["Аналитик данных", "Data Scientist", "ML инженер", "Разработчик ИИ"],
          "subjects": ["Введение в Data Science", "ИИ", "Машинное обучение", "Глубокое обучение", "Визуализация данных"]
        },
        {
          "name": "Искусственный интеллект и робототехника",
          "degree": "Бакалавр",
          "duration": "4 года",
          "language": "Английский",
          "careers": ["Разработчик ИИ", "Инженер робототехники", "Специалист по компьютерному зрению"],
          "subjects": ["Машинное обучение", "Основы робототехники", "ИИ и глубокое обучение", "Компьютерное зрение", "IoT"]
        },
        {
          "name": "Менеджмент в информационных технологиях",
          "degree": "Бакалавр",
          "duration": "4 года",
          "language": "Английский",
          "careers": ["IT-менеджер", "Руководитель проекта", "Бизнес-аналитик", "IT-консультант"],
          "subjects": ["Основы бизнеса", "Управление процессами", "Менеджмент", "Маркетинг", "Управление рисками"]
        },
        {
          "name": "Кибербезопасность и этичный хакинг",
          "degree": "Бакалавр",
          "duration": "4 года",
          "language": "Английский",
          "careers": ["Аналитик кибербезопасности", "Этичный хакер", "Менеджер ИБ", "Консультант по безопасности"],
          "subjects": ["Основы кибербезопасности", "Этичный хакинг и пентест", "Цифровая криминалистика", "Облачная безопасность"]
        },
        {
          "name": "Основы креативных индустрий",
          "degree": "Бакалавр",
          "duration": "4 года",
          "language": "Английский",
          "careers": ["Креативный директор", "Менеджер по маркетингу", "Координатор проекта"],
          "subjects": ["Цифровой дизайн", "3D Дизайн", "Game Дизайн", "VR Дизайн", "Motion Дизайн"]
        }
      ]
    },
    "Факультет гуманитарных наук": {
      "riasec_types": ["A", "S"],
      "additional_ort": "не требуется",
      "programs": [
        {
          "name": "Лингвистика (перевод и переводоведение)",
          "degree": "Бакалавр",
          "duration": "4 года",
          "language": "Английский",
          "careers": ["Переводчик", "Лингвист", "Редактор", "Преподаватель языков"],
          "subjects": ["Нейролингвистика", "Синхронный перевод", "Художественный перевод", "Психолингвистика"]
        },
        {
          "name": "Филология (Английский язык и литература)",
          "degree": "Бакалавр",
          "duration": "4 года",
          "language": "Английский",
          "careers": ["Преподаватель", "Редактор", "Переводчик", "Контент-менеджер"],
          "subjects": ["Введение в литературоведение", "Стилистика", "Сравнительное языкознание", "Семантика"]
        },
        {
          "name": "Педагогика (начальное образование)",
          "degree": "Бакалавр",
          "duration": "4 года",
          "language": "Английский",
          "careers": ["Учитель начальных классов", "Педагог"],
          "subjects": ["STEM образование", "Ораторское мастерство", "Основы научных исследований"]
        },
        {
          "name": "STEM-образование",
          "degree": "Бакалавр",
          "duration": "4 года",
          "language": "Английский",
          "careers": ["Учитель STEM", "Педагог в школе или лицее"],
          "subjects": ["Введение в STEM", "Новые направления физики/химии/биологии", "Цифровая образовательная среда"]
        }
      ]
    },
    "Факультет экономики и управления": {
      "riasec_types": ["E", "C"],
      "additional_ort": "не требуется",
      "programs": [
        {
          "name": "Экономика (Международная экономика и бизнес)",
          "degree": "Бакалавр",
          "duration": "4 года",
          "language": "Английский",
          "careers": ["Аналитик", "Экономист", "Финансист"],
          "subjects": ["Экономика", "Международная экономика", "Эконометрика", "Бухгалтерский учет и аудит"]
        },
        {
          "name": "Экономика (Международные финансы и аудит)",
          "degree": "Бакалавр",
          "duration": "4 года",
          "language": "Английский",
          "careers": ["Финансист", "Банкир", "Аудитор", "Инвестиционный аналитик"],
          "subjects": ["Финансовый менеджмент", "Банковское дело", "Финансовый анализ", "Аудит"]
        },
        {
          "name": "Менеджмент",
          "degree": "Бакалавр",
          "duration": "4 года",
          "language": "Английский",
          "careers": ["Менеджер", "Бизнес-аналитик", "HR специалист", "Специалист по маркетингу"],
          "subjects": ["Инновационный менеджмент", "Управление проектом", "Кризисное управление"]
        },
        {
          "name": "Юриспруденция (Международное и бизнес-право)",
          "degree": "Бакалавр",
          "duration": "4 года",
          "language": "Английский",
          "careers": ["Юрист", "Адвокат", "Специалист международных организаций"],
          "subjects": ["Права человека", "Международное публичное право", "Международное инвестиционное право"]
        },
        {
          "name": "Экономика окружающей среды",
          "degree": "Бакалавр",
          "duration": "4 года",
          "language": "Английский",
          "careers": ["Эколог-аналитик", "Специалист по устойчивому развитию", "Экологический консультант"],
          "subjects": ["Экономика природных ресурсов", "Зелёная экономика", "Экологическое право"]
        },
        {
          "name": "Менеджмент в индустрии гостеприимства и туризма",
          "degree": "Бакалавр",
          "duration": "4 года",
          "language": "Английский",
          "careers": ["Менеджер по туризму", "Администратор гостиницы", "Координатор туристических программ"],
          "subjects": ["Туризм в Кыргызстане", "Международный туризм", "Устойчивое развитие туризма"]
        }
      ]
    },
    "Медицинский факультет": {
      "riasec_types": ["I", "S"],
      "additional_ort": "Биология ≥60 и Химия ≥60",
      "programs": [
        {
          "name": "Лечебное дело",
          "degree": "Бакалавр",
          "duration": "6 лет",
          "language": "Английский",
          "careers": ["Врач", "Преподаватель медицины", "Медицинский администратор"],
          "subjects": ["Анатомия", "Физиология", "Биохимия", "Клинические науки", "Патология"]
        }
      ]
    },
    "Факультет социальных наук": {
      "riasec_types": ["S", "E"],
      "additional_ort": "не требуется",
      "programs": [
        {
          "name": "Психология",
          "degree": "Бакалавр",
          "duration": "4 года",
          "language": "Английский",
          "careers": ["Психолог", "Консультант", "HR специалист", "Нейропсихолог"],
          "subjects": ["Психология личности", "Социальная психология", "Нейропсихология", "Психогенетика"]
        },
        {
          "name": "Журналистика (Медиа, коммуникация и дизайн)",
          "degree": "Бакалавр",
          "duration": "4 года",
          "language": "Английский",
          "careers": ["Репортёр", "Редактор", "Продюсер", "PR специалист"],
          "subjects": ["Связи с общественностью", "Межкультурная коммуникация", "Аналитическая журналистика"]
        },
        {
          "name": "Международные отношения",
          "degree": "Бакалавр",
          "duration": "4 года",
          "language": "Английский",
          "careers": ["Дипломат", "Специалист МО", "Аналитик", "Сотрудник НПО"],
          "subjects": ["История МО", "Международное право", "Глобализация мировой политики"]
        }
      ]
    }
  },
  "riasec_to_faculty": {
    "R": ["Факультет инженерии и информатики"],
    "I": ["Факультет инженерии и информатики", "Медицинский факультет"],
    "A": ["Факультет гуманитарных наук"],
    "S": ["Факультет гуманитарных наук", "Медицинский факультет", "Факультет социальных наук"],
    "E": ["Факультет экономики и управления", "Факультет социальных наук"],
    "C": ["Факультет экономики и управления"]
  }
}
```

### Contact Info (Update human_handoff.py)

```
Приемная комиссия МУА
Адрес:  ул.Анкара (Горький) 1/10, мкр. «Тунгуч», г.Бишкек (D-блок, 1 этаж)
Телефон: +996 555 820 000 (WhatsApp)
Email:  admission@alatoo.edu.kg
Портал: https://2020.edu.gov.kg/vuz
Часы:   по графику работы приемной комиссии
```

---

## Data Ingestion — No Scraper

The university site (alatoo.edu.kg) blocks automated scrapers. Replace
`data_ingestion/scraper.py` with local file ingestion only.

### New Directory Structure

```
data/
├── raw/
│   ├── pdfs/           ← Place official university PDFs here
│   │   └── README.txt  ← Instructions for which PDFs to add
│   └── manual/         ← .txt files with manually copied content
│       ├── admissions_general.txt
│       ├── ort_discounts.txt
│       ├── faculty_engineering.txt
│       ├── faculty_education.txt
│       ├── faculty_economics.txt
│       ├── faculty_medicine.txt
│       ├── faculty_social.txt
│       ├── faq.txt
│       └── contact.txt
├── ort_thresholds.json  ← Replace with real data above
└── riasec_mapping.json  ← Replace with real data above
```

Replace `data_ingestion/scraper.py` content with:
```python
"""
scraper.py — DISABLED
The university website blocks automated scrapers.
Use local file ingestion via pdf_extractor.py and manual .txt files instead.
Run: python data_ingestion/embedder.py
"""
def scrape_all():
    print("[scraper] Web scraping disabled — site blocks bots.")
    print("[scraper] Add PDFs to data/raw/pdfs/ and text to data/raw/manual/")
    return [], {}
```

Update `data_ingestion/embedder.py` → `run_full_ingestion()` to:
1. Skip scrape_all() or call it knowing it returns empty
2. Scan `data/raw/manual/*.txt` as additional text sources
3. Everything else stays the same

---

## config.yaml — Replace config.py

Create `config.yaml` at project root. Update all modules to read from it.

```yaml
llm:
  model: "qwen3:14b"
  base_url: "http://localhost:11434"
  temperature: 0.1
  num_predict: 512
  num_ctx: 4096

embeddings:
  dense_model: "BAAI/bge-m3"
  ablation_model: "intfloat/multilingual-e5-large"
  device: "cuda"
  normalize: true

retrieval:
  bm25_weight: 0.3
  dense_weight: 0.7
  top_k: 5

chunking:
  chunk_size: 500
  chunk_overlap: 50

chroma:
  persist_dir: "./data/chromadb"
  collection_name: "university_docs"

classifier:
  model_name: "metinovadilet/KyrgyzBert"
  finetuned_path: "./classifier/model"
  labels:
    ort_validator: 0
    orientation_engine: 1
    program_comparator: 2
    human_handoff: 3
  confidence_threshold: 0.75

stt:
  model: "nineninesix/kyrgyz-whisper-medium"
  device: "cuda"
  compute_type: "float16"
  contextual_bias: "Саламатсызбы, ОРТ, программист, факультет, экономика, медицина, юрист"

tts:
  model: "nineninesix/kani-tts-400m-ky"
  enabled_platforms: ["web"]
  output_dir: "./audio_responses"

orientation:
  max_questions: 5

agent:
  max_iterations: 10
  thinking_tools: ["Professional_Orientation_Engine"]
  fast_tools: ["ORT_Validator", "Program_Comparator_RAG", "Human_Handoff_Trigger"]
  fallback_message: "Извините, у меня нет точного ответа на этот вопрос. Пожалуйста, обратитесь в приёмную комиссию: +996 555 820 000 (WhatsApp) или admission@alatoo.edu.kg"

telegram:
  bot_token: ""
  officer_chat_id: ""
  webhook_url: ""

api:
  host: "0.0.0.0"
  port: 8000
  session_ttl_seconds: 3600

data:
  riasec_mapping: "./data/riasec_mapping.json"
  ort_thresholds: "./data/ort_thresholds.json"
  raw_pdfs: "./data/raw/pdfs"
  raw_manual: "./data/raw/manual"

guardrails:
  injection_keywords:
    - "ignore previous"
    - "ignore all"
    - "jailbreak"
    - "pretend you are"
    - "act as"
    - "system prompt"
    - "forget your instructions"
    - "you are now"
    - "disregard"
    - "override instructions"
    - "100% скидка"
    - "ты зачислен"
    - "вы зачислены"
  domain_keywords:
    - "поступление"
    - "факультет"
    - "специальность"
    - "орт"
    - "ort"
    - "стипендия"
    - "скидка"
    - "syllabus"
    - "программа"
    - "обучение"
    - "университет"
    - "аlatoo"
    - "ала-тоо"
    - "муа"
    - "карьера"
    - "профессия"
    - "проходной балл"
    - "tuition"
    - "admissions"
    - "faculty"
    - "career"
    - "бакалавр"
    - "диплом"
    - "кабылуу"
    - "факультет"
    - "адистик"
```

---

## New Component 1 — LLM with Thinking Mode

```python
# agent/core.py — replace Grok ChatOpenAI with this

from langchain_ollama import ChatOllama
import yaml

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

def get_llm(thinking: bool = False) -> ChatOllama:
    """
    thinking=True  → Qwen3 extended reasoning (for RIASEC diagnosis)
    thinking=False → direct fast response (ORT checks, comparisons)
    """
    return ChatOllama(
        model=cfg["llm"]["model"],
        base_url=cfg["llm"]["base_url"],
        temperature=cfg["llm"]["temperature"],
        num_predict=cfg["llm"]["num_predict"],
        num_ctx=cfg["llm"]["num_ctx"],
    )

def format_prompt(text: str, thinking: bool = False) -> str:
    """Prepend Qwen3 thinking mode prefix."""
    return ("/think " if thinking else "/no_think ") + text
```

---

## New Component 2 — Hybrid Retrieval

```python
# retrieval/chroma_store.py — new file

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import Optional
import yaml

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

def load_embeddings(use_ablation: bool = False) -> HuggingFaceEmbeddings:
    model = (cfg["embeddings"]["ablation_model"] if use_ablation
             else cfg["embeddings"]["dense_model"])
    return HuggingFaceEmbeddings(
        model_name=model,
        model_kwargs={"device": cfg["embeddings"]["device"]},
        encode_kwargs={"normalize_embeddings": cfg["embeddings"]["normalize"]},
    )

def get_retriever(
    docs: list[Document],
    faculty_filter: Optional[str] = None,
    use_ablation: bool = False,
) -> EnsembleRetriever:
    """BM25 handles exact Kyrgyz terms; BGE-m3 handles semantic meaning."""
    search_kwargs = {"k": cfg["retrieval"]["top_k"]}
    if faculty_filter:
        search_kwargs["filter"] = {"faculty": faculty_filter}

    dense = Chroma(
        collection_name=cfg["chroma"]["collection_name"],
        embedding_function=load_embeddings(use_ablation),
        persist_directory=cfg["chroma"]["persist_dir"],
    ).as_retriever(search_kwargs=search_kwargs)

    bm25 = BM25Retriever.from_documents(
        docs, k=cfg["retrieval"]["top_k"]
    )

    return EnsembleRetriever(
        retrievers=[bm25, dense],
        weights=[cfg["retrieval"]["bm25_weight"],
                 cfg["retrieval"]["dense_weight"]],
    )
```

---

## New Component 3 — KyrgyzBERT Intent Classifier

This is the NLP course fine-tuning contribution. Routes student queries to the
correct tool using a lightweight classifier instead of relying purely on
Qwen3's ReAct reasoning.

```
Query → KyrgyzBERT (~10ms, CPU) → confidence ≥ 0.75?
    YES → dispatch to tool directly  (fast path)
    NO  → Qwen3 ReAct agent decides  (safe fallback)
```

Training data format (200+ examples, 50+ per class, Russian AND Kyrgyz):
```json
[
  {"text": "Менин ОРТ балым 145. Кире аламбы?", "label": "ort_validator", "label_id": 0},
  {"text": "Мой ОРТ 138. Можно поступить на инженерию?", "label": "ort_validator", "label_id": 0},
  {"text": "Кандай скидка алса болот ОРТ боюнча?", "label": "ort_validator", "label_id": 0},
  {"text": "Мен кайсы факультетти тандасам билбейм", "label": "orientation_engine", "label_id": 1},
  {"text": "Не знаю какую специальность выбрать", "label": "orientation_engine", "label_id": 1},
  {"text": "IT факультетин юридикалык менен салыштыр", "label": "program_comparator", "label_id": 2},
  {"text": "Сравни психологию и журналистику", "label": "program_comparator", "label_id": 2},
  {"text": "Мне нужен живой сотрудник", "label": "human_handoff", "label_id": 3},
  {"text": "Адамды чакырыңыз", "label": "human_handoff", "label_id": 3}
]
```

Generate synthetic examples using Qwen3-14B locally before training:
```python
prompt = """Сгенерируй 10 реалистичных запросов студента на русском И кыргызском языках
для намерения: {intent}
Контекст: чат-бот приемной комиссии Ала-Тоо Университета.
Верни JSON массив: [{{"text":"...","label":"{label}","label_id":{id}}}]"""
```

After generating data, train with `classifier/train.py`.
Target metrics: accuracy > 85%, f1_macro > 0.83 on eval set.
Expected training time on RTX 4080: ~40 minutes.

```python
# agent/router.py — new file

from classifier.predict import predict_intent
import yaml

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

def route_query(text: str) -> str:
    """
    Returns tool name if confidence high enough, else 'react_agent'.
    Fast path saves ~2 seconds per simple query.
    """
    intent, confidence = predict_intent(text)
    threshold = cfg["classifier"]["confidence_threshold"]
    return intent if confidence >= threshold else "react_agent"
```

---

## New Component 4 — STT (Simplified, No Router)

nineninesix/kyrgyz-whisper-medium handles Kyrgyz + Russian + English natively.
No language detection or routing needed — single model handles all three.

```python
# voice/stt.py — replace entire file

from faster_whisper import WhisperModel
import yaml, tempfile, os
from pathlib import Path

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

_model = None

def _get_model():
    global _model
    if _model is None:
        print(f"[stt] Loading {cfg['stt']['model']}...")
        _model = WhisperModel(
            cfg["stt"]["model"],
            device=cfg["stt"]["device"],
            compute_type=cfg["stt"]["compute_type"],
        )
    return _model

def transcribe(audio_path: str) -> str:
    """Transcribe audio. Auto-detects Kyrgyz/Russian/English."""
    model = _get_model()
    try:
        segments, info = model.transcribe(
            str(audio_path),
            initial_prompt=cfg["stt"]["contextual_bias"],
            language=None,      # auto-detect
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        print(f"[stt] Detected: {info.language} | Text: {text[:80]}")
        return text
    except Exception as e:
        print(f"[stt] Error: {e}")
        return ""

def transcribe_bytes(audio_bytes: bytes, suffix: str = ".ogg") -> str:
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        return transcribe(tmp_path)
    finally:
        os.unlink(tmp_path)
```

---

## New Component 5 — TTS

```python
# tts/kani_tts.py — new file

from kani_tts import KaniTTS
import yaml, os
from pathlib import Path

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

_model = None

def load():
    global _model
    print(f"[tts] Loading {cfg['tts']['model']}...")
    _model = KaniTTS(cfg["tts"]["model"])

def speak(text: str, filename: str = "response.wav") -> str:
    """
    Convert text to audio. Returns path to wav file.
    Only called when platform is in cfg["tts"]["enabled_platforms"].
    """
    output_dir = Path(cfg["tts"]["output_dir"])
    output_dir.mkdir(exist_ok=True)
    output_path = str(output_dir / filename)

    audio, _ = _model(text)
    _model.save_audio(audio, output_path)
    return output_path

def is_enabled(platform: str) -> bool:
    return platform in cfg["tts"]["enabled_platforms"]
```

Update `api/chat_endpoint.py` to call TTS for web platform:
```python
# In chat endpoint, after getting reply from agent:
if tts.is_enabled(platform):
    audio_path = tts.speak(reply)
    return ChatResponse(reply=reply, audio_path=audio_path)
else:
    return ChatResponse(reply=reply)
```

TTS is disabled for Telegram (text-only). Enabled for web kiosk.

---

## Updated ORT Validator Logic

The original ort_validator.py used placeholder data and a budget/paid split.
МУА has NO budget places. Replace the logic entirely:

```python
# Key logic change in agent/tools/ort_validator.py

def check_ort_eligibility(score: int, faculty: str = None) -> str:
    """
    МУА admissions logic:
    - Minimum threshold: 110 (all faculties)
    - Engineering/IT: also need Math≥60 and Physics≥60
    - Medicine: also need Biology≥60 and Chemistry≥60
    - No budget places — discounts based on ORT score
    """
    MIN_THRESHOLD = 110

    if score < MIN_THRESHOLD:
        return (
            f"❌ Ваш балл ОРТ ({score}) ниже минимального порога {MIN_THRESHOLD} баллов "
            f"для поступления в МУА. Рассмотрите поступление в IT&Business колледж МУА "
            f"(без ОРТ, срок 1г10м, затем 2-й курс университета)."
        )

    # Calculate discount
    discount = get_discount(score)
    discount_msg = f"Скидка на обучение: {discount}%" if discount > 0 else "Без скидки"

    # Faculty-specific additional requirements
    if faculty and "инженер" in faculty.lower() or "информатик" in faculty.lower():
        additional = "⚠️ Также нужны доп.предметы ОРТ: Математика ≥60 и Физика ≥60."
    elif faculty and "медицин" in faculty.lower() or "лечебн" in faculty.lower():
        additional = "⚠️ Также нужны доп.предметы ОРТ: Биология ≥60 и Химия ≥60."
    else:
        additional = ""

    return (
        f"✅ Ваш балл ОРТ ({score}) превышает минимальный порог ({MIN_THRESHOLD}). "
        f"Вы можете подать документы в МУА. {discount_msg}. {additional} "
        f"⚠️ Окончательное решение принимает приёмная комиссия. "
        f"Подача через портал: 2020.edu.gov.kg/vuz"
    )
```

---

## Adaptive RIASEC Engine (Replace orientation_engine.py)

Use LLM-generated questions instead of fixed survey_questions from riasec_mapping.json.
Each question is generated based on all previous answers targeting uncertain RIASEC dimensions.

```python
# Key change: replace static questions with dynamic generation

def _generate_next_question(self, history: list) -> str:
    llm = get_llm(thinking=True)
    is_first = len(history) == 0
    history_text = "\n".join(
        f"Q{i+1}: {h['question']}\nОтвет: {h['answer']}"
        for i, h in enumerate(history)
    )
    content = (
        "Первый вопрос — широкий, ситуационный, о предпочтениях студента."
        if is_first else
        f"Предыдущие ответы:\n{history_text}\n\n"
        "Определи неясные RIASEC-типы. Задай наиболее дискриминирующий следующий вопрос."
    )
    prompt = format_prompt(
        f"Ты консультант по профориентации (Holland RIASEC: "
        f"R=Реалистичный I=Исследовательский A=Артистический "
        f"S=Социальный E=Предприимчивый C=Конвентциональный).\n"
        f"{content}\n"
        f"Вопрос №{len(history)+1} из 5. На русском. Ситуационный. "
        f"Ровно 4 варианта — каждый отражает разный тип.\n"
        f'Верни ТОЛЬКО JSON без markdown: '
        f'{{"question":"текст","options":['
        f'{{"text":"вариант","riasec":"R"}},'
        f'{{"text":"вариант","riasec":"I"}},'
        f'{{"text":"вариант","riasec":"A"}},'
        f'{{"text":"вариант","riasec":"S"}}]}}',
        thinking=True,
    )
    raw = llm.invoke(prompt).content
    # Strip /think tokens if present
    if "</think>" in raw:
        raw = raw.split("</think>")[-1].strip()
    return raw
```

After 5 questions, generate result mapping to REAL faculty names from riasec_mapping.json.

---

## Updated requirements.txt

Add these packages (keep everything existing):
```
# Already in requirements.txt — keep:
langchain==0.3.x
langchain-community==0.3.x
chromadb
sentence-transformers
faster-whisper
fastapi
gradio
python-telegram-bot
python-dotenv
pydantic

# REMOVE:
langchain-openai          ← no longer needed for main LLM

# ADD:
langchain-ollama          ← ChatOllama for Qwen3
langchain-huggingface     ← HuggingFaceEmbeddings
rank_bm25                 ← BM25 sparse retrieval
kani-tts                  ← TTS
transformers              ← KyrgyzBERT classifier
torch                     ← classifier training
scikit-learn              ← classifier evaluation metrics
accelerate                ← faster training
pyyaml                    ← config.yaml reading
```

---

## Updated docker-compose.yml

Remove XAI_API_KEY. Add Ollama service:

```yaml
services:
  ollama:
    image: ollama/ollama:latest
    volumes:
      - ollama_models:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  agent:
    build: .
    depends_on: [ollama]
    environment:
      - TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN}
      - OFFICER_CHAT_ID=${OFFICER_CHAT_ID}
      - TELEGRAM_WEBHOOK_URL=${TELEGRAM_WEBHOOK_URL}
    volumes:
      - ./data/chromadb:/app/data/chromadb
      - ./data/raw:/app/data/raw

volumes:
  ollama_models:
```

---

## Build Order (Follow This Sequence)

### Phase 1 — LLM Verification (30 min)
```bash
curl http://localhost:11434/api/tags   # confirm qwen3:14b loaded
python -c "
from langchain_ollama import ChatOllama
llm = ChatOllama(model='qwen3:14b', base_url='http://localhost:11434')
print(llm.invoke('/no_think Привет! Ты работаешь?').content)
"
```

### Phase 2 — Replace config.py with config.yaml (30 min)
1. Create `config.yaml` with content above
2. Update all imports: replace `from config import X` with yaml loading
3. Remove `config.py` or keep as backward-compat shim

### Phase 3 — Real Data Files (30 min)
1. Replace `data/ort_thresholds.json` with real data from this file
2. Replace `data/riasec_mapping.json` with real faculty data from this file
3. Create `data/raw/manual/` directory
4. Manually copy key content into .txt files:
   - admissions_general.txt: ORT rules, documents required, English test
   - ort_discounts.txt: full discount table
   - faq.txt: FAQ content from website
   - One .txt per faculty with program descriptions

### Phase 4 — Disable Scraper, Test Ingestion (1 hr)
1. Replace scraper.py with disabled stub
2. Run `python data_ingestion/embedder.py` on manual .txt files
3. Verify ChromaDB has >50 chunks
4. Test query: "порог ОРТ для поступления" → should return admission info

### Phase 5 — Hybrid Retrieval (1 hr)
```bash
pip install rank_bm25 langchain-huggingface
```
1. Create `retrieval/chroma_store.py`
2. Update `data_ingestion/embedder.py` to support BM25 index
3. Test: Kyrgyz query "ОРТ упайы" → verify BM25 finds exact term

### Phase 6 — Update ORT Validator (1 hr)
1. Replace logic with real МУА admission rules
2. No budget/paid split — just threshold + discount
3. Test: score 145 → eligible + 0% discount
4. Test: score 195 → eligible + 25% discount
5. Test: score 90 → not eligible, suggest college

### Phase 7 — KyrgyzBERT Classifier (4 hrs)
```bash
pip install transformers torch scikit-learn accelerate
```
1. Generate 200 training examples via Qwen3 (50 per class, RU+KG)
2. Save to `classifier/training_data.json`
3. Run `classifier/train.py` (~40 min)
4. Verify accuracy > 85%
5. Create `agent/router.py`

### Phase 8 — STT Replacement (1 hr)
```bash
pip install faster-whisper
```
1. Replace `voice/stt.py` with simplified single-model version
2. Test on a 30-second Kyrgyz-Russian audio clip
3. Verify no language router needed

### Phase 9 — TTS Addition (1 hr)
```bash
pip install kani-tts
```
1. Create `tts/kani_tts.py`
2. Update `api/chat_endpoint.py` to return audio for web platform
3. Update `ui/kiosk.py` to play audio responses
4. Verify TTS disabled for Telegram

### Phase 10 — Adaptive RIASEC (2 hrs)
1. Replace `orientation_engine.py` with LLM-generated questions
2. Test 5-question flow manually via CLI
3. Verify faculty recommendations use real МУА faculty names

### Phase 11 — Update Human Handoff (30 min)
1. Update contact info to real МУА details
2. Test Telegram notification with session state

### Phase 12 — Evaluation Update (1 hr)
1. Update `golden_dataset.json` with real faculty names
2. Add classifier ablation: KyrgyzBERT vs pure ReAct accuracy
3. Run ablation A: BGE-m3 alone vs BM25+BGE-m3 ensemble

---

## Evaluation Updates

Add to `evaluation/evaluate.py`:

```python
# Ablation D: KyrgyzBERT classifier vs pure ReAct routing
def evaluate_classifier_vs_react(scenarios):
    """
    Compare:
    A) route_query() → KyrgyzBERT classifier (fast path)
    B) pure ReAct agent tool selection
    Measure: accuracy, latency, token consumption
    """
```

Golden dataset `golden_dataset.json` — update all expected_result_contains to use
real faculty names:
- "CS" → "Факультет инженерии и информатики"
- "Economics" → "Факультет экономики и управления"
- "Law" → "Юриспруденция (Международное и бизнес-право)"
- "Education" → "Факультет гуманитарных наук"
- "Psychology" → "Психология"
- "Medicine" → "Медицинский факультет"

---

## Code Quality Rules

- All hyperparameters in config.yaml — zero magic numbers in code
- Type hints on every function signature
- Tool docstrings must say exactly when agent SHOULD and SHOULD NOT call them
- Session state never stored inside tool classes
- Temperature 0.1 everywhere
- ChromaDB 0 results → cfg["agent"]["fallback_message"], never hallucinate
- All model loading once at FastAPI startup via lifespan, not per-request
- TTS is optional — always check is_enabled(platform) before calling speak()

---

## Absolute Constraints

- NO langchain_google_genai, NO langchain_openai for main LLM
- NO Grok API, NO Gemini API, NO OpenAI API anywhere
- NO web scraping of alatoo.edu.kg
- NO LangChain 0.1/0.2 import paths
- NO session state inside tool classes
- NO hardcoded values — everything in config.yaml
- NO localStorage in any frontend JavaScript
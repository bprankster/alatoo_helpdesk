# Ala-Too University Admissions Helpdesk Agent

An AI-powered admissions assistant for Ala-Too International University (Bishkek, Kyrgyzstan). The system answers questions about admissions, ORT scores, tuition, and programmes in Kyrgyz, Russian, and English — with voice input (Kyrgyz Whisper STT), voice output (Kyrgyz TTS), and an adaptive career orientation test (Holland RIASEC).

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                FastAPI  (port 8000)                  │
│  POST /chat · POST /voice · POST /telegram           │
│  GET  /kiosk  (Gradio web UI, mounted as ASGI)       │
└──────────────┬───────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────┐
│              ReAct Agent  (agent/core.py)            │
│  Qwen3-14B via Ollama  ·  LangChain AgentExecutor   │
│                                                      │
│  Tools:                                              │
│  ├─ ORT_Validator          (ort_validator.py)        │
│  ├─ University_KB_Search   (kb_search.py)            │
│  ├─ Program_Comparator_RAG (program_comparator.py)   │
│  ├─ Professional_Orientation_Engine (orientation…)   │
│  └─ Human_Handoff_Trigger  (human_handoff.py)        │
└──────────────┬───────────────────────────────────────┘
               │
       ┌───────┴────────┐
       ▼                ▼
┌─────────────┐  ┌──────────────────────────┐
│  ChromaDB   │  │  Ollama (local, port 11434)│
│  BGE-M3     │  │  Qwen3:14b               │
│  BM25       │  └──────────────────────────┘
│  Hybrid RAG │
└─────────────┘

Voice pipeline:
  Microphone → faster-whisper (kyrgyz-whisper-medium) → Agent → kani-tts-400m-ky → Audio
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 12 GB | 16 GB |
| RAM | 16 GB | 32 GB |
| Disk | 30 GB | 50 GB |
| CUDA | 11.8+ | 12.1+ |
| OS | Ubuntu 22.04 | Ubuntu 22.04 / 24.04 |
| Python | 3.10 | 3.10 |

> **Note:** Qwen3:14b needs ~9 GB VRAM. BGE-M3 needs ~3 GB. TTS loads/unloads per request (Ollama is evicted from VRAM before TTS runs).

---

## Prerequisites

### 1. Ollama + Qwen3

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull the LLM (≈9 GB)
ollama pull qwen3:14b
```

### 2. Python 3.10 virtualenv

```bash
python3.10 -m venv venv
source venv/bin/activate
```

---

## Installation

```bash
git clone https://github.com/bprankster/alatoo_helpdesk.git
cd alatoo_helpdesk

source venv/bin/activate

# Step 1 — base packages (works on any OS)
pip install -r requirements.txt

# Step 2 — GPU/server packages (Ubuntu + CUDA only)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-server.txt

# Step 3 — Kyrgyz TTS (install v1 only, no deps override)
pip install kani-tts==1.0.1 --no-deps
```

> **Important:** Do NOT install `kani-tts-2`. Both packages write to the same `kani_tts/` namespace. Only `kani-tts==1.0.1` produces correct audio with the `nineninesix/kani-tts-400m-ky` model.

---

## Configuration

All settings live in `config.yaml`. Key sections:

```yaml
llm:
  model: "qwen3:14b"
  base_url: "http://localhost:11434"

embeddings:
  dense_model: "BAAI/bge-m3"
  device: "cuda"

tts:
  model: "nineninesix/kani-tts-400m-ky"
  enabled_platforms: ["web"]   # add "telegram" to enable there too
  output_dir: "./audio_out"

stt:
  model: "nineninesix/kyrgyz-whisper-medium"
  device: "cuda"

telegram:
  bot_token: ""          # set your BotFather token
  officer_chat_id: ""    # Telegram chat ID for human handoff
  webhook_url: ""        # public HTTPS URL for webhook
```

---

## Data Ingestion

The knowledge base is built from PDFs and manually written text files.

### Directory structure

```
data/
├── raw/
│   ├── pdfs/          ← Place university PDF brochures here
│   └── manual/        ← Place .txt files with hand-written facts here
├── chromadb/          ← Auto-generated vector store (git-ignored)
├── ort_thresholds.json
└── riasec_mapping.json
```

### Running ingestion

```bash
source venv/bin/activate
python data_ingestion/embedder.py
```

This extracts PDFs, chunks text (500 chars / 50 overlap), embeds with BGE-M3, and persists to ChromaDB. Re-run whenever you add new documents.

---

## Running the Server

```bash
source venv/bin/activate
python -m api.main
```

The server starts on `http://0.0.0.0:8000`.

| Endpoint | Description |
|----------|-------------|
| `GET /kiosk` | Web kiosk UI (Gradio) |
| `POST /chat` | JSON chat API |
| `POST /voice` | Audio upload → answer |
| `GET /health` | Health check |
| `POST /telegram` | Telegram webhook receiver |

### Run in background

```bash
nohup python -m api.main > /tmp/alatoo_server.log 2>&1 &
tail -f /tmp/alatoo_server.log
```

---

## Project Structure

```
alatoo_helpdesk/
├── agent/
│   ├── core.py               # ReAct agent, language detection, Kyrgyz cleanup
│   ├── guardrails.py         # Injection & off-topic filtering
│   ├── router.py             # KyrgyzBERT intent classifier (optional fast path)
│   ├── session.py            # Per-user session state with RIASEC progress
│   └── tools/
│       ├── kb_search.py          # Hybrid BM25 + BGE-M3 RAG search
│       ├── ort_validator.py      # ORT score → admission check
│       ├── program_comparator.py # Side-by-side programme comparison
│       ├── orientation_engine.py # Adaptive Holland RIASEC survey (5 questions)
│       └── human_handoff.py      # Telegram escalation to admissions officer
├── api/
│   ├── main.py               # FastAPI app + Gradio mount + lifespan startup
│   ├── chat_endpoint.py      # REST chat/voice routes
│   └── telegram_bot.py       # Telegram webhook handler
├── ui/
│   └── kiosk.py              # Gradio web kiosk (multilingual, TTS player, examples)
├── tts/
│   └── kani_tts.py           # kani-tts wrapper: load-on-demand, VRAM management
├── voice/
│   └── stt.py                # faster-whisper wrapper
├── data_ingestion/
│   ├── embedder.py           # PDF → chunks → BGE-M3 → ChromaDB
│   ├── pdf_extractor.py      # PyPDF text extraction
│   ├── chunker.py            # Sliding window chunker
│   └── refresh.py            # Re-index helper
├── retrieval/
│   └── chroma_store.py       # Hybrid BM25 + dense retrieval
├── classifier/
│   ├── train.py              # Fine-tune KyrgyzBERT intent classifier
│   ├── predict.py            # Inference
│   └── dataset.py            # Training data
├── evaluation/
│   └── evaluate.py           # RAG evaluation (ROUGE, accuracy)
├── data/
│   ├── ort_thresholds.json   # ORT score thresholds per programme
│   └── riasec_mapping.json   # Holland RIASEC → MUA faculty/programme mapping
├── config.yaml               # All settings
├── config.py                 # Python constants loaded from config.yaml
├── requirements.txt          # Base deps (any OS)
└── requirements-server.txt   # GPU deps (Ubuntu + CUDA only)
```

---

## Key Design Decisions

### VRAM Management
Qwen3:14b and BGE-M3 together exhaust a 16 GB GPU. TTS uses a separate 400M model. Before loading TTS, the system calls `POST /api/generate` with `keep_alive: 0` to evict Ollama from VRAM, synthesizes audio, then unloads TTS. BGE-M3 stays resident since it's needed for every query.

### TTS Dependencies
`kani-tts==1.0.1` and `kani-tts-2` both install into the `kani_tts/` namespace. Installing both causes v2 to overwrite v1 files, making the model produce garbage audio. Install **only** `kani-tts==1.0.1 --no-deps`. Compatible versions: `transformers==4.57.6`, `tokenizers==0.22.0`.

### Kyrgyz Language Quality
Qwen3 conflates Kyrgyz and Kazakh. The system applies two fixes:
1. Explicit negative instructions in every Kyrgyz-mode prompt ("do not use ғ/қ — those are Kazakh letters").
2. Post-processing via `_fix_kyrgyz()` in `agent/core.py` that strips Kazakh characters and replaces common Kazakh word forms.

### Orientation Survey
The 5-question Holland RIASEC survey generates each question adaptively using the previous answers to target uncertain dimensions. Questions are generated by Qwen3 (`/no_think` mode for clean JSON output). If LLM generation fails, a bank of 5 static fallback questions is used. Results are scored by faculty-RIASEC overlap (not flat lookup), so the primary recommendation is always the best-matching faculty, with specific programme names and career paths listed.

### Language Detection
The agent detects Kyrgyz by checking for unique Kyrgyz Unicode characters (ң/ү/ө) or a vocabulary list of ~40 Kyrgyz-specific words. Russian is detected by Cyrillic range. Everything else is treated as English.

---

## Telegram Bot Setup

1. Create a bot with [@BotFather](https://t.me/BotFather), copy the token.
2. Set `telegram.bot_token` and `telegram.officer_chat_id` in `config.yaml`.
3. Expose the server publicly (nginx + HTTPS, or ngrok for testing).
4. Set `telegram.webhook_url` to your public URL.
5. Restart the server — the webhook is registered at startup.

---

## Classifier (Optional Fast Path)

Set `agent.use_classifier: true` in `config.yaml` to enable KyrgyzBERT intent routing (bypasses ReAct for clear-cut queries like ORT validation).

To fine-tune the classifier:

```bash
pip install -r requirements-classifier.txt
python classifier/train.py
```

Trained weights are saved to `classifier/model/` (git-ignored; copy manually between machines).

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `CUDA out of memory` at startup | Ollama holding VRAM from previous run | `curl -X POST http://localhost:11434/api/generate -d '{"model":"qwen3:14b","keep_alive":0}'` |
| TTS audio is garbage / wrong language | `kani-tts-2` installed alongside `kani-tts` | `pip uninstall kani-tts kani-tts-2 -y && pip install kani-tts==1.0.1 --no-deps` |
| `ImportError: tokenizers>=0.21,<0.22` | Wrong tokenizers version | `pip install tokenizers==0.22.0` |
| Server starts but ChromaDB empty | Ingestion not run | `python data_ingestion/embedder.py` |
| Kyrgyz responses contain ғ/қ | LLM Kazakh contamination | Already mitigated by `_fix_kyrgyz()` in `agent/core.py` |
| Orientation gives wrong first question | ReAct agent not calling tool | `start_orientation()` in `kiosk.py` now calls the engine directly, bypassing ReAct |

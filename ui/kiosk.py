"""
kiosk.py — Gradio web kiosk UI.

Two tabs: Chat (quick questions + voice sidebar + TTS) + Orientation (Radio answer buttons).
Language selector as compact dropdown in header.
"""

import os
import re
import sys
import uuid
from pathlib import Path

_GRADIO_TMP = Path(__file__).parent.parent / "tmp" / "gradio"
_GRADIO_TMP.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("GRADIO_TEMP_DIR", str(_GRADIO_TMP))

import gradio as gr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent import guardrails
from agent.core import run_agent
from agent.session import clear_session, get_session

TITLE = "Ала-Тоо Университети"

# ── Language constants ──────────────────────────────────────────────────────────
LANG_KY = "🇰🇬 Кыргызча"
LANG_RU = "🇷🇺 Русский"
LANG_EN = "🇬🇧 English"
DEFAULT_LANG = LANG_RU

STRINGS = {
    LANG_KY: {
        "chat_ph":            "Суроолоруңузду жазыңыз…",
        "send":               "Жөнөт",
        "send_voice":         "🎙 Жөнөт",
        "reset":              "Тазалоо",
        "not_recognized":     "Үн таанылган жок. Кайра аракет кылыңыз.",
        "orient_start_label": "🎯 Кесип тандоо тестин баштоо",
        "orient_ph":          "же өз сөзүңүз менен жазыңыз…",
        "orient_send":        "Жооп бер",
        "orient_reset":       "Жаңыдан баштоо",
        "orient_trigger":     "Мен кесип тандоо тестин өтүп, ылайыктуу адистикти аныктоого жардам берүүнү суранам.",
        "orient_user_label":  "🎯 Кесип тандоо тести",
        "orient_hint":        "Тест 5 суроодон турат. Вариантты тандаңыз же өз жообуңузду жазыңыз.",
        "voice_section":      "🎙 Үн киргизүү",
        "tts_section":        "🔊 Жооп угуу",
        "quick_label":        "⚡ Тез суроолор",
    },
    LANG_RU: {
        "chat_ph":            "Введите вопрос…",
        "send":               "Отправить",
        "send_voice":         "🎙 Отправить",
        "reset":              "Очистить",
        "not_recognized":     "Речь не распознана. Попробуйте ещё раз.",
        "orient_start_label": "🎯 Начать тест профориентации",
        "orient_ph":          "или напишите своими словами…",
        "orient_send":        "Ответить",
        "orient_reset":       "Начать заново",
        "orient_trigger":     "Я хочу пройти тест на профориентацию. Помоги определить подходящую специальность.",
        "orient_user_label":  "🎯 Тест профориентации",
        "orient_hint":        "Тест из 5 вопросов. Выберите вариант или напишите своими словами.",
        "voice_section":      "🎙 Голосовой ввод",
        "tts_section":        "🔊 Аудио ответ",
        "quick_label":        "⚡ Быстрые вопросы",
    },
    LANG_EN: {
        "chat_ph":            "Type your question…",
        "send":               "Send",
        "send_voice":         "🎙 Send",
        "reset":              "Clear",
        "not_recognized":     "Could not recognise speech. Please try again.",
        "orient_start_label": "🎯 Start career orientation test",
        "orient_ph":          "or write in your own words…",
        "orient_send":        "Answer",
        "orient_reset":       "Restart",
        "orient_trigger":     "I want to take the career orientation test to find the best major for me.",
        "orient_user_label":  "🎯 Career orientation test",
        "orient_hint":        "5-question test. Choose an option or write your own answer.",
        "voice_section":      "🎙 Voice input",
        "tts_section":        "🔊 Audio reply",
        "quick_label":        "⚡ Quick questions",
    },
}

# ── Quick questions (example chips per language) ────────────────────────────────

EXAMPLES_KY = [
    "Менин ОРТ упайым 145. МУАга поступление кыла аламбы?",
    "ОРТ 183 болсо кандай скидка берилет?",
    "МУАга поступление үчүн кандай документтер керек?",
    "МУАда кайсы факультеттер бар?",
    "IT жана Инженерия факультети жөнүндө айтып бер",
    "Медицина факультетине кирүү үчүн эмне керек?",
    "Экономика факультети жөнүндө маалымат бер",
    "Кесиптик тест өткүм келет. Кайсы адистик мага ылайыктуу?",
]

EXAMPLES_RU = [
    "Мой ОРТ 145 — могу поступить в МУА?",
    "Какая скидка при ОРТ 183?",
    "Какие документы нужны для поступления в МУА?",
    "МУАда кайсы факультеттер бар?",
    "IT жана Инженерия факультети жөнүндө айтып бер",
    "Медицина факультетине кирүү үчүн эмне керек?",
    "Кесиптик тест өткүм келет. Кайсы адистик мага ылайыктуу?",
]

EXAMPLES_EN = [
    "My ORT score is 145. Can I apply to Ala-Too University?",
    "What discount do I get with ORT score 183?",
    "What documents are required for admission to Ala-Too?",
    "What faculties are available at Ala-Too University?",
    "Tell me about the Engineering and IT faculty at Ala-Too",
    "I want to take a career orientation test to find the right major",
]

# ── CSS ─────────────────────────────────────────────────────────────────────────
CSS = """
/* ── Reset & base ───────────────────────────────────────── */
.gradio-container {
    max-width: 100% !important;
    margin: 0 !important;
    padding: 0 0 32px !important;
    background: #EEF2F7 !important;
    font-family: 'Segoe UI', -apple-system, system-ui, sans-serif !important;
}
body { background: #EEF2F7 !important; margin: 0 !important; }
footer { display: none !important; }

/* Constrain tabs content to 1020px centered */
.tabs {
    max-width: 1020px !important;
    margin: 0 auto !important;
    padding: 0 12px !important;
}

/* ── Header row ─────────────────────────────────────────── */
#header-row {
    gap: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}
#header-row > .wrap, #header-row > div {
    gap: 0 !important; padding: 0 !important;
}

/* ── Language dropdown ──────────────────────────────────── */
#lang-col {
    background: #001A4D !important;
    border-radius: 0 !important;
    padding: 0 20px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-end !important;
    min-height: 82px !important;
    max-width: 230px !important;
}
#lang-col > div, #lang-col .block, #lang-col .wrap {
    background: transparent !important;
    border: none !important; padding: 0 !important; box-shadow: none !important;
}

/* Gradio 5 custom dropdown (not a native <select>) */
#lang-dd {
    background: rgba(255,255,255,0.12) !important;
    border: 1.5px solid rgba(255,255,255,0.35) !important;
    border-radius: 8px !important;
    min-width: 160px !important;
    cursor: pointer !important;
    transition: border-color 0.15s, background 0.15s !important;
}
#lang-dd:hover, #lang-dd:focus-within {
    background: rgba(255,255,255,0.22) !important;
    border-color: rgba(255,255,255,0.70) !important;
}
/* All text/input elements inside dropdown → white */
#lang-dd .block, #lang-dd .wrap, #lang-dd input,
#lang-dd button, #lang-dd span, #lang-dd div {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: white !important;
    font-size: 0.86rem !important;
    min-height: unset !important;
}
#lang-dd svg, #lang-dd svg path, #lang-dd svg polyline {
    stroke: rgba(255,255,255,0.80) !important;
}
/* Dropdown options list */
#lang-dd ul, #lang-dd .options, #lang-dd [data-testid="dropdown-menu"] {
    background: #002060 !important;
    border: 1px solid rgba(255,255,255,0.20) !important;
    border-radius: 6px !important;
}
#lang-dd li, #lang-dd .item {
    color: white !important;
    font-size: 0.86rem !important;
}
#lang-dd li:hover, #lang-dd .item:hover {
    background: rgba(255,255,255,0.15) !important;
    color: white !important;
}

/* ── Tab nav ────────────────────────────────────────────── */
.tab-nav {
    background: white !important;
    border: none !important;
    border-bottom: 2px solid #E8EDF4 !important;
    border-radius: 12px 12px 0 0 !important;
    padding: 0 20px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
}
.tab-nav button {
    font-size: 0.92rem !important;
    font-weight: 500 !important;
    color: #94A3B8 !important;
    padding: 14px 26px !important;
    border: none !important;
    background: transparent !important;
    border-bottom: 3px solid transparent !important;
    margin-bottom: -2px !important;
    border-radius: 0 !important;
    transition: color 0.15s !important;
    letter-spacing: 0.01em !important;
}
.tab-nav button.selected {
    color: #1A3A6E !important;
    font-weight: 700 !important;
    border-bottom-color: #1A3A6E !important;
}
.tab-nav button:hover:not(.selected) { color: #475569 !important; }

/* ── Tab content panel ──────────────────────────────────── */
.tabitem {
    background: white !important;
    border: 1.5px solid #E8EDF4 !important;
    border-top: none !important;
    border-radius: 0 0 14px 14px !important;
    padding: 20px 20px 16px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
}

/* ── Chatbot ────────────────────────────────────────────── */
#chatbot, #orient-chatbot {
    border: 1.5px solid #E8EDF4 !important;
    border-radius: 12px !important;
    background: #FAFCFF !important;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.04) !important;
}

/* ── Text inputs ────────────────────────────────────────── */
.chat-input textarea, .orient-input textarea {
    border: 1.5px solid #D1DCE8 !important;
    border-radius: 10px !important;
    font-size: 0.93rem !important;
    background: #FAFCFF !important;
    padding: 11px 15px !important;
    resize: none !important;
    line-height: 1.55 !important;
    transition: border-color 0.15s, box-shadow 0.15s !important;
    color: #1E293B !important;
}
.chat-input textarea:focus, .orient-input textarea:focus {
    border-color: #1A3A6E !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(26,58,110,0.10) !important;
    background: white !important;
}

/* ── Primary send button ────────────────────────────────── */
.send-btn button {
    background: linear-gradient(135deg, #1A3A6E, #2352A0) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.90rem !important;
    min-height: 48px !important;
    padding: 0 20px !important;
    letter-spacing: 0.01em !important;
    transition: opacity 0.15s, transform 0.12s, box-shadow 0.15s !important;
    box-shadow: 0 2px 6px rgba(26,58,110,0.30) !important;
    white-space: nowrap !important;
}
.send-btn button:hover {
    opacity: 0.92 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(26,58,110,0.36) !important;
}
.send-btn button:active { transform: translateY(0) !important; }

/* ── Example chips (quick questions) ───────────────────── */
.examples-holder table { border: none !important; }
.examples-holder table td { padding: 2px 4px !important; border: none !important; }
.examples-holder table td button, table.examples td button {
    font-size: 0.80rem !important;
    border-radius: 16px !important;
    padding: 5px 13px !important;
    background: white !important;
    border: 1.5px solid #CBD5E1 !important;
    color: #334155 !important;
    font-weight: 500 !important;
    white-space: nowrap !important;
    transition: all 0.13s !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
}
.examples-holder table td button:hover, table.examples td button:hover {
    background: #EEF4FF !important;
    border-color: #93B4E8 !important;
    color: #1A3A6E !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 2px 6px rgba(26,58,110,0.12) !important;
}

/* ── Gold orientation start button ─────────────────────── */
.orient-start button {
    background: linear-gradient(135deg, #B8811E, #D4A832 50%, #B8811E) !important;
    color: #1C0F00 !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 0.96rem !important;
    min-height: 54px !important;
    width: 100% !important;
    letter-spacing: 0.01em !important;
    box-shadow: 0 3px 10px rgba(196,151,42,0.32) !important;
    transition: transform 0.12s, box-shadow 0.15s, opacity 0.15s !important;
}
.orient-start button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 18px rgba(196,151,42,0.42) !important;
    opacity: 0.95 !important;
}

/* ── Ghost / secondary buttons ──────────────────────────── */
.ghost-btn button {
    background: transparent !important;
    color: #94A3B8 !important;
    border: 1.5px solid #E2E8F0 !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    min-height: 36px !important;
    padding: 0 14px !important;
    transition: all 0.15s !important;
    white-space: nowrap !important;
}
.ghost-btn button:hover {
    color: #64748B !important;
    border-color: #CBD5E1 !important;
    background: #F8FAFC !important;
}

/* ── Voice sidebar ──────────────────────────────────────── */
#voice-sidebar {
    background: #F6F9FC !important;
    border: 1.5px solid #E8EDF4 !important;
    border-radius: 12px !important;
    padding: 16px 14px 12px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
}
.sidebar-label {
    font-size: 0.74rem !important;
    font-weight: 700 !important;
    color: #64748B !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    margin: 0 0 10px 0 !important;
    display: block !important;
}
.sidebar-divider {
    border: none !important;
    border-top: 1px solid #E2E8F0 !important;
    margin: 14px 0 !important;
}
.voice-send-btn button {
    background: #475569 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 0.86rem !important;
    font-weight: 600 !important;
    min-height: 40px !important;
    width: 100% !important;
    margin-top: 8px !important;
    transition: background 0.15s !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.12) !important;
}
.voice-send-btn button:hover { background: #334155 !important; }

/* ── TTS audio player ───────────────────────────────────── */
#tts-player { border-radius: 8px !important; overflow: hidden !important; }

/* ── Orientation radio options ──────────────────────────── */
#orient-options .wrap, #orient-options > div {
    display: flex !important;
    flex-direction: column !important;
    gap: 8px !important;
    padding: 6px 0 10px !important;
    background: transparent !important;
    border: none !important;
}
#orient-options label {
    background: white !important;
    border: 1.5px solid #E2E8F0 !important;
    border-radius: 10px !important;
    padding: 13px 18px !important;
    cursor: pointer !important;
    font-size: 0.92rem !important;
    color: #374151 !important;
    transition: all 0.15s !important;
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
    line-height: 1.45 !important;
}
#orient-options label:hover {
    border-color: #1A3A6E !important;
    background: #EEF4FF !important;
    color: #1A3A6E !important;
    transform: translateX(4px) !important;
    box-shadow: 0 3px 10px rgba(26,58,110,0.12) !important;
}
#orient-options label:has(input:checked) {
    background: #1A3A6E !important;
    border-color: #1A3A6E !important;
    color: white !important;
}
#orient-options input[type="radio"] {
    position: absolute !important; opacity: 0 !important;
    pointer-events: none !important; width: 0 !important; height: 0 !important;
}
"""

HEADER_LEFT_HTML = """
<div style="background:linear-gradient(135deg,#001A4D 0%,#1A3A6E 100%);
    border-radius:0;padding:14px 24px;
    display:flex;align-items:center;gap:18px;
    font-family:'Segoe UI',system-ui,sans-serif;flex:1;min-width:0;">
  <img src="/ui_assets/logo.png" onerror="this.style.display='none'"
       style="height:52px;width:auto;object-fit:contain;flex-shrink:0;
              filter:drop-shadow(0 1px 3px rgba(0,0,0,0.25));" alt="МУА">
  <div style="flex:1;min-width:0;">
    <div style="font-size:1.08rem;font-weight:700;color:#FFFFFF;
                line-height:1.25;white-space:nowrap;letter-spacing:0.01em;">
      Ала-Тоо Эл Аралык Университети
    </div>
    <div style="font-size:0.72rem;color:rgba(255,255,255,0.55);
                margin-top:3px;letter-spacing:0.02em;">
      Ala-Too International University &nbsp;·&nbsp; Bishkek, Kyrgyzstan
    </div>
  </div>
  <div style="text-align:right;font-size:0.73rem;line-height:2.1;flex-shrink:0;">
    <a href="https://wa.me/996555820000"
       style="color:rgba(255,255,255,0.82);text-decoration:none;
              transition:color 0.15s;">📞 +996 555 820 000</a><br>
    <a href="mailto:admission@alatoo.edu.kg"
       style="color:rgba(255,255,255,0.82);text-decoration:none;">
       📧 admission@alatoo.edu.kg</a>
  </div>
</div>
"""

GOLD_LINE_HTML = (
    '<div style="height:3px;'
    'background:linear-gradient(90deg,#8B5E10,#D4A832 35%,#F0CC5A 50%,#D4A832 65%,#8B5E10);'
    'margin:0;"></div>'
)

FOOTER_HTML = """
<div style="max-width:1020px;margin:10px auto 0;padding:12px 12px 4px;
    text-align:center;font-size:0.72rem;color:#94A3B8;
    border-top:1px solid #E8EDF4;font-family:system-ui;line-height:1.8;">
  🏫 ул. Анкара 1/10, мкр. «Тунгуч», Бишкек, D-блок, 1 этаж &nbsp;·&nbsp;
  <a href="https://wa.me/996555820000"
     style="color:#1A3A6E;text-decoration:none;font-weight:500;">+996 555 820 000</a>
  &nbsp;·&nbsp;
  <a href="mailto:admission@alatoo.edu.kg"
     style="color:#1A3A6E;text-decoration:none;font-weight:500;">admission@alatoo.edu.kg</a>
</div>
"""


# ── Helpers ─────────────────────────────────────────────────────────────────────

def _generate_user_id() -> str:
    return f"web_{uuid.uuid4().hex[:12]}"

def _msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}

def _s(lang: str, key: str) -> str:
    return STRINGS.get(lang, STRINGS[LANG_RU]).get(key, "")

def _parse_orient_options(text: str) -> list[str]:
    options = []
    for line in text.split("\n"):
        m = re.match(r"^\s*([А-ГA-D])\.\s*(.+)", line)
        if m:
            options.append(f"{m.group(1)}. {m.group(2).strip()}")
    return options

def _tts_available() -> bool:
    try:
        from tts.kani_tts import is_enabled
        return is_enabled("web")
    except Exception:
        return False

def _ollama_unload():
    """Tell Ollama to release the LLM from VRAM before TTS loads."""
    try:
        import urllib.request, json, yaml
        cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        model = cfg["llm"]["model"]
        base_url = cfg["llm"]["base_url"].rstrip("/")
        payload = json.dumps({"model": model, "keep_alive": 0}).encode()
        req = urllib.request.Request(
            f"{base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
        print(f"[tts] Ollama unloaded {model} from VRAM")
    except Exception as e:
        print(f"[tts] Ollama unload failed (continuing): {e}")


def _run_tts(text: str) -> str | None:
    try:
        from tts.kani_tts import speak
        import uuid, shutil
        _ollama_unload()          # free VRAM before loading TTS model
        src = speak(text)
        if src is None:
            return None
        # Copy into Gradio's temp dir so it's served via /file= endpoint
        dst = str(_GRADIO_TMP / f"{uuid.uuid4().hex[:12]}_tts.wav")
        shutil.copy2(src, dst)
        print(f"[tts] ready: {dst}")
        return dst
    except Exception as e:
        print(f"[tts] error: {e}")
    return None

def tts_for_last_reply(history: list, lang: str):
    print(f"[tts] tts_for_last_reply called: lang={lang!r} history_len={len(history) if history else 0}")
    if not history:
        return gr.update(visible=False)
    last = history[-1]
    if not (isinstance(last, dict) and last.get("role") == "assistant"):
        return gr.update(visible=False)
    text = last.get("content", "")
    if not text.strip():
        return gr.update(visible=False)
    audio = _run_tts(text)
    return gr.update(value=audio, visible=audio is not None)

def _sidebar_section_html(label: str) -> str:
    return f'<span class="sidebar-label">{label}</span>'

def _orient_hint_html(lang: str) -> str:
    hint = _s(lang, "orient_hint")
    return (
        f'<p style="font-size:0.79rem;color:#94A3B8;margin:4px 0 12px;'
        f'padding:0;line-height:1.5;">{hint}</p>'
    )



# ── Callbacks ────────────────────────────────────────────────────────────────────

def change_language(lang: str):
    s = STRINGS.get(lang, STRINGS[LANG_RU])
    return (
        gr.update(placeholder=s["chat_ph"]),
        gr.update(value=s["send"]),
        gr.update(value=s["send_voice"]),
        gr.update(value=s["reset"]),
        gr.update(value=s["orient_start_label"]),
        gr.update(placeholder=s["orient_ph"]),
        gr.update(value=s["orient_send"]),
        gr.update(value=s["orient_reset"]),
        gr.update(value=_orient_hint_html(lang)),
        gr.update(visible=(lang == LANG_KY)),   # col_ky
        gr.update(visible=(lang == LANG_RU)),   # col_ru
        gr.update(visible=(lang == LANG_EN)),   # col_en
        gr.update(value=None, visible=False),   # tts_player — hide on lang switch
        lang,                                   # lang_state
    )


def respond(message: str, history: list, user_id: str):
    if not message.strip():
        return "", history, user_id
    guard = guardrails.check(message)
    if guard.blocked or guard.off_topic:
        return "", history + [_msg("user", message), _msg("assistant", guard.reply)], user_id
    session = get_session(user_id)
    reply = run_agent(message, session)
    return "", history + [_msg("user", message), _msg("assistant", reply)], user_id


def respond_voice(audio_path: str | None, history: list, user_id: str, lang: str):
    if audio_path is None:
        return history, user_id
    from voice.stt import transcribe
    text = transcribe(audio_path)
    if not text:
        return history + [_msg("assistant", _s(lang, "not_recognized"))], user_id
    guard = guardrails.check(text)
    if guard.blocked or guard.off_topic:
        return history + [_msg("user", f"🎙 {text}"), _msg("assistant", guard.reply)], user_id
    session = get_session(user_id)
    reply = run_agent(text, session)
    return history + [_msg("user", f"🎙 {text}"), _msg("assistant", reply)], user_id


def start_orientation(history: list, user_id: str, lang: str):
    s = STRINGS.get(lang, STRINGS[LANG_RU])
    clear_session(user_id + ":orient")
    session = get_session(user_id + ":orient")

    # Bypass ReAct — call orientation engine directly so the first question always appears
    from agent.core import _set_active_session
    from agent.tools.orientation_engine import orientation_engine
    _set_active_session(session)
    session.add_message("user", s["orient_user_label"])
    reply = orientation_engine(s["orient_trigger"])
    session.add_message("assistant", reply)

    new_history = [_msg("user", s["orient_user_label"]), _msg("assistant", reply)]
    options = _parse_orient_options(reply)
    return new_history, user_id, gr.update(choices=options, value=None, visible=bool(options))


def select_orient_option(evt: gr.SelectData, history: list, user_id: str):
    choice = evt.value if evt else None
    if not choice:
        return history, user_id, gr.update()
    letter = choice[0]
    session = get_session(user_id + ":orient")
    reply = run_agent(letter, session)
    new_history = history + [_msg("user", choice), _msg("assistant", reply)]
    options = _parse_orient_options(reply)
    return new_history, user_id, gr.update(choices=options, value=None, visible=bool(options))


def respond_orientation(message: str, history: list, user_id: str):
    if not message.strip():
        return "", history, user_id, gr.update()
    guard = guardrails.check(message)
    if guard.blocked:
        return "", history + [_msg("user", message), _msg("assistant", guard.reply)], user_id, gr.update()
    session = get_session(user_id + ":orient")
    reply = run_agent(message, session)
    options = _parse_orient_options(reply)
    return "", history + [_msg("user", message), _msg("assistant", reply)], user_id, gr.update(choices=options, value=None, visible=bool(options))


def reset_chat(user_id: str):
    clear_session(user_id)
    return [], _generate_user_id(), gr.update(value=None, visible=False)


def reset_orientation(user_id: str):
    clear_session(user_id + ":orient")
    return [], user_id, gr.update(choices=[], value=None, visible=False)


# ── Layout ───────────────────────────────────────────────────────────────────────

def build_demo() -> gr.Blocks:
    default_s = STRINGS[DEFAULT_LANG]
    _tts_on = _tts_available()

    with gr.Blocks(title=TITLE, theme=gr.themes.Base(), css=CSS) as demo:

        user_id_state = gr.State(_generate_user_id)
        lang_state    = gr.State(DEFAULT_LANG)

        # ── Header ──────────────────────────────────────────────────────────────
        with gr.Row(elem_id="header-row", equal_height=True):
            gr.HTML(HEADER_LEFT_HTML)
            with gr.Column(scale=0, min_width=200, elem_id="lang-col"):
                lang_dd = gr.Dropdown(
                    choices=[LANG_KY, LANG_RU, LANG_EN],
                    value=DEFAULT_LANG,
                    label=None,
                    container=False,
                    interactive=True,
                    elem_id="lang-dd",
                )
        gr.HTML(GOLD_LINE_HTML)

        # ── Tabs ────────────────────────────────────────────────────────────────
        with gr.Tabs():

            # ─── Chat tab ───────────────────────────────────────────────────────
            with gr.TabItem("💬 Чат / Chat"):
                with gr.Row(equal_height=False):

                    # Main chat column
                    with gr.Column(scale=5):
                        chatbot = gr.Chatbot(
                            type="messages", height=420,
                            show_label=False, elem_id="chatbot",
                            bubble_full_width=False,
                        )
                        # TTS player — appears below chatbot after each reply
                        tts_player = gr.Audio(
                            label="🔊 Аудио ответ",
                            type="filepath",
                            autoplay=True,
                            visible=False,
                            interactive=False,
                            container=True,
                            elem_id="tts-player",
                        )
                        with gr.Row():
                            msg_input = gr.Textbox(
                                placeholder=default_s["chat_ph"],
                                lines=1, max_lines=4,
                                show_label=False, container=False,
                                scale=5, elem_classes=["chat-input"],
                            )
                            send_btn = gr.Button(
                                default_s["send"],
                                scale=1, min_width=110,
                                elem_classes=["send-btn"],
                            )
                        with gr.Row():
                            reset_btn = gr.Button(
                                default_s["reset"],
                                elem_classes=["ghost-btn"],
                                scale=0, min_width=90,
                            )

                        # ── Quick questions (example chips) ─────────────────────
                        with gr.Column(visible=(DEFAULT_LANG == LANG_KY)) as col_ky:
                            gr.Examples(EXAMPLES_KY, inputs=msg_input,
                                        label="💬 Суроолор мисалдары",
                                        examples_per_page=len(EXAMPLES_KY))
                        with gr.Column(visible=(DEFAULT_LANG == LANG_RU)) as col_ru:
                            gr.Examples(EXAMPLES_RU, inputs=msg_input,
                                        label="💬 Примеры вопросов",
                                        examples_per_page=len(EXAMPLES_RU))
                        with gr.Column(visible=(DEFAULT_LANG == LANG_EN)) as col_en:
                            gr.Examples(EXAMPLES_EN, inputs=msg_input,
                                        label="💬 Example questions",
                                        examples_per_page=len(EXAMPLES_EN))

                    # Voice sidebar
                    with gr.Column(scale=2, min_width=200, elem_id="voice-sidebar"):
                        gr.HTML(_sidebar_section_html(default_s["voice_section"]))
                        audio_input = gr.Audio(
                            label=None,
                            type="filepath",
                            sources=["microphone", "upload"],
                            container=False,
                        )
                        send_voice_btn = gr.Button(
                            default_s["send_voice"],
                            elem_classes=["voice-send-btn"],
                        )

            # ─── Orientation tab ────────────────────────────────────────────────
            with gr.TabItem("🎯 Профориентация / Orientation"):
                orient_chatbot = gr.Chatbot(
                    type="messages", height=360,
                    show_label=False, elem_id="orient-chatbot",
                    bubble_full_width=False,
                )
                start_orient_btn = gr.Button(
                    default_s["orient_start_label"],
                    elem_classes=["orient-start"],
                )
                orient_options = gr.Radio(
                    choices=[], value=None, label=None,
                    container=False, visible=False, interactive=True,
                    elem_id="orient-options",
                )
                orient_hint = gr.HTML(_orient_hint_html(DEFAULT_LANG))
                with gr.Row():
                    orient_input = gr.Textbox(
                        placeholder=default_s["orient_ph"],
                        lines=1, max_lines=3,
                        show_label=False, container=False,
                        scale=5, elem_classes=["orient-input"],
                    )
                    orient_send_btn = gr.Button(
                        default_s["orient_send"],
                        scale=1, min_width=110,
                        elem_classes=["send-btn"],
                    )
                orient_reset_btn = gr.Button(
                    default_s["orient_reset"],
                    elem_classes=["ghost-btn"],
                )

        gr.HTML(FOOTER_HTML)

        # ── Event bindings ───────────────────────────────────────────────────────

        lang_dd.change(
            fn=change_language,
            inputs=[lang_dd],
            outputs=[
                msg_input, send_btn, send_voice_btn, reset_btn,
                start_orient_btn, orient_input, orient_send_btn, orient_reset_btn,
                orient_hint,
                col_ky, col_ru, col_en,
                tts_player,
                lang_state,
            ],
        )

        _tts_outputs = [tts_player] if _tts_on else []

        # ── Helper to wire send + optional TTS ──────────────────────────────────
        def _wire(event):
            if _tts_on:
                event.then(fn=tts_for_last_reply,
                           inputs=[chatbot, lang_state],
                           outputs=_tts_outputs)

        # Chat — text input
        _wire(send_btn.click(fn=respond, inputs=[msg_input, chatbot, user_id_state],
                             outputs=[msg_input, chatbot, user_id_state]))
        _wire(msg_input.submit(fn=respond, inputs=[msg_input, chatbot, user_id_state],
                               outputs=[msg_input, chatbot, user_id_state]))

        # Chat — voice
        _wire(send_voice_btn.click(fn=respond_voice,
                                   inputs=[audio_input, chatbot, user_id_state, lang_state],
                                   outputs=[chatbot, user_id_state]))

        # Chat — reset
        reset_btn.click(fn=reset_chat, inputs=[user_id_state],
                        outputs=[chatbot, user_id_state, tts_player])


        # ── Orientation ──────────────────────────────────────────────────────────
        start_orient_btn.click(
            fn=start_orientation,
            inputs=[orient_chatbot, user_id_state, lang_state],
            outputs=[orient_chatbot, user_id_state, orient_options],
        )
        orient_options.select(
            fn=select_orient_option,
            inputs=[orient_chatbot, user_id_state],
            outputs=[orient_chatbot, user_id_state, orient_options],
        )
        orient_send_btn.click(
            fn=respond_orientation,
            inputs=[orient_input, orient_chatbot, user_id_state],
            outputs=[orient_input, orient_chatbot, user_id_state, orient_options],
        )
        orient_input.submit(
            fn=respond_orientation,
            inputs=[orient_input, orient_chatbot, user_id_state],
            outputs=[orient_input, orient_chatbot, user_id_state, orient_options],
        )
        orient_reset_btn.click(
            fn=reset_orientation,
            inputs=[user_id_state],
            outputs=[orient_chatbot, user_id_state, orient_options],
        )

    return demo


if __name__ == "__main__":
    build_demo().launch(server_port=7860)

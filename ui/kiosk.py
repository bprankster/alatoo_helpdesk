"""
kiosk.py — Gradio web kiosk UI.

Two tabs: Chat (voice sidebar + TTS) + Orientation (Radio answer buttons).
Language selector as compact dropdown in header.
"""

import os
import re
import sys
import uuid
from pathlib import Path

# Must come before `import gradio` to redirect upload temp dir
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
        "or_type":            "же өз жообуңузду жазыңыз:",
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
        "or_type":            "или напишите своими словами:",
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
        "or_type":            "or write in your own words:",
    },
}

EXAMPLES_KY = [
    "Менин ОРТ балым 138. CS факультетине кире аламбы?",
    "ОРТ 195 болсо кандай скидка берилет?",
    "Медицина факультетине кирүү үчүн кандай предметтер керек?",
    "IT жана Экономика программаларын салыштырып бер",
    "МУАга документтерди кантип жана качан тапшырам?",
    "Кибербезопасность адистиги кандай мүмкүнчүлүктөрдү берет?",
    "Юриспруденция факультетине минималдуу ОРТ упайы канча?",
    "Психология жана Журналистика программаларынын айырмасы эмне?",
    "МУАда окуу акысы канча? Бөлүп төлөсө болобу?",
    "Ала-Тоо университетинде жатакана барбы?",
    "Инженерия факультетине кирүү үчүн математикадан канча упай керек?",
    "Адамды чакырыңыз — кабылуу боюнча сүйлөшкүм келет",
]
EXAMPLES_RU = [
    "Мой ОРТ 145 — могу поступить на Computer Science?",
    "Какая скидка при ОРТ 183?",
    "Какие документы нужны для поступления в МУА?",
    "Сравни программы Кибербезопасность и Психология",
    "Расскажи про факультет инженерии и информатики",
    "Хочу поговорить с сотрудником приёмной комиссии",
]
EXAMPLES_EN = [
    "What is the minimum ORT score to apply to Ala-Too?",
    "What discount do I get with ORT score 200?",
    "Compare Computer Engineering and Data Science programs",
    "What documents are required for admission?",
    "I'd like to speak to an admissions officer",
]

# ── CSS ─────────────────────────────────────────────────────────────────────────
CSS = """
/* ── Base ──────────────────────────────────────────── */
.gradio-container {
    max-width: 980px !important;
    margin: 0 auto !important;
    padding: 0 10px 24px !important;
    background: #F1F5F9 !important;
    font-family: 'Segoe UI', -apple-system, system-ui, sans-serif !important;
}
body { background: #F1F5F9 !important; }
footer { display: none !important; }

/* ── Header row — flush, no gap ─────────────────────── */
#header-row {
    gap: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}
#header-row > .wrap,
#header-row > div {
    gap: 0 !important;
    padding: 0 !important;
}

/* ── Language dropdown column (dark blue) ───────────── */
#lang-col {
    background: #002366 !important;
    border-radius: 0 12px 0 0 !important;
    padding: 0 18px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-end !important;
    min-height: 78px !important;
    max-width: 200px !important;
}
#lang-col > div,
#lang-col .block,
#lang-col .wrap {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    box-shadow: none !important;
}
#lang-dd select {
    background: rgba(255,255,255,0.10) !important;
    border: 1.5px solid rgba(255,255,255,0.28) !important;
    color: #FFFFFF !important;
    border-radius: 8px !important;
    font-size: 0.84rem !important;
    padding: 6px 10px !important;
    cursor: pointer !important;
    min-width: 148px !important;
    outline: none !important;
    transition: border-color 0.15s !important;
}
#lang-dd select:hover,
#lang-dd select:focus {
    border-color: rgba(255,255,255,0.55) !important;
}
#lang-dd option {
    background: #002F6C !important;
    color: white !important;
}

/* ── Tab nav ──────────────────────────────────────── */
.tab-nav {
    background: white !important;
    border: none !important;
    border-bottom: 2px solid #E5E7EB !important;
    border-radius: 12px 12px 0 0 !important;
    padding: 0 16px !important;
    gap: 0 !important;
}
.tab-nav button {
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    color: #9CA3AF !important;
    padding: 12px 24px !important;
    border: none !important;
    background: transparent !important;
    border-bottom: 2.5px solid transparent !important;
    margin-bottom: -2px !important;
    border-radius: 0 !important;
    transition: color 0.12s !important;
}
.tab-nav button.selected {
    color: #002366 !important;
    font-weight: 700 !important;
    border-bottom-color: #002366 !important;
}
.tab-nav button:hover:not(.selected) { color: #4B5563 !important; }

/* ── Tab content ─────────────────────────────────── */
.tabitem {
    background: white !important;
    border: 1.5px solid #E5E7EB !important;
    border-top: none !important;
    border-radius: 0 0 12px 12px !important;
    padding: 16px 18px 14px !important;
}

/* ── Chatbots ────────────────────────────────────── */
#chatbot, #orient-chatbot {
    border: none !important;
    background: transparent !important;
}

/* ── Text inputs ─────────────────────────────────── */
.chat-input textarea, .orient-input textarea {
    border: 1.5px solid #E2E8F0 !important;
    border-radius: 10px !important;
    font-size: 0.92rem !important;
    background: white !important;
    padding: 10px 14px !important;
    resize: none !important;
    line-height: 1.5 !important;
    transition: border-color 0.12s, box-shadow 0.12s !important;
}
.chat-input textarea:focus, .orient-input textarea:focus {
    border-color: #002366 !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(0,35,102,0.08) !important;
}

/* ── Primary send button ─────────────────────────── */
.send-btn button {
    background: #002366 !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    min-height: 46px !important;
    padding: 0 18px !important;
    transition: background 0.12s !important;
    white-space: nowrap !important;
}
.send-btn button:hover { background: #003087 !important; }

/* ── Gold orientation start button ──────────────── */
.orient-start button {
    background: linear-gradient(135deg, #C4972A, #DBA830) !important;
    color: #1C0F00 !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    min-height: 52px !important;
    width: 100% !important;
    box-shadow: 0 2px 8px rgba(196,151,42,0.28) !important;
    transition: transform 0.12s, box-shadow 0.12s !important;
}
.orient-start button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(196,151,42,0.38) !important;
}

/* ── Ghost / secondary buttons ───────────────────── */
.ghost-btn button {
    background: transparent !important;
    color: #9CA3AF !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 8px !important;
    font-size: 0.8rem !important;
    min-height: 36px !important;
    transition: all 0.12s !important;
    white-space: nowrap !important;
}
.ghost-btn button:hover { color: #6B7280 !important; border-color: #D1D5DB !important; }

/* ── Voice sidebar ───────────────────────────────── */
#voice-sidebar {
    background: #F8FAFC !important;
    border: 1.5px solid #E2E8F0 !important;
    border-radius: 12px !important;
    padding: 14px 14px 10px !important;
}
.sidebar-label {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    color: #64748B !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    margin: 0 0 8px 0 !important;
}
.sidebar-divider {
    border: none !important;
    border-top: 1px solid #E2E8F0 !important;
    margin: 12px 0 !important;
}
.voice-send-btn button {
    background: #475569 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 0.84rem !important;
    font-weight: 600 !important;
    min-height: 38px !important;
    width: 100% !important;
    margin-top: 6px !important;
    transition: background 0.12s !important;
}
.voice-send-btn button:hover { background: #334155 !important; }

/* ── TTS audio player ────────────────────────────── */
#tts-player {
    border-radius: 8px !important;
    overflow: hidden !important;
}

/* ── Orientation answer radio options ────────────── */
#orient-options .wrap,
#orient-options > div {
    display: flex !important;
    flex-direction: column !important;
    gap: 8px !important;
    padding: 4px 0 8px !important;
    background: transparent !important;
    border: none !important;
}
#orient-options label {
    background: white !important;
    border: 1.5px solid #E2E8F0 !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
    cursor: pointer !important;
    font-size: 0.9rem !important;
    color: #374151 !important;
    transition: all 0.15s !important;
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
    line-height: 1.4 !important;
}
#orient-options label:hover {
    border-color: #002366 !important;
    background: #EFF6FF !important;
    color: #002366 !important;
    transform: translateX(3px) !important;
    box-shadow: 0 2px 8px rgba(0,35,102,0.12) !important;
}
#orient-options label:has(input:checked) {
    background: #002366 !important;
    border-color: #002366 !important;
    color: white !important;
}
#orient-options input[type="radio"] {
    position: absolute !important;
    opacity: 0 !important;
    pointer-events: none !important;
    width: 0 !important;
    height: 0 !important;
}

/* ── Example chips ───────────────────────────────── */
.examples-holder table { border: none !important; }
.examples-holder table td { padding: 2px 3px !important; border: none !important; }
.examples-holder table td button, table.examples td button {
    font-size: 0.78rem !important;
    border-radius: 14px !important;
    padding: 4px 12px !important;
    background: #F8FAFC !important;
    border: 1px solid #E2E8F0 !important;
    color: #475569 !important;
    font-weight: 500 !important;
    white-space: nowrap !important;
    transition: all 0.12s !important;
}
.examples-holder table td button:hover, table.examples td button:hover {
    background: #EFF6FF !important;
    border-color: #BFDBFE !important;
    color: #1D4ED8 !important;
}
"""

HEADER_LEFT_HTML = """
<div style="background:#002366;border-radius:12px 0 0 0;padding:14px 22px;
    display:flex;align-items:center;gap:16px;font-family:'Segoe UI',system-ui,sans-serif;
    flex:1;min-width:0;">
  <img src="/ui_assets/logo.png" onerror="this.style.display='none'"
       style="height:50px;width:auto;object-fit:contain;flex-shrink:0;" alt="МУА">
  <div style="flex:1;min-width:0;">
    <div style="font-size:1.1rem;font-weight:700;color:#FFFFFF;line-height:1.2;white-space:nowrap;">
      Ала-Тоо Эл Аралык Университети
    </div>
    <div style="font-size:0.73rem;color:rgba(255,255,255,0.6);margin-top:2px;">
      Ala-Too International University &nbsp;·&nbsp; Bishkek
    </div>
  </div>
  <div style="text-align:right;font-size:0.73rem;line-height:2;flex-shrink:0;">
    <a href="https://wa.me/996555820000"
       style="color:rgba(255,255,255,0.8);text-decoration:none;">📞 +996 555 820 000</a><br>
    <a href="mailto:admission@alatoo.edu.kg"
       style="color:rgba(255,255,255,0.8);text-decoration:none;">📧 admission@alatoo.edu.kg</a>
  </div>
</div>
"""

GOLD_LINE_HTML = (
    '<div style="height:3px;background:linear-gradient(90deg,#C4972A,#E8B842 50%,#C4972A);'
    'margin:0;border-radius:0;"></div>'
)

FOOTER_HTML = """
<div style="text-align:center;padding:10px 4px 4px;font-size:0.73rem;color:#94A3B8;
    border-top:1px solid #E2E8F0;margin-top:8px;font-family:system-ui;">
  🏫 ул. Анкара 1/10, мкр. «Тунгуч», Бишкек, D-блок, 1 этаж &nbsp;·&nbsp;
  <a href="https://wa.me/996555820000" style="color:#002366;text-decoration:none;">+996 555 820 000</a>
  &nbsp;·&nbsp;
  <a href="mailto:admission@alatoo.edu.kg" style="color:#002366;text-decoration:none;">admission@alatoo.edu.kg</a>
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
    """Extract A/B/C/D option lines from an orientation question response."""
    options = []
    for line in text.split("\n"):
        m = re.match(r"^\s*([A-D])\.\s*(.+)", line)
        if m:
            options.append(f"{m.group(1)}. {m.group(2).strip()}")
    return options

def _tts_available() -> bool:
    """Return True if kani-tts is installed and enabled for web."""
    try:
        from tts.kani_tts import is_enabled
        return is_enabled("web")
    except Exception:
        return False

def _run_tts(text: str) -> str | None:
    """Generate TTS audio. Returns wav path or None."""
    try:
        from tts.kani_tts import speak
        return speak(text)
    except Exception as e:
        print(f"[tts] error: {e}")
    return None

def tts_for_last_reply(history: list):
    """Called via .then() after text response — generates audio for the last assistant message."""
    if not history:
        return gr.update(visible=False)
    last = history[-1]
    if isinstance(last, dict) and last.get("role") == "assistant":
        text = last.get("content", "")
    else:
        return gr.update(visible=False)
    audio = _run_tts(text)
    return gr.update(value=audio, visible=audio is not None)

def _sidebar_section_html(label: str) -> str:
    return f'<p class="sidebar-label">{label}</p>'

def _orient_hint_html(lang: str) -> str:
    hint = _s(lang, "orient_hint")
    return (
        f'<p style="font-size:0.78rem;color:#9CA3AF;margin:4px 0 10px;padding:0;">'
        f'{hint}</p>'
    )


# ── Callbacks ────────────────────────────────────────────────────────────────────

def change_language(lang: str):
    s = STRINGS.get(lang, STRINGS[LANG_RU])
    return (
        gr.update(placeholder=s["chat_ph"]),           # msg_input
        gr.update(value=s["send"]),                     # send_btn
        gr.update(value=s["send_voice"]),               # send_voice_btn
        gr.update(value=s["reset"]),                    # reset_btn
        gr.update(value=s["orient_start_label"]),       # start_orient_btn
        gr.update(placeholder=s["orient_ph"]),          # orient_input
        gr.update(value=s["orient_send"]),              # orient_send_btn
        gr.update(value=s["orient_reset"]),             # orient_reset_btn
        gr.update(value=_orient_hint_html(lang)),       # orient_hint
        gr.update(visible=(lang == LANG_KY)),           # col_ky
        gr.update(visible=(lang == LANG_RU)),           # col_ru
        gr.update(visible=(lang == LANG_EN)),           # col_en
        lang,                                           # lang_state
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
    reply = run_agent(s["orient_trigger"], session)
    new_history = [_msg("user", s["orient_user_label"]), _msg("assistant", reply)]
    options = _parse_orient_options(reply)
    options_update = gr.update(choices=options, value=None, visible=bool(options))
    return new_history, user_id, options_update


def select_orient_option(evt: gr.SelectData, history: list, user_id: str):
    """Handle a click on one of the A/B/C/D Radio option buttons.
    Uses gr.SelectData so it only fires on explicit user click, never on programmatic reset."""
    choice = evt.value if evt else None
    if not choice:
        return history, user_id, gr.update()
    letter = choice[0]  # "A. some text" → "A"
    session = get_session(user_id + ":orient")
    reply = run_agent(letter, session)
    new_history = history + [_msg("user", choice), _msg("assistant", reply)]
    options = _parse_orient_options(reply)
    options_update = gr.update(choices=options, value=None, visible=bool(options))
    return new_history, user_id, options_update


def respond_orientation(message: str, history: list, user_id: str):
    """Free-text fallback for orientation — skips off-topic guardrail."""
    if not message.strip():
        return "", history, user_id, gr.update()
    guard = guardrails.check(message)
    if guard.blocked:
        return "", history + [_msg("user", message), _msg("assistant", guard.reply)], user_id, gr.update()
    session = get_session(user_id + ":orient")
    reply = run_agent(message, session)
    options = _parse_orient_options(reply)
    options_update = gr.update(choices=options, value=None, visible=bool(options))
    return "", history + [_msg("user", message), _msg("assistant", reply)], user_id, options_update


def reset_chat(user_id: str):
    clear_session(user_id)
    return [], _generate_user_id(), gr.update(value=None, visible=False)  # tts_player


def reset_orientation(user_id: str):
    clear_session(user_id + ":orient")
    return [], user_id, gr.update(choices=[], value=None, visible=False)


# ── Layout ───────────────────────────────────────────────────────────────────────

def build_demo() -> gr.Blocks:
    default_s = STRINGS[DEFAULT_LANG]

    _tts_on = _tts_available()  # checked once at startup

    with gr.Blocks(title=TITLE, theme=gr.themes.Base(), css=CSS) as demo:

        user_id_state = gr.State(_generate_user_id)
        lang_state    = gr.State(DEFAULT_LANG)

        # ── Header: logo/title on left, language dropdown on right ──────────────
        with gr.Row(elem_id="header-row", equal_height=True):
            gr.HTML(HEADER_LEFT_HTML)
            with gr.Column(scale=0, min_width=190, elem_id="lang-col"):
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

            # ─── Chat tab ──────────────────────────────────────────────────────
            with gr.TabItem("💬 Чат / Chat"):
                with gr.Row(equal_height=False):

                    # Main chat column
                    with gr.Column(scale=5):
                        chatbot = gr.Chatbot(
                            type="messages", height=440,
                            show_label=False, elem_id="chatbot",
                            bubble_full_width=False,
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
                                scale=1, min_width=90,
                            )

                        # Language-filtered example chips
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

                    # Voice sidebar (right)
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
                        with gr.Column(visible=_tts_on, elem_id="tts-section"):
                            gr.HTML('<hr class="sidebar-divider">')
                            gr.HTML(_sidebar_section_html(default_s["tts_section"]))
                            tts_player = gr.Audio(
                                label=None,
                                type="filepath",
                                autoplay=True,
                                visible=False,
                                interactive=False,
                                container=False,
                                elem_id="tts-player",
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

                # Clickable answer option buttons (auto-submit on click)
                orient_options = gr.Radio(
                    choices=[],
                    value=None,
                    label=None,
                    container=False,
                    visible=False,
                    interactive=True,
                    elem_id="orient-options",
                )

                orient_hint = gr.HTML(_orient_hint_html(DEFAULT_LANG))

                # Free-text fallback input
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
                lang_state,
            ],
        )

        # Chat — text (text shows immediately, TTS loads after via .then())
        _tts_outputs = [tts_player] if _tts_on else []
        _tts_fn      = tts_for_last_reply if _tts_on else None

        _send_ev = send_btn.click(
            fn=respond,
            inputs=[msg_input, chatbot, user_id_state],
            outputs=[msg_input, chatbot, user_id_state],
        )
        if _tts_on:
            _send_ev.then(fn=_tts_fn, inputs=[chatbot], outputs=_tts_outputs)

        _submit_ev = msg_input.submit(
            fn=respond,
            inputs=[msg_input, chatbot, user_id_state],
            outputs=[msg_input, chatbot, user_id_state],
        )
        if _tts_on:
            _submit_ev.then(fn=_tts_fn, inputs=[chatbot], outputs=_tts_outputs)

        # Chat — voice
        _voice_ev = send_voice_btn.click(
            fn=respond_voice,
            inputs=[audio_input, chatbot, user_id_state, lang_state],
            outputs=[chatbot, user_id_state],
        )
        if _tts_on:
            _voice_ev.then(fn=_tts_fn, inputs=[chatbot], outputs=_tts_outputs)

        # Chat — reset
        reset_btn.click(
            fn=reset_chat,
            inputs=[user_id_state],
            outputs=[chatbot, user_id_state, tts_player],
        )

        # Orientation — start
        start_orient_btn.click(
            fn=start_orientation,
            inputs=[orient_chatbot, user_id_state, lang_state],
            outputs=[orient_chatbot, user_id_state, orient_options],
        )

        # Orientation — Radio option click (auto-submits on explicit user click only)
        orient_options.select(
            fn=select_orient_option,
            inputs=[orient_chatbot, user_id_state],
            outputs=[orient_chatbot, user_id_state, orient_options],
        )

        # Orientation — free-text fallback
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

        # Orientation — reset
        orient_reset_btn.click(
            fn=reset_orientation,
            inputs=[user_id_state],
            outputs=[orient_chatbot, user_id_state, orient_options],
        )

    return demo


if __name__ == "__main__":
    build_demo().launch(server_port=7860)

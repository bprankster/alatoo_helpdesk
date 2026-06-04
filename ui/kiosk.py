"""
kiosk.py — Gradio web kiosk UI for the admissions agent.

Mounts at /kiosk via FastAPI.
Each browser tab gets its own isolated session via gr.State().
Logo: place ui/assets/logo.png — served at /ui_assets/logo.png.
"""

import os
import sys
import uuid
from pathlib import Path

import gradio as gr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent import guardrails
from agent.core import run_agent
from agent.session import clear_session, get_session

_GRADIO_TMP = Path(__file__).parent.parent / "tmp" / "gradio"
_GRADIO_TMP.mkdir(parents=True, exist_ok=True)

TITLE = "Ала-Тоо Университети — Кабылуу"

# ── Language constants ─────────────────────────────────────────────────────────
LANG_KY = "🇰🇬  Кыргызча"
LANG_RU = "🇷🇺  Русский"
LANG_EN = "🇬🇧  English"
DEFAULT_LANG = LANG_RU

STRINGS = {
    LANG_KY: {
        "placeholder": "Суроолоруңузду жазыңыз…",
        "send": "➤  Жөнөт",
        "send_voice": "🎙️  Үнүмдү жөнөт",
        "orient": "🎯  Кесип тандоо тести",
        "reset": "🔄  Жаңы суббат",
        "voice_title": "### 🎙️ Үн менен жазуу",
        "voice_hint": "Кыргыз, орус же англис тилинде сүйлөңүз.",
        "examples_label": "💬 Суроолор мисалдары",
        "orient_trigger": "Мен кесип тандоо тестин өтүп, ылайыктуу адистикти аныктоого жардам берүүнү суранам.",
        "not_recognized": "Үн таанылган жок. Кайра аракет кылыңыз.",
    },
    LANG_RU: {
        "placeholder": "Введите вопрос…",
        "send": "➤  Отправить",
        "send_voice": "🎙️  Отправить голос",
        "orient": "🎯  Тест профориентации",
        "reset": "🔄  Новая сессия",
        "voice_title": "### 🎙️ Голосовой ввод",
        "voice_hint": "Говорите на кыргызском, русском или английском.",
        "examples_label": "💬 Примеры вопросов",
        "orient_trigger": "Я хочу пройти тест на профориентацию. Помоги определить подходящую специальность.",
        "not_recognized": "Речь не распознана. Попробуйте ещё раз.",
    },
    LANG_EN: {
        "placeholder": "Type your question…",
        "send": "➤  Send",
        "send_voice": "🎙️  Send voice",
        "orient": "🎯  Career orientation test",
        "reset": "🔄  New session",
        "voice_title": "### 🎙️ Voice input",
        "voice_hint": "Speak in Kyrgyz, Russian, or English.",
        "examples_label": "💬 Example questions",
        "orient_trigger": "I want to take the career orientation test to find the best major for me.",
        "not_recognized": "Could not recognise speech. Please try again.",
    },
}


def _voice_hint_html(lang: str) -> str:
    text = STRINGS.get(lang, STRINGS[LANG_RU])["voice_hint"]
    return (
        f'<div style="font-size:0.74rem;color:#64748b;line-height:1.6;'
        f'background:#F0F4F8;border-radius:8px;padding:8px 10px;'
        f'border:1px solid #dde3ee;margin-top:8px;">{text}</div>'
    )


def _s(lang: str, key: str) -> str:
    return STRINGS.get(lang, STRINGS[LANG_RU]).get(key, "")


# ── Examples per language ───────────────────────────────────────────────────────
EXAMPLES_KY = [
    "Менин ОРТ балым 138. CS факультетине кире аламбы?",
    "ОРТ 195 болсо кандай скидка берилет?",
    "Медицина факультетине кирүү үчүн кандай предметтер керек?",
    "IT жана Экономика программаларын салыштырып бер",
    "МУАга документтерди кантип жана качан тапшырам?",
    "Кайсы адистикти тандасам билбейм — жардам бер",
    "Адамды чакырыңыз — кабылуу боюнча сүйлөшкүм келет",
]

EXAMPLES_RU = [
    "Мой ОРТ 145 — могу поступить на Computer Science?",
    "Какая скидка при ОРТ 183?",
    "Какие документы нужны для поступления в МУА?",
    "Сравни программы Кибербезопасность и Психология",
    "Расскажи про факультет инженерии и информатики",
    "Не знаю какую специальность выбрать — помоги разобраться",
    "Хочу поговорить с сотрудником приёмной комиссии",
]

EXAMPLES_EN = [
    "What is the minimum ORT score to apply to Ala-Too?",
    "What discount do I get with ORT score 200?",
    "Compare Computer Engineering and Data Science programs",
    "What documents are required for admission?",
    "I'm not sure which major to choose — can you help?",
    "I'd like to speak to an admissions officer",
]

# ── Styles ─────────────────────────────────────────────────────────────────────
CSS = """
.gradio-container {
    max-width: 1060px !important;
    margin: 0 auto !important;
    padding: 0 !important;
    background: #F0F4F8 !important;
    font-family: 'Segoe UI', system-ui, sans-serif !important;
}
body, .dark { background: #F0F4F8 !important; }
footer { display: none !important; }

#chatbox {
    border-radius: 14px !important;
    border: 1px solid #dde3ee !important;
    background: white !important;
    box-shadow: 0 1px 8px rgba(0,35,102,0.07) !important;
}

/* ── Language selector ── */
#lang-select { margin: 0 0 12px !important; }
#lang-select > div.wrap {
    display: flex !important;
    gap: 8px !important;
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    flex-wrap: nowrap !important;
}
#lang-select label {
    flex: 1 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 11px 8px !important;
    border: 1.5px solid #dde3ee !important;
    border-radius: 10px !important;
    background: white !important;
    font-size: 0.92rem !important;
    font-weight: 500 !important;
    color: #475569 !important;
    cursor: pointer !important;
    transition: all 0.15s !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
}
#lang-select label:has(input[type="radio"]:checked) {
    background: #002366 !important;
    border-color: #002366 !important;
    color: white !important;
    font-weight: 700 !important;
    box-shadow: 0 3px 10px rgba(0,35,102,0.28) !important;
}
#lang-select label:hover:not(:has(input[type="radio"]:checked)) {
    border-color: #94a3b8 !important;
    background: #f8fafc !important;
}
#lang-select input[type="radio"] { display: none !important; }

/* ── Proforientation button ── */
#btn-orient button {
    background: linear-gradient(135deg, #C4972A 0%, #E8B842 100%) !important;
    border: none !important;
    color: #001544 !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border-radius: 10px !important;
    min-height: 54px !important;
    box-shadow: 0 3px 10px rgba(196,151,42,0.4) !important;
    transition: transform 0.12s, box-shadow 0.12s !important;
}
#btn-orient button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 5px 16px rgba(196,151,42,0.5) !important;
}

/* ── Reset button ── */
#btn-reset button {
    background: white !important;
    border: 1.5px solid #cbd5e1 !important;
    color: #64748b !important;
    font-size: 0.85rem !important;
    border-radius: 10px !important;
    min-height: 54px !important;
}
#btn-reset button:hover { border-color: #94a3b8 !important; color: #475569 !important; }

/* ── Send text button ── */
#btn-send button {
    background: #002366 !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    border-radius: 10px !important;
    min-height: 52px !important;
    border: none !important;
    transition: background 0.15s !important;
}
#btn-send button:hover { background: #003087 !important; }

/* ── Text input ── */
#txt-input textarea {
    border-radius: 10px !important;
    border: 1.5px solid #dde3ee !important;
    font-size: 0.95rem !important;
    background: white !important;
    padding: 12px 14px !important;
    transition: border-color 0.15s !important;
}
#txt-input textarea:focus { border-color: #002366 !important; outline: none !important; }

/* ── Voice panel card ── */
#voice-card {
    background: white;
    border: 1px solid #dde3ee;
    border-radius: 14px;
    padding: 16px 14px 12px;
    box-shadow: 0 1px 6px rgba(0,35,102,0.06);
}

/* ── Big voice send button ── */
#btn-voice button {
    background: #002366 !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    border-radius: 10px !important;
    min-height: 70px !important;
    border: none !important;
    width: 100% !important;
    transition: background 0.15s !important;
    margin-top: 6px !important;
}
#btn-voice button:hover { background: #003087 !important; }

/* ── Example chips ── */
.examples-holder table { border: none !important; }
.examples-holder table td { padding: 3px 4px !important; border: none !important; }
.examples-holder table td button,
table.examples td button {
    font-size: 0.80rem !important;
    border-radius: 20px !important;
    padding: 5px 14px !important;
    background: white !important;
    border: 1.5px solid #c8d4e8 !important;
    color: #002366 !important;
    font-weight: 500 !important;
    white-space: nowrap !important;
    transition: all 0.13s !important;
}
.examples-holder table td button:hover,
table.examples td button:hover {
    background: #002366 !important;
    color: white !important;
    border-color: #002366 !important;
}
"""

HEADER_HTML = """
<div style="
    background: linear-gradient(135deg, #001544 0%, #002366 55%, #003580 100%);
    border-radius: 14px; overflow: hidden;
    box-shadow: 0 4px 20px rgba(0,35,102,0.28);
    margin-bottom: 14px; font-family: system-ui, sans-serif;
">
  <div style="padding: 18px 26px 14px; display:flex; align-items:center; gap:18px;">
    <img src="/ui_assets/logo.png"
         onerror="this.style.display='none'"
         style="height:60px; width:auto; object-fit:contain; flex-shrink:0; filter:brightness(1.05);"
         alt="МУА">
    <div style="flex:1;">
      <div style="font-size:1.4rem; font-weight:800; color:white; letter-spacing:-0.01em; line-height:1.15;">
        Ала-Тоо Эл Аралык Университети
      </div>
      <div style="font-size:0.82rem; color:rgba(255,255,255,0.65); margin-top:3px;">
        Ala-Too International University &nbsp;·&nbsp; Бишкек, Кыргызстан
      </div>
    </div>
    <div style="text-align:right; font-size:0.79rem; color:rgba(255,255,255,0.75); line-height:2;">
      📞 +996 555 820 000<br>
      📧 admission@alatoo.edu.kg
    </div>
  </div>
  <div style="height:3px; background:linear-gradient(90deg,#C4972A,#E8B842 50%,#C4972A);"></div>
  <div style="background:rgba(0,0,0,0.22);padding:7px 26px;display:flex;gap:22px;flex-wrap:wrap;
      font-size:0.77rem;color:rgba(255,255,255,0.75);">
    <span>🎓 Кабылуу 2024–2025</span>
    <span>🤖 AI жардамчы / ИИ-помощник</span>
    <span>📋 5 факультет · 23 программа</span>
  </div>
</div>
"""

FOOTER_HTML = """
<div style="text-align:center;padding:12px 4px 8px;font-size:0.76rem;color:#94a3b8;
    border-top:1px solid #e2e8f0;margin-top:8px;font-family:system-ui;">
  🏫 ул. Анкара (Горький) 1/10, мкр. «Тунгуч», Бишкек, D-блок, 1 этаж
  &nbsp;·&nbsp;
  <a href="https://wa.me/996555820000" style="color:#002366;text-decoration:none;">📞 +996 555 820 000</a>
  &nbsp;·&nbsp;
  <a href="mailto:admission@alatoo.edu.kg" style="color:#002366;text-decoration:none;">📧 admission@alatoo.edu.kg</a>
</div>
"""


# ── Helpers ────────────────────────────────────────────────────────────────────

def _generate_user_id() -> str:
    return f"web_{uuid.uuid4().hex[:12]}"


def _msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


# ── Callbacks ──────────────────────────────────────────────────────────────────

def change_language(lang: str):
    s = STRINGS.get(lang, STRINGS[LANG_RU])
    return (
        gr.update(placeholder=s["placeholder"]),
        gr.update(value=s["send"]),
        gr.update(value=s["send_voice"]),
        gr.update(value=s["orient"]),
        gr.update(value=s["reset"]),
        gr.update(value=s["voice_title"]),
        gr.update(value=_voice_hint_html(lang)),
        gr.update(visible=(lang == LANG_KY)),
        gr.update(visible=(lang == LANG_RU)),
        gr.update(visible=(lang == LANG_EN)),
        lang,
    )


def respond(message: str, history: list, user_id: str) -> tuple[str, list, str]:
    if not message.strip():
        return "", history, user_id
    guard = guardrails.check(message)
    if guard.blocked or guard.off_topic:
        return "", history + [_msg("user", message), _msg("assistant", guard.reply)], user_id
    session = get_session(user_id)
    reply = run_agent(message, session)
    return "", history + [_msg("user", message), _msg("assistant", reply)], user_id


def respond_voice(audio_path: str | None, history: list, user_id: str, lang: str) -> tuple[list, str]:
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


def start_orientation(history: list, user_id: str, lang: str) -> tuple[list, str]:
    s = STRINGS.get(lang, STRINGS[LANG_RU])
    session = get_session(user_id)
    reply = run_agent(s["orient_trigger"], session)
    return history + [
        _msg("user", s["orient"]),
        _msg("assistant", reply),
    ], user_id


def reset_session(user_id: str) -> tuple[list, str]:
    clear_session(user_id)
    return [], _generate_user_id()


# ── Layout ─────────────────────────────────────────────────────────────────────

def build_demo() -> gr.Blocks:
    default_s = STRINGS[DEFAULT_LANG]

    with gr.Blocks(
        title=TITLE,
        theme=gr.themes.Base(),
        css=CSS,
        tmp_dir=str(_GRADIO_TMP),
    ) as demo:

        user_id_state = gr.State(_generate_user_id)
        lang_state = gr.State(DEFAULT_LANG)

        gr.HTML(HEADER_HTML)

        # ── Language selector ───────────────────────────────────────────────────
        lang_radio = gr.Radio(
            choices=[LANG_KY, LANG_RU, LANG_EN],
            value=DEFAULT_LANG,
            label=None,
            container=False,
            elem_id="lang-select",
        )

        # ── Quick-action bar ────────────────────────────────────────────────────
        with gr.Row():
            orient_btn = gr.Button(default_s["orient"], elem_id="btn-orient", scale=3)
            reset_btn = gr.Button(default_s["reset"], elem_id="btn-reset", scale=1)

        # ── Chat + voice ────────────────────────────────────────────────────────
        with gr.Row(equal_height=False):

            with gr.Column(scale=5):
                chatbot = gr.Chatbot(
                    type="messages",
                    height=530,
                    show_label=False,
                    elem_id="chatbox",
                    bubble_full_width=False,
                )
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder=default_s["placeholder"],
                        scale=6,
                        container=False,
                        lines=1,
                        max_lines=4,
                        show_label=False,
                        elem_id="txt-input",
                    )
                    send_btn = gr.Button(
                        default_s["send"],
                        scale=1,
                        min_width=110,
                        elem_id="btn-send",
                    )

            with gr.Column(scale=2, min_width=210, elem_id="voice-card"):
                voice_title_md = gr.Markdown(default_s["voice_title"])
                audio_input = gr.Audio(
                    label=None,
                    type="filepath",
                    sources=["microphone", "upload"],
                )
                send_voice_btn = gr.Button(
                    default_s["send_voice"],
                    size="lg",
                    elem_id="btn-voice",
                )
                voice_hint = gr.HTML(_voice_hint_html(DEFAULT_LANG))

        # ── Examples — one set per language ─────────────────────────────────────
        with gr.Column(visible=(DEFAULT_LANG == LANG_KY)) as col_ky:
            gr.Examples(
                examples=EXAMPLES_KY,
                inputs=msg_input,
                label="💬 Суроолор мисалдары",
                examples_per_page=len(EXAMPLES_KY),
            )
        with gr.Column(visible=(DEFAULT_LANG == LANG_RU)) as col_ru:
            gr.Examples(
                examples=EXAMPLES_RU,
                inputs=msg_input,
                label="💬 Примеры вопросов",
                examples_per_page=len(EXAMPLES_RU),
            )
        with gr.Column(visible=(DEFAULT_LANG == LANG_EN)) as col_en:
            gr.Examples(
                examples=EXAMPLES_EN,
                inputs=msg_input,
                label="💬 Example questions",
                examples_per_page=len(EXAMPLES_EN),
            )

        gr.HTML(FOOTER_HTML)

        # ── Event bindings ──────────────────────────────────────────────────────
        lang_radio.change(
            fn=change_language,
            inputs=[lang_radio],
            outputs=[
                msg_input, send_btn, send_voice_btn, orient_btn, reset_btn,
                voice_title_md, voice_hint,
                col_ky, col_ru, col_en,
                lang_state,
            ],
        )

        send_btn.click(
            fn=respond,
            inputs=[msg_input, chatbot, user_id_state],
            outputs=[msg_input, chatbot, user_id_state],
        )
        msg_input.submit(
            fn=respond,
            inputs=[msg_input, chatbot, user_id_state],
            outputs=[msg_input, chatbot, user_id_state],
        )
        send_voice_btn.click(
            fn=respond_voice,
            inputs=[audio_input, chatbot, user_id_state, lang_state],
            outputs=[chatbot, user_id_state],
        )
        orient_btn.click(
            fn=start_orientation,
            inputs=[chatbot, user_id_state, lang_state],
            outputs=[chatbot, user_id_state],
        )
        reset_btn.click(
            fn=reset_session,
            inputs=[user_id_state],
            outputs=[chatbot, user_id_state],
        )

    return demo


if __name__ == "__main__":
    build_demo().launch(server_port=7860)

"""
kiosk.py — Gradio web kiosk UI (two-tab: Chat + Orientation).

Mounts at /kiosk via FastAPI.
Orientation runs in a separate session so single-letter answers (A/B/C/D)
never hit the off-topic guardrail.
"""

import os
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

# ── Language constants ─────────────────────────────────────────────────────────
LANG_KY = "🇰🇬 Кыргызча"
LANG_RU = "🇷🇺 Русский"
LANG_EN = "🇬🇧 English"
DEFAULT_LANG = LANG_RU

STRINGS = {
    LANG_KY: {
        "chat_ph":            "Суроолоруңузду жазыңыз…",
        "send":               "Жөнөт",
        "voice_btn":          "🎙 Үн",
        "reset":              "Тазалоо",
        "not_recognized":     "Үн таанылган жок. Кайра аракет кылыңыз.",
        "orient_start_label": "🎯 Кесип тандоо тестин баштоо",
        "orient_ph":          "А, Б, В же Г… же өз сөзүңүз менен",
        "orient_send":        "Жооп",
        "orient_reset":       "Жаңыдан баштоо",
        "orient_trigger":     "Мен кесип тандоо тестин өтүп, ылайыктуу адистикти аныктоого жардам берүүнү суранам.",
        "orient_user_label":  "🎯 Кесип тандоо тести",
    },
    LANG_RU: {
        "chat_ph":            "Введите вопрос…",
        "send":               "Отправить",
        "voice_btn":          "🎙 Голос",
        "reset":              "Очистить",
        "not_recognized":     "Речь не распознана. Попробуйте ещё раз.",
        "orient_start_label": "🎯 Начать тест профориентации",
        "orient_ph":          "A, B, C или D — или своими словами…",
        "orient_send":        "Ответить",
        "orient_reset":       "Начать заново",
        "orient_trigger":     "Я хочу пройти тест на профориентацию. Помоги определить подходящую специальность.",
        "orient_user_label":  "🎯 Тест профориентации",
    },
    LANG_EN: {
        "chat_ph":            "Type your question…",
        "send":               "Send",
        "voice_btn":          "🎙 Voice",
        "reset":              "Clear",
        "not_recognized":     "Could not recognise speech. Please try again.",
        "orient_start_label": "🎯 Start career orientation test",
        "orient_ph":          "A, B, C or D — or in your own words…",
        "orient_send":        "Answer",
        "orient_reset":       "Restart",
        "orient_trigger":     "I want to take the career orientation test to find the best major for me.",
        "orient_user_label":  "🎯 Career orientation test",
    },
}

ORIENT_HINTS = {
    LANG_KY: "Тест 5 суроодон турат. А/Б/В/Г же өз сөзүңүз менен жооп бериңиз.",
    LANG_RU: "Тест из 5 вопросов. Отвечайте A/B/C/D или своими словами.",
    LANG_EN: "5-question test. Answer A/B/C/D or write in your own words.",
}

def _orient_hint_html(lang: str) -> str:
    return (
        f'<p style="font-size:0.77rem;color:#9CA3AF;margin:4px 0 10px;padding:0;">'
        f'{ORIENT_HINTS.get(lang, ORIENT_HINTS[LANG_RU])}</p>'
    )

# ── Examples ───────────────────────────────────────────────────────────────────
EXAMPLES_KY = [
    "Менин ОРТ балым 138. CS факультетине кире аламбы?",
    "ОРТ 195 болсо кандай скидка берилет?",
    "Медицина факультетине кирүү үчүн кандай предметтер керек?",
    "IT жана Экономика программаларын салыштырып бер",
    "МУАга документтерди кантип жана качан тапшырам?",
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

# ── CSS ────────────────────────────────────────────────────────────────────────
CSS = """
/* ── Base ─────────────────────────────────────────── */
.gradio-container {
    max-width: 920px !important;
    margin: 0 auto !important;
    padding: 0 10px 20px !important;
    background: #F1F5F9 !important;
    font-family: 'Segoe UI', -apple-system, system-ui, sans-serif !important;
}
body { background: #F1F5F9 !important; }
footer { display: none !important; }

/* ── Language pills ─────────────────────────────────── */
#lang-select { margin: 8px 0 4px !important; }
#lang-select .wrap {
    display: flex !important; gap: 6px !important;
    background: transparent !important; border: none !important; padding: 0 !important;
}
#lang-select label {
    padding: 5px 14px !important;
    border: 1.5px solid #E2E8F0 !important; border-radius: 20px !important;
    background: white !important; font-size: 0.82rem !important; font-weight: 500 !important;
    color: #64748B !important; cursor: pointer !important; transition: all 0.12s !important;
    display: flex !important; align-items: center !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04) !important;
}
#lang-select label:has(input[type="radio"]:checked) {
    background: #002366 !important; border-color: #002366 !important;
    color: white !important; font-weight: 600 !important;
    box-shadow: 0 2px 6px rgba(0,35,102,0.2) !important;
}
#lang-select label:hover:not(:has(input[type="radio"]:checked)) {
    border-color: #94A3B8 !important; background: #F8FAFC !important;
}
#lang-select input[type="radio"] { display: none !important; }

/* ── Tab navigation ─────────────────────────────────── */
.tab-nav {
    background: white !important; border: none !important;
    border-bottom: 2px solid #E5E7EB !important;
    border-radius: 12px 12px 0 0 !important;
    padding: 0 14px !important; gap: 0 !important;
}
.tab-nav button {
    font-size: 0.9rem !important; font-weight: 500 !important;
    color: #9CA3AF !important; padding: 11px 22px !important;
    border: none !important; background: transparent !important;
    border-bottom: 2.5px solid transparent !important;
    margin-bottom: -2px !important; border-radius: 0 !important;
    transition: color 0.12s !important;
}
.tab-nav button.selected {
    color: #002366 !important; font-weight: 700 !important;
    border-bottom-color: #002366 !important;
}
.tab-nav button:hover:not(.selected) { color: #4B5563 !important; }

/* ── Tab content ─────────────────────────────────────── */
.tabitem {
    background: white !important;
    border: 1.5px solid #E5E7EB !important; border-top: none !important;
    border-radius: 0 0 12px 12px !important; padding: 16px 18px 10px !important;
}

/* ── Chatbots ─────────────────────────────────────────── */
#chatbot, #orient-chatbot {
    border: none !important; background: transparent !important;
}

/* ── Text inputs ──────────────────────────────────────── */
.chat-input textarea, .orient-input textarea {
    border: 1.5px solid #E2E8F0 !important; border-radius: 10px !important;
    font-size: 0.92rem !important; background: white !important;
    padding: 10px 14px !important; resize: none !important; line-height: 1.5 !important;
    transition: border-color 0.12s, box-shadow 0.12s !important;
}
.chat-input textarea:focus, .orient-input textarea:focus {
    border-color: #002366 !important; outline: none !important;
    box-shadow: 0 0 0 3px rgba(0,35,102,0.08) !important;
}

/* ── Primary button (Send / Answer) ─────────────────── */
.send-btn button {
    background: #002366 !important; color: white !important;
    border: none !important; border-radius: 10px !important;
    font-weight: 600 !important; font-size: 0.88rem !important;
    min-height: 46px !important; padding: 0 16px !important;
    transition: background 0.12s !important; white-space: nowrap !important;
}
.send-btn button:hover { background: #003087 !important; }

/* ── Gold orientation start button ──────────────────── */
.orient-start button {
    background: linear-gradient(135deg, #C4972A, #DBA830) !important;
    color: #1C0F00 !important; border: none !important; border-radius: 10px !important;
    font-weight: 700 !important; font-size: 0.95rem !important;
    min-height: 52px !important; width: 100% !important;
    box-shadow: 0 2px 8px rgba(196,151,42,0.28) !important;
    transition: transform 0.12s, box-shadow 0.12s !important;
}
.orient-start button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(196,151,42,0.38) !important;
}

/* ── Voice button ─────────────────────────────────────── */
.voice-btn button {
    background: white !important; color: #374151 !important;
    border: 1.5px solid #D1D5DB !important; border-radius: 10px !important;
    font-weight: 500 !important; font-size: 0.86rem !important;
    min-height: 42px !important; transition: all 0.12s !important;
}
.voice-btn button:hover {
    background: #EFF6FF !important; border-color: #93C5FD !important; color: #002366 !important;
}

/* ── Ghost / secondary buttons ───────────────────────── */
.ghost-btn button {
    background: transparent !important; color: #9CA3AF !important;
    border: 1px solid #E5E7EB !important; border-radius: 8px !important;
    font-size: 0.8rem !important; min-height: 38px !important;
    transition: all 0.12s !important; white-space: nowrap !important;
}
.ghost-btn button:hover { color: #6B7280 !important; border-color: #D1D5DB !important; }

/* ── Example chips ─────────────────────────────────────── */
.examples-holder table { border: none !important; }
.examples-holder table td { padding: 2px 3px !important; border: none !important; }
.examples-holder table td button, table.examples td button {
    font-size: 0.78rem !important; border-radius: 14px !important;
    padding: 4px 12px !important; background: #F8FAFC !important;
    border: 1px solid #E2E8F0 !important; color: #475569 !important;
    font-weight: 500 !important; white-space: nowrap !important; transition: all 0.12s !important;
}
.examples-holder table td button:hover, table.examples td button:hover {
    background: #EFF6FF !important; border-color: #BFDBFE !important; color: #1D4ED8 !important;
}
"""

HEADER_HTML = """
<div style="background:#002366;border-radius:12px 12px 0 0;padding:14px 22px;
    display:flex;align-items:center;gap:16px;font-family:'Segoe UI',system-ui,sans-serif;">
  <img src="/ui_assets/logo.png" onerror="this.style.display='none'"
       style="height:50px;width:auto;object-fit:contain;flex-shrink:0;" alt="МУА">
  <div style="flex:1;min-width:0;">
    <div style="font-size:1.15rem;font-weight:700;color:#FFFFFF;line-height:1.2;">
      Ала-Тоо Эл Аралык Университети
    </div>
    <div style="font-size:0.75rem;color:rgba(255,255,255,0.6);margin-top:2px;">
      Ala-Too International University &nbsp;·&nbsp; Bishkek
    </div>
  </div>
  <div style="text-align:right;font-size:0.74rem;line-height:1.9;flex-shrink:0;">
    <a href="https://wa.me/996555820000"
       style="color:rgba(255,255,255,0.8);text-decoration:none;">📞 +996 555 820 000</a><br>
    <a href="mailto:admission@alatoo.edu.kg"
       style="color:rgba(255,255,255,0.8);text-decoration:none;">📧 admission@alatoo.edu.kg</a>
  </div>
</div>
<div style="height:3px;background:linear-gradient(90deg,#C4972A,#E8B842 50%,#C4972A);"></div>
"""

FOOTER_HTML = """
<div style="text-align:center;padding:10px 4px 4px;font-size:0.73rem;color:#94A3B8;
    border-top:1px solid #E2E8F0;margin-top:8px;font-family:system-ui;">
  🏫 ул. Анкара 1/10, мкр. «Тунгуч», Бишкек, D-блок, 1 этаж &nbsp;·&nbsp;
  <a href="https://wa.me/996555820000" style="color:#002366;text-decoration:none;">+996 555 820 000</a>
  &nbsp;·&nbsp;
  <a href="mailto:admission@alatoo.edu.kg" style="color:#002366;text-decoration:none;">admission@alatoo.edu.kg</a>
</div>
"""


# ── Helpers ────────────────────────────────────────────────────────────────────

def _generate_user_id() -> str:
    return f"web_{uuid.uuid4().hex[:12]}"

def _msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}

def _s(lang: str, key: str) -> str:
    return STRINGS.get(lang, STRINGS[LANG_RU]).get(key, "")


# ── Callbacks ──────────────────────────────────────────────────────────────────

def change_language(lang: str):
    s = STRINGS.get(lang, STRINGS[LANG_RU])
    return (
        gr.update(placeholder=s["chat_ph"]),                                         # msg_input
        gr.update(value=s["send"]),                                                   # send_btn
        gr.update(value=s["voice_btn"]),                                              # send_voice_btn
        gr.update(value=s["reset"]),                                                  # reset_btn
        gr.update(value=s["orient_start_label"]),                                     # start_orient_btn
        gr.update(placeholder=s["orient_ph"]),                                        # orient_input
        gr.update(value=s["orient_send"]),                                            # orient_send_btn
        gr.update(value=s["orient_reset"]),                                           # orient_reset_btn
        gr.update(value=_orient_hint_html(lang)),                                     # orient_hint
        gr.update(visible=(lang == LANG_KY)),                                         # col_ky
        gr.update(visible=(lang == LANG_RU)),                                         # col_ru
        gr.update(visible=(lang == LANG_EN)),                                         # col_en
        lang,                                                                          # lang_state
    )


def respond(message: str, history: list, user_id: str) -> tuple[str, list, str]:
    if not message.strip():
        return "", history, user_id
    guard = guardrails.check(message)
    if guard.blocked:
        return "", history + [_msg("user", message), _msg("assistant", guard.reply)], user_id
    if guard.off_topic:
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
    if guard.blocked:
        return history + [_msg("user", f"🎙 {text}"), _msg("assistant", guard.reply)], user_id
    if guard.off_topic:
        return history + [_msg("user", f"🎙 {text}"), _msg("assistant", guard.reply)], user_id
    session = get_session(user_id)
    reply = run_agent(text, session)
    return history + [_msg("user", f"🎙 {text}"), _msg("assistant", reply)], user_id


def respond_orientation(message: str, history: list, user_id: str) -> tuple[str, list, str]:
    """
    Orientation tab handler.
    Only blocks prompt injection — never applies off-topic guardrail.
    This lets single-letter answers like 'B' pass through without being rejected.
    Uses a separate session key (':orient') so orientation state is isolated
    from the main chat session.
    """
    if not message.strip():
        return "", history, user_id
    guard = guardrails.check(message)
    if guard.blocked:
        return "", history + [_msg("user", message), _msg("assistant", guard.reply)], user_id
    session = get_session(user_id + ":orient")
    reply = run_agent(message, session)
    return "", history + [_msg("user", message), _msg("assistant", reply)], user_id


def start_orientation(history: list, user_id: str, lang: str) -> tuple[list, str]:
    s = STRINGS.get(lang, STRINGS[LANG_RU])
    clear_session(user_id + ":orient")
    session = get_session(user_id + ":orient")
    reply = run_agent(s["orient_trigger"], session)
    return [
        _msg("user", s["orient_user_label"]),
        _msg("assistant", reply),
    ], user_id


def reset_chat(user_id: str) -> tuple[list, str]:
    clear_session(user_id)
    return [], _generate_user_id()


def reset_orientation(user_id: str) -> tuple[list, str]:
    clear_session(user_id + ":orient")
    return [], user_id


# ── Layout ─────────────────────────────────────────────────────────────────────

def build_demo() -> gr.Blocks:
    default_s = STRINGS[DEFAULT_LANG]

    with gr.Blocks(title=TITLE, theme=gr.themes.Base(), css=CSS) as demo:

        user_id_state = gr.State(_generate_user_id)
        lang_state = gr.State(DEFAULT_LANG)

        # ── Header ─────────────────────────────────────────────────────────────
        gr.HTML(HEADER_HTML)

        # ── Language pills ──────────────────────────────────────────────────────
        lang_radio = gr.Radio(
            choices=[LANG_KY, LANG_RU, LANG_EN],
            value=DEFAULT_LANG,
            label=None,
            container=False,
            elem_id="lang-select",
        )

        # ── Tabs ────────────────────────────────────────────────────────────────
        with gr.Tabs():

            # ─── Chat tab ──────────────────────────────────────────────────────
            with gr.TabItem("💬 Чат / Chat"):
                chatbot = gr.Chatbot(
                    type="messages", height=460, show_label=False,
                    elem_id="chatbot", bubble_full_width=False,
                )
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder=default_s["chat_ph"],
                        lines=1, max_lines=4, show_label=False,
                        container=False, scale=5,
                        elem_classes=["chat-input"],
                    )
                    send_btn = gr.Button(
                        default_s["send"], scale=1, min_width=100,
                        elem_classes=["send-btn"],
                    )
                with gr.Row():
                    audio_input = gr.Audio(
                        label=None, type="filepath",
                        sources=["microphone", "upload"], scale=3,
                    )
                    send_voice_btn = gr.Button(
                        default_s["voice_btn"], scale=1, min_width=90,
                        elem_classes=["voice-btn"],
                    )
                    reset_btn = gr.Button(
                        default_s["reset"], scale=1, min_width=80,
                        elem_classes=["ghost-btn"],
                    )

                # Language-filtered examples
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

            # ─── Orientation tab ────────────────────────────────────────────────
            with gr.TabItem("🎯 Профориентация / Orientation"):
                orient_chatbot = gr.Chatbot(
                    type="messages", height=400, show_label=False,
                    elem_id="orient-chatbot", bubble_full_width=False,
                )
                start_orient_btn = gr.Button(
                    default_s["orient_start_label"],
                    elem_classes=["orient-start"],
                )
                with gr.Row():
                    orient_input = gr.Textbox(
                        placeholder=default_s["orient_ph"],
                        lines=1, max_lines=3, show_label=False,
                        container=False, scale=5,
                        elem_classes=["orient-input"],
                    )
                    orient_send_btn = gr.Button(
                        default_s["orient_send"], scale=1, min_width=100,
                        elem_classes=["send-btn"],
                    )
                orient_hint = gr.HTML(_orient_hint_html(DEFAULT_LANG))
                orient_reset_btn = gr.Button(
                    default_s["orient_reset"],
                    elem_classes=["ghost-btn"],
                )

        gr.HTML(FOOTER_HTML)

        # ── Event bindings ──────────────────────────────────────────────────────
        lang_radio.change(
            fn=change_language,
            inputs=[lang_radio],
            outputs=[
                msg_input, send_btn, send_voice_btn, reset_btn,
                start_orient_btn, orient_input, orient_send_btn, orient_reset_btn,
                orient_hint,
                col_ky, col_ru, col_en,
                lang_state,
            ],
        )

        # Chat
        send_btn.click(
            fn=respond, inputs=[msg_input, chatbot, user_id_state],
            outputs=[msg_input, chatbot, user_id_state],
        )
        msg_input.submit(
            fn=respond, inputs=[msg_input, chatbot, user_id_state],
            outputs=[msg_input, chatbot, user_id_state],
        )
        send_voice_btn.click(
            fn=respond_voice,
            inputs=[audio_input, chatbot, user_id_state, lang_state],
            outputs=[chatbot, user_id_state],
        )
        reset_btn.click(
            fn=reset_chat, inputs=[user_id_state],
            outputs=[chatbot, user_id_state],
        )

        # Orientation
        start_orient_btn.click(
            fn=start_orientation,
            inputs=[orient_chatbot, user_id_state, lang_state],
            outputs=[orient_chatbot, user_id_state],
        )
        orient_send_btn.click(
            fn=respond_orientation,
            inputs=[orient_input, orient_chatbot, user_id_state],
            outputs=[orient_input, orient_chatbot, user_id_state],
        )
        orient_input.submit(
            fn=respond_orientation,
            inputs=[orient_input, orient_chatbot, user_id_state],
            outputs=[orient_input, orient_chatbot, user_id_state],
        )
        orient_reset_btn.click(
            fn=reset_orientation, inputs=[user_id_state],
            outputs=[orient_chatbot, user_id_state],
        )

    return demo


if __name__ == "__main__":
    build_demo().launch(server_port=7860)

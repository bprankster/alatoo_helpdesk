"""
kiosk.py — Gradio web kiosk UI for the admissions agent.

Mounts at /kiosk via FastAPI.
Each browser tab gets its own isolated session via gr.State().
"""

import os
import sys
import uuid

import gradio as gr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent import guardrails
from agent.core import run_agent
from agent.session import clear_session, get_session


TITLE = "Ала-Тоо Университети — Кабылуу / Приёмная комиссия"
PLACEHOLDER = "Суроолоруңузду жазыңыз / Введите вопрос / Type your question…"

EXAMPLES = [
    # ── Kyrgyz ────────────────────────────────────────────
    "Менин ОРТ балым 138. CS факультетине кире аламбы?",
    "ОРТ 195 болсо кандай скидка берилет?",
    "Медицина факультетине кирүү үчүн кандай предметтер керек?",
    "Мен кайсы адистикти тандашымды билбейм — жардамчы бол",
    "IT жана Экономика программаларын салыштырып бер",
    "Инженерия факультетинде кандай адистиктер бар?",
    "МУАга документтерди кантип жана качан тапшырам?",
    "Адамды чакырыңыз — сотрудник менен сүйлөшкүм келет",
    # ── Russian ────────────────────────────────────────────
    "Мой ОРТ 145 — могу поступить на Computer Science?",
    "Какая скидка при ОРТ 183?",
    "Какие документы нужны для поступления в МУА?",
    "Я не знаю, какую специальность выбрать. Помоги.",
    "Сравни программы Кибербезопасность и Психология",
    "Расскажи про факультет инженерии и информатики",
    # ── English ────────────────────────────────────────────
    "What is the minimum ORT score to apply?",
    "Compare Computer Science and Economics programs",
]

CSS = """
/* ── Global ───────────────────────────────────────────────────── */
.gradio-container {
    max-width: 1020px !important;
    margin: 0 auto !important;
    padding-top: 8px !important;
}

/* ── Send button ──────────────────────────────────────────────── */
#btn-send {
    min-height: 52px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
}

/* ── BIG voice button ────────────────────────────────────────── */
#btn-voice {
    min-height: 76px !important;
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    letter-spacing: 0.01em !important;
    margin-top: 10px !important;
}

/* ── Example chips ───────────────────────────────────────────── */
.examples-holder .examples table td button,
.examples table td button {
    font-size: 0.82rem !important;
    border-radius: 20px !important;
    padding: 5px 14px !important;
    background: #f1f5f9 !important;
    border: 1px solid #cbd5e1 !important;
    transition: background 0.15s, border-color 0.15s !important;
    white-space: nowrap !important;
}
.examples-holder .examples table td button:hover,
.examples table td button:hover {
    background: #dbeafe !important;
    border-color: #93c5fd !important;
    color: #1d4ed8 !important;
}

/* ── Chatbot ──────────────────────────────────────────────────── */
#chatbox {
    border-radius: 12px !important;
    border: 1.5px solid #e2e8f0 !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06) !important;
}

/* ── Reset button ─────────────────────────────────────────────── */
#btn-reset {
    border-radius: 8px !important;
    font-size: 0.85rem !important;
}
"""

HEADER_HTML = """
<div style="
    background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 58%, #0ea5e9 100%);
    border-radius: 14px;
    padding: 24px 30px 20px;
    margin-bottom: 14px;
    box-shadow: 0 4px 22px rgba(37,99,235,0.28);
    font-family: system-ui, sans-serif;
">
  <div style="display:flex; align-items:center; gap:14px;">
    <span style="font-size:2.6rem; line-height:1;">🎓</span>
    <div>
      <div style="
          font-size:1.5rem; font-weight:700; color:white;
          letter-spacing:-0.01em; line-height:1.2; margin-bottom:5px;">
        Ала-Тоо Университети
      </div>
      <div style="
          font-size:0.95rem; color:rgba(255,255,255,0.88);
          font-weight:400; line-height:1.4;">
        Кабылуу боюнча AI жардамчы &nbsp;·&nbsp;
        ИИ-помощник приёмной комиссии &nbsp;·&nbsp;
        AI Admissions Assistant
      </div>
    </div>
  </div>
  <div style="
      margin-top:12px; padding-top:10px;
      border-top:1px solid rgba(255,255,255,0.2);
      display:flex; gap:20px; flex-wrap:wrap;
      font-size:0.8rem; color:rgba(255,255,255,0.75);
  ">
    <span>🇰🇬 Кыргызча</span>
    <span>🇷🇺 Русский</span>
    <span>🇬🇧 English</span>
    <span>📞 +996 555 820 000 (WhatsApp)</span>
    <span>📧 admission@alatoo.edu.kg</span>
  </div>
</div>
"""

VOICE_HINT_HTML = """
<div style="
    font-size:0.78rem; color:#475569; line-height:1.6;
    background:#f0f7ff; border-radius:8px; padding:10px 12px;
    border:1px solid #bfdbfe; margin-top:6px;
">
  🎙 Кыргыз, орус же англис тилинде сүйлөңүз.<br>
  Говорите по-кыргызски, по-русски или по-английски.<br>
  <span style="color:#94a3b8">Speak in any of the three languages.</span>
</div>
"""

FOOTER_HTML = """
<div style="
    text-align:center; padding:14px 8px 6px;
    font-size:0.78rem; color:#94a3b8;
    border-top:1px solid #e2e8f0; margin-top:4px;
    font-family:system-ui,sans-serif;
">
  🏫 ул. Анкара (Горький) 1/10, мкр. «Тунгуч», Бишкек, D-блок, 1 этаж
  &nbsp;·&nbsp;
  📞 <a href="https://wa.me/996555820000" style="color:#3b82f6;text-decoration:none;">+996 555 820 000</a>
  &nbsp;·&nbsp;
  📧 <a href="mailto:admission@alatoo.edu.kg" style="color:#3b82f6;text-decoration:none;">admission@alatoo.edu.kg</a>
</div>
"""


# ── Session helpers ────────────────────────────────────────────────────────────

def _generate_user_id() -> str:
    return f"web_{uuid.uuid4().hex[:12]}"


def _msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


# ── Agent callbacks ────────────────────────────────────────────────────────────

def respond(message: str, history: list, user_id: str) -> tuple[str, list, str]:
    if not message.strip():
        return "", history, user_id

    guard = guardrails.check(message)
    if guard.blocked or guard.off_topic:
        history = history + [_msg("user", message), _msg("assistant", guard.reply)]
        return "", history, user_id

    session = get_session(user_id)
    reply = run_agent(message, session)
    history = history + [_msg("user", message), _msg("assistant", reply)]
    return "", history, user_id


def respond_voice(audio_path: str | None, history: list, user_id: str) -> tuple[list, str]:
    if audio_path is None:
        return history, user_id

    from voice.stt import transcribe
    text = transcribe(audio_path)

    if not text:
        err = "Речь не распознана / Үн таанылган жок. Попробуйте ещё раз или напишите текстом."
        return history + [_msg("assistant", err)], user_id

    guard = guardrails.check(text)
    if guard.blocked or guard.off_topic:
        return history + [_msg("user", f"🎙 {text}"), _msg("assistant", guard.reply)], user_id

    session = get_session(user_id)
    reply = run_agent(text, session)
    return history + [_msg("user", f"🎙 {text}"), _msg("assistant", reply)], user_id


def reset_session(user_id: str) -> tuple[list, str]:
    clear_session(user_id)
    return [], _generate_user_id()


# ── UI layout ──────────────────────────────────────────────────────────────────

def build_demo() -> gr.Blocks:
    with gr.Blocks(
        title=TITLE,
        theme=gr.themes.Soft(
            primary_hue="blue",
            neutral_hue="slate",
        ),
        css=CSS,
    ) as demo:

        user_id_state = gr.State(_generate_user_id)

        # Header
        gr.HTML(HEADER_HTML)

        # ── Main row: chat (left) + voice panel (right) ────────────────────────
        with gr.Row(equal_height=False):

            # Chat column
            with gr.Column(scale=5):
                chatbot = gr.Chatbot(
                    type="messages",
                    height=500,
                    show_label=False,
                    elem_id="chatbox",
                    bubble_full_width=False,
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder=PLACEHOLDER,
                        scale=6,
                        container=False,
                        lines=1,
                        max_lines=5,
                        show_label=False,
                    )
                    send_btn = gr.Button(
                        "➤ Жөнөт / Отправить",
                        variant="primary",
                        scale=1,
                        min_width=140,
                        elem_id="btn-send",
                    )

            # Voice panel column
            with gr.Column(scale=2, min_width=220):
                gr.Markdown("### 🎙️ Үн менен жазуу\n*Голосовой ввод / Voice input*")

                audio_input = gr.Audio(
                    label="Микрофон / Файл",
                    type="filepath",
                    sources=["microphone", "upload"],
                )

                send_voice_btn = gr.Button(
                    "🎙️  Жөнөт үнүмдү\nОтправить голос",
                    variant="primary",
                    size="lg",
                    elem_id="btn-voice",
                )

                gr.HTML(VOICE_HINT_HTML)

        # ── Bottom bar ─────────────────────────────────────────────────────────
        with gr.Row():
            clear_btn = gr.Button(
                "🔄  Жаңы суббат / Новая сессия",
                variant="stop",
                size="sm",
                elem_id="btn-reset",
            )

        # ── Example questions ──────────────────────────────────────────────────
        gr.Examples(
            examples=EXAMPLES,
            inputs=msg_input,
            label="💬 Суроолор мисалдары / Примеры вопросов / Example questions",
        )

        gr.HTML(FOOTER_HTML)

        # ── Event bindings ─────────────────────────────────────────────────────
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
            inputs=[audio_input, chatbot, user_id_state],
            outputs=[chatbot, user_id_state],
        )
        clear_btn.click(
            fn=reset_session,
            inputs=[user_id_state],
            outputs=[chatbot, user_id_state],
        )

    return demo


if __name__ == "__main__":
    build_demo().launch(server_port=7860)

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


TITLE = "Ала-Тоо Университети — Приёмная комиссия"
DESCRIPTION = (
    "Добро пожаловать! Я помогу вам с вопросами поступления, "
    "выбором специальности и сравнением программ.\n\n"
    "Кош келиңиздер! Кабылуу, адистик тандоо жана программаларды салыштыруу боюнча жардам берем."
)
PLACEHOLDER = "Введите вопрос на русском или кыргызском языке…"
EXAMPLES = [
    "Мой ОРТ 145. Могу ли я поступить на Computer Science?",
    "Я не знаю, какую специальность выбрать.",
    "Сравни программы CS и Economics.",
    "Хочу поговорить с сотрудником приёмной комиссии.",
    "Менин ОРТ балым 138. CS факультетине кире аламбы?",
]


def _generate_user_id() -> str:
    return f"web_{uuid.uuid4().hex[:12]}"


def respond(message: str, history: list, user_id: str) -> tuple[str, list, str]:
    """
    Called by Gradio on each user message submission.

    Returns: (cleared_input, updated_history, user_id)
    """
    if not message.strip():
        return "", history, user_id

    guard = guardrails.check(message)
    if guard.blocked or guard.off_topic:
        history = history + [(message, guard.reply)]
        return "", history, user_id

    session = get_session(user_id)
    reply = run_agent(message, session)
    history = history + [(message, reply)]
    return "", history, user_id


def respond_voice(audio_path: str | None, history: list, user_id: str) -> tuple[list, str]:
    """Transcribe uploaded voice file and route to agent."""
    if audio_path is None:
        return history, user_id

    from voice.stt import transcribe
    text = transcribe(audio_path)

    if not text:
        err = "Не удалось распознать речь. Попробуйте ещё раз или напишите текстом."
        return history + [(None, err)], user_id

    guard = guardrails.check(text)
    if guard.blocked or guard.off_topic:
        return history + [(f"🎙 {text}", guard.reply)], user_id

    session = get_session(user_id)
    reply = run_agent(text, session)
    return history + [(f"🎙 {text}", reply)], user_id


def reset_session(user_id: str) -> tuple[list, str]:
    """Clear the session and start fresh."""
    clear_session(user_id)
    new_id = _generate_user_id()
    return [], new_id


def build_demo() -> gr.Blocks:
    with gr.Blocks(
        title=TITLE,
        theme=gr.themes.Soft(primary_hue="blue"),
        css=".gradio-container { max-width: 860px; margin: auto; }",
    ) as demo:

        gr.Markdown(f"# {TITLE}\n{DESCRIPTION}")

        # Per-tab state
        user_id_state = gr.State(_generate_user_id)

        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    label="Чат / Chat",
                    height=480,
                    show_label=False,
                    bubble_full_width=False,
                )
            with gr.Column(scale=1, min_width=160):
                gr.Markdown("### Голос / Voice")
                audio_input = gr.Audio(
                    label="Загрузите аудио",
                    type="filepath",
                    sources=["upload", "microphone"],
                )
                send_voice_btn = gr.Button("Отправить голос", variant="secondary")

        with gr.Row():
            msg_input = gr.Textbox(
                label="",
                placeholder=PLACEHOLDER,
                scale=5,
                container=False,
            )
            send_btn = gr.Button("Отправить", variant="primary", scale=1)

        with gr.Row():
            clear_btn = gr.Button("Сбросить сессию / Reset", variant="stop", size="sm")

        gr.Examples(
            examples=EXAMPLES,
            inputs=msg_input,
            label="Примеры вопросов / Example questions",
        )

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

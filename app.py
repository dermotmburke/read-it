"""Gradio UI for MLX Kokoro text-to-speech.

Architecture note: MLX requires all GPU ops to run on the main thread
(the one that created Stream(gpu, 0) at import time). Gradio's handlers
run in a thread pool, so we flip things: Gradio server runs in a
background thread, main thread loops over inference tasks from a queue.
"""
import traceback
import queue
import threading
import numpy as np
import gradio as gr

from mlx_audio.tts.utils import load_model
from speak import split_into_chunks, VOICES, DEFAULT_MODEL, DEFAULT_VOICE

VOICE_CHOICES = [
    (f"{v}  ·  {cat}", v)
    for cat, voices in VOICES.items()
    for v in voices
]

print("Loading Kokoro model (one-time)...")
model = load_model(DEFAULT_MODEL)
SAMPLE_RATE = model.sample_rate
print(f"Model ready  ·  sample rate: {SAMPLE_RATE} Hz")

# Queue for passing work TO the main thread, and getting results back
_task_queue: queue.Queue = queue.Queue()


def _run_on_main_thread(fn, *args):
    """Submit fn(*args) to the main thread and block until done."""
    box: dict = {"done": threading.Event()}
    _task_queue.put((fn, args, box))
    box["done"].wait()
    if "error" in box:
        raise RuntimeError(box["error"])
    return box["result"]


def load_file(file_path):
    if not file_path:
        return gr.update()
    with open(file_path) as f:
        return f.read()


def generate(text, voice, speed):
    print(f"generate() called: voice={voice!r} speed={speed}")
    try:
        text = (text or "").strip()
        if not text:
            return gr.update(), "No text provided."

        voice_id = voice.split("  ·  ")[0].strip() if "  ·  " in (voice or "") else voice

        text = (
            text.replace("—", " - ")
                .replace("–", " - ")
                .replace("‘", "'")
                .replace("’", "'")
                .replace("“", '"')
                .replace("”", '"')
        )

        chunks = split_into_chunks(text)
        print(f"chunks: {len(chunks)}, voice: {voice_id}")

        def _infer():
            parts = []
            for i, chunk in enumerate(chunks):
                print(f"  chunk {i+1}/{len(chunks)}: {chunk[:60]!r}")
                for result in model.generate(
                    text=chunk, voice=voice_id, speed=speed, lang_code="en"
                ):
                    parts.append(np.array(result.audio).flatten())
            return parts

        audio_parts = _run_on_main_thread(_infer)

        if not audio_parts:
            return gr.update(), "Model returned no audio."

        audio = np.concatenate(audio_parts)
        print(f"done: {len(audio)} samples")
        return (SAMPLE_RATE, audio), ""

    except Exception:
        msg = traceback.format_exc()
        print("ERROR:\n" + msg)
        return gr.update(), msg


# ── UI ───────────────────────────────────────────────────────────────────────

with gr.Blocks(title="MLX TTS") as demo:
    gr.Markdown("## MLX Text-to-Speech\nKokoro-82M · local · Apple Silicon")

    with gr.Row():
        with gr.Column(scale=3):
            file_in = gr.File(
                label="Drop a .txt file (or paste below)",
                file_types=[".txt"],
                file_count="single",
            )
            text_in = gr.Textbox(
                label="Text",
                placeholder="Paste text here…",
                lines=14,
                max_lines=40,
            )
            with gr.Row():
                voice_in = gr.Dropdown(
                    label="Voice",
                    choices=VOICE_CHOICES,
                    value=DEFAULT_VOICE,
                    scale=2,
                )
                speed_in = gr.Slider(
                    label="Speed",
                    minimum=0.5,
                    maximum=2.0,
                    step=0.05,
                    value=1.0,
                    scale=1,
                )
            speak_btn = gr.Button("Speak", variant="primary", size="lg")

        with gr.Column(scale=2):
            audio_out = gr.Audio(label="Output", autoplay=True)
            error_out = gr.Textbox(label="Log / Error", lines=10)

    file_in.change(load_file, file_in, text_in)
    speak_btn.click(generate, [text_in, voice_in, speed_in], [audio_out, error_out])


# ── Entry point ──────────────────────────────────────────────────────────────

def _run_gradio():
    demo.launch(theme=gr.themes.Soft(), prevent_thread_lock=True)


if __name__ == "__main__":
    # Start Gradio in a background thread so the main thread stays free for MLX
    t = threading.Thread(target=_run_gradio, daemon=True)
    t.start()

    print("Server running — main thread handling MLX inference")
    while True:
        fn, args, box = _task_queue.get()
        try:
            box["result"] = fn(*args)
        except Exception:
            box["error"] = traceback.format_exc()
            print("Inference error:\n" + box["error"])
        box["done"].set()

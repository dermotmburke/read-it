"""Gradio TTS server — PyTorch/Kokoro backend, runs on Linux CPU."""
import os
import traceback
import numpy as np
import gradio as gr
from kokoro import KPipeline

SAMPLE_RATE = 24000
DEFAULT_VOICE = "af_heart"

VOICES = {
    "American Female": ["af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica",
                        "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky"],
    "American Male":   ["am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
                        "am_michael", "am_onyx", "am_puck", "am_santa"],
    "British Female":  ["bf_alice", "bf_emma", "bf_isabella", "bf_lily"],
    "British Male":    ["bm_daniel", "bm_fable", "bm_george", "bm_lewis"],
}
VOICE_CHOICES = [(f"{v}  ·  {cat}", v) for cat, vs in VOICES.items() for v in vs]

# lang_code: 'a' = American English, 'b' = British English
def _lang_code(voice_id: str) -> str:
    return "b" if voice_id.startswith("b") else "a"


print("Loading Kokoro pipeline...")
# Pipelines are cheap to create; we keep one per lang_code
_pipelines: dict[str, KPipeline] = {}

def _get_pipeline(voice_id: str) -> KPipeline:
    lc = _lang_code(voice_id)
    if lc not in _pipelines:
        _pipelines[lc] = KPipeline(lang_code=lc)
    return _pipelines[lc]

# Warm up American English pipeline on startup
_get_pipeline("af_heart")
print("Ready.")


def split_chunks(text: str, max_chars: int = 600) -> list[str]:
    import re
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current = [], ""
    for s in sentences:
        candidate = (current + " " + s).strip() if current else s
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = s if len(s) <= max_chars else s  # oversized sentences pass through
    if current:
        chunks.append(current)
    return chunks


def load_file(file_path):
    if not file_path:
        return gr.update()
    with open(file_path) as f:
        return f.read()


def generate(text, voice, speed):
    print(f"generate(): voice={voice!r} speed={speed}")
    try:
        text = (text or "").strip()
        if not text:
            return gr.update(), "No text provided."

        voice_id = voice.split("  ·  ")[0].strip() if "  ·  " in (voice or "") else voice

        text = (
            text.replace("—", " - ").replace("–", " - ")
                .replace("‘", "'").replace("’", "'")
                .replace("“", '"').replace("”", '"')
        )

        pipeline = _get_pipeline(voice_id)
        chunks = split_chunks(text)
        print(f"  {len(chunks)} chunk(s)")

        audio_parts = []
        for i, chunk in enumerate(chunks):
            print(f"  chunk {i+1}/{len(chunks)}: {chunk[:60]!r}")
            for _, _, audio in pipeline(chunk, voice=voice_id, speed=speed):
                audio_parts.append(audio.flatten())

        if not audio_parts:
            return gr.update(), "Model returned no audio."

        result = np.concatenate(audio_parts)
        print(f"  done: {len(result)} samples ({len(result)/SAMPLE_RATE:.1f}s)")
        return (SAMPLE_RATE, result), ""

    except Exception:
        msg = traceback.format_exc()
        print("ERROR:\n" + msg)
        return gr.update(), msg


with gr.Blocks(title="TTS") as demo:
    gr.Markdown("## Text-to-Speech\nKokoro-82M · local")

    with gr.Row():
        with gr.Column(scale=3):
            file_in = gr.File(label="Drop a .txt file", file_types=[".txt"], file_count="single")
            text_in = gr.Textbox(label="Text", placeholder="Paste text here…", lines=14, max_lines=40)
            with gr.Row():
                voice_in = gr.Dropdown(label="Voice", choices=VOICE_CHOICES, value=DEFAULT_VOICE, scale=2)
                speed_in = gr.Slider(label="Speed", minimum=0.5, maximum=2.0, step=0.05, value=1.0, scale=1)
            speak_btn = gr.Button("Speak", variant="primary", size="lg")

        with gr.Column(scale=2):
            audio_out = gr.Audio(label="Output", autoplay=True)
            error_out = gr.Textbox(label="Log", lines=8)

    file_in.change(load_file, file_in, text_in)
    speak_btn.click(generate, [text_in, voice_in, speed_in], [audio_out, error_out])


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # bind to all interfaces so Docker can reach it
        server_port=int(os.environ.get("PORT", 7860)),
        theme=gr.themes.Soft(),
    )

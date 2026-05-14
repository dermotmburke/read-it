"""Gradio UI for MLX text-to-speech.

Architecture note: MLX requires all GPU ops to run on the main thread
(the one that created Stream(gpu, 0) at import time). Gradio's handlers
run in a thread pool, so we flip things: Gradio server runs in a
background thread, main thread loops over inference tasks from a queue.
"""
import os
import tempfile
import traceback
import queue
import threading
from dataclasses import dataclass, field
import numpy as np
import gradio as gr
from scipy.io.wavfile import write as wav_write
from pydub import AudioSegment

from mlx_audio.tts.utils import load_model
from speak import split_into_chunks, VOICES, DEFAULT_VOICE

_MP3_PATH = os.path.join(tempfile.gettempdir(), "read_it_output.mp3")


def _to_mp3(audio: np.ndarray, sample_rate: int) -> str:
    tmp_wav = _MP3_PATH.replace(".mp3", ".wav")
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    wav_write(tmp_wav, sample_rate, pcm)
    AudioSegment.from_wav(tmp_wav).export(_MP3_PATH, format="mp3", bitrate="192k")
    os.unlink(tmp_wav)
    return _MP3_PATH


# ── Model registry ────────────────────────────────────────────────────────────

@dataclass
class ModelSpec:
    id: str                          # HuggingFace repo id passed to load_model()
    label: str                       # Dropdown display label
    voices: list                     # [(display, value), ...]; empty = no named voices
    default_voice: str               # "" if no voices
    chunk_chars: int | None          # None = natural sentence split; 0 = no split; N = chunk size
    hint: str                        # Hint box text; "" = hidden
    eager: bool                      # Load at startup vs first use
    use_speed: bool                  # Whether to pass speed= to generate()
    default_speed: float             # Initial slider value for this model
    preprocess_text: bool            # Whether to normalise em-dashes / smart quotes
    generate_kwargs: dict = field(default_factory=dict)  # Extra kwargs for generate()


_KOKORO_VOICES = [
    (f"{v}  ·  {cat}", v)
    for cat, voices in VOICES.items()
    for v in voices
]

MODELS: list[ModelSpec] = [
    ModelSpec(
        id="mlx-community/kokoro-82M-bf16",
        label="Kokoro  ·  fast",
        voices=_KOKORO_VOICES,
        default_voice=DEFAULT_VOICE,
        chunk_chars=None,
        hint="",
        eager=True,
        use_speed=True,
        default_speed=0.9,
        preprocess_text=True,
        generate_kwargs={"lang_code": "en"},
    ),
    ModelSpec(
        id="mlx-community/orpheus-3b-0.1-ft-4bit",
        label="Orpheus  ·  expressive",
        voices=[
            ("tara  ·  American Female", "tara"),
            ("leah  ·  American Female", "leah"),
            ("jess  ·  American Female", "jess"),
            ("mia  ·  American Female", "mia"),
            ("zoe  ·  American Female", "zoe"),
            ("leo  ·  American Male", "leo"),
            ("dan  ·  American Male", "dan"),
            ("zac  ·  American Male", "zac"),
        ],
        default_voice="tara",
        chunk_chars=250,
        hint=(
            "Emotion tags: <laugh> <chuckle> <sigh> <gasp> <cough> <cries> — "
            "drop them anywhere in the text."
        ),
        eager=False,
        use_speed=True,
        default_speed=1.1,
        preprocess_text=False,
        # ~175 tokens/sec; 250 chars ≈ 20s speech → 3500 tokens; 4800 for headroom
        generate_kwargs={"max_tokens": 4800},
    ),
]

MODEL_OPTIONS = [(spec.label, spec.id) for spec in MODELS]

_model_cache: dict = {}


def _spec_by_id(model_id: str) -> ModelSpec:
    for spec in MODELS:
        if spec.id == model_id:
            return spec
    raise KeyError(f"Unknown model: {model_id!r}")


def _get_model(spec: ModelSpec):
    if spec.id not in _model_cache:
        print(f"Loading {spec.label} (first use)…")
        _model_cache[spec.id] = load_model(spec.id)
        print(f"{spec.label} ready.")
    return _model_cache[spec.id]


for _spec in MODELS:
    if _spec.eager:
        print(f"Loading {_spec.label} (one-time)…")
        _model_cache[_spec.id] = load_model(_spec.id)
        print(f"{_spec.label} ready  ·  sample rate: {_model_cache[_spec.id].sample_rate} Hz")

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


def generate(text, model_id, voice, speed):
    print(f"generate(): model={model_id!r} voice={voice!r} speed={speed}")
    no_dl = gr.update(visible=False)

    def _out(audio=gr.update(), error="", status="", dl=no_dl):
        return audio, error, status, dl

    try:
        spec = _spec_by_id(model_id)
        text = (text or "").strip()
        if not text:
            yield _out(error="No text provided.")
            return

        is_cached = spec.id in _model_cache
        loading_msg = (
            f"Downloading {spec.label} model…" if not is_cached
            else f"Generating with {spec.label}…"
        )
        yield _out(status=loading_msg)

        if spec.preprocess_text:
            text = (
                text.replace("—", " - ").replace("–", " - ")
                    .replace("'", "'").replace("'", "'")
                    .replace('"', '"').replace('"', '"')
            )

        if spec.chunk_chars == 0:
            chunks = [text]
        elif spec.chunk_chars is None:
            chunks = split_into_chunks(text)
        else:
            chunks = split_into_chunks(text, spec.chunk_chars)

        voice_id = voice.split("  ·  ")[0].strip() if "  ·  " in (voice or "") else voice
        print(f"  {len(chunks)} chunk(s), voice: {voice_id!r}")

        def _infer():
            mdl = _get_model(spec)
            parts = []
            for i, chunk in enumerate(chunks):
                print(f"  chunk {i+1}/{len(chunks)}: {chunk[:60]!r}")
                kwargs = {"text": chunk, **spec.generate_kwargs}
                if spec.voices:
                    kwargs["voice"] = voice_id
                if spec.use_speed:
                    kwargs["speed"] = speed
                for result in mdl.generate(**kwargs):
                    parts.append(np.array(result.audio).flatten())
            return parts, mdl.sample_rate

        audio_parts, sample_rate = _run_on_main_thread(_infer)

        if not audio_parts:
            yield _out(error="Model returned no audio.")
            return

        audio = np.clip(np.concatenate(audio_parts), -1.0, 1.0)
        print(f"  done: {len(audio)} samples @ {sample_rate} Hz")
        mp3_path = _to_mp3(audio, sample_rate)
        yield _out(audio=(sample_rate, audio), dl=gr.update(value=mp3_path, visible=True))

    except Exception:
        msg = traceback.format_exc()
        print("ERROR:\n" + msg)
        yield _out(error=msg)


def update_voices(model_id):
    """Switch voice dropdown, hint, and speed slider when model changes."""
    spec = _spec_by_id(model_id)
    voice_update = (
        gr.update(choices=spec.voices, value=spec.default_voice, visible=True)
        if spec.voices
        else gr.update(choices=[("—", "")], value="", visible=False)
    )
    hint_update = gr.update(value=spec.hint, visible=bool(spec.hint))
    speed_update = gr.update(visible=spec.use_speed, value=spec.default_speed)
    return voice_update, hint_update, speed_update


# ── Design: Sonic Brutalism ───────────────────────────────────────────────────

_HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700;900&display=swap" rel="stylesheet">
<script>
(function() {
    var saved = localStorage.getItem('read-it-theme');
    var prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    document.documentElement.setAttribute('data-theme', saved || (prefersDark ? 'dark' : 'light'));

    var DARK_CSS =
        '* { color: #ffffff !important; }' +
        '.label-wrap > span, label > span, .label-wrap span, label span { color: #ffffff !important; }' +
        '.wrap, .wrap-inner, .secondary-wrap, .token, select { color: #ffffff !important; }' +
        '.wrap input, .wrap-inner input, input[readonly] { color: #ffffff !important; }' +
        'svg, svg *, path, circle, rect, polyline, polygon { fill: #ffffff !important; }' +
        '.gradio-container, :root {' +
        '  --neutral-50:#000; --neutral-100:#111; --neutral-200:#222; --neutral-300:#ccc;' +
        '  --neutral-400:#ffffff; --neutral-500:#fff; --neutral-600:#fff; --neutral-700:#fff; --neutral-800:#fff; --neutral-900:#fff;' +
        '  --body-text-color:#ffffff; --body-text-color-subdued:#cccccc;' +
        '  --background-fill-primary:#000000; --background-fill-secondary:#111111;' +
        '  --block-background-fill:#000000; --input-background-fill:#000000;' +
        '  --border-color-primary:#ffffff; --color-accent:#ffffff;' +
        '  --button-secondary-background-fill:#000000; --button-secondary-text-color:#ffffff;' +
        '}' +
        '.component-wrapper, .standard-player, .controls, .control-wrapper, .play-pause-wrapper, .settings-wrapper, .waveform-wrapper { background: #000000 !important; }' +
        '.play-btn, .play-pause-button, .action, .rewind, .skip, .icon, .icon-wrap, .timestamp, .text-button, .playback, .volume, #time, #duration { color: #ffffff !important; fill: #ffffff !important; }' +
        'body, .gradio-container, .contain, .block, .panel, .form,' +
        ' .wrap, .wrap-inner, .secondary-wrap, .token,' +
        ' textarea, select, input, .upload-container, ul.options,' +
        ' .waveform-container, #app-header, #theme-toggle' +
        ' { background: #000000 !important; background-color: #000000 !important; }' +
        '.block, .panel, .form { border-color: #ffffff !important; }' +
        'textarea { border-color: #ffffff !important; }' +
        '.upload-container { border-color: #ffffff !important; }' +
        '.wrap-inner { border-color: #ffffff !important; }' +
        '.waveform-container { border-color: #ffffff !important; }' +
        '#app-header { border-bottom-color: #ffffff !important; }' +
        '#theme-toggle { border-color: #ffffff !important; }' +
        '#theme-toggle:hover { background: #ffffff !important; color: #000000 !important; }' +
        'input[type="range"] { accent-color: #ffffff !important; }' +
        'html[data-theme="dark"] .gradio-container .contain #speak-btn { background: #ffffff !important; color: #000000 !important; border-color: #ffffff !important; box-shadow: 4px 4px 0 #ffffff !important; }' +
        'html[data-theme="dark"] .gradio-container .contain #speak-btn:hover { box-shadow: 2px 2px 0 #ffffff !important; }' +
        '#dl-btn { background: #000000 !important; background-color: #000000 !important; color: #ffffff !important; border-color: #ffffff !important; box-shadow: 2px 2px 0px 0px #ffffff !important; }' +
        'ul.options li:hover, ul.options li.selected { background: #ffffff !important; color: #000000 !important; }' +
        '#ri-modal-box { background: #000000 !important; border-color: #ffffff !important; box-shadow: 8px 8px 0 #ffffff !important; }' +
        '#ri-modal-title, #ri-modal-status { color: #ffffff !important; }' +
        '#ri-modal-track { background: #333333 !important; }' +
        '#ri-modal-scan { background: #ffffff !important; }';

    function syncDark() {
        var el = document.getElementById('ri-dm');
        if (el) el.remove();
        if (document.documentElement.getAttribute('data-theme') !== 'dark') return;
        var s = document.createElement('style');
        s.id = 'ri-dm';
        s.textContent = DARK_CSS;
        (document.body || document.documentElement).appendChild(s);
    }

    window._riSyncColors = syncDark;

    function onReady() { setTimeout(syncDark, 300); }
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', onReady);
    } else {
        onReady();
    }
})();

// ── Processing modal ─────────────────────────────────────────────────────────
(function() {
    function riSetup() {
        var btn = document.querySelector('#speak-btn');
        if (!btn) { setTimeout(riSetup, 500); return; }
        btn.addEventListener('click', function() {
            var m = document.getElementById('ri-modal');
            var s = document.getElementById('ri-modal-status');
            if (m) m.style.display = 'flex';
            if (s) s.textContent = 'Generating audio…';
            var started = Date.now();
            var poll = setInterval(function() {
                var modal = document.getElementById('ri-modal');
                if (!modal || modal.style.display !== 'flex') { clearInterval(poll); return; }
                if (Date.now() - started < 1500) return;
                var soTxt = (document.querySelector('#status-out')?.textContent || '').trim();
                var errTxt = (document.querySelector('#error-box textarea')?.value || '').trim();
                if (errTxt || soTxt === '' || soTxt === 'Error') {
                    modal.style.display = 'none';
                    clearInterval(poll);
                    if (!errTxt && soTxt !== 'Error') {
                        [300, 800, 1600, 2800].forEach(function(d) {
                            setTimeout(function() {
                                if (!document.querySelector('button[aria-label="Pause"]')) {
                                    var btn = document.querySelector('button[aria-label="Play"]');
                                    if (btn) btn.click();
                                }
                            }, d);
                        });
                    }
                }
            }, 800);
        });
    }
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function() { setTimeout(riSetup, 800); });
    } else {
        setTimeout(riSetup, 800);
    }
})();

function toggleTheme() {
    var html = document.documentElement;
    var isDark = html.getAttribute('data-theme') === 'dark';
    html.setAttribute('data-theme', isDark ? 'light' : 'dark');
    localStorage.setItem('read-it-theme', isDark ? 'light' : 'dark');
    if (window._riSyncColors) window._riSyncColors();
}

</script>
"""

_CSS = """
/* ── Global: Space Grotesk, zero radius ── */
*, *::before, *::after {
    font-family: 'Space Grotesk', sans-serif !important;
    border-radius: 0px !important;
}

/* ── Light base ── */
html { color-scheme: light; }
html[data-theme="dark"] { color-scheme: dark; }

.gradio-container {
    background: #ffffff !important;
    max-width: 100% !important;
    padding: 0 !important;
}

/* ── App header ── */
#app-header {
    background: #ffffff;
    border-bottom: 4px solid #000000;
    padding: 20px 32px;
    display: flex;
    align-items: center;
    gap: 24px;
}

/* ── Header title ── */
#app-title { color: #000000; }
html[data-theme="dark"] #app-title { color: #ffffff; }

/* ── Theme toggle ── */
#theme-toggle {
    margin-left: auto;
    background: #ffffff;
    color: #000000;
    border: 2px solid #000000;
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    padding: 6px 14px;
    cursor: pointer;
    transition: background 0.08s, color 0.08s;
}
#theme-toggle:hover { background: #000000; color: #ffffff; }
#theme-toggle::after { content: 'DARK'; }
html[data-theme="dark"] #theme-toggle::after { content: 'LIGHT'; }

/* ── Content area padding ── */
.contain {
    padding: 28px 32px 56px !important;
}

/* ── Blocks / panels ── */
.block, .panel, .form {
    border: 2px solid #000000 !important;
    box-shadow: none !important;
    background: #ffffff !important;
}

/* ── Labels ── */
.label-wrap > span, label > span {
    font-size: 0.62rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    color: #000000 !important;
}

/* ── Textareas ── */
textarea {
    background: #ffffff !important;
    color: #000000 !important;
    border: 2px solid #000000 !important;
    box-shadow: none !important;
    font-size: 0.95rem !important;
}
textarea:focus {
    border-width: 4px !important;
    outline: none !important;
    box-shadow: none !important;
}

/* ── File upload ── */
.upload-container {
    border: 2px dashed #000000 !important;
    background: #ffffff !important;
}

/* ── Dropdowns ── */
.wrap, .wrap-inner, .secondary-wrap, .token, select {
    background: #ffffff !important;
    color: #000000 !important;
}
.wrap-inner {
    border: 2px solid #000000 !important;
}
.wrap input, .wrap-inner input, input[readonly] {
    background: #ffffff !important;
    color: #000000 !important;
    font-weight: 600 !important;
}
ul.options {
    border: 2px solid #000000 !important;
    box-shadow: 4px 4px 0px 0px #000000 !important;
    background: #ffffff !important;
    color: #000000 !important;
}
ul.options li { color: #000000 !important; }
ul.options li:hover, ul.options li.selected {
    background: #000000 !important;
    color: #ffffff !important;
}

/* ── Slider ── */
input[type='range'] { accent-color: #000000 !important; }

/* ── SPEAK button — brutalist hard shadow ── */
#speak-btn {
    background: #000000 !important;
    color: #ffffff !important;
    font-weight: 900 !important;
    font-size: 1.2rem !important;
    letter-spacing: 0.22em !important;
    text-transform: uppercase !important;
    border: 2px solid #000000 !important;
    box-shadow: 4px 4px 0px 0px #000000 !important;
    padding: 18px 32px !important;
    transition: box-shadow 0.08s, transform 0.08s !important;
    width: 100% !important;
}
#speak-btn:hover {
    box-shadow: 2px 2px 0px 0px #000000 !important;
    transform: translate(2px, 2px) !important;
}
#speak-btn:active {
    box-shadow: none !important;
    transform: translate(4px, 4px) !important;
}

/* ── Audio waveform ── */
.waveform-container {
    border: 2px solid #000000 !important;
    background: #ffffff !important;
}

/* ── Download button ── */
#dl-btn {
    background: #ffffff !important;
    color: #000000 !important;
    font-weight: 900 !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    border: 2px solid #000000 !important;
    box-shadow: 2px 2px 0px 0px #000000 !important;
    padding: 12px 24px !important;
    width: 100% !important;
    transition: box-shadow 0.08s, transform 0.08s !important;
}
#dl-btn:hover {
    box-shadow: 1px 1px 0px 0px #000000 !important;
    transform: translate(1px, 1px) !important;
}
#dl-btn:active {
    box-shadow: none !important;
    transform: translate(2px, 2px) !important;
}

/* ── Log / error box ── */
#error-box textarea {
    font-size: 0.72rem !important;
    line-height: 1.6 !important;
}

/* ── Hide Gradio branding ── */
footer, .built-with { display: none !important; }

/* ── Processing modal ── */
#ri-modal {
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.82);
    z-index: 9999;
    align-items: center;
    justify-content: center;
}
#ri-modal-box {
    background: #ffffff;
    border: 4px solid #000000;
    box-shadow: 8px 8px 0 #000000;
    padding: 48px 56px;
    min-width: 340px;
    max-width: 520px;
    width: 90%;
}
#ri-modal-title {
    font-size: 2.4rem;
    font-weight: 900;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #000000;
    margin-bottom: 28px;
    line-height: 1;
}
#ri-modal-track {
    height: 6px;
    background: #e0e0e0;
    position: relative;
    overflow: hidden;
    margin-bottom: 20px;
}
#ri-modal-scan {
    position: absolute;
    top: 0;
    left: -40%;
    width: 40%;
    height: 100%;
    background: #000000;
    animation: ri-scan 1.3s cubic-bezier(0.4, 0, 0.6, 1) infinite;
}
#ri-modal-status {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #000000;
}
@keyframes ri-scan {
    0%   { left: -40%; }
    100% { left: 140%; }
}

/* dark mode is handled entirely by JS-injected <style id="ri-dm"> */
html[data-theme="dark"] { color-scheme: dark; }
"""


# ── UI ───────────────────────────────────────────────────────────────────────

with gr.Blocks(title="read-it") as demo:
    gr.HTML('<div id="ri-modal"><div id="ri-modal-box"><div id="ri-modal-title">Generating</div><div id="ri-modal-track"><div id="ri-modal-scan"></div></div><div id="ri-modal-status">Initialising…</div></div></div>')

    gr.HTML("""
        <div id="app-header">
            <svg id="app-title" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1860 400" height="52" width="242" aria-label="READ-IT" role="img" fill="currentColor">
                <!-- R: short mid bar stops at x=200, leaving gap before leg -->
                <rect x="25"  y="50"  width="85"  height="300"/>
                <rect x="25"  y="50"  width="260" height="85"/>
                <rect x="200" y="50"  width="85"  height="139"/>
                <rect x="25"  y="189" width="175" height="85"/>
                <rect x="175" y="274" width="110" height="76"/>
                <!-- E -->
                <rect x="335" y="50"  width="85"  height="300"/>
                <rect x="335" y="50"  width="260" height="85"/>
                <rect x="335" y="189" width="200" height="85"/>
                <rect x="335" y="265" width="260" height="85"/>
                <!-- A -->
                <rect x="645" y="50"  width="85"  height="300"/>
                <rect x="820" y="50"  width="85"  height="300"/>
                <rect x="645" y="50"  width="260" height="85"/>
                <rect x="645" y="189" width="260" height="85"/>
                <!-- D -->
                <rect x="955"  y="50"  width="85"  height="300"/>
                <rect x="955"  y="50"  width="260" height="85"/>
                <rect x="955"  y="265" width="260" height="85"/>
                <rect x="1130" y="135" width="85"  height="130"/>
                <!-- I -->
                <rect x="1265" y="50"  width="260" height="85"/>
                <rect x="1352" y="50"  width="85"  height="300"/>
                <rect x="1265" y="265" width="260" height="85"/>
                <!-- T -->
                <rect x="1575" y="50" width="260" height="85"/>
                <rect x="1662" y="50" width="85"  height="300"/>
            </svg>
            <span style="font-size:0.68rem;font-weight:700;letter-spacing:0.16em;
                         text-transform:uppercase;opacity:0.55;">
                LOCAL TTS &nbsp;·&nbsp; MLX &nbsp;·&nbsp; APPLE SILICON
            </span>
            <button id="theme-toggle" onclick="toggleTheme()"></button>
        </div>
    """)

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
            model_in = gr.Dropdown(
                label="Model",
                choices=MODEL_OPTIONS,
                value=MODELS[0].id,
            )
            with gr.Row():
                voice_in = gr.Dropdown(
                    label="Voice",
                    choices=MODELS[0].voices,
                    value=MODELS[0].default_voice,
                    scale=2,
                )
                speed_in = gr.Slider(
                    label="Speed",
                    minimum=0.5,
                    maximum=2.0,
                    step=0.05,
                    value=MODELS[0].default_speed,
                    scale=1,
                )
            hint_out = gr.Textbox(
                label="Emotion tags",
                value="",
                interactive=False,
                visible=False,
                lines=1,
            )
            speak_btn = gr.Button(
                "SPEAK",
                variant="primary",
                size="lg",
                elem_id="speak-btn",
            )

        with gr.Column(scale=2):
            status_out = gr.Markdown("", elem_id="status-out")
            audio_out = gr.Audio(label="Output", autoplay=True)
            dl_btn = gr.DownloadButton(
                "DOWNLOAD MP3",
                visible=False,
                size="lg",
                elem_id="dl-btn",
            )
            error_out = gr.Textbox(label="Log / Error", lines=6, elem_id="error-box")

    file_in.change(load_file, file_in, text_in)
    model_in.change(update_voices, model_in, [voice_in, hint_out, speed_in])
    speak_btn.click(
        generate,
        [text_in, model_in, voice_in, speed_in],
        [audio_out, error_out, status_out, dl_btn],
    )


# ── Entry point ──────────────────────────────────────────────────────────────

def _run_gradio():
    demo.launch(
        server_name="0.0.0.0",
        prevent_thread_lock=True,
        css=_CSS,
        head=_HEAD,
        theme=gr.themes.Base(),
    )


if __name__ == "__main__":
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

# read-it

Local text-to-speech on Apple Silicon, running entirely on-device via MLX. Gradio UI at `http://localhost:7860`, persistent via a macOS LaunchAgent.

---

## Models

| Model | Style | Size | Speed |
|---|---|---|---|
| **Kokoro-82M** | Natural, non-autoregressive | 82M params | ~20× real-time |
| **Orpheus-3B** | Expressive, emotion tags | 3B params (4-bit) | ~3× real-time |

Kokoro loads at startup. Orpheus loads lazily on first use (~1.7 GB download).

Adding a new model is one `ModelSpec(...)` block in `app.py` — see [Architecture](#architecture).

---

## Requirements

- macOS, Apple Silicon (M1 or later)
- Python 3.13 via Homebrew
- `espeak-ng` and `ffmpeg` via Homebrew

---

## Setup

```bash
brew install espeak-ng ffmpeg

python3.13 -m venv .venv
source .venv/bin/activate

pip install spacy --prefer-binary
pip install espeakng-loader num2words phonemizer-fork
pip install misaki --no-deps
pip install mlx-audio gradio pydub scipy
pip install pytest playwright
playwright install chromium
```

> **Note:** `misaki[en]`'s full install fails on Python 3.13 because `spacy-curated-transformers` pulls a `blis` version that doesn't compile against the new C API. The manual steps above work around it.

---

## Run

```bash
.venv/bin/python3 app.py
# → http://localhost:7860
```

### Run on login (LaunchAgent)

```bash
cp com.dermotburke.tts.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.dermotburke.tts.plist
```

```bash
# Restart after changes
launchctl kickstart -k gui/$(id -u)/com.dermotburke.tts

# Logs
tail -f ~/Library/Logs/tts.log

# Status
launchctl list com.dermotburke.tts
```

---

## CLI

```bash
./speak myfile.txt
./speak myfile.txt --voice bf_emma --speed 1.1
./speak --list-voices
```

---

## Testing

Tests run a headless Chromium browser against the live service and verify audio generation end-to-end.

```bash
# Service must be running on localhost:7860
make test

# Wire up pre-push hook (run once)
make install-hooks
```

The pre-push hook blocks pushes unless all 8 tests pass.

---

## Architecture

**MLX requires all GPU ops on the main thread.** Gradio handlers run in a thread pool, so the design flips this around: Gradio runs in a background thread, the main thread loops over inference tasks from a `queue.Queue`.

### Adding a model

Define a `ModelSpec` in the `MODELS` list in `app.py`:

```python
ModelSpec(
    id="mlx-community/some-model",   # HuggingFace repo
    label="Name  ·  style",          # Dropdown label
    voices=[("name  ·  accent", "id"), ...],  # [] if no named voices
    default_voice="id",
    chunk_chars=None,    # None = sentence split, 0 = no split, N = fixed chars
    hint="",             # Shown in UI hint box
    eager=False,         # True = load at startup
    use_speed=True,      # False if model has no speed param
    default_speed=1.0,
    preprocess_text=False,
    generate_kwargs={},  # Extra kwargs passed to model.generate()
)
```

---

## Voices

### Kokoro (28 voices)

| Prefix | Accent |
|---|---|
| `af_` | American Female |
| `am_` | American Male |
| `bf_` | British Female |
| `bm_` | British Male |

Recommended: `af_heart` (default), `bf_emma`, `am_michael`.

### Orpheus (8 voices)

`tara` (default), `leah`, `jess`, `mia`, `zoe`, `leo`, `dan`, `zac`

Supports emotion tags anywhere in text: `<laugh>` `<chuckle>` `<sigh>` `<gasp>` `<cough>` `<cries>`

---

## Project layout

```
.
├── app.py                      # Gradio UI + model registry
├── speak.py                    # Core TTS logic + chunking
├── speak                       # CLI shell wrapper
├── com.dermotburke.tts.plist   # macOS LaunchAgent
├── requirements.txt
├── CLAUDE.md                   # AI contributor guidelines
├── Makefile                    # make test / make install-hooks
├── scripts/
│   └── pre-push                # Git hook — runs tests before push
└── tests/
    └── test_e2e.py             # Playwright end-to-end tests
```

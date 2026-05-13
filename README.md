# read-it

Local text-to-speech using the [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) model. Two flavours:

| | Mac (MLX) | Server (PyTorch) |
|---|---|---|
| **Target** | Apple Silicon | Linux / Docker |
| **Backend** | MLX | PyTorch CPU |
| **Speed** | ~15–20× real-time | ~2–4× real-time |
| **Entry point** | `app.py` | `server/app.py` |

Both expose the same Gradio UI at `http://localhost:7860`.

---

## Mac — MLX (Apple Silicon)

### Requirements

- macOS, Apple Silicon (M1 or later)
- Python 3.13 via Homebrew
- `espeak-ng` via Homebrew

### Setup

```bash
# Install system dependency
brew install espeak-ng

# Create venv and install packages
python3.13 -m venv .venv
source .venv/bin/activate

pip install spacy --prefer-binary
pip install espeakng-loader num2words phonemizer-fork
pip install misaki --no-deps
pip install mlx-audio gradio
```

> **Note:** `misaki[en]`'s full install fails on Python 3.13 because `spacy-curated-transformers` pulls a version of `blis` that doesn't compile against the new C API. The manual install steps above work around this.

### Run

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
# Manage
launchctl unload ~/Library/LaunchAgents/com.dermotburke.tts.plist  # stop
launchctl load   ~/Library/LaunchAgents/com.dermotburke.tts.plist  # start
tail -f ~/Library/Logs/tts.log                                      # logs
```

### CLI (no UI)

```bash
./speak myfile.txt
./speak myfile.txt --voice bf_emma --speed 1.1
./speak --list-voices
```

---

## Server — Docker (Linux / homelab)

Tested on AMD Ryzen / x86-64. Uses the PyTorch CPU backend — no CUDA or ROCm required.

### Requirements

- Docker + Docker Compose

### Run

```bash
cd server/
docker compose up -d --build
# → http://<host-ip>:7860
```

Model weights (~330 MB) are downloaded on first start and cached in a Docker volume so rebuilds are instant.

### Configuration

| Env var | Default | Description |
|---|---|---|
| `PORT` | `7860` | Port the server listens on |
| `HF_HOME` | `/cache/huggingface` | Model cache location (mount a volume here) |

---

## Voices

28 voices across four accents:

| Code prefix | Accent |
|---|---|
| `af_` | American Female |
| `am_` | American Male |
| `bf_` | British Female |
| `bm_` | British Male |

Recommended starting voices: `af_heart` (warm, default), `bf_emma` (British female), `am_michael` (American male).

Full list: `./speak --list-voices`

---

## Project layout

```
.
├── app.py                      # Gradio UI — MLX/Mac version
├── speak.py                    # CLI — MLX/Mac version
├── speak                       # Shell wrapper for speak.py
├── com.dermotburke.tts.plist   # macOS LaunchAgent
├── requirements.txt            # Pinned Mac deps
└── server/
    ├── app.py                  # Gradio UI — PyTorch/Docker version
    ├── Dockerfile
    ├── docker-compose.yml
    └── requirements.txt
```

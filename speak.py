#!/usr/bin/env python3
"""Read any text file aloud with Kokoro MLX TTS."""
import argparse
import re
import sys

from mlx_audio.tts.generate import generate_audio
from mlx_audio.tts.utils import load_model

DEFAULT_MODEL = "mlx-community/Kokoro-82M-bf16"
DEFAULT_VOICE = "af_heart"
MAX_CHUNK_CHARS = 600

VOICES = {
    "American Female": ["af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica",
                        "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky"],
    "American Male":   ["am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
                        "am_michael", "am_onyx", "am_puck", "am_santa"],
    "British Female":  ["bf_alice", "bf_emma", "bf_isabella", "bf_lily"],
    "British Male":    ["bm_daniel", "bm_fable", "bm_george", "bm_lewis"],
}


def split_into_chunks(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split text at sentence boundaries, keeping chunks under max_chars."""
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks, current = [], ""
    for sentence in sentences:
        candidate = (current + " " + sentence).strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(sentence) > max_chars:
                # Long sentence: split on commas
                parts = re.split(r"(?<=,)\s+", sentence)
                sub = ""
                for part in parts:
                    candidate = (sub + " " + part).strip() if sub else part
                    if len(candidate) <= max_chars:
                        sub = candidate
                    else:
                        if sub:
                            chunks.append(sub)
                        sub = part
                current = sub
            else:
                current = sentence

    if current:
        chunks.append(current)
    return chunks


def main():
    parser = argparse.ArgumentParser(
        description="Read a text file aloud using MLX Kokoro TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python speak.py essay.txt\n"
               "  python speak.py article.txt --voice bf_emma --speed 1.1\n"
               "  python speak.py notes.txt --voice am_michael\n"
               "  python speak.py --list-voices",
    )
    parser.add_argument("file", nargs="?", help="Text file to read aloud")
    parser.add_argument(
        "--voice", default=DEFAULT_VOICE,
        help=f"Voice to use (default: {DEFAULT_VOICE}). Use --list-voices to see all options.",
    )
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed (default: 1.0)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace model repo or local path")
    parser.add_argument("--list-voices", action="store_true", help="Show available voices and exit")
    args = parser.parse_args()

    if args.list_voices:
        for category, voices in VOICES.items():
            print(f"{category}: {', '.join(voices)}")
        return

    if not args.file:
        parser.error("a text file is required")

    try:
        with open(args.file) as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Error: file not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    if not text.strip():
        print("Error: file is empty", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model (first run will download ~330 MB)...")
    model = load_model(args.model)

    chunks = split_into_chunks(text)
    total = len(chunks)
    print(f"Voice: {args.voice}  |  Speed: {args.speed}x  |  Chunks: {total}\n")

    for i, chunk in enumerate(chunks, 1):
        if total > 1:
            print(f"[{i}/{total}] {chunk[:70]}{'...' if len(chunk) > 70 else ''}")
        generate_audio(
            text=chunk,
            model=model,
            voice=args.voice,
            speed=args.speed,
            play=True,
            stream=True,
            verbose=False,
        )


if __name__ == "__main__":
    main()

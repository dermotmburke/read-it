# read-it — Claude Code Instructions

## Verify every change by running tests

**Before reporting any task complete, run the test suite:**

```bash
make test
```

All 8 tests must pass. The service must be running on localhost:7860 first — restart it if you changed `app.py`:

```bash
launchctl kickstart -k gui/$(id -u)/com.dermotburke.tts
sleep 6
make test
```

If a test fails, fix it before finishing. Do not skip tests or mark work done without a green run.

---

## Service management

```bash
# Restart
launchctl kickstart -k gui/$(id -u)/com.dermotburke.tts

# Check status / PID
launchctl list com.dermotburke.tts

# Tail logs
tail -f ~/Library/Logs/tts.log
```

The plist at `~/Library/LaunchAgents/com.dermotburke.tts.plist` must be kept in sync with `com.dermotburke.tts.plist` in the repo root. If you edit the plist, copy it and reload:

```bash
cp com.dermotburke.tts.plist ~/Library/LaunchAgents/
launchctl unload ~/Library/LaunchAgents/com.dermotburke.tts.plist
launchctl load   ~/Library/LaunchAgents/com.dermotburke.tts.plist
```

---

## Architecture

`app.py` has one hard constraint: **MLX requires all GPU ops on the main thread**. Gradio handlers run in a thread pool, so the design flips this around:

- Gradio server runs in a **background thread**
- The **main thread** loops over inference tasks from a `queue.Queue`
- `_run_on_main_thread(fn, *args)` submits work and blocks until done

Do not move inference calls off the main thread or add `async` to the inference path.

---

## Gradio 6 gotchas

These caused hard-to-debug issues — don't repeat them:

**`elem_id` lands on the element itself, not a wrapper.**
In Gradio 6, `elem_id="speak-btn"` on a `gr.Button` sets `id="speak-btn"` directly on the `<button>` element. CSS selectors must be `#speak-btn { }` not `#speak-btn button { }`. Same for `#dl-btn`.

**`js=` on `.click()` is an input preprocessor, not a side-effect hook.**
If you pass `js=` to `speak_btn.click(fn=..., inputs=..., ...)`, Gradio 6 uses the JS function's return value as the inputs to the Python function. Returning nothing (`undefined`) sends `None` for all inputs — silently breaking the slider. Use a plain `addEventListener` in `_HEAD` instead.

**`.then(fn=None, ...)` crashes.**
Gradio 6 does not handle `fn=None` in `.then()`. It raises `TypeError` in `preprocess_data`. Use a real no-op function or avoid `.then()` entirely.

**`css=`, `head=`, `theme=` belong in `launch()`, not `Blocks()`.**
Gradio 6 moved these parameters from `gr.Blocks()` to `demo.launch()`. They are silently ignored (or cause a warning) if passed to `Blocks()`.

---

## Models

| Model | ID | Notes |
|---|---|---|
| Kokoro | `mlx-community/Kokoro-82M-bf16` | Fast, non-autoregressive. Loaded eagerly at startup. |
| Orpheus | `mlx-community/orpheus-3b-0.1-ft-4bit` | LLM-based, expressive. Loaded lazily on first use. |

**Orpheus token limit:** The Orpheus `generate()` defaults to `max_tokens=1200` (~8–15 s of audio). We pass `max_tokens=4800` and cap chunks at 250 chars. Do not remove these overrides.

**Chunk sizes:** `speak.py:MAX_CHUNK_CHARS = 600` for Kokoro. Orpheus uses 250 chars (set in `app.py:generate()`).

---

## Test suite

Tests live in `tests/test_e2e.py` and use Playwright against the live service.

```bash
make test          # run tests
make install-hooks # wire up pre-push git hook (run once after cloning)
```

The pre-push hook in `scripts/pre-push` blocks pushes unless the service is reachable and all tests pass.

When adding a new UI feature, add a corresponding test in `tests/test_e2e.py`.

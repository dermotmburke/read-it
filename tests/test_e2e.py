"""End-to-end UI tests for the read-it TTS service.

Requires the service to be running on localhost:7860.
Run with: pytest tests/test_e2e.py -v
"""
import requests
import pytest
from playwright.sync_api import sync_playwright, Page

BASE_URL = "http://localhost:7860"
MODAL_SELECTOR = "#ri-modal"
SPEAK_BTN = "#speak-btn"
STATUS_OUT = "#status-out"
ERROR_BOX = "#error-box textarea"
DL_BTN = "#dl-btn"

GENERATION_TIMEOUT = 120_000  # 2 min — Orpheus can be slow


@pytest.fixture(scope="module")
def page():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        pg = browser.new_page()
        pg.goto(BASE_URL, wait_until="networkidle")
        pg.wait_for_timeout(3000)
        yield pg
        browser.close()


def _wait_for_modal_dismiss(page: Page, timeout: int = GENERATION_TIMEOUT):
    page.wait_for_function(
        f"document.querySelector('{MODAL_SELECTOR}').style.display !== 'flex'",
        timeout=timeout,
    )


def _dom_state(page: Page) -> dict:
    return page.evaluate(f"""() => ({{
        modalDisplay: document.querySelector('{MODAL_SELECTOR}')?.style.display,
        modalStatus: document.getElementById('ri-modal-status')?.textContent,
        audioExists: !!document.querySelector('audio'),
        playBtnVisible: !!document.querySelector('button[aria-label="Play"]'),
        pauseBtnVisible: !!document.querySelector('button[aria-label="Pause"]'),
        dlVisible: !!(document.querySelector('{DL_BTN}')?.offsetParent),
        statusOut: (document.querySelector('{STATUS_OUT}')?.textContent || '').trim(),
        errorBox: (document.querySelector('{ERROR_BOX}')?.value || '').trim(),
    }})""")


class TestModal:
    def test_modal_hidden_on_load(self, page):
        state = _dom_state(page)
        assert state["modalDisplay"] != "flex", "Modal should not be visible on page load"

    def test_modal_shows_on_click(self, page):
        page.locator("textarea").first.fill("Hello, this is a quick test.")
        page.locator(SPEAK_BTN).click()
        page.wait_for_timeout(600)
        state = _dom_state(page)
        assert state["modalDisplay"] == "flex", "Modal should appear immediately after clicking SPEAK"
        assert state["modalStatus"] == "Generating audio…", (
            f"Modal status should say 'Generating audio…', got: '{state['modalStatus']}'"
        )

    def test_modal_dismisses_after_generation(self, page):
        _wait_for_modal_dismiss(page)
        state = _dom_state(page)
        assert state["modalDisplay"] != "flex", "Modal should dismiss once generation completes"


class TestGeneration:
    def test_no_error_after_generation(self, page):
        state = _dom_state(page)
        assert state["errorBox"] == "", f"Error box should be empty, got: '{state['errorBox'][:100]}'"

    def test_audio_player_present(self, page):
        state = _dom_state(page)
        assert state["audioExists"], "An <audio> element should be present after generation"

    def test_audio_starts_playing(self, page):
        # Give the auto-play click time to register
        page.wait_for_timeout(3000)
        state = _dom_state(page)
        assert state["pauseBtnVisible"], (
            "Pause button should be visible (audio playing) after generation — "
            "if Play button is still showing, autoplay did not trigger"
        )

    def test_download_button_visible(self, page):
        state = _dom_state(page)
        assert state["dlVisible"], "Download MP3 button should be visible after generation"


class TestKokoroVoice:
    """Verify a non-default Kokoro voice also produces audio."""

    def test_alternate_voice(self, page):
        page.locator("textarea").first.fill("Testing an alternate voice.")
        # Switch to af_heart via the voice dropdown
        voice_dd = page.locator("text=Voice").locator("..").locator("input")
        voice_dd.click()
        page.locator("text=af_heart").first.click()

        page.locator(SPEAK_BTN).click()
        _wait_for_modal_dismiss(page)
        page.wait_for_timeout(3000)

        state = _dom_state(page)
        assert state["errorBox"] == "", "No errors for alternate voice"
        assert state["pauseBtnVisible"], "Audio should play for alternate voice"


class TestAPI:
    """Verify the REST API endpoint at POST /api/tts."""

    API_URL = f"{BASE_URL}/api/tts"

    def test_returns_mp3(self):
        resp = requests.post(self.API_URL, json={"text": "Hello from the API."}, timeout=120)
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text[:200]}"
        assert resp.headers["content-type"] == "audio/mpeg"
        assert len(resp.content) > 1000, "MP3 response body is suspiciously small"

    def test_custom_voice(self):
        resp = requests.post(
            self.API_URL,
            json={"text": "Testing a custom voice.", "voice": "af_heart"},
            timeout=120,
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/mpeg"

    def test_invalid_voice_returns_400(self):
        resp = requests.post(
            self.API_URL,
            json={"text": "Hello.", "voice": "nonexistent_voice"},
            timeout=10,
        )
        assert resp.status_code == 400

    def test_empty_text_returns_400(self):
        resp = requests.post(self.API_URL, json={"text": ""}, timeout=10)
        assert resp.status_code == 400

    def test_speed_out_of_range_returns_400(self):
        resp = requests.post(self.API_URL, json={"text": "Hello.", "speed": 5.0}, timeout=10)
        assert resp.status_code == 400

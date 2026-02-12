import json
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List

import cv2
import numpy as np
import requests


# ----------------------------
# Pre-coded fallback responses
# ----------------------------
FALLBACK_RESPONSES = {
    "neutral": "You look neutral. If you're focusing, that's great. Want a quick goal for the next 5 minutes?",
    "happiness": "You look happy—nice! Keep that momentum. What’s one small win you can do next?",
    "surprise": "You look surprised. Did something unexpected happen? Take a breath and decide your next step calmly.",
    "sadness": "You look sad. It might help to pause for a moment. Try 3 slow breaths and do one small, easy task.",
    "anger": "You look tense/angry. Try unclenching your jaw and shoulders. A short break can reset your focus.",
    "disgust": "You look uncomfortable. If something’s bothering you, step back, reassess, and simplify the next action.",
    "fear": "You look anxious. Try grounding: name 3 things you see, 2 you feel, 1 you hear—then pick one tiny next step.",
    "contempt": "You look skeptical/annoyed. If something feels inefficient, define the problem in one sentence and adjust.",
}

# If emotion label is unknown for any reason:
DEFAULT_FALLBACK = "I couldn't classify the emotion reliably. If you want, tell me what you're feeling and I’ll respond."


# ----------------------------
# Config
# ----------------------------
@dataclass
class AIConfig:
    gemini_api_key: str = ""
    groq_api_key: str = ""
    gemini_model: str = "gemini-1.5-flash-001"
    groq_model: str = "llama-3.3-70b-versatile"


def load_config(path: str) -> AIConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return AIConfig(
        gemini_api_key=data.get("gemini_api_key", ""),
        groq_api_key=data.get("groq_api_key", ""),
        gemini_model=data.get("gemini_model", "gemini-1.5-flash-001"),
        groq_model=data.get("groq_model", "llama-3.3-70b-versatile"),
    )


# ----------------------------
# Gemini + Groq clients
# ----------------------------
class GeminiClient:
    """
    Gemini REST: POST
    https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key=API_KEY
    """
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    def generate(self, prompt: str, timeout_s: float = 10.0) -> Tuple[bool, str, Optional[int], Optional[str]]:
        if not self.api_key:
            return False, "Gemini API key missing.", None, None

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": prompt}]}
            ]
        }

        try:
            r = requests.post(url, json=payload, timeout=timeout_s)
            status = r.status_code
            txt = r.text

            # Quota/rate limit typically 429 RESOURCE_EXHAUSTED
            if status == 429:
                return False, "Gemini quota/rate limit reached (429).", status, txt

            if status < 200 or status >= 300:
                return False, f"Gemini HTTP {status}.", status, txt

            data = r.json()
            # Typical structure: candidates[0].content.parts[0].text
            candidates = data.get("candidates", [])
            if not candidates:
                return False, "Gemini returned no candidates.", status, txt

            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if not parts:
                return False, "Gemini returned empty content parts.", status, txt

            out_text = parts[0].get("text", "").strip()
            if not out_text:
                return False, "Gemini returned empty text.", status, txt

            return True, out_text, status, None

        except requests.RequestException as e:
            return False, f"Gemini request error: {e}", None, None


class GroqClient:
    """
    Groq OpenAI-compatible chat completions:
    POST https://api.groq.com/openai/v1/chat/completions
    Authorization: Bearer GROQ_API_KEY
    """
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    def chat(self, prompt: str, timeout_s: float = 10.0) -> Tuple[bool, str, Optional[int], Optional[str]]:
        if not self.api_key:
            return False, "Groq API key missing.", None, None

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a kindergarten teacher analyzing a live children webcam feed. Keep replies short, sweet and in one sentence."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
            "max_tokens": 120,
        }

        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
            status = r.status_code
            txt = r.text

            if status < 200 or status >= 300:
                return False, f"Groq HTTP {status}.", status, txt

            data = r.json()
            choices = data.get("choices", [])
            if not choices:
                return False, "Groq returned no choices.", status, txt

            msg = choices[0].get("message", {})
            out_text = (msg.get("content") or "").strip()
            if not out_text:
                return False, "Groq returned empty text.", status, txt

            return True, out_text, status, None

        except requests.RequestException as e:
            return False, f"Groq request error: {e}", None, None


# ----------------------------
# Emotion panel controller
# ----------------------------
@dataclass
class EmotionAIPanel:
    config_path: str = "api_keys.json"
    window_name: str = "AI Coach"
    width: int = 520
    height: int = 480

    collect_seconds: float = 5.0
    min_samples: int = 10

    _cfg: AIConfig = field(default_factory=AIConfig, init=False)
    _gemini: Optional[GeminiClient] = field(default=None, init=False)
    _groq: Optional[GroqClient] = field(default=None, init=False)

    _counts: Dict[str, int] = field(default_factory=dict, init=False)
    _collect_start: float = field(default_factory=time.time, init=False)

    _inflight: bool = field(default=False, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    # UI state
    last_emotion: str = "neutral"
    last_ai_text: str = "Waiting for 5s emotion summary..."
    last_source: str = "idle"
    last_error: str = ""

    def setup(self):
        self._cfg = load_config(self.config_path)
        self._gemini = GeminiClient(self._cfg.gemini_api_key, self._cfg.gemini_model)
        self._groq = GroqClient(self._cfg.groq_api_key, self._cfg.groq_model)

        self._counts.clear()
        self._collect_start = time.time()

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.width, self.height)

    def update_emotion_sample(self, emotion: str):
        # called every frame (or whenever you have a stable emotion label)
        if not emotion:
            return
        self._counts[emotion] = self._counts.get(emotion, 0) + 1

        now = time.time()
        if (now - self._collect_start) >= self.collect_seconds:
            self._collect_start = now

            total = sum(self._counts.values())
            if total >= self.min_samples and not self._inflight:
                dom = max(self._counts.items(), key=lambda kv: kv[1])[0]
                self._counts.clear()
                self._trigger_ai(dom, total)
            else:
                # reset counts anyway to avoid stale accumulation
                self._counts.clear()

    def _trigger_ai(self, dominant_emotion: str, total_samples: int):
        with self._lock:
            if self._inflight:
                return
            self._inflight = True

        self.last_emotion = dominant_emotion
        self.last_ai_text = "Thinking..."
        self.last_source = "calling"
        self.last_error = ""

        t = threading.Thread(
            target=self._ai_worker,
            args=(dominant_emotion, total_samples),
            daemon=True,
        )
        t.start()

    def _ai_worker(self, dominant_emotion: str, total_samples: int):
        prompt = (
            "You are a kindergarten teacher analyzing a live webcam feed. "
            f"The dominant facial emotion over the last 5 seconds is: {dominant_emotion} "
            f"(samples={total_samples}). "
            "Give a short supportive suggestion for the child. Make it short, concise and sweet. "
            "Text only, one sentence, must be under 50 words."
        )

        # 1) Try Gemini
        ok, text, status, raw = self._gemini.generate(prompt, timeout_s=10.0) if self._gemini else (False, "Gemini not configured.", None, None)
        if ok:
            self._set_result(text, source="gemini")
            return

        # Gemini failed: decide whether to fall back (quota or other error)
        gemini_err = text
        gemini_quota = (status == 429) or ("RESOURCE_EXHAUSTED" in (raw or "")) or ("resource_exhausted" in (raw or "").lower())
        # if quota OR any error, fall back to Groq (as you requested)
        ok2, text2, status2, raw2 = self._groq.chat(prompt, timeout_s=10.0) if self._groq else (False, "Groq not configured.", None, None)
        if ok2:
            self._set_result(text2, source="groq", error=f"Gemini failed: {gemini_err}")
            return

        # 3) Final fallback: pre-coded
        fallback = FALLBACK_RESPONSES.get(dominant_emotion, DEFAULT_FALLBACK)
        err_msg = f"Gemini failed: {gemini_err} | Groq failed: {text2}"
        self._set_result(fallback, source="local", error=err_msg)

    def _set_result(self, text: str, source: str, error: str = ""):
        with self._lock:
            self.last_ai_text = text.strip()
            self.last_source = source
            self.last_error = error.strip()
            self._inflight = False

    def draw_panel(self) -> np.ndarray:
        """
        Returns an image you can show in the 2nd window.
        """
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        def put_line(s: str, y: int, scale: float = 0.6, thick: int = 1):
            cv2.putText(img, s, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)

        put_line("Live Feed", 30, scale=0.9, thick=1)
        put_line(f"Dominant emotion (5s): {self.last_emotion}", 65, scale=0.65, thick=2)
        put_line(f"Source: {self.last_source}", 90, scale=0.6, thick=1)

        # Wrap AI text
        y = 130
        put_line("Response:", y, scale=0.65, thick=2)
        y += 28

        lines = self._wrap_text(self.last_ai_text, max_chars=46)
        for ln in lines[:10]:
            put_line(ln, y, scale=0.55, thick=1)
            y += 22

        if self.last_error:
            y = self.height - 60
            put_line("Note:", y, scale=0.55, thick=1)
            y += 22
            err_lines = self._wrap_text(self.last_error, max_chars=46)
            for ln in err_lines[:2]:
                put_line(ln, y, scale=0.5, thick=1)
                y += 20

        return img

    @staticmethod
    def _wrap_text(text: str, max_chars: int = 50) -> List[str]:
        words = (text or "").split()
        if not words:
            return [""]
        lines = []
        cur = []
        cur_len = 0
        for w in words:
            add = len(w) + (1 if cur else 0)
            if cur_len + add <= max_chars:
                cur.append(w)
                cur_len += add
            else:
                lines.append(" ".join(cur))
                cur = [w]
                cur_len = len(w)
        if cur:
            lines.append(" ".join(cur))
        return lines

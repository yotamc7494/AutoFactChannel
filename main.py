#!/usr/bin/env python3
"""
auto_shorts_generator.py - retention-first VERTICAL (9:16) "5 facts you didn't know" YouTube Shorts generator.

Same architecture as weekly_recap.py (minus upload and thumbnail):
  A. Topic: a RANDOM topics.json entry (title + the 5 facts). Stateless on purpose (the server's filesystem resets
     on every run), so a topic can come up again; with a ~180 topic pool that is an occasional repeat.
     Writer LLM (one call for the intro, one call PER fact) -> master timeline (script.json)
        scene 1  = intro: topic image + "Here are 5 facts you didn't know about X"
        scene 2-6 = one scene per fact; every narration opens with the fact itself, then explains it
  B. Images: image_gen.ImageGen (Cloudflare FLUX.2 klein, Pollinations fallback), fetched in a thread while TTS runs.
  C. TTS (Kokoro ONNX, isolated worker), synthesised sentence by sentence so every caption knows exactly when its
     sentence is spoken.
  D. Render: one isolated OS process per scene (several in parallel); every scene is an MP4 chunk that already
     contains its audio (TTS + SFX). Ken Burns zoom, pop-in captions, quick motion-blurred slide transitions.
  E. Assemble: ffmpeg concat demuxer (stream copy) + optional ducked background music.

Usage
  python auto_shorts_generator.py [--day 42] [--topics topics.json] [--output out.mp4]
                                  [--work-dir DIR] [--save-script s.json] [--load-script s.json]
                                  [--no-music] [--no-sfx] [--zoom 1.30] [--zoom-start 1.0]
                                  [--zoom-curve linear|ease_out|ease_in_out]
                                  [--transitions random|none|left,right,up,down,in,out]
                                  [--transition-seconds 0.25] [--transition-blur 0.5] [--pop-seconds 0.22]
Needs image_gen.py (your ImageGen module) and topics.json next to this file.
Env
  GEMINI_API_KEY, CLOUDFLARE_ACCOUNT_ID + CLOUDFLARE_API_TOKEN and/or POLLINATIONS_API_KEY (see image_gen.py)
  TOPICS_FILE, SHORT_ZOOM, SHORT_ZOOM_START, SHORT_ZOOM_CURVE, SHORT_TRANSITION_SECONDS, SHORT_TRANSITION_BLUR,
  SHORT_RENDER_WORKERS (optional)
Optional assets (never fatal if missing)
  assets/music/*.mp3, assets/sfx/<swoosh|impact|ding|heartbeat>.wav|mp3, assets/fonts/*.ttf (caption font; Anton
  or Impact recommended)
"""
from __future__ import annotations

import argparse
import datetime
import gc
import json
import logging
import math
import os
import pickle
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import wave
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
from pydantic import BaseModel, ConfigDict, Field

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*_a, **_k):
        return False

try:
    import httpx  # installed with google-genai; used to recognise network errors
except ImportError:  # pragma: no cover
    httpx = None

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("AutoShorts")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return float(default)


# =====================================================================
# 1. CONFIG   (the "KNOBS" blocks are the ones to edit)
# =====================================================================
TOPICS_FILE = Path(os.environ.get("TOPICS_FILE", BASE_DIR / "topics.json"))

# --- LLM (same rotation as weekly_recap.py) ---
GEMINI_MODELS = ["gemini-3.5-flash-lite", "gemini-flash-lite-latest"]
LLM_MAX_ROUNDS = 5
LLM_BASE_DELAY = 2.0
LLM_MAX_DELAY = 30.0
LLM_RETRY_STATUS = {408, 429, 500, 502, 503, 504}
WRITER_MAX_ROUNDS = 3  # rounds per Writer call

# --- Video (9:16 vertical) ---
VIDEO_W, VIDEO_H = 1080, 1920
OUTPUT_FPS = 30
X264_PRESET = "veryfast"
X264_CRF = 19
FFMPEG_THREADS = 2
RENDER_IN_SUBPROCESS = True
RENDER_WORKERS = int(os.environ.get("SHORT_RENDER_WORKERS", max(1, min(3, (os.cpu_count() or 2) // 2))))
BG_MUSIC_VOLUME = 0.10
BG_MUSIC_FADE_OUT = 2.0
USE_MUSIC = True
USE_SFX = True

# --- KNOBS: Ken Burns zoom --------------------------------------------------------------------------------------
# The image starts at ZOOM_START and ends the scene at ZOOM_END (1.30 = 30% closer). Set ZOOM_END = ZOOM_START to
# switch the zoom off.
ZOOM_START = _env_float("SHORT_ZOOM_START", 1.00)
ZOOM_END = _env_float("SHORT_ZOOM", 1.30)
ZOOM_CURVE = os.environ.get("SHORT_ZOOM_CURVE", "linear")  # linear | ease_out (fast start) | ease_in_out

# --- KNOBS: transitions between images --------------------------------------------------------------------------
# left/right/up/down = slide (left: the picture travels left, the next one enters from the right edge)
# in = push-through zoom in, out = pull-back zoom out
TRANSITIONS = ("up", "down", "left", "right", "in", "out")
TRANSITION_SPEC = "random"  # random | none | comma list that is cycled, e.g. "left,right,in"
TRANSITION_SECONDS = _env_float("SHORT_TRANSITION_SECONDS", 0.25)  # quick; 0.45 was the old value
# Motion blur during a transition, as the camera "shutter" in frames of travel (0 = off, 0.5 = film look,
# 1.0 = heavy streaks).
TRANSITION_BLUR = _env_float("SHORT_TRANSITION_BLUR", 0.5)

# --- KNOBS: caption pop animation -------------------------------------------------------------------------------
CAPTION_WORDS_MAX = 2  # words per caption chunk
POP_SECONDS = 0.10  # shorter pop feels closer to the spoken word; 0 disables it
POP_FROM_SCALE = 0.50  # starts at 50% size ...
POP_OVERSHOOT = 3.0  # ... overshoots to ~112% and settles (0 = no overshoot, just a scale-up)

# --- Pacing ---
FACTS_COUNT = 5
FACT_WORDS_MIN, FACT_WORDS_MAX = 8, 35  # safety bounds for broken output, the prompt aims at 20-30
MAX_VIDEO_SECONDS = 58.0
SEGMENT_PAD = 0.2  # silence after the last word of a scene
END_PAD = 0.5  # hold on the last picture
SENTENCE_GAP = 0.12
SPEED_SCALE_CAP = 1.25  # TTS may be sped up by at most this factor to fit MAX_VIDEO_SECONDS
# Used ONLY if Gemini is completely unavailable. The normal path is 100% AI-written.
FALLBACK_OPENING = "Here are 5 facts about {subject} that you probably never knew."

# --- Voices (Kokoro) ---
HOST_VOICE, HOST_SPEED = "am_liam", 1.15
SFX_GAIN = 0.2

# Pan paths (start, end) as a 0..1 position of the crop box inside its slack. One per scene, cycled.
MOTIONS = [
    ((0.50, 0.50), (0.50, 0.50)),  # centred push-in
    ((0.15, 0.60), (0.85, 0.40)),  # drift right
    ((0.50, 0.90), (0.50, 0.10)),  # tilt up
    ((0.85, 0.50), (0.15, 0.50)),  # drift left
    ((0.30, 0.30), (0.70, 0.70)),  # diagonal
    ((0.70, 0.30), (0.30, 0.70)),  # other diagonal
]


# =====================================================================
# 1b. SCHEMAS  (topics.json entry -> IntroPlan / FactSegment -> master timeline)
# =====================================================================
class _Model(BaseModel):
    model_config = ConfigDict(use_enum_values=True, validate_default=True)


class SfxType(str, Enum):
    none = "none"
    impact = "impact"
    ding = "ding"
    heartbeat = "heartbeat"


class IntroPlan(_Model):
    subject: str = Field(description="The topic as a short natural noun phrase (1-5 words), e.g. 'deep space'")
    title: str = Field(description="YouTube Short title, 30-70 characters, a curiosity hook that stays true to the "
                                   "facts. No hashtags, no emojis, no ALL CAPS words")
    opening_line: str = Field(description="First words spoken in the video: 1-2 short sentences, 8-20 words. Names the "
                                          "subject, promises five facts, gives a reason to keep watching")
    image_prompt: str = Field(description="One striking photorealistic scene that represents the whole topic, 9:16")
    description: str = Field(description="YouTube description, 40-90 words, conversational, no claims beyond the "
                                         "supplied facts, ends with 3-5 relevant hashtags including #Shorts")
    tags: str = Field(description="6 to 10 comma-separated tags")


class FactSegment(_Model):
    script: str = Field(description="Spoken narration: the fact first, then a short explanation. 14-18 words")
    image_prompt: str = Field(description="One concrete photorealistic scene for a text-to-image model, 9:16")
    sfx: SfxType = Field(default=SfxType.none, description="Sound effect when the picture lands")


class ShortTimeline(_Model):
    day: int
    title: str
    subject: str
    description: str
    tags: str
    facts: List[str]  # the 5 facts exactly as they appear in topics.json
    segments: List[FactSegment]  # segment 0 = intro, segments 1..5 = the facts


# =====================================================================
# 2. GENERIC HELPERS
# =====================================================================
def ev(x: Any) -> str:
    return str(x.value if isinstance(x, Enum) else x)


def ascii_clean(text: str) -> str:
    text = text or ""
    for src, dst in (("\u2018", "'"), ("\u2019", "'"), ("\u201c", '"'), ("\u201d", '"'),
                     ("\u2013", "-"), ("\u2014", "-"), ("\u2026", "...")):
        text = text.replace(src, dst)
    text = text.replace("\\", "")
    return re.sub(r"\s+", " ", re.sub(r"[^\x20-\x7E]+", "", text)).strip()


def shorten(text: str, n: int) -> str:
    return text if len(text) <= n else text[:max(1, n - 3)].rstrip() + "..."


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if s.strip()]


def clean_narration(text: str) -> str:
    """Symbols -> words so that the spoken audio and the captions show exactly the same text."""
    text = text or ""
    for src, dst in (("\u00b0C", " degrees Celsius"), ("\u00b0F", " degrees Fahrenheit"), ("\u00b0", " degrees"),
                     ("%", " percent"), ("&", " and "), ("+", " plus "), ("=", " equals "), ("/", " ")):
        text = text.replace(src, dst)
    text = re.sub(r"\$\s?(\d[\d,.]*)(\s*(?:thousand|million|billion|trillion))?",
                  lambda m: f"{m.group(1)}{m.group(2) or ''} dollars", text)
    text = ascii_clean(text)
    text = re.sub(r"[*#_`~^<>\[\]{}|]", "", text)
    text = re.sub(r"\s+[-]+\s+", ", ", text).replace(";", ",")
    return re.sub(r"\s+", " ", text).strip()


def clamp01(v: float) -> float:
    return min(1.0, max(0.0, v))


def lerp(a: float, b: float, p: float) -> float:
    return a + (b - a) * p


def ease_out(p: float) -> float:
    return 1 - (1 - p) ** 3


def ease_in_out(p: float) -> float:
    return 3 * p * p - 2 * p ** 3


def ease_out_back(p: float, k: float = 1.70158) -> float:
    p -= 1
    return 1 + (k + 1) * p ** 3 + k * p ** 2


def curve(kind: str, p: float) -> float:
    p = clamp01(p)
    if kind == "ease_out":
        return ease_out(p)
    if kind == "ease_in_out":
        return ease_in_out(p)
    return p


def fmt_ts(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 60:02d}:{s % 60:02d}"


def run_memory_worker(kind: str, *args: str) -> None:
    """Start one isolated worker, wait for exit, reclaim its RAM before the next stage."""
    env = os.environ.copy()
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[var] = "2"
    env["TOKENIZERS_PARALLELISM"] = "false"
    cmd = [sys.executable, str(Path(__file__).resolve()), f"--{kind}-worker", *map(str, args)]
    logger.info("Starting isolated %s worker", kind)
    completed = subprocess.run(cmd, env=env, check=False)
    gc.collect()
    if completed.returncode:
        msg = " (OS killed it; likely OOM)" if completed.returncode in (-9, 137) else ""
        raise RuntimeError(f"{kind} worker exit code {completed.returncode}{msg}; check the logs")


@lru_cache(maxsize=1)
def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        raise RuntimeError("ffmpeg not found. Install ffmpeg or `pip install imageio-ffmpeg`.")


def decode_audio(path: str, sr: int = 22050) -> np.ndarray:
    """Decode any audio file to mono float32 samples via ffmpeg."""
    proc = subprocess.run(
        [ffmpeg_exe(), "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"],
        capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg could not decode {path}: {proc.stderr.decode(errors='ignore')}")
    return np.frombuffer(proc.stdout, dtype=np.float32)


def write_wav_mono(path: str, samples: np.ndarray, sr: int) -> None:
    data = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(data.tobytes())


# =====================================================================
# 3. TOPIC SELECTION  (topics.json: [{day, title, facts: [5 strings]}, ...])
#    Plain random pick from the curated pool. No state is stored anywhere.
#    "day" is just a unique id here (it no longer means day of the year).
# =====================================================================
def _normalize_topic(entry: Any) -> Optional[dict]:
    """A valid topics.json entry -> {day, title, facts}; anything malformed -> None."""
    try:
        day = int(entry["day"])
        title = ascii_clean(str(entry.get("title", "")))
        facts = [ascii_clean(str(f)) for f in entry.get("facts", []) if ascii_clean(str(f))]
    except (KeyError, TypeError, ValueError, AttributeError):
        return None
    if not title or len(facts) != FACTS_COUNT:
        return None
    return {"day": day, "title": title, "facts": facts}


def load_topic_pool(topics_path: Path) -> List[dict]:
    """Every valid, unique-id entry of topics.json (bad entries are skipped with a warning, not fatal)."""
    try:
        raw = json.loads(topics_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Could not read {topics_path}: {exc}") from exc
    if not isinstance(raw, list):
        raise RuntimeError(f"{topics_path} must contain a JSON list of topics")
    pool: List[dict] = []
    seen = set()
    for entry in raw:
        topic = _normalize_topic(entry)
        if topic is None:
            logger.warning("Skipping malformed topics.json entry (needs day, title and exactly %d facts): %s",
                           FACTS_COUNT, shorten(str(entry), 80))
            continue
        if topic["day"] in seen:
            logger.warning("Skipping duplicate topic id %s", topic["day"])
            continue
        seen.add(topic["day"])
        pool.append(topic)
    if not pool:
        raise RuntimeError(f"{topics_path} has no valid topics")
    return pool


def select_topic(topics_path: Path, day: Optional[int] = None) -> dict:
    """Random topic from the pool. An explicit `day` forces that exact id (manual override)."""
    pool = load_topic_pool(topics_path)
    if day is not None:
        for topic in pool:
            if topic["day"] == day:
                logger.info("Manual override: topic %s: %s", day, topic["title"])
                return topic
        raise RuntimeError(f"{topics_path.name} has no topic with day/id {day}")
    topic = random.choice(pool)
    logger.info("Random topic %s (pool of %d): %s", topic["day"], len(pool), topic["title"])
    return topic


def subject_from_title(title: str) -> str:
    """'5 Mind-Blowing Facts About Deep Space' -> 'Deep Space' (fallback when the LLM is unavailable)."""
    m = re.match(r"^\s*(?:5|five)\s+(?:[\w'\-]+\s+){0,4}?(?:facts?|secrets?|things?)\s+"
                 r"(?:about|on|behind|from|of|to celebrate)\s+(.+)$", title, re.I)
    if m:
        return m.group(1).strip(" .!?")
    return re.sub(r"^\s*(?:5|five)\s+", "", title, flags=re.I).strip(" .!?")


# =====================================================================
# 4. LLM  (model rotation + exponential backoff, identical to weekly_recap.py)
# =====================================================================
class LLMUnavailableError(RuntimeError):
    """Every model kept failing after all retry rounds."""


@lru_cache(maxsize=1)
def get_gemini_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing!")
    from google import genai
    return genai.Client(api_key=api_key)


def _status_code(exc: Exception) -> Optional[int]:
    for attr in ("code", "status_code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    return None


def _is_transient(exc: Exception) -> bool:
    code = _status_code(exc)
    if code is not None:
        return code in LLM_RETRY_STATUS
    if httpx is not None and isinstance(exc, httpx.TransportError):
        return True
    return isinstance(exc, (ConnectionError, TimeoutError))


def ask_llm(prompt, config=None, parse: Optional[Callable] = None, max_rounds: int = LLM_MAX_ROUNDS):
    """Each round tries every model once; transient errors move on, permanent ones drop the model.
    `parse` failures (bad JSON, wrong length, editorial problems...) are retried too."""
    client = get_gemini_client()
    kwargs = {"config": config} if config else {}
    models = list(GEMINI_MODELS)
    last_exc: Optional[Exception] = None
    for round_no in range(1, max_rounds + 1):
        for model in list(models):
            try:
                response = client.models.generate_content(model=model, contents=prompt, **kwargs)
            except Exception as e:
                last_exc = e
                if _is_transient(e):
                    logger.warning(f"{model} temporarily unavailable (round {round_no}/{max_rounds}): {e}")
                    continue
                logger.error(f"{model} failed with a non-retryable error, removing it from rotation: {e}")
                models.remove(model)
                if not models:
                    raise
                continue
            if parse is None:
                return response
            try:
                return parse(response)
            except Exception as e:
                last_exc = e
                logger.warning(f"{model} returned unusable output (round {round_no}/{max_rounds}): {e}")
        if round_no < max_rounds:
            delay = min(LLM_MAX_DELAY, LLM_BASE_DELAY * 2 ** (round_no - 1)) * random.uniform(0.5, 1.0)
            logger.info(f"Retrying in {delay:.1f}s (round {round_no + 1}/{max_rounds})...")
            time.sleep(delay)
    raise LLMUnavailableError(
        f"Gemini still failing after {max_rounds} rounds across {GEMINI_MODELS}. Last error: {last_exc}") from last_exc


def _gen_config(types, schema, temperature: float):
    kwargs = dict(response_mime_type="application/json", response_schema=schema, temperature=temperature)
    try:  # the SDK's automatic function calling is not needed here and only produces warnings
        return types.GenerateContentConfig(
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True), **kwargs)
    except Exception:
        return types.GenerateContentConfig(**kwargs)


# =====================================================================
# 5. SCRIPT NORMALISATION / VALIDATION / LLM PROMPTS  (Writer)
# =====================================================================
FILLER = re.compile(
    r"\b(?:did you know|in conclusion|let'?s dive in|buckle up|without further ado|welcome back|hey guys|"
    r"you won'?t believe|mind[- ]blowing fact|here'?s (?:a|one) (?:fact|thing))\b", re.I)
GREETING = re.compile(r"^\s*(?:hey|hi|hello|welcome|what'?s up|today (?:we|i))\b", re.I)
STOPWORDS = {"about", "after", "also", "been", "because", "before", "being", "could", "does", "during", "each",
             "even", "every", "from", "have", "into", "just", "like", "made", "many", "more", "most", "much",
             "only", "other", "over", "same", "some", "such", "than", "that", "their", "them", "then", "there",
             "these", "they", "this", "those", "through", "under", "very", "were", "what", "when", "where",
             "which", "while", "will", "with", "would", "your"}


def _content_stems(text: str) -> set:
    return {w[:5] for w in re.findall(r"[a-z0-9]+", text.lower()) if (len(w) > 3 or w.isdigit()) and w not in STOPWORDS}


def _numbers(text: str) -> set:
    return {m.replace(",", "") for m in re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", text)}


def normalize_fact(seg: FactSegment) -> None:
    seg.script = clean_narration(seg.script)
    seg.image_prompt = ascii_clean(seg.image_prompt)
    words = len(seg.script.split())
    if not FACT_WORDS_MIN <= words <= FACT_WORDS_MAX:
        raise ValueError(f"narration has {words} words (allowed {FACT_WORDS_MIN}-{FACT_WORDS_MAX})")
    if len(seg.image_prompt.split()) < 6:
        raise ValueError("image_prompt is too short to be a usable scene")
    if ev(seg.sfx) not in {s.value for s in SfxType}:
        seg.sfx = SfxType.none.value


def parse_fact(response, fact: str, near_misses: Optional[list] = None, last: bool = False) -> FactSegment:
    """The narration must open with the supplied fact, keep its numbers and sound like a person, not a textbook."""
    seg = FactSegment.model_validate_json(response.text)
    normalize_fact(seg)
    issues = []
    first = split_sentences(seg.script)[0] if split_sentences(seg.script) else seg.script
    if FILLER.search(seg.script) or GREETING.search(seg.script):
        issues.append("filler phrase or greeting")
    wanted = _content_stems(fact)
    if wanted and len(wanted & _content_stems(first)) / len(wanted) < 0.5:
        issues.append("first sentence does not state the fact")
    script_numbers = _numbers(seg.script)
    if not _numbers(fact) <= script_numbers:
        issues.append("changed or dropped a number from the fact")
    if len(split_sentences(seg.script)) < 2:
        issues.append("no explanation after the fact")
    if last and not re.search(r"\bfollow\b", seg.script, re.I):
        issues.append("missing follow invitation")
    if issues:
        if near_misses is not None:
            near_misses.append((len(issues), seg))
        raise ValueError("; ".join(issues))
    return seg


def normalize_intro(p: IntroPlan) -> IntroPlan:
    p.subject = ascii_clean(p.subject).strip(" .!?\"'")
    if not 1 <= len(p.subject.split()) <= 6 or re.search(r"\bfacts?\b|^\d", p.subject, re.I):
        raise ValueError(f"unusable subject '{p.subject}'")

    p.title = re.sub(r"#\S+", "", ascii_clean(p.title)).strip(" \"'")
    if not 20 <= len(p.title) <= 100 or len(p.title.split()) < 3:
        raise ValueError(f"title has {len(p.title)} characters (allowed 20-100)")
    if sum(1 for w in p.title.split() if len(w) > 3 and w.isupper()) >= 2:
        raise ValueError("title uses ALL CAPS words")

    p.opening_line = clean_narration(p.opening_line)
    words = len(p.opening_line.split())
    if not 6 <= words <= 26:
        raise ValueError(f"opening line has {words} words (allowed 6-26)")
    if GREETING.search(p.opening_line):
        raise ValueError("opening line starts with a greeting")
    if not re.search(r"\b(?:5|five)\b", p.opening_line, re.I):
        raise ValueError("opening line does not promise the five facts")
    stems = _content_stems(p.subject)
    if stems and not stems & _content_stems(p.opening_line):
        raise ValueError("opening line does not mention the subject")

    p.image_prompt = ascii_clean(p.image_prompt)
    if len(p.image_prompt.split()) < 8:
        raise ValueError("image_prompt is too short")

    p.description = ascii_clean(p.description)
    if not 25 <= len(p.description.split()) <= 140:
        raise ValueError("description length is out of range")
    p.tags = ascii_clean(p.tags)
    return p


def write_intro(entry: dict) -> IntroPlan:
    from google.genai import types
    logger.info("Writer: title, opening line and description for '%s'", entry["title"])
    facts = "\n".join(f"  {i + 1}. {f}" for i, f in enumerate(entry["facts"]))
    prompt = f"""
            You are packaging a YouTube Short. Its working title is "{entry['title']}" and it presents these five facts:
        {facts}

            Return:
            - subject: the topic as a short natural noun phrase (1-5 words), e.g. 'deep space', 'your brain',
              'ancient Egypt'.
            - title: the YouTube title. Write a fresh, original hook of 30-70 characters that makes someone stop
              scrolling: a question, a surprising claim or a bold contrast drawn from the facts. It must be true to the
              facts and must NOT simply copy the working title. No hashtags, no emojis, no ALL CAPS words.
            - opening_line: the very first words spoken in the video (8-20 words, one or two short sentences). Name
              the subject, tell the viewer five facts are coming and give them a reason to stay. Invent it fresh in a
              natural spoken voice: no greeting, no 'did you know', and do NOT fall back on a stock formula like
              'Here are 5 facts about ...'.
            - image_prompt: ONE striking photorealistic establishing scene that represents the WHOLE topic and works
              as the opening shot of the video.
            - description: 40-90 words. A hook first, then tease what the viewer will learn without listing all five
              facts word for word, then a closing line of 3-5 relevant hashtags that includes #Shorts. No claims beyond
              the supplied facts.
            - tags: 6-10 comma-separated tags.

            {STYLE_GUIDE}
            """
    try:
        return ask_llm(prompt, config=_gen_config(types, IntroPlan, 0.9),
                       parse=lambda r: normalize_intro(IntroPlan.model_validate_json(r.text)),
                       max_rounds=WRITER_MAX_ROUNDS)
    except LLMUnavailableError:
        subject = subject_from_title(entry["title"])
        logger.warning("Intro writer unavailable; using a plain fallback intro for '%s'", subject)
        return IntroPlan(subject=subject, title=shorten(entry["title"], 100),
                         opening_line=FALLBACK_OPENING.format(subject=subject), tags="facts, did you know, shorts",
                         description=entry["title"] + ". " + " ".join(entry["facts"][:2]) + " #Shorts #Facts",
                         image_prompt=f"Cinematic photorealistic wide establishing shot that represents {subject}, "
                                      "dramatic lighting, rich detail, vertical composition")


STYLE_GUIDE = """
        VOICE: You are a smart, charismatic, high-energy YouTuber telling a friend something that just blew your mind,
        NOT a documentary narrator. Use 'you' naturally. Short, punchy sentences. Contractions and everyday English.
        Energy comes from the fact itself, not from fake hype or shouting. Never write like an encyclopedia entry.

        TRUTH: The supplied facts are the ONLY source of claims. Never change, round, drop or add numbers, names or
        dates. You may add widely known, safe context in the explanation, but if you are not certain, explain why the
        fact matters or how it feels instead of inventing new numbers or claims.

        TTS: Natural spoken English, simple punctuation (periods, commas, question marks). No emojis, hashtags,
        bullet symbols, parentheses or markdown. Digits are fine for years and ordinary numbers. Spell out units.

        IMAGE PROMPTS (text-to-image, vertical 9:16): ONE concrete photorealistic scene that makes the subject visible
        at a glance: subject, setting, camera and lens, lighting, mood, under 60 words. State an object count only when
        it matters ('exactly three'). Never include text, letters, logos, watermarks, real named people, celebrities,
        gore, copyrighted characters, collages or split screens.
        """


def write_fact(entry: dict, subject: str, index: int, previous_tail: str) -> FactSegment:
    from google.genai import types
    fact = entry["facts"][index]
    last = index == len(entry["facts"]) - 1
    logger.info("Writer: fact %d/%d", index + 1, len(entry["facts"]))
    closing = ("LAST FACT: after the explanation add ONE closing takeaway sentence about the whole topic, then a short "
               "invitation to follow for more (6-10 words in total for the closing line).") if last else ""
    overview = "\n".join(f"  {i + 1}. {f}" for i, f in enumerate(entry["facts"]))
    prompt = f"""
            You are a charismatic YouTuber writing ONE fact segment (fact {index + 1} of {len(entry['facts'])}) of a Short
            about {subject}. The intro has already been spoken, so there is NO greeting and NO lead-in.

            THE FACT (use its exact numbers, names and dates): {fact}

            STRUCTURE:
            1. The FIRST sentence states the fact itself, plainly and completely, so the viewer instantly knows what it
               is. Smooth the wording for speech but keep every number, name and date exactly as given. Never open with
               'did you know', 'here's something' or a vague teaser.
            2. Then ONE micro-explanation sentence (5-8 words): why it is true or why it matters.
            Total 14-18 words strictly. {closing}
            Example of the pattern: 'The Rosetta Stone unlocked ancient Egypt. It let scholars finally read hieroglyphs.'

            Also choose image_prompt (see the guide) and sfx (none unless a sound effect truly helps: impact for a shocking
            reveal, ding for a satisfying detail, heartbeat for something eerie or alive).
            {('Previous segment ended: ' + previous_tail) if previous_tail else ''}
            ALL FIVE FACTS (for context only, do not mention the others):
        {overview}

            {STYLE_GUIDE}
            """
    near_misses: list = []
    try:
        return ask_llm(prompt, config=_gen_config(types, FactSegment, 0.7),
                       parse=lambda r: parse_fact(r, fact, near_misses, last), max_rounds=WRITER_MAX_ROUNDS)
    except LLMUnavailableError:
        if not near_misses:
            raise
        score, seg = min(near_misses, key=lambda x: x[0])
        logger.warning("Fact %d: accepting best available draft (%d editorial issues)", index + 1, score)
        return seg


def generate_timeline(entry: dict) -> ShortTimeline:
    intro = write_intro(entry)
    logger.info("AI title: %s", intro.title)
    logger.info("AI opening line: %s", intro.opening_line)
    segments = [FactSegment(script=intro.opening_line, image_prompt=intro.image_prompt, sfx=SfxType.none.value)]
    tail = intro.opening_line
    for i in range(len(entry["facts"])):
        seg = write_fact(entry, intro.subject, i, tail)
        segments.append(seg)
        tail = shorten(seg.script, 160)
    return ShortTimeline(day=entry["day"], title=intro.title, subject=intro.subject, description=intro.description,
                         tags=intro.tags, facts=list(entry["facts"]), segments=segments)


# =====================================================================
# 6. IMAGES  (your image_gen.ImageGen: Cloudflare FLUX.2 klein, Pollinations zimage fallback)
# =====================================================================
def placeholder_image(index: int, out_path: Path) -> None:
    """Dark gradient fallback so one failed image never kills the whole video."""
    rng = random.Random(index * 9973 + 17)
    top = np.array([rng.randint(8, 50) for _ in range(3)], dtype=np.float32)
    bottom = np.array([rng.randint(40, 120) for _ in range(3)], dtype=np.float32)
    t = np.linspace(0.0, 1.0, VIDEO_H, dtype=np.float32)[:, None, None]
    arr = np.repeat(top * (1 - t) + bottom * t, VIDEO_W, axis=1).astype(np.uint8)
    Image.fromarray(arr, "RGB").save(out_path, format="PNG")


def generate_images(segments: Sequence[FactSegment], work_dir: Path) -> List[str]:
    try:
        from image_gen import ImageGen
    except ImportError as exc:
        raise RuntimeError("image_gen.py must sit next to auto_shorts_generator.py") from exc
    out_dir = work_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    gen = ImageGen(output_dir=out_dir, final_width=VIDEO_W, final_height=VIDEO_H)

    def one(i: int) -> Optional[Path]:
        try:
            return gen.generate(segments[i].image_prompt, index=i + 1, basename="scene")
        except Exception as e:  # noqa: BLE001 - one bad image must not cost the whole video
            logger.error("Image %d failed: %s", i + 1, e)
            return None

    with ThreadPoolExecutor(max_workers=gen.max_workers) as pool:
        results = list(pool.map(one, range(len(segments))))
    failed = [i for i, p in enumerate(results) if p is None]
    if len(failed) > 2:
        raise RuntimeError(f"{len(failed)}/{len(segments)} images failed; check the image provider keys and quotas")
    paths: List[str] = []
    for i, p in enumerate(results):
        if p is None:
            p = out_dir / f"placeholder_{i + 1:02d}.png"
            placeholder_image(i, p)
            logger.warning("Using a gradient placeholder for scene %d", i + 1)
        paths.append(str(p))
    logger.info("Images ready. %s", gen.summary_text())
    return paths


# =====================================================================
# 7. TEXT-TO-SPEECH (Kokoro-82M ONNX, isolated worker, sentence-level timings)
# =====================================================================
KOKORO_DIR = BASE_DIR / "kokoro_assets"  # shared with weekly_recap.py when both scripts live in one folder
_KOKORO_BASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
KOKORO_MODEL_NAME = "kokoro-v1.0.int8.onnx"
KOKORO_VOICES_NAME = "voices-v1.0.bin"


def _download(url: str, dest: Path) -> None:
    import requests
    logger.info("Downloading %s ...", dest.name)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, allow_redirects=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    tmp.replace(dest)


def _get_or_download_kokoro_assets() -> Tuple[str, str]:
    KOKORO_DIR.mkdir(parents=True, exist_ok=True)
    model_path, voices_path = KOKORO_DIR / KOKORO_MODEL_NAME, KOKORO_DIR / KOKORO_VOICES_NAME
    for path in (model_path, voices_path):
        if not path.exists():
            _download(f"{_KOKORO_BASE}/{path.name}", path)
    return str(model_path), str(voices_path)


@lru_cache(maxsize=1)
def get_kokoro_onnx():
    from kokoro_onnx import Kokoro
    model_path, voices_path = _get_or_download_kokoro_assets()
    return Kokoro(model_path, voices_path)


def clean_script_for_tts(text: str) -> str:
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    text = re.sub(r"\s+[-\u2013\u2014]+\s+|[\u2013\u2014]", ", ", text).replace(";", ",").replace("...", ".")
    return text.replace("\n", " ").strip()


def generate_voiceovers(items: List[dict]) -> List[dict]:
    """items = [{text, voice, speed, path}] -> [{duration, sentences: [[start, end], ...]}]. Every sentence is
    synthesised on its own and joined with a short pause, so every caption knows when its sentence is spoken.
    Runs inside the isolated TTS worker."""
    import soundfile as sf
    kokoro = get_kokoro_onnx()
    try:
        available = set(kokoro.get_voices())
    except Exception:
        available = set()
    results = []
    logger.info("Generating %d Kokoro clips", len(items))
    for it in items:
        voice = it["voice"] if (not available or it["voice"] in available) else "am_liam"
        if voice != it["voice"]:
            logger.warning("Voice %s missing, using %s", it["voice"], voice)
        sentences = split_sentences(it["text"]) or [it["text"]]
        parts, times, cursor, sr = [], [], 0.0, 24000
        for i, sent in enumerate(sentences):
            clean = clean_script_for_tts(sent)
            samples = np.zeros(0, dtype=np.float32)
            if re.search(r"[A-Za-z0-9]", clean):
                samples, sr = kokoro.create(clean, voice=voice, speed=float(it["speed"]), lang="en-us")
                samples = np.asarray(samples, dtype=np.float32)
            dur = len(samples) / sr
            times.append([cursor, cursor + dur])
            if len(samples):
                parts.append(samples)
                cursor += dur
            if i < len(sentences) - 1:
                parts.append(np.zeros(int(SENTENCE_GAP * sr), dtype=np.float32))
                cursor += SENTENCE_GAP
        if not parts:
            raise RuntimeError(f"Kokoro returned empty audio for {it['path']}")
        audio = np.concatenate(parts)
        sf.write(str(it["path"]), audio, sr, subtype="PCM_16")
        results.append({"duration": len(audio) / sr, "sentences": times})
        logger.info("Created %s (%.2fs, %s, %d sentences)", Path(it["path"]).name, results[-1]["duration"], voice,
                    len(sentences))
    return results


def run_tts(items: List[dict], work_dir: Path) -> List[dict]:
    in_file, out_file = work_dir / "tts_in.json", work_dir / "tts_out.json"
    payload = json.dumps(items)
    if (in_file.exists() and out_file.exists() and in_file.read_text(encoding="utf-8") == payload
            and all(Path(i["path"]).exists() for i in items)):
        logger.info("Reusing the voiceover from the previous run")
        return json.loads(out_file.read_text(encoding="utf-8"))
    in_file.write_text(payload, encoding="utf-8")
    out_file.unlink(missing_ok=True)
    run_memory_worker("tts", str(in_file), str(out_file))
    return json.loads(out_file.read_text(encoding="utf-8"))


def voice_offsets(n: int, transitions_on: bool) -> List[float]:
    """The first scene starts speaking at 00:00; later scenes wait half a transition so the new picture lands first."""
    return [0.0] + [(TRANSITION_SECONDS * 0.5 if transitions_on else 0.0)] * (n - 1)


def scene_durations(tts: List[dict], offsets: List[float]) -> List[float]:
    pads = [SEGMENT_PAD] * len(tts)
    pads[-1] = END_PAD
    return [o + r["duration"] + p for o, r, p in zip(offsets, tts, pads)]


def run_tts_fitted(segments: Sequence[FactSegment], work_dir: Path, transitions_on: bool) -> Tuple[List[dict], List[dict]]:
    """Natural speed first; if the Short would exceed MAX_VIDEO_SECONDS, re-voice slightly faster (capped)."""
    scale = 1.0
    items: List[dict] = []
    tts: List[dict] = []
    for _ in range(3):
        items = [{"text": s.script, "voice": HOST_VOICE, "speed": round(HOST_SPEED * scale, 3),
                  "path": str(work_dir / f"segment_{i}.wav")} for i, s in enumerate(segments)]
        tts = run_tts(items, work_dir)
        total = sum(scene_durations(tts, voice_offsets(len(tts), transitions_on)))
        if total <= MAX_VIDEO_SECONDS or scale >= SPEED_SCALE_CAP:
            break
        scale = min(SPEED_SCALE_CAP, scale * total / MAX_VIDEO_SECONDS * 1.01)
        logger.info("%.1fs is over the %.0fs budget; re-voicing at speed x%.2f", total, MAX_VIDEO_SECONDS, scale)
    total = sum(scene_durations(tts, voice_offsets(len(tts), transitions_on)))
    if total > 60.0:
        logger.warning("Final runtime %.1fs is over 60s. Consider lowering the words per fact.", total)
    return items, tts


# =====================================================================
# 7b. WORD-LEVEL ALIGNMENT  (optional WhisperX; uses the already generated WAV)
# =====================================================================
# `pip install whisperx` enables real word timing. Without it the existing
# approximate sentence/character timing still works, with a warning.
_WORD_ALIGN_VERSION = 1


def _align_one_voiceover(item: dict, info: dict, model, metadata, whisperx, device: str):
    """Return [[start,end], ...] in WAV-local seconds, or None if alignment failed."""
    sentences = split_sentences(item["text"]) or [item["text"]]
    sentence_times = info["sentences"]
    if len(sentences) != len(sentence_times):
        return None
    transcript = [
        {"start": float(bounds[0]), "end": float(bounds[1]), "text": clean_script_for_tts(sentence)}
        for sentence, bounds in zip(sentences, sentence_times)
    ]
    # We already know the exact script: no speech-to-text transcription needed.
    aligned = whisperx.align(transcript, model, metadata, str(item["path"]), device,
                             interpolate_method="linear")
    raw_words = aligned.get("word_segments", [])
    expected = sum(len(sentence.split()) for sentence in sentences)
    if len(raw_words) != expected:
        logger.warning("Alignment word count mismatch for %s (%d vs %d); using approximate captions",
                       Path(item["path"]).name, len(raw_words), expected)
        return None

    result = []
    index = 0
    for sentence, (s0, s1) in zip(sentences, sentence_times):
        s0, s1 = float(s0), float(s1)
        count = len(sentence.split())
        batch = raw_words[index:index + count]
        index += count
        good = True
        prev = s0 - 0.10
        for w in batch:
            a, b = w.get("start"), w.get("end")
            if (not isinstance(a, (int, float)) or not isinstance(b, (int, float))
                    or not math.isfinite(a) or not math.isfinite(b)
                    or a < s0 - 0.12 or b > s1 + 0.12 or a > b or a < prev - 0.06):
                good = False
                break
            prev = a
        if not good:
            # One sentence with unalignable tokens (often numerals) should not
            # discard accurate word timings in the other sentences.
            logger.warning("Incomplete alignment for a sentence in %s; falling back for that sentence",
                           Path(item["path"]).name)
            weights = [max(1, len(w.strip('.,!?;:'))) for w in sentence.split()]
            unit = (s1 - s0) / max(1, sum(weights))
            cursor = s0
            for weight in weights:
                result.append([cursor, cursor + weight * unit])
                cursor += weight * unit
        else:
            result.extend([[max(s0, float(w["start"])), min(s1, float(w["end"]))] for w in batch])
    return result


def alignment_worker_main(in_json: str, out_json: str) -> None:
    """Load the alignment model in an isolated worker to free its RAM afterwards."""
    import torch
    import whisperx

    payload = json.loads(Path(in_json).read_text(encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, metadata = whisperx.load_align_model(language_code="en", device=device)
    result = []
    for item, info in zip(payload["items"], payload["tts"]):
        try:
            words = _align_one_voiceover(item, info, model, metadata, whisperx, device)
        except Exception as exc:
            logger.warning("Word alignment failed for %s: %s", Path(item["path"]).name, exc)
            words = None
        result.append(words)
    Path(out_json).write_text(json.dumps(result), encoding="utf-8")


def run_word_alignment(items: List[dict], tts: List[dict], work_dir: Path) -> List[Optional[list]]:
    """Cache the aligned word times; returns None per scene on any failure."""
    import importlib.util
    if importlib.util.find_spec("whisperx") is None:
        logger.warning("WhisperX is not installed: captions are only estimated. "
                       "Run `pip install whisperx` for word-level audio sync.")
        return [None] * len(items)

    in_path = work_dir / "alignment_in.json"
    out_path = work_dir / "alignment_out.json"
    payload = json.dumps({"version": _WORD_ALIGN_VERSION, "items": items, "tts": tts,
                          "wav_stats": [[Path(it["path"]).stat().st_size,
                                         Path(it["path"]).stat().st_mtime_ns] for it in items]},
                         sort_keys=True)
    if (in_path.exists() and out_path.exists()
            and in_path.read_text(encoding="utf-8") == payload):
        logger.info("Reusing cached word alignment")
        return json.loads(out_path.read_text(encoding="utf-8"))
    in_path.write_text(payload, encoding="utf-8")
    out_path.unlink(missing_ok=True)
    try:
        run_memory_worker("align", str(in_path), str(out_path))
        aligned = json.loads(out_path.read_text(encoding="utf-8"))
        if len(aligned) != len(items):
            raise ValueError("The alignment worker returned the wrong number of scenes")
        return aligned
    except Exception as exc:
        logger.warning("Word alignment unavailable (%s). Falling back to estimated timings.", exc)
        return [None] * len(items)


# =====================================================================
# 8. SOUND EFFECTS  (user files in assets/sfx win; otherwise synthesised, so nothing is ever missing)
# =====================================================================
SFX_SR = 44100


def synth_sfx(name: str, sr: int = SFX_SR) -> np.ndarray:
    rng = np.random.default_rng(5)

    def tt(d):
        return np.arange(int(d * sr)) / sr

    def norm(x, peak=0.8):
        return x / (np.abs(x).max() + 1e-9) * peak

    def swoosh(d=0.45):
        t = tt(d)
        a = 0.03 + 0.55 * np.sin(np.pi * t / d) ** 2  # cut-off sweeps up then down
        noise, y, acc = rng.standard_normal(len(t)), np.zeros(len(t)), 0.0
        for i in range(len(t)):
            acc += a[i] * (noise[i] - acc)
            y[i] = acc
        return norm(y * np.sin(np.pi * t / d) ** 2)

    def boom(d=1.4):
        t = tt(d)
        phase = 2 * np.pi * np.cumsum(38 + 70 * np.exp(-t * 5)) / sr
        thump = np.sin(phase) * np.exp(-t * 3.2)
        grit = np.convolve(rng.standard_normal(len(t)), np.ones(40) / 40, mode="same") * np.exp(-t * 9)
        return norm(thump + 0.6 * grit)

    def bell(f, d=0.9):
        t = tt(d)
        return norm(np.sin(2 * np.pi * f * t) * np.exp(-t * 6) + 0.5 * np.sin(2 * np.pi * 2.76 * f * t) * np.exp(-t * 9))

    if name == "swoosh":
        return swoosh()
    if name == "impact":
        return boom()
    if name == "ding":
        return bell(1568.0)
    if name == "heartbeat":
        buf = np.zeros(int(1.7 * sr))
        t = tt(0.35)
        thump = np.sin(2 * np.pi * 58 * t) * np.exp(-t * 22)
        for at, g in ((0.0, 1.0), (0.26, 0.75), (0.95, 1.0), (1.21, 0.75)):
            i = int(at * sr)
            buf[i:i + len(thump)] += thump[:len(buf) - i] * g
        return norm(buf, 0.9)
    return np.zeros(1, dtype=np.float32)


def get_sfx_path(name: str, work_dir: Path) -> Optional[str]:
    if not USE_SFX or not name or name == "none":
        return None
    for ext in ("wav", "mp3", "ogg"):
        p = BASE_DIR / "assets" / "sfx" / f"{name}.{ext}"
        if p.exists():
            return str(p)
    out = work_dir / f"sfx_{name}.wav"
    if not out.exists():
        try:
            write_wav_mono(str(out), synth_sfx(name), SFX_SR)
        except Exception as e:
            logger.warning(f"Could not create SFX {name}: {e}")
            return None
    return str(out)


# =====================================================================
# 9. DRAWING PRIMITIVES  (fonts, captions, scrim, Ken Burns, transitions with motion blur)
# =====================================================================
_FONT_CANDIDATES = [
    "C:/Windows/Fonts/impact.ttf", "C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/segoeuib.ttf",
    "/System/Library/Fonts/Supplemental/Impact.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
]


def find_font(explicit: Optional[str] = None) -> Optional[str]:
    candidates: List[str] = [explicit] if explicit else []
    fonts_dir = BASE_DIR / "assets" / "fonts"
    if fonts_dir.is_dir():
        candidates += [str(p) for p in sorted(fonts_dir.glob("*.ttf"))] + [str(p) for p in sorted(fonts_dir.glob("*.otf"))]
    candidates += _FONT_CANDIDATES
    for path in candidates:
        if path and Path(path).is_file():
            return path
    return None


def load_font(path: Optional[str], size: int):
    if path:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int, stroke: int) -> str:
    lines: List[str] = []
    current = ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        if not current or draw.textlength(trial, font=font) + 2 * stroke <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return "\n".join(lines)


def render_text_overlay(font_path: Optional[str], text: str, *, start_size: int, min_size: int, max_width: int,
                        fill: Tuple[int, int, int, int], stroke_ratio: float = 0.09, shadow: bool = True,
                        pad: int = 36) -> Image.Image:
    scratch = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
    size = start_size
    while True:
        font = load_font(font_path, size)
        stroke = round(size * stroke_ratio)
        spacing = round(size * 0.12)
        wrapped = wrap_text(scratch, text, font, max_width, stroke)
        left, top, right, bottom = scratch.multiline_textbbox(
            (0, 0), wrapped, font=font, stroke_width=stroke, spacing=spacing, align="center")
        left, top = int(math.floor(left)), int(math.floor(top))
        right, bottom = int(math.ceil(right)), int(math.ceil(bottom))
        if (right - left) <= max_width or size <= min_size:
            break
        size -= 4
    width, height = (right - left) + 2 * pad, (bottom - top) + 2 * pad
    origin = (pad - left, pad - top)
    main = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    ImageDraw.Draw(main).multiline_text(origin, wrapped, font=font, fill=fill, stroke_width=stroke,
                                        stroke_fill=(0, 0, 0, 255), spacing=spacing, align="center")
    if not shadow:
        return main
    shade = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    ImageDraw.Draw(shade).multiline_text((origin[0], origin[1] + 7), wrapped, font=font, fill=(0, 0, 0, 210),
                                         stroke_width=stroke, stroke_fill=(0, 0, 0, 210), spacing=spacing,
                                         align="center")
    return Image.alpha_composite(shade.filter(ImageFilter.GaussianBlur(7)), main)


def caption_overlay(font_path: Optional[str], text: str) -> Tuple[Image.Image, Tuple[int, int]]:
    img = render_text_overlay(font_path, text, start_size=112, min_size=60, max_width=VIDEO_W - 150,
                              fill=(255, 255, 255, 255))
    return img, ((VIDEO_W - img.width) // 2, int(VIDEO_H * 0.5 - img.height / 2))


def apply_scrim(img: Image.Image) -> Image.Image:
    """Bake a soft dark band behind the caption zone and a slight shade at the bottom for legibility."""
    arr = np.asarray(img, dtype=np.float32)
    h = arr.shape[0]
    y = np.linspace(0.0, 1.0, h, dtype=np.float32)
    dim = 1.0 - 0.35 * np.exp(-((y - 0.5) ** 2) / (2 * 0.12 ** 2))
    bottom = y > 0.80
    dim[bottom] *= 1.0 - 0.30 * ((y[bottom] - 0.80) / 0.20)
    arr *= dim[:, None, None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def load_scene_image(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if img.size != (VIDEO_W, VIDEO_H):
        img = ImageOps.fit(img, (VIDEO_W, VIDEO_H), method=Image.Resampling.LANCZOS)
    return apply_scrim(img)


def kenburns_frame(base: Image.Image, style: dict, pan: Tuple[Tuple[float, float], Tuple[float, float]],
                   p: float) -> Image.Image:
    e = curve(style["zoom_curve"], p)
    zoom = max(1.0, lerp(style["zoom_start"], style["zoom_end"], e))
    px, py = lerp(pan[0][0], pan[1][0], e), lerp(pan[0][1], pan[1][1], e)
    box_w, box_h = VIDEO_W / zoom, VIDEO_H / zoom
    x0 = min(max((VIDEO_W - box_w) * px, 0.0), VIDEO_W - box_w)
    y0 = min(max((VIDEO_H - box_h) * py, 0.0), VIDEO_H - box_h)
    box = (x0, y0, min(float(VIDEO_W), x0 + box_w), min(float(VIDEO_H), y0 + box_h))
    # A float crop box makes Pillow sample sub-pixel positions: no zoom jitter.
    return base.resize((VIDEO_W, VIDEO_H), Image.Resampling.BICUBIC, box=box)


def _zoom_center(img: Image.Image, s: float) -> Image.Image:
    """s >= 1: crop the centre (sub-pixel) and scale back to full size."""
    w, h = img.size
    bw, bh = w / s, h / s
    x0, y0 = (w - bw) / 2, (h - bh) / 2
    return img.resize((w, h), Image.Resampling.BICUBIC, box=(x0, y0, x0 + bw, y0 + bh))


def box_blur_axis(arr: np.ndarray, length: int, axis: int) -> np.ndarray:
    """Directional box blur of a uint8 image along one axis (cumulative-sum trick, O(n), edges replicated)."""
    length = int(length) | 1
    if length < 3:
        return arr
    r = length // 2
    pad = [(0, 0)] * arr.ndim
    pad[axis] = (r, r)
    c = np.cumsum(np.pad(arr, pad, mode="edge"), axis=axis, dtype=np.int32)
    zero_shape = list(c.shape)
    zero_shape[axis] = 1
    c = np.concatenate([np.zeros(zero_shape, dtype=np.int32), c], axis=axis)
    n = arr.shape[axis]
    hi = [slice(None)] * arr.ndim
    lo = [slice(None)] * arr.ndim
    hi[axis], lo[axis] = slice(length, length + n), slice(0, n)
    return ((c[tuple(hi)] - c[tuple(lo)] + length // 2) // length).astype(np.uint8)


def _slide(old: Image.Image, new: Image.Image, kind: str, e: float) -> Image.Image:
    w, h = new.size
    canvas = Image.new("RGB", (w, h))
    ox = oy = nx = ny = 0
    if kind == "left":  # everything travels left, the next picture enters from the right edge
        ox, nx = -round(e * w), round((1 - e) * w)
    elif kind == "right":
        ox, nx = round(e * w), -round((1 - e) * w)
    elif kind == "up":
        oy, ny = -round(e * h), round((1 - e) * h)
    else:  # down
        oy, ny = round(e * h), -round((1 - e) * h)
    canvas.paste(old, (ox, oy))
    canvas.paste(new, (nx, ny))
    return canvas


def _zoom_transition(old: Image.Image, new: Image.Image, kind: str, e: float) -> Image.Image:
    w, h = new.size
    if kind == "in":  # push-through: the old picture rushes at the camera while the new one settles
        new_l = _zoom_center(new, lerp(1.25, 1.0, e))
        old_l = _zoom_center(old, lerp(1.0, 1.6, e))
        return Image.blend(new_l, old_l, 1.0 - e)
    canvas = _zoom_center(new, lerp(1.12, 1.0, e))  # "out": the old picture shrinks away and reveals the new one
    s = lerp(1.0, 0.55, e)
    small = old.resize((max(1, round(w * s)), max(1, round(h * s))), Image.Resampling.BILINEAR)
    mask = Image.new("L", small.size, int(round(255 * (1.0 - e))))
    canvas.paste(small, ((w - small.width) // 2, (h - small.height) // 2), mask)
    return canvas


def transition_frame(old: Image.Image, new: Image.Image, kind: str, p: float, td: float = 0.25,
                     blur: float = 0.0, fps: int = OUTPUT_FPS) -> Image.Image:
    """CapCut-style transition between two full frames, with motion blur. p = 0 shows `old`, p = 1 shows `new`.
    `blur` is the shutter length in frames of travel (0 = sharp). Slides get a directional blur whose length follows
    the picture's speed; zooms average a few sub-frames."""
    p = clamp01(p)
    e = ease_in_out(p)
    frames_in_transition = max(td * fps, 1e-6)
    if kind in ("left", "right", "up", "down"):
        frame = _slide(old, new, kind, e)
        if blur > 0:
            horizontal = kind in ("left", "right")
            speed = 6.0 * p * (1.0 - p) * (VIDEO_W if horizontal else VIDEO_H) / frames_in_transition  # px per frame
            length = int(speed * blur)
            if length >= 3:
                limit = (VIDEO_W if horizontal else VIDEO_H) // 3
                blurred = box_blur_axis(np.asarray(frame), min(length, limit), axis=1 if horizontal else 0)
                frame = Image.fromarray(np.ascontiguousarray(blurred))
        return frame
    if kind in ("in", "out"):
        if blur > 0:
            span = blur / frames_in_transition  # share of the transition that one shutter covers
            n = 6
            acc = None
            for i in range(n):
                pi = clamp01(p + (i / (n - 1) - 0.5) * span)
                a = np.asarray(_zoom_transition(old, new, kind, ease_in_out(pi)), dtype=np.float32)
                acc = a if acc is None else acc + a
            return Image.fromarray((acc / n + 0.5).astype(np.uint8))
        return _zoom_transition(old, new, kind, e)
    return new


# =====================================================================
# 10. CAPTIONS  (sentence timings -> short word groups that pop in)
# =====================================================================
@dataclass(frozen=True)
class CaptionCue:
    start: float
    end: float
    text: str


def build_caption_cues(sentences: List[str], times: List[List[float]], offset: float, speech_dur: float,
                       max_words: int = CAPTION_WORDS_MAX,
                       word_times: Optional[List[List[float]]] = None) -> List[CaptionCue]:
    """Two-word caption groups; prefer measured word timestamps over guesses."""
    expected_words = sum(len(s.split()) for s in sentences)
    if word_times is not None and len(word_times) != expected_words:
        word_times = None
    if len(times) != len(sentences):
        total_chars = sum(len(s) for s in sentences) or 1
        times, cur = [], 0.0
        for s in sentences:
            d = speech_dur * len(s) / total_chars
            times.append([cur, cur + d])
            cur += d

    cues: List[CaptionCue] = []
    word_idx = 0
    for sent, (s0, s1) in zip(sentences, times):
        words = sent.split()
        if not words:
            continue
        chunks: List[List[str]] = []
        cur_chunk: List[str] = []
        for w in words:
            cur_chunk.append(w)
            if len(cur_chunk) >= max_words or w[-1] in ",;:":
                chunks.append(cur_chunk)
                cur_chunk = []
        if cur_chunk:
            if chunks and len(cur_chunk) == 1 and len(chunks[-1]) < max_words and chunks[-1][-1][-1] not in ",;:":
                chunks[-1] += cur_chunk
            else:
                chunks.append(cur_chunk)

        # Fallback: the original duration-by-character estimate.
        weights = [float(sum(len(w) for w in ch) + len(ch) + (3 if ch[-1][-1] in ",;:" else 0))
                   for ch in chunks]
        unit = max(0.2, s1 - s0) / max(1.0, sum(weights))
        t = offset + s0
        for ch, wt in zip(chunks, weights):
            if word_times is not None:
                first = word_times[word_idx]
                last = word_times[word_idx + len(ch) - 1]
                start, end = offset + first[0], offset + last[1]
            else:
                start, end = t, t + wt * unit
            text = re.sub(r"[.,;:]+$", "", " ".join(ch)).upper()
            cues.append(CaptionCue(start, end, text))
            t += wt * unit
            word_idx += len(ch)

    # Hold each caption through the small gap before the next word starts.
    out = [CaptionCue(c.start, max(c.end, n.start), c.text) for c, n in zip(cues, cues[1:])]
    if cues:
        out.append(CaptionCue(cues[-1].start, cues[-1].end + 0.25, cues[-1].text))
    return out

# =====================================================================
# 11. SCENE  (frame_at(t) -> RGB uint8 array; t is scene-local)
# =====================================================================
class ShortScene:
    def __init__(self, args: Dict[str, Any], duration: float, start_time: float):
        self.args, self.duration = args, duration
        self.style = args["style"]
        font = find_font(self.style.get("font_path"))
        self.base = load_scene_image(args["image_path"])
        self.pan = MOTIONS[args["motion_index"] % len(MOTIONS)]
        sentences = split_sentences(args["seg"]["script"])
        self.cues = build_caption_cues(sentences, args["sentence_times"], args["voice_offset"], args["speech_dur"],
                                       self.style["caption_words"], args.get("word_times"))
        self.cue_imgs = [caption_overlay(font, c.text) for c in self.cues]
        self.hold: Optional[Image.Image] = None
        self.transition = args.get("transition")
        prev = args.get("prev")
        if prev and self.transition:  # the previous scene's last frame, frozen, is what slides/zooms away
            self.hold = kenburns_frame(load_scene_image(prev["image_path"]), self.style,
                                       MOTIONS[prev["motion_index"] % len(MOTIONS)], 1.0)

    def _draw_caption(self, frame: Image.Image, t: float) -> None:
        idx = next((i for i, c in enumerate(self.cues) if c.start <= t < c.end), None)
        if idx is None:
            return
        img, (x, y) = self.cue_imgs[idx]
        dt = t - self.cues[idx].start
        pop = self.style["pop_seconds"]
        if pop > 0 and dt < pop:
            p = clamp01(dt / pop)
            s = max(0.05, lerp(self.style["pop_from"], 1.0, ease_out_back(p, self.style["pop_overshoot"])))
            w, h = img.size
            sw, sh = max(1, round(w * s)), max(1, round(h * s))
            scaled = img.resize((sw, sh), Image.Resampling.BILINEAR)
            alpha = clamp01(p * 2.5)
            if alpha < 1.0:
                scaled.putalpha(scaled.getchannel("A").point(lambda v, a=alpha: int(v * a)))
            frame.paste(scaled, (round(x + w / 2 - sw / 2), round(y + h / 2 - sh / 2)), scaled)
        else:
            frame.paste(img, (x, y), img)

    def frame_at(self, t: float) -> np.ndarray:
        frame = kenburns_frame(self.base, self.style, self.pan, t / max(self.duration, 1e-6))
        td = self.style["transition_seconds"]
        if self.hold is not None and td > 0 and t < td:
            frame = transition_frame(self.hold, frame, self.transition, t / td, td, self.style["transition_blur"])
        self._draw_caption(frame, t)  # captions stay put on top of the sliding pictures
        return np.asarray(frame)


# =====================================================================
# 12. VIDEO WRITING  (each scene -> one MP4 chunk WITH its own audio; one isolated process per scene)
# =====================================================================
@dataclass
class SceneJob:
    out_path: str
    start_time: float  # position on the global timeline (seconds)
    duration: float
    args: Dict[str, Any] = field(default_factory=dict)


def frame_range(start_time: float, duration: float) -> Tuple[int, int]:
    """Global frame indices [first, last). Cumulative rounding => no drift between scenes."""
    return round(start_time * OUTPUT_FPS), round((start_time + duration) * OUTPUT_FPS)


def build_scene_audio(job: SceneJob, n_frames: int) -> str:
    """Mono 44.1 kHz wav of exactly n_frames/FPS seconds: voice + whoosh + sfx, each at its own offset."""
    sr = 44100
    total = int(round(n_frames / OUTPUT_FPS * sr))
    mix = np.zeros(total, dtype=np.float32)
    for path, gain, at in job.args["audio_layers"]:
        if path and os.path.exists(path):
            try:
                s = decode_audio(path, sr) * gain
                i = int(round(at * sr))
                m = max(0, min(len(s), total - i))
                if m:
                    mix[i:i + m] += s[:m]
            except Exception as e:
                logger.warning(f"Could not mix {path}: {e}")
    peak = float(np.abs(mix).max()) if total else 0.0
    if peak > 0.98:
        mix *= 0.98 / peak
    write_wav_mono(job.args["mix_path"], mix, sr)
    return job.args["mix_path"]


def render_scene_job(job: SceneJob) -> str:
    """Render one scene to an MP4 whose audio track is already embedded."""
    first, last = frame_range(job.start_time, job.duration)
    n_frames = max(1, last - first)
    mix_path = build_scene_audio(job, n_frames)
    scene = ShortScene(job.args, job.duration, job.start_time)
    cmd = [
        ffmpeg_exe(), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{VIDEO_W}x{VIDEO_H}", "-r", str(OUTPUT_FPS), "-i", "-",
        "-i", mix_path,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-preset", X264_PRESET, "-crf", str(X264_CRF), "-pix_fmt", "yuv420p",
        "-threads:v", str(FFMPEG_THREADS), "-filter_threads", "1",
        "-af", "pan=stereo|c0=c0|c1=c0", "-c:a", "aac", "-b:a", "160k", "-ar", "44100",
        "-t", f"{n_frames / OUTPUT_FPS:.6f}", job.out_path,
    ]
    with tempfile.TemporaryFile() as err_file:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=err_file)
        try:
            for i in range(n_frames):
                t = max(0.0, (first + i) / OUTPUT_FPS - job.start_time)
                proc.stdin.write(np.ascontiguousarray(scene.frame_at(t)))
            proc.stdin.close()
            proc.wait()
        except BrokenPipeError:
            proc.wait()
        except BaseException:
            proc.kill()
            proc.wait()
            raise
        if proc.returncode != 0:
            err_file.seek(0)
            raise RuntimeError(f"ffmpeg failed for {Path(job.out_path).name}: {err_file.read().decode(errors='ignore')}")
    del scene
    gc.collect()
    return job.out_path


def job_signature(job: SceneJob) -> str:
    raw = json.dumps([job.start_time, job.duration, job.args], sort_keys=True, default=str)
    return f"{zlib.crc32(raw.encode('utf-8')):08x}"


def render_scene_jobs(jobs: List[SceneJob]) -> List[str]:
    """Scenes are skipped only if they were rendered with exactly the same script, timing and style."""
    paths = []
    for job in jobs:
        marker = Path(job.out_path + ".done")
        sig = job_signature(job)
        if marker.exists() and marker.read_text() == sig and Path(job.out_path).exists():
            logger.info(f"Scene already rendered, skipping: {Path(job.out_path).name}")
        else:
            render_scene_job(job)
            marker.write_text(sig)
        paths.append(job.out_path)
    return paths


def render_all_scenes(jobs: List[SceneJob], work_dir: Path) -> List[str]:
    """One isolated OS process per scene (several at once): each scene dies with its process."""
    workers = max(1, RENDER_WORKERS) if RENDER_IN_SUBPROCESS else 1
    logger.info("Rendering %d scenes (%d at a time)", len(jobs), workers)

    def run_one(k: int) -> int:
        if RENDER_IN_SUBPROCESS:
            input_file = work_dir / f"render_scene_{k}.pkl"
            with input_file.open("wb") as f:
                pickle.dump([jobs[k]], f, protocol=pickle.HIGHEST_PROTOCOL)
            try:
                run_memory_worker("render", str(input_file))
            finally:
                input_file.unlink(missing_ok=True)
        else:
            render_scene_jobs([jobs[k]])
        logger.info(f"Scene {k + 1}/{len(jobs)} rendered")
        return k

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(run_one, range(len(jobs))))
    return [j.out_path for j in jobs]


# =====================================================================
# 13. FINAL ASSEMBLY  (concat demuxer of A/V chunks; stream copy; optional ducked music on the audio only)
# =====================================================================
def pick_music() -> Optional[str]:
    if not USE_MUSIC:
        return None
    folder = BASE_DIR / "assets" / "music"
    tracks = sorted(folder.glob("*.mp3")) if folder.exists() else []
    return str(random.choice(tracks)) if tracks else None


def assemble_final_video(scene_paths: List[str], total_seconds: float, output_path: str, work_dir: Path) -> None:
    list_file = work_dir / "scenes.txt"
    list_file.write_text("".join("file '{}'\n".format(Path(p).resolve().as_posix().replace("'", "'\\''"))
                                 for p in scene_paths))
    base = [ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(list_file)]
    music = pick_music()
    if music:
        logger.info("Mixing background music (ducked under the voice): %s", music)
        fade = max(0.0, total_seconds - BG_MUSIC_FADE_OUT)
        graph = (
            "[0:a]aformat=sample_rates=44100:channel_layouts=stereo,asplit=2[vo][sc];"
            f"[1:a]aformat=sample_rates=44100:channel_layouts=stereo,atrim=duration={total_seconds:.3f},"
            f"asetpts=PTS-STARTPTS,volume={BG_MUSIC_VOLUME},afade=t=out:st={fade:.3f}:d={BG_MUSIC_FADE_OUT}[bg0];"
            "[bg0][sc]sidechaincompress=threshold=0.03:ratio=8:attack=30:release=500[bg];"
            "[vo][bg]amix=inputs=2:duration=first:dropout_transition=0,volume=2[aout]")
        cmd = base + ["-stream_loop", "-1", "-i", music, "-filter_complex", graph, "-map", "0:v:0", "-map", "[aout]",
                      "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-t",
                      f"{total_seconds:.3f}", str(output_path)]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode == 0:
            return
        logger.warning("Music mix failed, assembling without music: %s", proc.stderr.decode(errors="ignore")[-400:])
    cmd = base + ["-c", "copy", "-movflags", "+faststart", str(output_path)]
    logger.info(f"Assembling final video: {output_path}")
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg assembly failed: {proc.stderr.decode(errors='ignore')}")


# =====================================================================
# 14. PIPELINE
# =====================================================================
def current_style() -> Dict[str, Any]:
    """Everything the isolated render workers need (module globals do not travel to subprocesses)."""
    return {"zoom_start": ZOOM_START, "zoom_end": ZOOM_END, "zoom_curve": ZOOM_CURVE,
            "transition_seconds": TRANSITION_SECONDS, "transition_blur": TRANSITION_BLUR,
            "pop_seconds": POP_SECONDS, "pop_from": POP_FROM_SCALE, "pop_overshoot": POP_OVERSHOOT,
            "caption_words": CAPTION_WORDS_MAX, "font_path": find_font()}


def choose_transitions(count: int, spec: str, seed: int) -> List[Optional[str]]:
    """One transition per boundary between two scenes. Deterministic per script, so a resumed run matches."""
    spec = (spec or "random").strip().lower()
    if spec == "none" or count <= 0 or TRANSITION_SECONDS <= 0:
        return [None] * max(0, count)
    wanted = [s.strip() for s in spec.split(",") if s.strip() in TRANSITIONS]
    if wanted:
        return [wanted[i % len(wanted)] for i in range(count)]
    rng = random.Random(seed)
    result: List[str] = []
    for _ in range(count):
        options = [t for t in TRANSITIONS if not result or t != result[-1]]
        result.append(rng.choice(options))
    return result


def run_shorts_pipeline(output_filename: Optional[str] = None, day: Optional[int] = None,
                        date: Optional[datetime.date] = None, topics_path: Optional[str] = None,
                        upload=False) -> str:
    """day = force one topic id (manual override); otherwise a random topic. date is deprecated and ignored."""
    logger.info("Launching AutoShorts pipeline (vertical 1080x1920)...")
    if date is not None:
        logger.warning("'date' is ignored: topics are now picked randomly.")
    work_dir = Path(tempfile.mkdtemp(prefix="auto_shorts_"))  # everything intermediate lives here and is deleted
    try:
        # ---- Phase A: random topic -> Writers -> master timeline ----
        master = generate_timeline(select_topic(Path(topics_path) if topics_path else TOPICS_FILE, day))
        for seg in master.segments[1:]:  # segment 0 is the AI opening line
            normalize_fact(seg)
        segs = list(master.segments)
        n = len(segs)
        logger.info("=" * 60)
        logger.info(f"Day {master.day} | {master.title}")
        for i, s in enumerate(segs):
            logger.info(f"  {'intro' if i == 0 else 'fact ' + str(i)} [{s.sfx}] :: {shorten(s.script, 100)}")
        logger.info("=" * 60)

        # ---- Phase B+C: images (network) and voices (CPU, isolated worker) at the same time ----
        seed = zlib.crc32((master.title + "".join(master.facts)).encode("utf-8"))
        transitions = choose_transitions(n - 1, TRANSITION_SPEC, seed)
        transitions_on = any(transitions)
        with ThreadPoolExecutor(max_workers=1) as pool:
            image_future = pool.submit(generate_images, segs, work_dir)
            speech_items, tts = run_tts_fitted(segs, work_dir, transitions_on)
            image_paths = image_future.result()
        aligned_words = run_word_alignment(speech_items, tts, work_dir)
        offsets = voice_offsets(n, transitions_on)
        durations = [r["duration"] for r in tts]
        seg_durs = scene_durations(tts, offsets)
        total = sum(seg_durs)
        logger.info("Narration %.1fs; video runtime %.1fs", sum(durations), total)

        # ---- Phase D: timeline -> scene jobs ----
        style = current_style()
        jobs: List[SceneJob] = []
        cursor = 0.0
        motion_shift = seed % len(MOTIONS)
        for k, s in enumerate(segs):
            kind = transitions[k - 1] if k > 0 else None
            layers = [[speech_items[k]["path"], 1.0, offsets[k]]]
            if kind:
                layers.append([get_sfx_path("swoosh", work_dir), SFX_GAIN, 0.0])
            sfx_path = get_sfx_path(ev(s.sfx), work_dir)
            if sfx_path:
                layers.append([sfx_path, SFX_GAIN, (TRANSITION_SECONDS * 0.6) if kind else 0.05])
            prev = {"image_path": image_paths[k - 1], "motion_index": k - 1 + motion_shift} if kind else None
            jobs.append(SceneJob(str(work_dir / f"scene_{k}.mp4"), cursor, seg_durs[k], args=dict(
                seg=s.model_dump(), index=k, image_path=image_paths[k], motion_index=k + motion_shift,
                transition=kind, prev=prev, sentence_times=tts[k]["sentences"], voice_offset=offsets[k],
                speech_dur=durations[k], word_times=aligned_words[k],
                audio_layers=layers, mix_path=str(work_dir / f"mix_{k}.wav"),
                style=style)))
            cursor += seg_durs[k]
        logger.info(f"Final length: {cursor:.1f}s ({fmt_ts(cursor)}) in {len(jobs)} scenes; transitions: "
                    f"{', '.join(t or '-' for t in transitions)}")

        if not output_filename:
            out_dir = BASE_DIR / "output"
            out_dir.mkdir(parents=True, exist_ok=True)
            output_filename = str(out_dir / f"SHORT_DAY{master.day:03d}_{datetime.datetime.now():%Y%m%d_%H%M%S}.mp4")
        Path(output_filename).parent.mkdir(parents=True, exist_ok=True)

        # ---- Phase E: render + assemble (the ONLY file that leaves the temp folder) ----
        scene_paths = render_all_scenes(jobs, work_dir)
        assemble_final_video(scene_paths, cursor, output_filename, work_dir)
        logger.info(f"SUCCESS! Short saved as: {output_filename} ({fmt_ts(cursor)})")

        if upload:
            from uploader import upload_video
            tags = [t.strip() for t in master.tags.split(",") if t.strip()]
            logger.info("Uploading with AI title: %s", master.title)
            upload_video(file_path=output_filename, title=shorten(master.title, 100),
                         description=master.description, tags=tags)
        return output_filename
    finally:
        logger.info("Cleaning up temporary files...")
        shutil.rmtree(work_dir, ignore_errors=True)


# =====================================================================
# 15. ENTRY POINTS (parent CLI + isolated workers)
# =====================================================================
def tts_worker_main(in_json: str, out_json: str) -> None:
    items = json.loads(Path(in_json).read_text(encoding="utf-8"))
    Path(out_json).write_text(json.dumps(generate_voiceovers(items)), encoding="utf-8")


def render_worker_main(pkl_path: str) -> None:
    # This pickle is written by this very application into its own private temp directory.
    with open(pkl_path, "rb") as f:
        jobs = pickle.load(f)
    render_scene_jobs(jobs if isinstance(jobs, list) else [jobs])


def main() -> None:
    global USE_MUSIC, USE_SFX, ZOOM_START, ZOOM_END, ZOOM_CURVE, TRANSITION_SPEC, TRANSITION_SECONDS
    global TRANSITION_BLUR, POP_SECONDS, HOST_VOICE, RENDER_WORKERS
    parser = argparse.ArgumentParser(description="Generate the vertical '5 facts you didn't know' Short of the day.")
    parser.add_argument("--day", type=int, default=None,
                        help="force this topics.json id (default: random topic)")
    parser.add_argument("--date", default=None, help="DEPRECATED and ignored (topics are picked randomly now)")
    parser.add_argument("--topics", default=None, help=f"topics file (default {TOPICS_FILE.name})")
    parser.add_argument("--output", default=None, help="output mp4 path (default: output/SHORT_DAY###_<timestamp>.mp4)")
    parser.add_argument("--no-music", action="store_true", help="do not mix background music")
    parser.add_argument("--no-sfx", action="store_true", help="no whoosh / sound effects")
    parser.add_argument("--zoom", type=float, default=None, help=f"zoom reached at the end of each scene (default {ZOOM_END})")
    parser.add_argument("--zoom-start", type=float, default=None, help=f"zoom at the start of each scene (default {ZOOM_START})")
    parser.add_argument("--zoom-curve", choices=["linear", "ease_out", "ease_in_out"], default=None)
    parser.add_argument("--transitions", default=None,
                        help="random (default) | none | comma list cycled in order: up,down,left,right,in,out")
    parser.add_argument("--transition-seconds", type=float, default=None, help=f"default {TRANSITION_SECONDS}")
    parser.add_argument("--transition-blur", type=float, default=None,
                        help=f"motion blur shutter in frames, 0 = off (default {TRANSITION_BLUR})")
    parser.add_argument("--pop-seconds", type=float, default=None, help=f"caption pop length, 0 = off (default {POP_SECONDS})")
    parser.add_argument("--voice", default=None, help=f"Kokoro voice (default {HOST_VOICE}; am_onyx is deeper)")
    parser.add_argument("--render-workers", type=int, default=None, help="scenes rendered in parallel")
    cli = parser.parse_args()
    if cli.no_music:
        USE_MUSIC = False
    if cli.no_sfx:
        USE_SFX = False
    if cli.zoom is not None:
        ZOOM_END = max(1.0, cli.zoom)
    if cli.zoom_start is not None:
        ZOOM_START = max(1.0, cli.zoom_start)
    if cli.zoom_curve:
        ZOOM_CURVE = cli.zoom_curve
    if cli.transitions is not None:
        TRANSITION_SPEC = cli.transitions
    if cli.transition_seconds is not None:
        TRANSITION_SECONDS = max(0.0, cli.transition_seconds)
    if cli.transition_blur is not None:
        TRANSITION_BLUR = max(0.0, cli.transition_blur)
    if cli.pop_seconds is not None:
        POP_SECONDS = max(0.0, cli.pop_seconds)
    if cli.voice:
        HOST_VOICE = cli.voice
    if cli.render_workers:
        RENDER_WORKERS = max(1, cli.render_workers)
    if cli.date:
        logger.warning("--date is deprecated and ignored: topics are picked randomly.")
    run_shorts_pipeline(output_filename=cli.output, day=cli.day, topics_path=cli.topics, upload=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--tts-worker":
        if len(sys.argv) != 4:
            raise SystemExit("Usage: --tts-worker input.json durations.json")
        tts_worker_main(sys.argv[2], sys.argv[3])
    elif len(sys.argv) > 1 and sys.argv[1] == "--render-worker":
        if len(sys.argv) != 3:
            raise SystemExit("Usage: --render-worker scene_jobs.pkl")
        render_worker_main(sys.argv[2])
    elif len(sys.argv) > 1 and sys.argv[1] == "--align-worker":
        if len(sys.argv) != 4:
            raise SystemExit("Usage: --align-worker input.json output.json")
        alignment_worker_main(sys.argv[2], sys.argv[3])
    else:
        main()
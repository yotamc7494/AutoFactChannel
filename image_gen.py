#!/usr/bin/env python3
"""
image_gen.py - ImageGen for AutoFactChannel.

Primary:  Cloudflare Workers AI  @cf/black-forest-labs/flux-2-klein-4b  (free daily neurons)
Fallback: Pollinations           model "zimage"                          (free key, paced)

Required .env:
    CLOUDFLARE_ACCOUNT_ID=...
    CLOUDFLARE_API_TOKEN=...
    POLLINATIONS_API_KEY=...      # secret key (sk_...), not the pk_ app key

Requires: pip install requests pillow python-dotenv
"""
from __future__ import annotations

import base64
import hashlib
import io
import logging
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence
from urllib.parse import quote

import requests
from PIL import Image, ImageStat

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

log = logging.getLogger("ImageGen")

DEFAULT_STYLE_SUFFIX = ", photorealistic, sharp focus, 9:16 vertical composition"
RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS
BREAKER_THRESHOLD = 4        # consecutive failures before a provider is paused
BREAKER_COOLDOWN_S = 300     # pause length after the breaker trips


# --------------------------------------------------------------------------- errors
class ImageGenError(Exception):
    """Raised when no provider could produce a valid image."""


class ProviderError(Exception):
    def __init__(self, message: str, status: Optional[int] = None, retryable: bool = False,
                 fatal: bool = False, quota_exhausted: bool = False,
                 retry_after: Optional[float] = None):
        super().__init__(message)
        self.status = status
        self.retryable = retryable          # worth retrying the same provider
        self.fatal = fatal                  # bad credentials: stop using provider this session
        self.quota_exhausted = quota_exhausted
        self.retry_after = retry_after


def _seconds_until_utc_midnight() -> float:
    now = datetime.now(timezone.utc)
    nxt = (now + timedelta(days=1)).replace(hour=0, minute=1, second=0, microsecond=0)
    return (nxt - now).total_seconds()


# --------------------------------------------------------------------------- providers
class BaseProvider:
    name = "base"
    env_vars: tuple = ()
    max_prompt_chars = 1800
    min_interval = 0.0          # minimum seconds between request starts
    max_concurrency = 1
    max_attempts = 2            # per image, per provider (transient errors only)
    quota_patterns: Optional[re.Pattern] = None

    def __init__(self, width: int, height: int, timeout: float):
        self.width, self.height, self.timeout = width, height, timeout
        self._sem = threading.Semaphore(self.max_concurrency)
        self._pace_lock = threading.Lock()
        self._last_start = 0.0
        self._state_lock = threading.Lock()
        self._consecutive_failures = 0
        self._blocked_until = 0.0
        self._blocked_reason = ""

    # ---- configuration / state
    @classmethod
    def configured(cls) -> bool:
        return all(os.getenv(v) for v in cls.env_vars)

    @classmethod
    def missing_env(cls) -> List[str]:
        return [v for v in cls.env_vars if not os.getenv(v)]

    def available(self) -> bool:
        with self._state_lock:
            return time.time() >= self._blocked_until

    def blocked_reason(self) -> str:
        with self._state_lock:
            remaining = max(0, int(self._blocked_until - time.time()))
            return f"{self._blocked_reason} (~{remaining}s left)"

    def block(self, seconds: float, reason: str) -> None:
        with self._state_lock:
            self._blocked_until = max(self._blocked_until, time.time() + seconds)
            self._blocked_reason = reason
        log.warning("Provider %s paused for ~%ds: %s", self.name, int(seconds), reason)

    def quota_block_seconds(self) -> float:
        return _seconds_until_utc_midnight()

    def record_success(self) -> None:
        with self._state_lock:
            self._consecutive_failures = 0

    def record_failure(self) -> None:
        trip = False
        with self._state_lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= BREAKER_THRESHOLD:
                self._consecutive_failures = 0
                trip = True
        if trip:
            self.block(BREAKER_COOLDOWN_S, f"{BREAKER_THRESHOLD} consecutive failures")

    # ---- request plumbing
    def _pace(self) -> None:
        if self.min_interval <= 0:
            return
        with self._pace_lock:
            wait = self.min_interval - (time.monotonic() - self._last_start)
            if wait > 0:
                time.sleep(wait)
            self._last_start = time.monotonic()

    def _raise_for(self, resp: requests.Response) -> None:
        status = resp.status_code
        body = resp.text[:300].replace("\n", " ")
        msg = f"HTTP {status}: {body}"
        ra = resp.headers.get("Retry-After", "")
        retry_after = float(ra) if ra.replace(".", "", 1).isdigit() else None
        if self.quota_patterns and self.quota_patterns.search(resp.text or ""):
            raise ProviderError(msg, status, quota_exhausted=True)
        if status in (401, 403):
            raise ProviderError(msg, status, fatal=True)
        if status == 402:
            raise ProviderError(msg, status, quota_exhausted=True)
        if status == 429 or status >= 500:
            raise ProviderError(msg, status, retryable=True, retry_after=retry_after)
        raise ProviderError(msg, status)      # 400/404/422 etc: not worth retrying

    def _request(self, prompt: str, seed: int) -> bytes:
        raise NotImplementedError

    def generate(self, prompt: str, seed: int) -> bytes:
        prompt = prompt[: self.max_prompt_chars]
        last: Optional[ProviderError] = None
        for attempt in range(1, self.max_attempts + 1):
            with self._sem:
                self._pace()
                try:
                    return self._request(prompt, seed + (attempt - 1) * 7919)
                except ProviderError as exc:
                    last = exc
                except requests.RequestException as exc:
                    last = ProviderError(f"network error: {exc}", retryable=True)
            if last.fatal or last.quota_exhausted or not last.retryable or attempt == self.max_attempts:
                break
            delay = min(last.retry_after or (3 * 2 ** attempt), 60)
            log.warning("%s attempt %d/%d failed (%s); retrying in %.0fs", self.name, attempt,
                        self.max_attempts, last, delay)
            time.sleep(delay)
        assert last is not None
        raise last


class CloudflareFlux2Klein(BaseProvider):
    name = "cloudflare_flux2_klein"
    env_vars = ("CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN")
    max_prompt_chars = 1800            # API limit is 2048
    max_concurrency = 3
    quota_patterns = re.compile(r"daily free allocation|\b4006\b|used up your", re.I)

    def _request(self, prompt: str, seed: int) -> bytes:
        acct = os.environ["CLOUDFLARE_ACCOUNT_ID"]
        token = os.environ["CLOUDFLARE_API_TOKEN"]
        url = (f"https://api.cloudflare.com/client/v4/accounts/{acct}/ai/run/"
               "@cf/black-forest-labs/flux-2-klein-4b")
        files = {  # (None, value) => plain multipart form fields
            "prompt": (None, prompt),
            "width": (None, str(self.width)),
            "height": (None, str(self.height)),
            "seed": (None, str(seed)),
        }
        resp = requests.post(url, headers={"Authorization": f"Bearer {token}"}, files=files,
                             timeout=self.timeout)
        if resp.status_code != 200:
            self._raise_for(resp)
        if resp.headers.get("Content-Type", "").lower().startswith("image/"):
            return resp.content
        try:
            payload = resp.json()
        except ValueError:
            raise ProviderError(f"Non-JSON response: {resp.text[:200]!r}")
        if payload.get("success") is False:
            errors = str(payload.get("errors"))
            raise ProviderError(f"API error: {errors[:300]}",
                                quota_exhausted=bool(self.quota_patterns.search(errors)))
        result = payload.get("result") or payload
        b64 = result.get("image") if isinstance(result, dict) else None
        if not b64:
            raise ProviderError(f"No image in response: {str(payload)[:300]}")
        return base64.b64decode(b64 + "=" * (-len(b64) % 4))


class PollinationsZImage(BaseProvider):
    name = "pollinations_zimage"
    env_vars = ("POLLINATIONS_API_KEY",)
    max_prompt_chars = 1000            # prompt travels in the URL
    min_interval = 6.0                 # free balance rate-limits aggressively
    max_concurrency = 1
    max_attempts = 3

    def quota_block_seconds(self) -> float:
        return 3600.0                  # Pollen balance refills on a schedule

    def _request(self, prompt: str, seed: int) -> bytes:
        key = os.environ["POLLINATIONS_API_KEY"]
        url = f"https://gen.pollinations.ai/image/{quote(prompt, safe='')}"
        params = {"model": "zimage", "width": self.width, "height": self.height,
                  "seed": seed, "nologo": "true"}
        resp = requests.get(url, params=params, headers={"Authorization": f"Bearer {key}"},
                            timeout=self.timeout)
        if resp.status_code != 200:
            self._raise_for(resp)
        if not resp.headers.get("Content-Type", "").lower().startswith("image/"):
            raise ProviderError(f"Expected an image, got {resp.headers.get('Content-Type')}: "
                                f"{resp.text[:200]!r}")
        return resp.content


# --------------------------------------------------------------------------- ImageGen
class ImageGen:
    """Generates 9:16 images with automatic provider fallback.

        gen = ImageGen(output_dir="work/images")
        paths = gen.generate_batch([item.image_prompt for item in script.items], basename="fact")
    """

    PROVIDER_CLASSES = (CloudflareFlux2Klein, PollinationsZImage)   # priority order

    def __init__(self, output_dir: str | Path = "assets/images", width: int = 768, height: int = 1344,
                 final_width: int = 1080, final_height: int = 1920, max_workers: int = 3,
                 request_timeout: float = 180.0, style_suffix: str = DEFAULT_STYLE_SUFFIX,
                 reuse_existing: bool = True, base_seed: Optional[int] = None):
        self.output_dir = Path(output_dir)
        self.width, self.height = width, height
        self.final_width, self.final_height = final_width, final_height
        self.max_workers = max(1, max_workers)
        self.style_suffix = style_suffix
        self.reuse_existing = reuse_existing
        self.base_seed = base_seed if base_seed is not None else random.randint(1, 10_000_000)
        self._stats_lock = threading.Lock()
        self._stats: Dict[str, Dict[str, float]] = {}

        self.providers: List[BaseProvider] = []
        for cls in self.PROVIDER_CLASSES:
            if cls.configured():
                self.providers.append(cls(width, height, request_timeout))
            else:
                log.warning("Provider %s disabled - missing env var(s): %s", cls.name,
                            ", ".join(cls.missing_env()))
        if not self.providers:
            raise ImageGenError("No image provider configured. Set CLOUDFLARE_ACCOUNT_ID + "
                                "CLOUDFLARE_API_TOKEN and/or POLLINATIONS_API_KEY in .env")
        log.info("ImageGen ready. Providers (priority order): %s | request %dx%d -> final %dx%d | seed %d",
                 ", ".join(p.name for p in self.providers), width, height, final_width, final_height,
                 self.base_seed)

    # ---- public API
    def generate(self, prompt: str, index: int = 1, basename: str = "scene",
                 output_dir: Optional[str | Path] = None) -> Path:
        """Generate (or reuse) one image and return its path."""
        prepared = self._prepare_prompt(prompt)
        out_dir = Path(output_dir) if output_dir else self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{basename}_{index:02d}_{self._prompt_hash(prepared)}.png"

        if self.reuse_existing and path.exists() and path.stat().st_size > 10_000:
            log.info("[%s %02d] reusing existing image %s", basename, index, path.name)
            return path

        seed = self.base_seed + index
        errors: List[str] = []
        for provider in self.providers:
            if not provider.available():
                errors.append(f"{provider.name}: skipped, {provider.blocked_reason()}")
                continue
            start = time.perf_counter()
            try:
                raw = provider.generate(prepared, seed)
                final = self._validate_and_normalize(raw)
            except ProviderError as exc:
                self._record(provider.name, ok=False, seconds=time.perf_counter() - start)
                provider.record_failure()
                errors.append(f"{provider.name}: {exc}")
                if exc.fatal:
                    provider.block(24 * 3600, f"credentials rejected ({exc.status})")
                elif exc.quota_exhausted:
                    provider.block(provider.quota_block_seconds(), "quota exhausted")
                log.warning("[%s %02d] %s failed (%s) -> trying next provider", basename, index,
                            provider.name, exc)
                continue
            self._save_atomic(final, path)
            elapsed = time.perf_counter() - start
            provider.record_success()
            self._record(provider.name, ok=True, seconds=elapsed)
            log.info("[%s %02d] OK via %s in %.1fs -> %s", basename, index, provider.name, elapsed, path.name)
            return path

        raise ImageGenError(f"All providers failed for {basename} {index:02d}: " + " | ".join(errors))

    def generate_batch(self, prompts: Sequence[str], basename: str = "scene",
                       output_dir: Optional[str | Path] = None) -> List[Path]:
        """Generate one image per prompt concurrently; returns paths in prompt order.

        Every image is attempted even if some fail (so finished ones are cached for the next run);
        if any failed, ImageGenError is raised at the end listing all failures.
        """
        if not prompts:
            return []
        results: Dict[int, Path] = {}
        failures: Dict[int, str] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="imggen") as pool:
            futures = {pool.submit(self.generate, p, i, basename, output_dir): i
                       for i, p in enumerate(prompts, start=1)}
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results[idx] = fut.result()
                except Exception as exc:       # noqa: BLE001 - collect everything, raise once below
                    failures[idx] = str(exc)
                    log.error("[%s %02d] FAILED: %s", basename, idx, exc)
        if failures:
            detail = "; ".join(f"#{i}: {m}" for i, m in sorted(failures.items()))
            raise ImageGenError(f"{len(failures)}/{len(prompts)} images failed. {detail}")
        log.info("Batch complete: %d images. %s", len(prompts), self.summary_text())
        return [results[i] for i in range(1, len(prompts) + 1)]

    def summary(self) -> Dict[str, Dict[str, float]]:
        with self._stats_lock:
            return {k: dict(v) for k, v in self._stats.items()}

    def summary_text(self) -> str:
        parts = []
        for name, s in self.summary().items():
            avg = s["seconds"] / max(1, s["ok"] + s["fail"])
            parts.append(f"{name}: {int(s['ok'])} ok / {int(s['fail'])} failed (avg {avg:.1f}s)")
        return " | ".join(parts) or "no requests made"

    # ---- internals
    def _prepare_prompt(self, prompt: str) -> str:
        text = re.sub(r"\s+", " ", (prompt or "").strip())
        if not text:
            raise ImageGenError("Empty image prompt")
        if "9:16" not in text and self.style_suffix:
            text = text.rstrip(" .,") + self.style_suffix
        return text

    def _prompt_hash(self, prepared: str) -> str:
        key = f"{prepared}|{self.final_width}x{self.final_height}"
        return hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]

    def _record(self, name: str, ok: bool, seconds: float) -> None:
        with self._stats_lock:
            s = self._stats.setdefault(name, {"ok": 0, "fail": 0, "seconds": 0.0})
            s["ok" if ok else "fail"] += 1
            s["seconds"] += seconds

    def _validate_and_normalize(self, raw: bytes) -> Image.Image:
        try:
            img = Image.open(io.BytesIO(raw))
            img.load()
        except Exception as exc:               # noqa: BLE001
            raise ProviderError(f"returned data is not a valid image: {exc}")
        if min(img.size) < 256:
            raise ProviderError(f"image too small: {img.size}")
        img = img.convert("RGB")
        if max(ImageStat.Stat(img.convert("L")).stddev) < 3.0:
            raise ProviderError("image is blank/uniform")
        return self._to_916(img)

    def _to_916(self, img: Image.Image) -> Image.Image:
        """Center-crop to exact 9:16 (only if needed), then resize to the final size."""
        w, h = img.size
        target = self.final_width / self.final_height
        ratio = w / h
        if abs(ratio - target) / target > 0.005:
            if ratio > target:                  # too wide -> crop sides
                new_w = round(h * target)
                left = (w - new_w) // 2
                img = img.crop((left, 0, left + new_w, h))
            else:                               # too tall -> crop top/bottom
                new_h = round(w / target)
                top = (h - new_h) // 2
                img = img.crop((0, top, w, top + new_h))
        if img.size != (self.final_width, self.final_height):
            img = img.resize((self.final_width, self.final_height), RESAMPLE)
        return img

    @staticmethod
    def _save_atomic(img: Image.Image, path: Path) -> None:
        tmp = path.with_name(path.stem + ".tmp.png")
        img.save(tmp, format="PNG")
        tmp.replace(path)


# --------------------------------------------------------------------------- smoke test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                        datefmt="%H:%M:%S")
    test_prompts = [
        "A cinematic 35mm macro photograph of an ancient Roman golden coin lying on worn marble, "
        "side-lighting casting dramatic shadows",
        "A single ripe red apple on a plain wooden table, soft natural window light",
        "A dimly lit ancient library interior, exactly three white marble busts on stone pedestals, "
        "an open leather-bound book on a wooden desk, dust particles in a shaft of light",
    ]
    gen = ImageGen(output_dir="imagegen_smoke_test")
    for p in gen.generate_batch(test_prompts, basename="smoke"):
        print("saved:", p)
    print(gen.summary_text())
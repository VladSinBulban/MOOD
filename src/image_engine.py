"""Prompt-composable local photo editing for MOOD.

This module uses curated JSON recipes and classic image processing. It is fully local
and free, but it is not a generative model: it cannot invent realistic new objects or
replace scene content the way an image-editing AI can.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

ROOT_DIR = Path(__file__).resolve().parent.parent
SCENARIOS_PATH = ROOT_DIR / "scenarios.json"

BASE_SETTINGS = {
    "exposure": 0.0, "contrast": 1.0, "saturation": 1.0,
    "temperature": 0.0, "tint": 0.0, "vignette": 0.04,
    "grain": 0.0, "fade": 0.0,
}
BASE_EFFECTS = {"fog": 0.0, "glow": 0.0, "dust": 0.0, "rain": 0.0, "snow": 0.0}


def load_scenario_data() -> dict[str, Any]:
    """Load and validate the JSON recipe library."""
    if not SCENARIOS_PATH.exists():
        raise FileNotFoundError(f"MOOD scenario library is missing: {SCENARIOS_PATH}")
    with SCENARIOS_PATH.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data.get("scenarios"), list) or not data["scenarios"]:
        raise ValueError("scenarios.json must contain a non-empty 'scenarios' list.")
    return data


def _normalise(text: str) -> str:
    text = text.casefold().replace("’", "'")
    text = re.sub(r"[^\w\s'-]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _contains_phrase(text: str, phrase: str) -> bool:
    phrase = _normalise(phrase)
    if not phrase:
        return False
    return re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text) is not None


def _scenario_score(text: str, scenario: dict[str, Any]) -> tuple[float, str | None]:
    best = 0.0
    best_alias = None
    for alias in scenario.get("aliases", []):
        normal_alias = _normalise(str(alias))
        if normal_alias and _contains_phrase(text, normal_alias):
            # Prefer more informative multiword phrases over short, generic words.
            score = 1.0 + 0.24 * len(normal_alias.split()) + min(len(normal_alias), 32) / 100
            if score > best:
                best, best_alias = score, normal_alias
    return best, best_alias


def _blend_settings(base: dict[str, float], chosen: list[tuple[float, dict[str, Any]]]) -> dict[str, float]:
    """Blend selected recipes by family so secondary descriptors complement the main look."""
    result = dict(base)
    if not chosen:
        return result
    # Main recipe establishes the overall grade. Secondary recipes have a restrained influence.
    chosen = sorted(chosen, key=lambda item: item[0], reverse=True)
    primary = chosen[0][1].get("settings", {})
    result.update(primary)
    for score, scenario in chosen[1:]:
        weight = min(0.34, 0.16 + (score - 1.0) * 0.06)
        for key, value in scenario.get("settings", {}).items():
            if key not in result:
                result[key] = value
                continue
            current = float(result[key])
            target = float(value)
            # Saturation/exposure are scale/offset values; interpolate them gently.
            result[key] = current * (1.0 - weight) + target * weight
    return result


def interpret_prompt(prompt: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
    """Interpret a prompt into a grade, layered effects, and a transparent explanation."""
    if not prompt or not prompt.strip():
        raise ValueError("Enter a creative direction.")
    data = data or load_scenario_data()
    text = _normalise(prompt)
    scored = []
    for scenario in data["scenarios"]:
        score, alias = _scenario_score(text, scenario)
        if score:
            scored.append((score, scenario, alias))

    # Keep only the strongest matching recipe per family. Variants get a boost for exact phrases.
    best_by_family: dict[str, tuple[float, dict[str, Any], str]] = {}
    for score, scenario, alias in scored:
        family = scenario.get("family", "other")
        current = best_by_family.get(family)
        if current is None or score > current[0]:
            best_by_family[family] = (score, scenario, alias or scenario["id"])
    selected = sorted(best_by_family.values(), key=lambda item: item[0], reverse=True)
    if not selected:
        # Unknown prompt: don't falsely claim the engine understood it.
        return {
            "settings": dict(data.get("settings_defaults", BASE_SETTINGS)),
            "effects": dict(data.get("effect_defaults", BASE_EFFECTS)),
            "matched": [],
            "message": "No specific scenario matched; applying a gentle neutral enhancement.",
            "recognized": False,
        }

    settings = _blend_settings(data.get("settings_defaults", BASE_SETTINGS), [(s, p) for s, p, _ in selected])
    effects = dict(data.get("effect_defaults", BASE_EFFECTS))
    # Effects are cumulative, but cap them so multiple words don't overwhelm the image.
    for _, scenario, _ in selected:
        for key, value in scenario.get("effects", {}).items():
            effects[key] = float(effects.get(key, 0.0)) + float(value) * (0.78 if key in effects and effects[key] else 1.0)

    matched = []
    for score, scenario, alias in selected:
        matched.append({"id": scenario["id"], "name": scenario["name"], "alias": alias, "score": round(score, 2)})

    for modifier in data.get("modifiers", []):
        if any(_contains_phrase(text, _normalise(alias)) for alias in modifier.get("aliases", [])):
            for key, delta in modifier.get("settings", {}).items():
                if key in ("contrast", "saturation"):
                    settings[key] = float(settings.get(key, 1.0)) + float(delta)
                else:
                    settings[key] = float(settings.get(key, 0.0)) + float(delta)
            for key, delta in modifier.get("effects", {}).items():
                effects[key] = float(effects.get(key, 0.0)) + float(delta)

    # Simple direct modifiers make free-form instructions more flexible than a fixed preset.
    if any(_contains_phrase(text, p) for p in ("slightly", "subtle", "a little", "gentle")):
        settings["vignette"] = float(settings.get("vignette", 0.0)) * 0.78
        effects["fog"] = float(effects.get("fog", 0.0)) * 0.78
    if any(_contains_phrase(text, p) for p in ("very", "intense", "extreme", "dramatic")):
        settings["contrast"] = 1.0 + (float(settings.get("contrast", 1.0)) - 1.0) * 1.15
        settings["vignette"] = float(settings.get("vignette", 0.0)) * 1.10
    if "not too dark" in text or "keep it bright" in text:
        settings["exposure"] = max(float(settings.get("exposure", 0.0)), 0.05)
    if "preserve the colors" in text or "keep natural colors" in text:
        settings["saturation"] = 1.0 + (float(settings.get("saturation", 1.0)) - 1.0) * 0.45

    # Clamp combined recipe values before image processing.
    limits = {"exposure":(-1.25, 1.0), "contrast":(0.55, 1.85), "saturation":(0.0, 1.8),
              "temperature":(-0.8, 0.8), "tint":(-0.5, 0.5), "vignette":(0.0, 0.82),
              "grain":(0.0, 0.16), "fade":(0.0, 0.35)}
    for key, (low, high) in limits.items():
        settings[key] = float(np.clip(float(settings.get(key, BASE_SETTINGS.get(key, 0.0)),), low, high))
    for key in BASE_EFFECTS:
        effects[key] = float(np.clip(float(effects.get(key, 0.0)), 0.0, 0.52))

    names = [item["name"] for item in matched]
    return {"settings": settings, "effects": effects, "matched": matched,
            "message": "Combined local recipes: " + ", ".join(names[:4]), "recognized": True}


def adjust_image(image: Image.Image, **settings: float) -> Image.Image:
    """Apply color grade and cinematic finishing effects."""
    settings = dict(settings)
    exposure = float(np.clip(settings.get("exposure", 0.0), -2, 2))
    contrast = float(np.clip(settings.get("contrast", 1.0), 0, 2))
    saturation = float(np.clip(settings.get("saturation", 1.0), 0, 2))
    temperature = float(np.clip(settings.get("temperature", 0.0), -1, 1))
    tint = float(np.clip(settings.get("tint", 0.0), -1, 1))
    vignette = float(np.clip(settings.get("vignette", 0.0), 0, 1))
    grain = float(np.clip(settings.get("grain", 0.0), 0, 0.25))
    fade = float(np.clip(settings.get("fade", 0.0), 0, 0.6))
    result = ImageOps.exif_transpose(image).convert("RGB")
    if exposure:
        result = ImageEnhance.Brightness(result).enhance(2 ** exposure)
    if contrast != 1:
        result = ImageEnhance.Contrast(result).enhance(contrast)
    if saturation != 1:
        result = ImageEnhance.Color(result).enhance(saturation)
    pixels = np.asarray(result, dtype=np.float32).copy()
    pixels[:, :, 0] += temperature * 30 + tint * 12
    pixels[:, :, 1] -= tint * 20
    pixels[:, :, 2] -= temperature * 30 - tint * 12
    height, width = pixels.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    distance = np.sqrt(((xx - (width - 1) / 2) / max(width / 2, 1)) ** 2 + ((yy - (height - 1) / 2) / max(height / 2, 1)) ** 2)
    distance = np.clip(distance / np.sqrt(2), 0, 1)
    if vignette:
        pixels *= (1 - vignette * distance ** 1.7)[:, :, None]
    if fade:
        pixels = pixels * (1 - fade) + 128 * fade
    if grain:
        rng = np.random.default_rng(2026)
        pixels += rng.normal(0, grain * 255, (height, width, 1))
    return Image.fromarray(np.uint8(np.clip(pixels, 0, 255)))


def _blend_overlay(image: Image.Image, color: tuple[int, int, int], alpha: np.ndarray) -> Image.Image:
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    overlay = np.empty_like(pixels)
    overlay[:] = color
    alpha = np.clip(alpha, 0, 0.8).astype(np.float32)
    if alpha.ndim == 2:
        alpha = alpha[:, :, None]
    result = pixels * (1.0 - alpha) + overlay * alpha
    return Image.fromarray(np.uint8(np.clip(result, 0, 255)))


def apply_fog(image: Image.Image, strength: float) -> Image.Image:
    if strength <= 0.005:
        return image
    width, height = image.size
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, width, dtype=np.float32)[None, :]
    broad = (0.18 + 0.82 * (1 - y) ** 1.45) * (0.88 + 0.12 * np.cos(x * np.pi * 2.0))
    # Multiple low-frequency bands prevent the haze from looking like a flat overlay.
    bands = 0.08 * np.sin(x * np.pi * 3.0 + y * 2.0) + 0.05 * np.sin(x * np.pi * 7.0 - y * 4.0)
    alpha = np.clip(strength * (broad + bands), 0, 0.58)
    return _blend_overlay(image, (203, 213, 222), alpha)


def apply_glow(image: Image.Image, strength: float) -> Image.Image:
    if strength <= 0.005:
        return image
    radius = max(2.0, min(image.size) * 0.018)
    blurred = image.filter(ImageFilter.GaussianBlur(radius=radius))
    # Bright areas contribute more to bloom than deep shadows.
    bright = ImageEnhance.Brightness(blurred).enhance(1.12)
    return Image.blend(image.convert("RGB"), ImageChops_screen(image, bright), min(0.42, strength * 0.7))


def ImageChops_screen(base: Image.Image, blend: Image.Image) -> Image.Image:
    from PIL import ImageChops
    return ImageChops.screen(base.convert("RGB"), blend.convert("RGB"))


def apply_particles(image: Image.Image, amount: float, kind: str) -> Image.Image:
    if amount <= 0.005:
        return image
    width, height = image.size
    rng = np.random.default_rng(1147 if kind == "snow" else 7331 if kind == "rain" else 2609)
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    count = max(1, int(width * height / 6500 * amount * 5))
    if kind == "rain":
        for _ in range(min(count, 1800)):
            x = int(rng.integers(0, width)); y = int(rng.integers(0, height))
            length = int(rng.integers(5, 18))
            alpha = int(rng.integers(22, 75) * min(amount * 2.0, 1.0))
            draw.line((x, y, x - 3, y + length), fill=(195, 210, 225, alpha), width=1)
    else:
        for _ in range(min(count, 2200)):
            x = int(rng.integers(0, width)); y = int(rng.integers(0, height))
            radius = int(rng.integers(1, 4)) if kind == "snow" else int(rng.integers(1, 3))
            alpha = int(rng.integers(18, 100) * min(amount * 2.0, 1.0))
            color = (240, 245, 250, alpha) if kind == "snow" else (226, 201, 163, alpha)
            draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color)
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def apply_effects(image: Image.Image, effects: dict[str, float]) -> Image.Image:
    result = image
    result = apply_fog(result, effects.get("fog", 0.0))
    result = apply_glow(result, effects.get("glow", 0.0))
    result = apply_particles(result, effects.get("dust", 0.0), "dust")
    result = apply_particles(result, effects.get("rain", 0.0), "rain")
    result = apply_particles(result, effects.get("snow", 0.0), "snow")
    return result


def local_edit(image: Image.Image, prompt: str) -> tuple[Image.Image, str]:
    """Compatibility entry point used by app.py's local fallback."""
    if image is None:
        raise ValueError("Upload a photograph first.")
    interpreted = interpret_prompt(prompt)
    original = ImageOps.exif_transpose(image).convert("RGB")
    result = adjust_image(original, **interpreted["settings"])
    result = apply_effects(result, interpreted["effects"])
    if not interpreted["recognized"]:
        result = ImageEnhance.Contrast(result).enhance(1.04)
        return result, "Neutral local enhancement (prompt not recognized)"
    names = [item["name"] for item in interpreted["matched"]]
    return result, " + ".join(names[:3]) + " (composed local recipes)"


def scenario_count() -> int:
    return len(load_scenario_data()["scenarios"])

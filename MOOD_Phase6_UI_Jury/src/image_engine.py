
"""MOOD Engine 2.0 — prompt-composable local photo editing.

Runs locally with Pillow and NumPy. No paid APIs or network access.
This engine applies image-processing looks; it does not generate new scene content.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import (
    Image,
    ImageChops,
    ImageDraw,
    ImageEnhance,
    ImageFilter,
    ImageOps,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
SCENARIOS_PATH = ROOT_DIR / "scenarios.json"

BASE_SETTINGS = {
    "exposure": 0.0,
    "contrast": 1.0,
    "saturation": 1.0,
    "temperature": 0.0,
    "tint": 0.0,
    "vignette": 0.04,
    "grain": 0.0,
    "fade": 0.0,
}

BASE_EFFECTS = {
    "fog": 0.0,
    "glow": 0.0,
    "dust": 0.0,
    "rain": 0.0,
    "snow": 0.0,
}

# Direct semantic looks complement recipes from scenarios.json.
# Each look has a recognizable visual treatment of its own.
LOOK_ALIASES = {
    "night": (
        "night", "at night", "nighttime", "midnight",
        "dark night", "moonlit", "moonlight",
    ),
    "cyberpunk": (
        "cyberpunk", "cyber punk", "neon city",
        "futuristic neon", "neon lights", "sci fi city",
    ),
    "sunset": (
        "sunset", "golden hour", "warm sunset",
        "golden light", "sunrise", "sunset glow",
    ),
    "thriller": (
        "psychological thriller", "thriller",
        "ominous", "sinister", "eerie", "creepy",
        "unsettling", "dark atmosphere",
    ),
    "vintage": (
        "vintage", "old film", "retro", "analog film",
        "aged photograph", "1970s", "1980s film",
    ),
    "noir": (
        "film noir", "black and white", "black-and-white",
        "monochrome", "noir", "grayscale", "greyscale",
    ),
    "dreamy": (
        "dreamy", "dreamlike", "dream like",
        "ethereal", "soft dream", "fairytale",
        "fairy tale",
    ),
    "underwater": (
        "underwater", "under water", "deep ocean",
        "under the sea", "ocean depths",
    ),
    "winter": (
        "snowy", "snowfall", "winter", "snowstorm",
        "icy landscape", "frozen world",
    ),
    "rain": (
        "rain", "rainy", "raining", "heavy rain",
        "rainstorm", "wet streets", "rain soaked",
    ),
    "storm": (
        "storm", "thunderstorm", "stormy",
        "dark storm clouds", "approaching storm",
    ),
    "fire": (
        "firelight", "fire lit", "fiery", "inferno",
        "burning orange", "lit by fire",
    ),
    "desert": (
        "desert", "dust storm", "dusty atmosphere",
        "sandy haze", "arid landscape",
    ),
    "horror": (
        "horror", "horror movie", "horror film",
        "nightmare", "haunted", "ghostly",
    ),
    "bleach_bypass": (
        "bleach bypass", "bleach bypass film",
        "gritty cinematic", "gritty movie",
    ),
    "romantic": (
        "romantic", "soft romantic", "romantic film",
        "warm and intimate",
    ),
    "autumn": (
        "autumn", "fall colors", "fall colours",
        "autumn leaves", "autumnal",
    ),
    "toxic": (
        "toxic", "radioactive", "acid green",
        "toxic green", "mutant atmosphere",
    ),
    "vaporwave": (
        "vaporwave", "vapor wave", "80s neon",
        "retro neon", "synthwave", "synth wave",
    ),
    "cold": (
        "cold", "icy blue", "frozen", "cold blue",
        "winter chill",
    ),
}

LOOK_DESCRIPTIONS = {
    "night": "Cinematic night grade",
    "cyberpunk": "Cyberpunk neon grade",
    "sunset": "Golden-hour grade",
    "thriller": "Psychological thriller grade",
    "vintage": "Vintage film grade",
    "noir": "Black-and-white noir grade",
    "dreamy": "Dreamlike soft-focus grade",
    "underwater": "Underwater teal grade",
    "winter": "Winter atmosphere",
    "rain": "Rainy cinematic grade",
    "storm": "Storm atmosphere",
    "fire": "Firelight grade",
    "desert": "Desert haze grade",
    "horror": "Horror film grade",
    "bleach_bypass": "Bleach-bypass film grade",
    "romantic": "Warm romantic grade",
    "autumn": "Autumn color grade",
    "toxic": "Toxic green grade",
    "vaporwave": "Vaporwave neon grade",
    "cold": "Cold blue grade",
}


def load_scenario_data() -> dict[str, Any]:
    """Load and validate the scenario recipe library."""
    if not SCENARIOS_PATH.exists():
        raise FileNotFoundError(
            f"MOOD scenario library is missing: {SCENARIOS_PATH}"
        )

    with SCENARIOS_PATH.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)

    if not isinstance(data.get("scenarios"), list) or not data["scenarios"]:
        raise ValueError(
            "scenarios.json must contain a non-empty 'scenarios' list."
        )

    return data


def _normalise(text: str) -> str:
    text = text.casefold().replace("â€™", "'")
    text = re.sub(r"[^\w\s'-]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _contains_phrase(text: str, phrase: str) -> bool:
    phrase = _normalise(phrase)
    if not phrase:
        return False

    return (
        re.search(
            r"(?<!\w)" + re.escape(phrase) + r"(?!\w)",
            text,
        )
        is not None
    )


def _scenario_score(
    text: str, scenario: dict[str, Any]
) -> tuple[float, str | None]:
    best = 0.0
    best_alias = None

    for alias in scenario.get("aliases", []):
        normal_alias = _normalise(str(alias))

        if normal_alias and _contains_phrase(text, normal_alias):
            score = (
                1.0
                + 0.24 * len(normal_alias.split())
                + min(len(normal_alias), 32) / 100
            )
            if score > best:
                best = score
                best_alias = normal_alias

    return best, best_alias


def _blend_settings(
    base: dict[str, float],
    chosen: list[tuple[float, dict[str, Any]]],
) -> dict[str, float]:
    result = dict(base)
    if not chosen:
        return result

    chosen = sorted(chosen, key=lambda item: item[0], reverse=True)
    result.update(chosen[0][1].get("settings", {}))

    for score, scenario in chosen[1:]:
        weight = min(0.34, 0.16 + (score - 1.0) * 0.06)

        for key, value in scenario.get("settings", {}).items():
            if key not in result:
                result[key] = value
                continue

            current = float(result[key])
            target = float(value)
            result[key] = current * (1.0 - weight) + target * weight

    return result


def _detect_looks(text: str) -> list[str]:
    """Return distinct looks detected in the prompt."""
    found = []

    for look, aliases in LOOK_ALIASES.items():
        if any(_contains_phrase(text, alias) for alias in aliases):
            found.append(look)

    # Explicit black-and-white requests take priority over color grades.
    if "noir" in found:
        found = [
            look for look in found
            if look not in {"cyberpunk", "sunset", "fire", "vaporwave"}
        ]

    return found


def interpret_prompt(
    prompt: str,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Combine JSON recipes, direct looks, and simple prompt modifiers."""
    if not prompt or not prompt.strip():
        raise ValueError("Enter a creative direction.")

    data = data or load_scenario_data()
    text = _normalise(prompt)

    scored = []
    for scenario in data["scenarios"]:
        score, alias = _scenario_score(text, scenario)
        if score:
            scored.append((score, scenario, alias))

    best_by_family: dict[
        str, tuple[float, dict[str, Any], str]
    ] = {}

    for score, scenario, alias in scored:
        family = scenario.get("family", "other")
        current = best_by_family.get(family)

        if current is None or score > current[0]:
            best_by_family[family] = (
                score,
                scenario,
                alias or scenario["id"],
            )

    selected = sorted(
        best_by_family.values(),
        key=lambda item: item[0],
        reverse=True,
    )
    looks = _detect_looks(text)

    # Direct looks can recognize a prompt even when no JSON recipe matches.
    if not selected and not looks:
        return {
            "settings": dict(data.get("settings_defaults", BASE_SETTINGS)),
            "effects": dict(data.get("effect_defaults", BASE_EFFECTS)),
            "matched": [],
            "looks": [],
            "message": (
                "No specific scenario matched; applying a gentle "
                "neutral enhancement."
            ),
            "recognized": False,
        }

    settings = _blend_settings(
        data.get("settings_defaults", BASE_SETTINGS),
        [(score, scenario) for score, scenario, _ in selected],
    )
    effects = dict(data.get("effect_defaults", BASE_EFFECTS))

    for _, scenario, _ in selected:
        for key, value in scenario.get("effects", {}).items():
            existing = float(effects.get(key, 0.0))
            multiplier = 0.78 if existing else 1.0
            effects[key] = existing + float(value) * multiplier

    matched = []
    for score, scenario, alias in selected:
        matched.append(
            {
                "id": scenario["id"],
                "name": scenario["name"],
                "alias": alias,
                "score": round(score, 2),
            }
        )

    for modifier in data.get("modifiers", []):
        if any(
            _contains_phrase(text, _normalise(alias))
            for alias in modifier.get("aliases", [])
        ):
            for key, delta in modifier.get("settings", {}).items():
                if key in ("contrast", "saturation"):
                    settings[key] = (
                        float(settings.get(key, 1.0)) + float(delta)
                    )
                else:
                    settings[key] = (
                        float(settings.get(key, 0.0)) + float(delta)
                    )

            for key, delta in modifier.get("effects", {}).items():
                effects[key] = float(effects.get(key, 0.0)) + float(delta)

    if any(
        _contains_phrase(text, phrase)
        for phrase in ("slightly", "subtle", "a little", "gentle")
    ):
        settings["vignette"] = float(settings.get("vignette", 0.0)) * 0.78
        for key in effects:
            effects[key] *= 0.78

    if any(
        _contains_phrase(text, phrase)
        for phrase in ("very", "intense", "extreme", "dramatic", "strong")
    ):
        settings["contrast"] = 1.0 + (
            float(settings.get("contrast", 1.0)) - 1.0
        ) * 1.15
        settings["vignette"] = float(settings.get("vignette", 0.0)) * 1.10
        for key in effects:
            effects[key] *= 1.18

    if "not too dark" in text or "keep it bright" in text:
        settings["exposure"] = max(
            float(settings.get("exposure", 0.0)), 0.05
        )

    if "preserve the colors" in text or "keep natural colors" in text:
        settings["saturation"] = 1.0 + (
            float(settings.get("saturation", 1.0)) - 1.0
        ) * 0.45

    limits = {
        "exposure": (-1.25, 1.0),
        "contrast": (0.55, 1.85),
        "saturation": (0.0, 1.8),
        "temperature": (-0.8, 0.8),
        "tint": (-0.5, 0.5),
        "vignette": (0.0, 0.82),
        "grain": (0.0, 0.16),
        "fade": (0.0, 0.35),
    }

    for key, (low, high) in limits.items():
        settings[key] = float(
            np.clip(
                float(settings.get(key, BASE_SETTINGS.get(key, 0.0))),
                low,
                high,
            )
        )

    for key in BASE_EFFECTS:
        effects[key] = float(
            np.clip(float(effects.get(key, 0.0)), 0.0, 0.52)
        )

    names = [item["name"] for item in matched]
    descriptions = [LOOK_DESCRIPTIONS[look] for look in looks]
    explanations = names[:4] + descriptions[:4]

    return {
        "settings": settings,
        "effects": effects,
        "matched": matched,
        "looks": looks,
        "message": "Combined looks: " + ", ".join(explanations),
        "recognized": True,
    }


def adjust_image(image: Image.Image, **settings: float) -> Image.Image:
    """Apply exposure, contrast, color balance, vignette, and grain."""
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

    distance = np.sqrt(
        ((xx - (width - 1) / 2) / max(width / 2, 1)) ** 2
        + ((yy - (height - 1) / 2) / max(height / 2, 1)) ** 2
    )
    distance = np.clip(distance / np.sqrt(2), 0, 1)

    if vignette:
        pixels *= (1 - vignette * distance ** 1.7)[:, :, None]

    if fade:
        pixels = pixels * (1 - fade) + 128 * fade

    if grain:
        rng = np.random.default_rng(2026)
        noise = rng.normal(0, grain * 255, (height, width, 1))
        pixels += noise

    return Image.fromarray(np.uint8(np.clip(pixels, 0, 255)))


def _blend_overlay(
    image: Image.Image,
    color: tuple[int, int, int],
    alpha: np.ndarray,
) -> Image.Image:
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    overlay = np.empty_like(pixels)
    overlay[:] = color

    alpha = np.clip(alpha, 0, 0.8).astype(np.float32)
    if alpha.ndim == 2:
        alpha = alpha[:, :, None]

    result = pixels * (1.0 - alpha) + overlay * alpha
    return Image.fromarray(np.uint8(np.clip(result, 0, 255)))


def _color_grade(
    image: Image.Image,
    shadow_color: tuple[int, int, int],
    highlight_color: tuple[int, int, int],
    strength: float,
) -> Image.Image:
    """Tint shadows and highlights differently while retaining image detail."""
    if strength <= 0:
        return image.convert("RGB")

    pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    luminance = (
        0.2126 * pixels[:, :, 0]
        + 0.7152 * pixels[:, :, 1]
        + 0.0722 * pixels[:, :, 2]
    ) / 255.0

    shadows = np.clip(1.0 - luminance * 1.65, 0, 1)[:, :, None]
    highlights = np.clip((luminance - 0.28) / 0.72, 0, 1)[:, :, None]

    shadow = np.asarray(shadow_color, dtype=np.float32)
    highlight = np.asarray(highlight_color, dtype=np.float32)

    pixels += (shadow - 128) * shadows * strength
    pixels += (highlight - 128) * highlights * strength * 0.65

    return Image.fromarray(np.uint8(np.clip(pixels, 0, 255)))


def _apply_night(image: Image.Image) -> Image.Image:
    result = ImageEnhance.Brightness(image).enhance(0.52)
    result = ImageEnhance.Contrast(result).enhance(1.25)
    result = ImageEnhance.Color(result).enhance(0.76)
    result = _color_grade(result, (45, 72, 135), (125, 160, 210), 0.65)
    return adjust_image(result, vignette=0.38)


def _apply_cyberpunk(image: Image.Image) -> Image.Image:
    result = ImageEnhance.Contrast(image).enhance(1.24)
    result = ImageEnhance.Color(result).enhance(1.55)
    result = _color_grade(result, (10, 120, 180), (245, 40, 155), 0.90)

    pixels = np.asarray(result, dtype=np.float32)
    brightness = pixels.max(axis=2)
    mask = np.clip((brightness - 145) / 90, 0, 1)[:, :, None]
    neon = pixels.copy()
    neon[:, :, 0] = np.minimum(255, neon[:, :, 0] * 1.15 + 15)
    neon[:, :, 2] = np.minimum(255, neon[:, :, 2] * 1.18 + 12)

    result = Image.fromarray(np.uint8(np.clip(pixels * (1 - mask * 0.30) + neon * mask * 0.30, 0, 255)))
    blurred = result.filter(ImageFilter.GaussianBlur(max(2, min(result.size) * 0.012)))
    bloom = ImageChops.screen(result, blurred)
    return Image.blend(result, bloom, 0.24)


def _apply_sunset(image: Image.Image) -> Image.Image:
    result = ImageEnhance.Color(image).enhance(1.20)
    result = ImageEnhance.Contrast(result).enhance(1.10)

    height, width = result.height, result.width
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    gradient = np.broadcast_to(
        np.clip((1.0 - y) ** 1.6, 0, 1), (height, width)
    )
    return _blend_overlay(result, (255, 126, 48), gradient * 0.34)


def _apply_thriller(image: Image.Image) -> Image.Image:
    result = ImageEnhance.Color(image).enhance(0.68)
    result = ImageEnhance.Contrast(result).enhance(1.32)
    result = _color_grade(result, (47, 75, 76), (172, 156, 125), 0.44)
    return adjust_image(result, vignette=0.43, grain=0.025)


def _apply_vintage(image: Image.Image) -> Image.Image:
    result = ImageOps.grayscale(image).convert("RGB")
    result = ImageOps.colorize(
        ImageOps.grayscale(result),
        black=(38, 28, 22),
        white=(244, 218, 172),
    ).convert("RGB")
    result = ImageEnhance.Contrast(result).enhance(0.92)
    result = adjust_image(result, fade=0.07, vignette=0.24, grain=0.035)
    return result


def _apply_noir(image: Image.Image) -> Image.Image:
    result = ImageOps.grayscale(image).convert("RGB")
    result = ImageEnhance.Contrast(result).enhance(1.65)
    result = ImageEnhance.Brightness(result).enhance(0.94)
    return adjust_image(result, vignette=0.38, grain=0.012)


def _apply_dreamy(image: Image.Image) -> Image.Image:
    result = ImageEnhance.Color(image).enhance(1.12)
    result = ImageEnhance.Contrast(result).enhance(0.88)

    blur = result.filter(
        ImageFilter.GaussianBlur(max(2, min(result.size) * 0.012))
    )
    result = Image.blend(result, blur, 0.20)

    bright_blur = result.filter(
        ImageFilter.GaussianBlur(max(4, min(result.size) * 0.025))
    )
    result = ImageChops.screen(result, bright_blur)
    return Image.blend(image.convert("RGB"), result, 0.48)


def _apply_underwater(image: Image.Image) -> Image.Image:
    result = ImageEnhance.Color(image).enhance(0.78)
    result = ImageEnhance.Contrast(result).enhance(1.08)
    result = _color_grade(result, (15, 95, 130), (80, 175, 190), 0.88)
    return _blend_overlay(result, (0, 92, 122), np.full(result.size[::-1], 0.10, dtype=np.float32))


def _apply_winter(image: Image.Image) -> Image.Image:
    result = ImageEnhance.Color(image).enhance(0.68)
    result = _color_grade(result, (80, 120, 170), (180, 205, 230), 0.60)
    return result


def _apply_storm(image: Image.Image) -> Image.Image:
    result = ImageEnhance.Brightness(image).enhance(0.70)
    result = ImageEnhance.Contrast(result).enhance(1.30)
    result = ImageEnhance.Color(result).enhance(0.72)
    result = _color_grade(result, (66, 83, 103), (150, 166, 178), 0.55)
    return result


def _apply_fire(image: Image.Image) -> Image.Image:
    result = ImageEnhance.Color(image).enhance(1.20)
    return _color_grade(result, (145, 65, 22), (255, 190, 80), 0.75)


def _apply_desert(image: Image.Image) -> Image.Image:
    result = ImageEnhance.Color(image).enhance(0.82)
    result = _blend_overlay(
        result, (198, 151, 92),
        np.full((result.height, result.width), 0.25, dtype=np.float32),
    )
    return result


def _apply_bleach_bypass(image: Image.Image) -> Image.Image:
    result = ImageEnhance.Color(image).enhance(0.50)
    result = ImageEnhance.Contrast(result).enhance(1.55)
    return adjust_image(result, grain=0.012, vignette=0.20)


def _apply_romantic(image: Image.Image) -> Image.Image:
    result = ImageEnhance.Color(image).enhance(1.12)
    result = _color_grade(result, (95, 70, 80), (255, 208, 175), 0.45)
    return _apply_dreamy(result)


def _apply_autumn(image: Image.Image) -> Image.Image:
    result = ImageEnhance.Color(image).enhance(1.22)
    return _color_grade(result, (125, 80, 35), (238, 166, 66), 0.60)


def _apply_toxic(image: Image.Image) -> Image.Image:
    result = ImageEnhance.Color(image).enhance(1.35)
    return _color_grade(result, (50, 150, 38), (165, 220, 65), 0.82)


def _apply_vaporwave(image: Image.Image) -> Image.Image:
    result = ImageEnhance.Color(image).enhance(1.35)
    result = _color_grade(result, (55, 100, 190), (230, 95, 190), 0.75)
    return result


def _apply_cold(image: Image.Image) -> Image.Image:
    result = ImageEnhance.Color(image).enhance(0.80)
    return _color_grade(result, (55, 105, 180), (145, 190, 235), 0.65)


LOOK_FUNCTIONS = {
    "night": _apply_night,
    "cyberpunk": _apply_cyberpunk,
    "sunset": _apply_sunset,
    "thriller": _apply_thriller,
    "vintage": _apply_vintage,
    "noir": _apply_noir,
    "dreamy": _apply_dreamy,
    "underwater": _apply_underwater,
    "winter": _apply_winter,
    "rain": _apply_storm,
    "storm": _apply_storm,
    "fire": _apply_fire,
    "desert": _apply_desert,
    "horror": _apply_thriller,
    "bleach_bypass": _apply_bleach_bypass,
    "romantic": _apply_romantic,
    "autumn": _apply_autumn,
    "toxic": _apply_toxic,
    "vaporwave": _apply_vaporwave,
    "cold": _apply_cold,
}


def apply_fog(image: Image.Image, strength: float) -> Image.Image:
    if strength <= 0.005:
        return image

    width, height = image.size
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, width, dtype=np.float32)[None, :]

    broad = (0.18 + 0.82 * (1 - y) ** 1.45) * (
        0.88 + 0.12 * np.cos(x * np.pi * 2.0)
    )
    bands = (
        0.08 * np.sin(x * np.pi * 3.0 + y * 2.0)
        + 0.05 * np.sin(x * np.pi * 7.0 - y * 4.0)
    )

    alpha = np.clip(strength * (broad + bands), 0, 0.58)
    return _blend_overlay(image, (203, 213, 222), alpha)


def apply_glow(image: Image.Image, strength: float) -> Image.Image:
    if strength <= 0.005:
        return image

    radius = max(2.0, min(image.size) * 0.018)
    blurred = image.filter(ImageFilter.GaussianBlur(radius=radius))

    # Glow responds to brighter pixels, not just a flat white overlay.
    bright = ImageEnhance.Brightness(blurred).enhance(1.30)
    screen = ImageChops.screen(image.convert("RGB"), bright)
    return Image.blend(image.convert("RGB"), screen, min(0.58, strength * 0.9))


def apply_particles(
    image: Image.Image, amount: float, kind: str
) -> Image.Image:
    if amount <= 0.005:
        return image

    width, height = image.size
    seeds = {"snow": 1147, "rain": 7331, "dust": 2609}
    rng = np.random.default_rng(seeds.get(kind, 2609))

    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    scale = width * height / 6500
    count = max(1, int(scale * amount * 7))

    if kind == "rain":
        for _ in range(min(count, 3500)):
            x = int(rng.integers(0, width))
            y = int(rng.integers(0, height))
            length = int(rng.integers(7, 24))
            alpha = int(rng.integers(35, 115) * min(amount * 2.4, 1.0))
            draw.line(
                (x, y, x - 4, y + length),
                fill=(190, 210, 230, alpha),
                width=1,
            )
    else:
        for _ in range(min(count, 3000)):
            x = int(rng.integers(0, width))
            y = int(rng.integers(0, height))

            if kind == "snow":
                radius = int(rng.integers(1, 4))
                color = (240, 245, 255)
                alpha = int(
                    rng.integers(50, 150) * min(amount * 2.4, 1.0)
                )
            else:
                radius = int(rng.integers(1, 3))
                color = (226, 201, 163)
                alpha = int(
                    rng.integers(25, 100) * min(amount * 2.4, 1.0)
                )

            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=(*color, alpha),
            )

    return Image.alpha_composite(
        image.convert("RGBA"), overlay
    ).convert("RGB")


def apply_effects(
    image: Image.Image, effects: dict[str, float]
) -> Image.Image:
    result = image.convert("RGB")

    result = apply_fog(result, effects.get("fog", 0.0))
    result = apply_glow(result, effects.get("glow", 0.0))
    result = apply_particles(result, effects.get("dust", 0.0), "dust")
    result = apply_particles(result, effects.get("rain", 0.0), "rain")
    result = apply_particles(result, effects.get("snow", 0.0), "snow")

    return result


def _apply_looks(
    image: Image.Image, looks: list[str]
) -> Image.Image:
    """Apply the primary look, then complementary looks."""
    result = image.convert("RGB")

    if not looks:
        return result

    # Apply the main look first. Extra looks are blended so combinations
    # remain useful instead of repeatedly crushing contrast and exposure.
    primary = LOOK_FUNCTIONS[looks[0]](result)

    if len(looks) == 1:
        return primary

    result = Image.blend(result, primary, 0.88)

    for look in looks[1:]:
        secondary = LOOK_FUNCTIONS[look](result)
        result = Image.blend(result, secondary, 0.40)

    return result


def local_edit(
    image: Image.Image, prompt: str
) -> tuple[Image.Image, str]:
    """Compatibility entry point used by app.py's local fallback."""
    if image is None:
        raise ValueError("Upload a photograph first.")

    interpreted = interpret_prompt(prompt)
    original = ImageOps.exif_transpose(image).convert("RGB")

    if not interpreted["recognized"]:
        result = adjust_image(original, **interpreted["settings"])
        result = ImageEnhance.Contrast(result).enhance(1.04)
        return result, "Neutral local enhancement (prompt not recognized)"

    result = adjust_image(original, **interpreted["settings"])
    result = _apply_looks(result, interpreted.get("looks", []))
    result = apply_effects(result, interpreted["effects"])

    names = [item["name"] for item in interpreted["matched"]]
    names.extend(
        LOOK_DESCRIPTIONS[look]
        for look in interpreted.get("looks", [])
    )

    return result, " + ".join(names[:4]) + " (composed local recipes — MOOD Engine 2.0)"


def scenario_count() -> int:
    """Return the number of recipes in scenarios.json."""
    return len(load_scenario_data()["scenarios"])

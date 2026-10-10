"""MOOD's cinematic image-processing engine."""

from PIL import Image, ImageEnhance
import numpy as np


PRESETS = {
    "natural": {
        "exposure": 0.0, "contrast": 1.0, "saturation": 1.0,
        "temperature": 0.0, "tint": 0.0,
        "vignette": 0.0, "grain": 0.0, "fade": 0.0,
    },
    "batman": {
        "exposure": -0.55, "contrast": 1.65, "saturation": 0.48,
        "temperature": -0.42, "tint": 0.08,
        "vignette": 0.72, "grain": 0.07, "fade": 0.04,
    },
    "noir": {
        "exposure": -0.35, "contrast": 1.8, "saturation": 0.0,
        "temperature": 0.0, "tint": 0.0,
        "vignette": 0.8, "grain": 0.12, "fade": 0.0,
    },
    "horror": {
        "exposure": -0.65, "contrast": 1.55, "saturation": 0.55,
        "temperature": -0.25, "tint": -0.12,
        "vignette": 0.82, "grain": 0.13, "fade": 0.02,
    },
    "cyberpunk": {
        "exposure": -0.15, "contrast": 1.55, "saturation": 1.65,
        "temperature": -0.2, "tint": 0.35,
        "vignette": 0.48, "grain": 0.04, "fade": 0.0,
    },
    "nostalgic": {
        "exposure": 0.12, "contrast": 0.88, "saturation": 0.78,
        "temperature": 0.38, "tint": 0.06,
        "vignette": 0.3, "grain": 0.1, "fade": 0.22,
    },
    "dreamy": {
        "exposure": 0.28, "contrast": 0.78, "saturation": 0.85,
        "temperature": 0.18, "tint": 0.12,
        "vignette": 0.12, "grain": 0.01, "fade": 0.18,
    },
    "warm": {
        "exposure": 0.22, "contrast": 1.12, "saturation": 1.3,
        "temperature": 0.48, "tint": 0.02,
        "vignette": 0.12, "grain": 0.0, "fade": 0.0,
    },
    "cold": {
        "exposure": -0.12, "contrast": 1.22, "saturation": 0.82,
        "temperature": -0.48, "tint": 0.02,
        "vignette": 0.2, "grain": 0.02, "fade": 0.0,
    },
    "moody": {
        "exposure": -0.42, "contrast": 1.48, "saturation": 0.68,
        "temperature": -0.12, "tint": 0.02,
        "vignette": 0.55, "grain": 0.06, "fade": 0.04,
    },
}


def adjust_image(
    image: Image.Image,
    exposure=0.0,
    contrast=1.0,
    saturation=1.0,
    temperature=0.0,
    tint=0.0,
    vignette=0.0,
    grain=0.0,
    fade=0.0,
) -> Image.Image:
    """Apply adjustable colour grading and cinematic effects."""
    exposure = float(np.clip(exposure, -2, 2))
    contrast = float(np.clip(contrast, 0, 2))
    saturation = float(np.clip(saturation, 0, 2))
    temperature = float(np.clip(temperature, -1, 1))
    tint = float(np.clip(tint, -1, 1))
    vignette = float(np.clip(vignette, 0, 1))
    grain = float(np.clip(grain, 0, 0.25))
    fade = float(np.clip(fade, 0, 0.6))

    result = image.convert("RGB")

    if exposure:
        result = ImageEnhance.Brightness(result).enhance(2 ** exposure)
    if contrast != 1:
        result = ImageEnhance.Contrast(result).enhance(contrast)
    if saturation != 1:
        result = ImageEnhance.Color(result).enhance(saturation)

    pixels = np.asarray(result, dtype=np.float32).copy()

    # Colour temperature and tint.
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

    # Darken image corners for a cinematic vignette.
    if vignette:
        factor = 1 - vignette * (distance ** 1.7)
        pixels *= factor[:, :, None]

    # Lift dark tones for a faded film look.
    if fade:
        pixels = pixels * (1 - fade) + 128 * fade

    # Add subtle, reproducible film grain.
    if grain:
        rng = np.random.default_rng(42)
        noise = rng.normal(0, grain * 255, (height, width, 1))
        pixels += noise

    pixels = np.clip(pixels, 0, 255).astype(np.uint8)
    return Image.fromarray(pixels)


def apply_preset(image: Image.Image, preset_name: str) -> Image.Image:
    """Apply a named cinematic look."""
    name = preset_name.strip().lower()
    if name not in PRESETS:
        raise ValueError(
            f"Unknown preset: {preset_name!r}. "
            f"Choose from: {', '.join(PRESETS)}"
        )
    return adjust_image(image, **PRESETS[name])

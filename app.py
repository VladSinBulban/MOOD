
import os
import io
import json
import base64
import urllib.request
import urllib.error

import gradio as gr
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from dotenv import load_dotenv


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
MODEL = "gemini-nano-banana-2.1"
API_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
PORT = 7862


# ============================================================
# IMAGE HELPERS
# ============================================================

def prepare_image(image):
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((1536, 1536))

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=88)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def rgb_adjust(image, red=1.0, green=1.0, blue=1.0):
    """Adjust individual RGB channels."""
    arr = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    arr[:, :, 0] *= red
    arr[:, :, 1] *= green
    arr[:, :, 2] *= blue
    return Image.fromarray(np.uint8(np.clip(arr, 0, 255)))


def vignette(image, strength=0.3):
    """Darken image edges."""
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]

    yy, xx = np.mgrid[0:h, 0:w]
    xx = (xx - w / 2) / max(w / 2, 1)
    yy = (yy - h / 2) / max(h / 2, 1)

    radius = np.sqrt(xx * xx + yy * yy)
    mask = np.clip(1 - strength * radius ** 1.7, 0.45, 1.0)
    arr *= mask[..., None]

    return Image.fromarray(np.uint8(np.clip(arr, 0, 255)))


def blend_color(image, color, strength):
    """Blend an image with a chosen color."""
    overlay = Image.new("RGB", image.size, color)
    return Image.blend(image.convert("RGB"), overlay, strength)


# ============================================================
# LOCAL EDITING ENGINE
# ============================================================

def local_edit(image, prompt):
    """
    Free, rule-based image effects.

    Important: these are conventional image filters, not generative AI.
    They cannot add realistic new objects, scenery, or lighting sources.
    """

    image = ImageOps.exif_transpose(image).convert("RGB")
    text = prompt.lower().strip()

    if not text:
        raise ValueError("Enter a creative direction.")

    # --------------------------------------------------------
    # BLACK AND WHITE
    # --------------------------------------------------------
    if any(k in text for k in (
        "black and white", "black-and-white", "monochrome",
        "grayscale", "greyscale"
    )):
        result = ImageOps.grayscale(image).convert("RGB")
        return ImageEnhance.Contrast(result).enhance(1.18), "Black and white"

    # --------------------------------------------------------
    # FOG AND MIST
    # --------------------------------------------------------
    if any(k in text for k in (
        "fog", "foggy", "mist", "misty", "haze", "hazy"
    )):
        arr = np.asarray(image, dtype=np.float32).copy()
        h, w = arr.shape[:2]

        # Mist increases toward the upper part of the image.
        y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
        x = np.linspace(0, 1, w, dtype=np.float32)[None, :]

        strength = 0.08 + 0.38 * (1 - y) ** 1.5
        strength *= 0.92 + 0.08 * np.cos(x * np.pi * 2)
        strength = np.clip(strength, 0.0, 0.58)
        strength = np.broadcast_to(strength[..., None], (h, w, 1))

        fog_color = np.array([205, 216, 224], dtype=np.float32)
        arr = arr * (1 - strength) + fog_color * strength

        result = Image.fromarray(np.uint8(np.clip(arr, 0, 255)))
        result = ImageEnhance.Contrast(result).enhance(0.84)
        return result, "Fog / mist approximation"

    # --------------------------------------------------------
    # RAIN / STORM
    # --------------------------------------------------------
    if any(k in text for k in (
        "rain", "rainy", "storm", "stormy", "overcast"
    )):
        result = ImageEnhance.Color(image).enhance(0.68)
        result = ImageEnhance.Contrast(result).enhance(0.92)
        result = rgb_adjust(result, 0.94, 0.98, 1.06)
        result = blend_color(result, (105, 125, 145), 0.10)
        return result, "Rainy / overcast color treatment"

    # --------------------------------------------------------
    # SNOW / WINTER
    # --------------------------------------------------------
    if any(k in text for k in (
        "snow", "snowy", "winter", "icy", "ice cold"
    )):
        result = ImageEnhance.Color(image).enhance(0.72)
        result = rgb_adjust(result, 0.91, 1.00, 1.12)
        result = ImageEnhance.Brightness(result).enhance(1.07)
        result = blend_color(result, (205, 225, 245), 0.08)
        return result, "Cold winter color treatment"

    # --------------------------------------------------------
    # NIGHT / MOONLIGHT
    # --------------------------------------------------------
    if any(k in text for k in (
        "night", "nighttime", "moonlight", "midnight"
    )):
        result = ImageEnhance.Brightness(image).enhance(0.55)
        result = ImageEnhance.Color(result).enhance(0.72)
        result = ImageEnhance.Contrast(result).enhance(1.16)
        result = rgb_adjust(result, 0.78, 0.88, 1.14)
        return vignette(result, 0.34), "Night color treatment"

    # --------------------------------------------------------
    # SUNSET / GOLDEN HOUR
    # --------------------------------------------------------
    if any(k in text for k in (
        "sunset", "golden hour", "sunrise", "warm light"
    )):
        result = rgb_adjust(image, 1.13, 1.04, 0.86)
        result = ImageEnhance.Color(result).enhance(1.12)
        result = ImageEnhance.Contrast(result).enhance(1.05)
        return result, "Golden-hour color treatment"

    # --------------------------------------------------------
    # LOVE ISLAND / TROPICAL / SUMMER
    # --------------------------------------------------------
    if any(k in text for k in (
        "love island", "tropical", "summer", "beach",
        "holiday", "resort", "paradise"
    )):
        result = ImageEnhance.Color(image).enhance(1.28)
        result = ImageEnhance.Contrast(result).enhance(1.10)
        result = rgb_adjust(result, 1.025, 1.025, 1.00)
        return result, "Tropical / reality-TV inspired grade"

    # --------------------------------------------------------
    # CYBERPUNK / NEON
    # --------------------------------------------------------
    if any(k in text for k in (
        "cyberpunk", "neon", "synthwave", "futuristic"
    )):
        result = ImageEnhance.Contrast(image).enhance(1.22)
        result = ImageEnhance.Color(result).enhance(1.30)
        result = rgb_adjust(result, 1.08, 0.93, 1.15)
        result = blend_color(result, (45, 20, 95), 0.10)
        return result, "Neon-inspired color treatment"

    # --------------------------------------------------------
    # HORROR / DARK / EERIE
    # --------------------------------------------------------
    if any(k in text for k in (
        "horror", "scary", "eerie", "creepy", "ominous",
        "dark cinematic", "dark and moody"
    )):
        result = ImageEnhance.Brightness(image).enhance(0.72)
        result = ImageEnhance.Contrast(result).enhance(1.28)
        result = ImageEnhance.Color(result).enhance(0.68)
        result = rgb_adjust(result, 0.94, 1.00, 1.06)
        return vignette(result, 0.38), "Dark cinematic treatment"

    # --------------------------------------------------------
    # VINTAGE / ANALOG FILM
    # --------------------------------------------------------
    if any(k in text for k in (
        "vintage", "retro", "analog", "film look", "old photo"
    )):
        result = ImageEnhance.Color(image).enhance(0.70)
        result = ImageEnhance.Contrast(result).enhance(0.86)
        result = rgb_adjust(result, 1.08, 1.01, 0.91)
        result = blend_color(result, (224, 190, 135), 0.07)
        return result, "Vintage film treatment"

    # --------------------------------------------------------
    # DREAMY / SOFT FOCUS
    # --------------------------------------------------------
    if any(k in text for k in (
        "dreamy", "dreamlike", "soft focus", "ethereal"
    )):
        blurred = image.filter(ImageFilter.GaussianBlur(radius=2.5))
        result = Image.blend(image, blurred, 0.35)
        result = ImageEnhance.Brightness(result).enhance(1.03)
        return result, "Dreamy soft-focus treatment"

    # --------------------------------------------------------
    # UNDERWATER / OCEAN
    # --------------------------------------------------------
    if any(k in text for k in (
        "underwater", "under water", "ocean", "deep sea"
    )):
        result = rgb_adjust(image, 0.78, 0.97, 1.16)
        result = ImageEnhance.Color(result).enhance(0.82)
        result = blend_color(result, (0, 85, 125), 0.18)
        return result, "Underwater-inspired color treatment"

    # --------------------------------------------------------
    # BRIGHT / VIBRANT
    # --------------------------------------------------------
    if any(k in text for k in (
        "vibrant", "colorful", "colourful", "bright colors"
    )):
        result = ImageEnhance.Color(image).enhance(1.38)
        result = ImageEnhance.Contrast(result).enhance(1.10)
        result = ImageEnhance.Brightness(result).enhance(1.05)
        return result, "Vibrant color treatment"

    # --------------------------------------------------------
    # SOFT / MUTED
    # --------------------------------------------------------
    if any(k in text for k in (
        "muted", "desaturated", "subtle colors", "soft colors"
    )):
        result = ImageEnhance.Color(image).enhance(0.68)
        result = ImageEnhance.Contrast(result).enhance(0.92)
        return result, "Muted color treatment"

    # --------------------------------------------------------
    # SHARP / HIGH CONTRAST
    # --------------------------------------------------------
    if any(k in text for k in (
        "high contrast", "dramatic contrast", "sharp"
    )):
        result = ImageEnhance.Contrast(image).enhance(1.25)
        result = ImageEnhance.Sharpness(result).enhance(1.5)
        return result, "Contrast and sharpness treatment"

    # --------------------------------------------------------
    # NEUTRAL DEFAULT — NO AUTOMATIC WARMING
    # --------------------------------------------------------
    # Don't pretend to understand a prompt we cannot interpret.
    # Apply only a modest contrast adjustment, not a warm filter.
    result = ImageEnhance.Contrast(image).enhance(1.06)
    return result, "Generic local treatment (prompt not recognized)"


# ============================================================
# GEMINI IMAGE EDITING
# ============================================================

def build_prompt(user_prompt):
    return f"""
You are MOOD, a professional AI photo editor.

User request:
{user_prompt}

Edit the provided photograph to follow the request.
Interpret the requested atmosphere, lighting, environment and style.
For fog, create believable depth and variable mist rather than a
uniform grey overlay. For a scene transformation, change the scene
when necessary instead of only adjusting warmth or saturation.
Preserve important subjects and composition unless asked otherwise.
Return an edited image, not a textual explanation.
""".strip()


def extract_image(response):
    candidates = []

    def collect(obj):
        if isinstance(obj, dict):
            for key in ("output_image", "image"):
                value = obj.get(key)
                if isinstance(value, dict):
                    data = value.get("data")
                    if isinstance(data, str):
                        candidates.append(data)

            data = obj.get("data")
            if isinstance(data, str):
                candidates.append(data)

            for key in ("content", "steps", "output", "model_output"):
                value = obj.get(key)
                if isinstance(value, (dict, list)):
                    collect(value)

        elif isinstance(obj, list):
            for item in obj:
                collect(item)

    collect(response)

    for data in candidates:
        try:
            if data.startswith("data:") and "," in data:
                data = data.split(",", 1)[1]

            raw = base64.b64decode(data, validate=True)
            with Image.open(io.BytesIO(raw)) as result:
                result.load()
                return result.convert("RGB")
        except Exception:
            pass

    raise RuntimeError(
        "Gemini returned no recognizable image. "
        "The API response format may have changed."
    )


def gemini_edit(image, prompt):
    if not API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    payload = {
        "model": MODEL,
        "input": [
            {"type": "text", "text": build_prompt(prompt)},
            {
                "type": "image",
                "mime_type": "image/jpeg",
                "data": prepare_image(image),
            },
        ],
    }

    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": API_KEY,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.loads(response.read().decode("utf-8"))

    except urllib.error.HTTPError as error:
        try:
            body = error.read().decode("utf-8")
            parsed = json.loads(body)
            message = parsed.get("error", {}).get(
                "message", body[:500]
            )
        except Exception:
            message = "Image API request failed."

        raise RuntimeError(
            f"Gemini HTTP {error.code}: {message}"
        ) from error

    except urllib.error.URLError as error:
        raise RuntimeError(
            f"Network error: {error.reason}"
        ) from error

    except TimeoutError as error:
        raise RuntimeError("Gemini request timed out.") from error

    return extract_image(result)


# ============================================================
# MAIN EDIT HANDLER
# ============================================================

def edit_photo(image, prompt, progress=gr.Progress()):
    if image is None:
        raise gr.Error("Upload a photograph first.")

    if not prompt or not prompt.strip():
        raise gr.Error("Describe the effect you want.")

    prompt = prompt.strip()
    progress(0.1, desc="Preparing photograph...")

    if API_KEY:
        try:
            progress(0.25, desc="Requesting AI image edit...")
            result = gemini_edit(image, prompt)
            progress(1.0, desc="AI edit complete.")

            return result, (
                "AI EDIT COMPLETE\n"
                "The result was returned by Gemini."
            )

        except Exception as error:
            print(f"[MOOD] Gemini error: {error}")

            progress(0.75, desc="AI unavailable; applying local effect...")

            try:
                result, effect_name = local_edit(image, prompt)
                return result, (
                    f"LOCAL FALLBACK: {effect_name}\n"
                    "This is a conventional image effect, not generative AI.\n\n"
                    f"Gemini error: {str(error)[:350]}"
                )
            except Exception as local_error:
                print(f"[MOOD] Local editing error: {local_error}")
                raise gr.Error(
                    f"Both editing methods failed: {local_error}"
                ) from local_error

    result, effect_name = local_edit(image, prompt)

    return result, (
        f"LOCAL DEMO: {effect_name}\n"
        "Gemini is not configured. This is not a generative AI edit."
    )


# ============================================================
# DESIGN
# ============================================================

CSS = """
.gradio-container {
    max-width: 1250px !important;
    margin: auto !important;
}

body, .gradio-container {
    background: #101014 !important;
    color: #f4f2f8 !important;
}

#hero {
    padding: 28px 4px 20px 4px;
}

#brand {
    color: #b7a1ff;
    font-size: 13px;
    font-weight: 800;
    letter-spacing: 5px;
}

#headline {
    font-size: clamp(36px, 5vw, 56px);
    line-height: 1.08;
    font-weight: 800;
    letter-spacing: -2px;
    margin: 12px 0;
}

#subhead {
    color: #a5a2b1;
    font-size: 16px;
    line-height: 1.6;
}

.panel {
    background: #19191f;
    border: 1px solid #2d2b36;
    border-radius: 18px;
    padding: 18px;
}

.panel-title {
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2px;
    color: #c0b0ff;
    margin-bottom: 10px;
}

.primary-btn {
    background: #a78bfa !important;
    color: #171321 !important;
    border: 0 !important;
    font-weight: 800 !important;
    min-height: 48px !important;
}

footer {
    display: none !important;
}

@media (max-width: 700px) {
    #headline {
        letter-spacing: -1px;
    }
}
"""


# ============================================================
# GRADIO APP
# ============================================================

with gr.Blocks(
    title="MOOD — AI Creative Studio",
    theme=gr.themes.Base(
        primary_hue="violet",
        secondary_hue="slate",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ),
    css=CSS,
) as demo:

    gr.HTML("""
        <div id="hero">
            <div id="brand">MOOD / CREATIVE STUDIO</div>
            <div id="headline">Make it feel different.</div>
            <div id="subhead">
                One photograph. Your imagination.
                Describe the atmosphere, style, or feeling you want.
            </div>
        </div>
    """)

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=280):
            with gr.Group(elem_classes=["panel"]):
                gr.HTML(
                    '<div class="panel-title">01 / YOUR PHOTOGRAPH</div>'
                )
                input_image = gr.Image(
                    label="Original image",
                    type="pil",
                    height=330,
                )

            with gr.Group(elem_classes=["panel"]):
                gr.HTML(
                    '<div class="panel-title">02 / CREATIVE DIRECTION</div>'
                )
                prompt = gr.Textbox(
                    label="Describe your vision",
                    placeholder=(
                        "Try: foggy morning, rainy street, "
                        "cyberpunk night, vintage film..."
                    ),
                    lines=4,
                    max_lines=8,
                )

                edit_button = gr.Button(
                    "Transform photograph →",
                    variant="primary",
                    elem_classes=["primary-btn"],
                )

        with gr.Column(scale=1, min_width=280):
            with gr.Group(elem_classes=["panel"]):
                gr.HTML(
                    '<div class="panel-title">03 / YOUR RESULT</div>'
                )
                output_image = gr.Image(
                    label="Edited photograph",
                    type="pil",
                    height=430,
                    interactive=False,
                )
                status = gr.Textbox(
                    label="Editing status",
                    value="Ready when you are.",
                    interactive=False,
                    lines=5,
                )

    gr.HTML("""
        <div style="
            color:#777482;
            font-size:12px;
            text-align:center;
            padding:24px 0;
        ">
            MOOD CREATIVE STUDIO · AI EDITING + LOCAL FALLBACK
        </div>
    """)

    edit_button.click(
        fn=edit_photo,
        inputs=[input_image, prompt],
        outputs=[output_image, status],
    )

    prompt.submit(
        fn=edit_photo,
        inputs=[input_image, prompt],
        outputs=[output_image, status],
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    demo.queue(max_size=8)
    demo.launch(
        server_name="127.0.0.1",
        server_port=PORT,
        inbrowser=True,
        show_error=True,
    )

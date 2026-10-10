
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

from src.image_engine import local_edit

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


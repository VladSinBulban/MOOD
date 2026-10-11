
import os
import io
import json
import base64
import tempfile
import copy
from datetime import datetime
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
PORT = int(os.getenv("PORT", os.getenv("MOOD_PORT", "7862")))
MAX_UPLOAD_MB = max(5, min(100, int(os.getenv("MOOD_MAX_UPLOAD_MB", "25"))))
MAX_IMAGE_PIXELS = max(1_000_000, min(50_000_000, int(os.getenv("MOOD_MAX_IMAGE_PIXELS", "25000000"))))
APP_VERSION = "6.0.0"
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


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


def apply_intensity(original, edited, intensity):
    """Blend the edited image with the original to apply the slider reliably."""
    original = ImageOps.exif_transpose(original).convert("RGB")
    edited = edited.convert("RGB")
    if edited.size != original.size:
        edited = edited.resize(original.size, Image.Resampling.LANCZOS)
    strength = max(0.0, min(100.0, float(intensity))) / 100.0
    return Image.blend(original, edited, strength)


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
# PHASE 6: INPUT VALIDATION AND PRIVACY-SAFE OPERATIONS
# ============================================================

def validate_image(image, *, label="Image"):
    """Reject missing, unsupported, or excessively large image inputs."""
    if image is None:
        raise gr.Error(f"{label}: upload a photograph first.")
    if not isinstance(image, Image.Image):
        try:
            image = Image.fromarray(np.asarray(image))
        except Exception as exc:
            raise gr.Error(f"{label}: unsupported image input.") from exc
    width, height = image.size
    if width < 1 or height < 1:
        raise gr.Error(f"{label}: the image has invalid dimensions.")
    if width * height > MAX_IMAGE_PIXELS:
        raise gr.Error(f"{label}: {width}×{height} exceeds the configured {MAX_IMAGE_PIXELS:,}-pixel safety limit. Resize it or adjust MOOD_MAX_IMAGE_PIXELS.")
    return ImageOps.exif_transpose(image).convert("RGB")


def validate_prompt(prompt):
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Describe the effect you want.")
    if len(prompt) > 4000:
        raise gr.Error("Keep the creative direction under 4,000 characters.")
    return prompt


def release_readiness_report():
    """Give practical launch checks without displaying secrets."""
    return (f"MOOD release {APP_VERSION}\n"
            f"Default bind: 127.0.0.1 (this computer only)\n"
            f"Configured port: {PORT}\n"
            f"Image pixel safety limit: {MAX_IMAGE_PIXELS:,}\n"
            f"Batch maximum: 30 photos\n"
            f"Gemini: {'configured (secret hidden)' if API_KEY else 'not configured; local mode available'}\n\n"
            "Before distributing commercially: review every dependency license, the Gemini API terms and billing, privacy/data-retention policy, and any third-party model/brand rights. Do not expose this development server directly to the public internet; use a hardened deployment with authentication, TLS, request limits, and secret management.")


# ============================================================
# MAIN EDIT HANDLER
# ============================================================

def edit_photo(image, prompt, intensity=65, progress=gr.Progress()):
    image = validate_image(image, label="Source photo")
    prompt = validate_prompt(prompt)
    intensity = max(0, min(100, int(intensity)))
    intensity_hint = "subtle" if intensity <= 33 else ("intense" if intensity >= 70 else "balanced")
    effective_prompt = f"{prompt}, {intensity_hint} effect intensity ({intensity}%)"
    progress(0.1, desc="Preparing photograph...")

    if API_KEY:
        try:
            progress(0.25, desc="Requesting AI image edit...")
            result = gemini_edit(image, effective_prompt)
            result = apply_intensity(image, result, intensity)
            progress(1.0, desc="AI edit complete.")

            return result, (
                "AI EDIT COMPLETE\n"
                "The result was returned by Gemini."
            )

        except Exception as error:
            print(f"[MOOD] Gemini error: {error}")

            progress(0.75, desc="AI unavailable; applying local effect...")

            try:
                result, effect_name = local_edit(image, effective_prompt)
                result = apply_intensity(image, result, intensity)
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

    result, effect_name = local_edit(image, effective_prompt)
    result = apply_intensity(image, result, intensity)

    return result, (
        f"LOCAL DEMO: {effect_name}\n"
        "Gemini is not configured. This is not a generative AI edit."
    )


def sync_source_preview(image):
    """Refresh the original preview and clear stale edits when the source changes."""
    if image is None:
        return None, None, "Ready when you are. Upload a photograph and describe the mood."
    original = ImageOps.exif_transpose(image).convert("RGB")
    return original, None, "Source updated. Your original is preserved; create a new edit when ready."


def reset_preview(image):
    """Reset the result preview to the unedited source image."""
    if image is None:
        return None, "Upload a photograph before resetting the preview."
    original = ImageOps.exif_transpose(image).convert("RGB")
    return original, "Preview reset to the original photograph. Your source image is unchanged."


# ============================================================
# SCENARIO LIBRARY
# ============================================================

BUILTIN_SCENARIOS = [
    {"name": "Cinematic", "family": "mood", "aliases": ["cinematic lighting", "film still"]},
    {"name": "Vintage film", "family": "style", "aliases": ["vintage film", "analog grain"]},
    {"name": "Dreamy", "family": "mood", "aliases": ["dreamlike", "soft glow"]},
    {"name": "Natural color", "family": "color", "aliases": ["natural colors", "realistic skin tones"]},
    {"name": "Psychological thriller", "family": "mood", "aliases": ["psychological thriller", "eerie tension"]},
]


def load_scenarios():
    """Load the project's full scenarios.json, with a safe built-in fallback."""
    path = os.path.join(BASE_DIR, "scenarios.json")
    try:
        with open(path, "r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
        scenarios = payload.get("scenarios", []) if isinstance(payload, dict) else []
        valid = [item for item in scenarios if isinstance(item, dict) and item.get("name")]
        return valid or BUILTIN_SCENARIOS
    except (OSError, ValueError, TypeError):
        return BUILTIN_SCENARIOS


SCENARIOS = load_scenarios()


def get_scenario_category(item):
    family = str(item.get("family", "other")).lower()
    name = str(item.get("name", "")).lower()
    aliases = " ".join(str(x) for x in item.get("aliases", [])).lower()
    searchable = f"{name} {aliases}"
    if family.startswith("reference_"):
        return "Reference worlds"
    if any(word in searchable for word in ("portrait", "selfie", "skin tone", "people photography")):
        return "Portrait & people"
    if any(word in searchable for word in ("product", "packshot", "still life")):
        return "Product & still life"
    if any(word in searchable for word in ("landscape", "mountain", "forest", "coast", "beach", "nature", "desert", "underwater")):
        return "Landscape & nature"
    if family == "mood" or any(word in searchable for word in ("cinema", "cinematic", "thriller", "film noir")):
        return "Cinematic & mood"
    labels = {
        "style": "Styles & film", "color": "Color grades", "effect": "Effects",
        "time": "Time of day", "weather": "Weather & atmosphere", "subject": "Photography",
        "variant": "Creative looks",
    }
    return labels.get(family, "Other looks")


SCENARIO_CATEGORIES = ["All categories"] + sorted({get_scenario_category(item) for item in SCENARIOS})


def filter_scenarios(search_text, category):
    query = (search_text or "").strip().lower()
    category = category or "All categories"
    matches = []
    for item in SCENARIOS:
        if category != "All categories" and get_scenario_category(item) != category:
            continue
        haystack = " ".join([
            str(item.get("name", "")),
            str(item.get("family", "")),
            " ".join(str(x) for x in item.get("aliases", [])),
        ]).lower()
        if query and query not in haystack:
            continue
        matches.append(item.get("name", ""))
    return gr.update(choices=matches, value=(matches[0] if matches else None))


def use_scenario(name):
    if not name:
        return ""
    item = next((entry for entry in SCENARIOS if entry.get("name") == name), None)
    if not item:
        return name
    aliases = item.get("aliases", [])
    cues = ", ".join(str(value) for value in aliases[:3] if value)
    return f"{item['name']}. Visual direction: {cues}. Preserve the main subject and realistic details."



# ============================================================
# PHASE 3: PROFESSIONAL ADJUSTMENTS, MASKING, HISTORY, CUSTOM LOOKS
# ============================================================

CUSTOM_LOOKS_PATH = os.path.join(BASE_DIR, "mood_custom_looks.json")


def _control_settings(exposure, contrast, saturation, temperature, highlights, shadows, sharpness, denoise):
    return {
        "exposure": float(exposure), "contrast": float(contrast),
        "saturation": float(saturation), "temperature": float(temperature),
        "highlights": float(highlights), "shadows": float(shadows),
        "sharpness": float(sharpness), "denoise": float(denoise),
    }


def apply_professional_adjustments(image, exposure=0, contrast=1, saturation=1,
                                   temperature=0, highlights=0, shadows=0,
                                   sharpness=1, denoise=0, mask=None):
    """Apply non-destructive-style global or mask-weighted photo adjustments."""
    if image is None:
        raise ValueError("Create or upload an image before adjusting it.")
    source = ImageOps.exif_transpose(image).convert("RGB")
    result = source
    if denoise > 0:
        softened = result.filter(ImageFilter.MedianFilter(size=3))
        result = Image.blend(result, softened, max(0.0, min(1.0, float(denoise))))
    if abs(float(exposure)) > 1e-4:
        result = ImageEnhance.Brightness(result).enhance(2 ** max(-3.0, min(3.0, float(exposure))))
    if abs(float(contrast) - 1.0) > 1e-4:
        result = ImageEnhance.Contrast(result).enhance(max(0.0, min(3.0, float(contrast))))
    if abs(float(saturation) - 1.0) > 1e-4:
        result = ImageEnhance.Color(result).enhance(max(0.0, min(3.0, float(saturation))))
    if abs(float(temperature)) > 1e-4:
        t = max(-100.0, min(100.0, float(temperature))) / 100.0
        result = rgb_adjust(result, red=1.0 + 0.18 * t, green=1.0 + 0.015 * t, blue=1.0 - 0.18 * t)
    if abs(float(highlights)) > 1e-4 or abs(float(shadows)) > 1e-4:
        arr = np.asarray(result.convert("RGB"), dtype=np.float32)
        luminance = arr.mean(axis=2, keepdims=True) / 255.0
        shadow_weight = np.clip((0.68 - luminance) / 0.68, 0, 1) ** 1.35
        highlight_weight = np.clip((luminance - 0.34) / 0.66, 0, 1) ** 1.35
        arr += shadow_weight * float(shadows) * 0.85
        arr += highlight_weight * float(highlights) * 0.85
        result = Image.fromarray(np.uint8(np.clip(arr, 0, 255)))
    if abs(float(sharpness) - 1.0) > 1e-4:
        result = ImageEnhance.Sharpness(result).enhance(max(0.0, min(4.0, float(sharpness))))
    result = result.convert("RGB")
    if mask is not None:
        if isinstance(mask, dict):
            mask = mask.get("composite") or mask.get("background")
        if mask is not None:
            if not isinstance(mask, Image.Image):
                try:
                    mask = Image.fromarray(np.asarray(mask))
                except Exception:
                    mask = None
            if mask is not None:
                mask = ImageOps.exif_transpose(mask).convert("L").resize(source.size, Image.Resampling.LANCZOS)
                # White mask pixels receive the adjustment; black pixels remain untouched.
                result = Image.composite(result, source, mask)
    return result


def _history_push(history, image):
    history = list(history or [])
    if image is not None:
        history.append(ImageOps.exif_transpose(image).convert("RGB"))
        history = history[-20:]
    return history


def apply_controls(base_image, current_image, exposure, contrast, saturation, temperature, highlights, shadows, sharpness, denoise, mask, history):
    if current_image is None:
        raise gr.Error("Create an edit or upload a photo first.")
    history = _history_push(history, current_image)
    source = base_image if base_image is not None else current_image
    result = apply_professional_adjustments(source, exposure, contrast, saturation, temperature, highlights, shadows, sharpness, denoise, mask)
    return result, history, f"Adjustments applied · {result.width} × {result.height} px. Undo is available."


def undo_edit(image, history, redo_history):
    history, redo_history = list(history or []), list(redo_history or [])
    if image is None or not history:
        return image, history, redo_history, "Nothing to undo yet. Apply an adjustment or create an edit first."
    redo_history.append(ImageOps.exif_transpose(image).convert("RGB"))
    previous = history.pop()
    return previous, history, redo_history, "Undid the last image operation."


def redo_edit(image, history, redo_history):
    history, redo_history = list(history or []), list(redo_history or [])
    if not redo_history:
        return image, history, redo_history, "Nothing to redo."
    if image is not None:
        history.append(ImageOps.exif_transpose(image).convert("RGB"))
        history = history[-20:]
    next_image = redo_history.pop()
    return next_image, history, redo_history, "Redo complete."


def reset_adjustment_controls():
    return 0, 1.0, 1.0, 0, 0, 0, 1.0, 0.0, "Adjustment controls reset. The image itself has not changed."


def save_custom_look(name, exposure, contrast, saturation, temperature, highlights, shadows, sharpness, denoise):
    name = (name or "").strip()
    if not name:
        raise gr.Error("Enter a name for your custom look.")
    if len(name) > 48:
        raise gr.Error("Keep the custom look name under 49 characters.")
    settings = _control_settings(exposure, contrast, saturation, temperature, highlights, shadows, sharpness, denoise)
    try:
        with open(CUSTOM_LOOKS_PATH, "r", encoding="utf-8") as handle:
            looks = json.load(handle)
        if not isinstance(looks, dict):
            looks = {}
    except (OSError, ValueError):
        looks = {}
    looks[name] = settings
    with open(CUSTOM_LOOKS_PATH, "w", encoding="utf-8") as handle:
        json.dump(looks, handle, indent=2, ensure_ascii=False)
    return gr.update(choices=sorted(looks), value=name), f"Saved custom look: {name}"


def load_custom_looks():
    try:
        with open(CUSTOM_LOOKS_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return sorted(data) if isinstance(data, dict) else []
    except (OSError, ValueError):
        return []


def apply_saved_look(name):
    if not name:
        return 0, 1.0, 1.0, 0, 0, 0, 1.0, 0.0, "Choose a saved look first."
    try:
        with open(CUSTOM_LOOKS_PATH, "r", encoding="utf-8") as handle:
            look = json.load(handle).get(name)
    except (OSError, ValueError, AttributeError):
        look = None
    if not isinstance(look, dict):
        return 0, 1.0, 1.0, 0, 0, 0, 1.0, 0.0, "That saved look could not be loaded."
    return (look.get("exposure", 0), look.get("contrast", 1), look.get("saturation", 1),
            look.get("temperature", 0), look.get("highlights", 0), look.get("shadows", 0),
            look.get("sharpness", 1), look.get("denoise", 0), f"Loaded custom look: {name}. Press Apply adjustments to use it.")


# ============================================================
# EXPORT
# ============================================================

def export_image(image, image_format="PNG", quality=95):
    if image is None:
        raise gr.Error("Create an edit before exporting it.")
    fmt = (image_format or "PNG").upper()
    extension = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}.get(fmt)
    if extension is None:
        raise gr.Error("Choose PNG, JPEG, or WebP.")
    export_dir = os.path.join(tempfile.gettempdir(), "MOOD_exports")
    os.makedirs(export_dir, exist_ok=True)
    filename = f"MOOD_edit_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}{extension}"
    path = os.path.join(export_dir, filename)
    result = ImageOps.exif_transpose(image).convert("RGB")
    options = {}
    if fmt == "JPEG":
        options = {"quality": max(50, min(100, int(quality))), "optimize": True}
    elif fmt == "WEBP":
        options = {"quality": max(50, min(100, int(quality))), "method": 6}
    elif fmt == "PNG":
        result = ImageOps.exif_transpose(image).convert("RGBA") if image.mode == "RGBA" else result
        options = {"optimize": True}
    result.save(path, format=fmt, **options)
    return path, f"Export ready · {fmt} · {result.width} × {result.height} px"


def clear_session():
    return None, None, None, "", 65, [], [], [], None, "Session cleared. Upload a photograph to begin again."




# ============================================================
# PHASE 4: BATCH WORKFLOW, PORTABLE PROJECTS, AND DIAGNOSTICS
# ============================================================

def _open_uploaded_image(item):
    """Open an uploaded Gradio file or a path as a detached RGB PIL image."""
    path = item if isinstance(item, (str, os.PathLike)) else getattr(item, "name", None)
    if not path or not os.path.isfile(path):
        raise ValueError("One of the uploaded files has no readable path.")
    if os.path.getsize(path) > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"Image exceeds the {MAX_UPLOAD_MB} MB upload limit.")
    with Image.open(path) as im:
        im.load()
        return ImageOps.exif_transpose(im).convert("RGB")


def _edit_one_for_batch(image, prompt, intensity):
    """Edit one image for batch mode, with an honest local fallback."""
    note = ""
    if API_KEY:
        try:
            edited = gemini_edit(image, prompt)
            return apply_intensity(image, edited, intensity), "Gemini AI"
        except Exception as exc:
            note = f"Gemini unavailable ({str(exc)[:180]}); local fallback used."
    edited, effect = local_edit(image, prompt)
    return apply_intensity(image, edited, intensity), (f"Local: {effect}. {note}".strip())


def process_batch(files, prompt, intensity, progress=gr.Progress()):
    """Apply the same direction to multiple photographs and package results."""
    if not files:
        raise gr.Error("Choose at least one photo for batch processing.")
    prompt = validate_prompt(prompt)
    if len(files) > 30:
        raise gr.Error("For reliability, process up to 30 photos per batch.")
    workdir = tempfile.mkdtemp(prefix="mood_batch_")
    output_dir = os.path.join(workdir, "edited")
    os.makedirs(output_dir, exist_ok=True)
    records = []
    total = len(files)
    for index, item in enumerate(files, 1):
        source_name = os.path.basename(item if isinstance(item, (str, os.PathLike)) else getattr(item, "name", f"photo_{index}.jpg"))
        stem = os.path.splitext(source_name)[0] or f"photo_{index}"
        safe_stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stem)[:70] or f"photo_{index}"
        try:
            progress((index - 1) / total, desc=f"Editing {index}/{total}: {source_name[:36]}")
            file_path = item if isinstance(item, (str, os.PathLike)) else getattr(item, "name", None)
            if not file_path or not os.path.isfile(file_path):
                raise ValueError("Uploaded file is no longer available.")
            if os.path.getsize(file_path) > MAX_UPLOAD_MB * 1024 * 1024:
                raise ValueError(f"File exceeds the {MAX_UPLOAD_MB} MB upload limit.")
            original = validate_image(_open_uploaded_image(item), label=source_name)
            edited, method = _edit_one_for_batch(original, prompt, max(0, min(100, int(intensity))))
            output_name = f"{index:02d}_{safe_stem}.jpg"
            output_path = os.path.join(output_dir, output_name)
            edited.save(output_path, format="JPEG", quality=95, optimize=True)
            records.append({"source": source_name, "output": output_name, "status": "success", "method": method, "size": list(edited.size)})
        except Exception as exc:
            records.append({"source": source_name, "status": "failed", "error": str(exc)[:400]})
    manifest = {
        "application": "MOOD", "workflow": "batch", "created_at": datetime.now().isoformat(timespec="seconds"),
        "prompt": prompt.strip(), "intensity": int(intensity), "results": records,
        "note": "Local fallback uses conventional image processing and is not generative AI."
    }
    with open(os.path.join(workdir, "batch_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    archive = os.path.join(tempfile.gettempdir(), f"MOOD_batch_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.zip")
    import zipfile
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, names in os.walk(workdir):
            for name in names:
                full = os.path.join(root, name)
                zf.write(full, os.path.relpath(full, workdir))
    succeeded = sum(r["status"] == "success" for r in records)
    failed = total - succeeded
    progress(1.0, desc="Batch package ready")
    summary = f"Batch complete · {succeeded}/{total} succeeded · {failed} failed. Results and batch_manifest.json are in the ZIP."
    return archive, summary


def save_project_bundle(original, edited, prompt, intensity, exposure, contrast, saturation, temperature, highlights, shadows, sharpness, denoise, mask):
    """Create a portable project ZIP with images and an editable JSON manifest."""
    if original is None and edited is None:
        raise gr.Error("Upload or edit a photo before saving a project.")
    workdir = tempfile.mkdtemp(prefix="mood_project_")
    if original is not None:
        ImageOps.exif_transpose(original).convert("RGB").save(os.path.join(workdir, "original.png"), format="PNG", optimize=True)
    if edited is not None:
        ImageOps.exif_transpose(edited).convert("RGB").save(os.path.join(workdir, "edited.png"), format="PNG", optimize=True)
    if mask is not None:
        try:
            if isinstance(mask, dict):
                mask = mask.get("composite") or mask.get("background")
            if mask is not None:
                ImageOps.exif_transpose(mask).convert("L").save(os.path.join(workdir, "selective_mask.png"), format="PNG", optimize=True)
        except Exception:
            pass
    manifest = {
        "application": "MOOD", "project_version": 1, "created_at": datetime.now().isoformat(timespec="seconds"),
        "prompt": prompt or "", "intensity": int(intensity),
        "adjustments": _control_settings(exposure, contrast, saturation, temperature, highlights, shadows, sharpness, denoise),
        "files": {"original": "original.png" if original is not None else None, "edited": "edited.png" if edited is not None else None,
                  "mask": "selective_mask.png" if os.path.exists(os.path.join(workdir, "selective_mask.png")) else None},
        "notes": "The project contains the image snapshots and adjustment values. The API key is never included."
    }
    with open(os.path.join(workdir, "mood_project.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    import zipfile
    path = os.path.join(tempfile.gettempdir(), f"MOOD_project_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.zip")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in os.listdir(workdir):
            zf.write(os.path.join(workdir, name), name)
    return path, "Portable project saved. It includes image snapshots and settings, never your API key."


def load_project_bundle(upload):
    """Restore the images and controls from a MOOD project ZIP."""
    if not upload:
        raise gr.Error("Choose a MOOD project ZIP first.")
    path = upload if isinstance(upload, (str, os.PathLike)) else getattr(upload, "name", None)
    if not path:
        raise gr.Error("The selected project file cannot be read.")
    import zipfile
    try:
        with zipfile.ZipFile(path, "r") as zf:
            names = set(zf.namelist())
            if "mood_project.json" not in names:
                raise ValueError("This ZIP does not contain mood_project.json.")
            manifest = json.loads(zf.read("mood_project.json").decode("utf-8"))
            if manifest.get("application") != "MOOD" or manifest.get("project_version") != 1:
                raise ValueError("Unsupported MOOD project format/version.")
            def read_image(member):
                if not member or member not in names:
                    return None
                with zf.open(member) as fh:
                    with Image.open(fh) as im:
                        im.load()
                        return ImageOps.exif_transpose(im).convert("RGB")
            original = read_image(manifest.get("files", {}).get("original"))
            edited = read_image(manifest.get("files", {}).get("edited"))
            mask = read_image(manifest.get("files", {}).get("mask"))
    except (OSError, ValueError, KeyError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        raise gr.Error(f"Could not load project: {exc}") from exc
    settings = manifest.get("adjustments", {})
    values = [settings.get("exposure", 0), settings.get("contrast", 1), settings.get("saturation", 1),
              settings.get("temperature", 0), settings.get("highlights", 0), settings.get("shadows", 0),
              settings.get("sharpness", 1), settings.get("denoise", 0)]
    status = f"Project loaded · source {'restored' if original is not None else 'not included'} · result {'restored' if edited is not None else 'not included'}."
    return original, original, edited, manifest.get("prompt", ""), manifest.get("intensity", 65), mask, *values, status


def run_diagnostics():
    """Report environment and optional engine availability without exposing secrets."""
    try:
        import gradio as _gr
        gradio_version = getattr(_gr, "__version__", "unknown")
    except Exception:
        gradio_version = "unavailable"
    try:
        import PIL as _pil
        pillow_version = getattr(_pil, "__version__", "unknown")
    except Exception:
        pillow_version = "unavailable"
    try:
        import numpy as _np
        numpy_version = getattr(_np, "__version__", "unknown")
    except Exception:
        numpy_version = "unavailable"
    scenario_status = f"{len(SCENARIOS)} scenarios loaded" if SCENARIOS else "scenario library unavailable; built-in fallback active"
    key_status = "Configured (key value hidden)" if API_KEY else "Not configured — local editing remains available"
    looks_status = f"{len(load_custom_looks())} saved custom looks"
    return (f"MOOD diagnostics · release {APP_VERSION}\n"
            f"Python: {os.sys.version.split()[0]}\nGradio: {gradio_version}\nPillow: {pillow_version}\nNumPy: {numpy_version}\n"
            f"Gemini: {key_status}\nScenario library: {scenario_status}\nCustom looks: {looks_status}\n"
            f"Project folder: {BASE_DIR}\nUpload limit: {MAX_UPLOAD_MB} MB per file\nImage pixel limit: {MAX_IMAGE_PIXELS:,}\nDefault network exposure: loopback only\nAPI keys are not displayed or included in project exports.")


def save_jury_feedback(overall, edit_task, export_task, batch_task, clarity, notes):
    """Create a downloadable, local text report from the jury evaluation form."""
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = f"""MOOD JURY EVALUATION
Submitted: {stamp}

OVERALL IMPRESSION
{overall or "Not answered"}

SINGLE-PHOTO EDIT TASK
{edit_task or "Not answered"}

EXPORT TASK
{export_task or "Not answered"}

TWO-PHOTO BATCH TASK
{batch_task or "Not answered"}

UI CLARITY
{clarity or "Not answered"}

NOTES / BUGS / SUGGESTIONS
{(notes or "").strip() or "No additional notes"}

---
This report was generated by MOOD. It does not include photos, API keys, or personal account data.
"""
    fd, path = tempfile.mkstemp(prefix="mood_jury_feedback_", suffix=".txt", text=True)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(report)
    return path, "Feedback report prepared. Download it and send it to the project owner."


# ============================================================
# DESIGN
# ============================================================

CSS = """
:root {
  color-scheme: dark;
  --mood-bg: #050807;
  --mood-panel: #0b110e;
  --mood-line: rgba(145, 190, 164, .15);
  --mood-green: #00f5a0;
  --mood-text: #f1f8f4;
  --mood-muted: #91a69a;
  --body-background-fill: #050807 !important;
  --background-fill-primary: #070b09 !important;
  --background-fill-secondary: #0b110e !important;
  --border-color-primary: #26372d !important;
  --border-color-accent: #00f5a0 !important;
  --color-accent: #00f5a0 !important;
  --button-primary-background-fill: #00f5a0 !important;
  --button-primary-background-fill-hover: #54ffc0 !important;
  --button-primary-text-color: #04150d !important;
}

html {
  background: #050807 !important;
}

html, body, .gradio-container, .main, .contain, .app {
  background:
    radial-gradient(ellipse at 8% 0%, rgba(0, 245, 160, .075), transparent 35%),
    radial-gradient(ellipse at 94% 22%, rgba(0, 130, 90, .065), transparent 32%),
    #050807 !important;
  color: var(--mood-text) !important;
  font-family: Inter, ui-sans-serif, system-ui, sans-serif !important;
}

.gradio-container {
  max-width: 1660px !important;
  margin: 0 auto !important;
  padding: 0 32px 34px !important;
  font-size: 15px !important;
}

footer, .built-with {
  display: none !important;
}

* {
  scrollbar-color: #294536 #070b09;
}

::selection {
  background: rgba(0, 245, 160, .25);
  color: #fff;
}

#topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 19px 2px;
  border-bottom: 1px solid rgba(145, 190, 164, .14);
  margin-bottom: 25px;
}

#brandmark {
  display: flex;
  align-items: center;
  gap: 12px;
  font-size: 21px;
  font-weight: 850;
  letter-spacing: 3px;
}

#brandicon {
  width: 38px;
  height: 38px;
  display: grid;
  place-items: center;
  border-radius: 12px;
  color: #03130c;
  background: linear-gradient(145deg, #65ffbf, #00d98a);
  font-size: 21px;
  font-weight: 900;
  box-shadow:
    0 0 25px rgba(0, 245, 160, .19),
    inset 0 1px 0 rgba(255,255,255,.5);
}

#topnote {
  color: #8ca497;
  font-size: 11px;
  letter-spacing: 2px;
  text-transform: uppercase;
}

#hero {
  padding: 0 2px 12px;
}

#eyebrow {
  color: #00f5a0;
  font-size: 11px;
  font-weight: 800;
  letter-spacing: 2.8px;
  text-transform: uppercase;
}

#headline {
  font-size: clamp(34px, 4vw, 52px);
  line-height: 1.05;
  font-weight: 820;
  letter-spacing: -2px;
  margin: 8px 0;
  color: #f1f8f4;
}

#headline span {
  color: #00f5a0;
  text-shadow: 0 0 32px rgba(0, 245, 160, .16);
}

#subhead {
  max-width: 700px;
  color: #a4b6ab;
  font-size: 16px;
  line-height: 1.7;
}

.panel {
  background: linear-gradient(
    145deg,
    rgba(19, 30, 24, .86),
    rgba(9, 15, 11, .92)
  ) !important;
  border: 1px solid rgba(145, 190, 164, .17) !important;
  border-radius: 20px !important;
  padding: 23px !important;
  box-shadow:
    inset 0 1px 0 rgba(255,255,255,.035),
    0 16px 44px rgba(0,0,0,.17) !important;
  backdrop-filter: blur(18px);
  -webkit-backdrop-filter: blur(18px);
}

.panel-title {
  font-size: 11px;
  font-weight: 850;
  letter-spacing: 1.9px;
  color: #00f5a0;
  margin-bottom: 12px;
  text-transform: uppercase;
}

.panel-note {
  color: #91a69a;
  font-size: 13px;
  line-height: 1.7;
  margin: 6px 0 16px;
}

.primary-btn {
  background: linear-gradient(115deg, #68ffc1, #00e995) !important;
  color: #03150d !important;
  border: 1px solid rgba(145, 255, 204, .8) !important;
  font-size: 15px !important;
  font-weight: 850 !important;
  min-height: 54px !important;
  border-radius: 13px !important;
  box-shadow:
    0 0 28px rgba(0, 245, 160, .13),
    inset 0 1px 0 rgba(255,255,255,.42) !important;
  transition:
    filter .18s ease,
    transform .18s ease,
    box-shadow .18s ease !important;
}

.primary-btn:hover {
  filter: brightness(1.07) !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 0 34px rgba(0, 245, 160, .22) !important;
}

button, [role="button"] {
  border-radius: 11px !important;
  border-color: #293c31 !important;
  color: #e7f2eb !important;
  font-size: 13px !important;
  transition:
    border-color .18s ease,
    background .18s ease,
    color .18s ease !important;
}

button:hover, [role="button"]:hover {
  border-color: rgba(0, 245, 160, .65) !important;
}

button:focus-visible, [role="button"]:focus-visible {
  outline: 2px solid #00f5a0 !important;
  outline-offset: 3px !important;
}

input, textarea, .wrap, .form, .input-container {
  background: rgba(5, 10, 7, .78) !important;
  border-color: #293c31 !important;
  color: #f1f8f4 !important;
}

input, textarea {
  border-radius: 11px !important;
}

label, .label-wrap, .svelte-1gfkn6j {
  color: #dce9e0 !important;
  font-size: 14px !important;
}

input:focus, textarea:focus {
  border-color: #00f5a0 !important;
  box-shadow: 0 0 0 2px rgba(0, 245, 160, .12) !important;
}

input::placeholder, textarea::placeholder {
  color: #71877a !important;
  opacity: 1 !important;
}

.image-container, .image-frame {
  border-radius: 15px !important;
  overflow: hidden !important;
}

#statusbox textarea {
  color: #c5d7cb !important;
  font-size: 13px !important;
  line-height: 1.65 !important;
}

#footerline {
  border-top: 1px solid rgba(145, 190, 164, .13);
  margin-top: 25px;
  padding: 20px 2px 4px;
  color: #71877a;
  font-size: 10px;
  letter-spacing: 2px;
  text-align: center;
}

.block {
  border-radius: 14px !important;
}

.gradio-container .tabs,
.gradio-container .tab-nav,
.gradio-container [data-testid="block-info"] {
  border-color: rgba(145, 190, 164, .15) !important;
}

.gradio-container .slider input,
input[type="range"] {
  accent-color: #00f5a0 !important;
}

.library-row {
  margin-top: 8px !important;
}

#export-status textarea {
  min-height: 44px !important;
}

.workspace-tabs {
  margin-top: 2px !important;
}

.workspace-tabs > .tab-nav,
.workspace-tabs .tab-nav {
  gap: 7px !important;
  border-bottom: 1px solid rgba(145, 190, 164, .14) !important;
  margin-bottom: 12px !important;
  padding-bottom: 7px !important;
}

.workspace-tabs button[role="tab"] {
  background: rgba(13, 21, 16, .78) !important;
  border: 1px solid rgba(145, 190, 164, .13) !important;
  border-radius: 12px !important;
  padding: 12px 17px !important;
  color: #9cafa4 !important;
  font-weight: 700 !important;
}

.workspace-tabs button[role="tab"][aria-selected="true"] {
  background: linear-gradient(
    135deg,
    rgba(0, 245, 160, .13),
    rgba(0, 245, 160, .045)
  ) !important;
  border-color: rgba(0, 245, 160, .55) !important;
  color: #63ffc0 !important;
  box-shadow:
    inset 0 1px 0 rgba(255,255,255,.04),
    0 0 22px rgba(0,245,160,.045) !important;
}

.workspace-intro {
  padding: 2px 2px 12px;
}

.workspace-kicker {
  color: #00f5a0;
  font-size: 10px;
  font-weight: 850;
  letter-spacing: 2.2px;
  margin-bottom: 7px;
}

.workspace-heading {
  color: #f1f8f4;
  font-size: 26px;
  font-weight: 820;
  letter-spacing: -.8px;
  margin-bottom: 6px;
}

.workspace-copy {
  color: #9aafa1;
  font-size: 13px;
  line-height: 1.7;
  max-width: 760px;
}

.accordion, details {
  border-color: rgba(145, 190, 164, .16) !important;
  border-radius: 13px !important;
}

hr {
  border-color: rgba(145, 190, 164, .13) !important;
}

@media (max-width: 760px) {
  .gradio-container {
    padding: 0 12px 22px !important;
  }

  #topbar {
    padding: 15px 2px;
    margin-bottom: 20px;
  }

  #brandmark {
    font-size: 18px;
    letter-spacing: 2px;
  }

  #brandicon {
    width: 34px;
    height: 34px;
  }

  #headline {
    font-size: clamp(30px, 8vw, 44px);
    letter-spacing: -1.5px;
    margin: 7px 0;
  }

  #topnote {
    display: none;
  }

  #hero {
    padding-bottom: 12px;
  }

  .panel {
    padding: 15px !important;
    border-radius: 16px !important;
  }

  #subhead {
    font-size: 15px;
  }

  .workspace-tabs .tab-nav {
    gap: 6px !important;
  }

  .workspace-tabs button[role="tab"] {
    padding: 10px 12px !important;
  }

  .workspace-heading {
    font-size: 23px;
  }
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    transition-duration: .01ms !important;
    animation-duration: .01ms !important;
  }
}
"""

# ============================================================
# GRADIO APP
# ============================================================

with gr.Blocks(
    title="MOOD — Creative Photo Studio · Phase 6",
) as demo:

    history_state = gr.State([])
    redo_state = gr.State([])
    adjustment_base_state = gr.State(None)

    gr.HTML("""
        <div id="topbar">
            <div id="brandmark"><div id="brandicon">M</div><span>MOOD</span></div>
            <div id="topnote">IMAGE LAB&nbsp; / &nbsp;PHASE 6</div>
        </div>
        <div id="hero">
            <div id="eyebrow">A focused creative tool</div>
            <div id="headline">Make it feel<br><span>different.</span></div>
            <div id="subhead">A focused photo-editing workspace. Start with one photo, make your edit, then use separate workspaces for adjustments, batch processing, projects, and jury evaluation.</div>
        </div>
    """)

    with gr.Tabs(elem_classes=["workspace-tabs"]):
        # --------------------------------------------------------
        # EDITOR: the only workspace most users need to start with
        # --------------------------------------------------------
        with gr.Tab("Editor", id="editor"):
            gr.HTML('<div class="workspace-intro"><div class="workspace-kicker">WORKSPACE 01</div><div class="workspace-heading">Edit a photo</div><div class="workspace-copy">Upload, describe the look, and compare your result. Advanced tools live in their own tabs.</div></div>')
            with gr.Row(equal_height=False):
                with gr.Column(scale=4, min_width=280):
                    with gr.Group(elem_classes=["panel"]):
                        gr.HTML('<div class="panel-title">01 / Source photo</div>')
                        gr.HTML('<div class="panel-note">Choose a JPG, PNG, or WebP image to begin.</div>')
                        input_image = gr.Image(label="Drop a photo here", type="pil", height=280)
                    with gr.Group(elem_classes=["panel"]):
                        gr.HTML('<div class="panel-title">02 / Creative direction</div>')
                        prompt = gr.Textbox(
                            label="Describe the mood",
                            placeholder="e.g. cinematic rainy evening, reflections on the street, natural skin tones",
                            lines=3, max_lines=6,
                        )
                        with gr.Accordion("Browse the look library", open=False):
                            gr.HTML('<div class="panel-note">Search and choose a starting point. You can edit the prompt afterward.</div>')
                            scenario_search = gr.Textbox(
                                label="Search looks",
                                placeholder="portrait, rainy neon, product, golden hour",
                                lines=1, max_lines=1,
                            )
                            scenario_category = gr.Dropdown(
                                choices=SCENARIO_CATEGORIES, value="All categories",
                                label="Category", interactive=True,
                            )
                            scenario_choice = gr.Dropdown(
                                choices=[item.get("name", "") for item in SCENARIOS],
                                label=f"Scenario library · {len(SCENARIOS)} looks",
                                interactive=True, filterable=True,
                            )
                            use_scenario_button = gr.Button("Use selected look", size="sm")
                        intensity = gr.Slider(
                            minimum=0, maximum=100, value=65, step=5,
                            label="Effect intensity",
                            info="0 = minimal effect · 100 = strongest effect",
                        )
                        edit_button = gr.Button(
                            "Transform photo  →", variant="primary",
                            elem_classes=["primary-btn"],
                        )
                with gr.Column(scale=6, min_width=320):
                    with gr.Group(elem_classes=["panel"]):
                        with gr.Row():
                            gr.HTML('<div class="panel-title">03 / Before & after</div>')
                            gr.HTML('<div style="margin-left:auto;color:#aab5af;font-size:11px;letter-spacing:1px">COMPARE</div>')
                        with gr.Row(equal_height=False):
                            with gr.Column(scale=1, min_width=130):
                                original_preview = gr.Image(
                                    label="Original", type="pil", height=330,
                                    interactive=False,
                                )
                            with gr.Column(scale=1, min_width=130):
                                output_image = gr.Image(
                                    label="Edited result", type="pil", height=330,
                                    interactive=False,
                                )
                        with gr.Row():
                            reset_button = gr.Button("↺ Reset preview", size="sm")
                            clear_button = gr.Button("Clear session", size="sm")
                        status = gr.Textbox(
                            label="Status",
                            value="Ready. Upload a photo and describe the mood.",
                            interactive=False, lines=2, elem_id="statusbox",
                        )
                        gr.HTML('<div class="panel-note">Your original remains unchanged. Reset returns the preview to the source photo.</div>')
                    with gr.Group(elem_classes=["panel"]):
                        gr.HTML('<div class="panel-title">04 / Export</div>')
                        with gr.Row():
                            export_format = gr.Dropdown(
                                choices=["PNG", "JPEG", "WEBP"], value="PNG",
                                label="Format", interactive=True,
                            )
                            export_quality = gr.Slider(
                                minimum=50, maximum=100, value=95, step=5,
                                label="Quality (JPEG / WebP)",
                            )
                        export_button = gr.Button("Prepare download  ↓", variant="primary", elem_classes=["primary-btn"])
                        download_file = gr.File(label="Download edited image", interactive=False)
                        export_status = gr.Textbox(
                            label="Export status",
                            value="Create an edit, then export it.",
                            interactive=False, lines=1, elem_id="export-status",
                        )

        # --------------------------------------------------------
        # ADJUSTMENTS: advanced controls are kept out of the editor
        # --------------------------------------------------------
        with gr.Tab("Adjustments", id="adjustments"):
            gr.HTML('<div class="workspace-intro"><div class="workspace-kicker">WORKSPACE 02</div><div class="workspace-heading">Fine-tune the result</div><div class="workspace-copy">These controls apply to the current edited image. Use a mask when you only want to adjust selected areas.</div></div>')
            with gr.Group(elem_classes=["panel"]):
                gr.HTML('<div class="panel-title">Light and color</div>')
                with gr.Row():
                    exposure_control = gr.Slider(-2.0, 2.0, value=0, step=0.1, label="Exposure")
                    contrast_control = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Contrast")
                    saturation_control = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Saturation")
                with gr.Row():
                    temperature_control = gr.Slider(-100, 100, value=0, step=5, label="Cool ↔ warm")
                    highlights_control = gr.Slider(-100, 100, value=0, step=5, label="Highlights")
                    shadows_control = gr.Slider(-100, 100, value=0, step=5, label="Shadows")
                with gr.Row():
                    sharpness_control = gr.Slider(0, 3, value=1, step=0.1, label="Sharpness")
                    denoise_control = gr.Slider(0, 1, value=0, step=0.05, label="Noise reduction")
            with gr.Group(elem_classes=["panel"]):
                gr.HTML('<div class="panel-title">Selective adjustments</div>')
                gr.HTML('<div class="panel-note">Optional grayscale mask: white areas receive adjustments; black areas remain unchanged.</div>')
                selective_mask = gr.Image(label="Optional adjustment mask", type="pil", height=180)
                with gr.Row():
                    apply_adjustments_button = gr.Button("Apply adjustments", variant="primary", elem_classes=["primary-btn"])
                    reset_controls_button = gr.Button("Reset controls")
                    undo_button = gr.Button("↶ Undo")
                    redo_button = gr.Button("↷ Redo")
                adjustment_status = gr.Textbox(label="Adjustment status", value="Ready for precise adjustments.", interactive=False, lines=1)
            with gr.Group(elem_classes=["panel"]):
                gr.HTML('<div class="panel-title">Saved looks</div>')
                with gr.Row():
                    custom_look_name = gr.Textbox(label="Look name", placeholder="e.g. Soft warm portrait", scale=2)
                    save_look_button = gr.Button("Save current look", scale=1)
                with gr.Row():
                    custom_look_choice = gr.Dropdown(choices=load_custom_looks(), label="Saved custom looks", interactive=True, scale=2)
                    load_look_button = gr.Button("Load look controls", scale=1)

        # --------------------------------------------------------
        # BATCH: isolated from single-image editing
        # --------------------------------------------------------
        with gr.Tab("Batch", id="batch"):
            gr.HTML('<div class="workspace-intro"><div class="workspace-kicker">WORKSPACE 03</div><div class="workspace-heading">Process multiple photos</div><div class="workspace-copy">Apply one direction to a group of images and download the results together.</div></div>')
            with gr.Group(elem_classes=["panel"]):
                gr.HTML('<div class="panel-title">Batch job</div>')
                gr.HTML('<div class="panel-note">Select up to 30 images. Gemini requests can take longer and may incur API usage; local mode uses conventional image effects.</div>')
                batch_files = gr.File(label="Choose photos", file_count="multiple", file_types=["image"], interactive=True)
                batch_prompt = gr.Textbox(label="Batch direction", placeholder="e.g. natural warm editorial color grade", lines=2)
                batch_intensity = gr.Slider(0, 100, value=65, step=5, label="Batch intensity")
                batch_button = gr.Button("Process batch  →", variant="primary", elem_classes=["primary-btn"])
                batch_download = gr.File(label="Download batch ZIP", interactive=False)
                batch_status = gr.Textbox(label="Batch status", interactive=False, lines=3)

        # --------------------------------------------------------
        # PROJECTS: save and restore sessions
        # --------------------------------------------------------
        with gr.Tab("Projects", id="projects"):
            gr.HTML('<div class="workspace-intro"><div class="workspace-kicker">WORKSPACE 04</div><div class="workspace-heading">Project files</div><div class="workspace-copy">Save a portable session or reopen a previously saved MOOD project.</div></div>')
            with gr.Row(equal_height=False):
                with gr.Column(min_width=280):
                    with gr.Group(elem_classes=["panel"]):
                        gr.HTML('<div class="panel-title">Save current project</div>')
                        gr.HTML('<div class="panel-note">Includes source/result snapshots, prompt, intensity, mask, and adjustment settings. API keys are never included.</div>')
                        project_save_button = gr.Button("Save project ZIP  ↓", variant="primary", elem_classes=["primary-btn"])
                        project_download = gr.File(label="Download project", interactive=False)
                        project_status = gr.Textbox(label="Save status", interactive=False, lines=2)
                with gr.Column(min_width=280):
                    with gr.Group(elem_classes=["panel"]):
                        gr.HTML('<div class="panel-title">Open a project</div>')
                        project_upload = gr.File(label="Choose a MOOD project ZIP", file_types=[".zip"], interactive=True)
                        project_load_button = gr.Button("Load project", size="lg", variant="primary", elem_classes=["primary-btn"])
                        project_load_status = gr.Textbox(label="Load status", value="Choose a project ZIP to restore a session.", interactive=False, lines=2)

        # --------------------------------------------------------
        # JURY: an explicit, repeatable evaluation path
        # --------------------------------------------------------
        with gr.Tab("Jury test", id="jury"):
            gr.HTML('<div class="workspace-intro"><div class="workspace-kicker">EVALUATION</div><div class="workspace-heading">Test MOOD</div><div class="workspace-copy">Please follow the same tasks in order. The goal is to evaluate the current app, not to guess how it is supposed to work.</div></div>')
            with gr.Group(elem_classes=["panel"]):
                gr.HTML('<div class="panel-title">Test protocol</div>')
                gr.Markdown("""
**1 · Edit one photo**
- Open **Editor**, upload a non-sensitive sample photo, enter a short direction, and select **Transform photo**.
- Check whether the status explains what happened and whether the result visibly changes the image.

**2 · Compare and export**
- Compare the original and edited previews.
- Export the edited result, download it, and open the file to confirm it works.

**3 · Process two photos**
- Open **Batch**, select exactly two sample images, enter one direction, and run the batch.
- Download the ZIP and confirm both edited images are present.

**4 · Explore controls**
- Open **Adjustments** and try one change, then Undo/Redo.
- Open **Projects** and save a project if time permits.

**Please record actual outcomes.** If something fails, note the workspace, the action taken, and the error message. Avoid confidential images.
""")
            with gr.Group(elem_classes=["panel"]):
                gr.HTML('<div class="panel-title">Evaluation feedback</div>')
                gr.HTML('<div class="panel-note">Submitting sends these ratings and notes to the MOOD app instance so it can create a downloadable text report. It does not send feedback to a separate analytics service or include photos/API keys. If MOOD is hosted online, the host processes this feedback.</div>')
                jury_overall = gr.Radio(
                    choices=["Very good", "Good", "Okay", "Poor", "Could not complete"],
                    label="Overall experience",
                )
                jury_edit = gr.Radio(
                    choices=["Passed", "Partially passed", "Failed", "Not tested"],
                    value="Not tested", label="Single-photo edit",
                )
                jury_export = gr.Radio(
                    choices=["Passed", "Partially passed", "Failed", "Not tested"],
                    value="Not tested", label="Export and downloaded file",
                )
                jury_batch = gr.Radio(
                    choices=["Passed", "Partially passed", "Failed", "Not tested"],
                    value="Not tested", label="Two-photo batch",
                )
                jury_clarity = gr.Radio(
                    choices=["Very clear", "Mostly clear", "Confusing", "Very confusing"],
                    label="Was the UI easy to understand?",
                )
                jury_notes = gr.Textbox(
                    label="Issues and suggestions",
                    placeholder="What confused you? What failed? What should be clearer?",
                    lines=4, max_lines=10,
                )
                jury_submit = gr.Button("Create feedback report  ↓", variant="primary", elem_classes=["primary-btn"])
                jury_report_file = gr.File(label="Download feedback report", interactive=False)
                jury_report_status = gr.Textbox(label="Feedback report status", interactive=False, lines=1)

        # --------------------------------------------------------
        # SETTINGS: privacy and diagnostics are secondary
        # --------------------------------------------------------
        with gr.Tab("Settings", id="settings"):
            gr.HTML('<div class="workspace-intro"><div class="workspace-kicker">CONFIGURATION</div><div class="workspace-heading">Settings & diagnostics</div><div class="workspace-copy">Privacy details, configuration checks, and troubleshooting live here rather than in the editing flow.</div></div>')
            with gr.Group(elem_classes=["panel"]):
                gr.HTML('<div class="panel-title">Privacy</div>')
                gr.Markdown("""
- MOOD runs as a local app by default and binds to your computer.
- If Gemini is configured, the selected image and prompt are sent to Google for AI editing.
- Local effects run on this computer.
- Exports, projects, and feedback reports are generated as downloadable files.
- Do not expose the development server directly to the public internet.
""")
                release_info = gr.Textbox(label="Release readiness & configuration", value="Run the release check for configuration details and distribution reminders.", interactive=False, lines=7)
                release_info_button = gr.Button("Run release check")
            with gr.Group(elem_classes=["panel"]):
                gr.HTML('<div class="panel-title">Diagnostics</div>')
                gr.HTML('<div class="panel-note">Check the environment, Gemini configuration, scenario library, and saved custom looks without revealing secrets.</div>')
                diagnostics_button = gr.Button("Run diagnostics")
                diagnostics_output = gr.Textbox(label="System report", value="Run diagnostics when needed.", interactive=False, lines=7)

    gr.HTML('<div id="footerline">MOOD&nbsp; / &nbsp;PHASE 6 · CLEAN WORKSPACES · JURY EVALUATION</div>')

    edit_button.click(
        fn=edit_photo,
        inputs=[input_image, prompt, intensity],
        outputs=[output_image, status],
    ).then(lambda img: img, inputs=[output_image], outputs=[adjustment_base_state]).then(lambda: ([], []), outputs=[history_state, redo_state])
    prompt.submit(
        fn=edit_photo,
        inputs=[input_image, prompt, intensity],
        outputs=[output_image, status],
    ).then(lambda img: img, inputs=[output_image], outputs=[adjustment_base_state]).then(lambda: ([], []), outputs=[history_state, redo_state])

    input_image.change(
        fn=sync_source_preview,
        inputs=[input_image],
        outputs=[original_preview, output_image, status],
    ).then(lambda: None, outputs=[adjustment_base_state]).then(lambda: ([], []), outputs=[history_state, redo_state])
    reset_button.click(
        fn=reset_preview,
        inputs=[input_image],
        outputs=[output_image, status],
    ).then(lambda img: img, inputs=[output_image], outputs=[adjustment_base_state]).then(lambda: ([], []), outputs=[history_state, redo_state])
    clear_button.click(
        fn=clear_session,
        inputs=[],
        outputs=[input_image, original_preview, output_image, prompt, intensity, history_state, redo_state, selective_mask, adjustment_base_state, status],
    )

    scenario_search.change(
        fn=filter_scenarios,
        inputs=[scenario_search, scenario_category],
        outputs=[scenario_choice],
    )
    scenario_category.change(
        fn=filter_scenarios,
        inputs=[scenario_search, scenario_category],
        outputs=[scenario_choice],
    )
    use_scenario_button.click(fn=use_scenario, inputs=[scenario_choice], outputs=[prompt])
    scenario_choice.change(fn=use_scenario, inputs=[scenario_choice], outputs=[prompt])

    apply_adjustments_button.click(
        fn=apply_controls,
        inputs=[adjustment_base_state, output_image, exposure_control, contrast_control, saturation_control, temperature_control,
                highlights_control, shadows_control, sharpness_control, denoise_control, selective_mask, history_state],
        outputs=[output_image, history_state, adjustment_status],
    ).then(lambda: [], outputs=[redo_state])

    undo_button.click(fn=undo_edit, inputs=[output_image, history_state, redo_state],
                      outputs=[output_image, history_state, redo_state, adjustment_status])
    redo_button.click(fn=redo_edit, inputs=[output_image, history_state, redo_state],
                      outputs=[output_image, history_state, redo_state, adjustment_status])
    reset_controls_button.click(fn=reset_adjustment_controls,
        outputs=[exposure_control, contrast_control, saturation_control, temperature_control,
                 highlights_control, shadows_control, sharpness_control, denoise_control, adjustment_status])
    save_look_button.click(fn=save_custom_look,
        inputs=[custom_look_name, exposure_control, contrast_control, saturation_control, temperature_control,
                highlights_control, shadows_control, sharpness_control, denoise_control],
        outputs=[custom_look_choice, adjustment_status])
    load_look_button.click(fn=apply_saved_look, inputs=[custom_look_choice],
        outputs=[exposure_control, contrast_control, saturation_control, temperature_control,
                 highlights_control, shadows_control, sharpness_control, denoise_control, adjustment_status])


    batch_button.click(fn=process_batch, inputs=[batch_files, batch_prompt, batch_intensity], outputs=[batch_download, batch_status])
    project_save_button.click(
        fn=save_project_bundle,
        inputs=[input_image, output_image, prompt, intensity, exposure_control, contrast_control, saturation_control,
                temperature_control, highlights_control, shadows_control, sharpness_control, denoise_control, selective_mask],
        outputs=[project_download, project_status],
    )
    project_load_button.click(
        fn=load_project_bundle,
        inputs=[project_upload],
        outputs=[input_image, original_preview, output_image, prompt, intensity, selective_mask,
                 exposure_control, contrast_control, saturation_control, temperature_control, highlights_control,
                 shadows_control, sharpness_control, denoise_control, project_status],
    ).then(lambda img: img, inputs=[output_image], outputs=[adjustment_base_state]).then(lambda: ([], []), outputs=[history_state, redo_state])
    jury_submit.click(
        fn=save_jury_feedback,
        inputs=[jury_overall, jury_edit, jury_export, jury_batch, jury_clarity, jury_notes],
        outputs=[jury_report_file, jury_report_status],
    )

    release_info_button.click(fn=release_readiness_report, outputs=[release_info])
    diagnostics_button.click(fn=run_diagnostics, outputs=[diagnostics_output])

    export_button.click(
        fn=export_image,
        inputs=[output_image, export_format, export_quality],
        outputs=[download_file, export_status],
    )

# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    host = os.getenv("MOOD_HOST", "0.0.0.0" if os.getenv("RENDER") else "127.0.0.1").strip() or "127.0.0.1"
    if host not in {"127.0.0.1", "localhost", "::1"}:
        print("[MOOD SECURITY WARNING] MOOD_HOST exposes the app beyond loopback. Do not use this for public hosting without authentication, TLS, and a hardened deployment.")
    print(f"[MOOD] Release {APP_VERSION}; listening on {host}:{PORT}; upload limit {MAX_UPLOAD_MB} MB; image pixel limit {MAX_IMAGE_PIXELS:,}.")
    demo.queue(max_size=8, default_concurrency_limit=1)
    demo.launch(
        server_name=host,
        server_port=PORT,
        inbrowser=not bool(os.getenv("RENDER")),
        show_error=False,
        share=False,
        theme=gr.themes.Base(
            primary_hue="green",
            secondary_hue="green",
            neutral_hue="zinc",
            font=gr.themes.GoogleFont("Inter"),
        ),
        css=CSS,
    )

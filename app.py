
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
# MAIN EDIT HANDLER
# ============================================================

def edit_photo(image, prompt, intensity=65, progress=gr.Progress()):
    if image is None:
        raise gr.Error("Upload a photograph first.")

    if not prompt or not prompt.strip():
        raise gr.Error("Describe the effect you want.")

    prompt = prompt.strip()
    intensity = int(intensity)
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
    if not path:
        raise ValueError("One of the uploaded files has no readable path.")
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
    if not (prompt or "").strip():
        raise gr.Error("Enter a creative direction for the batch.")
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
            original = _open_uploaded_image(item)
            edited, method = _edit_one_for_batch(original, prompt.strip(), int(intensity))
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
    return ("MOOD diagnostics\n"
            f"Python: {os.sys.version.split()[0]}\nGradio: {gradio_version}\nPillow: {pillow_version}\nNumPy: {numpy_version}\n"
            f"Gemini: {key_status}\nScenario library: {scenario_status}\nCustom looks: {looks_status}\n"
            f"Project folder: {BASE_DIR}\nAPI keys are not displayed or included in project exports.")


# ============================================================
# DESIGN
# ============================================================

CSS = """
:root {
  color-scheme: dark;
  --mood-bg: #050706;
  --mood-panel: #0b0f0c;
  --mood-line: #202923;
  --mood-green: #00FF9A;
  --mood-text: #f5f7f6;
  --mood-muted: #aab5af;
  --body-background-fill: #050706 !important;
  --background-fill-primary: #050706 !important;
  --background-fill-secondary: #0b0f0c !important;
  --border-color-primary: #202923 !important;
  --border-color-accent: #00FF9A !important;
  --color-accent: #00FF9A !important;
  --button-primary-background-fill: #00FF9A !important;
  --button-primary-background-fill-hover: #36ffb2 !important;
  --button-primary-text-color: #03130c !important;
}
html, body, .gradio-container, .main, .contain, .app {
  background: #050706 !important; color: #f5f7f6 !important;
  font-family: Inter, ui-sans-serif, system-ui, sans-serif !important;
}
.gradio-container { max-width: 1600px !important; margin: 0 auto !important; padding: 0 34px 32px !important; font-size: 16px !important; }
footer, .built-with { display: none !important; }
* { scrollbar-color: #29342d #050706; }
#topbar { display:flex; align-items:center; justify-content:space-between; padding:24px 2px; border-bottom:1px solid #202923; margin-bottom:38px; }
#brandmark { display:flex; align-items:center; gap:12px; font-size:22px; font-weight:850; letter-spacing:3px; }
#brandicon { width:36px; height:36px; display:grid; place-items:center; border-radius:9px; color:#03130c; background:#00FF9A; font-size:21px; font-weight:900; box-shadow:0 0 24px #00FF9A24; }
#topnote { color:#aab5af; font-size:13px; letter-spacing:2px; text-transform:uppercase; }
#hero { padding:8px 2px 34px; }
#eyebrow { color:#00FF9A; font-size:13px; font-weight:750; letter-spacing:2.5px; text-transform:uppercase; }
#headline { font-size:clamp(46px,6vw,76px); line-height:1.02; font-weight:820; letter-spacing:-3px; margin:16px 0 18px; color:#f5f7f6; }
#headline span { color:#00FF9A; }
#subhead { max-width:720px; color:#b4beb8; font-size:18px; line-height:1.7; }
.panel { background:#0b0f0c !important; border:1px solid #202923 !important; border-radius:16px !important; padding:24px !important; box-shadow:none !important; }
.panel-title { font-size:14px; font-weight:800; letter-spacing:1.6px; color:#00FF9A; margin-bottom:14px; text-transform:uppercase; }
.panel-note { color:#aab5af; font-size:14px; line-height:1.65; margin:6px 0 16px; }
.primary-btn { background:#00FF9A !important; color:#03130c !important; border:1px solid #00FF9A !important; font-size:16px !important; font-weight:800 !important; min-height:56px !important; border-radius:10px !important; box-shadow:none !important; transition:filter .15s ease, transform .15s ease !important; }
.primary-btn:hover { filter:brightness(1.08) !important; transform:translateY(-1px) !important; }
button, [role="button"] { border-radius:9px !important; border-color:#29342d !important; color:#f5f7f6 !important; font-size:14px !important; transition:border-color .15s ease, background .15s ease !important; }
button:hover, [role="button"]:hover { border-color:#00FF9A !important; }
input, textarea, .wrap, .form, .input-container { background:#070a08 !important; border-color:#29342d !important; color:#f5f7f6 !important; }
label, .label-wrap, .svelte-1gfkn6j { color:#e2e8e4 !important; font-size:15px !important; }
input:focus, textarea:focus { border-color:#00FF9A !important; box-shadow:0 0 0 1px #00FF9A40 !important; }
input::placeholder, textarea::placeholder { color:#87958c !important; opacity:1 !important; }
.image-container, .image-frame { border-radius:12px !important; overflow:hidden !important; }
#statusbox textarea { color:#d0d9d3 !important; font-size:14px !important; line-height:1.6 !important; }
#footerline { border-top:1px solid #202923; margin-top:28px; padding:22px 2px 4px; color:#89968e; font-size:12px; letter-spacing:2px; text-align:center; }
.block { border-radius:12px !important; }
.gradio-container .tabs, .gradio-container .tab-nav, .gradio-container [data-testid="block-info"] { border-color:#202923 !important; }
.gradio-container .slider input, input[type="range"] { accent-color:#00FF9A !important; }
.library-row { margin-top: 8px !important; }
#export-status textarea { min-height: 44px !important; }
@media (max-width:760px) {
  .gradio-container { padding:0 12px 22px !important; }
  #headline { font-size:clamp(42px,11vw,60px); letter-spacing:-1.7px; }
  #topnote { display:none; }
  .panel { padding:15px !important; }
  #subhead { font-size:16px; }
}
"""

# ============================================================
# GRADIO APP
# ============================================================

with gr.Blocks(
    title="MOOD — Creative Photo Studio",
    theme=gr.themes.Base(
        primary_hue="green",
        secondary_hue="green",
        neutral_hue="zinc",
        font=gr.themes.GoogleFont("Inter"),
    ),
    css=CSS,
) as demo:

    history_state = gr.State([])
    redo_state = gr.State([])
    adjustment_base_state = gr.State(None)

    gr.HTML("""
        <div id="topbar">
            <div id="brandmark"><div id="brandicon">M</div><span>MOOD</span></div>
            <div id="topnote">IMAGE LAB&nbsp; / &nbsp;PHASE 4</div>
        </div>
        <div id="hero">
            <div id="eyebrow">A focused creative tool</div>
            <div id="headline">Make it feel<br><span>different.</span></div>
            <div id="subhead">One photo. One direction. A whole new mood. Explore scenarios, fine-tune color and light, apply selective masks, build custom looks, process batches, save portable projects, and export.</div>
        </div>
    """)

    with gr.Row(equal_height=False):
        with gr.Column(scale=5, min_width=300):
            with gr.Group(elem_classes=["panel"]):
                gr.HTML('<div class="panel-title">01 / Source photograph</div>')
                gr.HTML('<div class="panel-note">Start with a photo you love. JPG, PNG, and WebP are supported.</div>')
                input_image = gr.Image(
                    label="Drop your image here",
                    type="pil",
                    height=310,
                )

            with gr.Group(elem_classes=["panel"]):
                gr.HTML('<div class="panel-title">02 / Direct the mood</div>')
                prompt = gr.Textbox(
                    label="Your creative direction",
                    placeholder="e.g. cinematic rainy evening, reflections on the street, natural skin tones",
                    lines=3,
                    max_lines=7,
                )
                gr.HTML('<div class="panel-note">Browse the full scenario library. Search by look, subject, atmosphere, or reference.</div>')
                scenario_search = gr.Textbox(
                    label="Search looks",
                    placeholder="e.g. portrait, rainy neon, product, golden hour",
                    lines=1,
                    max_lines=1,
                )
                scenario_category = gr.Dropdown(
                    choices=SCENARIO_CATEGORIES,
                    value="All categories",
                    label="Category",
                    interactive=True,
                )
                scenario_choice = gr.Dropdown(
                    choices=[item.get("name", "") for item in SCENARIOS],
                    label=f"Scenario library · {len(SCENARIOS)} looks",
                    interactive=True,
                    filterable=True,
                )
                use_scenario_button = gr.Button("Use selected look", size="sm")
                intensity = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=65,
                    step=5,
                    label="Effect intensity",
                    info="Lower for a gentle grade · higher for a stronger look",
                )
                edit_button = gr.Button(
                    "Transform photograph  →",
                    variant="primary",
                    elem_classes=["primary-btn"],
                )

        with gr.Column(scale=6, min_width=320):
            with gr.Group(elem_classes=["panel"]):
                with gr.Row():
                    gr.HTML('<div class="panel-title">03 / Compare your result</div>')
                    gr.HTML('<div style="margin-left:auto;color:#858095;font-size:11px;letter-spacing:1px">BEFORE / AFTER</div>')
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=140):
                        original_preview = gr.Image(
                            label="Original",
                            type="pil",
                            height=390,
                            interactive=False,
                        )
                    with gr.Column(scale=1, min_width=140):
                        output_image = gr.Image(
                            label="Edited result",
                            type="pil",
                            height=390,
                            interactive=False,
                        )
                with gr.Row():
                    reset_button = gr.Button("↺ Reset preview", size="sm")
                    clear_button = gr.Button("Clear session", size="sm")
                gr.HTML('<div style="color:#aab5af;font-size:12px;margin:0 0 12px">Reset restores the result preview; Clear session removes the current photo and prompt.</div>')
                status = gr.Textbox(
                    label="Session status",
                    value="Ready when you are. Upload a photograph and describe the mood.",
                    interactive=False,
                    lines=3,
                    elem_id="statusbox",
                )
                gr.HTML('<div class="panel-note">Tip: compare the original and edited versions side by side. Your uploaded source stays intact when you edit or reset.</div>')
            with gr.Group(elem_classes=["panel"]):
                gr.HTML('<div class="panel-title">04 / Export your edit</div>')
                with gr.Row():
                    export_format = gr.Dropdown(
                        choices=["PNG", "JPEG", "WEBP"], value="PNG", label="File format", interactive=True
                    )
                    export_quality = gr.Slider(
                        minimum=50, maximum=100, value=95, step=5, label="Quality (JPEG / WebP)"
                    )
                export_button = gr.Button("Prepare download  ↓", variant="primary", elem_classes=["primary-btn"])
                download_file = gr.File(label="Your exported image", interactive=False)
                export_status = gr.Textbox(
                    label="Export status", value="Export the edited result when you are ready.",
                    interactive=False, lines=1, elem_id="export-status"
                )

    with gr.Group(elem_classes=["panel"]):
        gr.HTML('<div class="panel-title">05 / Professional adjustments</div>')
        gr.HTML('<div class="panel-note">Fine-tune the current result. Use an optional grayscale mask to apply adjustments only to white areas (black areas stay unchanged).</div>')
        with gr.Row():
            exposure_control = gr.Slider(-2.0, 2.0, value=0, step=0.1, label="Exposure (stops)")
            contrast_control = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Contrast")
            saturation_control = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Saturation")
        with gr.Row():
            temperature_control = gr.Slider(-100, 100, value=0, step=5, label="Cool ↔ Warm")
            highlights_control = gr.Slider(-100, 100, value=0, step=5, label="Highlights")
            shadows_control = gr.Slider(-100, 100, value=0, step=5, label="Shadows")
        with gr.Row():
            sharpness_control = gr.Slider(0, 3, value=1, step=0.1, label="Sharpness")
            denoise_control = gr.Slider(0, 1, value=0, step=0.05, label="Noise reduction")
        selective_mask = gr.Image(label="Optional selective mask · white = adjust, black = protect", type="pil", height=160)
        with gr.Row():
            apply_adjustments_button = gr.Button("Apply adjustments", variant="primary", elem_classes=["primary-btn"])
            reset_controls_button = gr.Button("Reset controls")
            undo_button = gr.Button("↶ Undo")
            redo_button = gr.Button("↷ Redo")
        with gr.Row():
            custom_look_name = gr.Textbox(label="Custom look name", placeholder="e.g. Soft warm portrait", scale=2)
            save_look_button = gr.Button("Save current look", scale=1)
        with gr.Row():
            custom_look_choice = gr.Dropdown(choices=load_custom_looks(), label="Saved custom looks", interactive=True, scale=2)
            load_look_button = gr.Button("Load look controls", scale=1)
        adjustment_status = gr.Textbox(label="Adjustment status", value="Ready for precise adjustments.", interactive=False, lines=1)



    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=300):
            with gr.Group(elem_classes=["panel"]):
                gr.HTML('<div class="panel-title">06 / Batch studio</div>')
                gr.HTML('<div class="panel-note">Apply the same creative direction to up to 30 images and download one ZIP with edited images plus a processing manifest. Gemini requests may take longer and can incur API usage.</div>')
                batch_files = gr.File(label="Choose multiple photos", file_count="multiple", file_types=["image"], interactive=True)
                batch_prompt = gr.Textbox(label="Batch direction", placeholder="e.g. natural warm editorial color grade", lines=2)
                batch_intensity = gr.Slider(0, 100, value=65, step=5, label="Batch intensity")
                batch_button = gr.Button("Process batch →", variant="primary", elem_classes=["primary-btn"])
                batch_download = gr.File(label="Batch ZIP", interactive=False)
                batch_status = gr.Textbox(label="Batch status", interactive=False, lines=2)
        with gr.Column(scale=1, min_width=300):
            with gr.Group(elem_classes=["panel"]):
                gr.HTML('<div class="panel-title">07 / Portable project</div>')
                gr.HTML('<div class="panel-note">Save original and edited snapshots, prompt, intensity, selective mask, and adjustment settings in a ZIP. Reload the ZIP later to restore the session. Your API key is never included.</div>')
                project_save_button = gr.Button("Save project ZIP ↓", variant="primary", elem_classes=["primary-btn"])
                project_download = gr.File(label="Saved project", interactive=False)
                project_status = gr.Textbox(label="Project status", interactive=False, lines=2)
                project_upload = gr.File(label="Load a MOOD project ZIP", file_types=[".zip"], interactive=True)
                project_load_button = gr.Button("Load project", size="lg")
    with gr.Group(elem_classes=["panel"]):
        gr.HTML('<div class="panel-title">08 / Diagnostics</div>')
        gr.HTML('<div class="panel-note">Check the app environment, optional Gemini configuration, scenario library, and saved custom looks without revealing secrets.</div>')
        diagnostics_button = gr.Button("Run diagnostics")
        diagnostics_output = gr.Textbox(label="System report", value="Run diagnostics when you need to check MOOD's environment.", interactive=False, lines=7)


    gr.HTML('<div id="footerline">MOOD&nbsp; / &nbsp;CREATIVE IMAGE LAB</div>')

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
    demo.queue(max_size=8)
    demo.launch(
        server_name="127.0.0.1",
        server_port=PORT,
        inbrowser=True,
        show_error=True,
    )


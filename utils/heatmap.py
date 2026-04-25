import cv2
import numpy as np
from PIL import Image
import torch
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ─────────────────────────────────────────────────────────────────────────────
# NO MEDIAPIPE — mp.solutions was removed in mediapipe 0.10+.
# We use OpenCV's built-in Haar cascade instead (zero extra dependencies).
# ─────────────────────────────────────────────────────────────────────────────

try:
    _FACE_CASCADE = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
except Exception:
    _FACE_CASCADE = None

# Face region definitions as fractions of the detected face bounding box.
# (y_start, y_end, x_start, x_end) — all 0.0 to 1.0 relative to face box.
FACE_REGIONS = {
    "Forehead":  (0.00, 0.22, 0.15, 0.85),
    "Left Eye":  (0.22, 0.42, 0.10, 0.46),
    "Right Eye": (0.22, 0.42, 0.54, 0.90),
    "Nose":      (0.38, 0.62, 0.33, 0.67),
    "Lips":      (0.62, 0.78, 0.22, 0.78),
    "Jaw":       (0.78, 1.00, 0.10, 0.90),
}

# SPEED: ViT reshape transform — converts token sequence to 2D spatial grid
# WHY needed? Standard GradCAM expects [B, C, H, W] feature maps from CNNs.
# ViTs output [B, N_tokens, embedding_dim] — a sequence, not a grid.
# This transform reshapes it so GradCAM can produce a spatial heatmap.
def _vit_reshape_transform(tensor, height=14, width=14):
    """
    Reshapes ViT output tokens into a 2D spatial grid.
    ViT-base splits a 224×224 image into 14×14 = 196 patches.
    Token 0 is the [CLS] classification token — we skip it.
    Remaining 196 tokens correspond to the 14×14 patch grid.
    """
    # tensor shape: [batch, num_tokens, embed_dim]
    # Skip CLS token (index 0), reshape the rest to [B, H, W, C]
    result = tensor[:, 1:, :].reshape(
        tensor.size(0), height, width, tensor.size(2)
    )
    # Permute to [B, C, H, W] as expected by GradCAM
    result = result.permute(0, 3, 1, 2)
    return result


def generate_face_heatmap(image_path, fake_score, breakdown):
    """
    Generates a real Grad-CAM++ heatmap showing which image regions
    caused the model to predict "fake", then scores each facial zone.

    WHY GradCAMPlusPlus instead of plain GradCAM?
    GradCAM++ averages gradients across the spatial dimension.
    For ViTs this gives cleaner, more localised heatmaps — less noise.

    Returns: (PIL Image with heatmap overlay, dict of region percentages)
    On failure: (None, {})
    """
    from utils.model import get_model_and_extractor

    try:
        model, feature_extractor, fake_idx = get_model_and_extractor()

        # Load image in two formats needed by different steps
        img_pil = Image.open(image_path).convert("RGB")
        img_np = np.array(img_pil).astype(np.float32) / 255.0   # 0-1 float for blending

        # Prepare input tensor
        inputs = feature_extractor(images=img_pil, return_tensors="pt")
        input_tensor = inputs["pixel_values"]   # [1, 3, 224, 224]

        # ── Target layer: last encoder block's layernorm ─────────────────
        # WHY this layer? It's the last layer before the classifier head.
        # It has the richest semantic spatial information — the model has
        # already decided "fake" here, so its attention map shows exactly
        # which regions contributed to that decision.
        try:
            target_layer = model.vit.encoder.layer[-1].layernorm_before
        except AttributeError:
            try:
                target_layer = model.vit.layernorm
            except AttributeError:
                # Generic fallback for other architectures
                layers = [m for m in model.modules() if hasattr(m, "weight")]
                target_layer = layers[-3]

        targets = [ClassifierOutputTarget(fake_idx)]

        # ── Run GradCAM++ with ViT reshape transform ─────────────────────
        with GradCAMPlusPlus(
            model=model,
            target_layers=[target_layer],
            reshape_transform=_vit_reshape_transform   # KEY: makes ViT work
        ) as cam:
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0]   # Remove batch dim → [H, W]

        # Resize CAM map to original image size
        h, w = img_np.shape[:2]
        cam_resized = cv2.resize(grayscale_cam, (w, h))

        # ── Build colored overlay ─────────────────────────────────────────
        # COLORMAP_JET: blue=low (real) → green → yellow → red=high (fake)
        # image_weight=0.5: 50% original, 50% heatmap overlay
        visualization = show_cam_on_image(
            img_np,
            cam_resized,
            use_rgb=True,
            colormap=cv2.COLORMAP_JET,
            image_weight=0.5
        )

        # ── Per-region scores ─────────────────────────────────────────────
        region_scores = _calculate_region_scores(image_path, cam_resized, h, w)

        return Image.fromarray(visualization), region_scores

    except Exception as e:
        print(f"[heatmap] Failed on {image_path}: {e}")
        return None, {}


def _detect_face_box(image_path):
    """
    Detects the main face in the image using OpenCV Haar cascade.
    Returns (x, y, w, h) or None if no face found.
    """
    if _FACE_CASCADE is None or _FACE_CASCADE.empty():
        return None
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        return None
    faces = _FACE_CASCADE.detectMultiScale(
        img_gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40)
    )
    if len(faces) == 0:
        return None
    # Return largest detected face
    return tuple(sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0])


def _calculate_region_scores(image_path, cam_map, img_h, img_w):
    """
    Scores each facial region by averaging the Grad-CAM activation
    inside that region.

    With face detected: uses actual face bounding box → accurate zones.
    Without face: falls back to full-image fractions → still useful.
    """
    face_box = _detect_face_box(image_path)

    if face_box:
        fx, fy, fw, fh = face_box
        scores = {}
        for name, (y0f, y1f, x0f, x1f) in FACE_REGIONS.items():
            py0 = max(0, int(fy + y0f * fh))
            py1 = min(img_h, int(fy + y1f * fh))
            px0 = max(0, int(fx + x0f * fw))
            px1 = min(img_w, int(fx + x1f * fw))
            region = cam_map[py0:py1, px0:px1]
            scores[name] = round(float(np.mean(region)) * 100, 1) if region.size > 0 else 0.0
        return scores
    else:
        return _fallback_scores(cam_map)


def _fallback_scores(cam_map):
    """Geometric fallback when no face bounding box is detected."""
    h, w = cam_map.shape
    def z(y0f, y1f, x0f, x1f):
        r = cam_map[int(y0f*h):int(y1f*h), int(x0f*w):int(x1f*w)]
        return round(float(np.mean(r)) * 100, 1) if r.size > 0 else 0.0
    return {
        "Forehead":  z(0.05, 0.25, 0.25, 0.75),
        "Left Eye":  z(0.28, 0.42, 0.15, 0.45),
        "Right Eye": z(0.28, 0.42, 0.55, 0.85),
        "Nose":      z(0.42, 0.62, 0.35, 0.65),
        "Lips":      z(0.62, 0.76, 0.28, 0.72),
        "Jaw":       z(0.76, 0.95, 0.15, 0.85),
    }


def generate_signal_heatmap_data(frame_results):
    """
    Aggregates region scores across all frames that have them.
    Called by app.py to feed the signal breakdown chart.
    """
    if not frame_results:
        return {}
    totals, counts = {}, {}
    for r in frame_results:
        for region, score in r.get("region_scores", {}).items():
            totals[region] = totals.get(region, 0.0) + score
            counts[region] = counts.get(region, 0) + 1
    if not totals:
        return {}
    return {k: round(totals[k] / counts[k], 1) for k in totals}

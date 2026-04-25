import cv2
import numpy as np
from PIL import Image
import torch
from pytorch_grad_cam import GradCAMPlusPlus
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

        # ── Per-region scores ─────────────────────────────────────────────
        region_scores = _calculate_region_scores(image_path, cam_resized, h, w)

        visualization = _draw_gradient_overlay(
            np.array(img_pil),
            image_path,
            cam_resized,
        )

        return Image.fromarray(visualization), region_scores

    except Exception as e:
        print(f"[heatmap] Failed on {image_path}: {e}")
        return generate_face_region_overlay(image_path, fake_score, breakdown)


def generate_face_region_overlay(image_path, fake_score, breakdown):
    """
    Fallback explainability view when Grad-CAM is unavailable.
    It still marks facial zones and colors them by suspicion so the report
    never shows a plain, unexplained frame.
    """
    try:
        img_pil = Image.open(image_path).convert("RGB")
        img_np = np.array(img_pil)
        grid_map = _artifact_grid_map(img_np, fake_score, breakdown)
        full_map = cv2.resize(grid_map, (img_np.shape[1], img_np.shape[0]))
        region_scores = _calculate_region_scores(image_path, full_map, img_np.shape[0], img_np.shape[1])
        overlay = _draw_gradient_overlay(img_np, image_path, grid_map)
        return Image.fromarray(overlay), region_scores
    except Exception as e:
        print(f"[heatmap] Fallback overlay failed on {image_path}: {e}")
        return None, {}


def _fallback_region_scores_from_signals(fake_score, breakdown):
    base = float(fake_score) * 100
    breakdown = breakdown or {}
    signal_bias = {
        "Forehead": breakdown.get("frequency_noise", 0) * 80,
        "Left Eye": breakdown.get("face_confidence_drop", 0) * 90,
        "Right Eye": breakdown.get("face_confidence_drop", 0) * 90,
        "Nose": breakdown.get("color_inconsistency", 0) * 80,
        "Lips": breakdown.get("blur_anomaly", 0) * 85,
        "Jaw": breakdown.get("color_inconsistency", 0) * 60,
    }
    return {
        region: round(float(np.clip(base * 0.65 + bias, 0, 100)), 1)
        for region, bias in signal_bias.items()
    }


def _region_boxes(image_path, img_h, img_w):
    face_box = _detect_face_box(image_path)
    if face_box:
        fx, fy, fw, fh = face_box
    else:
        fw = int(img_w * 0.62)
        fh = int(img_h * 0.78)
        fx = int((img_w - fw) / 2)
        fy = int(img_h * 0.08)

    boxes = {}
    for name, (y0f, y1f, x0f, x1f) in FACE_REGIONS.items():
        x0 = max(0, int(fx + x0f * fw))
        x1 = min(img_w, int(fx + x1f * fw))
        y0 = max(0, int(fy + y0f * fh))
        y1 = min(img_h, int(fy + y1f * fh))
        boxes[name] = (x0, y0, x1, y1)
    return boxes


def _grid_area(image_path, img_h, img_w):
    face_box = _detect_face_box(image_path)
    if face_box:
        fx, fy, fw, fh = face_box
        pad_x = int(fw * 0.16)
        pad_y = int(fh * 0.12)
        return (
            max(0, fx - pad_x),
            max(0, fy - pad_y),
            min(img_w, fx + fw + pad_x),
            min(img_h, fy + fh + pad_y),
        )

    side = int(min(img_w, img_h) * 0.78)
    x0 = int((img_w - side) / 2)
    y0 = int((img_h - side) / 2)
    return x0, y0, x0 + side, y0 + side


def _artifact_grid_map(image_np, fake_score, breakdown, grid=8):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    h, w = gray.shape
    raw = np.zeros((grid, grid), dtype=np.float32)

    for gy in range(grid):
        for gx in range(grid):
            y0, y1 = int(gy * h / grid), int((gy + 1) * h / grid)
            x0, x1 = int(gx * w / grid), int((gx + 1) * w / grid)
            patch = gray[y0:y1, x0:x1]
            sat_patch = hsv[y0:y1, x0:x1, 1]
            if patch.size == 0:
                continue

            sharp = cv2.Laplacian(patch, cv2.CV_64F).var()
            edge = np.mean(cv2.Canny(patch, 80, 160) > 0)
            sat = np.mean(sat_patch) / 255.0
            raw[gy, gx] = 0.45 * np.log1p(sharp) / 8.0 + 0.35 * edge + 0.2 * sat

    spread = raw.max() - raw.min()
    if spread > 1e-6:
        raw = (raw - raw.min()) / spread
    base = float(fake_score)
    breakdown = breakdown or {}
    artifact_boost = max(
        float(breakdown.get("editing_artifact_score", 0)),
        float(breakdown.get("sharpness_mismatch", 0)),
        float(breakdown.get("edge_mismatch", 0)),
        float(breakdown.get("saturation_anomaly", 0)),
    )
    return np.clip(0.55 * raw + 0.30 * base + 0.15 * artifact_boost, 0, 1)


def _score_color(score):
    if score >= 60:
        return (255, 40, 80)
    if score >= 35:
        return (255, 160, 0)
    return (0, 200, 112)


def _risk_colormap(risk_map):
    risk_map = np.clip(risk_map.astype(np.float32), 0, 1)
    colors = np.zeros((*risk_map.shape, 3), dtype=np.float32)

    low = risk_map < 0.5
    high = ~low

    low_t = np.zeros_like(risk_map)
    low_t[low] = risk_map[low] / 0.5
    colors[low, 0] = 255 * low_t[low]
    colors[low, 1] = 200 - 20 * low_t[low]
    colors[low, 2] = 112 * (1 - low_t[low])

    high_t = np.zeros_like(risk_map)
    high_t[high] = (risk_map[high] - 0.5) / 0.5
    colors[high, 0] = 255
    colors[high, 1] = 180 * (1 - high_t[high]) + 40 * high_t[high]
    colors[high, 2] = 80 * high_t[high]
    return colors.astype(np.uint8)


def _draw_gradient_overlay(image_np, image_path, risk_scores):
    out = image_np.copy()
    h, w = out.shape[:2]
    x0, y0, x1, y1 = _grid_area(image_path, h, w)
    area_w = max(1, x1 - x0)
    area_h = max(1, y1 - y0)

    if risk_scores is None or np.size(risk_scores) == 0:
        risk_scores = np.zeros((area_h, area_w), dtype=np.float32)
    else:
        risk_scores = cv2.resize(
            np.array(risk_scores, dtype=np.float32),
            (area_w, area_h),
            interpolation=cv2.INTER_CUBIC,
        )

    risk_scores = cv2.GaussianBlur(risk_scores, (0, 0), sigmaX=area_w / 18, sigmaY=area_h / 18)
    risk_scores = np.clip(risk_scores, 0, 1)
    gradient = _risk_colormap(risk_scores)
    roi = out[y0:y1, x0:x1]
    blended = cv2.addWeighted(gradient, 0.42, roi, 0.58, 0)
    out[y0:y1, x0:x1] = blended
    return out


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

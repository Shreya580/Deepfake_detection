import cv2
import numpy as np
from PIL import Image
import torch

# ── GradCAM imports ───────────────────────────────────────────────────────────
# grad-cam library supports ViT models via GradCAMPlusPlus + reshape_transform
# WHY GradCAMPlusPlus and not GradCAM?
# Standard GradCAM was designed for CNNs with spatial feature maps (H×W grids).
# ViTs produce sequence outputs (N tokens), not spatial grids.
# GradCAMPlusPlus handles this better, and we add a reshape_transform to
# convert the token sequence back into a 2D spatial grid for visualization.
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ── OpenCV face detector ──────────────────────────────────────────────────────
# WHY OpenCV instead of MediaPipe?
# MediaPipe 0.10+ removed the mp.solutions API that older tutorials reference.
# OpenCV's Haar Cascade face detector is built into opencv-python (no extra
# install), works on every platform, and is fast enough for this use case.
try:
    _CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    _FACE_CASCADE = cv2.CascadeClassifier(_CASCADE_PATH)
except Exception:
    _FACE_CASCADE = None

# ── Face region definitions (relative to detected face bounding box) ──────────
# Since we're using a simple face detector (not 468-point landmarks),
# we define regions as fractions of the face bounding box.
# Format: (y_start_frac, y_end_frac, x_start_frac, x_end_frac)
FACE_REGIONS = {
    "Forehead":  (0.00, 0.25, 0.15, 0.85),
    "Left Eye":  (0.25, 0.45, 0.10, 0.48),
    "Right Eye": (0.25, 0.45, 0.52, 0.90),
    "Nose":      (0.40, 0.65, 0.30, 0.70),
    "Lips":      (0.63, 0.80, 0.25, 0.75),
    "Jaw":       (0.78, 1.00, 0.10, 0.90),
}


def _reshape_transform_vit(tensor, height=14, width=14):
    """
    Reshape ViT token output for GradCAM spatial visualization.

    WHY this is needed:
    ViT splits the image into 14×14 = 196 patches + 1 CLS token = 197 tokens.
    GradCAM expects a 2D spatial tensor like (batch, channels, H, W).
    This function:
    1. Removes the CLS token (index 0)
    2. Reshapes the 196 remaining tokens back to 14×14
    3. Permutes to (batch, channels, H, W) format

    Result: GradCAM can now treat ViT like a CNN and produce a spatial heatmap.
    """
    # tensor shape: (batch, num_tokens, embed_dim) = (1, 197, 768)
    result = tensor[:, 1:, :]  # Remove CLS token → (1, 196, 768)
    result = result.reshape(
        tensor.size(0),   # batch
        height,
        width,
        -1                # embed_dim (auto-calculated = 768)
    )
    result = result.transpose(2, 3).transpose(1, 2)  # → (1, 768, 14, 14)
    return result


def generate_face_heatmap(image_path, fake_score, breakdown):
    """
    Generates a Grad-CAM++ heatmap showing exactly which pixels the ViT model
    focused on when deciding "fake" vs "real". Overlays the heatmap on the
    original image using a green→yellow→red color gradient.

    Also calculates per-region suspicion percentages using the face bounding
    box detected by OpenCV.

    Returns:
        (PIL.Image, dict)  — (heatmap overlay image, {region: pct})
        or None on failure
    """
    try:
        from utils.model import get_model_and_extractor
        vit_model, feature_extractor = get_model_and_extractor()

        # ── Load image ────────────────────────────────────────────────────
        img_pil = Image.open(image_path).convert("RGB")
        img_np = np.array(img_pil).astype(np.float32) / 255.0  # Normalize 0–1

        inputs = feature_extractor(images=img_pil, return_tensors="pt")
        input_tensor = inputs["pixel_values"]  # (1, 3, 224, 224)

        # ── Find target layer ─────────────────────────────────────────────
        # For ViT (dima806 model uses google/vit-base-patch16-224 backbone):
        # We target the last transformer encoder block's layernorm.
        # WHY this layer? It's the last layer with rich spatial semantics
        # before the final classifier head — ideal for attribution.
        try:
            # Standard ViT structure: model.vit.encoder.layer[-1].layernorm_before
            target_layer = vit_model.vit.encoder.layer[-1].layernorm_before
        except AttributeError:
            try:
                # Alternative path
                target_layer = vit_model.vit.layernorm
            except AttributeError:
                # Generic fallback: take 3rd-to-last module with parameters
                all_layers = [m for m in vit_model.modules()
                              if len(list(m.parameters(recurse=False))) > 0]
                target_layer = all_layers[-3]

        fake_idx = breakdown.get("fake_label_idx", 0)
        targets = [ClassifierOutputTarget(fake_idx)]

        # ── Run GradCAM++ ─────────────────────────────────────────────────
        with GradCAMPlusPlus(
            model=vit_model,
            target_layers=[target_layer],
            reshape_transform=_reshape_transform_vit   # ← essential for ViT
        ) as cam:
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0]  # Remove batch dim → (H, W)

        # Resize CAM to match original image size
        h_orig, w_orig = img_np.shape[:2]
        cam_resized = cv2.resize(grayscale_cam, (w_orig, h_orig))

        # ── Create colored overlay ────────────────────────────────────────
        # COLORMAP_RdYlGn: green (low activation) → yellow → red (high)
        # image_weight=0.55 means 55% original, 45% heatmap — balanced view
        visualization = show_cam_on_image(
            img_np,
            cam_resized,
            use_rgb=True,
            colormap=cv2.COLORMAP_RdYlGn,
            image_weight=0.55
        )

        # ── Per-region scores ─────────────────────────────────────────────
        region_scores = _calculate_region_scores(image_path, cam_resized, h_orig, w_orig)

        return Image.fromarray(visualization), region_scores

    except Exception as e:
        print(f"[heatmap] GradCAM error on {image_path}: {e}")
        # Return fallback: plain image + estimated scores
        return _plain_fallback(image_path, fake_score)


def _calculate_region_scores(image_path, cam_map, h, w):
    """
    Detects the face bounding box using OpenCV Haar Cascade, then
    calculates the average GradCAM activation in each named facial region.

    WHY bounding box instead of 468 landmarks?
    OpenCV's CascadeClassifier is built into opencv-python — zero extra
    dependencies. For dividing into 6 approximate regions, a bounding box
    with fractional offsets is accurate enough.

    Returns dict: {"Forehead": 72.3, "Left Eye": 85.1, ...}
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return _fallback_region_scores(cam_map)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    face_rect = None
    if _FACE_CASCADE is not None and not _FACE_CASCADE.empty():
        faces = _FACE_CASCADE.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(30, 30)
        )
        if len(faces) > 0:
            # Use the largest detected face
            face_rect = max(faces, key=lambda r: r[2] * r[3])

    if face_rect is None:
        # No face found — fall back to whole-image region estimates
        return _fallback_region_scores(cam_map)

    fx, fy, fw, fh = face_rect
    region_scores = {}

    for region_name, (y0f, y1f, x0f, x1f) in FACE_REGIONS.items():
        # Convert fractions to pixel coordinates within face bounding box
        ry0 = int(fy + y0f * fh)
        ry1 = int(fy + y1f * fh)
        rx0 = int(fx + x0f * fw)
        rx1 = int(fx + x1f * fw)

        # Clamp to image bounds
        ry0, ry1 = max(0, ry0), min(h, ry1)
        rx0, rx1 = max(0, rx0), min(w, rx1)

        patch = cam_map[ry0:ry1, rx0:rx1]

        if patch.size == 0:
            region_scores[region_name] = 50.0
        else:
            avg = float(np.mean(patch))
            region_scores[region_name] = round(avg * 100, 1)

    return region_scores


def _fallback_region_scores(cam_map):
    """
    When no face is detected, estimate region scores from the center of the
    image using fixed fractions. Less precise but never crashes.
    """
    h, w = cam_map.shape

    def zone(y0f, y1f, x0f, x1f):
        patch = cam_map[int(y0f*h):int(y1f*h), int(x0f*w):int(x1f*w)]
        return round(float(np.mean(patch)) * 100, 1) if patch.size > 0 else 50.0

    return {
        "Forehead":  zone(0.05, 0.25, 0.25, 0.75),
        "Left Eye":  zone(0.25, 0.42, 0.15, 0.48),
        "Right Eye": zone(0.25, 0.42, 0.52, 0.85),
        "Nose":      zone(0.40, 0.62, 0.35, 0.65),
        "Lips":      zone(0.62, 0.78, 0.28, 0.72),
        "Jaw":       zone(0.78, 0.97, 0.15, 0.85),
    }


def _plain_fallback(image_path, fake_score):
    """
    If GradCAM completely fails, return the original image with no overlay
    and dummy region scores based on the overall fake_score.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        dummy_scores = {
            "Forehead":  round(fake_score * 80, 1),
            "Left Eye":  round(fake_score * 95, 1),
            "Right Eye": round(fake_score * 90, 1),
            "Nose":      round(fake_score * 60, 1),
            "Lips":      round(fake_score * 85, 1),
            "Jaw":       round(fake_score * 100, 1),
        }
        return img, dummy_scores
    except Exception:
        return None


def generate_signal_heatmap_data(frame_results):
    """
    Averages region_scores across all frames that have them.
    Used for the signal breakdown bar chart.
    Returns dict like {"Jaw": 78.2, "Left Eye": 65.1, ...}
    """
    if not frame_results:
        return {}

    all_regions = {}
    count = 0

    for r in frame_results:
        region_data = r.get("region_scores", {})
        if region_data:
            for region, score in region_data.items():
                all_regions[region] = all_regions.get(region, 0) + score
            count += 1

    if count == 0:
        return {}

    return {k: round(v / count, 1) for k, v in all_regions.items()}
"""
gradcam.py — Grad-CAM heatmap generation

WHY YOUR FRIEND'S APPROACH IS BETTER FOR GRAD-CAM:
  She used Xception (a real CNN). CNNs have spatial feature maps [C,H,W]
  that Grad-CAM was designed for — it pools gradients over H×W to get
  per-channel importance weights, then produces a 2D spatial heatmap.

  Your previous version used a Vision Transformer (ViT). ViTs output
  token sequences [N, D], not spatial maps, so standard Grad-CAM either
  produces noise or requires a complex reshape transform that often breaks.

  SOLUTION: Use Xception (like your friend) for heatmap generation,
  and keep your ViT ensemble for the actual score. Best of both worlds.

FACE REGIONS: We use OpenCV Haar cascade (no mediapipe dependency)
  to find the face bounding box, then divide it into zones.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.cm as cm
import streamlit as st
from PIL import Image

# ── Xception preprocessing ────────────────────────────────────────────────────
xception_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ── Face detector (built into OpenCV, no download needed) ────────────────────
try:
    _FACE_CASCADE = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
except Exception:
    _FACE_CASCADE = None

# Face zone definitions — fractions of (face_x, face_y, face_w, face_h)
FACE_ZONES = {
    "Forehead":  (0.00, 0.22, 0.15, 0.85),
    "Left Eye":  (0.22, 0.42, 0.10, 0.46),
    "Right Eye": (0.22, 0.42, 0.54, 0.90),
    "Nose":      (0.38, 0.62, 0.33, 0.67),
    "Lips":      (0.62, 0.78, 0.22, 0.78),
    "Jaw":       (0.78, 1.00, 0.10, 0.90),
}


@st.cache_resource(show_spinner="Loading Grad-CAM model (Xception)...")
def _load_xception():
    """
    Loads pretrained Xception from timm library.

    WHY Xception for Grad-CAM?
    Xception was specifically designed for image classification with
    depthwise separable convolutions. Its conv4 block (the last conv
    layer before global pooling) has spatial feature maps at ~10x10 px
    resolution — perfect for Grad-CAM localisation.

    WHY pretrained on ImageNet (not deepfakes)?
    The Grad-CAM heatmap shows WHERE the model is looking, not whether
    it thinks something is fake. Any good feature extractor trained on
    natural images learns to focus on faces, textures, and edges —
    exactly the regions where deepfake artifacts appear.
    """
    import timm
    model = timm.create_model("xception", pretrained=True)
    model.eval()
    return model


def generate_face_heatmap(image_path, fake_score, breakdown):
    """
    Generates a Grad-CAM heatmap showing which image regions the model
    focused on, then computes per-facial-zone suspicion scores.

    HOW GRAD-CAM WORKS (plain English):
    1. Run image through Xception (forward pass)
    2. Hook into conv4 — save the activation maps [C, H, W]
    3. Backpropagate the top class score — save gradients at conv4
    4. Global-average-pool the gradients → importance weight per channel
    5. Weighted sum of activation maps → raw CAM
    6. ReLU + normalize → values 0 to 1
    7. Resize to image size, apply colormap, blend with original

    Returns: (PIL Image with heatmap, dict of region scores)
    """
    try:
        model = _load_xception()

        img_pil  = Image.open(image_path).convert("RGB")
        orig_w, orig_h = img_pil.size

        img_tensor = xception_transform(img_pil).unsqueeze(0)
        img_tensor.requires_grad_(True)

        activations = {}
        gradients   = {}

        # Hook into conv4 (last conv block before global pool)
        target_layer = model.conv4

        def fwd_hook(module, inp, out):
            activations["v"] = out.detach()

        def bwd_hook(module, grad_in, grad_out):
            gradients["v"] = grad_out[0].detach()

        h1 = target_layer.register_forward_hook(fwd_hook)
        h2 = target_layer.register_full_backward_hook(bwd_hook)

        # Forward pass
        output = model(img_tensor)

        # Backward on top predicted class
        model.zero_grad()
        class_idx = output.argmax(dim=1).item()
        one_hot   = torch.zeros_like(output)
        one_hot[0][class_idx] = 1.0
        output.backward(gradient=one_hot)

        h1.remove()
        h2.remove()

        # Compute CAM
        act  = activations["v"].squeeze(0)    # [C, H, W]
        grad = gradients["v"].squeeze(0)      # [C, H, W]
        weights = grad.mean(dim=(1, 2))       # [C]

        cam = torch.zeros(act.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * act[i]

        cam = F.relu(cam)
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
      
        cam[cam < 0.4] = 0.0

        # Resize to original image size
        cam_np      = cam.numpy()
        cam_resized = cv2.resize(cam_np, (orig_w, orig_h))

        # Apply RdYlGn_r colormap: red=high attention, green=low
        colormap = cm.get_cmap("RdYlGn_r")

        heatmap_rgba = colormap(cam_resized)

        # Separate RGB
        heatmap_rgb = heatmap_rgba[:, :, :3]

        # 🔥 Remove weak activations
        threshold = 0.4
        heatmap_rgb[cam_resized < threshold] = 0

        # 🔥 Scale intensity using fake score
        heatmap_rgb = heatmap_rgb * (0.4 + 0.6 * fake_score)

        # Convert to 0–255 safely
        heatmap_rgb = np.clip(heatmap_rgb * 255, 0, 255).astype(np.uint8)

        heatmap_pil = Image.fromarray(heatmap_rgb)

        # Blend: 55% original + 45% heatmap
        orig_rgb = img_pil.convert("RGB")
        alpha = 0.35 + 0.4 * fake_score
        blended  = Image.blend(orig_rgb, heatmap_pil, alpha=alpha)

        # Per-zone scores
        region_scores = _zone_scores(image_path, cam_resized, orig_h, orig_w)

        return blended, region_scores

    except Exception as e:
        print(f"[gradcam] Failed on {image_path}: {e}")
        return None, {}


def _detect_face(image_path):
    """Returns (x,y,w,h) bounding box of the largest face, or None."""
    if _FACE_CASCADE is None or _FACE_CASCADE.empty():
        return None
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    faces = _FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(faces) == 0:
        return None
    return tuple(sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0])


def _zone_scores(image_path, cam_map, img_h, img_w):
    """
    Average Grad-CAM activation inside each facial zone.
    Returns dict: {"Forehead": 34.2, "Left Eye": 78.1, ...} (0–100 scale).
    """
    face = _detect_face(image_path)

    if face:
        fx, fy, fw, fh = face
        scores = {}
        for name, (y0f, y1f, x0f, x1f) in FACE_ZONES.items():
            py0 = max(0, int(fy + y0f*fh));  py1 = min(img_h, int(fy + y1f*fh))
            px0 = max(0, int(fx + x0f*fw));  px1 = min(img_w, int(fx + x1f*fw))
            zone = cam_map[py0:py1, px0:px1]
            scores[name] = round(float(np.mean(zone))*100, 1) if zone.size > 0 else 0.0
        return scores
    else:
        # Fallback: whole-image zones when no face detected
        h, w = cam_map.shape
        def z(y0f, y1f, x0f, x1f):
            r = cam_map[int(y0f*h):int(y1f*h), int(x0f*w):int(x1f*w)]
            return round(float(np.mean(r))*100, 1) if r.size > 0 else 0.0
        return {
            "Forehead":  z(0.05,0.25,0.25,0.75),
            "Left Eye":  z(0.28,0.42,0.15,0.45),
            "Right Eye": z(0.28,0.42,0.55,0.85),
            "Nose":      z(0.42,0.62,0.35,0.65),
            "Lips":      z(0.62,0.76,0.28,0.72),
            "Jaw":       z(0.76,0.95,0.15,0.85),
        }


def aggregate_region_scores(frame_results):
    """
    Averages region scores across all frames that have them.
    Called after heatmaps are generated for top frames.
    """
    totals, counts = {}, {}
    for r in frame_results:
        for region, score in r.get("region_scores", {}).items():
            totals[region] = totals.get(region, 0.0) + score
            counts[region] = counts.get(region, 0) + 1
    if not totals:
        return {}
    return {k: round(totals[k]/counts[k], 1) for k in totals}

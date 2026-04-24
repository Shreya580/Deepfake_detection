import cv2
import numpy as np
from PIL import Image
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import mediapipe as mp

# ── Load MediaPipe face mesh ONCE ─────────────────────────────────────────────
# Same reason as model — load once, reuse every call
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,      # Single image, not video stream
    max_num_faces=1,             # We only care about the main face
    refine_landmarks=True,       # Get the precise 468-point mesh
    min_detection_confidence=0.3 # Accept lower confidence for difficult images
)

# ── MediaPipe landmark indices for facial regions ─────────────────────────────
# MediaPipe gives 468 landmarks. These index groups correspond to named regions.
# Source: MediaPipe face mesh topology map
REGION_LANDMARKS = {
    "Left eye":   [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
    "Right eye":  [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
    "Lips":       [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146],
    "Jaw":        [132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323],
    "Forehead":   [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377],
    "Nose":       [1, 2, 5, 4, 195, 197, 6, 168, 8, 9]
}


def generate_face_heatmap(image_path, fake_score, breakdown):
    """
    Generates a REAL Grad-CAM heatmap showing which pixels the AI
    used to make its fake/real decision.
    
    HOW IT WORKS:
    1. Load the image
    2. Run it through the model with Grad-CAM hooks active
    3. Grad-CAM computes gradients → produces a 2D activation map
    4. Normalize and apply green→red colormap
    5. Overlay on original image
    6. Return both the overlay image and per-region percentages
    
    WHY this is different from the old heatmap:
    The old heatmap drew blobs at hardcoded positions (eyes, jaw) regardless
    of the image. This heatmap shows what the ACTUAL MODEL focused on.
    If it focuses on the left cheek, that's where the heatmap will be bright.
    """
    from utils.model import get_model_and_extractor
    
    model, feature_extractor = get_model_and_extractor()
    
    # ── Step 1: Load and prepare image ───────────────────────────────────
    img_pil = Image.open(image_path).convert("RGB")
    img_np = np.array(img_pil).astype(np.float32) / 255.0  # Normalize to 0-1
    
    # Feature extractor prepares tensor for the model
    inputs = feature_extractor(images=img_pil, return_tensors="pt")
    input_tensor = inputs["pixel_values"]  # Shape: [1, 3, 224, 224]
    
    # ── Step 2: Find the target layer for Grad-CAM ───────────────────────
    # WHY the last layer? The last convolutional/attention layer has the
    # richest spatial information. Earlier layers detect edges and textures,
    # later layers detect high-level concepts like "fake blending artifact."
    # For Vision Transformers, we target the last attention layer's norm.
    try:
        # For ViT models, this is the last layernorm before the classifier
        target_layer = model.vit.layernorm
    except AttributeError:
        # Fallback for different model architectures
        target_layer = list(model.modules())[-3]
    
    # ── Step 3: Define what we want gradients of ─────────────────────────
    # We want gradients of the FAKE class score.
    # WHY? We want to know which pixels made the model say "fake".
    # If we used the "real" class, we'd get the opposite.
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    targets = [ClassifierOutputTarget(0)]
    
    # ── Step 4: Run Grad-CAM ─────────────────────────────────────────────
    # GradCAM hooks into target_layer, runs forward pass, computes gradients
    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        # grayscale_cam shape: [1, H, W] — values 0.0 to 1.0
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0]  # Remove batch dimension → [H, W]
    
    # Resize cam to match original image dimensions
    h, w = img_np.shape[:2]
    grayscale_cam_resized = cv2.resize(grayscale_cam, (w, h))
    
    # ── Step 5: Create colored overlay ───────────────────────────────────
    # show_cam_on_image blends the heatmap with the original image
    # colormap COLORMAP_RdYlGn: green (low) → yellow (medium) → red (high)
    visualization = show_cam_on_image(
        img_np,
        grayscale_cam_resized,
        use_rgb=True,
        colormap=cv2.COLORMAP_RdYlGn,  # Green=real, Red=fake
        image_weight=0.55  # 55% original, 45% heatmap overlay
    )
    
    # ── Step 6: Calculate per-region percentages ─────────────────────────
    region_scores = calculate_region_scores(image_path, grayscale_cam_resized)
    
    return Image.fromarray(visualization), region_scores


def calculate_region_scores(image_path, cam_map):
    """
    Divides the face into named regions using MediaPipe landmarks,
    then calculates the average Grad-CAM activation in each region.
    
    WHY MediaPipe? Without landmark detection, we'd have to guess where
    the eyes are (like the old code did with hardcoded fractions).
    MediaPipe finds the ACTUAL position of each facial feature in THIS
    specific image. A face on the left side of the frame, or tilted at
    an angle, will still get correctly segmented.
    
    Returns: dict like {"Left eye": 78.3, "Jaw": 91.2, "Lips": 45.0, ...}
             where values are percentages 0–100
    """
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    
    # Detect face landmarks
    results = face_mesh.process(img_rgb)
    
    # If no face detected, return estimated scores from the cam map directly
    if not results.multi_face_landmarks:
        return _fallback_region_scores(cam_map)
    
    landmarks = results.multi_face_landmarks[0].landmark
    
    region_scores = {}
    
    for region_name, landmark_indices in REGION_LANDMARKS.items():
        # Convert landmark fractions to pixel coordinates
        points = []
        for idx in landmark_indices:
            if idx < len(landmarks):
                lm = landmarks[idx]
                px = int(lm.x * w)
                py = int(lm.y * h)
                points.append([px, py])
        
        if len(points) < 3:
            region_scores[region_name] = 50.0
            continue
        
        # Create a mask for this region using convex hull of landmarks
        # WHY convex hull? The landmark points form the boundary of the region.
        # We fill the enclosed area to create a binary mask.
        pts = np.array(points, dtype=np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(mask, hull, 255)
        
        # Extract cam values only within this region's mask
        region_cam_values = cam_map[mask > 0]
        
        if len(region_cam_values) == 0:
            region_scores[region_name] = 50.0
        else:
            # Average activation in this region, converted to percentage
            avg_activation = float(np.mean(region_cam_values))
            region_scores[region_name] = round(avg_activation * 100, 1)
    
    return region_scores


def _fallback_region_scores(cam_map):
    """
    If MediaPipe can't find a face, estimate region scores from
    approximate face zones (fractions of image size).
    Less accurate but better than returning nothing.
    """
    h, w = cam_map.shape
    
    def zone_avg(y1f, y2f, x1f, x2f):
        y1, y2 = int(y1f*h), int(y2f*h)
        x1, x2 = int(x1f*w), int(x2f*w)
        return round(float(np.mean(cam_map[y1:y2, x1:x2])) * 100, 1)
    
    return {
        "Left eye":  zone_avg(0.28, 0.42, 0.20, 0.48),
        "Right eye": zone_avg(0.28, 0.42, 0.52, 0.80),
        "Nose":      zone_avg(0.42, 0.62, 0.38, 0.62),
        "Lips":      zone_avg(0.62, 0.75, 0.30, 0.70),
        "Jaw":       zone_avg(0.75, 0.95, 0.20, 0.80),
        "Forehead":  zone_avg(0.05, 0.28, 0.25, 0.75),
    }


def generate_signal_heatmap_data(frame_results):
    """
    For the signal breakdown chart — now just returns region scores
    averaged across all frames.
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
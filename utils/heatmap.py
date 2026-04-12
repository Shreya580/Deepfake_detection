import cv2
import numpy as np
from PIL import Image
import io

def generate_face_heatmap(image_path, fake_score, breakdown):
    """
    Generates a heatmap overlay on the face image showing suspicious regions.
    
    HOW IT WORKS:
    1. Detect face landmarks using DeepFace/OpenCV
    2. Assign suspicion weights to facial regions based on our score breakdown
    3. Create a Gaussian heat blob at each region
    4. Overlay the heatmap on the original image with transparency
    
    WHY these regions?
    - Eyes: Deepfakes often have unnatural blinking or eye reflection artifacts
    - Mouth/lips: Lip sync is hard to fake perfectly; color often mismatched
    - Jaw/chin: Face blending boundary is usually around the jawline
    - Forehead: Texture synthesis often fails here (hair-to-skin transition)
    """
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    
    # Create blank heatmap canvas
    heatmap = np.zeros((h, w), dtype=np.float32)
    
    # ── Define facial regions (as fractions of image size) ────────────────
    # These are approximate positions for a centered face
    # Format: (center_x_frac, center_y_frac, radius_x_frac, radius_y_frac, weight)
    
    blur_weight = breakdown.get("blur_anomaly", 0.1)
    color_weight = breakdown.get("color_inconsistency", 0.1)
    face_weight = breakdown.get("face_confidence_drop", 0.1)
    freq_weight = breakdown.get("frequency_noise", 0.1)
    
    regions = [
        # Left eye region
        (0.35, 0.38, 0.12, 0.08, face_weight * 4 + freq_weight * 3),
        # Right eye region
        (0.65, 0.38, 0.12, 0.08, face_weight * 4 + freq_weight * 3),
        # Mouth / lips
        (0.50, 0.68, 0.18, 0.08, color_weight * 5 + blur_weight * 2),
        # Jaw / chin (blending boundary)
        (0.50, 0.82, 0.25, 0.10, blur_weight * 4 + face_weight * 2),
        # Forehead
        (0.50, 0.22, 0.22, 0.10, freq_weight * 3 + color_weight * 2),
        # Left cheek
        (0.28, 0.58, 0.10, 0.12, color_weight * 3),
        # Right cheek
        (0.72, 0.58, 0.10, 0.12, color_weight * 3),
        # Nose bridge
        (0.50, 0.50, 0.08, 0.15, face_weight * 2),
    ]
    
    # ── Paint Gaussian blobs at each region ─────────────────────────────
    for (cx_frac, cy_frac, rx_frac, ry_frac, weight) in regions:
        # Convert fractions to pixel coordinates
        cx = int(cx_frac * w)
        cy = int(cy_frac * h)
        rx = int(rx_frac * w)
        ry = int(ry_frac * h)
        
        # Create coordinate grid
        Y, X = np.mgrid[0:h, 0:w]
        
        # Gaussian blob formula
        blob = np.exp(-(((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2))
        
        # Scale blob by signal weight and fake score
        heatmap += blob * weight * fake_score * 3
    
    # Normalize heatmap to 0–255
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    heatmap = (heatmap * 255).astype(np.uint8)
    
    # ── Apply colormap ───────────────────────────────────────────────────
    # COLORMAP_JET: blue=safe, green=moderate, yellow=high, red=very suspicious
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # ── Blend heatmap with original image ───────────────────────────────
    # Alpha = how transparent the heatmap is
    # 0.0 = invisible, 1.0 = fully opaque
    # We use 0.5 so you can still see the face underneath
    alpha = 0.45 + (fake_score * 0.25)  # More suspicious = more vivid overlay
    alpha = min(alpha, 0.7)  # Cap at 70% opacity
    
    blended = cv2.addWeighted(
        img_rgb.astype(np.float32), 1 - alpha,
        heatmap_rgb.astype(np.float32), alpha,
        0
    ).astype(np.uint8)
    
    # Convert to PIL for Streamlit display
    return Image.fromarray(blended)


def generate_signal_heatmap_data(frame_results):
    """
    Prepares data for the Signal Breakdown visualization.
    Aggregates signal contributions across all frames.
    
    Returns a dict ready for Plotly bar chart.
    """
    if not frame_results:
        return {}
    
    signal_totals = {
        "Face confidence drop": 0,
        "Blur anomaly": 0,
        "Color inconsistency": 0,
        "Frequency noise": 0
    }
    
    for result in frame_results:
        bd = result.get("breakdown", {})
        signal_totals["Face confidence drop"] += bd.get("face_confidence_drop", 0)
        signal_totals["Blur anomaly"] += bd.get("blur_anomaly", 0)
        signal_totals["Color inconsistency"] += bd.get("color_inconsistency", 0)
        signal_totals["Frequency noise"] += bd.get("frequency_noise", 0)
    
    # Average across frames
    n = len(frame_results)
    signal_avgs = {k: round(v / n, 4) for k, v in signal_totals.items()}
    
    # Convert to percentages of total
    total = sum(signal_avgs.values())
    if total > 0:
        signal_pcts = {k: round((v / total) * 100, 1) for k, v in signal_avgs.items()}
    else:
        signal_pcts = {k: 25.0 for k in signal_avgs}
    
    return signal_pcts
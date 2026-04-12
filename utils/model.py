import cv2
import numpy as np
from deepface import DeepFace
import os

# ─── SIGNAL 1: Face Detection Confidence ───────────────────────────────────
def get_face_confidence(image_path):
    """
    DeepFace detects faces and returns how confident it is.
    
    WHY this matters for deepfakes:
    Deepfake faces often have slightly wrong geometry — the face
    detection model will be less confident about detecting them cleanly.
    A real face: confidence ~0.98+
    A deepfake face: may drop to 0.7–0.85 due to blending artifacts.
    
    Returns: confidence float 0.0–1.0
    """
    try:
        result = DeepFace.extract_faces(
            img_path=image_path,
            detector_backend="opencv",  # Fastest, works on CPU
            enforce_detection=False      # Don't crash if no face found
        )
        
        if result and len(result) > 0:
            # DeepFace returns confidence as part of face region info
            confidence = result[0].get("confidence", 0.5)
            return float(confidence)
        else:
            return 0.0  # No face = suspicious
            
    except Exception:
        return 0.3  # Error = assume somewhat suspicious


# ─── SIGNAL 2: Blur / Sharpness Score ──────────────────────────────────────
def get_blur_score(image_path):
    """
    Measures image sharpness using Laplacian variance.
    
    WHY this catches deepfakes:
    Deepfake face-swapping algorithms often leave the face region
    slightly blurrier than the background. The face is "pasted in"
    and the blending creates softness at edges — especially around
    the jaw, hairline, and ears.
    
    How it works:
    Laplacian operator detects edges. High variance = sharp edges = real.
    Low variance = blurry = suspicious.
    
    Returns: anomaly score 0.0–1.0 (higher = more blurry = more suspicious)
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        return 0.5
    
    # Apply Laplacian (edge detector)
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    variance = laplacian.var()
    
    # Normalize: typical real face variance is 100–800
    # Very blurry (deepfake) might be 20–80
    # We invert so high score = suspicious
    normalized = np.clip(variance / 500.0, 0, 1)
    blur_anomaly = 1.0 - normalized  # Invert: low sharpness = high anomaly
    
    return float(blur_anomaly)


# ─── SIGNAL 3: Frequency Domain Noise ──────────────────────────────────────
def get_frequency_noise(image_path):
    """
    Analyzes high-frequency content using Fast Fourier Transform (FFT).
    
    WHY this catches deepfakes:
    AI-generated faces have unusual frequency patterns.
    Real photos have natural high-frequency content (textures, pores).
    GAN-generated faces often show grid-like artifacts in frequency space
    — called "GAN fingerprints" — invisible to the eye but detectable mathematically.
    
    Returns: noise score 0.0–1.0 (higher = more artificial frequency pattern)
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        return 0.5
    
    # Apply FFT to get frequency domain
    f = np.fft.fft2(img.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    
    # Look at high-frequency energy (outer ring of FFT)
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    
    # Create masks for center (low freq) and edges (high freq)
    Y, X = np.ogrid[:h, :w]
    distance = np.sqrt((X - cx)**2 + (Y - cy)**2)
    
    low_freq_mask = distance < 30   # Inner circle
    high_freq_mask = distance > 80  # Outer ring
    
    low_energy = magnitude[low_freq_mask].mean()
    high_energy = magnitude[high_freq_mask].mean()
    
    # Real images: high frequency has natural texture energy
    # Deepfakes: unusual ratio between low and high frequency
    if low_energy == 0:
        return 0.5
    
    ratio = high_energy / low_energy
    
    # Normalize (typical real ratio: 0.1–0.4)
    # Anomalous ratio (too high or too low) = suspicious
    normal_ratio = 0.25
    deviation = abs(ratio - normal_ratio) / normal_ratio
    score = np.clip(deviation, 0, 1)
    
    return float(score)


# ─── SIGNAL 4: Color Inconsistency ─────────────────────────────────────────
def get_color_inconsistency(image_path):
    """
    Compares color histogram of the face region vs background.
    
    WHY this catches deepfakes:
    When a face is swapped onto a body, the color grading (lighting,
    white balance, skin tone adaptation) is rarely perfect. The face
    region will have slightly different color distribution than the
    surrounding area.
    
    Returns: inconsistency score 0.0–1.0
    """
    img = cv2.imread(image_path)
    
    if img is None:
        return 0.5
    
    h, w = img.shape[:2]
    
    # Define face region (center ~50% of image) and background (border)
    center_y1, center_y2 = h // 4, 3 * h // 4
    center_x1, center_x2 = w // 4, 3 * w // 4
    
    face_region = img[center_y1:center_y2, center_x1:center_x2]
    
    # Background: combine top, bottom, left, right strips
    bg_parts = [
        img[:center_y1, :],
        img[center_y2:, :],
        img[:, :center_x1],
        img[:, center_x2:]
    ]
    background = np.concatenate([p.reshape(-1, 3) for p in bg_parts if p.size > 0])
    
    # Calculate color histograms for each channel (BGR)
    inconsistency_scores = []
    
    for channel in range(3):
        face_hist = np.histogram(face_region[:,:,channel].flatten(), bins=32, range=(0,256))[0]
        bg_hist = np.histogram(background[:,channel], bins=32, range=(0,256))[0]
        
        # Normalize histograms
        face_hist = face_hist / (face_hist.sum() + 1e-8)
        bg_hist = bg_hist / (bg_hist.sum() + 1e-8)
        
        # Chi-squared distance between histograms
        diff = face_hist - bg_hist
        chi_sq = np.sum(diff**2 / (bg_hist + 1e-8))
        
        inconsistency_scores.append(chi_sq)
    
    # Average across channels, normalize to 0–1
    avg_inconsistency = np.mean(inconsistency_scores)
    score = np.clip(avg_inconsistency / 10.0, 0, 1)
    
    return float(score)


# ─── COMBINED SCORER ────────────────────────────────────────────────────────
def score_frame(image_path):
    """
    Main function: combines all 4 signals into one fake probability score.
    
    Weights (must sum to 1.0):
    - face_confidence: 0.35  (most reliable signal)
    - blur_anomaly:    0.30  (second most reliable)
    - color_inconsist: 0.20  (medium reliability)
    - freq_noise:      0.15  (supporting signal)
    
    WHY these weights?
    Face confidence and blur are the two strongest indicators of a deepfake.
    Color and frequency analysis support but can have false positives
    (e.g. bad lighting in real videos can look like color inconsistency).
    
    Returns: dict with overall score and breakdown
    """
    face_conf = get_face_confidence(image_path)
    blur = get_blur_score(image_path)
    color = get_color_inconsistency(image_path)
    freq = get_frequency_noise(image_path)
    
    # Face confidence is INVERTED: high confidence = real = low fake score
    face_anomaly = 1.0 - face_conf
    
    # Weighted combination
    fake_score = (
        0.35 * face_anomaly +
        0.30 * blur +
        0.20 * color +
        0.15 * freq
    )
    
    # Clamp to valid range
    fake_score = float(np.clip(fake_score, 0.0, 1.0))
    
    return {
        "fake_score": round(fake_score, 4),
        "breakdown": {
            "face_confidence_drop": round(face_anomaly * 0.35, 4),
            "blur_anomaly": round(blur * 0.30, 4),
            "color_inconsistency": round(color * 0.20, 4),
            "frequency_noise": round(freq * 0.15, 4)
        },
        "raw_signals": {
            "face_confidence": round(face_conf, 4),
            "blur_score": round(blur, 4),
            "color_score": round(color, 4),
            "freq_score": round(freq, 4)
        }
    }


def analyze_all_frames(frame_paths, progress_callback=None):
    """
    Runs score_frame() on every extracted frame.
    Returns a list of results (one per frame) for the visualization layer.
    
    WHY progress_callback?
    Processing 60 frames takes time. We pass a callback so Streamlit
    can update a progress bar in real-time — users see something is happening.
    """
    results = []
    
    for i, path in enumerate(frame_paths):
        result = score_frame(path)
        result["frame_index"] = i
        result["frame_path"] = path
        result["frame_number"] = i + 1
        results.append(result)
        
        if progress_callback:
            progress_callback(i + 1, len(frame_paths))
    
    return results


def get_overall_verdict(frame_results):
    """
    Computes the final verdict from all frame scores.
    
    WHY weighted toward top 20%?
    If even a few frames show strong manipulation, the video is likely fake.
    A single frame that's 95% fake is more meaningful than 50 frames at 30%.
    We weight the top scores more heavily to catch partial manipulations.
    """
    if not frame_results:
        return {"overall_score": 0.0, "verdict": "Unknown", "color": "gray"}
    
    scores = [r["fake_score"] for r in frame_results]
    scores_sorted = sorted(scores, reverse=True)
    
    # Top 20% of frames
    top_count = max(1, len(scores_sorted) // 5)
    top_scores = scores_sorted[:top_count]
    
    # Weighted average: 60% weight on top frames, 40% on all frames
    top_avg = np.mean(top_scores)
    overall_avg = np.mean(scores)
    overall_score = round(0.6 * top_avg + 0.4 * overall_avg, 4)
    
    # Verdict categories
    if overall_score < 0.35:
        verdict = "Likely Real"
        color = "green"
    elif overall_score < 0.55:
        verdict = "Uncertain"
        color = "orange"
    else:
        verdict = "Likely Deepfake"
        color = "red"
    
    return {
        "overall_score": overall_score,
        "overall_percent": round(overall_score * 100, 1),
        "verdict": verdict,
        "color": color,
        "total_frames": len(scores),
        "fake_frames": sum(1 for s in scores if s > 0.6),
        "uncertain_frames": sum(1 for s in scores if 0.4 <= s <= 0.6),
        "real_frames": sum(1 for s in scores if s < 0.4),
        "max_score": round(max(scores), 4),
        "min_score": round(min(scores), 4)
    }
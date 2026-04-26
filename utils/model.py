import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

try:
    import cv2
except Exception:
    cv2 = None

# ─────────────────────────────────────────────────────────────────────────────
# SPEED FIX: Load models ONCE at module level.
# Loading inside score_frame() = 5s × 60 frames = 5 minutes wasted.
# Loading here = 5s total on startup, then all frames reuse it instantly.
#
# ACCURACY FIX: We use TWO models in an ensemble.
#
# WHY TWO MODELS?
# Model 1 (dima806) was trained on face-swap deepfakes (FaceForensics++).
#   → Good at detecting swapped faces in videos
#   → Weak on fully AI-generated faces (ChatGPT, Midjourney, DALL-E)
#     because those were never in its training data.
#
# Model 2 (haywoodsloan) was trained specifically on AI-generated images
# from diffusion models (Stable Diffusion, DALL-E, Midjourney).
#   → Good at detecting AI-generated content (your ChatGPT face case)
#   → Catches what model 1 misses
#
# Final score = average of both. This covers BOTH types of fakes.
# ─────────────────────────────────────────────────────────────────────────────

MODEL_1_NAME = "dima806/deepfake_vs_real_image_detection"
MODEL_2_NAME = "haywoodsloan/ai-image-detector-deploy"

print("[DeepScan] Loading model 1 (face-swap deepfake detector)...")
extractor1 = AutoImageProcessor.from_pretrained(MODEL_1_NAME)
model1 = AutoModelForImageClassification.from_pretrained(MODEL_1_NAME)
model1.eval()

print("[DeepScan] Loading model 2 (AI-generated image detector)...")
try:
    extractor2 = AutoImageProcessor.from_pretrained(MODEL_2_NAME)
    model2 = AutoModelForImageClassification.from_pretrained(MODEL_2_NAME)
    model2.eval()
    MODEL2_AVAILABLE = True
    print("[DeepScan] Both models loaded successfully.")
except Exception as e:
    print(f"[DeepScan] Model 2 unavailable ({e}). Using model 1 only.")
    MODEL2_AVAILABLE = False

# Find which output index = "fake" for model 1
LABELS1 = model1.config.id2label
FAKE_IDX1 = next(k for k, v in LABELS1.items() if "fake" in v.lower())

# Find fake index for model 2 if available
if MODEL2_AVAILABLE:
    LABELS2 = model2.config.id2label
    # Model 2 may use "artificial" or "fake" or "AI" as the fake label
    fake_keywords = ["fake", "artificial", "ai", "generated", "synthetic"]
    try:
        FAKE_IDX2 = next(
            k for k, v in LABELS2.items()
            if any(kw in v.lower() for kw in fake_keywords)
        )
    except StopIteration:
        # If no obvious fake label, use index 0
        FAKE_IDX2 = 0

print(f"[DeepScan] Model 1 labels: {LABELS1}  | Fake index: {FAKE_IDX1}")
if MODEL2_AVAILABLE:
    print(f"[DeepScan] Model 2 labels: {LABELS2}  | Fake index: {FAKE_IDX2}")


def get_model_and_extractor():
    """
    Returns model1, extractor1, and fake index for Grad-CAM in heatmap.py.
    We use model1 for Grad-CAM because it's a ViT with known layer structure.
    Model2 is only used for scoring, not heatmap generation.
    """
    return model1, extractor1, FAKE_IDX1


def _run_model1(img_pil):
    """Run face-swap deepfake model. Returns fake probability 0-1."""
    inputs = extractor1(images=img_pil, return_tensors="pt")
    with torch.no_grad():
        outputs = model1(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)[0]
    return float(probs[FAKE_IDX1].item())


def _run_model2(img_pil):
    """Run AI-generated image detector. Returns fake probability 0-1."""
    if not MODEL2_AVAILABLE:
        return None
    try:
        inputs = extractor2(images=img_pil, return_tensors="pt")
        with torch.no_grad():
            outputs = model2(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]
        return float(probs[FAKE_IDX2].item())
    except Exception as e:
        print(f"[DeepScan] Model 2 inference error: {e}")
        return None


def _grid_values(arr, fn, grid=4):
    h, w = arr.shape[:2]
    values = []
    for gy in range(grid):
        for gx in range(grid):
            y0, y1 = int(gy * h / grid), int((gy + 1) * h / grid)
            x0, x1 = int(gx * w / grid), int((gx + 1) * w / grid)
            patch = arr[y0:y1, x0:x1]
            if patch.size:
                values.append(float(fn(patch)))
    return np.array(values, dtype=np.float32)


def _estimate_editing_artifacts(img_pil):
    """
    Lightweight forensic score for local edits/composites.
    This is not a replacement for a trained tamper-localization model, but it
    catches cues the deepfake classifiers miss: inconsistent sharpness, heavy
    local saturation, edge-density jumps, and compression/noise mismatch.
    """
    if cv2 is None:
        return 0.0, {}

    img = np.array(img_pil.resize((384, 384))).astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    sharpness = _grid_values(
        gray,
        lambda p: cv2.Laplacian(p, cv2.CV_64F).var(),
        grid=4,
    )
    edges = _grid_values(
        gray,
        lambda p: np.mean(cv2.Canny(p, 80, 160) > 0),
        grid=4,
    )
    saturation = _grid_values(
        hsv[:, :, 1],
        lambda p: np.mean(p) / 255.0,
        grid=4,
    )

    high_sat_pixels = float(np.mean(hsv[:, :, 1] > 185))
    sharp_cv = float(np.std(sharpness) / (np.mean(sharpness) + 1e-6))
    edge_cv = float(np.std(edges) / (np.mean(edges) + 1e-6))
    sat_spread = float(np.percentile(saturation, 90) - np.percentile(saturation, 10))

    sharpness_mismatch = np.clip((sharp_cv - 0.55) / 1.1, 0, 1)
    edge_mismatch = np.clip((edge_cv - 0.45) / 1.0, 0, 1)
    saturation_anomaly = np.clip((sat_spread - 0.18) / 0.42, 0, 1)
    color_outlier = np.clip((high_sat_pixels - 0.08) / 0.32, 0, 1)

    score = float(np.clip(
        0.34 * sharpness_mismatch
        + 0.28 * edge_mismatch
        + 0.22 * saturation_anomaly
        + 0.16 * color_outlier,
        0,
        1,
    ))

    signals = {
        "editing_artifact_score": round(score, 4),
        "sharpness_mismatch": round(float(sharpness_mismatch), 4),
        "edge_mismatch": round(float(edge_mismatch), 4),
        "saturation_anomaly": round(float(saturation_anomaly), 4),
        "color_outlier": round(float(color_outlier), 4),
    }
    return score, signals


def score_frame(image_path):
    """
    Runs BOTH models on a single frame and averages the results.

    WHY average and not take the max?
    Taking the max would make everything look more fake (false positives).
    Averaging is more conservative and more accurate overall.
    If both models agree something is fake, the average will be high.
    If only one model thinks it's fake, the average reflects the uncertainty.

    SPEED NOTE: Both models run with torch.no_grad() which skips gradient
    computation — about 2x faster than a normal forward pass.
    """
    try:
        img_pil = Image.open(image_path).convert("RGB")

        score1 = _run_model1(img_pil)
        score2 = _run_model2(img_pil)

        # Keep the primary fake probability close to Isha's more reliable
        # implementation: the dima806 detector directly predicts face-swap
        # fake probability. The AI-image model is only allowed to add signal
        # when it is strongly confident, so it cannot wash out face-swap scores.
        ai_signal = score2 if score2 is not None and score2 >= 0.65 else 0.0
        final_score = round(max(score1, ai_signal), 4)

        artifact_score, artifact_signals = _estimate_editing_artifacts(img_pil)
        manipulation_score = round(max(
            final_score,
            0.72 * artifact_score + 0.28 * final_score,
        ), 4)

        return {
            "fake_score": manipulation_score,
            "breakdown": {
                "deepfake_model_score": round(score1, 4),
                "ai_generation_model_score": round(score2, 4) if score2 is not None else 0.5,
                **artifact_signals,
            },
            "raw_signals": {
                "model1_fake_prob": round(score1, 4),
                "model2_fake_prob": round(score2, 4) if score2 is not None else 0.5,
                "ensemble_score": final_score,
                "artifact_score": round(artifact_score, 4),
                "manipulation_score": manipulation_score,
            },
        }

    except Exception as e:
        print(f"[DeepScan] score_frame error on {image_path}: {e}")
        return {
            "fake_score": 0.5,
            "breakdown": {"deepfake_model_score": 0.5, "ai_generation_model_score": 0.5},
            "raw_signals": {"model1_fake_prob": 0.5, "model2_fake_prob": 0.5, "ensemble_score": 0.5},
        }


def analyze_all_frames(frame_paths, progress_callback=None):
    """
    Scores every frame. Also caches region_scores as empty dict so
    heatmap.py can optionally fill it in for the top frames only.

    SPEED IMPROVEMENT: We only run Grad-CAM heatmaps in app.py for the
    top 5 frames. Scoring all frames here is fast (~1-2s per frame on CPU).
    """
    results = []
    for i, path in enumerate(frame_paths):
        result = score_frame(path)
        result["frame_index"] = i
        result["frame_path"] = path
        result["frame_number"] = i + 1
        result["region_scores"] = {}   # filled later by heatmap for top frames
        results.append(result)
        if progress_callback:
            progress_callback(i + 1, len(frame_paths))
    return results


def get_overall_verdict(frame_results):
    """
    Computes final verdict weighted toward the worst-scoring frames.

    WHY top-20% weighting?
    A video with 50 clean frames and 10 heavily manipulated frames is still
    a deepfake. A simple average would give ~17% fake — misleadingly low.
    Weighting 60% on the top 20% of scores catches partial manipulations.
    """
    if not frame_results:
        return {"overall_score": 0.0, "verdict": "Unknown", "color": "gray"}

    scores = [r["fake_score"] for r in frame_results]
    scores_sorted = sorted(scores, reverse=True)

    top_count = max(1, len(scores_sorted) // 5)
    top_avg = float(np.mean(scores_sorted[:top_count]))
    overall_avg = float(np.mean(scores))
    overall_score = round(0.6 * top_avg + 0.4 * overall_avg, 4)

    if overall_score < 0.35:
        verdict, color = "Likely Real", "green"
    elif overall_score < 0.55:
        verdict, color = "Uncertain", "orange"
    else:
        verdict, color = "Likely Manipulated", "red"

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
        "min_score": round(min(scores), 4),
    }

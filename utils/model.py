import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

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

        if score2 is not None:
            # Ensemble: weighted average
            # Model 2 gets slightly more weight for AI-generated detection
            # since that's what this model specialises in and is the harder
            # case that model 1 fails on.
            final_score = round(0.45 * score1 + 0.55 * score2, 4)
        else:
            final_score = round(score1, 4)

        return {
            "fake_score": final_score,
            "breakdown": {
                "deepfake_model_score": round(score1, 4),
                "ai_generation_model_score": round(score2, 4) if score2 is not None else 0.5,
            },
            "raw_signals": {
                "model1_fake_prob": round(score1, 4),
                "model2_fake_prob": round(score2, 4) if score2 is not None else 0.5,
                "ensemble_score": final_score,
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
        verdict, color = "Likely Deepfake", "red"

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

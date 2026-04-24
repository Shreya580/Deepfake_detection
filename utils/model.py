import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# ── Load model ONCE when the module is imported ───────────────────────────────
# WHY here and not inside the function?
# Loading takes 3-5 seconds. If loaded inside score_frame(), it reloads every
# single frame → 60 frames × 4 seconds = 4 minutes wasted.
# Loading once at the top means it happens once when app.py starts.

MODEL_NAME = "dima806/deepfake_vs_real_image_detection"

print("Loading deepfake detection model... (first run downloads ~300MB)")
feature_extractor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
model.eval()  # Evaluation mode: disables dropout, fixes batch norm
print("Model loaded successfully.")

# ── Identify which output index = "fake" ─────────────────────────────────────
# The model has two classes. We check its label map to find the "fake" index.
LABELS = model.config.id2label  # e.g. {0: 'fake', 1: 'real'}
FAKE_LABEL_IDX = [k for k, v in LABELS.items() if 'fake' in v.lower()][0]
print(f"Label map: {LABELS}  →  FAKE index = {FAKE_LABEL_IDX}")


def score_frame(image_path):
    """
    Runs the HuggingFace ViT deepfake model on one image frame.

    Steps:
    1. Load image as PIL
    2. Run through feature extractor (resize + normalize)
    3. Forward pass through ViT → logits
    4. Softmax → probabilities
    5. Return fake probability

    Returns dict with fake_score, breakdown, raw_signals
    """
    try:
        img = Image.open(image_path).convert("RGB")
        inputs = feature_extractor(images=img, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)[0]
        fake_prob = float(probs[FAKE_LABEL_IDX].item())
        real_prob = 1.0 - fake_prob

        return {
            "fake_score": round(fake_prob, 4),
            "breakdown": {
                "ai_model_confidence": round(fake_prob, 4),
            },
            "raw_signals": {
                "fake_probability": round(fake_prob, 4),
                "real_probability": round(real_prob, 4),
            },
            "fake_label_idx": FAKE_LABEL_IDX,
        }

    except Exception as e:
        print(f"[model] Error scoring {image_path}: {e}")
        return {
            "fake_score": 0.5,
            "breakdown": {"ai_model_confidence": 0.5},
            "raw_signals": {"fake_probability": 0.5, "real_probability": 0.5},
            "fake_label_idx": FAKE_LABEL_IDX,
        }


def get_model_and_extractor():
    """
    Returns the loaded model and feature extractor.
    Called by heatmap.py so Grad-CAM can access the model object directly
    without reloading it a second time.
    """
    return model, feature_extractor


def analyze_all_frames(frame_paths, progress_callback=None):
    """Runs score_frame() on every extracted frame path."""
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
    Computes a final verdict score weighted toward the most suspicious frames.
    WHY weighted? A single manipulated frame matters more than many clean ones.
    We weight the top 20% of frames at 60% of the final score.
    """
    if not frame_results:
        return {"overall_score": 0.0, "verdict": "Unknown", "color": "gray"}

    scores = [r["fake_score"] for r in frame_results]
    scores_sorted = sorted(scores, reverse=True)

    top_count = max(1, len(scores_sorted) // 5)
    top_avg = np.mean(scores_sorted[:top_count])
    overall_avg = np.mean(scores)
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
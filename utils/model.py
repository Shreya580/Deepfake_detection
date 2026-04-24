import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# ── Load model ONCE when the module is imported ──────────────────────────────
# WHY here and not inside the function?
# Loading a model takes 3-5 seconds. If you load it inside score_frame(),
# it reloads every single frame = 60 frames × 4 seconds = 4 minutes wasted.
# Loading once at the top means it happens once when app.py starts.

MODEL_NAME = "dima806/deepfake_vs_real_image_detection"

print("Loading deepfake detection model... (first run downloads ~300MB)")
feature_extractor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
model.eval()  # Set to evaluation mode — disables dropout, fixes batch norm
print("Model loaded.")

# ── What labels does this model use? ─────────────────────────────────────────
# The model has two output classes. We need to know which index = "fake".
# Check model.config.id2label to see: {0: 'fake', 1: 'real'} or vice versa.
LABELS = model.config.id2label  # e.g. {0: 'fake', 1: 'real'}
FAKE_LABEL_IDX = [k for k, v in LABELS.items() if 'fake' in v.lower()][0]


def score_frame(image_path):
    """
    Runs the real deepfake detection model on one image frame.
    
    HOW IT WORKS:
    1. Load the image and resize to 224x224 (what the model expects)
    2. Run it through the feature extractor (normalizes pixel values)
    3. Pass through the ViT model → get raw output scores (logits)
    4. Apply softmax to convert logits to probabilities
    5. Return the probability for the "fake" class
    
    Returns: dict with fake_score, breakdown, raw_signals
    """
    try:
        # Load image as PIL (HuggingFace models expect PIL images)
        img = Image.open(image_path).convert("RGB")
        
        # Feature extractor: resizes to 224x224, normalizes pixel values
        # WHY normalize? The model was trained on normalized images.
        # Feeding raw 0-255 pixels would give wrong results.
        inputs = feature_extractor(images=img, return_tensors="pt")
        
        # Run the model
        # torch.no_grad() = don't compute gradients during forward pass
        # WHY? Gradients are only needed during training. Skipping them
        # saves memory and makes inference ~2x faster.
        with torch.no_grad():
            outputs = model(**inputs)
        
        # outputs.logits shape: [1, 2] = one image, two classes
        # Softmax converts raw scores to probabilities that sum to 1.0
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
            }
        }
    
    except Exception as e:
        print(f"Error scoring frame {image_path}: {e}")
        return {
            "fake_score": 0.5,
            "breakdown": {"ai_model_confidence": 0.5},
            "raw_signals": {"fake_probability": 0.5, "real_probability": 0.5}
        }


def get_model_and_extractor():
    """
    Returns the model and feature extractor objects.
    WHY this function? Grad-CAM in heatmap.py needs direct access to the
    model object — not just the score. This gives heatmap.py access
    without reloading the model a second time.
    """
    return model, feature_extractor


def analyze_all_frames(frame_paths, progress_callback=None):
    """Runs score_frame() on every extracted frame."""
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
    """Computes final verdict weighted toward highest-scoring frames."""
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
        "min_score": round(min(scores), 4)
    }
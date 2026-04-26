"""
model.py — Detection pipeline

WHAT YOUR FRIEND DID BETTER:
  She used @st.cache_resource which loads the model ONCE and caches it
  across Streamlit reruns. Your version loaded at module level which
  caused slow cold starts and couldn't show a loading spinner.

WHAT YOU HAD BETTER:
  Two-model ensemble (face-swap + AI-generated detector) gives broader coverage.

THIS FILE combines both: cache_resource + ensemble.
"""

import torch
import numpy as np
from PIL import Image
import streamlit as st
from transformers import AutoImageProcessor, AutoModelForImageClassification


# ─────────────────────────────────────────────────────────────────────────────
# @st.cache_resource means: load ONCE per session, reuse forever.
# WHY this matters: without it, the model reloads on every Streamlit rerun
# (every time user changes anything). With it, loading happens once (~5s),
# then every frame call is fast (~1-2s).
# ─────────────────────────────────────────────────────────────────────────────

MODEL_1 = "dima806/deepfake_vs_real_image_detection"   # Face-swap deepfakes
MODEL_2 = "haywoodsloan/ai-image-detector-deploy"      # AI-generated images


@st.cache_resource(show_spinner="Loading detection model 1 (face-swap)...")
def _load_model1():
    extractor = AutoImageProcessor.from_pretrained(MODEL_1)
    model     = AutoModelForImageClassification.from_pretrained(MODEL_1)
    model.eval()
    labels    = model.config.id2label
    fake_idx  = next(k for k, v in labels.items() if "fake" in v.lower())
    return extractor, model, fake_idx


@st.cache_resource(show_spinner="Loading detection model 2 (AI-generation)...")
def _load_model2():
    try:
        extractor = AutoImageProcessor.from_pretrained(MODEL_2)
        model     = AutoModelForImageClassification.from_pretrained(MODEL_2)
        model.eval()
        labels   = model.config.id2label
        keywords = ["fake", "artificial", "ai", "generated", "synthetic"]
        try:
            fake_idx = next(k for k, v in labels.items()
                           if any(kw in v.lower() for kw in keywords))
        except StopIteration:
            fake_idx = 0
        return extractor, model, fake_idx
    except Exception as e:
        st.warning(f"Model 2 unavailable — using model 1 only. ({e})")
        return None, None, None


def get_model_for_gradcam():
    """
    Returns (extractor, model, fake_idx) for model 1.
    Used by gradcam.py so it doesn't reload the model a second time.
    """
    return _load_model1()


def _run_model1(img_pil):
    extractor, model, fake_idx = _load_model1()
    inputs = extractor(images=img_pil, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)[0]
    return float(probs[fake_idx].item())


def _run_model2(img_pil):
    extractor, model, fake_idx = _load_model2()
    if model is None:
        return None
    try:
        inputs = extractor(images=img_pil, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]
        return float(probs[fake_idx].item())
    except Exception as e:
        print(f"[model2] inference error: {e}")
        return None


def score_frame(image_path):
    """
    Runs both models on a frame and returns an ensemble fake score.

    WHY ensemble?
    Model 1 catches face-swap deepfakes (videos, realistic face swaps).
    Model 2 catches AI-generated images (DALL-E, Midjourney, ChatGPT).
    Together they cover both main categories of fake media.

    Returns: dict with fake_score + per-model breakdown
    """
    try:
        img_pil = Image.open(image_path).convert("RGB")

        score1 = _run_model1(img_pil)
        score2 = _run_model2(img_pil)

        if score2 is not None:
            # Give model2 slightly more weight — AI-generated is harder to catch
            final = round(0.45 * score1 + 0.55 * score2, 4)
        else:
            final = round(score1, 4)

        return {
            "fake_score": final,
            "breakdown": {
                "Face-swap model":        round(score1, 4),
                "AI-generation model":    round(score2, 4) if score2 else "N/A",
            },
            "raw_signals": {
                "model1": score1,
                "model2": score2 or 0.5,
                "ensemble": final,
            }
        }

    except Exception as e:
        print(f"[score_frame] error: {e}")
        return {
            "fake_score": 0.5,
            "breakdown": {"Face-swap model": 0.5, "AI-generation model": "N/A"},
            "raw_signals": {"model1": 0.5, "model2": 0.5, "ensemble": 0.5},
        }


def analyze_all_frames(frame_paths, progress_callback=None):
    """Score every extracted frame. Progress callback updates the UI bar."""
    results = []
    for i, path in enumerate(frame_paths):
        result = score_frame(path)
        result["frame_index"]  = i
        result["frame_path"]   = path
        result["frame_number"] = i + 1
        result["region_scores"] = {}
        results.append(result)
        if progress_callback:
            progress_callback(i + 1, len(frame_paths))
    return results


def get_overall_verdict(frame_results):
    """
    Final verdict weighted toward the worst frames.
    60% weight on top-20% scores catches partial deepfakes.
    """
    if not frame_results:
        return {"overall_score": 0.0, "verdict": "Unknown", "color": "gray",
                "overall_percent": 0, "total_frames": 0, "fake_frames": 0,
                "uncertain_frames": 0, "real_frames": 0,
                "max_score": 0, "min_score": 0}

    scores = [r["fake_score"] for r in frame_results]
    top_n  = max(1, len(scores) // 5)
    top_avg = float(np.mean(sorted(scores, reverse=True)[:top_n]))
    avg     = float(np.mean(scores))
    final   = round(0.6 * top_avg + 0.4 * avg, 4)

    if final < 0.35:
        verdict, color = "Likely Real",    "green"
    elif final < 0.55:
        verdict, color = "Uncertain",      "orange"
    else:
        verdict, color = "Likely Deepfake","red"

    return {
        "overall_score":    final,
        "overall_percent":  round(final * 100, 1),
        "verdict":          verdict,
        "color":            color,
        "total_frames":     len(scores),
        "fake_frames":      sum(1 for s in scores if s > 0.6),
        "uncertain_frames": sum(1 for s in scores if 0.4 <= s <= 0.6),
        "real_frames":      sum(1 for s in scores if s < 0.4),
        "max_score":        round(max(scores), 4),
        "min_score":        round(min(scores), 4),
    }

import cv2
import os
import numpy as np
from PIL import Image


def extract_frames(video_path, output_folder="frames", max_frames=60, sample_every=15):
    if os.path.exists(output_folder):
        for f in os.listdir(output_folder):
            try:
                os.remove(os.path.join(output_folder, f))
            except Exception:
                pass
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    duration     = total_frames / fps if fps > 0 else 0

    frame_paths = []
    frame_count = 0
    saved_count = 0

    while cap.isOpened() and saved_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % sample_every == 0:
            resized    = cv2.resize(frame, (224, 224))
            frame_path = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_path, resized)
            frame_paths.append(frame_path)
            saved_count += 1
        frame_count += 1

    cap.release()

    return {
        "frame_paths":           frame_paths,
        "total_frames_in_video": total_frames,
        "frames_extracted":      saved_count,
        "fps":                   round(fps, 2),
        "duration_seconds":      round(duration, 2),
    }


def process_image(image_path, output_folder="frames"):
    os.makedirs(output_folder, exist_ok=True)
    for f in os.listdir(output_folder):
        try:
            os.remove(os.path.join(output_folder, f))
        except Exception:
            pass

    img         = Image.open(image_path).convert("RGB")
    img_resized = img.resize((224, 224))
    frame_path  = os.path.join(output_folder, "frame_0000.jpg")
    img_resized.save(frame_path)

    return {
        "frame_paths":           [frame_path],
        "total_frames_in_video": 1,
        "frames_extracted":      1,
        "fps":                   0,
        "duration_seconds":      0,
    }


def get_frame_thumbnail(frame_path, size=(400, 400)):
    img = Image.open(frame_path).convert("RGB")
    img = img.resize(size, Image.LANCZOS)
    return img

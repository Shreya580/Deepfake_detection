import cv2          # OpenCV for video reading
import os           # For folder/file operations
from PIL import Image  # For saving frames as images


def _resize_preserving_aspect_pil(img, max_size=768):
    img = img.copy()
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    return img


def _resize_preserving_aspect_cv(frame, max_size=768):
    h, w = frame.shape[:2]
    scale = min(max_size / max(h, w), 1.0)
    if scale >= 1.0:
        return frame
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

def extract_frames(video_path, output_folder="frames", max_frames=60, sample_every=15):
    """
    Extracts frames from a video file at regular intervals.
    
    WHY sample_every=15?
    Videos are typically 30fps. Sampling every 15 frames gives us
    2 frames per second — enough detail without overloading your laptop.
    
    WHY max_frames=60?
    60 frames × ~0.5 seconds each = covers a 30-second video.
    More than enough for a demo. Keeps processing time under 1 minute.
    """
    
    # Clear old frames from previous runs
    if os.path.exists(output_folder):
        for f in os.listdir(output_folder):
            os.remove(os.path.join(output_folder, f))
    os.makedirs(output_folder, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video metadata
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    frame_paths = []
    frame_count = 0
    saved_count = 0
    
    while cap.isOpened() and saved_count < max_frames:
        ret, frame = cap.read()
        
        if not ret:
            break  # End of video
        
        # Only save every Nth frame (sampling)
        if frame_count % sample_every == 0:
            # Save frame as JPEG
            frame_path = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
            preview = _resize_preserving_aspect_cv(frame, max_size=768)
            cv2.imwrite(frame_path, preview)
            frame_paths.append(frame_path)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    
    return {
        "frame_paths": frame_paths,
        "total_frames_in_video": total_frames,
        "frames_extracted": saved_count,
        "fps": round(fps, 2),
        "duration_seconds": round(duration, 2)
    }


def process_image(image_path, output_folder="frames"):
    """
    For single image input — treat it like a one-frame video.
    Resize and save to frames folder for consistent downstream processing.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Clear old frames
    for f in os.listdir(output_folder):
        os.remove(os.path.join(output_folder, f))
    
    # Open and resize. Keep enough detail for artifact checks; the model's
    # feature extractor will handle its own 224px input internally.
    img = Image.open(image_path).convert("RGB")
    img = _resize_preserving_aspect_pil(img, max_size=768)
    
    frame_path = os.path.join(output_folder, "frame_0000.jpg")
    img.save(frame_path)
    
    return {
        "frame_paths": [frame_path],
        "total_frames_in_video": 1,
        "frames_extracted": 1,
        "fps": 0,
        "duration_seconds": 0
    }


def get_frame_thumbnail(frame_path, size=(112, 112)):
    """
    Returns a smaller thumbnail version of a frame.
    Used in the Suspicious Frame Gallery visualization.
    WHY 112x112? Half of 224 — small enough to show 5 in a row.
    """
    img = Image.open(frame_path).convert("RGB")
    img.thumbnail(size)
    return img

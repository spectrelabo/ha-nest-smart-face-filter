"""
recognition.py - Shared face-recognition logic for the add-on.

Ported from face_recognition/analyze.py and face_recognition/create_embeddings.py
in the main repo, without the manual OpenCV .so preload hacks
(here the dependencies are installed normally via pip in the add-on Docker
image, which has its own isolated and complete Python environment).
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import cv2

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from insightface.app import FaceAnalysis


def load_known_embeddings(embeddings_dir: Path) -> dict:
    """Load into RAM all .npy files found in the specified directory."""
    known_faces = {}
    embeddings_dir = Path(embeddings_dir)
    if not embeddings_dir.exists():
        print(f"[WARNING] Embeddings directory not found: {embeddings_dir}")
        return known_faces

    for npy_file in embeddings_dir.glob("*.npy"):
        try:
            emb = np.load(npy_file)
            emb = emb / np.linalg.norm(emb)
            known_faces[npy_file.stem] = emb
        except Exception as e:
            print(f"[ERROR] Unable to load {npy_file}: {e}", file=sys.stderr)

    return known_faces


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(emb1, emb2)
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot_product / (norm1 * norm2))


def crop_best_face(img: np.ndarray, faces: list, pad_ratio: float = 0.25) -> np.ndarray:
    """Crop the face with the largest bounding box, with proportional padding."""
    best_face_obj = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    bbox = [int(b) for b in best_face_obj.bbox]
    h, w, _ = img.shape
    pad_w = int((bbox[2] - bbox[0]) * pad_ratio)
    pad_h = int((bbox[3] - bbox[1]) * pad_ratio)
    x1, y1 = max(0, bbox[0] - pad_w), max(0, bbox[1] - pad_h)
    x2, y2 = min(w, bbox[2] + pad_w), min(h, bbox[3] + pad_h)
    return img[y1:y2, x1:x2]


def extract_frames_from_video(video_bytes: bytes, max_frames: int = 30) -> list:
    """Decode an in-memory video buffer (mp4) and extract its frames as numpy arrays.

    Temporarily writes the buffer to a temporary file (required because
    cv2.VideoCapture does not support reading directly from in-memory bytes),
    decodes it with OpenCV, and returns a list of frames (up to max_frames,
    sampled uniformly if the video contains more).
    """
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            temp_path = tmp.name

        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            print("[WARNING] Unable to open the temporary video with OpenCV.", file=sys.stderr)
            return []

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        if not frames:
            print("[WARNING] No frames decoded from the video.", file=sys.stderr)
            return []

        if len(frames) > max_frames:
            indices = np.linspace(0, len(frames) - 1, max_frames).astype(int)
            frames = [frames[i] for i in indices]

        return frames
    except Exception as e:
        print(f"[ERROR] Error while extracting video frames: {e}", file=sys.stderr)
        return []
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass


def analyze_frames_for_best_match(engine: "FaceRecognitionEngine", frames: list, known_faces: dict) -> dict:
    """Run face recognition on each frame and return the best overall result.

    For each frame, detect faces and compute cosine similarity with every known face.
    Returns a dict with: {"identity": str|None, "score": float, "frame": np.ndarray|None, "faces": list}
    corresponding to the frame/face with the highest absolute match score across all frames.
    If no frame contains detectable faces, returns {"identity": None, "score": -1.0, "frame": None, "faces": []}.
    """
    best_identity = None
    best_score = -1.0
    best_frame = None
    best_faces = []

    for frame in frames:
        faces = engine.analyze_image(frame)
        if not faces:
            continue
        # Keep track of the first frame with faces in case there are no known embeddings
        if best_frame is None:
            best_frame = frame
            best_faces = faces
        for face in faces:
            for identity, known_emb in known_faces.items():
                score = cosine_similarity(face.normed_embedding, known_emb)
                if score > best_score:
                    best_score = score
                    best_identity = identity
                    best_frame = frame
                    best_faces = faces

    return {
        "identity": best_identity,
        "score": best_score,
        "frame": best_frame,
        "faces": best_faces,
    }


class FaceRecognitionEngine:
    """Wrapper around InsightFace FaceAnalysis, initialized only once."""

    def __init__(self, det_size: tuple = (320, 320)):
        self.app = FaceAnalysis(
            name="buffalo_s",
            providers=["CPUExecutionProvider"],
            allowed_modules=["detection", "recognition"],
        )
        self.app.prepare(ctx_id=0, det_size=det_size)

    def analyze_image(self, img: np.ndarray) -> list:
        """Detect all faces in the image and extract their embeddings."""
        return self.app.get(img)

    def extract_best_embedding(self, img: np.ndarray):
        """Return the embedding of the face with the largest bounding box, or None."""
        faces = self.analyze_image(img)
        if not faces:
            return None
        best_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return best_face.normed_embedding

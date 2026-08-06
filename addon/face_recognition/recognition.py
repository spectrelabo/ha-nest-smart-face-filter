"""
recognition.py - Logica condivisa di riconoscimento facciale per l'add-on.

Porting da face_recognition/analyze.py e face_recognition/create_embeddings.py
del repo principale, senza gli hack di preload manuale delle .so di OpenCV
(qui le dipendenze sono installate normalmente via pip nell'immagine Docker
dell'add-on, che ha il proprio ambiente Python isolato e completo).
"""

import os
import sys
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
    """Carica in RAM tutti i file .npy presenti nella directory specificata."""
    known_faces = {}
    embeddings_dir = Path(embeddings_dir)
    if not embeddings_dir.exists():
        print(f"[WARNING] Directory embedding non trovata: {embeddings_dir}")
        return known_faces

    for npy_file in embeddings_dir.glob("*.npy"):
        try:
            emb = np.load(npy_file)
            emb = emb / np.linalg.norm(emb)
            known_faces[npy_file.stem] = emb
        except Exception as e:
            print(f"[ERROR] Impossibile caricare {npy_file}: {e}", file=sys.stderr)

    return known_faces


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Calcola la cosine similarity tra due vettori."""
    dot_product = np.dot(emb1, emb2)
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot_product / (norm1 * norm2))


def crop_best_face(img: np.ndarray, faces: list, pad_ratio: float = 0.25) -> np.ndarray:
    """Ritaglia il volto con bounding box più ampia, con padding proporzionale."""
    best_face_obj = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    bbox = [int(b) for b in best_face_obj.bbox]
    h, w, _ = img.shape
    pad_w = int((bbox[2] - bbox[0]) * pad_ratio)
    pad_h = int((bbox[3] - bbox[1]) * pad_ratio)
    x1, y1 = max(0, bbox[0] - pad_w), max(0, bbox[1] - pad_h)
    x2, y2 = min(w, bbox[2] + pad_w), min(h, bbox[3] + pad_h)
    return img[y1:y2, x1:x2]


class FaceRecognitionEngine:
    """Wrapper su InsightFace FaceAnalysis, inizializzato una sola volta."""

    def __init__(self, det_size: tuple = (320, 320)):
        self.app = FaceAnalysis(
            name="buffalo_s",
            providers=["CPUExecutionProvider"],
            allowed_modules=["detection", "recognition"],
        )
        self.app.prepare(ctx_id=0, det_size=det_size)

    def analyze_image(self, img: np.ndarray) -> list:
        """Rileva tutti i volti nell'immagine e ne estrae gli embedding."""
        return self.app.get(img)

    def extract_best_embedding(self, img: np.ndarray):
        """Ritorna l'embedding del volto con bounding box più grande, o None."""
        faces = self.analyze_image(img)
        if not faces:
            return None
        best_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return best_face.normed_embedding

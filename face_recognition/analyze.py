#!/usr/bin/env python3
"""
analyze.py - [LEGACY/DEPRECATED] Event-driven face recognition script.

THIS SCRIPT IS NO LONGER THE RECOMMENDED PATH. On Home Assistant OS,
shell_command runs inside a Docker container isolated from the Terminal &
SSH add-on container, so the dependencies installed manually here are not
reliably reachable. Use the local "Face Recognition" add-on instead
(folder addon/face_recognition/ in this repo), which exposes the same logic
through an internal REST API.

This file remains in the repo only for users who already have a working manual
setup via SSH and do not want to migrate to the add-on yet.
"""

import os
import sys
import ctypes

# Preload shared C libraries (libopencv*.so) from the shared /config/ volume
lib_dir = "/config/face_recognition/lib"
if os.path.exists(lib_dir):
    os.environ["LD_LIBRARY_PATH"] = lib_dir + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    for lib in ["libopencv_core.so.412", "libopencv_imgproc.so.412", "libopencv_imgcodecs.so.412", "libopencv_ml.so.412", "libopencv_dnn.so.412"]:
        so_path = os.path.join(lib_dir, lib)
        if os.path.exists(so_path):
            try:
                ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
            except Exception:
                pass

shared_site = "/config/face_recognition/lib/site-packages"
if shared_site not in sys.path:
    sys.path.insert(0, shared_site)

user_site = "site.getusersitepackages()"
if user_site not in sys.path:
    sys.path.insert(0, user_site)

# Prevent OpenMP/Musl deadlocks on Alpine Linux
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import shutil
from datetime import datetime
from pathlib import Path
import numpy as np
import cv2
import requests

try:
    import insightface
    from insightface.app import FaceAnalysis
except ImportError:
    print("[ERROR] Missing 'insightface' dependency. Install with: pip install insightface onnxruntime opencv-python-headless numpy requests", file=sys.stderr)
    sys.exit(1)

# Default configuration
EMBEDDINGS_DIR = os.getenv("EMBEDDINGS_DIR", "/config/face_recognition/embeddings")
SNAPSHOT_PATH = os.getenv("SNAPSHOT_PATH", "/config/www/last_person.jpg")
THRESHOLD = float(os.getenv("THRESHOLD", "0.45"))
SAVE_UNKNOWN_FACES = os.getenv("SAVE_UNKNOWN_FACES", "true").lower() == "true"
UNKNOWN_FACES_DIR = os.getenv("UNKNOWN_FACES_DIR", "/config/face_recognition/unknown_faces")
MAX_SAVED_FACES = int(os.getenv("MAX_SAVED_FACES", "30"))

# Home Assistant REST API configuration
HA_URL = os.getenv("SUPERVISOR_TOKEN", None)  # HA Supervisor REST Endpoint
HA_BASE_URL = os.getenv("HA_BASE_URL", "http://supervisor/core")
HA_BEARER_TOKEN = os.getenv("SUPERVISOR_TOKEN") or os.getenv("HA_ACCESS_TOKEN", "")


def load_known_embeddings(embeddings_dir: Path) -> dict:
    """Load into RAM all .npy files found in the specified directory."""
    known_faces = {}
    if not embeddings_dir.exists():
        print(f"[WARNING] Embeddings directory not found: {embeddings_dir}")
        return known_faces

    for npy_file in embeddings_dir.glob("*.npy"):
        try:
            emb = np.load(npy_file)
            emb = emb / np.linalg.norm(emb)
            identity = npy_file.stem
            known_faces[identity] = emb
        except Exception as e:
            print(f"[ERROR] Unable to load {npy_file}: {e}", file=sys.stderr)

    return known_faces


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Calculate cosine similarity between two normalized vectors."""
    dot_product = np.dot(emb1, emb2)
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot_product / (norm1 * norm2))


def archive_unknown_face(snapshot_file: Path):
    """Archive the unknown-face snapshot so new faces can be learned later."""
    if not SAVE_UNKNOWN_FACES:
        return

    unknown_dir = Path(UNKNOWN_FACES_DIR)
    unknown_dir.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest_file = unknown_dir / f"unknown_{timestamp_str}.jpg"

    try:
        shutil.copy(snapshot_file, dest_file)
        print(f"[INFO] Snapshot archived at: {dest_file} (can be used to learn a new face).")

        # Enforce the maximum number of stored images (auto-pruning)
        saved_files = sorted(unknown_dir.glob("unknown_*.jpg"), key=os.path.getmtime)
        if len(saved_files) > MAX_SAVED_FACES:
            for old_file in saved_files[:-MAX_SAVED_FACES]:
                try:
                    old_file.unlink()
                    print(f"[CLEANUP] Removed old archived snapshot: {old_file.name}")
                except Exception as e:
                    print(f"[WARNING] Unable to remove {old_file}: {e}")
    except Exception as e:
        print(f"[ERROR] Error while archiving the snapshot: {e}", file=sys.stderr)


def notify_home_assistant(event_name: str, payload: dict):
    """Send a custom event to Home Assistant through the REST API."""
    if not HA_BEARER_TOKEN:
        print("[INFO] No HA token configured. API notification skipped.")
        return

    url = f"{HA_BASE_URL}/api/events/{event_name}"
    headers = {
        "Authorization": f"Bearer {HA_BEARER_TOKEN}",
        "Content-Type": "application/json",
    }
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=5)
        if response.status_code in (200, 201):
            print(f"[SUCCESS] Event '{event_name}' sent to Home Assistant (status: {response.status_code}).")
        else:
            print(f"[WARNING] Unexpected HA API response ({response.status_code}): {response.text}")
    except Exception as e:
        print(f"[ERROR] Unable to send event to Home Assistant: {e}", file=sys.stderr)


def main():
    start_time = time.perf_counter()
    print("=== [FACE RECOGNITION PIPELINE] Analysis Run ===")

    embeddings_path = Path(EMBEDDINGS_DIR)
    snapshot_file = Path(SNAPSHOT_PATH)

    # 1. Load known faces into RAM
    known_faces = load_known_embeddings(embeddings_path)
    print(f"[INFO] Known faces loaded into RAM: {len(known_faces)} ({list(known_faces.keys())})")

    # 2. Check whether the snapshot exists
    if not snapshot_file.is_file():
        print(f"[ERROR] Snapshot not found: {snapshot_file}", file=sys.stderr)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        print(f"[METRICS] Execution time: {elapsed_ms:.2f} ms")
        sys.exit(1)

    # 3. Initialize InsightFace
    init_start = time.perf_counter()
    app = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"], allowed_modules=["detection", "recognition"])
    app.prepare(ctx_id=0, det_size=(320, 320))
    init_ms = (time.perf_counter() - init_start) * 1000

    # 4. Decode image
    img = cv2.imread(SNAPSHOT_PATH)
    if img is None:
        try:
            from PIL import Image
            pil_img = Image.open(SNAPSHOT_PATH).convert("RGB")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"[ERROR] Unable to read the snapshot with OpenCV or PIL: {SNAPSHOT_PATH} ({e})")
            sys.exit(1)

    # 5. Face Detection & Feature Extraction
    infer_start = time.perf_counter()
    faces = app.get(img)
    infer_ms = (time.perf_counter() - infer_start) * 1000

    if not faces:
        print(f"[WARNING] No face detected in snapshot {snapshot_file}.")
        notify_home_assistant("no_face_detected", {
            "snapshot_path": str(snapshot_file),
            "message": "Person detected by Nest but no extractable face found in the snapshot."
        })
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        print(f"[METRICS] Detection: {infer_ms:.2f} ms | Total Execution Time: {elapsed_ms:.2f} ms")
        return

    # 6. Crop the main face for the zoomed-in preview in the notification
    best_face_obj = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    bbox = [int(b) for b in best_face_obj.bbox]
    h, w, _ = img.shape
    pad_w = int((bbox[2] - bbox[0]) * 0.25)
    pad_h = int((bbox[3] - bbox[1]) * 0.25)
    x1, y1 = max(0, bbox[0] - pad_w), max(0, bbox[1] - pad_h)
    x2, y2 = min(w, bbox[2] + pad_w), min(h, bbox[3] + pad_h)
    crop_img = img[y1:y2, x1:x2]
    
    crop_path = Path("/config/www/snapshots/last_face_crop.jpg")
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(crop_path), crop_img)

    # 7. Compare embeddings (Cosine Similarity)
    best_identity = None
    best_score = -1.0

    for face in faces:
        detected_emb = face.normed_embedding

        for identity, known_emb in known_faces.items():
            score = cosine_similarity(detected_emb, known_emb)
            if score > best_score:
                best_score = score
                best_identity = identity

    print(f"[RESULT] Best match: '{best_identity}' with similarity score = {best_score:.4f} (Threshold = {THRESHOLD})")

    if best_score >= THRESHOLD:
        print(f"[SUCCESS] RECOGNIZED face: {best_identity} ({best_score * 100:.1f}% confidence)")
        notify_home_assistant("known_person_detected", {
            "identity": best_identity,
            "confidence": round(best_score, 4),
            "snapshot_path": str(snapshot_file),
            "face_crop_path": "/local/snapshots/last_face_crop.jpg"
        })
    else:
        print(f"[ALERT] UNKNOWN face! Highest match {best_score:.4f} below threshold {THRESHOLD}.")
        archive_unknown_face(snapshot_file)
        notify_home_assistant("unknown_person_detected", {
            "highest_match": best_identity if best_identity else "none",
            "score": round(best_score, 4),
            "threshold": THRESHOLD,
            "snapshot_path": str(snapshot_file),
            "face_crop_path": "/local/snapshots/last_face_crop.jpg"
        })

    total_ms = (time.perf_counter() - start_time) * 1000
    print(f"[METRICS] Init: {init_ms:.2f} ms | Inference: {infer_ms:.2f} ms | Total Time: {total_ms:.2f} ms")


if __name__ == "__main__":
    main()

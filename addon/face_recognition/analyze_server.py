"""
analyze_server.py - Server Flask dell'add-on Face Recognition.

Espone via HTTP la stessa logica del vecchio face_recognition/analyze.py,
ma in esecuzione dentro il container Docker dedicato dell'add-on (dove
insightface/opencv sono installati normalmente via pip), invece che via
shell_command dentro il container di Home Assistant Core.
"""

import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import requests
from flask import Flask, jsonify, request

from recognition import (
    FaceRecognitionEngine,
    crop_best_face,
    load_known_embeddings,
    cosine_similarity,
    extract_frames_from_video,
    analyze_frames_for_best_match,
)

EMBEDDINGS_DIR = os.getenv("EMBEDDINGS_DIR", "/config/face_recognition/embeddings")
THRESHOLD = float(os.getenv("THRESHOLD", "0.45"))
SAVE_UNKNOWN_FACES = os.getenv("SAVE_UNKNOWN_FACES", "true").lower() == "true"
UNKNOWN_FACES_DIR = os.getenv("UNKNOWN_FACES_DIR", "/config/face_recognition/unknown_faces")
MAX_SAVED_FACES = int(os.getenv("MAX_SAVED_FACES", "30"))
CROP_OUTPUT_PATH = os.getenv("CROP_OUTPUT_PATH", "/config/www/snapshots/last_face_crop.jpg")

HA_BASE_URL = os.getenv("HA_BASE_URL", "http://supervisor/core")
HA_BEARER_TOKEN = os.getenv("SUPERVISOR_TOKEN", "")

app = Flask(__name__)
_engine = None  # inizializzato lazy in get_engine() per non caricare il modello negli unit test


def get_engine() -> FaceRecognitionEngine:
    global _engine
    if _engine is None:
        _engine = FaceRecognitionEngine()
    return _engine


def notify_home_assistant(event_name: str, payload: dict) -> None:
    """Invia un evento personalizzato a Home Assistant via Supervisor REST API."""
    if not HA_BEARER_TOKEN:
        print("[INFO] Nessun SUPERVISOR_TOKEN disponibile. Notifica ignorata.")
        return

    url = f"{HA_BASE_URL}/api/events/{event_name}"
    headers = {
        "Authorization": f"Bearer {HA_BEARER_TOKEN}",
        "Content-Type": "application/json",
    }
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=5)
        if response.status_code in (200, 201):
            print(f"[SUCCESS] Evento '{event_name}' inviato (status: {response.status_code}).")
        else:
            print(f"[WARNING] Risposta HA API non prevista ({response.status_code}): {response.text}")
    except Exception as e:
        print(f"[ERROR] Impossibile inviare evento a Home Assistant: {e}", file=sys.stderr)


def archive_unknown_face(snapshot_file: Path) -> None:
    """Archivia lo snapshot del volto sconosciuto per l'apprendimento futuro."""
    if not SAVE_UNKNOWN_FACES:
        return

    unknown_dir = Path(UNKNOWN_FACES_DIR)
    unknown_dir.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest_file = unknown_dir / f"unknown_{timestamp_str}.jpg"

    try:
        shutil.copy(snapshot_file, dest_file)
        saved_files = sorted(unknown_dir.glob("unknown_*.jpg"), key=os.path.getmtime)
        if len(saved_files) > MAX_SAVED_FACES:
            for old_file in saved_files[:-MAX_SAVED_FACES]:
                old_file.unlink(missing_ok=True)
    except Exception as e:
        print(f"[ERROR] Errore durante l'archiviazione dello snapshot: {e}", file=sys.stderr)


def archive_unknown_frame(frame: np.ndarray) -> None:
    """Archivia un frame numpy come immagine del volto sconosciuto per l'apprendimento futuro."""
    if not SAVE_UNKNOWN_FACES:
        return

    unknown_dir = Path(UNKNOWN_FACES_DIR)
    unknown_dir.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest_file = unknown_dir / f"unknown_{timestamp_str}.jpg"

    try:
        cv2.imwrite(str(dest_file), frame)
        saved_files = sorted(unknown_dir.glob("unknown_*.jpg"), key=os.path.getmtime)
        if len(saved_files) > MAX_SAVED_FACES:
            for old_file in saved_files[:-MAX_SAVED_FACES]:
                old_file.unlink(missing_ok=True)
    except Exception as e:
        print(f"[ERROR] Errore durante l'archiviazione del frame sconosciuto: {e}", file=sys.stderr)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or {}
    snapshot_path = data.get("snapshot_path")
    threshold = float(data.get("threshold", THRESHOLD))

    if not snapshot_path:
        return jsonify({"error": "snapshot_path is required"}), 400

    snapshot_file = Path(snapshot_path)
    if not snapshot_file.is_file():
        return jsonify({"error": f"snapshot not found: {snapshot_path}"}), 400

    if snapshot_file.stat().st_size < 1024:
        return jsonify({"error": f"snapshot too small/corrupted: {snapshot_path}"}), 400

    img = cv2.imread(str(snapshot_file))
    if img is None:
        return jsonify({"error": f"failed to decode image: {snapshot_path}"}), 400

    known_faces = load_known_embeddings(Path(EMBEDDINGS_DIR))
    engine = get_engine()
    faces = engine.analyze_image(img)

    if not faces:
        notify_home_assistant("no_face_detected", {
            "snapshot_path": str(snapshot_file),
            "message": "Persona rilevata da Nest ma nessun volto estraibile dallo snapshot.",
        })
        return jsonify({"identity": None, "score": -1.0, "matched": False})

    crop_img = crop_best_face(img, faces)
    Path(CROP_OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(CROP_OUTPUT_PATH, crop_img)

    best_identity = None
    best_score = -1.0
    for face in faces:
        for identity, known_emb in known_faces.items():
            score = cosine_similarity(face.normed_embedding, known_emb)
            if score > best_score:
                best_score = score
                best_identity = identity

    matched = best_score >= threshold

    if matched:
        notify_home_assistant("known_person_detected", {
            "identity": best_identity,
            "confidence": round(best_score, 4),
            "snapshot_path": str(snapshot_file),
            "face_crop_path": "/local/snapshots/last_face_crop.jpg",
        })
    else:
        archive_unknown_face(snapshot_file)
        notify_home_assistant("unknown_person_detected", {
            "highest_match": best_identity if best_identity else "none",
            "score": round(best_score, 4),
            "threshold": threshold,
            "snapshot_path": str(snapshot_file),
            "face_crop_path": "/local/snapshots/last_face_crop.jpg",
        })

    return jsonify({"identity": best_identity, "score": round(best_score, 4), "matched": matched})


@app.route("/analyze_event", methods=["POST"])
def analyze_event():
    """Analizza un evento Nest estraendo e processando tutti i frame del clip video."""
    data = request.get_json(silent=True) or {}
    device_id = data.get("device_id")
    event_id = data.get("event_id")
    threshold = float(data.get("threshold", THRESHOLD))

    if not device_id or not event_id:
        return jsonify({"error": "device_id and event_id are required"}), 400

    if not HA_BEARER_TOKEN:
        return jsonify({"error": "SUPERVISOR_TOKEN not available, cannot fetch Nest event media"}), 400

    url = f"{HA_BASE_URL}/api/nest/event_media/{device_id}/{event_id}"
    try:
        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {HA_BEARER_TOKEN}"},
            timeout=15,
        )
    except Exception as e:
        return jsonify({"error": f"failed to fetch Nest event media: {e}"}), 502

    if response.status_code != 200:
        return jsonify({"error": f"failed to fetch Nest event media: HTTP {response.status_code}"}), 502

    frames = extract_frames_from_video(response.content)
    if not frames:
        return jsonify({"error": "no frames could be decoded from event clip"}), 400

    known_faces = load_known_embeddings(Path(EMBEDDINGS_DIR))
    result = analyze_frames_for_best_match(get_engine(), frames, known_faces)

    if result["frame"] is None:
        notify_home_assistant("no_face_detected", {
            "device_id": device_id,
            "event_id": event_id,
            "message": "Persona rilevata da Nest ma nessun volto estraibile dal clip evento.",
        })
        return jsonify({"identity": None, "score": -1.0, "matched": False})

    crop_img = crop_best_face(result["frame"], result["faces"])
    Path(CROP_OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(CROP_OUTPUT_PATH, crop_img)

    best_score = result["score"]
    best_identity = result["identity"]
    matched = best_score >= threshold

    if matched:
        notify_home_assistant("known_person_detected", {
            "identity": best_identity,
            "confidence": round(best_score, 4),
            "device_id": device_id,
            "event_id": event_id,
            "face_crop_path": "/local/snapshots/last_face_crop.jpg",
        })
    else:
        archive_unknown_frame(result["frame"])
        notify_home_assistant("unknown_person_detected", {
            "highest_match": best_identity if best_identity else "none",
            "score": round(best_score, 4),
            "threshold": threshold,
            "device_id": device_id,
            "event_id": event_id,
            "face_crop_path": "/local/snapshots/last_face_crop.jpg",
        })

    return jsonify({"identity": best_identity, "score": round(best_score, 4), "matched": matched})


@app.route("/learn", methods=["POST"])
def learn():
    data = request.get_json(silent=True) or {}
    image_path = data.get("image_path")
    name = data.get("name")

    if not image_path or not name:
        return jsonify({"error": "image_path and name are required"}), 400

    image_file = Path(image_path)
    if not image_file.is_file():
        return jsonify({"error": f"image not found: {image_path}"}), 400

    identity_name = name.strip().lower().replace(" ", "_")

    img = cv2.imread(str(image_file))
    if img is None:
        return jsonify({"error": f"failed to decode image: {image_path}"}), 400

    engine = get_engine()
    embedding = engine.extract_best_embedding(img)
    if embedding is None:
        return jsonify({"error": "no face detected in image"}), 400

    output_dir = Path(EMBEDDINGS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"{identity_name}.npy", embedding)

    processed_dir = Path(UNKNOWN_FACES_DIR) / "learned"
    processed_dir.mkdir(parents=True, exist_ok=True)
    try:
        shutil.move(str(image_file), str(processed_dir / f"{identity_name}_{image_file.name}"))
    except Exception as e:
        print(f"[WARNING] Impossibile archiviare la foto sorgente: {e}", file=sys.stderr)

    return jsonify({"status": "learned", "identity": identity_name})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

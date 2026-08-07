import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def client(monkeypatch, tmp_path):
    """Crea il client Flask di test con FaceRecognitionEngine mockato
    (evita di caricare il vero modello InsightFace nei test unitari)."""
    monkeypatch.setenv("EMBEDDINGS_DIR", str(tmp_path / "embeddings"))
    monkeypatch.setenv("UNKNOWN_FACES_DIR", str(tmp_path / "unknown_faces"))
    monkeypatch.setenv("CURRENT_PENDING_FACE_PATH", str(tmp_path / "current_pending_face.jpg"))

    with patch("recognition.FaceRecognitionEngine") as mock_engine_cls:
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        import analyze_server
        import importlib
        importlib.reload(analyze_server)
        analyze_server.app.config["TESTING"] = True
        analyze_server._engine = mock_engine
        yield analyze_server.app.test_client(), mock_engine


def test_health_endpoint_returns_ok(client):
    test_client, _ = client
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_analyze_missing_snapshot_path_returns_400(client):
    test_client, _ = client
    response = test_client.post("/analyze", json={})
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_analyze_nonexistent_snapshot_returns_400(client, tmp_path):
    test_client, _ = client
    response = test_client.post(
        "/analyze", json={"snapshot_path": str(tmp_path / "missing.jpg")}
    )
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_learn_missing_fields_returns_400(client):
    test_client, _ = client
    response = test_client.post("/learn", json={"name": "mario_rossi"})
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_learn_nonexistent_image_returns_400(client, tmp_path):
    test_client, _ = client
    response = test_client.post(
        "/learn",
        json={"image_path": str(tmp_path / "missing.jpg"), "name": "mario_rossi"},
    )
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_learn_saves_embedding_file(client, tmp_path):
    test_client, mock_engine = client
    embeddings_dir = Path(os.environ["EMBEDDINGS_DIR"])
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    fake_image = tmp_path / "unknown_20260101_000000.jpg"
    fake_image.write_bytes(b"\xff\xd8\xff" + b"0" * 2000)  # placeholder JPEG-like bytes >1KB

    mock_engine.extract_best_embedding.return_value = np.array([1.0, 0.0, 0.0])

    with patch("analyze_server.cv2.imread", return_value=np.zeros((10, 10, 3), dtype=np.uint8)):
        response = test_client.post(
            "/learn", json={"image_path": str(fake_image), "name": "Mario Rossi"}
        )

    assert response.status_code == 200
    assert response.get_json() == {"status": "learned", "identity": "mario_rossi"}
    assert (embeddings_dir / "mario_rossi.npy").is_file()


# ---------------------------------------------------------------------------
# Tests for /analyze_event endpoint
# ---------------------------------------------------------------------------

def _make_fake_frames(n: int = 2):
    return [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(n)]


def _make_mock_face(embedding: np.ndarray):
    face = MagicMock()
    face.normed_embedding = embedding
    face.bbox = [10.0, 10.0, 50.0, 50.0]
    return face


def test_analyze_event_missing_device_id_returns_400(client):
    test_client, _ = client
    response = test_client.post("/analyze_event", json={"event_id": "evt123"})
    assert response.status_code == 400
    assert "device_id" in response.get_json().get("error", "")


def test_analyze_event_missing_event_id_returns_400(client):
    test_client, _ = client
    response = test_client.post("/analyze_event", json={"device_id": "dev123"})
    assert response.status_code == 400
    assert "event_id" in response.get_json().get("error", "")


def test_analyze_event_no_token_returns_400(client, monkeypatch):
    test_client, _ = client
    import analyze_server
    monkeypatch.setattr(analyze_server, "HA_BEARER_TOKEN", "")
    response = test_client.post("/analyze_event", json={"device_id": "dev123", "event_id": "evt123"})
    assert response.status_code == 400
    assert "SUPERVISOR_TOKEN" in response.get_json().get("error", "")


def test_analyze_event_ha_api_non_200_returns_502(client):
    test_client, _ = client
    import analyze_server
    mock_response = MagicMock()
    mock_response.status_code = 404
    with patch("analyze_server.requests.get", return_value=mock_response), \
         patch.object(analyze_server, "HA_BEARER_TOKEN", "fake-token"):
        response = test_client.post(
            "/analyze_event", json={"device_id": "dev123", "event_id": "evt123"}
        )
    assert response.status_code == 502
    assert "failed to fetch" in response.get_json().get("error", "")


def test_analyze_event_successful_match_returns_200(client, tmp_path, monkeypatch):
    test_client, mock_engine = client
    import analyze_server

    crop_path = str(tmp_path / "last_face_crop.jpg")
    monkeypatch.setattr(analyze_server, "CROP_OUTPUT_PATH", crop_path)

    fake_frames = _make_fake_frames(2)
    known_emb = np.array([1.0, 0.0, 0.0])
    mock_face = _make_mock_face(np.array([1.0, 0.0, 0.0]))  # perfect match
    mock_engine.analyze_image.return_value = [mock_face]

    # Save a known embedding
    emb_dir = Path(os.environ["EMBEDDINGS_DIR"])
    emb_dir.mkdir(parents=True, exist_ok=True)
    np.save(emb_dir / "mario_rossi.npy", known_emb)

    mock_http_resp = MagicMock()
    mock_http_resp.status_code = 200
    mock_http_resp.content = b"fake_video"

    with patch("analyze_server.requests.get", return_value=mock_http_resp), \
         patch("analyze_server.extract_frames_from_video", return_value=fake_frames), \
         patch("analyze_server.notify_home_assistant") as mock_notify, \
         patch("analyze_server.cv2.imwrite"), \
         patch.object(analyze_server, "HA_BEARER_TOKEN", "fake-token"):
        response = test_client.post(
            "/analyze_event",
            json={"device_id": "dev123", "event_id": "evt123", "threshold": 0.45},
        )

    assert response.status_code == 200
    data = response.get_json()
    assert data["matched"] is True
    assert data["identity"] == "mario_rossi"
    assert mock_notify.call_count == 1
    assert mock_notify.call_args[0][0] == "known_person_detected"


def test_analyze_event_no_face_detected_returns_matched_false(client, monkeypatch):
    test_client, mock_engine = client
    import analyze_server

    fake_frames = _make_fake_frames(2)
    mock_engine.analyze_image.return_value = []  # no faces in any frame

    mock_http_resp = MagicMock()
    mock_http_resp.status_code = 200
    mock_http_resp.content = b"fake_video"

    with patch("analyze_server.requests.get", return_value=mock_http_resp), \
         patch("analyze_server.extract_frames_from_video", return_value=fake_frames), \
         patch("analyze_server.notify_home_assistant") as mock_notify, \
         patch.object(analyze_server, "HA_BEARER_TOKEN", "fake-token"):
        response = test_client.post(
            "/analyze_event",
            json={"device_id": "dev123", "event_id": "evt123"},
        )

    assert response.status_code == 200
    data = response.get_json()
    assert data["matched"] is False
    assert data["identity"] is None
    assert data["score"] == -1.0
    mock_notify.assert_called_once()
    assert mock_notify.call_args[0][0] == "no_face_detected"


def test_analyze_event_no_frames_decoded_returns_400(client):
    test_client, _ = client
    import analyze_server

    mock_http_resp = MagicMock()
    mock_http_resp.status_code = 200
    mock_http_resp.content = b"fake_video"

    with patch("analyze_server.requests.get", return_value=mock_http_resp), \
         patch("analyze_server.extract_frames_from_video", return_value=[]), \
         patch.object(analyze_server, "HA_BEARER_TOKEN", "fake-token"):
        response = test_client.post(
            "/analyze_event",
            json={"device_id": "dev123", "event_id": "evt123"},
        )

    assert response.status_code == 400
    assert "no frames" in response.get_json().get("error", "")


# ---------------------------------------------------------------------------
# Tests for the unknown faces FIFO queue (/unknown_faces/queue, /skip, /discard)
# ---------------------------------------------------------------------------

def _write_fake_face(directory: Path, name: str, mtime_offset_seconds: float = 0) -> Path:
    """Crea un file jpg fittizio con una mtime controllabile per testare l'ordine FIFO."""
    directory.mkdir(parents=True, exist_ok=True)
    file_path = directory / name
    file_path.write_bytes(b"\xff\xd8\xff" + b"0" * 2000)
    if mtime_offset_seconds:
        current_time = os.path.getmtime(file_path)
        new_time = current_time + mtime_offset_seconds
        os.utime(file_path, (new_time, new_time))
    return file_path


def test_enforce_file_cap_removes_oldest_files_over_limit(client, tmp_path):
    test_client, _ = client
    import analyze_server

    directory = tmp_path / "capped"
    directory.mkdir()
    for i in range(5):
        _write_fake_face(directory, f"unknown_{i}.jpg", mtime_offset_seconds=i)

    analyze_server._enforce_file_cap(directory, "unknown_*.jpg", max_files=3)

    remaining = sorted(p.name for p in directory.glob("unknown_*.jpg"))
    assert remaining == ["unknown_2.jpg", "unknown_3.jpg", "unknown_4.jpg"]


def test_enforce_file_cap_noop_when_under_limit(client, tmp_path):
    test_client, _ = client
    import analyze_server

    directory = tmp_path / "capped"
    directory.mkdir()
    _write_fake_face(directory, "unknown_0.jpg")

    analyze_server._enforce_file_cap(directory, "unknown_*.jpg", max_files=30)

    assert len(list(directory.glob("unknown_*.jpg"))) == 1


def test_unknown_faces_queue_empty_returns_zero_count(client):
    test_client, _ = client
    response = test_client.get("/unknown_faces/queue")
    assert response.status_code == 200
    assert response.get_json() == {
        "pending_count": 0,
        "current_filename": None,
        "current_path": None,
        "image_url": None,
    }


def test_unknown_faces_queue_returns_oldest_file_as_current(client, tmp_path):
    test_client, _ = client
    unknown_dir = Path(os.environ["UNKNOWN_FACES_DIR"])
    older = _write_fake_face(unknown_dir, "unknown_older.jpg", mtime_offset_seconds=0)
    _write_fake_face(unknown_dir, "unknown_newer.jpg", mtime_offset_seconds=100)

    response = test_client.get("/unknown_faces/queue")
    data = response.get_json()

    assert data["pending_count"] == 2
    assert data["current_filename"] == "unknown_older.jpg"
    assert data["current_path"] == str(older)
    assert data["image_url"] == "/local/snapshots/current_pending_face.jpg"


def test_unknown_faces_skip_empty_queue_returns_404(client):
    test_client, _ = client
    response = test_client.post("/unknown_faces/skip")
    assert response.status_code == 404
    assert response.get_json() == {"error": "no pending unknown faces"}


def test_unknown_faces_skip_moves_current_to_back_of_queue(client):
    test_client, _ = client
    unknown_dir = Path(os.environ["UNKNOWN_FACES_DIR"])
    older = _write_fake_face(unknown_dir, "unknown_older.jpg", mtime_offset_seconds=0)
    _write_fake_face(unknown_dir, "unknown_newer.jpg", mtime_offset_seconds=100)

    response = test_client.post("/unknown_faces/skip")
    data = response.get_json()

    assert data["pending_count"] == 2
    assert data["current_filename"] == "unknown_newer.jpg"
    assert data["current_path"] == str(unknown_dir / "unknown_newer.jpg")
    assert data["image_url"]
    # Il file "skippato" deve esistere ancora (non viene cancellato)
    assert older.is_file()


def test_unknown_faces_discard_empty_queue_returns_404(client):
    test_client, _ = client
    response = test_client.post("/unknown_faces/discard")
    assert response.status_code == 404
    assert response.get_json() == {"error": "no pending unknown faces"}


def test_unknown_faces_discard_deletes_current_file(client):
    test_client, _ = client
    unknown_dir = Path(os.environ["UNKNOWN_FACES_DIR"])
    older = _write_fake_face(unknown_dir, "unknown_older.jpg", mtime_offset_seconds=0)
    _write_fake_face(unknown_dir, "unknown_newer.jpg", mtime_offset_seconds=100)

    response = test_client.post("/unknown_faces/discard")
    data = response.get_json()

    assert data["pending_count"] == 1
    assert data["current_filename"] == "unknown_newer.jpg"
    assert data["current_path"] == str(unknown_dir / "unknown_newer.jpg")
    assert data["image_url"]
    assert not older.is_file()


def test_learn_enforces_cap_on_learned_dir(client, tmp_path, monkeypatch):
    test_client, mock_engine = client
    monkeypatch.setattr("analyze_server.MAX_SAVED_FACES", 2)

    unknown_dir = Path(os.environ["UNKNOWN_FACES_DIR"])
    learned_dir = unknown_dir / "learned"
    learned_dir.mkdir(parents=True, exist_ok=True)
    for i in range(2):
        _write_fake_face(learned_dir, f"existing_{i}.jpg", mtime_offset_seconds=i)

    fake_image = tmp_path / "unknown_new.jpg"
    fake_image.write_bytes(b"\xff\xd8\xff" + b"0" * 2000)
    mock_engine.extract_best_embedding.return_value = np.array([1.0, 0.0, 0.0])

    with patch("analyze_server.cv2.imread", return_value=np.zeros((10, 10, 3), dtype=np.uint8)):
        response = test_client.post(
            "/learn", json={"image_path": str(fake_image), "name": "Mario Rossi"}
        )

    assert response.status_code == 200
    remaining = sorted(p.name for p in learned_dir.glob("*.jpg"))
    assert len(remaining) == 2
    # existing_0.jpg has the oldest mtime and must be the one evicted by the cap.
    assert "existing_0.jpg" not in remaining
    assert "existing_1.jpg" in remaining



@pytest.fixture
def client(monkeypatch, tmp_path):
    """Crea il client Flask di test con FaceRecognitionEngine mockato
    (evita di caricare il vero modello InsightFace nei test unitari)."""
    monkeypatch.setenv("EMBEDDINGS_DIR", str(tmp_path / "embeddings"))
    monkeypatch.setenv("UNKNOWN_FACES_DIR", str(tmp_path / "unknown_faces"))
    monkeypatch.setenv("CURRENT_PENDING_FACE_PATH", str(tmp_path / "current_pending_face.jpg"))

    with patch("recognition.FaceRecognitionEngine") as mock_engine_cls:
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        import analyze_server
        import importlib
        importlib.reload(analyze_server)
        analyze_server.app.config["TESTING"] = True
        analyze_server._engine = mock_engine
        yield analyze_server.app.test_client(), mock_engine


def test_health_endpoint_returns_ok(client):
    test_client, _ = client
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_analyze_missing_snapshot_path_returns_400(client):
    test_client, _ = client
    response = test_client.post("/analyze", json={})
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_analyze_nonexistent_snapshot_returns_400(client, tmp_path):
    test_client, _ = client
    response = test_client.post(
        "/analyze", json={"snapshot_path": str(tmp_path / "missing.jpg")}
    )
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_learn_missing_fields_returns_400(client):
    test_client, _ = client
    response = test_client.post("/learn", json={"name": "mario_rossi"})
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_learn_nonexistent_image_returns_400(client, tmp_path):
    test_client, _ = client
    response = test_client.post(
        "/learn",
        json={"image_path": str(tmp_path / "missing.jpg"), "name": "mario_rossi"},
    )
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_learn_saves_embedding_file(client, tmp_path):
    test_client, mock_engine = client
    embeddings_dir = Path(os.environ["EMBEDDINGS_DIR"])
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    fake_image = tmp_path / "unknown_20260101_000000.jpg"
    fake_image.write_bytes(b"\xff\xd8\xff" + b"0" * 2000)  # placeholder JPEG-like bytes >1KB

    mock_engine.extract_best_embedding.return_value = np.array([1.0, 0.0, 0.0])

    with patch("analyze_server.cv2.imread", return_value=np.zeros((10, 10, 3), dtype=np.uint8)):
        response = test_client.post(
            "/learn", json={"image_path": str(fake_image), "name": "Mario Rossi"}
        )

    assert response.status_code == 200
    assert response.get_json() == {"status": "learned", "identity": "mario_rossi"}
    assert (embeddings_dir / "mario_rossi.npy").is_file()

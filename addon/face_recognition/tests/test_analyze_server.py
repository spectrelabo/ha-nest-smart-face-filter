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



@pytest.fixture
def client(monkeypatch, tmp_path):
    """Crea il client Flask di test con FaceRecognitionEngine mockato
    (evita di caricare il vero modello InsightFace nei test unitari)."""
    monkeypatch.setenv("EMBEDDINGS_DIR", str(tmp_path / "embeddings"))
    monkeypatch.setenv("UNKNOWN_FACES_DIR", str(tmp_path / "unknown_faces"))

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

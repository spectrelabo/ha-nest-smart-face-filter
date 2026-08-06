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


def test_learn_saves_embedding_file(client, tmp_path, monkeypatch):
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

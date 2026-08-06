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

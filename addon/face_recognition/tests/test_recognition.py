# addon/face_recognition/tests/test_recognition.py
import os
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recognition import load_known_embeddings, cosine_similarity, extract_frames_from_video, analyze_frames_for_best_match


def test_cosine_similarity_identical_vectors_returns_one():
    vec = np.array([1.0, 0.0, 0.0])
    assert cosine_similarity(vec, vec) == 1.0


def test_cosine_similarity_orthogonal_vectors_returns_zero():
    vec1 = np.array([1.0, 0.0])
    vec2 = np.array([0.0, 1.0])
    assert cosine_similarity(vec1, vec2) == 0.0


def test_cosine_similarity_zero_vector_returns_zero():
    vec1 = np.array([0.0, 0.0])
    vec2 = np.array([1.0, 0.0])
    assert cosine_similarity(vec1, vec2) == 0.0


def test_load_known_embeddings_reads_normalized_npy_files(tmp_path):
    raw_vec = np.array([3.0, 4.0])  # norma = 5
    np.save(tmp_path / "mario_rossi.npy", raw_vec)

    result = load_known_embeddings(tmp_path)

    assert set(result.keys()) == {"mario_rossi"}
    np.testing.assert_allclose(result["mario_rossi"], np.array([0.6, 0.8]))


def test_load_known_embeddings_missing_dir_returns_empty_dict(tmp_path):
    missing_dir = tmp_path / "does_not_exist"
    result = load_known_embeddings(missing_dir)
    assert result == {}


# ---------------------------------------------------------------------------
# Tests for extract_frames_from_video
# ---------------------------------------------------------------------------

def _make_mp4_bytes(tmp_path, num_frames: int = 5, width: int = 64, height: int = 64) -> bytes:
    """Genera un breve file MP4 in memoria usando cv2.VideoWriter."""
    import cv2
    out_path = str(tmp_path / "test_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, 10.0, (width, height))
    for i in range(num_frames):
        color = (i * 50 % 256, 100, 200)
        frame = np.full((height, width, 3), color, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    with open(out_path, "rb") as f:
        return f.read()


def test_extract_frames_from_video_returns_frames(tmp_path):
    video_bytes = _make_mp4_bytes(tmp_path, num_frames=5)
    frames = extract_frames_from_video(video_bytes, max_frames=30)
    assert len(frames) == 5
    assert all(isinstance(f, np.ndarray) for f in frames)


def test_extract_frames_from_video_samples_uniformly_when_exceeds_max(tmp_path):
    video_bytes = _make_mp4_bytes(tmp_path, num_frames=20)
    frames = extract_frames_from_video(video_bytes, max_frames=5)
    assert len(frames) == 5


def test_extract_frames_from_video_invalid_bytes_returns_empty():
    frames = extract_frames_from_video(b"not a valid mp4", max_frames=30)
    assert frames == []


def test_extract_frames_from_video_empty_bytes_returns_empty():
    frames = extract_frames_from_video(b"", max_frames=30)
    assert frames == []


# ---------------------------------------------------------------------------
# Tests for analyze_frames_for_best_match
# ---------------------------------------------------------------------------

def _make_mock_face(embedding: np.ndarray):
    face = MagicMock()
    face.normed_embedding = embedding
    face.bbox = [10, 10, 50, 50]
    return face


def test_analyze_frames_no_faces_returns_none_frame():
    engine = MagicMock()
    engine.analyze_image.return_value = []
    frames = [np.zeros((64, 64, 3), dtype=np.uint8)] * 3
    result = analyze_frames_for_best_match(engine, frames, {"mario": np.array([1.0, 0.0, 0.0])})
    assert result["frame"] is None
    assert result["identity"] is None
    assert result["score"] == -1.0
    assert result["faces"] == []


def test_analyze_frames_picks_best_match_across_frames():
    engine = MagicMock()

    emb_low = np.array([0.0, 1.0, 0.0])   # cosine sim with mario = 0.0
    emb_high = np.array([1.0, 0.0, 0.0])  # cosine sim with mario = 1.0

    face_low = _make_mock_face(emb_low)
    face_high = _make_mock_face(emb_high)

    frame_low = np.zeros((64, 64, 3), dtype=np.uint8)
    frame_high = np.ones((64, 64, 3), dtype=np.uint8) * 128

    engine.analyze_image.side_effect = [
        [face_low],   # frame 0 → low score
        [face_high],  # frame 1 → high score
    ]

    known_faces = {"mario": np.array([1.0, 0.0, 0.0])}
    result = analyze_frames_for_best_match(engine, [frame_low, frame_high], known_faces)

    assert result["identity"] == "mario"
    assert result["score"] == pytest.approx(1.0, abs=1e-5)
    assert result["frame"] is frame_high


def test_analyze_frames_no_known_faces_returns_first_frame_with_face():
    engine = MagicMock()
    face = _make_mock_face(np.array([1.0, 0.0, 0.0]))
    frame_with_face = np.ones((64, 64, 3), dtype=np.uint8)

    engine.analyze_image.side_effect = [
        [],              # frame 0: no face
        [face],          # frame 1: face detected
    ]

    result = analyze_frames_for_best_match(engine, [np.zeros((64, 64, 3), dtype=np.uint8), frame_with_face], {})

    # With no known embeddings, no identity match — but frame should still be set
    assert result["frame"] is frame_with_face
    assert result["identity"] is None
    assert result["score"] == -1.0

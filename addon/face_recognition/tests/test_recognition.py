# addon/face_recognition/tests/test_recognition.py
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recognition import load_known_embeddings, cosine_similarity


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

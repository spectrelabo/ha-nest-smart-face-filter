#!/usr/bin/env python3
"""
create_embeddings.py - Script to extract and save the embedding of a known face.

Uses the InsightFace `buffalo_s` model on CPU (ONNX Runtime) to generate
a 512-dimensional NumPy array (.npy) for the specified identity.
"""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np

try:
    import insightface
    from insightface.app import FaceAnalysis
except ImportError:
    print("[ERROR] Unable to import 'insightface'. Install the dependencies with: pip install insightface onnxruntime opencv-python-headless numpy", file=sys.stderr)
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract the 512-dimensional face embedding from an image and save it as a .npy file"
    )
    parser.add_argument(
        "--image",
        "-i",
        required=True,
        type=str,
        help="Path of the source image containing the face (e.g. /path/to/image.jpg)"
    )
    parser.add_argument(
        "--name",
        "-n",
        required=True,
        type=str,
        help="Name or identifier of the person (e.g. mario_rossi)"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="/config/face_recognition/embeddings",
        type=str,
        help="Destination directory for the .npy file (default: /config/face_recognition/embeddings)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = Path(args.image)
    output_dir = Path(args.output_dir)
    identity_name = args.name.strip().lower().replace(" ", "_")

    if not image_path.is_file():
        print(f"[ERROR] Image not found at path: {image_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Initializing InsightFace model 'buffalo_s' (CPUExecutionProvider)...")
    app = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    print(f"[INFO] Loading image: {image_path}")
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[ERROR] Unable to decode the image with OpenCV: {image_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Running detection and feature extraction...")
    faces = app.get(img)

    if not faces:
        print(f"[ERROR] No face detected in image {image_path}.", file=sys.stderr)
        sys.exit(1)

    if len(faces) > 1:
        print(f"[WARNING] Found {len(faces)} faces. The primary face (largest bounding box) will be used.")

    # Select the face with the largest bounding box
    best_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    embedding = best_face.normed_embedding

    if embedding is None or len(embedding) == 0:
        print(f"[ERROR] Embedding extraction failed for image {image_path}.", file=sys.stderr)
        sys.exit(1)

    # Create the output directory if it does not exist
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{identity_name}.npy"

    np.save(output_file, embedding)
    print(f"[SUCCESS] Embedding for '{identity_name}' saved successfully!")
    print(f"          Path: {output_file}")
    print(f"          Vector size: {embedding.shape}")


if __name__ == "__main__":
    main()

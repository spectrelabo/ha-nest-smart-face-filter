#!/usr/bin/env python3
"""
learn_face.py - [LEGACY/DEPRECATED] CLI helper to learn new faces.

THIS SCRIPT IS NO LONGER THE RECOMMENDED PATH. Use the
ha_nest_smart_face_filter.learn_face service instead (it calls the local
"Face Recognition" add-on's POST /learn endpoint). This file remains in the
repo only for users who already have a working manual setup via SSH.
"""

import argparse
import sys
import shutil
from pathlib import Path

# Default paths
UNKNOWN_DIR = Path("/config/face_recognition/unknown_faces")
EMBEDDINGS_DIR = Path("/config/face_recognition/embeddings")
CREATE_SCRIPT = Path(__file__).parent / "create_embeddings.py"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert an unknown-face snapshot into a new known-face profile (.npy)"
    )
    parser.add_argument(
        "--image",
        "-i",
        type=str,
        help="Path of the archived image to learn from (e.g. /config/face_recognition/unknown_faces/unknown_20260805_110000.jpg)"
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Use the most recent unknown-face snapshot"
    )
    parser.add_argument(
        "--name",
        "-n",
        required=True,
        type=str,
        help="Name of the identity to assign to the face (e.g. giuseppe_verdi)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    identity_name = args.name.strip().lower().replace(" ", "_")

    if args.latest:
        saved_files = sorted(UNKNOWN_DIR.glob("unknown_*.jpg"), key=lambda f: f.stat().st_mtime, reverse=True)
        if not saved_files:
            print(f"[ERROR] No unknown-face snapshot found in: {UNKNOWN_DIR}", file=sys.stderr)
            sys.exit(1)
        target_image = saved_files[0]
        print(f"[INFO] Selected the most recent snapshot: {target_image.name}")
    elif args.image:
        target_image = Path(args.image)
    else:
        print("[ERROR] Specify --image or --latest to select the photo to learn from.", file=sys.stderr)
        sys.exit(1)

    if not target_image.is_file():
        print(f"[ERROR] Image not found: {target_image}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Generating embedding for identity '{identity_name}' from image {target_image.name}...")

    # Import create_embeddings or call it through a subprocess
    cmd = [
        sys.executable,
        str(CREATE_SCRIPT),
        "--image", str(target_image),
        "--name", identity_name,
        "--output-dir", str(EMBEDDINGS_DIR)
    ]

    import subprocess
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print(f"[SUCCESS] New face '{identity_name}' learned and registered successfully!")
        # Move the processed photo into the learned archive or remove it
        processed_dir = UNKNOWN_DIR / "learned"
        processed_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(target_image), str(processed_dir / f"{identity_name}_{target_image.name}"))
        print(f"[INFO] Source photo archived in: {processed_dir}")
    else:
        print(f"[ERROR] Error while extracting the embedding for '{identity_name}'.", file=sys.stderr)


if __name__ == "__main__":
    main()

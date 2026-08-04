#!/usr/bin/env python3
"""
create_embeddings.py - Script per estrarre e salvare l'embedding di un volto noto.

Utilizza il modello InsightFace `buffalo_s` su CPU (ONNX Runtime) per generare 
un array NumPy (.npy) a 512 dimensioni per l'identità specificata.
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
    print("[ERROR] Impossibile importare 'insightface'. Installa le dipendenze con: pip install insightface onnxruntime opencv-python-headless numpy", file=sys.stderr)
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estrae l'embedding facciale a 512 dimensioni da un'immagine e lo salva come file .npy"
    )
    parser.add_argument(
        "--image",
        "-i",
        required=True,
        type=str,
        help="Percorso dell'immagine sorgente contenente il volto (es. /path/to/image.jpg)"
    )
    parser.add_argument(
        "--name",
        "-n",
        required=True,
        type=str,
        help="Nome o identificativo della persona (es. mario_rossi)"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="/config/face_recognition/embeddings",
        type=str,
        help="Directory di destinazione per il file .npy (default: /config/face_recognition/embeddings)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = Path(args.image)
    output_dir = Path(args.output_dir)
    identity_name = args.name.strip().lower().replace(" ", "_")

    if not image_path.is_file():
        print(f"[ERROR] Immagine non trovata al percorso: {image_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Inizializzazione modello InsightFace 'buffalo_s' (CPUExecutionProvider)...")
    app = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    print(f"[INFO] Caricamento immagine: {image_path}")
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[ERROR] Impossibile decodificare l'immagine con OpenCV: {image_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Esecuzione detection e feature extraction...")
    faces = app.get(img)

    if not faces:
        print(f"[ERROR] Nessun volto rilevato nell'immagine {image_path}.", file=sys.stderr)
        sys.exit(1)

    if len(faces) > 1:
        print(f"[WARNING] Trovati {len(faces)} volti. Verrà utilizzato il volto primario (con bounding box più grande).")

    # Seleziona il volto con il bounding box più ampio
    best_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    embedding = best_face.normed_embedding

    if embedding is None or len(embedding) == 0:
        print(f"[ERROR] Estrazione dell'embedding fallita per l'immagine {image_path}.", file=sys.stderr)
        sys.exit(1)

    # Crea la directory di output se non esiste
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{identity_name}.npy"

    np.save(output_file, embedding)
    print(f"[SUCCESS] Embedding per '{identity_name}' salvato con successo!")
    print(f"          Percorso: {output_file}")
    print(f"          Dimensione vector: {embedding.shape}")


if __name__ == "__main__":
    main()

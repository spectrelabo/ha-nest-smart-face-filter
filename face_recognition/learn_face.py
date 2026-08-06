#!/usr/bin/env python3
"""
learn_face.py - [LEGACY/DEPRECATO] Helper CLI per apprendere nuovi volti.

QUESTO SCRIPT NON E' PIU' IL PERCORSO RACCOMANDATO. Usa invece il servizio
ha_nest_smart_face_filter.learn_face (chiama l'endpoint POST /learn del
Local Add-on "Face Recognition"). Questo file resta nel repo solo per chi
ha già un setup manuale funzionante via SSH.
"""

import argparse
import sys
import shutil
from pathlib import Path

# Percorsi predefiniti
UNKNOWN_DIR = Path("/config/face_recognition/unknown_faces")
EMBEDDINGS_DIR = Path("/config/face_recognition/embeddings")
CREATE_SCRIPT = Path(__file__).parent / "create_embeddings.py"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Converte uno snapshot di volto sconosciuto in un nuovo profilo di volto noto (.npy)"
    )
    parser.add_argument(
        "--image",
        "-i",
        type=str,
        help="Percorso dell'immagine archiviata da apprendere (es. /config/face_recognition/unknown_faces/unknown_20260805_110000.jpg)"
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Utilizza l'ultimo snapshot di volto sconosciuto più recente"
    )
    parser.add_argument(
        "--name",
        "-n",
        required=True,
        type=str,
        help="Nome dell'identità da assegnare al volto (es. giuseppe_verdi)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    identity_name = args.name.strip().lower().replace(" ", "_")

    if args.latest:
        saved_files = sorted(UNKNOWN_DIR.glob("unknown_*.jpg"), key=lambda f: f.stat().st_mtime, reverse=True)
        if not saved_files:
            print(f"[ERROR] Nessuno snapshot di volto sconosciuto trovato in: {UNKNOWN_DIR}", file=sys.stderr)
            sys.exit(1)
        target_image = saved_files[0]
        print(f"[INFO] Selezionato l'ultimo snapshot più recente: {target_image.name}")
    elif args.image:
        target_image = Path(args.image)
    else:
        print("[ERROR] Specificare --image oppure --latest per selezionare la foto da apprendere.", file=sys.stderr)
        sys.exit(1)

    if not target_image.is_file():
        print(f"[ERROR] Immagine non trovata: {target_image}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Generazione embedding per l'identità '{identity_name}' dall'immagine {target_image.name}...")

    # Esecuzione import di create_embeddings o chiamata subprocess
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
        print(f"[SUCCESS] Nuovo volto '{identity_name}' appreso e registrato con successo!")
        # Sposta la foto processata in archivio elaborati o rimuovila
        processed_dir = UNKNOWN_DIR / "learned"
        processed_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(target_image), str(processed_dir / f"{identity_name}_{target_image.name}"))
        print(f"[INFO] Foto di origine archiviata in: {processed_dir}")
    else:
        print(f"[ERROR] Errore durante l'estrazione dell'embedding per '{identity_name}'.", file=sys.stderr)


if __name__ == "__main__":
    main()

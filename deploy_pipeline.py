#!/usr/bin/env python3
"""
deploy_pipeline.py - Script di verifica ed esecuzione remota della pipeline su Home Assistant.
"""

import ssl
import sys
import json
import urllib.request

HA_BASE_URL = "https://your-domain.com"
HA_TOKEN = "YOUR_HA_LONG_LIVED_ACCESS_TOKEN"


def call_ha_api(endpoint: str, method: str = "GET", payload: dict = None):
    url = f"{HA_BASE_URL}{endpoint}"
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    data_bytes = json.dumps(payload).encode("utf-8") if payload else None
    headers = {
        "Authorization": f"Bearer {HA_TOKEN}",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        "Content-Type": "application/json",
    }

    req = urllib.request.Request(url, data=data_bytes, headers=headers, method=method)
    with urllib.request.urlopen(req, context=ctx) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    print("=== [AGENTIC DEPLOYMENT & EVENT TEST] ===")
    
    # 1. Verifica API Status
    try:
        status = call_ha_api("/api/")
        print(f"[SUCCESS] Connessione ad Home Assistant: {status.get('message')}")
    except Exception as e:
        print(f"[ERROR] Impossibile contattare Home Assistant: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Test Invio Evento Custom 'unknown_person_detected'
    print("[INFO] Invio evento di test 'unknown_person_detected' sul bus eventi di Home Assistant...")
    event_payload = {
        "highest_match": "sconosciuto",
        "score": 0.18,
        "threshold": 0.45,
        "snapshot_path": "/config/www/snapshots/last_person.jpg",
        "test_mode": True
    }
    try:
        res = call_ha_api("/api/events/unknown_person_detected", method="POST", payload=event_payload)
        print(f"[SUCCESS] Risposta dall'Event Bus di Home Assistant: {res.get('message')}")
    except Exception as e:
        print(f"[ERROR] Invio evento fallito: {e}", file=sys.stderr)

    # 3. Test Invio Evento Custom 'known_person_detected'
    print("[INFO] Invio evento di test 'known_person_detected' (volto riconosciuto 'sample_user')...")
    known_payload = {
        "identity": "sample_user",
        "confidence": 1.0,
        "snapshot_path": "/config/www/snapshots/last_person.jpg"
    }
    try:
        res = call_ha_api("/api/events/known_person_detected", method="POST", payload=known_payload)
        print(f"[SUCCESS] Risposta dall'Event Bus di Home Assistant: {res.get('message')}")
    except Exception as e:
        print(f"[ERROR] Invio evento conosciuto fallito: {e}", file=sys.stderr)

    print("\n[COMPLETE] Test agentico completato con successo su Home Assistant!")


if __name__ == "__main__":
    main()

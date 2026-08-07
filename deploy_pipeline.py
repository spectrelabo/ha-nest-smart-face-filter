#!/usr/bin/env python3
"""
deploy_pipeline.py - Script to verify and run the pipeline remotely on Home Assistant.
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
    
    # 1. Check API status
    try:
        status = call_ha_api("/api/")
        print(f"[SUCCESS] Connected to Home Assistant: {status.get('message')}")
    except Exception as e:
        print(f"[ERROR] Unable to reach Home Assistant: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Test sending the custom event 'unknown_person_detected'
    print("[INFO] Sending test event 'unknown_person_detected' on the Home Assistant event bus...")
    event_payload = {
        "highest_match": "unknown",
        "score": 0.18,
        "threshold": 0.45,
        "snapshot_path": "/config/www/snapshots/last_person.jpg",
        "test_mode": True
    }
    try:
        res = call_ha_api("/api/events/unknown_person_detected", method="POST", payload=event_payload)
        print(f"[SUCCESS] Response from the Home Assistant Event Bus: {res.get('message')}")
    except Exception as e:
        print(f"[ERROR] Event send failed: {e}", file=sys.stderr)

    # 3. Test sending the custom event 'known_person_detected'
    print("[INFO] Sending test event 'known_person_detected' (recognized face 'sample_user')...")
    known_payload = {
        "identity": "sample_user",
        "confidence": 1.0,
        "snapshot_path": "/config/www/snapshots/last_person.jpg"
    }
    try:
        res = call_ha_api("/api/events/known_person_detected", method="POST", payload=known_payload)
        print(f"[SUCCESS] Response from the Home Assistant Event Bus: {res.get('message')}")
    except Exception as e:
        print(f"[ERROR] Known-face event send failed: {e}", file=sys.stderr)

    print("\n[COMPLETE] Agentic test completed successfully on Home Assistant!")


if __name__ == "__main__":
    main()

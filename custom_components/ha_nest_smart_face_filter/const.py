"""Constants for the HA Nest Smart Face Filter integration."""

DOMAIN = "ha_nest_smart_face_filter"
DEFAULT_NAME = "HA Nest Smart Face Filter"

CONF_ADDON_HOST = "addon_host"
CONF_EMBEDDINGS_DIR = "embeddings_dir"
CONF_SNAPSHOT_PATH = "snapshot_path"
CONF_THRESHOLD = "threshold"

# HA Supervisor's internal DNS uses hyphens for add-on slugs (underscores → hyphens).
# The `a0d7b954` prefix is a Supervisor-generated repository hash placeholder — it will
# differ per installation. Verify the real hostname via Settings → Add-ons → Face Recognition
# → Info → Hostname and override it in the integration's Options if needed.
DEFAULT_ADDON_HOST = "http://a0d7b954-face-recognition:8080"

DEFAULT_EMBEDDINGS_DIR = "/config/face_recognition/embeddings"
DEFAULT_SNAPSHOT_PATH = "/config/www/last_person.jpg"
DEFAULT_THRESHOLD = 0.45

EVENT_UNKNOWN_PERSON_DETECTED = "unknown_person_detected"
EVENT_KNOWN_PERSON_DETECTED = "known_person_detected"
EVENT_NO_FACE_DETECTED = "no_face_detected"

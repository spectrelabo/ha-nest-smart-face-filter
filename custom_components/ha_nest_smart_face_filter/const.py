"""Constants for the HA Nest Smart Face Filter integration."""

DOMAIN = "ha_nest_smart_face_filter"
DEFAULT_NAME = "HA Nest Smart Face Filter"

CONF_EMBEDDINGS_DIR = "embeddings_dir"
CONF_SNAPSHOT_PATH = "snapshot_path"
CONF_THRESHOLD = "threshold"

DEFAULT_EMBEDDINGS_DIR = "/config/face_recognition/embeddings"
DEFAULT_SNAPSHOT_PATH = "/config/www/snapshots/last_person.jpg"
DEFAULT_THRESHOLD = 0.45

EVENT_UNKNOWN_PERSON_DETECTED = "unknown_person_detected"
EVENT_KNOWN_PERSON_DETECTED = "known_person_detected"
EVENT_NO_FACE_DETECTED = "no_face_detected"

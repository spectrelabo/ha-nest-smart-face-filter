"""The HA Nest Smart Face Filter integration."""
from __future__ import annotations

import logging
import asyncio
import os
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, ServiceCall

from .const import (
    DOMAIN,
    CONF_EMBEDDINGS_DIR,
    CONF_SNAPSHOT_PATH,
    CONF_THRESHOLD,
    DEFAULT_EMBEDDINGS_DIR,
    DEFAULT_SNAPSHOT_PATH,
    DEFAULT_THRESHOLD,
)

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up HA Nest Smart Face Filter from a config entry."""
    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = entry.data

    async def handle_analyze_snapshot(call: ServiceCall):
        """Service handler to trigger face recognition analysis."""
        snapshot = call.data.get(
            "snapshot_path",
            entry.data.get(CONF_SNAPSHOT_PATH, DEFAULT_SNAPSHOT_PATH),
        )
        threshold = call.data.get(
            "threshold", entry.data.get(CONF_THRESHOLD, DEFAULT_THRESHOLD)
        )
        embeddings_dir = entry.data.get(CONF_EMBEDDINGS_DIR, DEFAULT_EMBEDDINGS_DIR)

        _LOGGER.info("Analyzing snapshot: %s (threshold: %s)", snapshot, threshold)

        script_path = os.path.join(hass.config.config_dir, "face_recognition", "analyze.py")
        if os.path.exists(script_path):
            proc = await asyncio.create_subprocess_exec(
                "python3", script_path, "--image", snapshot, "--embeddings", embeddings_dir, "--threshold", str(threshold),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode == 0:
                _LOGGER.info("Face recognition completed: %s", stdout.decode().strip())
            else:
                _LOGGER.error("Face recognition failed: %s", stderr.decode().strip())
        else:
            _LOGGER.error("Analyze script not found at %s", script_path)

    hass.services.async_register(DOMAIN, "analyze_snapshot", handle_analyze_snapshot)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    hass.data[DOMAIN].pop(entry.entry_id, None)
    return True

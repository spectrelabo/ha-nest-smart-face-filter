"""The HA Nest Smart Face Filter integration."""
from __future__ import annotations

import logging
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
        _LOGGER.info("Executing HA Nest Smart Face Filter analysis service...")
        snapshot = call.data.get(
            "snapshot_path",
            entry.data.get(CONF_SNAPSHOT_PATH, DEFAULT_SNAPSHOT_PATH),
        )
        threshold = call.data.get(
            "threshold", entry.data.get(CONF_THRESHOLD, DEFAULT_THRESHOLD)
        )
        _LOGGER.info("Analyzing snapshot: %s (threshold: %s)", snapshot, threshold)

    hass.services.async_register(DOMAIN, "analyze_snapshot", handle_analyze_snapshot)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    hass.data[DOMAIN].pop(entry.entry_id, None)
    return True

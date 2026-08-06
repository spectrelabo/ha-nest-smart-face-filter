"""The HA Nest Smart Face Filter integration."""
from __future__ import annotations

import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .const import (
    DOMAIN,
    CONF_ADDON_HOST,
    CONF_EMBEDDINGS_DIR,
    CONF_SNAPSHOT_PATH,
    CONF_THRESHOLD,
    DEFAULT_ADDON_HOST,
    DEFAULT_EMBEDDINGS_DIR,
    DEFAULT_SNAPSHOT_PATH,
    DEFAULT_THRESHOLD,
)

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up HA Nest Smart Face Filter from a config entry."""
    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = entry.data

    def get_addon_host() -> str:
        return entry.data.get(CONF_ADDON_HOST, DEFAULT_ADDON_HOST)

    async def handle_analyze_snapshot(call: ServiceCall):
        """Service handler to trigger face recognition analysis via the add-on."""
        snapshot = call.data.get(
            "snapshot_path",
            entry.data.get(CONF_SNAPSHOT_PATH, DEFAULT_SNAPSHOT_PATH),
        )
        threshold = call.data.get(
            "threshold", entry.data.get(CONF_THRESHOLD, DEFAULT_THRESHOLD)
        )

        _LOGGER.info("Analyzing snapshot via add-on: %s (threshold: %s)", snapshot, threshold)

        session = async_get_clientsession(hass)
        url = f"{get_addon_host()}/analyze"
        try:
            async with session.post(
                url,
                json={"snapshot_path": snapshot, "threshold": threshold},
                timeout=30,
            ) as response:
                result = await response.json()
                if response.status == 200:
                    _LOGGER.info("Face recognition completed: %s", result)
                else:
                    _LOGGER.error("Face recognition add-on returned error: %s", result)
        except Exception as err:
            _LOGGER.error("Failed to reach face recognition add-on at %s: %s", url, err)

    async def handle_learn_face(call: ServiceCall):
        """Service handler to register a new known face via the add-on."""
        image_path = call.data["image_path"]
        name = call.data["name"]

        session = async_get_clientsession(hass)
        url = f"{get_addon_host()}/learn"
        try:
            async with session.post(
                url, json={"image_path": image_path, "name": name}, timeout=30
            ) as response:
                result = await response.json()
                if response.status == 200:
                    _LOGGER.info("Face learned: %s", result)
                else:
                    _LOGGER.error("Learn face add-on returned error: %s", result)
        except Exception as err:
            _LOGGER.error("Failed to reach face recognition add-on at %s: %s", url, err)

    hass.services.async_register(DOMAIN, "analyze_snapshot", handle_analyze_snapshot)
    hass.services.async_register(DOMAIN, "learn_face", handle_learn_face)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    hass.data[DOMAIN].pop(entry.entry_id, None)
    return True

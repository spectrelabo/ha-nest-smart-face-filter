"""The HA Nest Smart Face Filter integration."""
from __future__ import annotations

import logging

import aiohttp
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


async def _async_update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload the config entry when options are updated."""
    await hass.config_entries.async_reload(entry.entry_id)


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up HA Nest Smart Face Filter from a config entry."""
    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = entry.data

    def get_config() -> dict:
        """Merge config entry data with any options-flow overrides (options win)."""
        return {**entry.data, **entry.options}

    def get_addon_host() -> str:
        return get_config().get(CONF_ADDON_HOST, DEFAULT_ADDON_HOST)

    async def handle_analyze_snapshot(call: ServiceCall):
        """Service handler to trigger face recognition analysis via the add-on."""
        config = get_config()
        snapshot = call.data.get(
            "snapshot_path",
            config.get(CONF_SNAPSHOT_PATH, DEFAULT_SNAPSHOT_PATH),
        )
        threshold = call.data.get(
            "threshold", config.get(CONF_THRESHOLD, DEFAULT_THRESHOLD)
        )

        _LOGGER.info("Analyzing snapshot via add-on: %s (threshold: %s)", snapshot, threshold)

        session = async_get_clientsession(hass)
        url = f"{get_addon_host()}/analyze"
        try:
            async with session.post(
                url,
                json={"snapshot_path": snapshot, "threshold": threshold},
                timeout=aiohttp.ClientTimeout(total=30),
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
        session = async_get_clientsession(hass)
        url = f"{get_addon_host()}/learn"
        try:
            image_path = call.data["image_path"]
            name = call.data["name"]
            async with session.post(
                url, json={"image_path": image_path, "name": name}, timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                result = await response.json()
                if response.status == 200:
                    _LOGGER.info("Face learned: %s", result)
                else:
                    _LOGGER.error("Learn face add-on returned error: %s", result)
        except KeyError as err:
            _LOGGER.error("Missing required field for learn_face service call: %s", err)
        except Exception as err:
            _LOGGER.error("Failed to reach face recognition add-on at %s: %s", url, err)
    async def handle_analyze_event(call: ServiceCall):
        """Service handler to trigger face recognition on a Nest event clip via the add-on."""
        config = get_config()
        session = async_get_clientsession(hass)
        url = f"{get_addon_host()}/analyze_event"
        try:
            device_id = call.data["device_id"]
            event_id = call.data["event_id"]
            threshold = call.data.get("threshold", config.get(CONF_THRESHOLD, DEFAULT_THRESHOLD))
            _LOGGER.info(
                "Analyzing Nest event via add-on: device=%s event=%s threshold=%s",
                device_id, event_id, threshold,
            )
            async with session.post(
                url,
                json={"device_id": device_id, "event_id": event_id, "threshold": threshold},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                result = await response.json()
                if response.status == 200:
                    _LOGGER.info("Nest event face recognition completed: %s", result)
                else:
                    _LOGGER.error("Face recognition add-on returned error for analyze_event: %s", result)
        except KeyError as err:
            _LOGGER.error("Missing required field for analyze_event service call: %s", err)
        except Exception as err:
            _LOGGER.error("Failed to reach face recognition add-on at %s: %s", url, err)

    hass.services.async_register(DOMAIN, "analyze_snapshot", handle_analyze_snapshot)
    hass.services.async_register(DOMAIN, "analyze_event", handle_analyze_event)
    hass.services.async_register(DOMAIN, "learn_face", handle_learn_face)
    entry.async_on_unload(entry.add_update_listener(_async_update_listener))
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    hass.data[DOMAIN].pop(entry.entry_id, None)
    return True

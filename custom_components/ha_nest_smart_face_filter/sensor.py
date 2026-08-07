"""Sensor platform for HA Nest Smart Face Filter — coda dei volti sconosciuti."""
from __future__ import annotations

import logging
from datetime import timedelta

import aiohttp
from homeassistant.components.sensor import SensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import (
    CoordinatorEntity,
    DataUpdateCoordinator,
    UpdateFailed,
)

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

SCAN_INTERVAL = timedelta(seconds=60)


class PendingFacesCoordinator(DataUpdateCoordinator):
    """Coordinator che interroga l'endpoint /unknown_faces/queue dell'add-on."""

    def __init__(self, hass: HomeAssistant, addon_host: str) -> None:
        super().__init__(
            hass,
            _LOGGER,
            name="ha_nest_smart_face_filter_pending_faces",
            update_interval=SCAN_INTERVAL,
        )
        self.addon_host = addon_host

    async def _async_update_data(self) -> dict:
        session = async_get_clientsession(self.hass)
        url = f"{self.addon_host}/unknown_faces/queue"
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status != 200:
                    raise UpdateFailed(f"Add-on returned HTTP {response.status}")
                return await response.json()
        except Exception as err:
            raise UpdateFailed(f"Failed to reach add-on at {url}: {err}") from err


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback
) -> None:
    """Configura il sensore dei volti in attesa da una config entry."""
    coordinator = hass.data[DOMAIN][entry.entry_id]["coordinator"]
    async_add_entities([PendingFacesSensor(coordinator, entry)])


class PendingFacesSensor(CoordinatorEntity, SensorEntity):
    """Sensore che espone il conteggio dei volti sconosciuti in attesa di revisione."""

    _attr_name = "Volti in Attesa"
    _attr_icon = "mdi:face-recognition"

    def __init__(self, coordinator: PendingFacesCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator)
        self._attr_unique_id = f"{entry.entry_id}_pending_faces"

    @property
    def native_value(self):
        return (self.coordinator.data or {}).get("pending_count", 0)

    @property
    def extra_state_attributes(self):
        data = self.coordinator.data or {}
        return {
            "current_filename": data.get("current_filename"),
            "current_path": data.get("current_path"),
            "image_url": data.get("image_url"),
        }

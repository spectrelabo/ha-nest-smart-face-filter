"""Image platform for HA Nest Smart Face Filter.

Exposes the current pending unknown face as an authenticated HA image entity,
so that only authenticated users (and Lovelace dashboards) can view it.
The image is fetched on demand from the add-on's /unknown_faces/image endpoint
and served through Home Assistant's built-in image proxy – never stored under
/config/www or served without authentication.
"""
from __future__ import annotations

import logging
from datetime import datetime

import aiohttp
from homeassistant.components.image import ImageEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN, CONF_ADDON_HOST, DEFAULT_ADDON_HOST

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback
) -> None:
    """Set up the pending-face image entity from a config entry."""
    coordinator = hass.data[DOMAIN][entry.entry_id]["coordinator"]
    addon_host = {**entry.data, **entry.options}.get(CONF_ADDON_HOST, DEFAULT_ADDON_HOST)
    async_add_entities([PendingFaceImageEntity(hass, coordinator, entry, addon_host)])


class PendingFaceImageEntity(ImageEntity):
    """Image entity that proxies the current unknown face from the add-on.

    The image is only accessible to authenticated users through HA's image
    API (e.g. /api/image_proxy/image.<slug>). No file is written to www/.
    """

    _attr_icon = "mdi:face-recognition"
    _attr_content_type = "image/jpeg"

    def __init__(self, hass, coordinator, entry: ConfigEntry, addon_host: str) -> None:
        super().__init__(hass)
        self._coordinator = coordinator
        self._addon_host = addon_host
        self._attr_name = f"{entry.title} Pending Face"
        self._attr_unique_id = f"{entry.entry_id}_pending_face_image"
        self._image_url = f"{addon_host}/unknown_faces/image"
        # Track when the image last changed so HA knows to refresh the proxy cache.
        self._last_image_time: datetime | None = None

    @property
    def image_last_updated(self) -> datetime | None:
        """Return the time the image was last updated (drives cache invalidation)."""
        return self._last_image_time

    async def async_image(self) -> bytes | None:
        """Fetch and return the current pending face JPEG from the add-on."""
        session = async_get_clientsession(self.hass)
        try:
            async with session.get(
                self._image_url, timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    self._last_image_time = datetime.now()
                    return await response.read()
                if response.status == 404:
                    # No pending face in the queue — normal, not an error.
                    return None
                _LOGGER.warning(
                    "Add-on returned HTTP %s for pending face image", response.status
                )
                return None
        except Exception as err:
            _LOGGER.debug("Could not fetch pending face image from add-on: %s", err)
            return None

    async def async_added_to_hass(self) -> None:
        """Subscribe to coordinator updates to refresh image timestamp."""
        await super().async_added_to_hass()
        self.async_on_remove(
            self._coordinator.async_add_listener(self._handle_coordinator_update)
        )

    def _handle_coordinator_update(self) -> None:
        """Update the image timestamp when the coordinator detects a new face."""
        data = self._coordinator.data or {}
        if data.get("pending_count", 0) > 0:
            self._last_image_time = datetime.now()
        self.async_write_ha_state()

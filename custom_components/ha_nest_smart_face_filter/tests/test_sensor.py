"""Richiede pytest-homeassistant-custom-component installato nel venv di sviluppo."""
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


@pytest.mark.asyncio
async def test_coordinator_fetches_queue_state_from_addon(hass):
    from custom_components.ha_nest_smart_face_filter.sensor import PendingFacesCoordinator

    with patch(
        "custom_components.ha_nest_smart_face_filter.sensor.async_get_clientsession"
    ) as mock_session_getter:
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "pending_count": 2,
            "current_filename": "unknown_1.jpg",
            "current_path": "/config/face_recognition/unknown_faces/unknown_1.jpg",
            "image_url": "/local/snapshots/current_pending_face.jpg",
        })
        mock_session = MagicMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response
        mock_session.get.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_session_getter.return_value = mock_session

        coordinator = PendingFacesCoordinator(hass, "http://fake-addon:8080")
        await coordinator.async_refresh()

        assert coordinator.data["pending_count"] == 2
        assert coordinator.data["current_filename"] == "unknown_1.jpg"


@pytest.mark.asyncio
async def test_coordinator_marks_unavailable_when_addon_unreachable(hass):
    from custom_components.ha_nest_smart_face_filter.sensor import PendingFacesCoordinator
    from homeassistant.helpers.update_coordinator import UpdateFailed

    with patch(
        "custom_components.ha_nest_smart_face_filter.sensor.async_get_clientsession"
    ) as mock_session_getter:
        mock_session = MagicMock()
        mock_session.get.side_effect = Exception("connection refused")
        mock_session_getter.return_value = mock_session

        coordinator = PendingFacesCoordinator(hass, "http://fake-addon:8080")
        await coordinator.async_refresh()

        assert coordinator.last_update_success is False


def test_pending_faces_sensor_name_yields_slug_prefixed_entity_id():
    """The entity's derived object_id must be <entry_title_slug>_volti_in_attesa
    (e.g. sensor.nest_smart_face_filter_volti_in_attesa for the default entry
    title "Nest Smart Face Filter"), per the plan's documented entity contract,
    not a bare "volti_in_attesa" that would collide across config entries."""
    from homeassistant.util import slugify

    from custom_components.ha_nest_smart_face_filter.sensor import PendingFacesSensor

    coordinator = MagicMock()
    entry = MagicMock()
    entry.title = "Nest Smart Face Filter"
    entry.entry_id = "abc123"

    sensor = PendingFacesSensor(coordinator, entry)

    assert sensor.name == "Nest Smart Face Filter Volti in Attesa"
    assert slugify(sensor.name) == "nest_smart_face_filter_volti_in_attesa"
    assert sensor.unique_id == "abc123_pending_faces"

"""Richiede pytest-homeassistant-custom-component installato nel venv di sviluppo."""
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


@pytest.mark.asyncio
async def test_handle_analyze_snapshot_calls_addon_http_endpoint(hass):
    from custom_components.ha_nest_smart_face_filter import async_setup_entry
    from custom_components.ha_nest_smart_face_filter.const import DOMAIN
    from homeassistant.config_entries import ConfigEntryState

    entry = AsyncMock()
    entry.entry_id = "test_entry"
    entry.data = {}
    entry.options = {}
    entry.state = ConfigEntryState.SETUP_IN_PROGRESS

    with patch(
        "custom_components.ha_nest_smart_face_filter.async_get_clientsession"
    ) as mock_session_getter, patch(
        "custom_components.ha_nest_smart_face_filter.sensor.async_get_clientsession"
    ) as mock_sensor_session_getter, patch.object(
        hass.config_entries, "async_forward_entry_setups", new_callable=AsyncMock
    ):
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"identity": "alessio", "score": 0.9, "matched": True})
        mock_session = AsyncMock()
        mock_session.post.return_value.__aenter__.return_value = mock_response
        mock_session_getter.return_value = mock_session

        mock_queue_response = AsyncMock()
        mock_queue_response.status = 200
        mock_queue_response.json = AsyncMock(return_value={
            "pending_count": 0, "current_filename": None, "current_path": None, "image_url": None,
        })
        mock_sensor_session = MagicMock()
        mock_sensor_session.get.return_value.__aenter__.return_value = mock_queue_response
        mock_sensor_session.get.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_sensor_session_getter.return_value = mock_sensor_session

        await async_setup_entry(hass, entry)
        await hass.services.async_call(
            DOMAIN,
            "analyze_snapshot",
            {"snapshot_path": "/config/www/last_person.jpg"},
            blocking=True,
        )

        mock_session.post.assert_called_once()
        called_url = mock_session.post.call_args.args[0]
        assert called_url.endswith("/analyze")


@pytest.mark.asyncio
async def test_handle_skip_pending_face_calls_addon_and_refreshes_sensor(hass):
    from custom_components.ha_nest_smart_face_filter import async_setup_entry
    from custom_components.ha_nest_smart_face_filter.const import DOMAIN
    from custom_components.ha_nest_smart_face_filter.sensor import PendingFacesCoordinator
    from homeassistant.config_entries import ConfigEntryState

    entry = AsyncMock()
    entry.entry_id = "test_entry_skip"
    entry.data = {}
    entry.options = {}
    entry.state = ConfigEntryState.SETUP_IN_PROGRESS

    with patch(
        "custom_components.ha_nest_smart_face_filter.async_get_clientsession"
    ) as mock_session_getter, patch(
        "custom_components.ha_nest_smart_face_filter.sensor.async_get_clientsession"
    ) as mock_sensor_session_getter, patch.object(
        hass.config_entries, "async_forward_entry_setups", new_callable=AsyncMock
    ), patch.object(
        PendingFacesCoordinator, "async_request_refresh", new_callable=AsyncMock
    ) as mock_refresh:
        mock_queue_response = AsyncMock()
        mock_queue_response.status = 200
        mock_queue_response.json = AsyncMock(return_value={
            "pending_count": 1, "current_filename": "unknown_1.jpg",
            "current_path": "/config/face_recognition/unknown_faces/unknown_1.jpg",
            "image_url": "/local/snapshots/current_pending_face.jpg",
        })
        mock_sensor_session = MagicMock()
        mock_sensor_session.get.return_value.__aenter__.return_value = mock_queue_response
        mock_sensor_session.get.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_sensor_session_getter.return_value = mock_sensor_session

        mock_skip_response = AsyncMock()
        mock_skip_response.status = 200
        mock_skip_response.json = AsyncMock(return_value={
            "pending_count": 1, "current_filename": "unknown_2.jpg",
            "current_path": "/config/face_recognition/unknown_faces/unknown_2.jpg",
            "image_url": "/local/snapshots/current_pending_face.jpg",
        })
        mock_session = MagicMock()
        mock_session.post.return_value.__aenter__.return_value = mock_skip_response
        mock_session.post.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_session_getter.return_value = mock_session

        await async_setup_entry(hass, entry)
        await hass.services.async_call(DOMAIN, "skip_pending_face", {}, blocking=True)

        called_url = mock_session.post.call_args.args[0]
        assert called_url.endswith("/unknown_faces/skip")
        mock_refresh.assert_called_once()


@pytest.mark.asyncio
async def test_handle_discard_pending_face_calls_addon_and_refreshes_sensor(hass):
    from custom_components.ha_nest_smart_face_filter import async_setup_entry
    from custom_components.ha_nest_smart_face_filter.const import DOMAIN
    from custom_components.ha_nest_smart_face_filter.sensor import PendingFacesCoordinator
    from homeassistant.config_entries import ConfigEntryState

    entry = AsyncMock()
    entry.entry_id = "test_entry_discard"
    entry.data = {}
    entry.options = {}
    entry.state = ConfigEntryState.SETUP_IN_PROGRESS

    with patch(
        "custom_components.ha_nest_smart_face_filter.async_get_clientsession"
    ) as mock_session_getter, patch(
        "custom_components.ha_nest_smart_face_filter.sensor.async_get_clientsession"
    ) as mock_sensor_session_getter, patch.object(
        hass.config_entries, "async_forward_entry_setups", new_callable=AsyncMock
    ), patch.object(
        PendingFacesCoordinator, "async_request_refresh", new_callable=AsyncMock
    ) as mock_refresh:
        mock_queue_response = AsyncMock()
        mock_queue_response.status = 200
        mock_queue_response.json = AsyncMock(return_value={
            "pending_count": 0, "current_filename": None, "current_path": None, "image_url": None,
        })
        mock_sensor_session = MagicMock()
        mock_sensor_session.get.return_value.__aenter__.return_value = mock_queue_response
        mock_sensor_session.get.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_sensor_session_getter.return_value = mock_sensor_session

        mock_discard_response = AsyncMock()
        mock_discard_response.status = 200
        mock_discard_response.json = AsyncMock(return_value={
            "pending_count": 0, "current_filename": None, "current_path": None, "image_url": None,
        })
        mock_session = MagicMock()
        mock_session.post.return_value.__aenter__.return_value = mock_discard_response
        mock_session.post.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_session_getter.return_value = mock_session

        await async_setup_entry(hass, entry)
        await hass.services.async_call(DOMAIN, "discard_pending_face", {}, blocking=True)

        called_url = mock_session.post.call_args.args[0]
        assert called_url.endswith("/unknown_faces/discard")
        mock_refresh.assert_called_once()


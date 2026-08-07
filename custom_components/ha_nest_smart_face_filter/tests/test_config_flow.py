"""Requires pytest-homeassistant-custom-component installed in the development venv.

Regression test for a live bug found in production logs: opening the
integration's Options flow raised
`AttributeError: property 'config_entry' of 'OptionsFlowHandler' object has no
setter`. Since Home Assistant 2024.11, `OptionsFlow.config_entry` is a
read-only property computed by the base class from `self.hass`/`self.handler`;
assigning to it manually in `__init__` (as the old code did) crashes as soon
as HA tries to construct the flow.
"""
import sys
from pathlib import Path

import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


@pytest.mark.asyncio
async def test_options_flow_init_does_not_crash(hass, enable_custom_integrations):
    """The real production entry point: hass.config_entries.options.async_init.

    This exercises the exact code path (OptionsFlowManager.async_create_flow ->
    ConfigFlow.async_get_options_flow -> OptionsFlowHandler()) that crashed in
    the field, instead of constructing OptionsFlowHandler directly.
    """
    from custom_components.ha_nest_smart_face_filter.const import DOMAIN

    entry = MockConfigEntry(domain=DOMAIN, data={}, options={})
    entry.add_to_hass(hass)

    result = await hass.config_entries.options.async_init(entry.entry_id)

    assert result["type"] == "form"
    assert result["step_id"] == "init"

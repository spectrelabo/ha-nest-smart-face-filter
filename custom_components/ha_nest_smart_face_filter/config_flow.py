"""Config flow for HA Nest Smart Face Filter integration."""
from __future__ import annotations

import logging
import voluptuous as vol

from homeassistant import config_entries
from homeassistant.core import callback
import homeassistant.helpers.config_validation as cv

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


class ConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for HA Nest Smart Face Filter."""

    VERSION = 1

    async def async_step_user(self, user_input=None):
        """Handle the initial step."""
        errors = {}

        if user_input is not None:
            await self.async_set_unique_id(DOMAIN)
            self._abort_if_unique_id_configured()
            return self.async_create_entry(
                title="Nest Smart Face Filter",
                data=user_input,
            )

        data_schema = vol.Schema(
            {
                vol.Optional(
                    CONF_EMBEDDINGS_DIR, default=DEFAULT_EMBEDDINGS_DIR
                ): cv.string,
                vol.Optional(
                    CONF_SNAPSHOT_PATH, default=DEFAULT_SNAPSHOT_PATH
                ): cv.string,
                vol.Optional(
                    CONF_THRESHOLD, default=DEFAULT_THRESHOLD
                ): vol.All(vol.Coerce(float), vol.Range(min=0.1, max=0.9)),
            }
        )

        return self.async_show_form(
            step_id="user", data_schema=data_schema, errors=errors
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        return OptionsFlowHandler(config_entry)


class OptionsFlowHandler(config_entries.OptionsFlow):
    """Handle options flow for HA Nest Smart Face Filter."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        self.config_entry = config_entry

    async def async_step_init(self, user_input=None):
        """Manage the options."""
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        options = self.config_entry.options
        embeddings_dir = options.get(
            CONF_EMBEDDINGS_DIR,
            self.config_entry.data.get(CONF_EMBEDDINGS_DIR, DEFAULT_EMBEDDINGS_DIR),
        )
        snapshot_path = options.get(
            CONF_SNAPSHOT_PATH,
            self.config_entry.data.get(CONF_SNAPSHOT_PATH, DEFAULT_SNAPSHOT_PATH),
        )
        threshold = options.get(
            CONF_THRESHOLD,
            self.config_entry.data.get(CONF_THRESHOLD, DEFAULT_THRESHOLD),
        )

        options_schema = vol.Schema(
            {
                vol.Optional(CONF_EMBEDDINGS_DIR, default=embeddings_dir): cv.string,
                vol.Optional(CONF_SNAPSHOT_PATH, default=snapshot_path): cv.string,
                vol.Optional(
                    CONF_THRESHOLD, default=threshold
                ): vol.All(vol.Coerce(float), vol.Range(min=0.1, max=0.9)),
            }
        )

        return self.async_show_form(step_id="init", data_schema=options_schema)

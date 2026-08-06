"""Root conftest.py to ensure the repo root is on sys.path."""
import sys
from pathlib import Path

_ROOT = Path(__file__).parent

# Ensure the repo root is on sys.path so that custom_components is importable.
sys.path.insert(0, str(_ROOT))

# pytest-homeassistant-custom-component initializes a 'custom_components' namespace
# package pointing at its testing_config directory.  Extend its __path__ so that
# the local custom_components directory is also searched.
import custom_components  # noqa: E402
_local_cc = str(_ROOT / "custom_components")
if _local_cc not in custom_components.__path__:
    custom_components.__path__.insert(0, _local_cc)

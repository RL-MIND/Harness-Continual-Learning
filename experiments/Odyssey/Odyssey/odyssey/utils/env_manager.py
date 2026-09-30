import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "conf" / "config.json"
EXAMPLE_CONFIG_PATH = PROJECT_ROOT / "conf" / "config.example.json"

ENV_OVERRIDES = {
    "ODYSSEY_OPENAI_BASE_URL": "openai_base_url",
    "ODYSSEY_OPENAI_API_KEY": "openai_api_key",
    "ODYSSEY_OPENAI_MODEL": "openai_model",
    "ODYSSEY_MC_SERVER_HOST": "MC_SERVER_HOST",
    "ODYSSEY_MC_SERVER_PORT": "MC_SERVER_PORT",
    "ODYSSEY_NODE_SERVER_PORT": "NODE_SERVER_PORT",
}

class ConfigManager:
    def __init__(self):
        configured_path = os.environ.get("ODYSSEY_CONFIG")
        if configured_path:
            path = Path(configured_path).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"ODYSSEY_CONFIG does not exist: {path}")
        elif DEFAULT_CONFIG_PATH.is_file():
            path = DEFAULT_CONFIG_PATH
        else:
            path = EXAMPLE_CONFIG_PATH

        with path.open("r", encoding="utf-8") as handle:
            self.config = json.load(handle)

        for environment_name, config_name in ENV_OVERRIDES.items():
            value = os.environ.get(environment_name)
            if value is not None:
                self.config[config_name] = value

        self.path = path

    def __getitem__(self, key):
        return self.get(key)

    def get(self, key: str, default: Any = "") -> Any:
        return self.config.get(key, default)

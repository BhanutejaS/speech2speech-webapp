import os
import yaml
from dotenv import load_dotenv

class SpeechToSpeechConfig:
    """Loads YAML config and resolves ${ENV_VARS} automatically."""

    def __init__(self, path="config.yaml"):
        load_dotenv()
        with open(path, "r") as f:
            raw_cfg = yaml.safe_load(f)
        self._cfg = self._resolve_env_vars(raw_cfg)

        # assign attributes
        for k, v in self._cfg.items():
            setattr(self, k, v)

    def _resolve_env_vars(self, d):
        resolved = {}
        for k, v in d.items():
            if isinstance(v, dict):
                resolved[k] = self._resolve_env_vars(v)
            elif isinstance(v, str) and v.startswith("${") and v.endswith("}"):
                env_name = v[2:-1]
                resolved[k] = os.getenv(env_name, "")
            else:
                resolved[k] = v
        return resolved

    def __getitem__(self, item):
        return self._cfg.get(item)

    def __repr__(self):
        return f"<SpeechToSpeechConfig keys={list(self._cfg.keys())}>"

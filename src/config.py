"""Configuration loading.

All tunables live in ``configs/*.yaml``. A ``Config`` wraps the parsed dict with
attribute access (nested dicts become nested ``Config``) plus a dotted ``get``.
"""
from __future__ import annotations

import os
from typing import Any, Optional

import yaml

_ENV_PREFIX = "SENTIMENT_"


class Config:
    """Dict wrapper with attribute access and dotted-path ``get``."""

    def __init__(self, data: dict[str, Any]):
        self._data = dict(data)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            value = self._data[name]
        except KeyError as exc:
            raise AttributeError(f"no config key: {name}") from exc
        return _wrap(value)

    def get(self, path: str, default: Any = None) -> Any:
        """Dotted lookup, e.g. ``cfg.get("student.lora_r", 16)``."""
        node: Any = self._data
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def to_dict(self) -> dict[str, Any]:
        return _unwrap(self._data)

    def __repr__(self) -> str:  # pragma: no cover
        return f"Config({self._data!r})"


def _wrap(value: Any) -> Any:
    if isinstance(value, dict):
        return Config(value)
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def _unwrap(value: Any) -> Any:
    if isinstance(value, Config):
        return _unwrap(value._data)
    if isinstance(value, dict):
        return {k: _unwrap(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_unwrap(v) for v in value]
    return value


def _apply_env_overrides(data: dict[str, Any], prefix: str = _ENV_PREFIX) -> dict[str, Any]:
    """Allow env vars to override config at runtime.

    ``__`` separates nesting levels and is lower-cased; underscores *inside* a
    key are preserved: ``student.lora_r`` <- ``SENTIMENT_STUDENT__LORA_R``,
    ``data.gold_size`` <- ``SENTIMENT_DATA__GOLD_SIZE``.
    """
    out = data
    for env_key, env_val in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        dotted = env_key[len(prefix):].lower().replace("__", ".")
        _set_path(out, dotted, env_val)
    return out


def _set_path(data: dict[str, Any], dotted: str, value: str) -> None:
    parts = dotted.split(".")
    node = data
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def load_config(path: Optional[str] = None) -> Config:
    """Load a YAML config, applying env-var overrides. Default: ``configs/default.yaml``.

    Also loads ``.env`` (repo root) so API keys reach the teacher without a manual
    ``export``. ``python-dotenv`` is optional — if absent, rely on the environment.
    """
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        from dotenv import load_dotenv

        load_dotenv(os.path.join(_root, ".env"))
    except ImportError:
        pass

    if path is None:
        path = os.path.join(_root, "configs", "default.yaml")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    data = _apply_env_overrides(data)
    return Config(data)

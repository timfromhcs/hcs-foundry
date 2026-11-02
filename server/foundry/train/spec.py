"""The declarative training spec — one input for every method and backend.

Rules that matter in practice:

* ``extra="forbid"`` — a misspelled key fails loudly instead of being ignored (the Axolotl
  lesson: silent typos are how people train the wrong thing for a week).
* ``spec_hash`` is stable under key order, so a manifest can be tied to the exact configuration.
* Hyper-parameters are explicit; nothing is hidden in a global.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SPEC_VERSION = 1


class Method(str, Enum):
    SFT = "sft"
    INT4 = "int4"
    DPO = "dpo"
    DISTILL = "distill"
    RFT = "rft"
    ONPOLICY_DISTILL = "onpolicy_distill"
    GRPO = "grpo"


class Backend(str, Enum):
    FP32_LORA = "fp32_lora"
    INT4_STREAM = "int4_stream"
    INT4_STORE = "int4_store"
    REMOTE = "remote"


class Profile(str, Enum):
    CI_TINY = "ci-tiny"
    LOCAL_SMOKE = "local-smoke"
    LOCAL_REAL = "local-real"
    REMOTE = "remote"


class ModelRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, max_length=1024)
    revision_hash: str | None = Field(default=None, max_length=128)


class DataRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(min_length=1, max_length=128)
    manifest_hash: str = Field(default="", max_length=128)
    split: Literal["train"] = "train"
    min_rows: int = Field(default=2, ge=1, le=1_000_000)


class Hyper(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lr: float = Field(gt=0, le=1.0)
    lora_r: int = Field(ge=1, le=256)
    lora_alpha: int = Field(ge=1, le=512)
    micro_batch: int = Field(default=1, ge=1, le=64)
    grad_accum: int = Field(default=1, ge=1, le=256)
    seq_len: int = Field(default=128, ge=16, le=8192)
    max_steps: int = Field(default=2, ge=0, le=100_000)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)


class Budget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_wall_s: int = Field(default=3600, ge=1, le=7 * 24 * 3600)
    max_ram_bytes: int | None = Field(default=None, ge=0)
    max_disk_bytes: int | None = Field(default=None, ge=0)


class EvalPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    suites: list[str] = Field(default_factory=list, max_length=32)
    guard_suites: list[str] = Field(default_factory=list, max_length=32)
    every_n_steps: int = Field(default=0, ge=0, le=10_000)
    final: bool = True


class StopRules(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patience: int = Field(default=3, ge=1, le=100)
    min_delta: float = Field(default=0.0, ge=0, le=1)
    nan_abort: bool = True


class TrainSpec(BaseModel):
    """Everything a trainer needs, and nothing it has to guess."""

    model_config = ConfigDict(extra="forbid")

    spec_version: Literal[1] = SPEC_VERSION
    method: Method
    backend: Backend
    model: ModelRef
    data: DataRef
    hyper: Hyper
    budget: Budget = Field(default_factory=Budget)
    eval: EvalPlan = Field(default_factory=EvalPlan)
    stop: StopRules = Field(default_factory=StopRules)
    profile: Profile = Profile.LOCAL_SMOKE
    method_args: dict[str, Any] = Field(default_factory=dict)

    @field_validator("method_args")
    @classmethod
    def _method_args_must_be_flat(cls, value: dict[str, Any]) -> dict[str, Any]:
        if len(value) > 32:
            raise ValueError("method_args supports at most 32 keys")
        return value

    @model_validator(mode="after")
    def _backend_must_fit_method(self) -> "TrainSpec":
        if self.method is Method.INT4 and self.backend not in (Backend.INT4_STREAM, Backend.INT4_STORE):
            raise ValueError("method 'int4' needs backend int4_stream or int4_store")
        if self.method is Method.SFT and self.backend is Backend.INT4_STORE:
            raise ValueError("method 'sft' cannot use the int4_store backend")
        if self.backend is Backend.REMOTE and self.profile is not Profile.REMOTE:
            raise ValueError("the remote backend requires profile 'remote'")
        return self

    @property
    def spec_hash(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_yaml(self) -> str:
        """Minimal YAML writer (no PyYAML dependency): nested mappings, no anchors."""
        def render(value: Any, indent: int = 0) -> list[str]:
            pad = "  " * indent
            lines: list[str] = []
            if isinstance(value, dict):
                for key, item in value.items():
                    if isinstance(item, (dict, list)) and item:
                        lines.append(f"{pad}{key}:")
                        lines.extend(render(item, indent + 1))
                    elif isinstance(item, list):
                        lines.append(f"{pad}{key}: []")
                    else:
                        lines.append(f"{pad}{key}: {_scalar(item)}")
            elif isinstance(value, list):
                for item in value:
                    lines.append(f"{pad}- {_scalar(item)}")
            else:
                lines.append(f"{pad}{_scalar(value)}")
            return lines

        return "\n".join(render(self.model_dump(mode="json"))) + "\n"

    @classmethod
    def from_dict(cls, payload: dict) -> "TrainSpec":
        return cls.model_validate(payload)


def _scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if text == "":
        return '""'
    if any(char in text for char in ":#[]{}\n"):
        return json.dumps(text)
    return text


PROFILE_DEFAULTS: dict[Profile, dict[str, Any]] = {
    Profile.CI_TINY: {
        "hyper": {"lr": 1e-3, "lora_r": 2, "lora_alpha": 4, "seq_len": 24, "max_steps": 2},
        "budget": {"max_wall_s": 120},
    },
    Profile.LOCAL_SMOKE: {
        "hyper": {"lr": 2e-4, "lora_r": 4, "lora_alpha": 16, "seq_len": 128, "max_steps": 2},
        "budget": {"max_wall_s": 1800},
    },
    Profile.LOCAL_REAL: {
        "hyper": {"lr": 2e-4, "lora_r": 8, "lora_alpha": 16, "seq_len": 256, "max_steps": 200},
        "budget": {"max_wall_s": 12 * 3600},
    },
}


def spec_for_profile(
    *,
    method: Method,
    backend: Backend,
    model_path: str,
    dataset_id: str,
    profile: Profile = Profile.LOCAL_SMOKE,
    **overrides: Any,
) -> TrainSpec:
    """Build a spec from a profile's defaults, with explicit overrides."""
    base = {
        "method": method,
        "backend": backend,
        "profile": profile,
        "model": {"path": model_path},
        "data": {"dataset_id": dataset_id},
        "hyper": dict(PROFILE_DEFAULTS.get(profile, {}).get("hyper", {})),
        "budget": dict(PROFILE_DEFAULTS.get(profile, {}).get("budget", {})),
    }
    for key, value in overrides.items():
        if key in ("hyper", "budget", "eval", "stop", "model", "data") and isinstance(value, dict):
            base.setdefault(key, {}).update(value)
        else:
            base[key] = value
    # required hyper fields without a profile default
    base["hyper"].setdefault("lr", 2e-4)
    base["hyper"].setdefault("lora_r", 4)
    base["hyper"].setdefault("lora_alpha", 16)
    return TrainSpec.model_validate(base)


def _parse_simple_yaml(text: str) -> dict:
    """A deliberately small YAML subset: nested mappings, scalar lists, quoted strings.

    Used only when PyYAML is not installed. It handles the files this project writes
    (``TrainSpec.to_yaml``) and refuses anything it does not understand instead of guessing.
    """
    import json as json_module

    root: dict = {}
    stack: list[tuple[int, dict]] = [(-1, root)]
    pending_list: dict | None = None
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if line.startswith("- "):
            if pending_list is None:
                raise ValueError("list item without a key")
            pending_list.setdefault("__items__", []).append(_yaml_scalar(line[2:]))
            continue
        if ":" not in line:
            raise ValueError(f"cannot parse line: {raw!r}")
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if value == "":
            child: dict = {}
            parent[key] = child
            pending_list = child
            stack.append((indent, child))
        elif value == "[]":
            parent[key] = []
            pending_list = None
        else:
            parent[key] = _yaml_scalar(value)
            pending_list = None
    return _unwrap_lists(root)


def _unwrap_lists(node: dict) -> dict:
    for key, value in list(node.items()):
        if isinstance(value, dict):
            items = value.pop("__items__", None)
            if items is not None:
                node[key] = items
            else:
                _unwrap_lists(value)
    return node


def _yaml_scalar(text: str):
    import json as json_module

    if text.startswith(('"', "'")):
        return json_module.loads(text) if text.startswith('"') else text[1:-1]
    if text in ("true", "false"):
        return text == "true"
    if text in ("null", "~", "None"):
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def load_spec(path: str | "Path") -> TrainSpec:
    """Load a spec from JSON or YAML."""
    from pathlib import Path as _Path

    file = _Path(path).expanduser()
    text = file.read_text("utf-8")
    if file.suffix.lower() == ".json":
        return TrainSpec.model_validate(__import__("json").loads(text))
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        return TrainSpec.model_validate(_parse_simple_yaml(text))
    return TrainSpec.model_validate(yaml.safe_load(text))

"""Registry of trainers keyed by (method, backend)."""

from __future__ import annotations

from typing import Iterable

from .base import Trainer
from .spec import Backend, Method, TrainSpec


class RegistryError(RuntimeError):
    pass


_REGISTRY: dict[tuple[Method, Backend], type[Trainer]] = {}


def register(trainer_cls: type[Trainer]) -> type[Trainer]:
    method = trainer_cls.method
    backend = trainer_cls.backend
    key = (Method(method), Backend(backend))
    if key in _REGISTRY:
        raise RegistryError(f"a trainer for {key[0].value}/{key[1].value} is already registered")
    _REGISTRY[key] = trainer_cls
    return trainer_cls


def get_trainer(spec: TrainSpec) -> Trainer:
    key = (spec.method, spec.backend)
    trainer_cls = _REGISTRY.get(key)
    if trainer_cls is None:
        available = ", ".join(f"{m.value}/{b.value}" for m, b in sorted(_REGISTRY))
        raise RegistryError(
            f"no trainer for {spec.method.value}/{spec.backend.value}; available: {available or 'none'}"
        )
    return trainer_cls()


def list_trainers() -> list[tuple[Method, Backend]]:
    return sorted(_REGISTRY)


def load_builtin_trainers() -> list[tuple[Method, Backend]]:
    """Import the adapter modules so their ``@register`` decorators run."""
    from .adapters import auto as _auto  # noqa: F401
    from .adapters import distill as _distill  # noqa: F401
    from .adapters import dpo as _dpo  # noqa: F401
    from .adapters import int4 as _int4  # noqa: F401
    from .adapters import sft as _sft  # noqa: F401

    return list_trainers()


def iter_trainers() -> Iterable[Trainer]:
    if not _REGISTRY:
        load_builtin_trainers()
    for trainer_cls in _REGISTRY.values():
        yield trainer_cls()

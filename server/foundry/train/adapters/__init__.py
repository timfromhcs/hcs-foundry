"""Adapters: thin trainers over the pre-v8 implementations."""

from . import auto, distill, dpo, int4, rft, sft  # noqa: F401

__all__ = ["auto", "distill", "dpo", "int4", "rft", "sft"]

"""Model adapters -- the only per-model surface in the codebase.

Importing this package registers all thirteen.  ``list_adapters()`` then gives
the roster, and ``get_adapter(key)`` an unloaded instance.
"""

from .base import (  # noqa: F401
    ModelAdapter,
    get_adapter,
    list_adapters,
    list_controls,
    list_api_models,
    register,
)
from . import (  # noqa: F401,E402
    aero,
    gemini,
    openai_audio,
    audio_flamingo,
    cascaded,
    gemma3n,
    moss,
    phi4,
    qwen,
    voxtral,
)

__all__ = ["ModelAdapter", "get_adapter", "list_adapters", "list_controls", "register"]

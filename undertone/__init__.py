"""UNDERTONE: multilingual needle-in-a-haystack for prominence-conditioned retrieval.

Replaces the retired ``longaudiobench`` task suite, whose ground truth was not
derived from the audio (random verdicts, constant speaker labels, timelines
generated before the audio existed).  Everything here keys off a real span in a
real recording.
"""

__version__ = "0.1.0"

from .env import Hardware, hf_token, resolve_hardware, versions
from .items import MCQItem, ItemPack, ROLES, CATEGORIES, LANGS
from .ladder import CONDITIONS, window_for

__all__ = [
    "Hardware",
    "hf_token",
    "resolve_hardware",
    "versions",
    "MCQItem",
    "ItemPack",
    "ROLES",
    "CATEGORIES",
    "LANGS",
    "CONDITIONS",
    "window_for",
]

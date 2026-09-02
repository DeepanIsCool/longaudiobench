"""
LongAudioBench: A Comprehensive Benchmark for Long-Context Audio Understanding

Tests 4 fundamental failure modes of Audio-LLMs:
1. Acoustic Needle-in-Haystack (ANiH) - Long-range acoustic retrieval
2. Speaker Diarization Drift - Long-term speaker identity tracking
3. Environmental Soundscape Timeline - Non-speech auditory scene analysis
4. Acoustic Narrative Coherence - Multi-hop cross-modal reasoning
"""

__version__ = "0.1.0"
__all__ = ["tasks", "metrics", "baselines", "configs"]
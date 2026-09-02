"""
Audio mixing utilities for LongAudioBench data generation.
"""

import numpy as np
import librosa
import soundfile as sf
from typing import List, Tuple, Optional
import random


def load_audio(path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Load audio file and resample to target sample rate."""
    audio, sr = librosa.load(path, sr=target_sr)
    return audio, sr


def save_audio(audio: np.ndarray, path: str, sr: int = 16000):
    """Save audio to file."""
    sf.write(path, audio, sr)


def mix_audio_at_timestamp(
    background: np.ndarray,
    foreground: np.ndarray,
    insertion_time: float,
    sr: int = 16000,
    snr_db: float = 10.0
) -> np.ndarray:
    """
    Mix foreground audio into background at specified timestamp with given SNR.
    
    Args:
        background: Background audio array
        foreground: Foreground (needle) audio array
        insertion_time: Time in seconds to insert foreground
        sr: Sample rate
        snr_db: Signal-to-noise ratio in dB (foreground vs background at insertion point)
    
    Returns:
        Mixed audio array
    """
    # Calculate sample positions
    insert_sample = int(insertion_time * sr)
    fg_len = len(foreground)
    
    # Ensure background is long enough
    required_len = insert_sample + fg_len
    if len(background) < required_len:
        # Pad background
        background = np.pad(background, (0, required_len - len(background)), mode='constant')
    
    # Extract background segment at insertion point
    bg_segment = background[insert_sample:insert_sample + fg_len]
    
    # Calculate scaling for target SNR
    # SNR = 20 * log10(rms_fg / rms_bg)
    rms_fg = np.sqrt(np.mean(foreground ** 2))
    rms_bg = np.sqrt(np.mean(bg_segment ** 2))
    
    if rms_bg > 0 and rms_fg > 0:
        target_ratio = 10 ** (snr_db / 20)
        scale_fg = target_ratio * rms_bg / rms_fg
        foreground_scaled = foreground * scale_fg
    else:
        foreground_scaled = foreground
    
    # Mix
    mixed = background.copy()
    mixed[insert_sample:insert_sample + fg_len] = bg_segment + foreground_scaled
    
    return mixed


def concatenate_audio_segments(segments: List[Tuple[np.ndarray, float, float]], 
                                total_duration: float,
                                sr: int = 16000,
                                crossfade_ms: float = 10.0) -> np.ndarray:
    """
    Concatenate audio segments at specified times.
    
    Args:
        segments: List of (audio_array, start_time, end_time)
        total_duration: Total output duration in seconds
        sr: Sample rate
        crossfade_ms: Crossfade duration in milliseconds
    
    Returns:
        Concatenated audio array
    """
    total_samples = int(total_duration * sr)
    output = np.zeros(total_samples)
    crossfade_samples = int(crossfade_ms / 1000 * sr)
    
    for audio, start_time, end_time in segments:
        start_sample = int(start_time * sr)
        end_sample = min(int(end_time * sr), total_samples)
        segment_len = end_sample - start_sample
        
        if segment_len <= 0:
            continue
        
        # Trim or pad audio to fit
        if len(audio) > segment_len:
            audio = audio[:segment_len]
        elif len(audio) < segment_len:
            audio = np.pad(audio, (0, segment_len - len(audio)), mode='constant')
        
        # Apply crossfade if overlapping
        if start_sample > 0 and crossfade_samples > 0:
            fade_in = np.linspace(0, 1, min(crossfade_samples, segment_len))
            audio[:len(fade_in)] *= fade_in
            
            # Crossfade with existing
            overlap_start = max(0, start_sample - crossfade_samples)
            overlap_len = start_sample - overlap_start
            if overlap_len > 0:
                fade_out = np.linspace(1, 0, overlap_len)
                output[overlap_start:start_sample] *= fade_out
        
        output[start_sample:end_sample] += audio
    
    # Normalize to prevent clipping
    max_val = np.max(np.abs(output))
    if max_val > 1.0:
        output = output / max_val * 0.95
    
    return output


def create_anih_audio(
    bg_path: str,
    needle_path: str,
    output_path: str,
    insertion_time: float,
    bg_duration: float = 3600.0,
    needle_duration: float = 0.5,
    snr_db: float = 10.0,
    sr: int = 16000
):
    """
    Create ANiH mixed audio file.
    
    Args:
        bg_path: Path to background audio (should be at least bg_duration long)
        needle_path: Path to needle audio (should be needle_duration long)
        output_path: Output path for mixed audio
        insertion_time: Time in seconds to insert needle
        bg_duration: Total background duration
        needle_duration: Needle duration
        snr_db: Signal-to-noise ratio
        sr: Sample rate
    """
    # Load background (trim or loop to bg_duration)
    bg_audio, _ = load_audio(bg_path, sr)
    bg_target_len = int(bg_duration * sr)
    
    if len(bg_audio) < bg_target_len:
        # Loop background
        repeats = (bg_target_len // len(bg_audio)) + 1
        bg_audio = np.tile(bg_audio, repeats)[:bg_target_len]
    else:
        bg_audio = bg_audio[:bg_target_len]
    
    # Load needle
    needle_audio, _ = load_audio(needle_path, sr)
    needle_target_len = int(needle_duration * sr)
    
    if len(needle_audio) > needle_target_len:
        needle_audio = needle_audio[:needle_target_len]
    elif len(needle_audio) < needle_target_len:
        needle_audio = np.pad(needle_audio, (0, needle_target_len - len(needle_audio)), mode='constant')
    
    # Mix
    mixed = mix_audio_at_timestamp(bg_audio, needle_audio, insertion_time, sr, snr_db)
    
    # Save
    save_audio(mixed, output_path, sr)
    print(f"[INFO] Created ANiH audio: {output_path} (needle at {insertion_time:.1f}s)")


def create_speaker_drift_audio(
    speaker_segments: List[Tuple[str, float, float]],  # (path, start, end)
    output_path: str,
    total_duration: float,
    sr: int = 16000
):
    """
    Create speaker drift audio by concatenating speaker turns.
    
    Args:
        speaker_segments: List of (audio_path, start_time, end_time)
        output_path: Output path
        total_duration: Total duration
        sr: Sample rate
    """
    segments = []
    for path, start, end in speaker_segments:
        audio, _ = load_audio(path, sr)
        segments.append((audio, start, end))
    
    mixed = concatenate_audio_segments(segments, total_duration, sr)
    save_audio(mixed, output_path, sr)
    print(f"[INFO] Created speaker drift audio: {output_path}")


def create_soundscape_audio(
    event_segments: List[Tuple[str, float, float]],  # (path, start, end)
    bg_path: Optional[str],
    output_path: str,
    total_duration: float,
    sr: int = 16000
):
    """
    Create soundscape audio by mixing events over background.
    
    Args:
        event_segments: List of (audio_path, start_time, end_time)
        bg_path: Optional background audio path
        output_path: Output path
        total_duration: Total duration
        sr: Sample rate
    """
    total_samples = int(total_duration * sr)
    output = np.zeros(total_samples)
    
    # Add background if provided
    if bg_path:
        bg_audio, _ = load_audio(bg_path, sr)
        if len(bg_audio) < total_samples:
            repeats = (total_samples // len(bg_audio)) + 1
            bg_audio = np.tile(bg_audio, repeats)[:total_samples]
        else:
            bg_audio = bg_audio[:total_samples]
        output = bg_audio.copy()
    
    # Add events
    for path, start, end in event_segments:
        audio, _ = load_audio(path, sr)
        start_sample = int(start * sr)
        end_sample = min(int(end * sr), total_samples)
        segment_len = end_sample - start_sample
        
        if segment_len <= 0:
            continue
        
        if len(audio) > segment_len:
            audio = audio[:segment_len]
        elif len(audio) < segment_len:
            audio = np.pad(audio, (0, segment_len - len(audio)), mode='constant')
        
        output[start_sample:end_sample] += audio
    
    # Normalize
    max_val = np.max(np.abs(output))
    if max_val > 1.0:
        output = output / max_val * 0.95
    
    save_audio(output, output_path, sr)
    print(f"[INFO] Created soundscape audio: {output_path}")


def create_narrative_coherence_audio(
    story_path: str,
    clue_path: str,
    output_path: str,
    clue_a_time: float,
    story_duration: float,
    clue_duration: float = 5.0,
    sr: int = 16000
):
    """
    Create narrative coherence audio by embedding clue into story.
    
    Args:
        story_path: Path to story audio
        clue_path: Path to acoustic clue
        output_path: Output path
        clue_a_time: Time to insert clue A
        story_duration: Total story duration
        clue_duration: Clue duration
        sr: Sample rate
    """
    # Load story
    story_audio, _ = load_audio(story_path, sr)
    target_len = int(story_duration * sr)
    
    if len(story_audio) < target_len:
        # Pad with silence
        story_audio = np.pad(story_audio, (0, target_len - len(story_audio)), mode='constant')
    else:
        story_audio = story_audio[:target_len]
    
    # Load clue
    clue_audio, _ = load_audio(clue_path, sr)
    clue_target_len = int(clue_duration * sr)
    
    if len(clue_audio) > clue_target_len:
        clue_audio = clue_audio[:clue_target_len]
    elif len(clue_audio) < clue_target_len:
        clue_audio = np.pad(clue_audio, (0, clue_target_len - len(clue_audio)), mode='constant')
    
    # Mix clue into story at clue_a_time (lower SNR so it's background)
    mixed = mix_audio_at_timestamp(story_audio, clue_audio, clue_a_time, sr, snr_db=5.0)
    
    save_audio(mixed, output_path, sr)
    print(f"[INFO] Created narrative coherence audio: {output_path}")
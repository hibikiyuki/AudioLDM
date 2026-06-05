"""Prompt pool for conditioning-mode IEC.

Provides 100+ diverse text prompts organized by category,
used for SLERP B-2 initialization and Micro-SLERP mutation.
"""

from __future__ import annotations

import random
from typing import List, Optional

EMOTION_MOOD: List[str] = [
    "happy upbeat music",
    "sad melancholic melody",
    "peaceful and calm ambient music",
    "tense and suspenseful music",
    "mysterious and eerie soundscape",
    "dark and ominous music",
    "nostalgic and wistful tune",
    "energetic and powerful music",
    "melancholic and reflective music",
    "aggressive and intense music",
    "playful and whimsical melody",
    "ethereal and dreamy soundscape",
    "dramatic and cinematic music",
    "hopeful and uplifting music",
    "romantic and tender music",
    "anxious and unsettling music",
    "triumphant and majestic music",
    "serene and tranquil music",
]

INSTRUMENTS: List[str] = [
    "solo piano melody",
    "violin string melody",
    "acoustic guitar strumming",
    "drum kit rhythm",
    "flute melody",
    "cello solo",
    "trumpet fanfare",
    "bass guitar groove",
    "synthesizer pad",
    "harp arpeggio",
    "pipe organ music",
    "saxophone jazz improvisation",
    "choir vocals",
    "marimba percussion",
    "banjo folk melody",
    "oboe classical melody",
    "french horn fanfare",
    "electric guitar riff",
]

GENRES: List[str] = [
    "jazz improvisation",
    "classical orchestral music",
    "electronic dance music",
    "ambient soundscape",
    "hip hop beat",
    "folk acoustic music",
    "rock music",
    "cinematic film score",
    "lofi hip hop",
    "full orchestral arrangement",
    "experimental avant-garde music",
    "blues music",
    "country music",
    "reggae rhythm",
    "bossa nova",
    "techno music",
]

ENVIRONMENTS: List[str] = [
    "sound of rain falling",
    "wind blowing through trees",
    "ocean waves crashing",
    "forest nature ambience",
    "city street noise",
    "crackling fire",
    "outer space ambient sounds",
    "crowd murmuring",
    "thunderstorm sounds",
    "birds chirping in morning",
    "waterfall sounds",
    "desert wind ambience",
]

TEXTURES: List[str] = [
    "reverberant echoing music",
    "dry and intimate recording",
    "layered and lush orchestration",
    "sparse and minimal music",
    "bright and shimmering music",
    "dark and heavy music",
    "warm and mellow music",
    "cold and distant music",
    "staccato rhythmic music",
    "legato flowing music",
    "glitchy electronic music",
    "lo-fi degraded audio",
    "rich and full-bodied music",
]

COMPOUND_PHRASES: List[str] = [
    "gentle piano melody with soft strings",
    "aggressive electronic beat with heavy bass",
    "peaceful nature ambience with soft flute",
    "dark orchestral tension with low brass",
    "upbeat jazz with saxophone and piano",
    "calm acoustic guitar with light percussion",
    "energetic rock guitar with drums",
    "mysterious ambient synthesizer pad",
    "emotional violin solo with orchestra",
    "bright and cheerful ukulele melody",
    "deep meditative drone with bells",
    "fast-paced electronic dance track",
    "slow and sorrowful piano ballad",
    "epic cinematic orchestra with choir",
    "funky bass groove with brass section",
    "soft and intimate acoustic guitar solo",
    "distorted electric guitar power chords",
    "warm jazz piano trio with bass and drums",
    "haunting minor key string quartet",
    "pulsating techno synthesizer with kick drum",
    "delicate harp arpeggio with strings",
    "driving cinematic percussion with brass",
    "gentle lullaby with music box melody",
    "tense thriller music with staccato strings",
    "jubilant fanfare with brass and timpani",
]

PROMPT_POOL: List[str] = (
    EMOTION_MOOD
    + INSTRUMENTS
    + GENRES
    + ENVIRONMENTS
    + TEXTURES
    + COMPOUND_PHRASES
)


def sample_prompts(n: int, exclude: Optional[List[str]] = None) -> List[str]:
    """Return n prompts sampled without replacement from the pool.

    Falls back to sampling with replacement if n exceeds available pool size.
    """
    pool = [p for p in PROMPT_POOL if p not in (exclude or [])]
    if n <= len(pool):
        return random.sample(pool, n)
    # If n > pool, exhaust pool then sample remainder with replacement
    result = list(pool)
    while len(result) < n:
        extra = random.sample(pool, min(len(pool), n - len(result)))
        result.extend(extra)
    return result[:n]

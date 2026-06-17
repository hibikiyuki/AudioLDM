"""Prompt pool for conditioning-mode IEC.

Provides diverse text prompts organized by category,
used for SLERP B-2 initialization and Micro-SLERP mutation.

除外基準 (eval_prompt_clap_consistency.py の結果に基づく):
  - CLAP自己整合スコア < 0.15 のプロンプト（AudioLDMが全く対応できない）
  - std > 0.07 かつ score < 0.20 のプロンプト（生成が不安定で誘導方向が信頼できない）

除外済み (15件, audioldm-m-full, n_samples=2, ddim_steps=200):
  crowd murmuring (0.052), city street noise (0.086), sound of rain falling (0.097),
  wind blowing through trees (0.098, std=0.137), sparse and minimal music (0.119),
  choir vocals (0.128), birds chirping in morning (0.127),
  peaceful nature ambience with soft flute (0.134),
  cello solo (0.152, std=0.076), acoustic guitar strumming (0.154, std=0.071),
  driving cinematic percussion with brass (0.169, std=0.082),
  thunderstorm sounds (0.175, std=0.094),
  jubilant fanfare with brass and timpani (0.179, std=0.081),
  big band jazz (0.241, std=0.070),
  upbeat marimba ensemble with bass (0.237)
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

import numpy as np

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
    "drum kit rhythm",
    "flute melody",
    "trumpet fanfare",
    "bass guitar groove",
    "synthesizer pad",
    "harp arpeggio",
    "pipe organ music",
    "saxophone jazz improvisation",
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
    "trip hop beat",
    "synthwave music",
    "drum and bass music",
    "new age ambient music",
    "baroque music",
]

ENVIRONMENTS: List[str] = [
    "ocean waves crashing",
    "forest nature ambience",
    "crackling fire",
    "outer space ambient sounds",
    "waterfall sounds",
    "desert wind ambience",
]

TEXTURES: List[str] = [
    "reverberant echoing music",
    "dry and intimate recording",
    "layered and lush orchestration",
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
    "gentle lullaby with music box melody",
    "tense thriller music with staccato strings",
    "dark trip hop beat with heavy bass",
    "melancholic piano melody with ambient strings",
    "atmospheric dark jazz with muted trumpet",
    "driving synthesizer arpeggio with drum machine",
    "eerie theremin melody with strings",
]

PROMPT_POOL: List[str] = (
    EMOTION_MOOD
    + INSTRUMENTS
    + GENRES
    + ENVIRONMENTS
    + TEXTURES
    + COMPOUND_PHRASES
)

# CLAP自己整合スコア (eval_prompt_clap_consistency.py, audioldm-m-full, n_samples=2, ddim_steps=200)
# AudioLDMが各プロンプトをどれだけ「理解して」生成できるかの代理指標。
# sample_prompts(weighted=True) でこのスコアに比例した加重サンプリングが使える。
CLAP_SCORES: Dict[str, float] = {
    # EMOTION_MOOD
    "happy upbeat music":                0.270356,
    "sad melancholic melody":            0.274085,
    "peaceful and calm ambient music":   0.215105,
    "tense and suspenseful music":       0.340176,
    "mysterious and eerie soundscape":   0.377456,
    "dark and ominous music":            0.291088,
    "nostalgic and wistful tune":        0.259007,
    "energetic and powerful music":      0.212136,
    "melancholic and reflective music":  0.308998,
    "aggressive and intense music":      0.307985,
    "playful and whimsical melody":      0.265392,
    "ethereal and dreamy soundscape":    0.348084,
    "dramatic and cinematic music":      0.266380,
    "hopeful and uplifting music":       0.248695,
    "romantic and tender music":         0.258224,
    "anxious and unsettling music":      0.337304,
    "triumphant and majestic music":     0.397673,
    "serene and tranquil music":         0.254482,
    # INSTRUMENTS
    "solo piano melody":                 0.388476,
    "violin string melody":              0.316207,
    "drum kit rhythm":                   0.155167,
    "flute melody":                      0.366682,
    "trumpet fanfare":                   0.269477,
    "bass guitar groove":                0.239529,
    "synthesizer pad":                   0.211709,
    "harp arpeggio":                     0.372814,
    "pipe organ music":                  0.295802,
    "saxophone jazz improvisation":      0.363927,
    "marimba percussion":                0.321221,
    "banjo folk melody":                 0.425010,
    "oboe classical melody":             0.360110,
    "french horn fanfare":               0.341180,
    "electric guitar riff":              0.173565,
    # GENRES
    "jazz improvisation":                0.311650,
    "classical orchestral music":        0.383334,
    "electronic dance music":            0.300455,
    "ambient soundscape":                0.420103,
    "hip hop beat":                      0.275584,
    "folk acoustic music":               0.234512,
    "rock music":                        0.172823,
    "cinematic film score":              0.231390,
    "lofi hip hop":                      0.389805,
    "full orchestral arrangement":       0.306560,
    "experimental avant-garde music":    0.225375,
    "blues music":                       0.246945,
    "country music":                     0.180125,
    "reggae rhythm":                     0.393068,
    "bossa nova":                        0.238551,
    "techno music":                      0.321679,
    "trip hop beat":                     0.269433,
    "synthwave music":                   0.254162,
    "drum and bass music":               0.342166,
    "new age ambient music":             0.400429,
    "baroque music":                     0.378774,
    # ENVIRONMENTS
    "ocean waves crashing":              0.222536,
    "forest nature ambience":            0.304518,
    "crackling fire":                    0.352843,
    "outer space ambient sounds":        0.408787,
    "waterfall sounds":                  0.277155,
    "desert wind ambience":              0.160392,
    # TEXTURES
    "reverberant echoing music":         0.325269,
    "dry and intimate recording":        0.264014,
    "layered and lush orchestration":    0.297605,
    "bright and shimmering music":       0.382657,
    "dark and heavy music":              0.304446,
    "warm and mellow music":             0.365552,
    "cold and distant music":            0.218721,
    "staccato rhythmic music":           0.424510,
    "legato flowing music":              0.340571,
    "glitchy electronic music":          0.291494,
    "lo-fi degraded audio":              0.338403,
    "rich and full-bodied music":        0.212249,
    # COMPOUND_PHRASES
    "gentle piano melody with soft strings":          0.336318,
    "aggressive electronic beat with heavy bass":     0.405237,
    "dark orchestral tension with low brass":         0.355652,
    "upbeat jazz with saxophone and piano":           0.253528,
    "calm acoustic guitar with light percussion":     0.277895,
    "energetic rock guitar with drums":               0.301580,
    "mysterious ambient synthesizer pad":             0.354118,
    "emotional violin solo with orchestra":           0.395296,
    "bright and cheerful ukulele melody":             0.424646,
    "deep meditative drone with bells":               0.260587,
    "fast-paced electronic dance track":              0.288190,
    "slow and sorrowful piano ballad":                0.319641,
    "epic cinematic orchestra with choir":            0.331712,
    "funky bass groove with brass section":           0.284213,
    "soft and intimate acoustic guitar solo":         0.194562,
    "distorted electric guitar power chords":         0.338638,
    "warm jazz piano trio with bass and drums":       0.225595,
    "haunting minor key string quartet":              0.303616,
    "pulsating techno synthesizer with kick drum":    0.384391,
    "delicate harp arpeggio with strings":            0.387216,
    "gentle lullaby with music box melody":           0.419894,
    "tense thriller music with staccato strings":     0.374108,
    "dark trip hop beat with heavy bass":             0.424408,
    "melancholic piano melody with ambient strings":  0.321864,
    "atmospheric dark jazz with muted trumpet":       0.340034,
    "driving synthesizer arpeggio with drum machine": 0.316943,
    "eerie theremin melody with strings":             0.351492,
}


def sample_prompts(
    n: int,
    exclude: Optional[List[str]] = None,
    rng: Optional[np.random.Generator] = None,
    weighted: bool = False,
) -> List[str]:
    """Return n prompts sampled without replacement from the pool.

    Falls back to sampling with replacement if n exceeds available pool size.

    Args:
        rng: optional seeded ``np.random.Generator``. When provided, sampling
            is fully determined by the generator's state (for reproducibility
            with a fixed seed). Otherwise the global ``random`` module is used.
        weighted: If True, sample proportional to CLAP_SCORES so that
            higher-scoring prompts (AudioLDMが得意なプロンプト) are drawn more
            frequently for SLERP B-2 initialization. Prompts absent from
            CLAP_SCORES receive the pool mean score as weight.
    """
    if n <= 0:
        return []
    pool = [p for p in PROMPT_POOL if p not in (exclude or [])]

    probs: Optional[np.ndarray] = None
    if weighted and pool:
        mean_score = float(np.mean(list(CLAP_SCORES.values())))
        raw = np.array(
            [CLAP_SCORES.get(p, mean_score) for p in pool], dtype=float
        )
        raw = np.clip(raw, 0.0, None)
        total = raw.sum()
        if total > 0:
            probs = raw / total

    if rng is not None:
        if n <= len(pool):
            return rng.choice(pool, size=n, replace=False, p=probs).tolist()
        result = rng.choice(pool, size=len(pool), replace=False, p=probs).tolist()
        while len(result) < n:
            extra = rng.choice(
                pool, size=min(len(pool), n - len(result)), replace=False, p=probs
            ).tolist()
            result.extend(extra)
        return result[:n]

    if n <= len(pool):
        if probs is not None:
            indices = np.random.choice(len(pool), size=n, replace=False, p=probs)
            return [pool[i] for i in indices]
        return random.sample(pool, n)

    # n > pool: exhaust then fill remainder
    if probs is not None:
        result = [pool[i] for i in np.random.permutation(len(pool))]
    else:
        result = list(pool)
    while len(result) < n:
        extra_size = min(len(pool), n - len(result))
        if probs is not None:
            indices = np.random.choice(len(pool), size=extra_size, replace=False, p=probs)
            result.extend([pool[i] for i in indices])
        else:
            result.extend(random.sample(pool, extra_size))
    return result[:n]

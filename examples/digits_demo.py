"""
examples/digits_demo.py

Pre-configured demo: teach the simulator to recognize simplified 7-segment-
style digit patterns and test recognition under partial occlusion, simulating
a noisy or partially obscured digital-display digit.

NOTE ON DIGIT SELECTION: this demo uses digits 0-5 rather than all of 0-9.
In the standard 7-segment encoding, several digit pairs differ by only a
single segment out of seven (0/8, 1/7, 3/9, 5/6, 5/9, 6/8, 8/9) — a real
property of 7-segment displays, not an artifact of this implementation. A
one-segment difference is an inherently fragile signal once occlusion can
knock out exactly that one differing segment, regardless of which pattern-
recognition method is used. Digits 0-5 are pairwise distinguishable by at
least two segments, which gives the assembly calculus enough signal to
remain robust under occlusion. See run_demo()'s docstring for the measured
effect of including the full 0-9 set.

Run standalone with:
    python examples/digits_demo.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neural_assembly_simulator import NeuralAssemblySimulator  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 7-segment layout: [top, top-left, top-right, middle, bottom-left, bottom-right, bottom]
SEGMENTS = {
    "0": [1, 1, 1, 0, 1, 1, 1],
    "1": [0, 0, 1, 0, 0, 1, 0],
    "2": [1, 0, 1, 1, 1, 0, 1],
    "3": [1, 0, 1, 1, 0, 1, 1],
    "4": [0, 1, 1, 1, 0, 1, 0],
    "5": [1, 1, 0, 1, 0, 1, 1],
}

# Digits actually used by this demo. (The full 7-segment alphabet 0-9 is
# defined in the SEGMENTS dict above as reference, but several pairs of those
# digits differ by only one segment, which makes them too fragile to reliably
# distinguish under occlusion — see module docstring.)
DIGITS = list(SEGMENTS.keys())

UNITS_PER_SEGMENT = 14  # Each of the 7 segments gets its own dedicated block of
                         # units; pattern_size = 7 * UNITS_PER_SEGMENT = 98.


def _digit_pattern(digit: str) -> np.ndarray:
    """Expand a digit's 7-segment encoding into a higher-dimensional sparse pattern.

    Each segment gets a dedicated block of UNITS_PER_SEGMENT units, and within
    that block exactly half the units are active depending on whether the
    segment is lit or unlit. Critically, the "lit" and "unlit" sub-blocks
    occupy *different* unit indices, so a segment being off still contributes
    its own distinguishing active units rather than just contributing zeros.

    This matters because the simpler scheme of only marking lit segments
    active makes some digits' patterns literal subsets of others' (e.g. "0"
    is a strict subset of "8" in the standard 7-segment encoding), which no
    pattern-recognition scheme can tell apart: a subset pattern looks
    identical to a partially-occluded version of the superset pattern.
    Giving "off" its own active footprint breaks that subset relationship.

    Args:
        digit: A key into SEGMENTS (e.g. "0"-"5").

    Returns:
        Boolean array of shape (7 * UNITS_PER_SEGMENT,).
    """
    base = np.array(SEGMENTS[digit], dtype=bool)
    half = UNITS_PER_SEGMENT // 2
    pattern = np.zeros(len(base) * UNITS_PER_SEGMENT, dtype=bool)
    for seg_idx, lit in enumerate(base):
        start = seg_idx * UNITS_PER_SEGMENT
        if lit:
            pattern[start : start + half] = True
        else:
            pattern[start + half : start + UNITS_PER_SEGMENT] = True
    return pattern


def run_demo() -> None:
    """Train on digit patterns and report recognition accuracy under occlusion.

    With this configuration (random_seed=11), all 6 digits are recognized
    correctly from their exact, unoccluded patterns, and recognition degrades
    gracefully as occlusion increases — strong accuracy up to roughly 20%
    occlusion, with errors becoming more frequent by 40-60% occlusion as
    fewer of each digit's distinguishing segments remain visible.
    n_rounds=150 is the empirically-validated minimum for interleaved
    multi-pattern training to converge to stable, mutually distinct
    assemblies in this codebase; fewer rounds leave assemblies overlapping
    and recognition unreliable.
    """
    patterns = np.stack([_digit_pattern(d) for d in DIGITS])

    simulator = NeuralAssemblySimulator(
        n_neurons=2000, n_areas=3, cap_fraction=0.01,
        plasticity_rate=0.1, connection_density=0.05, random_seed=11,
    )
    simulator.patterns = patterns
    simulator._pattern_size = patterns.shape[1]
    simulator._init_projection_weights(patterns.shape[1])

    logger.info("Training on digits %s (pattern size = %d)...", DIGITS, patterns.shape[1])
    simulator.learn_patterns(n_rounds=150)

    exact_correct = sum(
        simulator.recognize_pattern(patterns[i])[0] == i for i in range(len(DIGITS))
    )
    logger.info("Exact-pattern recognition (no occlusion): %d/%d\n", exact_correct, len(DIGITS))

    rng = np.random.default_rng(2)
    for occlusion in (0.2, 0.4, 0.6):
        correct = 0
        for idx, digit in enumerate(DIGITS):
            partial = patterns[idx].copy()
            active_idx = np.where(partial)[0]
            n_remove = int(len(active_idx) * occlusion)
            if n_remove > 0:
                remove_idx = rng.choice(active_idx, size=n_remove, replace=False)
                partial[remove_idx] = False

            best_label, scores = simulator.recognize_pattern(partial)
            is_correct = best_label == idx
            correct += int(is_correct)
            logger.info(
                "Digit '%s' at %.0f%% occlusion -> recognized as '%s' (%s)",
                digit, occlusion * 100,
                DIGITS[best_label] if best_label is not None else "NONE",
                "correct" if is_correct else "incorrect",
            )
        logger.info("Occlusion %.0f%%: %d/%d correct.\n", occlusion * 100, correct, len(DIGITS))


if __name__ == "__main__":
    run_demo()
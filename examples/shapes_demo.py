
## FILE 10: examples/shapes_demo.py
"""
examples/shapes_demo.py

Pre-configured demo: teach the simulator to recognize simple geometric
"shape" patterns (encoded as flattened binary grids) and test pattern
completion under occlusion.

Run standalone with:
    python examples/shapes_demo.py
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

GRID_SIDE = 10  # 10x10 = 100 units, matches default pattern_size


def _make_shape(name: str) -> np.ndarray:
    """Hand-craft simple shape patterns on a GRID_SIDE x GRID_SIDE grid."""
    grid = np.zeros((GRID_SIDE, GRID_SIDE), dtype=bool)

    if name == "cross":
        grid[GRID_SIDE // 2, :] = True
        grid[:, GRID_SIDE // 2] = True
    elif name == "square":
        grid[2:8, 2] = True
        grid[2:8, 7] = True
        grid[2, 2:8] = True
        grid[7, 2:8] = True
    elif name == "diagonal":
        for i in range(GRID_SIDE):
            grid[i, i] = True
    elif name == "circle_approx":
        center = GRID_SIDE / 2
        radius = GRID_SIDE / 2 - 1
        for i in range(GRID_SIDE):
            for j in range(GRID_SIDE):
                dist = np.sqrt((i - center) ** 2 + (j - center) ** 2)
                if abs(dist - radius) < 1.0:
                    grid[i, j] = True
    elif name == "checkerboard_corner":
        grid[0:5, 0:5] = np.indices((5, 5)).sum(axis=0) % 2 == 0
    else:
        raise ValueError(f"Unknown shape: {name}")

    return grid.flatten()


def run_demo() -> None:
    """Train on hand-crafted shapes and report recognition accuracy under occlusion."""
    shape_names = ["cross", "square", "diagonal", "circle_approx", "checkerboard_corner"]
    patterns = np.stack([_make_shape(name) for name in shape_names])

    simulator = NeuralAssemblySimulator(n_neurons=2000, n_areas=3, random_seed=7)
    simulator.patterns = patterns
    simulator._pattern_size = patterns.shape[1]
    simulator._init_projection_weights(patterns.shape[1])

    logger.info("Training on %d shapes...", len(shape_names))
    simulator.learn_patterns(n_rounds=150)

    rng = np.random.default_rng(1)
    for occlusion in (0.2, 0.4, 0.6):
        correct = 0
        for idx, name in enumerate(shape_names):
            partial = patterns[idx].copy()
            active_idx = np.where(partial)[0]
            n_remove = int(len(active_idx) * occlusion)
            remove_idx = rng.choice(active_idx, size=n_remove, replace=False)
            partial[remove_idx] = False

            best_label, scores = simulator.recognize_pattern(partial)
            is_correct = best_label == idx
            correct += int(is_correct)
            logger.info(
                "Shape '%s' at %.0f%% occlusion -> recognized as '%s' (%s)",
                name, occlusion * 100,
                shape_names[best_label] if best_label is not None else "NONE",
                "correct" if is_correct else "incorrect",
            )
        logger.info("Occlusion %.0f%%: %d/%d correct.\n", occlusion * 100, correct, len(shape_names))


if __name__ == "__main__":
    run_demo()
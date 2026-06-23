"""
tests/test_simulator.py

Unit tests for NeuralAssemblySimulator. Run with:
    pytest tests/test_simulator.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neural_assembly_simulator import NeuralAssemblySimulator  # noqa: E402


@pytest.fixture
def small_simulator() -> NeuralAssemblySimulator:
    """A small, fast simulator instance for testing."""
    return NeuralAssemblySimulator(
        n_neurons=300, n_areas=2, cap_fraction=0.05, plasticity_rate=0.2,
        connection_density=0.1, random_seed=0,
    )


class TestInitialization:
    def test_valid_init(self, small_simulator: NeuralAssemblySimulator) -> None:
        assert small_simulator.n_neurons == 300
        assert small_simulator.n_areas == 2
        assert small_simulator.k == max(1, int(300 * 0.05))

    def test_invalid_n_neurons_raises(self) -> None:
        with pytest.raises(ValueError):
            NeuralAssemblySimulator(n_neurons=0)

    def test_invalid_cap_fraction_raises(self) -> None:
        with pytest.raises(ValueError):
            NeuralAssemblySimulator(n_neurons=100, cap_fraction=1.5)


class TestPatternGeneration:
    def test_generate_patterns_shape(self, small_simulator: NeuralAssemblySimulator) -> None:
        patterns = small_simulator.generate_patterns(n_patterns=4, pattern_size=50)
        assert patterns.shape == (4, 50)
        assert patterns.dtype == bool

    def test_generate_patterns_invalid_args(self, small_simulator: NeuralAssemblySimulator) -> None:
        with pytest.raises(ValueError):
            small_simulator.generate_patterns(n_patterns=0, pattern_size=50)

    def test_patterns_are_sparse(self, small_simulator: NeuralAssemblySimulator) -> None:
        patterns = small_simulator.generate_patterns(n_patterns=3, pattern_size=100)
        active_fraction = patterns.sum(axis=1) / patterns.shape[1]
        assert np.all(active_fraction < 0.5)  # sparser than half-active


class TestLearning:
    def test_learn_patterns_requires_generation_first(
        self, small_simulator: NeuralAssemblySimulator
    ) -> None:
        with pytest.raises(RuntimeError):
            small_simulator.learn_patterns(n_rounds=3)

    def test_learn_patterns_produces_assemblies(
        self, small_simulator: NeuralAssemblySimulator
    ) -> None:
        small_simulator.generate_patterns(n_patterns=3, pattern_size=60)
        assemblies = small_simulator.learn_patterns(n_rounds=5)
        assert len(assemblies) == 3
        for assembly in assemblies.values():
            assert assembly.dtype == bool
            assert assembly.sum() == small_simulator.k

    def test_learned_assemblies_are_distinguishable(
        self, small_simulator: NeuralAssemblySimulator
    ) -> None:
        small_simulator.generate_patterns(n_patterns=3, pattern_size=80)
        assemblies = small_simulator.learn_patterns(n_rounds=15)
        overlap = small_simulator.assembly_overlap_matrix()
        # diagonal should be 1.0 (full self-overlap)
        assert np.allclose(np.diag(overlap), 1.0)


class TestRecognition:
    def test_recognize_pattern_requires_training(
        self, small_simulator: NeuralAssemblySimulator
    ) -> None:
        small_simulator.generate_patterns(n_patterns=2, pattern_size=50)
        with pytest.raises(RuntimeError):
            small_simulator.recognize_pattern(small_simulator.patterns[0])

    def test_recognize_exact_pattern_matches_self(
        self, small_simulator: NeuralAssemblySimulator
    ) -> None:
        small_simulator.generate_patterns(n_patterns=3, pattern_size=80)
        small_simulator.learn_patterns(n_rounds=20)
        for idx in range(3):
            best_label, scores = small_simulator.recognize_pattern(small_simulator.patterns[idx])
            assert best_label == idx
            assert scores[idx] == max(scores.values())

    def test_recognize_partial_pattern(self, small_simulator: NeuralAssemblySimulator) -> None:
        small_simulator.generate_patterns(n_patterns=3, pattern_size=80)
        small_simulator.learn_patterns(n_rounds=20)
        pattern = small_simulator.patterns[1].copy()
        active_idx = np.where(pattern)[0]
        pattern[active_idx[: len(active_idx) // 3]] = False  # remove ~1/3 of active units

        best_label, scores = small_simulator.recognize_pattern(pattern)
        assert best_label in (0, 1, 2)
        assert all(0.0 <= v <= 1.0 for v in scores.values())


class TestPatternCompletion:
    def test_pattern_completion_requires_training(
        self, small_simulator: NeuralAssemblySimulator
    ) -> None:
        small_simulator.generate_patterns(n_patterns=2, pattern_size=50)
        with pytest.raises(RuntimeError):
            small_simulator.pattern_completion(small_simulator.patterns[0])

    def test_pattern_completion_returns_full_pattern_shape(
        self, small_simulator: NeuralAssemblySimulator
    ) -> None:
        small_simulator.generate_patterns(n_patterns=3, pattern_size=80)
        small_simulator.learn_patterns(n_rounds=20)
        partial = small_simulator.patterns[0].copy()
        active_idx = np.where(partial)[0]
        partial[active_idx[: len(active_idx) // 2]] = False

        completed = small_simulator.pattern_completion(partial)
        assert completed.shape == partial.shape
        assert completed.dtype == bool


class TestUtilities:
    def test_activity_snapshot_shape(self, small_simulator: NeuralAssemblySimulator) -> None:
        small_simulator.generate_patterns(n_patterns=2, pattern_size=50)
        small_simulator.learn_patterns(n_rounds=3)
        snapshot = small_simulator.get_area_activity_snapshot()
        assert snapshot.shape == (small_simulator.n_areas, small_simulator.n_neurons)

    def test_overlap_matrix_symmetric(self, small_simulator: NeuralAssemblySimulator) -> None:
        small_simulator.generate_patterns(n_patterns=4, pattern_size=80)
        small_simulator.learn_patterns(n_rounds=10)
        overlap = small_simulator.assembly_overlap_matrix()
        assert np.allclose(overlap, overlap.T)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
"""
neural_assembly_simulator.py - Core Assembly Calculus Simulation
"""

from __future__ import annotations

import logging
from typing import Optional
import numpy as np
import streamlit as st
from pathlib import Path
import gzip
import struct
import io
import requests

logger = logging.getLogger(__name__)

# MNIST always uses 784 pixels (28x28). We keep this as a module constant
# so every code path agrees on the canonical size for real image datasets.
MNIST_SIZE = 784
MNIST_SIDE = 28


class BrainArea:
    def __init__(self, n_neurons: int, k: int) -> None:
        self.n_neurons = n_neurons
        self.k = k
        self.activation = np.zeros(n_neurons, dtype=bool)


class NeuralAssemblySimulator:
    def __init__(
        self,
        n_neurons: int = 10000,
        n_areas: int = 3,
        cap_fraction: float = 0.01,
        plasticity_rate: float = 0.10,
        connection_density: float = 0.05,
        random_seed: Optional[int] = 42,
    ) -> None:
        if n_neurons <= 0 or n_areas <= 0:
            raise ValueError("n_neurons and n_areas must both be positive integers.")
        if not (0 < cap_fraction < 1):
            raise ValueError("cap_fraction must be between 0 and 1.")

        self.n_neurons = n_neurons
        self.n_areas = n_areas
        self.cap_fraction = cap_fraction
        self.k = max(1, int(round(n_neurons * cap_fraction)))
        self.plasticity_rate = plasticity_rate
        self.connection_density = connection_density
        self.rng = np.random.default_rng(random_seed)

        self.areas = [BrainArea(n_neurons=n_neurons, k=self.k) for _ in range(n_areas)]

        self._input_weights: Optional[np.ndarray] = None
        self._pattern_size: Optional[int] = None
        self._inter_area_weights: list[np.ndarray] = []
        self._inter_area_nz_rows: list[np.ndarray] = []
        self._inter_area_col_sums: list[np.ndarray] = []
        self._input_nz_rows: Optional[list[np.ndarray]] = None
        self._input_col_sums: Optional[np.ndarray] = None

        self.assemblies: dict[int, np.ndarray] = {}
        self.patterns: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.dataset_name: str = "synthetic"

        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)

        logger.info("Initialized brain: %d areas x %d neurons, k=%d", n_areas, n_neurons, self.k)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _images_to_patterns(
        self, images: list[np.ndarray], labels_list: list[int]
    ) -> np.ndarray:
        """Convert a list of 28×28 uint8 images to (N, 784) bool patterns."""
        patterns = []
        for img in images:
            # Normalize to [0, 1] if needed
            arr = np.asarray(img, dtype=float)
            if arr.max() > 1.0:
                arr = arr / 255.0
            flat = arr.flatten()
            # Always 784 elements
            if len(flat) < MNIST_SIZE:
                flat = np.pad(flat, (0, MNIST_SIZE - len(flat)))
            elif len(flat) > MNIST_SIZE:
                flat = flat[:MNIST_SIZE]
            patterns.append(flat > 0.5)
        self.labels = np.array(labels_list, dtype=int)
        return np.array(patterns, dtype=bool)

    # ------------------------------------------------------------------
    # MNIST loading
    # ------------------------------------------------------------------

    def load_mnist(self, n_samples: int = 5, sample_size: int = 100) -> np.ndarray:
        """Load MNIST dataset.

        NOTE: ``sample_size`` is intentionally IGNORED for MNIST — we always
        store and return 784-pixel (28×28) patterns so the images are visible.
        The sidebar slider only affects *synthetic* and *custom* datasets.
        """
        self.dataset_name = "MNIST"
        real_size = MNIST_SIZE  # always 784

        # ---- 1. Try cache (must have been saved at 784 px) ----
        cached = self._load_from_cache("mnist_real", n_samples)
        if cached is not None and cached["patterns"].shape[1] == real_size:
            self.labels = cached.get("labels")
            patterns = cached["patterns"]
            self._init_patterns(patterns, real_size)
            st.success(f"✅ Loaded MNIST from cache! ({len(patterns)} patterns, 28×28)")
            return patterns

        # ---- 2. scikit-learn / OpenML ----
        try:
            from sklearn.datasets import fetch_openml
            st.info("📥 Downloading MNIST from OpenML (this may take a moment)…")
            X, y = fetch_openml(
                "mnist_784", version=1, return_X_y=True, as_frame=False, parser="auto"
            )
            selected_images, selected_labels = [], []
            for i in range(min(n_samples, 10)):
                idx = np.where(y == str(i))[0]
                if len(idx) == 0:
                    continue
                img = X[idx[0]].reshape(MNIST_SIDE, MNIST_SIDE)
                selected_images.append(img)
                selected_labels.append(i)
            if selected_images:
                patterns = self._images_to_patterns(selected_images, selected_labels)
                self._init_patterns(patterns, real_size)
                self._save_to_cache("mnist_real", patterns, self.labels)
                st.success(f"✅ MNIST loaded from OpenML! ({len(patterns)} digits, 28×28)")
                return patterns
        except Exception as e:
            st.warning(f"⚠️ OpenML failed: {str(e)[:60]}")

        # ---- 3. Direct download from Yann LeCun's mirror ----
        try:
            st.info("📥 Trying direct MNIST download…")
            url_img = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
            url_lbl = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
            r_img = requests.get(url_img, timeout=30)
            r_lbl = requests.get(url_lbl, timeout=30)
            if r_img.status_code == 200 and r_lbl.status_code == 200:
                with gzip.open(io.BytesIO(r_img.content), "rb") as f:
                    _, num, rows, cols = struct.unpack(">IIII", f.read(16))
                    all_images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
                with gzip.open(io.BytesIO(r_lbl.content), "rb") as f:
                    _, _ = struct.unpack(">II", f.read(8))
                    all_labels = np.frombuffer(f.read(), dtype=np.uint8)
                selected_images, selected_labels = [], []
                for i in range(min(n_samples, 10)):
                    idx = np.where(all_labels == i)[0]
                    if len(idx) == 0:
                        continue
                    selected_images.append(all_images[idx[0]])
                    selected_labels.append(i)
                if selected_images:
                    patterns = self._images_to_patterns(selected_images, selected_labels)
                    self._init_patterns(patterns, real_size)
                    self._save_to_cache("mnist_real", patterns, self.labels)
                    st.success(f"✅ MNIST downloaded directly! ({len(patterns)} digits, 28×28)")
                    return patterns
        except Exception as e:
            st.warning(f"⚠️ Direct download failed: {str(e)[:60]}")

        # ---- 4. TensorFlow/Keras ----
        try:
            from tensorflow.keras.datasets import mnist as tf_mnist  # type: ignore
            st.info("📥 Loading MNIST via TensorFlow…")
            (x_train, y_train), _ = tf_mnist.load_data()
            selected_images, selected_labels = [], []
            for i in range(min(n_samples, 10)):
                idx = np.where(y_train == i)[0]
                if len(idx) == 0:
                    continue
                selected_images.append(x_train[idx[0]])
                selected_labels.append(i)
            if selected_images:
                patterns = self._images_to_patterns(selected_images, selected_labels)
                self._init_patterns(patterns, real_size)
                self._save_to_cache("mnist_real", patterns, self.labels)
                st.success(f"✅ MNIST loaded via TensorFlow! ({len(patterns)} digits, 28×28)")
                return patterns
        except Exception as e:
            st.warning(f"⚠️ TensorFlow failed: {str(e)[:60]}")

        # ---- 5. Fallback: hand-crafted digit-like patterns at 784 px ----
        st.warning(
            "⚠️ All MNIST download methods failed. "
            "Generating digit-like patterns at 28×28 as a visual substitute."
        )
        return self._create_realistic_mnist_patterns(n_samples, real_size)

    # ------------------------------------------------------------------
    # Fashion-MNIST loading
    # ------------------------------------------------------------------

    def load_fashion_mnist(self, n_samples: int = 5, sample_size: int = 100) -> np.ndarray:
        """Load Fashion-MNIST dataset (always at 784 px)."""
        self.dataset_name = "Fashion-MNIST"
        real_size = MNIST_SIZE

        # ---- 1. Try cache ----
        cached = self._load_from_cache("fashion_real", n_samples)
        if cached is not None and cached["patterns"].shape[1] == real_size:
            self.labels = cached.get("labels")
            patterns = cached["patterns"]
            self._init_patterns(patterns, real_size)
            st.success(f"✅ Loaded Fashion-MNIST from cache! ({len(patterns)} patterns, 28×28)")
            return patterns

        # ---- 2. OpenML ----
        try:
            from sklearn.datasets import fetch_openml
            st.info("📥 Downloading Fashion-MNIST from OpenML…")
            X, y = fetch_openml(
                "Fashion-MNIST", version=1, return_X_y=True, as_frame=False, parser="auto"
            )
            selected_images, selected_labels = [], []
            for i in range(min(n_samples, 10)):
                idx = np.where(y == str(i))[0]
                if len(idx) == 0:
                    continue
                selected_images.append(X[idx[0]].reshape(MNIST_SIDE, MNIST_SIDE))
                selected_labels.append(i)
            if selected_images:
                patterns = self._images_to_patterns(selected_images, selected_labels)
                self._init_patterns(patterns, real_size)
                self._save_to_cache("fashion_real", patterns, self.labels)
                st.success(f"✅ Fashion-MNIST loaded from OpenML! ({len(patterns)} items, 28×28)")
                return patterns
        except Exception as e:
            st.warning(f"⚠️ OpenML failed: {str(e)[:60]}")

        # ---- 3. TensorFlow/Keras ----
        try:
            from tensorflow.keras.datasets import fashion_mnist as tf_fashion  # type: ignore
            st.info("📥 Loading Fashion-MNIST via TensorFlow…")
            (x_train, y_train), _ = tf_fashion.load_data()
            selected_images, selected_labels = [], []
            for i in range(min(n_samples, 10)):
                idx = np.where(y_train == i)[0]
                if len(idx) == 0:
                    continue
                selected_images.append(x_train[idx[0]])
                selected_labels.append(i)
            if selected_images:
                patterns = self._images_to_patterns(selected_images, selected_labels)
                self._init_patterns(patterns, real_size)
                self._save_to_cache("fashion_real", patterns, self.labels)
                st.success(f"✅ Fashion-MNIST loaded via TensorFlow! ({len(patterns)} items, 28×28)")
                return patterns
        except Exception as e:
            st.warning(f"⚠️ TensorFlow failed: {str(e)[:60]}")

        # ---- 4. Fallback at 784 px ----
        st.warning(
            "⚠️ All Fashion-MNIST download methods failed. "
            "Generating clothing-like patterns at 28×28 as a visual substitute."
        )
        return self._create_realistic_fashion_patterns(n_samples, real_size)

    # ------------------------------------------------------------------
    # Realistic fallback pattern generators (always 784 px)
    # ------------------------------------------------------------------

    def _create_realistic_mnist_patterns(self, n_patterns: int, sample_size: int) -> np.ndarray:
        """Create MNIST-like digit patterns at the given sample_size (should be 784)."""
        self.dataset_name = "mnist_like"
        size = MNIST_SIDE  # always draw at 28×28
        actual_size = size * size  # 784

        patterns, labels = [], []
        for digit in range(min(n_patterns, 10)):
            grid = np.zeros((size, size), dtype=float)
            cx, cy = size // 2, size // 2
            r = size // 3

            if digit == 0:
                for x in range(size):
                    for y in range(size):
                        d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                        if r - 2 < d < r + 2:
                            grid[x, y] = 1.0
            elif digit == 1:
                grid[3:size - 3, cx - 1: cx + 2] = 1.0
                grid[3:6, cx - 3: cx + 1] = 1.0
            elif digit == 2:
                # top bar
                grid[4:7, 7:21] = 1.0
                # right side top
                grid[4:14, 18:21] = 1.0
                # middle bar
                grid[12:15, 7:21] = 1.0
                # left side bottom
                grid[14:24, 7:10] = 1.0
                # bottom bar
                grid[22:25, 7:21] = 1.0
            elif digit == 3:
                for row in [4, 13, 23]:
                    grid[row: row + 3, 7:21] = 1.0
                grid[4:24, 18:21] = 1.0
            elif digit == 4:
                grid[4:14, 7:10] = 1.0
                grid[12:15, 7:21] = 1.0
                grid[4:24, 18:21] = 1.0
            elif digit == 5:
                grid[4:7, 7:21] = 1.0
                grid[4:14, 7:10] = 1.0
                grid[12:15, 7:21] = 1.0
                grid[14:24, 18:21] = 1.0
                grid[22:25, 7:21] = 1.0
            elif digit == 6:
                for x in range(size):
                    for y in range(size):
                        d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                        if r - 2 < d < r + 2:
                            grid[x, y] = 1.0
                grid[cx:, 7:10] = 1.0
            elif digit == 7:
                grid[4:7, 7:21] = 1.0
                grid[4:24, 18:21] = 1.0
            elif digit == 8:
                for x in range(size):
                    for y in range(size):
                        d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                        if r - 2 < d < r + 2:
                            grid[x, y] = 1.0
                grid[cx - 1: cx + 2, :] = 1.0
            elif digit == 9:
                for x in range(size):
                    for y in range(size):
                        d = np.sqrt((x - cx + 3) ** 2 + (y - cy) ** 2)
                        if r - 2 < d < r + 2:
                            grid[x, y] = 1.0
                grid[cx - 3:, 18:21] = 1.0

            flat = grid.flatten()
            # Ensure exactly actual_size elements
            if len(flat) < actual_size:
                flat = np.pad(flat, (0, actual_size - len(flat)))
            else:
                flat = flat[:actual_size]
            patterns.append(flat > 0.5)
            labels.append(digit)

        self.labels = np.array(labels, dtype=int)
        patterns_arr = np.array(patterns, dtype=bool)
        self._init_patterns(patterns_arr, actual_size)
        self._save_to_cache("mnist_real", patterns_arr, self.labels)
        return patterns_arr

    def _create_realistic_fashion_patterns(self, n_patterns: int, sample_size: int) -> np.ndarray:
        """Create Fashion-MNIST-like patterns at 28×28 (784 px)."""
        self.dataset_name = "fashion_like"
        size = MNIST_SIDE
        actual_size = size * size

        CLASS_NAMES = [
            "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
        ]

        patterns = []
        for i in range(min(n_patterns, 10)):
            grid = np.zeros((size, size), dtype=float)
            if i == 0:  # T-shirt
                grid[6:20, 6:22] = 1.0
                grid[6:13, 2:8] = 1.0
                grid[6:13, 20:26] = 1.0
                grid[4:7, 4:24] = 1.0
            elif i == 1:  # Trouser
                grid[4:15, 6:22] = 1.0
                grid[14:26, 6:13] = 1.0
                grid[14:26, 15:22] = 1.0
            elif i == 2:  # Pullover
                grid[6:22, 6:22] = 1.0
                grid[6:14, 2:8] = 1.0
                grid[6:14, 20:26] = 1.0
                grid[4:8, :] = 1.0
            elif i == 3:  # Dress
                grid[4:10, 8:20] = 1.0
                grid[10:26, 5:23] = 1.0
            elif i == 4:  # Coat
                grid[6:22, 6:22] = 1.0
                grid[6:16, 2:8] = 1.0
                grid[6:16, 20:26] = 1.0
                grid[4:8, :] = 1.0
                grid[12:22, 12:16] = 0.0  # lapel gap
            elif i == 5:  # Sandal
                grid[18:22, 4:24] = 1.0
                grid[10:18, 6:12] = 1.0
                grid[10:18, 16:22] = 1.0
                grid[8:12, 4:24] = 1.0
            elif i == 6:  # Shirt
                grid[6:20, 6:22] = 1.0
                grid[6:14, 2:8] = 1.0
                grid[6:14, 20:26] = 1.0
                grid[4:7, 4:24] = 1.0
                grid[8:16, 12:16] = 0.0  # button line
            elif i == 7:  # Sneaker
                grid[14:22, 3:25] = 1.0
                grid[8:16, 6:20] = 1.0
                grid[12:15, 3:8] = 1.0
            elif i == 8:  # Bag
                grid[6:24, 5:23] = 1.0
                grid[6:10, 5:9] = 0.0
                grid[6:10, 19:23] = 0.0
                grid[4:8, 9:19] = 1.0
            elif i == 9:  # Ankle boot
                grid[10:24, 5:23] = 1.0
                grid[4:12, 14:22] = 1.0
                grid[22:25, 3:9] = 1.0

            flat = grid.flatten()
            if len(flat) < actual_size:
                flat = np.pad(flat, (0, actual_size - len(flat)))
            else:
                flat = flat[:actual_size]
            patterns.append(flat > 0.5)

        patterns_arr = np.array(patterns, dtype=bool)
        self.labels = np.arange(len(patterns_arr), dtype=int)
        self._init_patterns(patterns_arr, actual_size)
        return patterns_arr

    # ------------------------------------------------------------------
    # Synthetic / custom
    # ------------------------------------------------------------------

    def generate_patterns(self, n_patterns: int = 5, pattern_size: int = 100) -> np.ndarray:
        """Generate synthetic sparse binary patterns."""
        self.dataset_name = "synthetic"
        active_fraction = 0.3
        n_active = max(1, int(pattern_size * active_fraction))
        patterns = np.zeros((n_patterns, pattern_size), dtype=bool)
        for i in range(n_patterns):
            active_idx = self.rng.choice(pattern_size, size=n_active, replace=False)
            patterns[i, active_idx] = True
        self.patterns = patterns
        self._pattern_size = pattern_size
        self._init_projection_weights(pattern_size)
        return patterns

    def load_custom_images(self, image_paths: list[str], sample_size: int = 100) -> np.ndarray:
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("PIL (Pillow) required for custom images.")
        self.dataset_name = "Custom Images"
        side = int(np.round(np.sqrt(sample_size)))
        actual_size = side * side
        patterns = []
        for path in image_paths:
            img = Image.open(path).convert("L")
            img = img.resize((side, side), Image.LANCZOS)
            arr = np.array(img, dtype=float) / 255.0
            flat = (arr.flatten() > 0.5)
            if len(flat) < actual_size:
                flat = np.pad(flat, (0, actual_size - len(flat)))
            elif len(flat) > actual_size:
                flat = flat[:actual_size]
            patterns.append(flat)
        patterns_arr = np.array(patterns, dtype=bool)
        self._init_patterns(patterns_arr, actual_size)
        return patterns_arr

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _save_to_cache(self, name: str, patterns: np.ndarray,
                       labels: Optional[np.ndarray] = None) -> None:
        try:
            cache_file = self.data_dir / f"{name}.npz"
            data: dict = {"patterns": patterns}
            if labels is not None:
                data["labels"] = labels
            np.savez(cache_file, **data)
        except Exception:
            pass

    def _load_from_cache(self, name: str, n_samples: int) -> Optional[dict]:
        cache_file = self.data_dir / f"{name}.npz"
        if cache_file.exists():
            try:
                data = np.load(cache_file, allow_pickle=True)
                patterns = data["patterns"][:n_samples]
                labels = data["labels"][:n_samples] if "labels" in data else None
                return {"patterns": patterns, "labels": labels}
            except Exception:
                pass
        return None

    # ------------------------------------------------------------------
    # Core simulation internals
    # ------------------------------------------------------------------

    def _init_patterns(self, patterns: np.ndarray, pattern_size: int) -> None:
        self.patterns = patterns
        self._pattern_size = pattern_size
        self._init_projection_weights(pattern_size)

    def _init_projection_weights(self, pattern_size: int) -> None:
        self._input_weights, self._input_nz_rows = self._random_dense_matrix(
            pattern_size, self.n_neurons
        )
        self._input_col_sums = self._input_weights.sum(axis=0)
        self._inter_area_weights = []
        self._inter_area_nz_rows = []
        self._inter_area_col_sums = []
        for _ in range(self.n_areas - 1):
            w, nz_rows = self._random_dense_matrix(self.n_neurons, self.n_neurons)
            self._inter_area_weights.append(w)
            self._inter_area_nz_rows.append(nz_rows)
            self._inter_area_col_sums.append(w.sum(axis=0))
        expected_active_inputs = pattern_size * self.connection_density
        self._weight_budget = max(expected_active_inputs, 1.0) * 1.5

    def _random_dense_matrix(
        self, n_rows: int, n_cols: int
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        n_active_per_col = max(1, int(round(n_rows * self.connection_density)))
        matrix = np.zeros((n_rows, n_cols), dtype=np.float32)
        nz_rows: list[np.ndarray] = []
        for j in range(n_cols):
            rows = self.rng.choice(n_rows, size=n_active_per_col, replace=False)
            matrix[rows, j] = 1.0
            nz_rows.append(rows)
        return matrix, nz_rows

    def _k_cap(self, scores: np.ndarray, k: int) -> np.ndarray:
        k = min(k, scores.shape[0])
        winners = np.argpartition(scores, -k)[-k:]
        activation = np.zeros_like(scores, dtype=bool)
        activation[winners] = True
        return activation

    def _project_step(
        self, pattern: np.ndarray, apply_plasticity: bool = True
    ) -> list[np.ndarray]:
        if self._input_weights is None:
            raise RuntimeError("Call generate_patterns() or load dataset before projecting.")
        activations: list[np.ndarray] = []
        active_inputs = np.where(pattern)[0]
        scores = self._input_weights[active_inputs, :].sum(axis=0)
        act0 = self._k_cap(scores, self.k)
        activations.append(act0)
        if apply_plasticity:
            active_targets = np.where(act0)[0]
            if active_inputs.size and active_targets.size:
                self._reinforce(
                    self._input_weights, self._input_col_sums,
                    self._input_nz_rows, active_inputs, active_targets,
                )
        prev_act = act0
        for i in range(self.n_areas - 1):
            w = self._inter_area_weights[i]
            col_sums = self._inter_area_col_sums[i]
            nz_rows = self._inter_area_nz_rows[i]
            active_src = np.where(prev_act)[0]
            scores = w[active_src, :].sum(axis=0)
            act_next = self._k_cap(scores, self.areas[i + 1].k)
            activations.append(act_next)
            if apply_plasticity:
                active_dst = np.where(act_next)[0]
                if active_src.size and active_dst.size:
                    self._reinforce(w, col_sums, nz_rows, active_src, active_dst)
            prev_act = act_next
        for area, act in zip(self.areas, activations):
            area.activation = act
        return activations

    def _reinforce(
        self,
        weight_matrix: np.ndarray,
        col_sums: np.ndarray,
        nz_rows: list[np.ndarray],
        active_src: np.ndarray,
        active_dst: np.ndarray,
    ) -> None:
        sub = weight_matrix[np.ix_(active_src, active_dst)]
        before = sub.sum(axis=0)
        sub = sub * (1.0 + self.plasticity_rate)
        weight_matrix[np.ix_(active_src, active_dst)] = sub
        after = sub.sum(axis=0)
        col_sums[active_dst] += after - before
        self._normalize_incoming_weights(weight_matrix, active_dst, col_sums, nz_rows)

    def _normalize_incoming_weights(
        self,
        weight_matrix: np.ndarray,
        target_indices: np.ndarray,
        col_sums: np.ndarray,
        nz_rows: list[np.ndarray],
    ) -> None:
        relevant_sums = col_sums[target_indices]
        nonzero = relevant_sums > 0
        if not np.any(nonzero):
            return
        budget = self._weight_budget
        scale = np.ones_like(relevant_sums)
        scale[nonzero] = budget / relevant_sums[nonzero]
        for col, s in zip(target_indices, scale):
            if s != 1.0:
                rows = nz_rows[col]
                weight_matrix[rows, col] *= s
        col_sums[target_indices] = relevant_sums * scale

    # ------------------------------------------------------------------
    # Public training / inference API
    # ------------------------------------------------------------------

    def learn_patterns(
        self, n_rounds: int = 150, progress_callback=None
    ) -> dict[int, np.ndarray]:
        if self.patterns is None:
            raise RuntimeError("Call generate_patterns() or load dataset before learn_patterns().")
        n_patterns = self.patterns.shape[0]
        final_activations: dict[int, np.ndarray] = {}
        self.training_snapshots: dict[int, list[np.ndarray]] = {p: [] for p in range(n_patterns)}
        for r in range(n_rounds):
            for p_idx in range(n_patterns):
                pattern = self.patterns[p_idx]
                activations = self._project_step(pattern, apply_plasticity=True)
                final_activations[p_idx] = activations[-1]
                self.training_snapshots[p_idx].append(activations[-1])
                if progress_callback is not None:
                    progress_callback(p_idx, r, n_patterns, n_rounds)
        self.assemblies = final_activations
        return self.assemblies

    def get_stability_history(self) -> list[dict]:
        if not getattr(self, "training_snapshots", None):
            return []
        history = []
        for p_idx, snapshots in self.training_snapshots.items():
            final = snapshots[-1]
            for r, act in enumerate(snapshots):
                overlap = np.logical_and(act, final).sum()
                stability = float(overlap) / float(max(1, final.sum()))
                history.append({"pattern": p_idx, "round": r, "stability": stability})
        return history

    def recognize_pattern(
        self, partial_pattern: np.ndarray
    ) -> tuple[Optional[int], dict[int, float]]:
        if not self.assemblies:
            raise RuntimeError("No assemblies learned yet.")
        activations = self._project_step(partial_pattern, apply_plasticity=False)
        probe_activation = activations[-1]
        scores: dict[int, float] = {}
        for label, assembly in self.assemblies.items():
            overlap = np.logical_and(probe_activation, assembly).sum()
            denom = max(1, assembly.sum())
            scores[label] = float(overlap) / float(denom)
        best_label = max(scores, key=scores.get) if scores else None
        if best_label is not None and scores[best_label] == 0.0:
            best_label = None
        return best_label, scores

    def pattern_completion(
        self, partial_pattern: np.ndarray, n_rounds: int = 5
    ) -> np.ndarray:
        if self.patterns is None or not self.assemblies:
            raise RuntimeError("Call generate_patterns() and learn_patterns() first.")
        current = partial_pattern.copy()
        for _ in range(n_rounds):
            self._project_step(current, apply_plasticity=False)
        best_label, scores = self.recognize_pattern(current)
        if best_label is None:
            return partial_pattern
        return self.patterns[best_label].copy()

    def get_area_activity_snapshot(self) -> np.ndarray:
        return np.stack([area.activation for area in self.areas], axis=0)

    def assembly_overlap_matrix(self) -> np.ndarray:
        labels = sorted(self.assemblies.keys())
        n = len(labels)
        mat = np.zeros((n, n), dtype=np.float32)
        for i, li in enumerate(labels):
            for j, lj in enumerate(labels):
                a, b = self.assemblies[li], self.assemblies[lj]
                union = np.logical_or(a, b).sum()
                inter = np.logical_and(a, b).sum()
                mat[i, j] = float(inter) / float(union) if union > 0 else 0.0
        return mat
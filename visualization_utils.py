"""
visualization_utils.py - Plotting and visualization helpers
"""

from __future__ import annotations

import logging
from typing import Optional
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


def _pattern_to_grid(pattern: np.ndarray) -> np.ndarray:
    """Convert a 1D binary pattern to a 2D grid for display."""
    pattern = np.asarray(pattern).astype(float)
    size = pattern.shape[0]

    # Known square sizes
    KNOWN_SIDES = {784: 28, 400: 20, 225: 15, 196: 14, 100: 10, 64: 8, 49: 7, 25: 5}
    if size in KNOWN_SIDES:
        side = KNOWN_SIDES[size]
        return pattern.reshape(side, side)

    # Try perfect square
    side = int(np.round(np.sqrt(size)))
    if side * side == size:
        return pattern.reshape(side, side)

    # Pad to next perfect square
    side = int(np.ceil(np.sqrt(size)))
    padded = np.zeros(side * side)
    padded[:size] = pattern
    return padded.reshape(side, side)


def plot_pattern_grid(pattern: np.ndarray, title: str = "Pattern", cmap: str = "Blues") -> plt.Figure:
    """Render a 1D binary pattern as a 2D grid image."""
    if pattern is None or len(pattern) == 0:
        pattern = np.zeros(100)

    pattern = np.asarray(pattern, dtype=float)

    # Ensure non-trivial range for display
    vmin, vmax = 0.0, 1.0
    if pattern.max() > 1.0:
        pattern = pattern / 255.0

    grid = _pattern_to_grid(pattern)

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.set_title(title, fontsize=11, color='white', pad=6, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.patch.set_facecolor('#0a1628')
    ax.set_facecolor('#0a1628')
    for spine in ax.spines.values():
        spine.set_color('#4a9eff')
        spine.set_linewidth(1.5)
    fig.tight_layout(pad=0.5)
    return fig


def plot_pattern_comparison(original: np.ndarray, partial: np.ndarray, completed: np.ndarray) -> plt.Figure:
    """Show original / partial / completed patterns side by side."""
    arrays = [np.asarray(a, dtype=float) for a in (original, partial, completed)]
    # Normalize if needed
    arrays = [a / 255.0 if a.max() > 1.0 else a for a in arrays]
    titles = ["Original", "Partial Input", "Completed"]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    fig.patch.set_facecolor('#0a1628')

    for ax, arr, title in zip(axes, arrays, titles):
        grid = _pattern_to_grid(arr)
        ax.imshow(grid, cmap="Blues", vmin=0, vmax=1, interpolation='nearest')
        ax.set_title(title, fontsize=12, color='white', fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor('#0a1628')
        for spine in ax.spines.values():
            spine.set_color('#4a9eff')
            spine.set_linewidth(1.5)
    fig.tight_layout()
    return fig


def plot_confidence_bars(scores: dict[int, float], highlight: Optional[int] = None) -> go.Figure:
    """Bar chart of recognition confidence scores."""
    if not scores:
        scores = {0: 0.0}
    labels = [f"Pattern {k}" for k in scores.keys()]
    values = list(scores.values())
    colors = ['#00ff88' if k == highlight else '#4a9eff' for k in scores.keys()]

    fig = go.Figure(data=[go.Bar(
        x=labels, y=values, marker_color=colors,
        text=[f"{v:.2f}" for v in values], textposition='outside', textfont_color='white'
    )])
    fig.update_layout(
        title="Recognition Confidence", title_font_color='white',
        yaxis_title="Confidence (assembly overlap)", yaxis_range=[0, 1.05],
        height=400, paper_bgcolor='#0a1628', plot_bgcolor='#0a1628',
        font_color='white', margin=dict(t=50, b=30, l=30, r=30),
        xaxis=dict(gridcolor='#1a3a6a', showgrid=True, tickfont_color='white', zeroline=False),
        yaxis=dict(gridcolor='#1a3a6a', showgrid=True, tickfont_color='white', zeroline=False),
        bargap=0.3,
    )
    return fig


def plot_activity_heatmap(activity: np.ndarray, area_labels: Optional[list[str]] = None) -> go.Figure:
    """Heatmap of per-area neural activity."""
    if activity is None or activity.size == 0:
        activity = np.zeros((1, 100))
    n_areas, n_neurons = activity.shape
    max_display = min(500, n_neurons)
    if n_neurons > max_display:
        idx = np.linspace(0, n_neurons - 1, max_display).astype(int)
        display_activity = activity[:, idx]
    else:
        display_activity = activity
    labels = area_labels or [f"Area {i}" for i in range(n_areas)]

    fig = go.Figure(data=go.Heatmap(
        z=display_activity.astype(int), y=labels,
        colorscale=[[0, '#0a1628'], [1, '#4a9eff']],
        showscale=True, colorbar=dict(title="Active", title_font_color='white', tickfont_color='white')
    ))
    fig.update_layout(
        title="Neural Activity (k-cap winners highlighted)", title_font_color='white',
        xaxis_title="Neuron index (subsampled)",
        height=250 + 40 * n_areas, paper_bgcolor='#0a1628', plot_bgcolor='#0a1628',
        font_color='white', margin=dict(t=50, b=30, l=60, r=30),
        xaxis=dict(gridcolor='#1a3a6a', showgrid=True, tickfont_color='white', zeroline=False),
        yaxis=dict(gridcolor='#1a3a6a', showgrid=True, tickfont_color='white', zeroline=False),
    )
    return fig


def plot_assembly_network(n_display_neurons: int = 40, active_indices: Optional[np.ndarray] = None,
                          area_label: str = "Area") -> go.Figure:
    """Draw an illustrative network graph highlighting an assembly."""
    if n_display_neurons < 3:
        n_display_neurons = 40
    G = nx.erdos_renyi_graph(n_display_neurons, p=0.08, seed=0)
    pos = nx.spring_layout(G, seed=0)
    active_set = set(active_indices.tolist()) if active_indices is not None else set()

    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines",
                            line=dict(width=0.5, color="#2a4a7a"), hoverinfo="none")

    node_x, node_y, node_color, node_size = [], [], [], []
    for n in G.nodes():
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        is_active = n in active_set
        node_color.append("#00ff88" if is_active else "#4a9eff")
        node_size.append(16 if is_active else 8)
    node_trace = go.Scatter(x=node_x, y=node_y, mode="markers", hoverinfo="none",
                            marker=dict(color=node_color, size=node_size,
                                        line=dict(width=1, color="#0a1628")))

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=f"{area_label}: Assembly Highlighted", title_font_color='white',
        showlegend=False, height=400, paper_bgcolor='#0a1628', plot_bgcolor='#0a1628',
        font_color='white', margin=dict(t=50, b=10, l=10, r=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    return fig


def plot_assembly_overlap_matrix(overlap: np.ndarray) -> go.Figure:
    """Heatmap of pairwise overlap between learned assemblies."""
    if overlap is None or overlap.size == 0:
        overlap = np.array([[1.0]])
    labels = [f"P{i}" for i in range(overlap.shape[0])]
    fig = go.Figure(data=go.Heatmap(
        z=overlap, x=labels, y=labels,
        colorscale=[[0, '#0a1628'], [0.5, '#1a3a6a'], [1, '#4a9eff']],
        zmin=0, zmax=1, text=np.round(overlap, 2),
        texttemplate="%{text}", textfont_color='white'
    ))
    fig.update_layout(
        title="Assembly Overlap Between Patterns", title_font_color='white',
        height=400, paper_bgcolor='#0a1628', plot_bgcolor='#0a1628',
        font_color='white', margin=dict(t=50, b=30, l=30, r=30),
        xaxis=dict(gridcolor='#1a3a6a', showgrid=True, tickfont_color='white', zeroline=False),
        yaxis=dict(gridcolor='#1a3a6a', showgrid=True, tickfont_color='white', zeroline=False),
    )
    return fig


def plot_training_timeline(history: list[dict]) -> go.Figure:
    """Line chart showing assembly stabilization over rounds."""
    if not history:
        history = [{"pattern": 0, "round": 0, "stability": 1.0}]
    fig = go.Figure()
    colors = ['#4a9eff', '#00ff88', '#ff6b6b', '#ffd93d', '#a66cff',
              '#ff6b9d', '#6bffb8', '#ffb86b']
    patterns = sorted(set(h["pattern"] for h in history))
    for idx, p in enumerate(patterns):
        rounds = [h["round"] for h in history if h["pattern"] == p]
        stability = [h["stability"] for h in history if h["pattern"] == p]
        color = colors[idx % len(colors)]
        fig.add_trace(go.Scatter(
            x=rounds, y=stability, mode="lines+markers", name=f"Pattern {p}",
            line=dict(color=color, width=2.5), marker=dict(color=color, size=8)
        ))
    fig.update_layout(
        title="Assembly Formation Timeline", title_font_color='white',
        xaxis_title="Training Round",
        yaxis_title="Stability (overlap with final assembly)",
        yaxis_range=[0, 1.05], height=400, paper_bgcolor='#0a1628', plot_bgcolor='#0a1628',
        font_color='white', margin=dict(t=50, b=40, l=50, r=30),
        legend=dict(font_color='white', bgcolor='rgba(10, 22, 40, 0.8)',
                    bordercolor='#1a3a6a', borderwidth=1),
        xaxis=dict(gridcolor='#1a3a6a', showgrid=True, tickfont_color='white', zeroline=False),
        yaxis=dict(gridcolor='#1a3a6a', showgrid=True, tickfont_color='white', zeroline=False),
    )
    return fig
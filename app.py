"""
app.py

Streamlit web application: Neural Assembly Pattern Learning with real datasets.
"""

from __future__ import annotations

import logging
import streamlit as st
import numpy as np
from config import load_config
from neural_assembly_simulator import NeuralAssemblySimulator
from visualization_utils import (
    plot_activity_heatmap,
    plot_assembly_network,
    plot_assembly_overlap_matrix,
    plot_confidence_bars,
    plot_pattern_comparison,
    plot_pattern_grid,
    plot_training_timeline,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration - DARK BLUE THEME
st.set_page_config(
    page_title="Neural Assembly Pattern Learning", 
    page_icon="🧠", 
    layout="wide"
)

# Custom CSS for COMPLETE DARK THEME
st.markdown("""
<style>
    /* FORCE DARK BACKGROUND ON EVERYTHING */
    html, body, .stApp, .stApp > header, .stApp > div,
    .main, .stAppViewContainer, .stAppViewContainer > div,
    section, .st-emotion-cache-1d391kg, .st-emotion-cache-12oz5g7 {
        background-color: #0a1628 !important;
    }
    
    /* SIDEBAR */
    .stSidebar, .css-1d391kg, .st-emotion-cache-1d391kg, 
    section[data-testid="stSidebar"], .st-emotion-cache-6qob1r {
        background-color: #0d1f3c !important;
    }
    
    /* DROPDOWN / SELECT BOX */
    .stSelectbox > div > div {
        background-color: #1a2a4a !important;
        border: 2px solid #4a9eff !important;
        border-radius: 8px !important;
        color: #ffffff !important;
        min-height: 38px !important;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #6ab0ff !important;
        box-shadow: 0 0 15px rgba(74, 158, 255, 0.2) !important;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #6ab0ff !important;
        box-shadow: 0 0 25px rgba(74, 158, 255, 0.3) !important;
    }
    
    .stSelectbox > div > div > div {
        color: #ffffff !important;
        background-color: transparent !important;
        font-weight: 500 !important;
        font-size: 15px !important;
        padding: 4px 8px !important;
    }
    
    .stSelectbox > div > div > div > svg {
        fill: #4a9eff !important;
        color: #4a9eff !important;
        width: 20px !important;
        height: 20px !important;
    }
    
    /* DROPDOWN MENU - POPUP LIST */
    div[data-baseweb="select"] > div {
        background-color: #1a2a4a !important;
        border: 2px solid #4a9eff !important;
        border-radius: 8px !important;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.9) !important;
        margin-top: 4px !important;
        max-height: 200px !important;
        overflow-y: auto !important;
    }
    
    div[data-baseweb="select"] > div > div {
        background-color: #1a2a4a !important;
        color: #ffffff !important;
        padding: 10px 16px !important;
        cursor: pointer !important;
        transition: all 0.2s !important;
        font-size: 15px !important;
        border-bottom: 1px solid #0d1f3c !important;
    }
    
    div[data-baseweb="select"] > div > div:hover {
        background-color: #2a4a7a !important;
        color: #ffffff !important;
        border-left: 4px solid #4a9eff !important;
        padding-left: 20px !important;
    }
    
    div[data-baseweb="select"] > div > div[aria-selected="true"] {
        background-color: #2a4a7a !important;
        color: #6ab0ff !important;
        font-weight: bold !important;
        border-left: 4px solid #4a9eff !important;
    }
    
    div[data-baseweb="select"] > div > div > div {
        color: #ffffff !important;
        font-weight: 500 !important;
    }
    
    div[data-baseweb="select"] > div::-webkit-scrollbar {
        width: 6px;
    }
    div[data-baseweb="select"] > div::-webkit-scrollbar-track {
        background: #0d1f3c;
    }
    div[data-baseweb="select"] > div::-webkit-scrollbar-thumb {
        background: #4a9eff;
        border-radius: 3px;
    }
    
    /* NUMBER INPUT */
    .stNumberInput > div > div > input {
        background-color: #1a2a4a !important;
        color: #ffffff !important;
        border: 2px solid #4a9eff !important;
        border-radius: 8px !important;
        padding: 8px 12px !important;
        font-size: 15px !important;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #6ab0ff !important;
        box-shadow: 0 0 20px rgba(74, 158, 255, 0.3) !important;
        outline: none !important;
    }
    
    .stNumberInput > div > div > button {
        background-color: #1a2a4a !important;
        border: 2px solid #4a9eff !important;
        color: #4a9eff !important;
    }
    
    .stNumberInput > div > div > button:hover {
        background-color: #2a4a7a !important;
        border-color: #6ab0ff !important;
        color: #ffffff !important;
    }
    
    /* SLIDERS */
    .stSlider > div > div > div {
        background-color: #4a9eff !important;
    }
    
    .stSlider > div > div > div > div {
        background-color: #4a9eff !important;
        border-color: #4a9eff !important;
        box-shadow: 0 0 15px rgba(74, 158, 255, 0.5) !important;
    }
    
    .stSlider label, .stSlider p {
        color: #c0d0e0 !important;
        font-weight: 500 !important;
    }
    
    /* ALL TEXT */
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown h1, .stMarkdown h2, 
    .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6,
    p, label, div, span, .stText, .stCaption, .stSelectbox label, 
    .stSlider label, .stNumberInput label, .stTextInput label,
    .stTextArea label, .stDateInput label, .stTimeInput label {
        color: #e0e8f0 !important;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #4a9eff !important;
    }
    
    .stSidebar .stMarkdown, .stSidebar p, .stSidebar label, 
    .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar .stSelectbox label,
    .stSidebar .stSlider label, .stSidebar .stNumberInput label {
        color: #c0d0e0 !important;
    }
    
    .stSidebar h1 { color: #4a9eff !important; }
    .stSidebar h2, .stSidebar h3 { color: #6ab0ff !important; }
    
    /* BUTTONS */
    .stButton > button {
        background-color: #1a3a6a !important;
        color: #ffffff !important;
        border: 2px solid #4a9eff !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        font-weight: bold !important;
        transition: all 0.3s !important;
        font-size: 16px !important;
    }
    
    .stButton > button:hover {
        background-color: #2a5a8a !important;
        border-color: #6ab0ff !important;
        box-shadow: 0 0 30px rgba(74, 158, 255, 0.4) !important;
        transform: translateY(-2px) !important;
        color: #ffffff !important;
    }
    
    .stButton > button:active {
        background-color: #0a3a5a !important;
        transform: translateY(0px) !important;
    }
    
    /* TABS */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0d1f3c !important;
        padding: 10px !important;
        border-radius: 10px !important;
        border: 1px solid #1a3a6a !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #0f2847 !important;
        color: #a0b0c0 !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        border: 1px solid #1a3a6a !important;
        font-weight: 500 !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #1a3a6a !important;
        color: #c0d0e0 !important;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #1a3a6a !important;
        color: #4a9eff !important;
        border: 2px solid #4a9eff !important;
        font-weight: bold !important;
    }
    
    /* PROGRESS BARS */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #1a3a6a, #4a9eff, #6ab0ff) !important;
    }
    
    .stProgress > div > div {
        background-color: #0f2847 !important;
    }
    
    /* ALERTS */
    .stAlert {
        background-color: #0f2847 !important;
        border: 1px solid #4a9eff !important;
        color: #e0e8f0 !important;
        border-radius: 8px !important;
    }
    
    /* DIVIDER */
    hr {
        border-color: #1a3a6a !important;
        border-width: 2px !important;
    }
    
    /* SCROLLBAR */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    ::-webkit-scrollbar-track {
        background: #0a1628;
        border-radius: 5px;
    }
    ::-webkit-scrollbar-thumb {
        background: #1a3a6a;
        border-radius: 5px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #4a9eff;
    }
    
    /* FILE UPLOADER */
    .stFileUploader {
        background-color: #0f2847 !important;
        border: 2px dashed #4a9eff !important;
        border-radius: 8px !important;
        padding: 20px !important;
    }
    
    .stFileUploader:hover {
        border-color: #6ab0ff !important;
        background-color: #1a3a6a !important;
    }
    
    .stFileUploader label {
        color: #c0d0e0 !important;
    }
</style>
""", unsafe_allow_html=True)

CFG = load_config("config.yaml")

# Session state initialization
def init_session_state() -> None:
    defaults = {
        "simulator": None,
        "patterns": None,
        "assemblies": None,
        "results": None,
        "training_history": None,
        "last_completion": None,
        "dataset_type": "synthetic",
        "init_error": None,
        "loading": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Sidebar configuration
def render_sidebar() -> dict:
    st.sidebar.title("🧠 Configuration")
    st.sidebar.markdown("---")

    st.sidebar.subheader("📊 Dataset")
    dataset_type = st.sidebar.selectbox(
        "Choose dataset",
        ["synthetic", "MNIST", "Fashion-MNIST", "Upload Custom Images"],
        help="Synthetic: Random patterns | MNIST: Handwritten digits | Fashion-MNIST: Clothing items"
    )
    
    if dataset_type == "Upload Custom Images":
        uploaded_files = st.sidebar.file_uploader(
            "Upload images", 
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Upload images of patterns you want the brain to learn"
        )
    else:
        uploaded_files = None
    
    n_patterns = st.sidebar.slider(
        "📈 Number of patterns/classes", min_value=2, max_value=10,
        value=min(CFG["patterns"]["n_patterns"], 5),
        help="For MNIST/Fashion-MNIST: how many digit/clothing classes to use"
    )

    st.sidebar.subheader("🧠 Brain Structure")
    n_neurons = st.sidebar.slider(
        "Neurons per area", min_value=200, max_value=10000,
        value=CFG["brain"]["n_neurons"], step=200,
    )
    n_areas = st.sidebar.slider(
        "Number of brain areas", min_value=1, max_value=5,
        value=CFG["brain"]["n_areas"],
    )
    cap_fraction = st.sidebar.slider(
        "Cap fraction (k / n)", min_value=0.001, max_value=0.05,
        value=CFG["brain"]["cap_fraction"], step=0.001, format="%.3f",
    )
    plasticity_rate = st.sidebar.slider(
        "Plasticity rate (β)", min_value=0.01, max_value=0.5,
        value=CFG["brain"]["plasticity_rate"], step=0.01,
    )
    pattern_size = st.sidebar.slider(
        "Pattern size (input units)", min_value=25, max_value=400,
        value=CFG["patterns"]["pattern_size"], step=25,
        help="Number of pixels to sample from each image"
    )

    st.sidebar.subheader("🎯 Training")
    n_rounds = st.sidebar.slider(
        "Training rounds per pattern", min_value=20, max_value=300,
        value=CFG["training"]["n_rounds"], step=10,
    )

    st.sidebar.divider()
    random_seed = st.sidebar.number_input(
        "🎲 Random seed", min_value=0, max_value=999999,
        value=CFG["brain"]["random_seed"],
    )

    return dict(
        dataset_type=dataset_type,
        uploaded_files=uploaded_files,
        n_patterns=n_patterns,
        n_neurons=n_neurons,
        n_areas=n_areas,
        cap_fraction=cap_fraction,
        plasticity_rate=plasticity_rate,
        pattern_size=pattern_size,
        n_rounds=n_rounds,
        random_seed=int(random_seed),
    )

# Actions
def action_init(params: dict) -> None:
    """Initialize simulator with real or synthetic data."""
    try:
        st.session_state.init_error = None
        
        simulator = NeuralAssemblySimulator(
            n_neurons=params["n_neurons"],
            n_areas=params["n_areas"],
            cap_fraction=params["cap_fraction"],
            plasticity_rate=params["plasticity_rate"],
            connection_density=CFG["brain"]["connection_density"],
            random_seed=params["random_seed"],
        )
        
        patterns = None
        dataset_display = params["dataset_type"]
        
        # Load data based on type
        if params["dataset_type"] == "synthetic":
            patterns = simulator.generate_patterns(
                n_patterns=params["n_patterns"], 
                pattern_size=params["pattern_size"]
            )
            st.success(f"✨ Generated {params['n_patterns']} synthetic patterns!")
            
        elif params["dataset_type"] == "MNIST":
            with st.spinner("📥 Loading MNIST dataset... This may take a moment."):
                patterns = simulator.load_mnist(
                    n_samples=params["n_patterns"],
                    sample_size=params["pattern_size"]
                )
            if patterns is not None and len(patterns) > 0:
                st.success(f"📊 Loaded {len(patterns)} digits from MNIST dataset!")
            else:
                st.warning("⚠️ MNIST download failed. Using synthetic patterns as fallback.")
                patterns = simulator.generate_patterns(
                    n_patterns=params["n_patterns"], 
                    pattern_size=params["pattern_size"]
                )
                dataset_display = "synthetic (fallback)"
            
        elif params["dataset_type"] == "Fashion-MNIST":
            with st.spinner("📥 Loading Fashion-MNIST dataset... This may take a moment."):
                patterns = simulator.load_fashion_mnist(
                    n_samples=params["n_patterns"],
                    sample_size=params["pattern_size"]
                )
            if patterns is not None and len(patterns) > 0:
                st.success(f"👕 Loaded {len(patterns)} items from Fashion-MNIST dataset!")
            else:
                st.warning("⚠️ Fashion-MNIST download failed. Using synthetic patterns as fallback.")
                patterns = simulator.generate_patterns(
                    n_patterns=params["n_patterns"], 
                    pattern_size=params["pattern_size"]
                )
                dataset_display = "synthetic (fallback)"
            
        elif params["dataset_type"] == "Upload Custom Images":
            if not params["uploaded_files"]:
                st.warning("Please upload images first.")
                return
            image_paths = []
            for uploaded_file in params["uploaded_files"]:
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                image_paths.append(temp_path)
            patterns = simulator.load_custom_images(
                image_paths=image_paths,
                sample_size=params["pattern_size"]
            )
            st.success(f"🖼️ Loaded {len(image_paths)} custom images!")
        
        # Store in session state
        st.session_state.simulator = simulator
        st.session_state.patterns = patterns
        st.session_state.assemblies = None
        st.session_state.results = None
        st.session_state.training_history = None
        st.session_state.dataset_type = dataset_display
        
        # Show pattern info
        if patterns is not None and len(patterns) > 0:
            st.info(f"📊 Pattern shape: {patterns.shape}")
            st.info(f"🔢 Number of patterns: {patterns.shape[0]}")
            st.info(f"📐 Pattern size: {patterns.shape[1]} units")
        
    except Exception as exc:
        logger.exception("Initialization failed")
        st.session_state.init_error = str(exc)
        st.error(f"❌ Failed to initialize: {exc}")
        st.info("💡 Try using 'synthetic' dataset instead, or check your internet connection.")

def action_train(n_rounds: int) -> None:
    simulator = st.session_state.simulator
    if simulator is None or st.session_state.patterns is None:
        st.warning("⚠️ Please initialize the brain and load data first.")
        return

    n_patterns = st.session_state.patterns.shape[0]
    progress_bar = st.progress(0.0, text="Starting training...")
    total_steps = n_patterns * n_rounds
    step_counter = {"count": 0}

    def callback(p_idx: int, r_idx: int, n_p: int, n_r: int) -> None:
        step_counter["count"] += 1
        frac = step_counter["count"] / total_steps
        progress_bar.progress(min(frac, 1.0), text=f"🧠 Training pattern {p_idx + 1}/{n_p}, round {r_idx + 1}/{n_r}")

    try:
        assemblies = simulator.learn_patterns(n_rounds=n_rounds, progress_callback=callback)
        progress_bar.progress(1.0, text="✅ Training complete!")

        st.session_state.assemblies = assemblies
        st.session_state.training_history = simulator.get_stability_history()
        st.success(f"🎯 Learned {n_patterns} assemblies over {n_rounds} rounds!")
    except Exception as exc:
        logger.exception("Training failed")
        st.error(f"❌ Training failed: {exc}")

def action_test(occlusion: float, pattern_idx: int) -> None:
    simulator = st.session_state.simulator
    if simulator is None or not st.session_state.assemblies:
        st.warning("⚠️ Please initialize and train the brain first.")
        return

    try:
        original = st.session_state.patterns[pattern_idx].copy()
        partial = original.copy()
        active_idx = np.where(partial)[0]
        n_to_remove = int(len(active_idx) * occlusion)
        if n_to_remove > 0:
            remove_idx = simulator.rng.choice(active_idx, size=n_to_remove, replace=False)
            partial[remove_idx] = False

        best_label, scores = simulator.recognize_pattern(partial)
        completed = simulator.pattern_completion(partial, n_rounds=CFG["training"]["completion_rounds"])

        st.session_state.results = {
            "original": original,
            "partial": partial,
            "completed": completed,
            "best_label": best_label,
            "scores": scores,
            "true_label": pattern_idx,
        }
    except Exception as exc:
        logger.exception("Testing failed")
        st.error(f"❌ Recognition/completion failed: {exc}")

# Main layout
def main() -> None:
    # Custom header with dark theme
    st.markdown("""
    <div style="text-align: center; padding: 20px 0; background: transparent;">
        <h1 style="color: #4a9eff; font-size: 3rem; text-shadow: 0 0 40px rgba(74, 158, 255, 0.3); margin: 0;">
            🧠 Neural Assembly Pattern Learning
        </h1>
        <p style="color: #80a0c0; font-size: 1.2rem; margin-top: 10px;">
            Real-world pattern recognition using Assembly Calculus with MNIST, Fashion-MNIST, and custom images
        </p>
    </div>
    """, unsafe_allow_html=True)

    params = render_sidebar()

    col_init, col_train, col_reset = st.sidebar.columns(3)
    with col_init:
        if st.button("🚀 Initialize", use_container_width=True):
            action_init(params)
    with col_train:
        if st.button("🎯 Train", use_container_width=True):
            action_train(params["n_rounds"])
    with col_reset:
        if st.button("🔄 Reset", use_container_width=True):
            for key in ("simulator", "patterns", "assemblies", "results", "training_history", "last_completion", "init_error"):
                st.session_state[key] = None
            st.rerun()

    tab_learn, tab_test, tab_visualize, tab_about = st.tabs(["📚 Learn", "🔍 Test", "📊 Visualize", "ℹ️ About"])

    # ============================================================
    # LEARN TAB - WITH MNIST/FASHION-MNIST SUPPORT
    # ============================================================
    with tab_learn:
        if st.session_state.patterns is None:
            st.info("💡 Click **Initialize** in the sidebar to load data and build the brain.")
        else:
            st.markdown(f"### 📊 Loaded Patterns ({st.session_state.dataset_type})")
            patterns = st.session_state.patterns
            
            # ============================================================
            # DEBUG: Show pattern info
            # ============================================================
            if patterns is not None and len(patterns) > 0:
                st.write("---")
                st.write("### 🔍 Pattern Information")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Number of Patterns", len(patterns))
                with col2:
                    st.metric("Size per Pattern", patterns[0].shape[0])
                with col3:
                    active = patterns[0].sum() if len(patterns) > 0 else 0
                    st.metric("Active Pixels (First)", int(active))
                
                # Detect dataset type from pattern size
                if patterns[0].shape[0] == 784:
                    st.success("✅ **MNIST/Fashion-MNIST detected!** (28×28 pixels)")
                elif patterns[0].shape[0] == 100:
                    st.info("📊 **Synthetic patterns** (10×10 pixels)")
                else:
                    st.info(f"📊 **Custom patterns** ({patterns[0].shape[0]} pixels)")
                st.write("---")
            # ============================================================
            
            if patterns is not None and len(patterns) > 0:
                # Display patterns in a grid
                cols = st.columns(min(5, patterns.shape[0]))
                for i in range(patterns.shape[0]):
                    with cols[i % len(cols)]:
                        label = f"Pattern {i}"
                        if (st.session_state.simulator and 
                            hasattr(st.session_state.simulator, 'labels') and 
                            st.session_state.simulator.labels is not None and
                            i < len(st.session_state.simulator.labels)):
                            label = f"Class {st.session_state.simulator.labels[i]}"
                        try:
                            fig = plot_pattern_grid(patterns[i], title=label)
                            st.pyplot(fig)
                        except Exception as e:
                            st.error(f"Error displaying pattern {i}: {str(e)}")
                            # Show raw data as fallback
                            st.write(f"First 20 values: {patterns[i][:20]}")
            else:
                st.warning("⚠️ No patterns loaded. Try initializing again with 'synthetic' data.")

            st.divider()
            
            # Show assemblies if trained
            if st.session_state.assemblies is None:
                st.info("🎯 Click **Train** in the sidebar to form assemblies for each pattern.")
            else:
                st.markdown("### 🧠 Learned Assemblies")
                left, right = st.columns(2)
                with left:
                    simulator = st.session_state.simulator
                    sample_assembly = st.session_state.assemblies[0]
                    active_sample = np.where(sample_assembly)[0]
                    display_n = CFG["visualization"]["network_display_neurons"]
                    sampled_active = active_sample[active_sample < display_n]
                    fig = plot_assembly_network(
                        n_display_neurons=display_n,
                        active_indices=sampled_active,
                        area_label=f"Final Area (Pattern 0)",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                with right:
                    if st.session_state.training_history:
                        fig = plot_training_timeline(st.session_state.training_history)
                        st.plotly_chart(fig, use_container_width=True)

                st.markdown("### 📊 Assembly Distinctiveness")
                overlap = st.session_state.simulator.assembly_overlap_matrix()
                fig = plot_assembly_overlap_matrix(overlap)
                st.plotly_chart(fig, use_container_width=True)

    # ============================================================
    # TEST TAB
    # ============================================================
    with tab_test:
        if not st.session_state.assemblies:
            st.info("🎯 Train the brain first (Learn tab / sidebar Train button).")
        else:
            n_patterns = st.session_state.patterns.shape[0]
            pattern_idx = st.selectbox(
                "🎯 Pattern to test", 
                options=list(range(n_patterns)), 
                format_func=lambda i: f"Pattern {i}" 
            )
            occlusion = st.slider(
                "🔍 Occlusion level", 
                min_value=0.0, max_value=0.9, value=0.4, step=0.05,
                help="Fraction of the pattern's active units to hide"
            )

            if st.button("🧪 Run Recognition & Completion", use_container_width=True):
                action_test(occlusion, pattern_idx)

            if st.session_state.results is not None:
                r = st.session_state.results
                st.markdown("### 🔄 Pattern Completion")
                fig = plot_pattern_comparison(r["original"], r["partial"], r["completed"])
                st.pyplot(fig)

                correct = r["best_label"] == r["true_label"]
                if r["best_label"] is None:
                    st.warning("⚠️ No confident match found — try lowering the occlusion level.")
                elif correct:
                    st.success(f"✅ Correctly recognized as Pattern {r['best_label']} (true label: {r['true_label']})! 🎉")
                else:
                    st.error(f"❌ Misrecognized as Pattern {r['best_label']} (true label: {r['true_label']}).")

                st.markdown("### 📊 Confidence Scores")
                fig = plot_confidence_bars(r["scores"], highlight=r["best_label"])
                st.plotly_chart(fig, use_container_width=True)

    # ============================================================
    # VISUALIZE TAB
    # ============================================================
    with tab_visualize:
        if st.session_state.simulator is None:
            st.info("💡 Initialize the brain first.")
        else:
            st.markdown("### 🔥 Current Neural Activity Heatmap")
            activity = st.session_state.simulator.get_area_activity_snapshot()
            fig = plot_activity_heatmap(activity, area_labels=[f"Area {i}" for i in range(activity.shape[0])])
            st.plotly_chart(fig, use_container_width=True)

    # ============================================================
    # ABOUT TAB
    # ============================================================
    with tab_about:
        st.markdown("""
        <div style="background-color: #0f2847; padding: 20px; border-radius: 10px; border: 1px solid #1a3a6a;">
        <h3 style="color: #4a9eff;">🧠 About this simulation</h3>
        
        <p style="color: #c0d0e0;">This app demonstrates the <strong style="color: #6ab0ff;">Assembly Calculus</strong> framework using real datasets:</p>
        
        <h4 style="color: #6ab0ff;">📊 Supported Datasets:</h4>
        <ul style="color: #c0d0e0;">
            <li><strong style="color: #4a9eff;">MNIST</strong>: Handwritten digits (0-9) - 28x28 grayscale images</li>
            <li><strong style="color: #4a9eff;">Fashion-MNIST</strong>: Clothing items - 28x28 grayscale images</li>
            <li><strong style="color: #4a9eff;">Custom Images</strong>: Upload your own images</li>
            <li><strong style="color: #4a9eff;">Synthetic</strong>: Random binary patterns (for comparison)</li>
        </ul>
        
        <h4 style="color: #6ab0ff;">⚙️ How it works:</h4>
        <ol style="color: #c0d0e0;">
            <li>Patterns (images) are projected into a simulated brain area</li>
            <li>Only the top-k most activated neurons fire (k-cap operation)</li>
            <li>Synapses between co-active neurons strengthen (Hebbian plasticity)</li>
            <li>Each pattern develops its own "neural assembly"</li>
            <li>The brain can recognize patterns from partial/noisy input</li>
        </ol>
        
        <h4 style="color: #6ab0ff;">📚 References:</h4>
        <ul style="color: #c0d0e0;">
            <li>Papadimitriou, C. H., et al. (2020). <em>Brain computation by assemblies of neurons.</em> PNAS.</li>
            <li>LeCun, Y., et al. (1998). <em>Gradient-based learning applied to document recognition.</em></li>
            <li>Xiao, H., et al. (2017). <em>Fashion-MNIST: A Novel Image Dataset for Machine Learning.</em></li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
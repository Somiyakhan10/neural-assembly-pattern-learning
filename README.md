# 🧠 Neural Assembly Pattern Learning

An interactive Streamlit simulation of the **Assembly Calculus** framework.

## Features

- **Multiple Datasets**: Synthetic, MNIST, Fashion-MNIST, and Custom Images
- **Real-time Training**: Watch neural assemblies form
- **Pattern Completion**: Test recognition with occluded patterns
- **Beautiful Visualizations**: Network graphs, heatmaps, confidence bars
- **Dark Theme**: Easy on the eyes

## Installation

```bash
pip install -r requirements.txt
streamlit run app.py

Usage
Select a dataset from the sidebar

Click "Initialize" to load patterns

Click "Train" to form neural assemblies

Test recognition in the "Test" tab

Explore visualizations

Deployment on Hugging Face Spaces
Create a new Space with Streamlit SDK

Upload all files

The Space will auto-deploy

Files
app.py - Main application

neural_assembly_simulator.py - Core simulation

visualization_utils.py - Plotting helpers

config.py - Configuration loader

config.yaml - Default settings

requirements.txt - Python dependencies

packages.txt - System packages

References
Papadimitriou, C. H., et al. (2020). Brain computation by assemblies of neurons. PNAS.

#  Neural Assembly Pattern Learning
[![Hugging Face](https://img.shields.io/badge/🤗-Neural%20Assembly%20Pattern%20Learning-ffd21e)](https://huggingface.co/spaces/Somiya855/Neural-Assembly-Pattern-Learning)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

##  Overview

Interactive simulation of the **Assembly Calculus** framework for visual pattern recognition. This app demonstrates how sparse populations of neurons can learn to recognize visual patterns through biologically-plausible mechanisms.

---

## Dashboard Overview

The main dashboard provides a clean interface with sidebar controls and multiple tabs for learning, testing, and visualization.

**Features:**
- Dark theme for comfortable viewing
- Sidebar controls for configuration
- Multiple tabs for different functionalities
- Real-time status updates

<img width="1891" height="856" alt="image" src="https://github.com/user-attachments/assets/5831a2b6-3893-4ea4-a627-9220c02422f2" />


---

##  Pattern Learning

The app supports multiple datasets including synthetic patterns, MNIST handwritten digits, and Fashion-MNIST clothing items.

**Learn Tab:**
- Displays loaded patterns as grids
- Shows pattern information (size, active pixels)
- Visualizes neural assemblies
<img width="1882" height="845" alt="image" src="https://github.com/user-attachments/assets/2d2254c4-cd3e-4363-adf0-11a3ff212c5c" />

---

##  Training Visualization

Watch neural assemblies form in real-time with interactive visualizations.

**Training Features:**
- Assembly formation timeline
- Stability tracking over rounds
- Assembly distinctiveness matrix

<img width="1847" height="825" alt="image" src="https://github.com/user-attachments/assets/73cd5cb0-bed3-412f-96dc-3c76eabec52d" />

---

##  Pattern Recognition

Test the brain's ability to recognize patterns from partial or occluded inputs.

**Test Tab:**
- Select patterns to test
- Adjust occlusion levels
- View pattern completion results
- Confidence score visualization

<img width="1898" height="839" alt="image" src="https://github.com/user-attachments/assets/25dfeb0f-5a9a-4a17-8ecc-2290968f6039" />

---

##  Neural Activity Heatmap

Visualize neural activity across different brain areas.

**Heatmap Features:**
- Per-area neural activity
- K-cap winners highlighted
- Interactive visualization

<img width="1349" height="779" alt="image" src="https://github.com/user-attachments/assets/a7f6bef1-4b66-416a-a1f2-5a0d15f4896b" />


---

##  Assembly Network

Visual representation of neural assemblies forming and connecting.

**Network Features:**
- Neurons as nodes
- Active assemblies highlighted
- Connection patterns visible

<img width="1290" height="708" alt="image" src="https://github.com/user-attachments/assets/fa4d908f-8ed3-41c6-8b1a-a25514591407" />


---

##  Confidence Scores

Recognition confidence visualization helps understand the brain's decision-making.

**Confidence Features:**
- Bar chart visualization
- Best match highlighted
- Confidence percentages displayed

<img width="1302" height="700" alt="image" src="https://github.com/user-attachments/assets/81f5c722-68c5-480a-ac7e-24ef3df39b23" />

---

##  Pattern Completion

Demonstration of the brain's ability to complete patterns from partial input.

**Completion Features:**
- Original pattern display
- Partial/occluded input
- Completed pattern reconstruction

<img width="1242" height="632" alt="image" src="https://github.com/user-attachments/assets/f5289a73-0d19-45e9-b769-36f5bbe10a7f" />

---

##  Configuration Sidebar

Full control over brain structure and training parameters.

**Configuration Options:**
- Dataset selection
- Brain structure (neurons, areas)
- Training parameters
- Random seed control

<img width="424" height="865" alt="image" src="https://github.com/user-attachments/assets/12eef704-885e-4e88-b83a-a1ca33cdb0c3" />

MNIST Dataset
Handwritten digits (0-4) - 28x28 grayscale images.
<img width="1296" height="728" alt="image" src="https://github.com/user-attachments/assets/4d459f19-d13b-4c44-b7bd-0d4325a02ab2" />
<img width="1322" height="514" alt="image" src="https://github.com/user-attachments/assets/8b37c0b4-59cf-4f49-9ae8-ff3f376982c9" />


Fashion-MNIST Dataset
<img width="1308" height="728" alt="image" src="https://github.com/user-attachments/assets/ade2dad2-3b48-493c-9d02-89619284cfe2" />
<img width="1335" height="632" alt="image" src="https://github.com/user-attachments/assets/576defa8-4d29-4a5e-b2ad-73d1c1a31d7e" />

---

##  Quick Start

```bash
# Clone the repository
git clone https://github.com/Somiyakhan10/neural-assembly-pattern-learning.git
cd neural-assembly-pattern-learning

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

Demo Link: https://huggingface.co/spaces/Somiya855/Neural-Assembly-Pattern-Learning

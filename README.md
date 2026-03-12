# PathGLS: Reference-free Evaluation for Pathology VLMs

## Overview

<p align="center">
  <img src="./images/pic.pdf" alt="PathGLS Architecture Overview" width="800">
</p>
**PathGLS** is a novel, reference-free evaluation framework designed to assess the trustworthiness of Vision-Language Models in computational pathology without relying on expert-annotated ground truths.

It evaluates pathology VLMs across three complementary dimensions:

1. **Grounding ($S_g$):** Fine-grained visual-text alignment using High-Resolution Multiple Instance Learning.
2. **Logic ($S_l$):** Entailment graph consistency using a domain-specific Natural Language Inference model to detect broken reasoning chains.
3. **Stability ($S_s$):** Output variance under adversarial visual perturbations and semantic prompt injections.

PathGLS supports both patch-level and whole-slide image level analysis.

## 1. System Requirements & Installation

PathGLS requires specific system-level libraries for WSI processing and specialized NLP models for logic extraction.

### Step 1.1: System Dependencies

For Debian/Ubuntu systems, the underlying C-library for reading WSIs is required:

```bash
sudo apt-get update
sudo apt-get install -y openslide-tools
Step 1.2: Python Environment
Create a virtual environment (Python 3.9+ recommended) and install standard dependencies:

Bash
conda create -n pathgls python=3.10 -y
conda activate pathgls
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
pip install -r requirements.txt
Step 1.3: Specialized Models
Download the specific SciSpacy model required by the Medical Knowledge Engine:

Bash
pip install [https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_md-0.5.1.tar.gz](https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_md-0.5.1.tar.gz)
2. Data Preparation
Before running the evaluation, you must configure the paths in config.py.

IMAGE_ROOT: Directory containing your images (e.g., .jpg, .png for ROI, or .svs, .ndpi, .tiff for WSIs).

INPUT_DATA_FILE: A JSON file containing the dataset metadata. The structure should be:

JSON
[
  {
    "image_id": "case_001",
    "original_path": "path/to/case_001.jpg",
    "text_control": "Optional: Ground truth or initial description."
  }
]
Model Configuration
You can switch the evaluation subject (the VLM being tested) by modifying MODEL_SOURCE in config.py:

"local": Uses the local VLM engine defined in LOCAL_MODEL_PATH.

"api": Uses a remote API (requires setting API_KEY, API_HOST, and API_MODEL_NAME).
```

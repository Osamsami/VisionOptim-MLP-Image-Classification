# VisionOptim

**A Multi-Layer Perceptron for image classification, implemented entirely from scratch in NumPy.**

VisionOptim trains a hand-rolled MLP (manual forward pass, manual backpropagation, ReLU/softmax, cross-entropy loss) on natural scene images, and compares how four different learning-rate strategies affect convergence and accuracy. No TensorFlow, PyTorch, or scikit-learn — every piece of the model and its evaluation metrics is implemented with plain NumPy.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/Osamsami/VisionOptim-MLP-Image-Classification/actions/workflows/ci.yml/badge.svg)](https://github.com/Osamsami/VisionOptim-MLP-Image-Classification/actions/workflows/ci.yml)
[![NumPy](https://img.shields.io/badge/built%20with-NumPy-013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)

---

## Overview

- **Model**: a 1-hidden-layer MLP (`input → Dense → ReLU → Dense → Softmax`), with forward pass, backpropagation, and parameter updates all implemented by hand — no autograd, no ML framework.
- **Optimization comparison**: the notebook trains four separate models under a constant learning rate, learning-rate scheduling (exponential decay), reduce-on-plateau, and an Adam-labeled configuration, and compares their loss/accuracy curves.
- **Evaluation**: every run reports both training accuracy and held-out test accuracy (on the dataset's `seg_test` split), plus a from-scratch confusion matrix and per-class precision/recall/F1 report.
- **Reusable pipeline**: the core MLP logic also lives in [`src/mlp.py`](src/mlp.py), used by a standalone [`train.py`](train.py) script that trains a model and saves its weights, and by [`app.py`](app.py), a Streamlit demo that loads those weights to classify an uploaded image.

## Dataset

[Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) — ~25,000 natural scene images across 6 classes: `buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`.

The dataset is **not included** in this repository (too large for version control). To use it:

1. Download and unzip it from Kaggle (link above).
2. Arrange it locally as:
   ```
   data/
     seg_train/<class_name>/*.jpg
     seg_test/<class_name>/*.jpg
   ```
3. Either keep it at `./data` relative to wherever you run the notebook/scripts from, or point `VISIONOPTIM_DATA_PATH` at its location.

## Model & Training Details

| | |
|---|---|
| Architecture | `Dense(input=64×64×3) → ReLU → Dense(hidden=128) → Softmax(6)` |
| Preprocessing | resize to 64×64, normalize to `[0, 1]`, flatten |
| Loss | categorical cross-entropy |
| Optimizers compared | constant LR, LR scheduling, reduce-on-plateau, Adam |
| Framework | none — forward/backward pass and gradients are hand-written NumPy |

## Results

Each of the four training runs in the notebook reports **both** training accuracy and held-out test accuracy (evaluated on `seg_test`, which the model never trains on) every epoch, along with a final train-vs-test comparison chart and a per-class precision/recall/F1 report. Actual numbers depend on the local run (dataset sample size, `limit_per_class`, epoch count, etc.) — run the notebook or `train.py` to reproduce them for your own setup. Training/loss curves and comparison charts from prior runs are saved under [`VisionOptim/visuals/`](VisionOptim/visuals/).

## Live Demo

A hosted Streamlit demo is planned for a later phase — link will be added here once deployed. In the meantime, run it locally:

```bash
streamlit run app.py
```

## Project Structure

```
.
├── VisionOptim/
│   ├── notebook/VisionOptim.ipynb   # main experiment notebook
│   └── visuals/                     # saved training/comparison plots
├── src/
│   └── mlp.py                       # shared from-scratch MLP implementation
├── train.py                         # standalone training script -> models/*.npz
├── app.py                           # Streamlit demo (upload image, get prediction)
├── models/                          # saved model weights (git-ignored)
├── requirements.txt
├── Dockerfile
└── .github/workflows/ci.yml         # installs deps, runs a training smoke test
```

## Installation & Usage

### 1. Clone and install dependencies

```bash
git clone https://github.com/Osamsami/VisionOptim-MLP-Image-Classification.git
cd VisionOptim-MLP-Image-Classification
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Get the dataset

See [Dataset](#dataset) above. Set `VISIONOPTIM_DATA_PATH` if it's not at `./data`:

```bash
export VISIONOPTIM_DATA_PATH=/path/to/dataset
```

### 3a. Run the notebook

```bash
jupyter notebook VisionOptim/notebook/VisionOptim.ipynb
```

### 3b. Or train from the command line

```bash
python train.py --epochs 20 --hidden-size 128
```

This saves weights to `models/mlp_weights.npz`.

### 4. Try the Streamlit demo

```bash
streamlit run app.py
```

Upload an image and get a predicted class from the trained model.

### Docker

```bash
docker build -t visionoptim .
docker run -p 8501:8501 -v /path/to/dataset:/app/data visionoptim
```

## Tech Stack

Python · NumPy · OpenCV · Matplotlib · Streamlit

## License

[MIT](LICENSE) © Osam Sami

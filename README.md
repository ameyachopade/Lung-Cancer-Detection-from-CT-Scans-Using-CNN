<div align="center">

# 🫁 Lung Cancer Detection from CT Scans Using CNN

**An interactive deep learning application for binary classification of lung CT scans**

[![Live Demo](https://api.gitlytics.dev/api/badge/tech.svg?slug=huggingface&style=icon_label_value&label=Hugging%20Face&variant=flat&value=Live%20Demo&value_color=%23FFD21E)](https://huggingface.co/spaces/ameyac11/Lung-Cancer-Detection-from-CT-Scans-Using-CNN)
[![Kaggle Model](https://api.gitlytics.dev/api/badge/tech.svg?slug=kaggle&style=icon_label_value&label=Kaggle&variant=flat&value=Models&value_color=%2320BEFF)](https://www.kaggle.com/models/ameyachopade/cnn-based-lung-cancer-detection-from-ct-scans/)
[![Kaggle Dataset](https://api.gitlytics.dev/api/badge/tech.svg?slug=kaggle&style=icon_label_value&label=Kaggle&variant=flat&value=Dataset&value_color=%2320BEFF)](https://www.kaggle.com/datasets/ameyachopade/ct-scan-binary-classification-dataset/data)
[![License](https://api.gitlytics.dev/api/badge/tech.svg?slug=license&style=icon_label_value&label=License&variant=flat&value=Apache%202.0&value_color=%23009688)](LICENSE)

<br>

[![Python](https://api.gitlytics.dev/api/badge/tech.svg?slug=python&style=icon_label_value&label=Language&variant=flat&value=Python%203.8%2B&value_color=%233776AB)](https://www.python.org/)
[![PyTorch](https://api.gitlytics.dev/api/badge/tech.svg?slug=pytorch&style=icon_label_value&label=AI%2FML&variant=flat&value=PyTorch&value_color=%23EE4C2C)](https://pytorch.org/)
[![Gradio](https://api.gitlytics.dev/api/badge/tech.svg?slug=gradio&style=icon_label_value&label=UI&variant=flat&value=Gradio&value_color=%23F97316)](https://gradio.app/)
[![ResNet](https://api.gitlytics.dev/api/badge/tech.svg?slug=pytorch&style=icon_label_value&label=Models&variant=flat&value=ResNet-18%2F34&value_color=%2300599C)](#-models)

*Fine-tuned **ResNet-18** and **ResNet-34** CNNs · Real-time inference · Gradio web UI*

[![Quick Start](https://api.gitlytics.dev/api/badge/tech.svg?slug=portfolio&style=icon_label_value&label=Section&value=Quick%20Start&variant=flat)](#-quick-start) · [![Features](https://api.gitlytics.dev/api/badge/tech.svg?slug=portfolio&style=icon_label_value&label=Section&value=Features&variant=flat)](#-features) · [![Project Structure](https://api.gitlytics.dev/api/badge/tech.svg?slug=portfolio&style=icon_label_value&label=Section&value=Project%20Structure&variant=flat)](#-project-structure) · [![Dataset](https://api.gitlytics.dev/api/badge/tech.svg?slug=portfolio&style=icon_label_value&label=Section&value=Dataset&variant=flat)](#-dataset) · [![License](https://api.gitlytics.dev/api/badge/tech.svg?slug=portfolio&style=icon_label_value&label=License&value=Section&variant=flat)](#-license)

</div>

---

## 📖 Overview

This project applies transfer learning with **ResNet** architectures to classify lung CT scan images as **Positive** (cancerous) or **Negative** (normal). A lightweight **Gradio** interface lets you upload scans, switch between models, and view confidence scores instantly - locally or on [![Hugging Face Spaces](https://api.gitlytics.dev/api/badge/tech.svg?slug=huggingface&style=icon_label_value&label=HF%20Space&value=Hugging%20Face%20Spaces&variant=flat)](https://huggingface.co/spaces/ameyac11/Lung-Cancer-Detection-from-CT-Scans-Using-CNN).

> ⚠️ **Disclaimer:** This tool is intended for **research and educational purposes only**. It is **not** a medical device and must **not** be used for clinical diagnosis or treatment decisions.

---

## 🔄 How It Works

```mermaid
flowchart TD
    A[CT Scan Upload] --> B[Preprocess 224x224]
    B --> C[ImageNet Normalize]
    C --> D{Model Selection}
    D -->|ResNet-18| E[ResNet-18 Inference]
    D -->|ResNet-34| F[ResNet-34 Inference]
    E --> G[Prediction and Confidence Scores]
    F --> G
```

| Stage | Details |
|:------|:--------|
| **Input** | Lung CT scan image (JPG/PNG) |
| **Preprocessing** | Resize to 224×224, ImageNet normalization |
| **Models** | ResNet-18 & ResNet-34 (ImageNet pre-trained, fine-tuned) |
| **Output** | Class label + confidence scores for both classes |
| **Weights** | Stored as `.safetensors` in `Model/` |

---

## ✨ Features

| | Feature | Description |
|:---:|:---|:---|
| 🔀 | **Multi-Model Support** | Toggle between **ResNet-18** and **ResNet-34** to compare predictions side by side |
| ⚡ | **Real-Time Inference** | Fast CPU/GPU classification powered by **PyTorch** |
| 🖥️ | **Interactive UI** | Drag-and-drop uploads, model selector, and clear confidence bars via **Gradio** - predictions for **Positive** (Benign or Malignant) and **Negative** (Normal) |
| 📂 | **Pre-loaded Examples** | 10 sample CT scans in `examples/` for instant testing - no downloads required |
| 🔒 | **Safe Weights** | Model checkpoints loaded with **Safetensors** for secure, efficient deserialization |

---

## ⚙️ Training Environment

The models were trained and evaluated using cloud computing infrastructure.

### Hardware
- NVIDIA Tesla T4 ×2 GPUs
- 30 GB Total GPU Memory

### Software
- Python
- PyTorch
- TorchVision
- CUDA

### Training Configuration
- Transfer Learning with ImageNet-pretrained weights
- Binary Classification (Positive vs Negative)
- CT Scan Image Input
- GPU-Accelerated Training

The use of dual NVIDIA Tesla T4 GPUs enabled efficient training and experimentation with ResNet-based architectures for lung cancer classification from CT scan images.

---

## 🚀 Quick Start

### Prerequisites

- Python **3.8+**
- [![Git LFS](https://api.gitlytics.dev/api/badge/tech.svg?slug=git&style=icon_label_value&label=Git&value=Git%20LFS&variant=flat)](https://git-lfs.com/) *(required to pull model weights in `Model/`)*

### 1️⃣ Clone the repository

```bash
git clone https://github.com/ameyachopade/Lung-Cancer-Detection-from-CT-Scans-Using-CNN.git
cd Lung-Cancer-Detection-from-CT-Scans-Using-CNN
git lfs pull
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

> 💡 **CPU-only PyTorch:** Uncomment the `--extra-index-url` line at the top of `requirements.txt` if you need the CPU-only wheel.

### 3️⃣ Launch the app

**Option A - Python script** *(recommended)*

```bash
python app.py
```

**Option B - Jupyter Notebook**

Open `app.ipynb` in Jupyter or VS Code, run all cells, and the Gradio server will start automatically.

🌐 Open your browser at **`http://127.0.0.1:7860`**

---

## 🧠 Models

| Model | Weights File | Architecture |
|:------|:-------------|:-------------|
| **ResNet-18** | `Model/resnet18_ct_v1.1.safetensors` | 18-layer residual network |
| **ResNet-34** | `Model/resnet34_ct_v1.safetensors` | 34-layer residual network *(default)* |

Both models use ImageNet pre-trained backbones with a replaced fully-connected layer for **2-class** binary classification.

📦 **Download separately:** [![Kaggle Models](https://api.gitlytics.dev/api/badge/tech.svg?slug=kaggle&style=icon_label_value&label=Kaggle&value=Kaggle%20Models&variant=flat)](https://www.kaggle.com/models/ameyachopade/cnn-based-lung-cancer-detection-from-ct-scans/)

---

## 📁 Project Structure

```
Lung-Cancer-Detection-from-CT-Scans-Using-CNN/
│
├── 📓 app.ipynb              # Gradio app (notebook version)
├── 🐍 app.py                 # Gradio app (script version)
├── 📋 requirements.txt       # Python dependencies
├── 📄 LICENSE                # Apache License 2.0
│
├── 🧠 Model/                 # Trained weights (Git LFS)
│   ├── resnet18_ct_v1.1.safetensors
│   └── resnet34_ct_v1.safetensors
│
└── 🖼️ examples/              # Sample CT scans for the demo UI
    ├── 1.jpg … 10.jpg
```

---

## 🔍 Dataset

This project uses a **derived binary-classification dataset** built from the [![IQ-OTH/NCCD Lung Ca...](https://api.gitlytics.dev/api/badge/tech.svg?slug=kaggle&style=icon_label_value&label=Kaggle&value=IQ-OTH%2FNCCD%20Lung%20Ca...&variant=flat)](https://www.kaggle.com/datasets/hamdallak/the-iqothnccd-lung-cancer-dataset).

| Original Class | Mapped To | Meaning |
|:---------------|:----------|:--------|
| Normal | **Negative** | No cancer detected |
| Benign + Malignant | **Positive** | Cancer indicators present |

The three-class source dataset was restructured into a **Positive / Negative** binary task to simplify lung cancer detection for deep learning and medical imaging research.

📦 **Access the derived dataset:** [![Kaggle - CT Scan Bi...](https://api.gitlytics.dev/api/badge/tech.svg?slug=kaggle&style=icon_label_value&label=Kaggle&value=Kaggle%20-%20CT%20Scan%20Bi...&variant=flat)](https://www.kaggle.com/datasets/ameyachopade/ct-scan-binary-classification-dataset/data)

> All credit for the original CT scan images belongs to the creators of the **IQ-OTH/NCCD Lung Cancer Dataset**.

---

## 🛠️ Tech Stack

<p align="center">
  <img src="https://api.gitlytics.dev/api/badge/tech.svg?slug=python&style=icon_label_value&label=Language&variant=flat&value=Python&value_color=%233776AB" alt="Python"/>
  <img src="https://api.gitlytics.dev/api/badge/tech.svg?slug=pytorch&style=icon_label_value&label=AI%2FML&variant=flat&value=PyTorch&value_color=%23EE4C2C" alt="PyTorch"/>
  <img src="https://api.gitlytics.dev/api/badge/tech.svg?slug=pytorch&style=icon_label_value&label=Vision&variant=flat&value=TorchVision&value_color=%23EE4C2C" alt="TorchVision"/>
  <img src="https://api.gitlytics.dev/api/badge/tech.svg?slug=gradio&style=icon_label_value&label=UI&variant=flat&value=Gradio&value_color=%23F97316" alt="Gradio"/>
  <img src="https://api.gitlytics.dev/api/badge/tech.svg?slug=huggingface&style=icon_label_value&label=Format&variant=flat&value=Safetensors&value_color=%23000000" alt="Safetensors"/>
  <img src="https://api.gitlytics.dev/api/badge/tech.svg?slug=python&style=icon_label_value&label=Images&variant=flat&value=Pillow&value_color=%23FFE873" alt="Pillow"/>
</p>

---

## 🌟 Support

If you find this project useful or interesting, please consider giving it a ⭐ on GitHub! Your support helps make the project more visible and encourages further development.

---

## 📄 License

This derived dataset and project code are distributed under the **[![Apache License 2.0](https://api.gitlytics.dev/api/badge/tech.svg?slug=portfolio&style=icon_label_value&label=License&value=Apache%202.0&variant=flat)](LICENSE)**.

Original CT scan images were sourced from the **IQ-OTH/NCCD Lung Cancer Dataset**. Users must review and comply with the terms, licensing, and attribution requirements of the original dataset in addition to Apache 2.0.

---

<div align="center">

**Built with ❤️ for medical imaging research**

[![Star this repo](https://api.gitlytics.dev/api/badge/tech.svg?slug=github&label_color=F1F3F5&style=icon_label_value&label=GitHub&value=Star%20this%20repo&variant=flat)](https://github.com/ameyachopade/Lung-Cancer-Detection-from-CT-Scans-Using-CNN) · [![Report an issue](https://api.gitlytics.dev/api/badge/tech.svg?slug=github&label_color=F1F3F5&style=icon_label_value&label=GitHub&value=Report%20an%20issue&variant=flat)](https://github.com/ameyachopade/Lung-Cancer-Detection-from-CT-Scans-Using-CNN/issues) · [![demo](https://api.gitlytics.dev/api/badge/tech.svg?slug=huggingface&style=icon_label_value&label=HF%20Space&value=demo&variant=flat)](https://huggingface.co/spaces/ameyac11/Lung-Cancer-Detection-from-CT-Scans-Using-CNN)

</div>

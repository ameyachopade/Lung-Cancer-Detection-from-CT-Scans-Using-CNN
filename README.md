# 🫁 Lung Cancer Detection from CT Scans Using CNN

An interactive deep learning application for classifying lung CT scans to detect the presence of cancer. This project utilizes fine-tuned **ResNet-18** and **ResNet-34** Convolutional Neural Network (CNN) models to provide real-time predictions with an intuitive Gradio web interface.

## 🚀 Live Demo & Models

* **Hugging Face Space (Live Demo):** [View and interact with the app here](https://huggingface.co/spaces/ameyac11/Lung-Cancer-Detection-from-CT-Scans-Using-CNN)
* **Kaggle Model:** [Access the trained models here](https://www.kaggle.com/models/ameyachopade/cnn-based-lung-cancer-detection-from-ct-scans/)
* **Kaggle Dataset:** [Access the dataset here](https://www.kaggle.com/datasets/ameyachopade/ct-scan-binary-classification-dataset/data)

## ✨ Features
* **Multi-Model Support:** Users can seamlessly toggle between **ResNet-18** and **ResNet-34** models to compare predictions.
* **Real-time Inference:** Fast and efficient image processing and classification using PyTorch.
* **Interactive UI:** Built using Gradio, enabling users to upload chest CT scan images and receive real-time predictions with confidence scores for **Positive** (Benign or Malignant) and **Negative** (Normal) classifications.
* **Pre-loaded Examples:** Includes sample CT scans for immediate testing without needing to download external images.

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

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AmeyC171/Lung-Cancer-Detection-from-CT-Scans-Using-CNN.git
   cd Lung-Cancer-Detection-from-CT-Scans-Using-CNN
   ```

2. **Install the required dependencies:**
   Make sure you have Python installed. Then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   You can run the application directly from the Jupyter Notebook:
   * Open `app.ipynb` in Jupyter Notebook or VS Code and run the cells.
   * The Gradio interface will launch on a local web server (usually `http://127.0.0.1:7860`).

## 📁 Repository Structure

* `app.ipynb`: The main Gradio application notebook containing the UI and inference logic.
* `Model/`: Contains the trained weights for the models (tracked via Git LFS).
  * `resnet18_ct_v1.pth`
  * `resnet18_ct_v1.1.safetensors`
  * `resnet34_ct_v1.safetensors`
* `examples/`: Sample CT scan images used for the Gradio examples.

## 🔍 Data Source

This dataset is a derived version of the IQ-OTH/NCCD Lung Cancer Dataset.

Original Dataset:
[The IQ-OTH/NCCD lung cancer dataset](https://www.kaggle.com/datasets/hamdallak/the-iqothnccd-lung-cancer-dataset)

The original dataset contains three classes:

- Normal
- Benign
- Malignant

For binary classification purposes, the dataset was restructured by merging the Benign and Malignant classes into a single **Positive** class, while the Normal class was retained as **Negative**.

This derived dataset was created to simplify lung cancer detection as a binary classification problem and to support deep learning, computer vision, and medical imaging research.

All credit for the original CT scan images belongs to the creators of the IQ-OTH/NCCD Lung Cancer Dataset.

---

## 📄 License

This derived dataset is distributed under the **Apache License 2.0**.

Original CT scan images were sourced from the IQ-OTH/NCCD Lung Cancer Dataset. Users should also review and comply with the terms, licensing information, and attribution requirements associated with the original dataset.

By using this dataset, you agree to comply with both the Apache 2.0 License for this derived work and any applicable terms of the original source dataset.

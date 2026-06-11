# 🫁 Lung Cancer Detection from CT Scans Using CNN

An interactive deep learning application for classifying lung CT scans to detect the presence of cancer. This project utilizes fine-tuned **ResNet-18** and **ResNet-34** Convolutional Neural Network (CNN) models to provide real-time predictions with an intuitive Gradio web interface.

## 🚀 Live Demo & Models

* **Hugging Face Space (Live Demo):** [View and interact with the app here](#) *(Add your Hugging Face link here)*
* **Kaggle Model / Dataset:** [Access the trained models and data here](#) *(Add your Kaggle link here)*

## ✨ Features
* **Multi-Model Support:** Users can seamlessly toggle between **ResNet-18** and **ResNet-34** models to compare predictions.
* **Real-time Inference:** Fast and efficient image processing and classification using PyTorch.
* **Interactive UI:** Built using Gradio, allowing for easy image uploads and displaying clear confidence scores for "Positive" (Cancerous) and "Negative" (Normal) classifications.
* **Pre-loaded Examples:** Includes sample CT scans for immediate testing without needing to download external images.

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
* `train.ipynb`: The notebook used for data preprocessing, training, and evaluating the ResNet models.
* `Model/`: Contains the trained weights for the models (tracked via Git LFS).
  * `resnet18_ct_v1.1.safetensors`
  * `resnet34_ct_v1.safetensors`
* `examples/`: Sample CT scan images used for the Gradio examples.

## 📄 License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for more details.

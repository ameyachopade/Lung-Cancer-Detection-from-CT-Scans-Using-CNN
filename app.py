import gradio as gr
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights

device = torch.device("cpu")
num_classes = 2

file_path = 'Classes/Classes.txt'
with open(file_path, 'r') as f:
    classes = [line.strip() for line in f]

model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("Model/Lung_Canacer_Model.pth", weights_only=True))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(image):
    img = Image.fromarray(image).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        predicted_idx = torch.argmax(probabilities).item()
    label = classes[predicted_idx]
    confidence = {classes[i]: float(probabilities[i]) for i in range(num_classes)}
    return label, confidence

example_images = [
    ["Dataset/Negative/Normal case (1).jpg"],
    ["Dataset/Negative/Normal case (5).jpg"],
    ["Dataset/Positive/Bengin case (1).jpg"],
    ["Dataset/Positive/Bengin case (10).jpg"],
    ["Dataset/Positive/Malignant case (1).jpg"],
    ["Dataset/Positive/Malignant case (5).jpg"],
]

with gr.Blocks(title="Lung Cancer Detection") as demo:
    gr.Markdown(
        """
        # 🫁 Lung Cancer Detection
        Upload a lung CT scan image to detect whether it shows signs of **cancer** or is **normal**.
        The model uses a ResNet-18 CNN trained on lung CT scan data.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(label="Upload CT Scan Image", type="numpy")
            predict_btn = gr.Button("🔍 Analyze", variant="primary")

        with gr.Column(scale=1):
            label_output = gr.Label(label="Prediction")
            confidence_output = gr.Label(label="Confidence Scores", num_top_classes=num_classes)

    gr.Examples(
        examples=example_images,
        inputs=image_input,
        label="Example CT Scan Images"
    )

    predict_btn.click(
        fn=predict,
        inputs=image_input,
        outputs=[label_output, confidence_output]
    )

if __name__ == "__main__":
    demo.launch()

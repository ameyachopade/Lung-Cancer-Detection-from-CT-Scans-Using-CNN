import gradio as gr
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights, ResNet34_Weights
from safetensors.torch import load_model

# Setup device and classes
device = torch.device("cpu")
num_classes = 2
classes = ["Positive", "Negative"]

# Load ResNet-18 model
resnet18 = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
resnet18.fc = nn.Linear(resnet18.fc.in_features, num_classes)
try:
    load_model(resnet18, "Model/resnet18_ct_v1.1.safetensors")
except Exception as e:
    print(f"Warning: Could not load ResNet18. Error: {e}")
resnet18 = resnet18.to(device)
resnet18.eval()

# Load ResNet-34 model
resnet34 = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
resnet34.fc = nn.Linear(resnet34.fc.in_features, num_classes)
try:
    load_model(resnet34, "Model/resnet34_ct_v1.safetensors")
except Exception as e:
    print(f"Warning: Could not load ResNet34. Error: {e}")
resnet34 = resnet34.to(device)
resnet34.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Prediction function
def predict(image, model_choice):
    if image is None:
        return "No image provided", {}
    
    # Process image
    img = Image.fromarray(image).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Select requested model
    selected_model = resnet18 if model_choice == "ResNet-18" else resnet34
    
    # Run inference
    with torch.no_grad():
        output = selected_model(img_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        predicted_idx = torch.argmax(probabilities).item()
        
    # Format results
    label = classes[predicted_idx]
    confidence = {classes[i]: float(probabilities[i]) for i in range(num_classes)}
    return label, confidence

# Define example image paths
example_images = [
    ["examples/1.jpg", "ResNet-34"],
    ["examples/2.jpg", "ResNet-34"],
    ["examples/3.jpg", "ResNet-34"],
    ["examples/4.jpg", "ResNet-34"],
    ["examples/5.jpg", "ResNet-34"],
    ["examples/6.jpg", "ResNet-34"],
    ["examples/7.jpg", "ResNet-34"],
    ["examples/8.jpg", "ResNet-34"],
    ["examples/9.jpg", "ResNet-34"],
    ["examples/10.jpg", "ResNet-34"],
]

# Build Gradio UI
with gr.Blocks(title="Lung Cancer Detection") as demo:
    gr.Markdown(
        """
        # 🫁 Lung Cancer Detection
        Upload a lung CT scan image to detect whether it shows signs of **cancer** or is **normal**.
        You can choose between **ResNet-18** and **ResNet-34** models.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(label="Upload CT Scan Image", type="numpy")
            model_selector = gr.Dropdown(
                choices=["ResNet-18", "ResNet-34"], 
                value="ResNet-34", 
                label="Select Model"
            )
            predict_btn = gr.Button("🔍 Analyze", variant="primary")

        with gr.Column(scale=1):
            label_output = gr.Label(label="Prediction")
            confidence_output = gr.Label(label="Confidence Scores", num_top_classes=num_classes)

    gr.Examples(
        examples=example_images,
        inputs=[image_input, model_selector],
        label="Example CT Scan Images"
    )

    # Link button to function
    predict_btn.click(
        fn=predict,
        inputs=[image_input, model_selector],
        outputs=[label_output, confidence_output]
    )

# Launch app
if __name__ == "__main__":
    demo.launch()

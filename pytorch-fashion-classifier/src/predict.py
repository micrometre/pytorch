import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from model import NeuralNetwork
from torchvision.transforms import ToTensor
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np


classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

def load_model(model_path, device):
    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def predict(model, image_tensor, device):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        pred = model(image_tensor)
        predicted_class = classes[pred[0].argmax(0)]
        confidence = torch.nn.functional.softmax(pred[0], dim=0).max().item()
        return predicted_class, confidence

def show_prediction(image_tensor, predicted_class, actual_class, confidence):
    """Display the image with prediction results"""
    # Convert tensor to numpy array and reshape for display
    image_np = image_tensor.squeeze().cpu().numpy()
    
    plt.figure(figsize=(8, 6))
    plt.imshow(image_np, cmap='gray')
    plt.title(f'Predicted: "{predicted_class}" (Confidence: {confidence:.2f})\nActual: "{actual_class}"', 
              fontsize=14, pad=20)
    plt.axis('off')
    
    # Save the prediction image
    os.makedirs('predictions', exist_ok=True)
    plt.savefig('predictions/prediction_result.png', bbox_inches='tight', dpi=150)
    plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "models/fashion_mnist_model.pth"  # Update with your model path
    model = load_model(model_path, device)

    x, y = test_data[0][0], test_data[0][1]
    
    # Get prediction using the existing predict function
    predicted_class, confidence = predict(model, x.unsqueeze(0), device)
    actual_class = classes[y]
    
    print(f'Predicted: "{predicted_class}", Actual: "{actual_class}"')
    print(f'Confidence: {confidence:.2f}')
    
    # Display the image with prediction
    show_prediction(x, predicted_class, actual_class, confidence)
    print("Prediction image saved to 'predictions/prediction_result.png'")


if __name__ == "__main__":
    main()
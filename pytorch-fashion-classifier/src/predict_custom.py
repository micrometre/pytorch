import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from model import NeuralNetwork

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

def load_model(model_path, device):
    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])
    custom_image = Image.open(image_path)
    custom_tensor = transform(custom_image).unsqueeze(0)
    return custom_tensor

def predict(model, image_tensor, device):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        pred = model(image_tensor)
        predicted_class = classes[pred[0].argmax(0)]
        confidence = torch.nn.functional.softmax(pred[0], dim=0).max().item()
        return predicted_class, confidence

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "models/fashion_mnist_model.pth"  # Update with your model path
    model = load_model(model_path, device)

    test_images_folder = "test_images"
    if os.path.exists(test_images_folder):
        image_files = [f for f in os.listdir(test_images_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if image_files:
            image_path = os.path.join(test_images_folder, image_files[0])
            print(f"\nTesting with custom image: {image_path}")
            custom_tensor = preprocess_image(image_path)
            predicted_class, confidence = predict(model, custom_tensor, device)
            print(f'Custom image predicted: "{predicted_class}" (confidence: {confidence:.2%})')
        else:
            print(f"No image files found in {test_images_folder} folder")
    else:
        print(f"Folder {test_images_folder} not found")

if __name__ == "__main__":
    main()
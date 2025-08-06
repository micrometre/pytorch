import torch
import torchvision.transforms as transforms

def get_device():
    """Get the appropriate device (CUDA if available, otherwise CPU)"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def resize_and_normalize(image):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform(image)

def load_image(image_path):
    from PIL import Image
    return Image.open(image_path)

def log_training_progress(epoch, loss, current, size):
    print(f"Epoch {epoch+1}, Loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def calculate_accuracy(predictions, labels):
    return (predictions.argmax(1) == labels).type(torch.float).sum().item()
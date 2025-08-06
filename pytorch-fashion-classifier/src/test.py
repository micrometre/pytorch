import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from model import NeuralNetwork
from utils import get_device
from data_loader import load_data

def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def main():
    # Set device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load data
    train_dataloader, test_dataloader = load_data()
    
    # Initialize model
    model = NeuralNetwork().to(device)
    
    # Load trained model
    model_path = "models/fashion_mnist_model.pth"
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded from {model_path}")
    except FileNotFoundError:
        print(f"Model file {model_path} not found. Please train the model first.")
        return
    
    # Set loss function
    loss_fn = nn.CrossEntropyLoss()
    
    print("Starting testing...")
    test(test_dataloader, model, loss_fn, device)

if __name__ == "__main__":
    main()
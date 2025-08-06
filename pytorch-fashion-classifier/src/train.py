import os
import torch
import torch.nn as nn
from tqdm import tqdm
from data_loader import load_data
from model import NeuralNetwork
from utils import get_device

def train(dataloader, model, loss_fn, optimizer, epochs, device):
    size = len(dataloader.dataset)
    model.train()
    for epoch in range(epochs):
        for batch, (X, y) in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs}"):
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                current = (batch + 1) * len(X)
                print(f"Epoch [{epoch + 1}/{epochs}], Batch [{batch + 1}/{len(dataloader)}], Loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")

def main():
    # Set device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load data
    train_dataloader, test_dataloader = load_data()
    
    # Initialize model
    model = NeuralNetwork().to(device)
    
    # Set loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    
    # Training parameters
    epochs = 5
    
    print("Starting training...")
    train(train_dataloader, model, loss_fn, optimizer, epochs, device)
    
    # Save the model
    os.makedirs("models", exist_ok=True)
    model_path = "models/fashion_mnist_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
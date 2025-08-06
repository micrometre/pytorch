import os                          # For file system operations (creating directories)
import torch                       # Core PyTorch library for tensor operations
import torch.nn as nn             # Neural network module for loss functions
from tqdm import tqdm             # Progress bar library for training visualization
from data_loader import load_data # Custom function to load FashionMNIST dataset
from model import NeuralNetwork   # Custom neural network model definition
from utils import get_device      # Utility function to determine GPU/CPU device

def train(dataloader, model, loss_fn, optimizer, epochs, device):
    """
    Train the neural network model using the provided data and parameters.
    
    Args:
        dataloader: DataLoader containing training batches
        model: Neural network model to train
        loss_fn: Loss function (e.g., CrossEntropyLoss)
        optimizer: Optimization algorithm (e.g., SGD, Adam)
        epochs: Number of complete passes through the dataset
        device: Device to run training on (CPU or GPU)
    """
    # Get total number of training samples for progress reporting
    size = len(dataloader.dataset)
    
    # Set model to training mode - enables dropout, batch norm training behavior
    model.train()
    
    # Outer loop: iterate through epochs (complete passes through dataset)
    for epoch in range(epochs):
        # Inner loop: iterate through batches within each epoch
        # tqdm provides a progress bar showing training progress
        for batch, (X, y) in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs}"):
            # Move data to the specified device (GPU/CPU) for computation
            # X: input images, y: corresponding labels
            X, y = X.to(device), y.to(device)

            # Forward pass: compute model predictions for the current batch
            pred = model(X)
            
            # Calculate loss: compare predictions with true labels
            # CrossEntropyLoss combines softmax and negative log likelihood
            loss = loss_fn(pred, y)

            # Backward pass: compute gradients
            # Clear gradients from previous iteration (PyTorch accumulates gradients)
            optimizer.zero_grad()
            
            # Compute gradients of loss with respect to model parameters
            loss.backward()
            
            # Update model parameters using computed gradients
            optimizer.step()

            # Print progress every 100 batches to monitor training
            if batch % 100 == 0:
                # Calculate number of samples processed so far
                current = (batch + 1) * len(X)
                print(f"Epoch [{epoch + 1}/{epochs}], Batch [{batch + 1}/{len(dataloader)}], Loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")

def main():
    """
    Main function that orchestrates the entire training process.
    """
    # Determine the best available device (GPU if available, otherwise CPU)
    device = get_device()
    print(f"Using device: {device}")
    
    # Load the FashionMNIST dataset and create data loaders
    # Returns tuple of (train_dataloader, test_dataloader)
    train_dataloader, test_dataloader = load_data()
    
    # Initialize the neural network model and move it to the specified device
    # .to(device) ensures model parameters are on the same device as input data
    model = NeuralNetwork().to(device)
    
    # Define loss function and optimizer
    # CrossEntropyLoss: combines softmax activation and negative log likelihood
    # Suitable for multi-class classification problems
    loss_fn = nn.CrossEntropyLoss()
    
    # SGD (Stochastic Gradient Descent) optimizer with learning rate 0.001
    # Updates model parameters based on computed gradients
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    
    # Set training hyperparameters
    epochs = 5  # Number of complete passes through the training dataset
    
    print("Starting training...")
    # Execute the training loop
    train(train_dataloader, model, loss_fn, optimizer, epochs, device)
    
    # Save the trained model for later use
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    model_path = "models/fashion_mnist_model.pth"
    
    # Save only the model's state dictionary (parameters) rather than the entire model
    # This is the recommended approach as it's more portable and efficient
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

# Execute main function only if script is run directly (not imported)
if __name__ == "__main__":
    main()
import torch                       # Core PyTorch library for tensor operations
import torch.nn as nn             # Neural network module for loss functions
from torch.utils.data import DataLoader  # Data loading utilities (imported but not directly used)
from torchvision import datasets  # Dataset utilities (imported but not directly used)
from torchvision.transforms import ToTensor  # Transform utilities (imported but not directly used)
from model import NeuralNetwork   # Custom neural network model definition
from utils import get_device      # Utility function to determine GPU/CPU device
from data_loader import load_data # Custom function to load FashionMNIST dataset

def test(dataloader, model, loss_fn, device):
    """
    Evaluate the trained neural network model on test data.
    
    Args:
        dataloader: DataLoader containing test batches
        model: Trained neural network model to evaluate
        loss_fn: Loss function used to calculate test loss
        device: Device to run evaluation on (CPU or GPU)
    """
    # Get total number of test samples and batches for metric calculations
    size = len(dataloader.dataset)        # Total number of test samples (10,000 for FashionMNIST)
    num_batches = len(dataloader)         # Number of batches in test set
    
    # Set model to evaluation mode - disables dropout, sets batch norm to eval mode
    # This ensures consistent behavior during inference
    model.eval()
    
    # Initialize metrics to track performance
    test_loss, correct = 0, 0
    
    # Disable gradient computation for efficiency during evaluation
    # torch.no_grad() reduces memory consumption and speeds up computation
    with torch.no_grad():
        # Iterate through all test batches
        for X, y in dataloader:
            # Move test data to the same device as the model
            # X: test images, y: true labels
            X, y = X.to(device), y.to(device)
            
            # Forward pass: get model predictions for test batch
            pred = model(X)
            
            # Accumulate test loss across all batches
            # .item() extracts scalar value from tensor
            test_loss += loss_fn(pred, y).item()
            
            # Calculate number of correct predictions in this batch
            # pred.argmax(1) gets the class with highest probability for each sample
            # Compare with true labels (y) and count matches
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    # Calculate average loss and accuracy across entire test set
    test_loss /= num_batches          # Average loss per batch
    correct /= size                   # Accuracy as fraction of correct predictions
    
    # Display test results in a formatted way
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def main():
    """
    Main function that orchestrates the model evaluation process.
    """
    # Determine the best available device (GPU if available, otherwise CPU)
    device = get_device()
    print(f"Using device: {device}")
    
    # Load the FashionMNIST dataset and create data loaders
    # We only need the test_dataloader for evaluation
    train_dataloader, test_dataloader = load_data()
    
    # Initialize the neural network model with the same architecture used for training
    # Move model to the specified device for computation
    model = NeuralNetwork().to(device)
    
    # Define path to the saved trained model
    model_path = "models/fashion_mnist_model.pth"
    
    # Load the trained model weights
    try:
        # Load the saved state dictionary (model parameters) from file
        # map_location ensures model loads on correct device (CPU/GPU)
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded from {model_path}")
    except FileNotFoundError:
        # Handle case where model file doesn't exist
        print(f"Model file {model_path} not found. Please train the model first.")
        return  # Exit function if no trained model is available
    
    # Set up the same loss function used during training
    # CrossEntropyLoss for multi-class classification
    loss_fn = nn.CrossEntropyLoss()
    
    print("Starting testing...")
    # Execute the evaluation process
    test(test_dataloader, model, loss_fn, device)

# Execute main function only if script is run directly (not imported)
if __name__ == "__main__":
    main()
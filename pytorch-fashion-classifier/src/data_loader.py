# Import necessary PyTorch modules for data handling
import torch
from torch.utils.data import DataLoader  # For creating data loaders that batch and shuffle data
from torchvision import datasets          # Pre-built datasets including FashionMNIST
from torchvision.transforms import ToTensor  # Transform to convert PIL images to tensors

def load_data(batch_size=64):
    """
    Load and prepare FashionMNIST dataset for training and testing.
    
    Args:
        batch_size (int): Number of samples per batch. Default is 64.
        
    Returns:
        tuple: A tuple containing (train_dataloader, test_dataloader)
    """
    
    # Load the training dataset
    # FashionMNIST contains 60,000 training images of 10 fashion categories
    training_data = datasets.FashionMNIST(
        root="data",           # Directory to store/load the dataset
        train=True,            # Load the training split (60,000 images)
        download=True,         # Download dataset if not already present
        transform=ToTensor(),  # Convert PIL images to PyTorch tensors and normalize to [0,1]
    )

    # Load the test dataset
    # FashionMNIST contains 10,000 test images for evaluation
    test_data = datasets.FashionMNIST(
        root="data",           # Same directory as training data
        train=False,           # Load the test split (10,000 images)
        download=True,         # Download if needed (will skip if already downloaded)
        transform=ToTensor(),  # Apply same transformation as training data
    )

    # Create data loader for training data
    # DataLoader handles batching, shuffling, and parallel data loading
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    
    # Create data loader for test data
    # Note: Test data typically doesn't need shuffling
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    # Return both data loaders for use in training and evaluation
    return train_dataloader, test_dataloader
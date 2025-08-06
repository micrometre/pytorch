# Import PyTorch's neural network module which contains layers and functions for building models
import torch.nn as nn

class NeuralNetwork(nn.Module):
    """
    A simple feedforward neural network for FashionMNIST classification.
    
    This network uses fully connected (linear) layers with ReLU activation
    to classify 28x28 grayscale images into 10 fashion categories.
    """
    
    def __init__(self):
        """
        Initialize the neural network architecture.
        
        The network consists of:
        - Input flattening layer (converts 28x28 images to 784-dimensional vectors)
        - Two hidden layers with 512 neurons each and ReLU activation
        - Output layer with 10 neurons (one for each fashion category)
        """
        # Call parent class constructor to properly initialize nn.Module
        super().__init__()
        
        # Flatten layer: converts 2D image tensors (28x28) into 1D vectors (784)
        # This is necessary because linear layers expect 1D input
        self.flatten = nn.Flatten()
        
        # Sequential container that chains layers together in order
        # Data flows through each layer sequentially during forward pass
        self.linear_relu_stack = nn.Sequential(
            # First hidden layer: 784 input features -> 512 output features
            # 28*28 = 784 (flattened image pixels) to 512 neurons
            nn.Linear(28*28, 512),
            
            # ReLU activation function: f(x) = max(0, x)
            # Introduces non-linearity and helps model learn complex patterns
            nn.ReLU(),
            
            # Second hidden layer: 512 input features -> 512 output features
            # Allows the model to learn more complex representations
            nn.Linear(512, 512),
            
            # Second ReLU activation for the second hidden layer
            nn.ReLU(),
            
            # Output layer: 512 input features -> 10 output features
            # 10 outputs correspond to the 10 FashionMNIST classes:
            # T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
            nn.Linear(512, 10)
        )

    def forward(self, x):
        """
        Define the forward pass of the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28)
                             representing a batch of grayscale images
        
        Returns:
            torch.Tensor: Output logits of shape (batch_size, 10)
                         Raw scores for each of the 10 classes (before softmax)
        """
        # Step 1: Flatten the input images from (batch_size, 1, 28, 28) to (batch_size, 784)
        # This converts each 28x28 image into a 784-dimensional vector
        x = self.flatten(x)
        
        # Step 2: Pass the flattened input through the linear layers with ReLU activations
        # The sequential stack applies: Linear -> ReLU -> Linear -> ReLU -> Linear
        logits = self.linear_relu_stack(x)
        
        # Step 3: Return raw logits (unnormalized scores)
        # Note: We return logits rather than probabilities because PyTorch's
        # CrossEntropyLoss expects raw logits and applies softmax internally
        return logits
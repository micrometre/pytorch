import os                                    # For creating directories and file operations
import torch                                 # Core PyTorch library for tensor operations
from PIL import Image                        # Python Imaging Library for image processing
import torchvision.transforms as transforms  # Image transformation utilities
from model import NeuralNetwork             # Custom neural network model definition
from torchvision.transforms import ToTensor  # Transform to convert images to tensors
from torchvision import datasets             # Pre-built datasets including FashionMNIST
import matplotlib.pyplot as plt             # Plotting library for visualization
import numpy as np                           # Numerical computing library

# Define the 10 FashionMNIST class labels in order (indices 0-9)
# These correspond to the output neurons of our neural network
classes = [
    "T-shirt/top",   # Index 0
    "Trouser",       # Index 1
    "Pullover",      # Index 2
    "Dress",         # Index 3
    "Coat",          # Index 4
    "Sandal",        # Index 5
    "Shirt",         # Index 6
    "Sneaker",       # Index 7
    "Bag",           # Index 8
    "Ankle boot",    # Index 9
]

# Load the FashionMNIST test dataset for making predictions
# This gives us access to test images with known labels for validation
test_data = datasets.FashionMNIST(
    root="data",           # Directory to store/load the dataset
    train=False,           # Load test split (10,000 images)
    download=True,         # Download if not already present
    transform=ToTensor(),  # Convert PIL images to PyTorch tensors
)

def load_model(model_path, device):
    """
    Load a trained neural network model from disk.
    
    Args:
        model_path (str): Path to the saved model file (.pth)
        device (torch.device): Device to load the model on (CPU/GPU)
    
    Returns:
        torch.nn.Module: Loaded model ready for inference
    """
    # Initialize model with same architecture used during training
    model = NeuralNetwork().to(device)
    
    # Load the saved model parameters (weights and biases)
    # map_location ensures model loads on the correct device
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Set model to evaluation mode for inference
    # Disables dropout and sets batch normalization to evaluation mode
    model.eval()
    
    return model

def predict(model, image_tensor, device):
    """
    Make a prediction on a single image using the trained model.
    
    Args:
        model (torch.nn.Module): Trained neural network model
        image_tensor (torch.Tensor): Input image tensor of shape (1, 1, 28, 28)
        device (torch.device): Device for computation (CPU/GPU)
    
    Returns:
        tuple: (predicted_class_name, confidence_score)
    """
    # Disable gradient computation for inference (saves memory and computation)
    with torch.no_grad():
        # Move image tensor to the same device as the model
        image_tensor = image_tensor.to(device)
        
        # Forward pass: get model predictions (logits)
        pred = model(image_tensor)
        
        # Get the predicted class by finding the index with highest score
        # pred[0] gets the first (and only) sample in the batch
        # argmax(0) returns the index of the maximum value
        predicted_class = classes[pred[0].argmax(0)]
        
        # Calculate confidence score using softmax to convert logits to probabilities
        # softmax normalizes the output to sum to 1, representing class probabilities
        # max().item() gets the highest probability as a Python float
        confidence = torch.nn.functional.softmax(pred[0], dim=0).max().item()
        
        return predicted_class, confidence

def show_prediction(image_tensor, predicted_class, actual_class, confidence):
    """
    Display the image with prediction results and save to file.
    
    Args:
        image_tensor (torch.Tensor): Input image tensor to display
        predicted_class (str): Model's predicted class name
        actual_class (str): True class name for comparison
        confidence (float): Model's confidence score (0-1)
    """
    # Convert tensor to numpy array for matplotlib visualization
    # squeeze() removes single-dimensional entries (batch and channel dimensions)
    # cpu() moves tensor to CPU if it's on GPU
    image_np = image_tensor.squeeze().cpu().numpy()
    
    # Create a figure for displaying the image and results
    plt.figure(figsize=(8, 6))
    
    # Display the grayscale image
    plt.imshow(image_np, cmap='gray')
    
    # Add title with prediction and actual results
    # f-string formatting for clean display of confidence score
    plt.title(f'Predicted: "{predicted_class}" (Confidence: {confidence:.2f})\nActual: "{actual_class}"', 
              fontsize=14, pad=20)
    
    # Remove axis ticks and labels for cleaner appearance
    plt.axis('off')
    
    # Create output directory for saving prediction images
    os.makedirs('predictions', exist_ok=True)
    
    # Save the visualization to file with high quality
    # bbox_inches='tight' removes extra whitespace
    # dpi=150 sets high resolution for clear image
    plt.savefig('predictions/prediction_result.png', bbox_inches='tight', dpi=150)
    
    # plt.show() is commented out - uncomment to display image interactively

def main():
    """
    Main function that orchestrates the prediction process.
    """
    # Determine the best available device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define path to the trained model file
    model_path = "models/fashion_mnist_model.pth"
    
    # Load the trained model from disk
    model = load_model(model_path, device)

    # Get the first test sample for demonstration
    # x: image tensor (1, 28, 28), y: class label (integer 0-9)
    x, y = test_data[0][0], test_data[0][1]
    
    # Make prediction on the sample image
    # unsqueeze(0) adds batch dimension: (1, 28, 28) -> (1, 1, 28, 28)
    predicted_class, confidence = predict(model, x.unsqueeze(0), device)
    
    # Convert numerical label to human-readable class name
    actual_class = classes[y]
    
    # Print prediction results to console
    print(f'Predicted: "{predicted_class}", Actual: "{actual_class}"')
    print(f'Confidence: {confidence:.2f}')
    
    # Visualize the prediction results and save to file
    show_prediction(x, predicted_class, actual_class, confidence)
    print("Prediction image saved to 'predictions/prediction_result.png'")

# Execute main function only if script is run directly (not imported)
if __name__ == "__main__":
    main()
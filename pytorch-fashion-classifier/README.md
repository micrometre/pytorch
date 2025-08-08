# PyTorch Fashion Classifier

This project is a PyTorch implementation of a fashion item classification model using the FashionMNIST dataset. The model is trained to classify images of clothing items into one of ten categories.

## Project Structure

```
pytorch-fashion-classifier
├── src
│   ├── __init__.py
│   ├── model.py          # Defines the neural network architecture
│   ├── train.py          # Contains the training loop
│   ├── test.py           # Evaluates the model on the test dataset
│   ├── predict.py        # Loads custom images and makes predictions
│   ├── data_loader.py    # Loads and preprocesses the FashionMNIST dataset
│   └── utils.py          # Utility functions for the project
├── test_images           # Directory for storing custom images for prediction
├── data                  # Directory for storing the downloaded dataset
├── models                # Directory for saving trained model checkpoints
├── requirements.txt      # Lists project dependencies
└── README.md             # Project documentation
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd pytorch-fashion-classifier
   ```

2. **Install the required packages:**
   It is recommended to create a virtual environment before installing the dependencies.
   ```
   pip install -r requirements.txt
   ```

3. **Download the dataset:**
   The dataset will be automatically downloaded when you run the training script.

## Usage

### Training the Model

To train the model, run the following command:
```
python src/train.py
```

### Testing the Model

To evaluate the model on the test dataset, run:
```
python src/test.py
```

### Making Predictions

To make predictions on custom images, place your images in the `test_images` directory and run:
```
python src/predict.py
```

## Creating a Custom Dataset Example

To test the model with your own images, you can create a basic custom dataset. Here's how to prepare and organize your images:

### Dataset Structure

Create the following directory structure for your custom dataset:

```
custom_dataset/
├── t-shirt/
│   ├── tshirt_001.jpg
│   ├── tshirt_002.png
│   └── ...
├── trouser/
│   ├── trouser_001.jpg
│   ├── trouser_002.png
│   └── ...
├── pullover/
│   └── ...
└── ... (other fashion categories)
```

### Image Requirements

For best results with the trained FashionMNIST model:

- **Format**: JPG, PNG, or other common image formats
- **Size**: Images will be automatically resized to 28x28 pixels
- **Color**: Grayscale preferred (color images will be converted)
- **Background**: Simple, clean backgrounds work best
- **Content**: Single fashion item per image, clearly visible


### Tips for Better Results

- **Image Quality**: Use clear, well-lit images
- **Single Items**: Ensure only one fashion item is visible per image
- **Similar Style**: Images similar to FashionMNIST style work best (simple, centered items)
- **Preprocessing**: Always apply the same preprocessing as the training data

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
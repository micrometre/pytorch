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

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
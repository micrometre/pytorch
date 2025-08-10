# Basic PyTorch Example

This project demonstrates basic usage of PyTorch, including tensor creation, operations, and a simple neural network. It serves as an introductory guide for those looking to understand the fundamentals of PyTorch.

## Project Structure

- `basic_pytorch_example.ipynb`: A Jupyter notebook that includes:
  - Importing the necessary libraries
  - Creating and manipulating tensors
  - Performing basic tensor operations
  - Defining a simple neural network
  - Executing a forward pass with sample data

- `requirements.txt`: A file that lists the dependencies required for the project. Make sure to install the necessary libraries before running the notebook.

## Installation

To set up the project, clone the repository and install the required packages using pip:

```bash
pip install -r requirements.txt
```

## Usage

1. Open the `basic_pytorch_example.ipynb` notebook in Jupyter Notebook or Jupyter Lab.
2. Run the cells sequentially to see how to create tensors, perform operations, and define a neural network.
3. To visualize training metrics, you can integrate TensorBoard by following these steps:
   - Ensure that TensorBoard is installed (it is included in `requirements.txt`).
   - Add TensorBoard logging to your training loop in the notebook.
   - Launch TensorBoard in your terminal with the command:
     ```bash
     tensorboard --logdir=runs
     ```
   - Open the provided URL in your web browser to visualize the metrics.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
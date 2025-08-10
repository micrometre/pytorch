# DeepLabV3 Semantic Segmentation - Basic Example

This repository contains a comprehensive Jupyter notebook demonstrating how to use the pre-trained DeepLabV3 model for semantic segmentation tasks using PyTorch.

## 🎯 Overview

DeepLabV3 is a state-of-the-art deep learning architecture for semantic segmentation that can classify each pixel in an image into different object categories. This example uses a pre-trained model with ResNet-101 backbone trained on the COCO dataset.

## 📋 What's Included

- **`deeplabv3_basic_example.ipynb`**: Complete tutorial notebook with step-by-step implementation
- Automatic image downloading and preprocessing
- Multiple visualization techniques
- Color-coded segmentation masks
- Results saving functionality

## 🚀 Features

- ✅ **Pre-trained Model Loading**: Uses torchvision's DeepLabV3 with ResNet-101
- ✅ **Image Preprocessing**: Proper normalization and resizing
- ✅ **GPU Support**: Automatically detects and uses CUDA if available
- ✅ **Multiple Visualizations**: Grayscale masks, colored masks, and overlays
- ✅ **Class Detection**: Identifies and labels detected object classes
- ✅ **Export Results**: Saves all outputs as image files
- ✅ **Error Handling**: Fallback options for image loading

## 📦 Requirements

### Python Packages

```bash
torch>=1.9.0
torchvision>=0.10.0
Pillow>=8.0.0
matplotlib>=3.3.0
numpy>=1.19.0
requests>=2.25.0
```

### System Requirements

- Python 3.7 or higher
- CUDA-compatible GPU (optional, but recommended for faster inference)
- Internet connection (for downloading sample images and pre-trained models)

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd deeplabv3
   ```

2. **Install dependencies**:
   ```bash
   pip install torch torchvision matplotlib pillow numpy requests
   ```

   Or using conda:
   ```bash
   conda install pytorch torchvision matplotlib pillow numpy requests -c pytorch
   ```

3. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook deeplabv3_basic_example.ipynb
   ```

## 📚 Notebook Sections

The notebook is organized into 7 main sections:

### 1. Import Required Libraries
- PyTorch and torchvision for deep learning
- PIL for image processing
- Matplotlib for visualization
- Device detection (GPU/CPU)

### 2. Load Pre-trained DeepLabV3 Model
- Downloads and loads the pre-trained model
- Sets up the model for inference
- Displays model information

### 3. Load and Preprocess Input Image
- Downloads a sample image or creates a test image
- Applies proper preprocessing (resize, normalize)
- Converts to tensor format

### 4. Perform Semantic Segmentation
- Runs inference on the preprocessed image
- Extracts segmentation predictions
- Converts to segmentation mask

### 5. Post-process and Visualize Results
- Resizes mask to match original image
- Creates side-by-side visualizations
- Shows original vs. segmented image

### 6. Apply Color Map to Segmentation Mask
- Creates colored segmentation masks
- Maps COCO classes to readable names
- Generates blended visualizations

### 7. Save Segmentation Output
- Creates output directory
- Saves all results as image files
- Provides file listing

## 🎨 Output Examples

The notebook generates several types of output:

- **Original Image**: The input image
- **Grayscale Mask**: Segmentation mask with class IDs
- **Colored Mask**: Each class represented by a different color
- **Overlay**: Original image with transparent segmentation overlay
- **Blended Result**: Artistic blend of original and colored mask

## 🏷️ Supported Classes

The model can detect 21 different classes from the COCO dataset:

- Background, Person, Bicycle, Car, Motorcycle
- Airplane, Bus, Train, Truck, Boat
- Traffic Light, Fire Hydrant, Stop Sign, Parking Meter
- Bench, Bird, Cat, Dog, Horse, Sheep, Cow

## 📊 Performance Notes

- **CPU**: Works on any system, but inference may be slow
- **GPU**: Significantly faster inference (recommended)
- **Memory**: Requires ~2GB RAM for model loading
- **Input Size**: Images are resized to 520x520 for processing

## 🔧 Customization

### Using Your Own Images

Replace the image URL in the notebook:

```python
# Replace this line in the notebook
url = "https://your-image-url-here.jpg"

# Or load a local image
img = Image.open("path/to/your/image.jpg").convert('RGB')
```

### Adjusting Output Size

Modify the preprocessing transforms:

```python
preprocess = transforms.Compose([
    transforms.Resize((your_height, your_width)),  # Change size here
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### Changing Color Palette

Customize the color mapping function:

```python
def create_color_palette(num_classes=21):
    # Your custom color scheme here
    palette = your_custom_colors
    return palette
```

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce input image size
   - Use CPU instead: `device = torch.device('cpu')`

2. **Internet Connection Issues**:
   - The notebook includes fallback image generation
   - Use local images instead of URLs

3. **Package Import Errors**:
   - Ensure all packages are installed: `pip install -r requirements.txt`
   - Check Python version compatibility

4. **Model Download Fails**:
   - Check internet connection
   - Clear PyTorch cache: `torch.hub.clear_cache()`

## 📖 References

- [DeepLabV3 Paper](https://arxiv.org/abs/1706.05587) - Original research paper
- [PyTorch Segmentation Models](https://pytorch.org/vision/stable/models.html#semantic-segmentation) - Official documentation
- [COCO Dataset](https://cocodataset.org/) - Dataset used for training

## 🤝 Contributing

Feel free to:
- Report bugs or issues
- Suggest improvements
- Add new features
- Share your results

## 📄 License

This project is open-source and available under the MIT License.

## 🙏 Acknowledgments

- PyTorch team for the excellent deep learning framework
- Research authors of DeepLabV3
- COCO dataset contributors

---

**Happy Segmenting!** 🎯

If you find this helpful, please consider starring the repository!

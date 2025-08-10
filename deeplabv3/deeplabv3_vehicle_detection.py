#!/usr/bin/env python3
"""
DeepLabV3 Semantic Segmentation - License Plate Vehicle Detection

This script demonstrates how to use a pre-trained DeepLabV3 model for detecting
vehicles with license plates (cars, motorcycles, trucks, buses) using PyTorch.

Author: Generated from Jupyter Notebook
Date: August 2025
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import requests
from io import BytesIO
import os
import argparse


def setup_device():
    """Setup and return the computing device (GPU/CPU)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return device


def load_model(device):
    """Load and setup the pre-trained DeepLabV3 model"""
    print("Loading pre-trained DeepLabV3 model...")
    model = models.segmentation.deeplabv3_resnet101(pretrained=True)
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Number of classes: {model.classifier[-1].out_channels}")
    return model


def setup_preprocessing():
    """Setup image preprocessing transforms"""
    preprocess = transforms.Compose([
        transforms.Resize((520, 520)),  # Resize to a standard size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return preprocess


def load_image(image_path):
    """Load a local image file with error handling"""
    try:
        img = Image.open(image_path).convert('RGB')
        print(f"Local image '{image_path}' loaded successfully!")
        return img
    except FileNotFoundError:
        print(f"Error: Image file '{image_path}' not found!")
        print(f"Please make sure '{image_path}' exists in the current directory.")
        raise
    except Exception as e:
        print(f"Error loading image: {e}")
        raise


def perform_segmentation(model, input_tensor, device):
    """Perform semantic segmentation inference"""
    print("Running semantic segmentation inference...")
    with torch.no_grad():
        output = model(input_tensor)
        
    # Get the segmentation predictions
    predictions = output['out']
    print(f"Output shape: {predictions.shape}")
    
    # Convert predictions to segmentation mask
    segmentation_mask = torch.argmax(predictions, dim=1).squeeze().cpu().numpy()
    print(f"Segmentation mask shape: {segmentation_mask.shape}")
    print(f"Unique classes in mask: {np.unique(segmentation_mask)}")
    
    print("Segmentation completed successfully!")
    return segmentation_mask


def create_color_palette(num_classes=21):
    """Create a color palette for segmentation visualization"""
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    palette[0] = [0, 0, 0]  # Background is black
    
    # Assign bright colors only to license plate vehicles
    vehicle_colors = {
        3: [255, 0, 0],    # Car - Red
        4: [0, 255, 0],    # Motorcycle - Green
        6: [0, 0, 255],    # Bus - Blue
        8: [255, 255, 0]   # Truck - Yellow
    }
    
    for class_id, color in vehicle_colors.items():
        if class_id < num_classes:
            palette[class_id] = color
    
    return palette


def apply_color_map(mask, palette, filter_vehicles=True):
    """Apply color mapping to segmentation mask, optionally filtering for vehicles only"""
    LICENSE_PLATE_VEHICLES = {3: 'car', 4: 'motorcycle', 6: 'bus', 8: 'truck'}
    
    height, width = mask.shape
    colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
    
    for class_id in np.unique(mask):
        if class_id < len(palette):
            if filter_vehicles:
                # Only color license plate vehicles
                if class_id in LICENSE_PLATE_VEHICLES:
                    colored_mask[mask == class_id] = palette[class_id]
                # Keep background black
                elif class_id == 0:
                    colored_mask[mask == class_id] = palette[class_id]
            else:
                colored_mask[mask == class_id] = palette[class_id]
    
    return colored_mask


def detect_vehicles(mask_resized):
    """Detect and analyze license plate vehicles in the segmentation mask"""
    LICENSE_PLATE_VEHICLES = {3: 'car', 4: 'motorcycle', 6: 'bus', 8: 'truck'}
    
    detected_vehicles = []
    for class_id in np.unique(mask_resized):
        if class_id in LICENSE_PLATE_VEHICLES:
            detected_vehicles.append((class_id, LICENSE_PLATE_VEHICLES[class_id]))
    
    print("🚗 Detected License Plate Vehicles:")
    if detected_vehicles:
        for class_id, vehicle_name in detected_vehicles:
            pixel_count = np.sum(mask_resized == class_id)
            percentage = (pixel_count / mask_resized.size) * 100
            print(f"  ✅ {vehicle_name.capitalize()} (Class {class_id}): {pixel_count:,} pixels ({percentage:.2f}% of image)")
    else:
        print("  ❌ No license plate vehicles detected in the image")
    
    print(f"\nTotal vehicle classes found: {len(detected_vehicles)}")
    print("Legend: 🔴 Car | 🟢 Motorcycle | 🔵 Bus | 🟡 Truck")
    
    return detected_vehicles


def create_visualization_image(original_image, colored_mask):
    """Create blended result with vehicles highlighted"""
    # Create the blended result
    blended = Image.blend(original_image, Image.fromarray(colored_mask), alpha=0.7)
    return blended


def save_results(blended, detected_vehicles, mask_resized):
    """Save only the vehicles highlighted result"""
    # Save the blended result (vehicles highlighted)
    blended.save("vehicles_highlighted.jpg")
    print("✓ Vehicle-highlighted result saved as 'vehicles_highlighted.jpg'")
    
    # Create a summary report
    vehicle_summary = {
        'total_pixels': mask_resized.size,
        'vehicles_found': detected_vehicles,
        'vehicle_pixel_count': sum(np.sum(mask_resized == class_id) for class_id, _ in detected_vehicles)
    }
    
    if vehicle_summary['vehicles_found']:
        vehicle_coverage = (vehicle_summary['vehicle_pixel_count'] / vehicle_summary['total_pixels']) * 100
        print(f"\n📊 Vehicle Detection Summary:")
        print(f"  - Total vehicles detected: {len(vehicle_summary['vehicles_found'])}")
        print(f"  - Vehicle coverage: {vehicle_coverage:.2f}% of image")
    else:
        print(f"\n📊 No license plate vehicles detected in this image.")


def main(image_path, save_outputs=True):
    """Main function to run the complete vehicle detection pipeline"""
    print("🚗 DeepLabV3 License Plate Vehicle Detection")
    print("=" * 50)
    
    # Setup device and model
    device = setup_device()
    model = load_model(device)
    
    # Setup preprocessing
    preprocess = setup_preprocessing()
    
    # Load and preprocess the image
    print(f"\n📷 Loading image: {image_path}")
    original_image = load_image(image_path)
    input_tensor = preprocess(original_image).unsqueeze(0).to(device)
    
    print(f"Original image size: {original_image.size}")
    print(f"Input tensor shape: {input_tensor.shape}")
    
    # Perform segmentation
    print("\n🔍 Running segmentation...")
    segmentation_mask = perform_segmentation(model, input_tensor, device)
    
    # Resize mask to match original image
    original_size = original_image.size  # (width, height)
    mask_resized = np.array(Image.fromarray(segmentation_mask.astype(np.uint8)).resize(original_size))
    
    # Create vehicle-focused visualizations
    print("\n🎨 Creating visualizations...")
    LICENSE_PLATE_VEHICLES = {3: 'car', 4: 'motorcycle', 6: 'bus', 8: 'truck'}
    
    # Create color palette and apply it
    color_palette = create_color_palette()
    colored_mask = apply_color_map(mask_resized, color_palette, filter_vehicles=True)
    
    # Create vehicle-only mask
    vehicle_only_mask = np.zeros_like(mask_resized)
    for class_id in LICENSE_PLATE_VEHICLES.keys():
        vehicle_only_mask[mask_resized == class_id] = class_id
    
    # Detect and analyze vehicles
    print("\n🚗 Analyzing detected vehicles...")
    detected_vehicles = detect_vehicles(mask_resized)
    
    # Create vehicles highlighted result
    print("\n📊 Creating vehicles highlighted image...")
    blended = create_visualization_image(original_image, colored_mask)
    
    # Save results
    if save_outputs:
        print("\n💾 Saving results...")
        save_results(blended, detected_vehicles, mask_resized)
    
    print("\n✅ Vehicle detection completed successfully!")
    return detected_vehicles, mask_resized, colored_mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepLabV3 License Plate Vehicle Detection")
    parser.add_argument("--image", "-i", default="test_images/4.jpg", help="Input image path (default: test_images/4.jpg)")
    parser.add_argument("--no-save", action="store_true", help="Don't save output files")
    
    args = parser.parse_args()
    
    try:
        main(
            image_path=args.image,
            save_outputs=not args.no_save
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        exit(1)

#!/bin/bash

echo "=== OCT Deep Learning Fingerprint System Setup ==="
echo "Setting up dependencies and environment..."

# Check if we're on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS system"
else
    echo "This script is optimized for macOS, but may work on other systems"
fi

# Install basic required packages
echo "Installing basic requirements..."
pip install "numpy<2.0" pyserial Pillow

echo ""
echo "Checking optional packages..."

# Check and install ONNX Runtime if desired
read -p "Do you want to install ONNX Runtime for deep learning features? (y/n): " install_onnx
if [[ $install_onnx =~ ^[Yy]$ ]]; then
    echo "Installing ONNX Runtime..."
    pip install onnxruntime
    echo "✓ ONNX Runtime installed"
else
    echo "Skipping ONNX Runtime - system will use basic feature extraction"
fi

# Check and install scikit-learn if desired
read -p "Do you want to install scikit-learn for advanced similarity calculations? (y/n): " install_sklearn
if [[ $install_sklearn =~ ^[Yy]$ ]]; then
    echo "Installing scikit-learn..."
    pip install scikit-learn
    echo "✓ Scikit-learn installed"
else
    echo "Skipping scikit-learn - system will use basic similarity calculation"
fi

echo ""
echo "Checking for VGG19 model..."

# Create models directory
mkdir -p models

# Check if VGG19 model exists
if [[ -f "models/vgg19-caffe2-9.onnx" ]]; then
    echo "✓ VGG19 model found"
else
    echo "VGG19 model not found"
    read -p "Do you want to download VGG19 model? (y/n): " download_model
    if [[ $download_model =~ ^[Yy]$ ]]; then
        echo "Downloading VGG19 model..."
        curl -L -o models/vgg19-caffe2-9.onnx \
          "https://github.com/onnx/models/raw/main/vision/classification/vgg/model/vgg19-caffe2-9.onnx"
        if [[ -f "models/vgg19-caffe2-9.onnx" ]]; then
            echo "✓ VGG19 model downloaded successfully"
        else
            echo "✗ Failed to download VGG19 model"
        fi
    else
        echo "Skipping VGG19 download - system will use basic feature extraction"
    fi
fi

echo ""
echo "Setup complete!"
echo ""
echo "Available files:"
echo "  - oct_deep_simple.py       : Main system (works with or without ONNX)"
echo "  - oct_deep_fingerprint.py  : Full-featured system (requires all dependencies)"
echo "  - README_DEEP_LEARNING.md  : Complete documentation"
echo ""
echo "To run the system:"
echo "  python3 oct_deep_simple.py"
echo ""
echo "System features:"

if [[ -f "models/vgg19-caffe2-9.onnx" ]] && command -v python3 -c "import onnxruntime" >/dev/null 2>&1; then
    echo "  ✓ Deep Learning mode (VGG19 + ONNX)"
else
    echo "  ✓ Basic Computer Vision mode"
fi

if command -v python3 -c "import sklearn" >/dev/null 2>&1; then
    echo "  ✓ Advanced similarity calculations"
else
    echo "  ✓ Basic similarity calculations"
fi

echo "  ✓ Arduino integration support"
echo "  ✓ B-scan folder processing"
echo "  ✓ User model training and verification"
echo "  ✓ GUI interface"
echo ""

#!/usr/bin/env zsh
set -euo pipefail

# Create models folder
mkdir -p models

# Download a couple of ImageNet-pretrained backbones from the ONNX Model Zoo
curl -L -o models/resnet50-v2-7.onnx \
  https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet50-v2-7.onnx

curl -L -o models/mobilenetv2-7.onnx \
  https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-7.onnx

echo "Downloaded ONNX models to ./models"

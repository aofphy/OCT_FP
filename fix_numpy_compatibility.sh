#!/bin/bash

echo "=== NumPy Compatibility Fix for OCT Fingerprint System ==="
echo ""

# Check current NumPy version
echo "🔍 Checking current NumPy version..."
python -c "import numpy; print(f'Current NumPy version: {numpy.__version__}')" 2>/dev/null || echo "❌ NumPy not found or import failed"

echo ""
echo "🔧 Fixing NumPy compatibility issue..."
echo "   This will downgrade NumPy to version < 2.0 for compatibility"
echo ""

read -p "Continue with NumPy downgrade? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "❌ Operation cancelled."
    exit 1
fi

echo ""
echo "📦 Uninstalling current NumPy..."
pip uninstall numpy -y

echo ""
echo "📦 Installing NumPy version 1.24.3..."
pip install "numpy==1.24.3"

echo ""
echo "📦 Reinstalling OpenCV with compatible NumPy..."
pip uninstall opencv-python -y
pip install opencv-python==4.8.0.76

echo ""
echo "📦 Reinstalling scikit-image with compatible NumPy..."
pip uninstall scikit-image -y
pip install scikit-image==0.20.0

echo ""
echo "✅ Compatibility fix completed!"
echo ""
echo "🧪 Testing imports..."
python -c "
try:
    import numpy
    print(f'✅ NumPy {numpy.__version__} imported successfully')
    import cv2
    print(f'✅ OpenCV {cv2.__version__} imported successfully')
    import skimage
    print(f'✅ scikit-image imported successfully')
    import serial
    print('✅ pyserial imported successfully')
    print('🎉 All imports successful!')
except Exception as e:
    print(f'❌ Import failed: {e}')
"

echo ""
echo "Now try running: python verifyFP_2_adruino.py"

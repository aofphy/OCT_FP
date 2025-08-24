@echo off
echo === OCT Fingerprint Arduino System - Dependency Installation ===
echo.

REM Check if pip is available
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ pip could not be found. Please install pip first.
    pause
    exit /b 1
)

echo 📦 Installing Python dependencies...

REM Install required packages
pip install opencv-python>=4.5.0
pip install numpy>=1.21.0
pip install Pillow>=8.0.0
pip install scikit-image>=0.18.0
pip install mss>=6.1.0
pip install pyserial>=3.5
pip install ttkthemes>=3.2.0

echo.
echo ✅ Installation completed!
echo.
echo Next steps:
echo 1. Connect your Arduino and upload the arduino_door_controller.ino
echo 2. Connect the relay module and magnetic door lock
echo 3. Run: python verifyFP_2_adruino.py
echo.
echo 📖 For detailed setup instructions, see ARDUINO_SETUP_GUIDE.md

pause

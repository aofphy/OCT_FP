#!/bin/bash

echo "🧪 Testing Arduino OCT Fingerprint System"
echo "======================================="

# Test environment
echo ""
echo "📋 Environment Check:"
python -c "
import sys
print(f'Python: {sys.version.split()[0]}')
try:
    import numpy; print(f'NumPy: {numpy.__version__}')
    import cv2; print(f'OpenCV: {cv2.__version__}') 
    import serial; print(f'PySerial: Available')
    import tkinter; print(f'Tkinter: Available')
    print('✅ All core modules available')
except Exception as e:
    print(f'❌ Import error: {e}')
"

echo ""
echo "🔌 Arduino Port Detection:"
python -c "
import serial.tools.list_ports
ports = serial.tools.list_ports.comports()
if ports:
    print('Available ports:')
    for port in ports:
        print(f'  - {port.device}: {port.description}')
else:
    print('No serial ports found')
"

echo ""
echo "🗄️  Database Check:"
if [ -f "fingerprint_db.sqlite" ]; then
    echo "✅ Database exists"
    python -c "
import sqlite3
conn = sqlite3.connect('fingerprint_db.sqlite')
cur = conn.cursor()
cur.execute('SELECT COUNT(*) FROM users')
users = cur.fetchone()[0]
cur.execute('SELECT COUNT(*) FROM fingerprints') 
fps = cur.fetchone()[0]
print(f'Users: {users}, Fingerprints: {fps}')
conn.close()
"
else
    echo "ℹ️  Database will be created on first run"
fi

echo ""
echo "📁 Arduino Files Check:"
files=("arduino_door_controller.ino" "ARDUINO_SETUP_GUIDE.md")
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ Missing: $file"
    fi
done

echo ""
echo "🚀 Starting Arduino OCT Fingerprint System..."
echo "   - Arduino integration enabled"
echo "   - Door control ready"
echo "   - Check admin panel for Arduino status"
echo ""

# Run the program
python verifyFP_2_adruino.py

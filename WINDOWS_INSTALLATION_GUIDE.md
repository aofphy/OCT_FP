# OCT Fingerprint System - Windows Installation Guide
# คู่มือติดตั้งระบบ OCT Fingerprint บน Windows

## ขั้นตอนการติดตั้งบน Windows

### 1. ติดตั้ง Python
- ดาวน์โหลด Python 3.8+ จาก https://python.org
- เลือก "Add Python to PATH" ตอนติดตั้ง
- ตรวจสอบการติดตั้ง: `python --version`

### 2. ติดตั้ง Package Dependencies
```cmd
# เปิด Command Prompt หรือ PowerShell
cd path\to\OCT_Fingerprint_folder

# ติดตั้ง packages พื้นฐาน
pip install -r requirements_windows.txt

# หรือติดตั้งแยกรายตัว
pip install opencv-python==4.8.1.78
pip install Pillow==10.0.1
pip install numpy==1.24.4
pip install onnxruntime==1.16.3
pip install scikit-learn==1.3.2
pip install pyserial==3.5
pip install mss==9.0.1
pip install glob2==0.7
```

### 3. ตรวจสอบการติดตั้ง ONNX Models
```cmd
# ตรวจสอบว่ามีไฟล์ model ในโฟลเดอร์ models/
dir models\*.onnx
```

### 4. เชื่อมต่อ Arduino (ถ้าต้องการ)
- ติดตั้ง Arduino IDE จาก https://arduino.cc
- ติดตั้ง CH340/CH341 Driver (สำหรับ Arduino clone)
- อัปโหลด arduino_door_controller_updated.ino ลง Arduino

### 5. รันระบบ
```cmd
python oct_deep_fixed.py
```

## ปัญหาที่พบบ่อยบน Windows

### Visual C++ Redistributable
หาก opencv-python ไม่ทำงาน:
```cmd
# ดาวน์โหลดและติดตั้ง Microsoft Visual C++ Redistributable
# https://aka.ms/vs/17/release/vc_redist.x64.exe
```

### ONNX Runtime Issues
หาก ONNX Runtime มีปัญหา:
```cmd
# ลองใช้ CPU-only version
pip uninstall onnxruntime
pip install onnxruntime
```

### Screen Capture Permission
หาก mss ไม่ทำงาน:
- ตรวจสอบ Windows Security settings
- อนุญาต Screen Recording permission

### Arduino Driver
หาก Arduino ไม่เชื่อมต่อ:
- ติดตั้ง CH340 Driver
- ตรวจสอบ Device Manager
- ลอง COM port อื่น

## การใช้งานบน Windows

### เปิดระบบ:
```cmd
# วิธีที่ 1: Command Prompt
cd C:\path\to\OCT_Fingerprint
python oct_deep_fixed.py

# วิธีที่ 2: PowerShell
cd "C:\path\to\OCT_Fingerprint"
python oct_deep_fixed.py

# วิธีที่ 3: สร้าง .bat file
echo python oct_deep_fixed.py > run_system.bat
```

### สร้าง Desktop Shortcut:
1. คลิกขวาบน Desktop → New → Shortcut
2. ใส่ path: `C:\path\to\python.exe C:\path\to\OCT_Fingerprint\oct_deep_fixed.py`
3. ตั้งชื่อ: "OCT Fingerprint System"

## Performance Tips บน Windows

### การ Optimize:
- ปิด Windows Defender Real-time protection ชั่วคราวเมื่อรัน
- ตั้งค่า Power Plan เป็น "High Performance"
- ปิด โปรแกรมที่ไม่จำเป็นเพื่อประหยัด RAM

### การ Troubleshoot:
- ตรวจสอบ Windows Event Viewer หากมี error
- รัน Command Prompt as Administrator
- ตรวจสอบ Firewall settings หาก Arduino ไม่ทำงาน

---
Created: August 25, 2025
For: Windows 10/11 (x64)

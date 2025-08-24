# OCT Fingerprint Verification System with Arduino Door Control

ระบบตรวจสอบลายนิ้วมือ OCT ที่เพิ่มความสามารถในการควบคุมประตูแม่เหล็กผ่าน Arduino เมื่อมีการยืนยันสำเร็จ

## ✨ คุณสมบัติใหม่ (New Features)

### 🔌 Arduino Integration
- **ควบคุมประตูแม่เหล็ก**: เปิดประตูอัตโนมัติเมื่อยืนยันลายนิ้วมือสำเร็จ
- **การเชื่อมต่ออัตโนมัติ**: ค้นหาและเชื่อมต่อกับ Arduino โดยอัตโนมัติ
- **การควบคุมเวลา**: ปรับเวลาเปิดประตูได้ (1-10 วินาที)
- **การทดสอบระบบ**: ทดสอบการทำงานของประตูแม่เหล็ก

### 🎛️ Control Panel
- **สถานะการเชื่อมต่อ**: แสดงสถานะ Arduino แบบ Real-time
- **การเลือกพอร์ต**: เลือกพอร์ต Arduino ด้วยตนเอง
- **การตั้งค่า**: ปรับแต่งพารามิเตอร์การทำงาน

## 📋 ความต้องการระบบ (System Requirements)

### Software
- Python 3.7+
- Arduino IDE (สำหรับอัพโหลดโค้ดไป Arduino)

### Hardware
- Arduino Uno/Nano/ESP32
- 1-Channel Relay Module (5V)
- Magnetic Door Lock (12V)
- 12V DC Power Supply
- Jumper Wires

### Python Dependencies
```
opencv-python>=4.5.0
numpy>=1.21.0
Pillow>=8.0.0
scikit-image>=0.18.0
mss>=6.1.0
pyserial>=3.5
ttkthemes>=3.2.0
```

## 🚀 การติดตั้ง (Installation)

### 1. ติดตั้ง Python Dependencies

#### Windows:
```bash
install_dependencies.bat
```

#### macOS/Linux:
```bash
chmod +x install_dependencies.sh
./install_dependencies.sh
```

#### Manual Installation:
```bash
pip install -r requirements.txt
```

### 2. ตั้งค่า Arduino
1. เปิด Arduino IDE
2. เชื่อมต่อ Arduino กับคอมพิวเตอร์
3. เปิดไฟล์ `arduino_door_controller.ino`
4. อัพโหลดโค้ดไปยัง Arduino
5. เชื่อมต่อ Relay Module ตาม [คู่มือการติดตั้ง](ARDUINO_SETUP_GUIDE.md)

### 3. เริ่มใช้งาน
```bash
python verifyFP_2_adruino.py
```

## 🔧 การใช้งาน (Usage)

### การลงทะเบียนผู้ใช้
1. ไปที่แท็บ "ลงทะเบียน"
2. สร้างผู้ใช้ใหม่
3. เลือกพื้นที่สำหรับจับภาพลายนิ้วมือ
4. จับภาพและบันทึกลายนิ้วมือ

### การยืนยันตัวตน
1. ไปที่แท็บ "ตรวจสอบ"
2. เลือกพื้นที่สำหรับการตรวจสอบ
3. เริ่มการตรวจสอบแบบ Live
4. เมื่อยืนยันสำเร็จ ประตูจะเปิดอัตโนมัติ

### การควบคุม Arduino
1. ไปที่แท็บ "ดูแลระบบ"
2. ตรวจสอบสถานะการเชื่อมต่อ Arduino
3. เลือกพอร์ตที่ถูกต้อง (หากจำเป็น)
4. ทดสอบการเปิดประตู
5. ปรับเวลาเปิดประตูตามต้องการ

## 📁 ไฟล์ในระบบ (File Structure)

```
OCT_Fingerprint/
├── verifyFP_2_adruino.py      # โปรแกรมหลักที่เพิ่ม Arduino control
├── arduino_door_controller.ino # โค้ด Arduino สำหรับควบคุม relay
├── ARDUINO_SETUP_GUIDE.md     # คู่มือการติดตั้ง Arduino
├── requirements.txt           # Python dependencies
├── install_dependencies.sh    # Script ติดตั้งสำหรับ macOS/Linux
├── install_dependencies.bat   # Script ติดตั้งสำหรับ Windows
├── fingerprint_db.sqlite     # ฐานข้อมูลลายนิ้วมือ
└── fingerprints/             # โฟลเดอร์เก็บภาพลายนิ้วมือ
```

## ⚡ การทำงานของระบบ (System Workflow)

1. **การยืนยันสำเร็จ**: ระบบพบลายนิ้วมือที่ตรงกัน (≥75% accuracy)
2. **ส่งคำสั่ง**: โปรแกรม Python ส่งคำสั่ง "OPEN_DOOR" ไป Arduino
3. **เปิด Relay**: Arduino เปิด relay เพื่อตัดไฟประตูแม่เหล็ก
4. **เปิดประตู**: ประตูแม่เหล็กเปิดได้เป็นเวลาที่กำหนด
5. **ปิดประตู**: หลังครบเวลา Arduino ส่งคำสั่ง "CLOSE_DOOR" เพื่อปิดประตู

## 🛠️ การแก้ไขปัญหา (Troubleshooting)

### Arduino ไม่เชื่อมต่อ
- ตรวจสอบสาย USB
- ตรวจสอบ Arduino Driver
- กดปุ่ม "ค้นหาพอร์ต" ใหม่
- ลองเปลี่ยนพอร์ต USB

### Relay ไม่ทำงาน
- ตรวจสอบการเชื่อมต่อสายไฟ
- ตรวจสอบแรงดันไฟฟ้า (5V)
- ตรวจสอบ LED บน Relay Module

### ประตูไม่เปิด
- ตรวจสอบ Power Supply 12V
- ตรวจสอบขั้วบวก/ลบของประตูแม่เหล็ก
- ทดสอบประตูด้วยการต่อโดยตรง

## ⚠️ ข้อควรระวัง (Safety Warnings)

- **ไฟฟ้า**: ใช้แรงดันที่ถูกต้อง (5V สำหรับ Arduino, 12V สำหรับประตู)
- **การเชื่อมต่อ**: ตรวจสอบการเชื่อมต่อก่อนเปิดไฟ
- **ความปลอดภัย**: ระบบนี้เป็นตัวอย่าง ควรมีระบบสำรองในการใช้งานจริง

## 📞 การสนับสนุน (Support)

สำหรับคำถามหรือปัญหาในการใช้งาน:
- อ่าน [คู่มือการติดตั้ง Arduino](ARDUINO_SETUP_GUIDE.md)
- ตรวจสอบ Serial Monitor ใน Arduino IDE
- ตรวจสอบ Console output ในโปรแกรม Python

## 📄 License

This project is for educational and demonstration purposes.

---
**หมายเหตุ**: ระบบนี้ออกแบบมาเพื่อการศึกษาและการสาธิต ในการใช้งานจริงควรมีการปรับปรุงด้านความปลอดภัยและความเชื่อถือได้เพิ่มเติม

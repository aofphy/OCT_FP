# Arduino Door Controller Setup Guide
## การตั้งค่า Arduino สำหรับควบคุมประตูแม่เหล็ก

### อุปกรณ์ที่จำเป็น (Hardware Requirements)
1. **Arduino Board**: Arduino Uno, Nano, หรือ ESP32
2. **Relay Module**: 1-Channel Relay Module (5V)
3. **Magnetic Door Lock**: Electric Magnetic Lock (12V)
4. **Power Supply**: 12V DC Power Adapter (สำหรับประตูแม่เหล็ก)
5. **USB Cable**: สำหรับเชื่อมต่อ Arduino กับคอมพิวเตอร์
6. **Jumper Wires**: สำหรับเชื่อมต่อสายไฟ

### การเชื่อมต่อสายไฟ (Wiring Diagram)

```
Arduino Uno          Relay Module
-----------          ------------
Pin 7        ------> IN (Input)
5V           ------> VCC
GND          ------> GND

Relay Module         12V Power Supply & Door Lock
------------         ---------------------------
COM          ------> 12V+ (Power Supply Positive)
NO           ------> Door Lock + (Positive)
                     Door Lock - ------> 12V- (Power Supply Negative)
```

### ขั้นตอนการติดตั้ง (Installation Steps)

#### 1. การอัพโหลดโค้ดไปยัง Arduino
1. เปิด Arduino IDE
2. เชื่อมต่อ Arduino กับคอมพิวเตอร์ผ่าน USB
3. เลือก Board และ Port ที่ถูกต้อง
4. เปิดไฟล์ `arduino_door_controller.ino`
5. กด Upload เพื่ออัพโหลดโค้ดไปยัง Arduino

#### 2. การเชื่อมต่อสายไฟ
1. **ระวัง**: ปิดไฟ Arduino และ Power Supply ก่อนเชื่อมต่อ
2. เชื่อมต่อสายไฟตาม Wiring Diagram ด้านบน
3. ตรวจสอบการเชื่อมต่อให้แน่ใจ
4. เปิด Power Supply 12V
5. เชื่อมต่อ Arduino กับคอมพิวเตอร์

#### 3. การติดตั้ง Python Serial Library
```bash
pip install pyserial
```

#### 4. การทดสอบระบบ
1. เปิดโปรแกรม Python (`python verifyFP_2_adruino.py`)
2. ไปที่แท็บ "ดูแลระบบ"
3. กดปุ่ม "ค้นหาพอร์ต" เพื่อหาพอร์ต Arduino
4. เลือกพอร์ตที่ถูกต้อง
5. กดปุ่ม "เชื่อมต่อ"
6. กดปุ่ม "ทดสอบเปิดประตู" เพื่อทดสอบ

### คำสั่งที่ Arduino รองรับ (Arduino Commands)

| คำสั่ง | คำอธิบาย |
|--------|----------|
| `OPEN_DOOR` | เปิดประตู (เปิด relay) |
| `CLOSE_DOOR` | ปิดประตู (ปิด relay) |
| `TEST` | ทดสอบการเชื่อมต่อ |
| `STATUS` | ตรวจสอบสถานะประตู |

### ข้อมูลเพิ่มเติม (Additional Information)

#### การตั้งค่า Relay
- **Relay Type**: Active HIGH (ส่งสัญญาณ HIGH เพื่อเปิด)
- **Relay Pin**: Digital Pin 7
- **LED Indicator**: Pin 13 (LED บอร์ดจะเปิดเมื่อประตูเปิด)

#### การปรับแต่ง
- **เวลาเปิดประตู**: สามารถปรับในโปรแกรม Python (1-10 วินาที)
- **พอร์ต Arduino**: โปรแกรมจะค้นหาอัตโนมัติ หรือเลือกด้วยตนเอง

#### การแก้ไขปัญหา (Troubleshooting)

**ปัญหา: ไม่พบพอร์ต Arduino**
- ตรวจสอบการเชื่อมต่อ USB
- ติดตั้ง Arduino Driver
- ลองเปลี่ยนสาย USB

**ปัญหา: Relay ไม่ทำงาน**
- ตรวจสอบการเชื่อมต่อสายไฟ
- ตรวจสอบแรงดันไฟฟ้า (5V สำหรับ relay)
- ตรวจสอบ LED บน relay module

**ปัญหา: ประตูไม่เปิด**
- ตรวจสอบ Power Supply 12V
- ตรวจสอบการเชื่อมต่อประตูแม่เหล็ก
- ตรวจสอบขั้วบวก/ลบของประตู

#### ข้อควรระวัง (Safety Warnings)
⚠️ **ข้อควรระวัง**:
- ใช้แรงดันไฟฟ้าที่ถูกต้อง (5V สำหรับ Arduino, 12V สำหรับประตู)
- ตรวจสอบการเชื่อมต่อก่อนเปิดไฟ
- อย่าแตะสายไฟขณะมีไฟฟ้า
- ใช้ประตูแม่เหล็กที่มีคุณภาพดี

#### Schema ของระบบ
```
Python Application
        |
        | (Serial Communication)
        v
Arduino Controller
        |
        | (Digital Signal)
        v
    Relay Module
        |
        | (12V Power Control)
        v
  Magnetic Door Lock
```

### การใช้งานในโปรแกรม
เมื่อการยืนยันลายนิ้วมือสำเร็จ ระบบจะ:
1. แสดงข้อความ "ยืนยันสำเร็จ"
2. ส่งคำสั่ง "OPEN_DOOR" ไปยัง Arduino
3. Arduino เปิด relay เพื่อตัดไฟประตูแม่เหล็ก
4. ประตูเปิดได้เป็นเวลาที่กำหนด (default: 3 วินาที)
5. Arduino ส่งคำสั่ง "CLOSE_DOOR" เพื่อปิดประตู

**หมายเหตุ**: ระบบนี้เป็นตัวอย่างเบื้องต้น ในการใช้งานจริงควรมีการป้องกันและความปลอดภัยเพิ่มเติม

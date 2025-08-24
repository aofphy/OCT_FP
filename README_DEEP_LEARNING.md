# ระบบตรวจสอบลายนิ้วมือ OCT แบบ Deep Learning

## คำอธิบายระบบ

ระบบนี้ได้รับการปรับปรุงให้ใช้เทคนิค Deep Learning แทนการใช้ HOG (Histogram of Oriented Gradients) ในการสร้างและตรวจสอบข้อมูลผู้ใช้

### คุณสมบัติหลัก

1. **การสร้างผู้ใช้แบบ Deep Learning**
   - เลือกโฟลเดอร์ B-scan ที่ต้องการ
   - ใช้โมเดล VGG19 pre-trained สำหรับสกัดฟีเจอร์
   - เทรนโมเดลเฉพาะสำหรับแต่ละผู้ใช้
   - บันทึกโมเดลสำหรับการตรวจสอบในอนาคต

2. **การตรวจสอบผู้ใช้**
   - อัปโหลดภาพ OCT สำหรับตรวจสอบ
   - เปรียบเทียบฟีเจอร์ด้วย Cosine Similarity
   - ปรับค่าเกณฑ์การตรวจสอบได้ (threshold)
   - เปิดประตูอัตโนมัติผ่าน Arduino (ถ้าเชื่อมต่อ)

3. **ระบบจัดการ**
   - ดูสถานะระบบและโมเดล
   - จัดการผู้ใช้ (ลบ, เทรนใหม่)
   - ดูประวัติการตรวจสอบ

### โครงสร้างไฟล์

```
oct_deep_simple.py          # ไฟล์หลักของระบบ
oct_deep_fingerprint.py     # เวอร์ชันที่ต้องใช้ advanced packages
requirements_deep.txt       # รายการ package ที่ต้องการ
models/
  vgg19-caffe2-9.onnx      # โมเดล VGG19 (ต้องดาวน์โหลดแยก)
user_models/               # โฟลเดอร์เก็บโมเดลผู้ใช้ (สร้างอัตโนมัติ)
oct_fingerprint_deep.sqlite # ฐานข้อมูล (สร้างอัตโนมัติ)
```

## การติดตั้งและใช้งาน

### ขั้นตอนที่ 1: เตรียมสภาพแวดล้อม

```bash
# ดาวน์โหลดโมเดล VGG19 (ถ้าต้องการใช้ Deep Learning เต็มรูปแบบ)
mkdir -p models
curl -L -o models/vgg19-caffe2-9.onnx \
  https://github.com/onnx/models/raw/main/vision/classification/vgg/model/vgg19-caffe2-9.onnx
```

### ขั้นตอนที่ 2: ติดตั้ง Packages (ถ้าต้องการ)

```bash
# สำหรับฟีเจอร์ Deep Learning เต็มรูปแบบ
pip install onnxruntime scikit-learn

# สำหรับฟีเจอร์พื้นฐาน (ระบบจะทำงานโดยไม่ต้องมี package เพิ่มเติม)
# จะใช้ OpenCV และ NumPy ที่มีอยู่แล้ว
```

### ขั้นตอนที่ 3: แก้ปัญหา NumPy version conflict

```bash
# ถ้าเจอปัญหา NumPy version conflict
pip install "numpy<2.0"

# หรือ
conda install "numpy<2.0"
```

### ขั้นตอนที่ 4: รันระบบ

```bash
python3 oct_deep_simple.py
```

## การใช้งาน

### การสร้างผู้ใช้ใหม่

1. ไปที่แท็บ "ลงทะเบียนผู้ใช้"
2. ใส่ชื่อผู้ใช้
3. กดปุ่ม "เลือกโฟลเดอร์" เพื่อเลือกโฟลเดอร์ B-scan
4. กดปุ่ม "สร้างผู้ใช้และเทรนโมเดล"
5. รอให้ระบบประมวลผลเสร็จ

### การตรวจสอบผู้ใช้

1. ไปที่แท็บ "ตรวจสอบผู้ใช้"
2. เลือกผู้ใช้ที่ต้องการตรวจสอบ
3. กดปุ่ม "เลือกภาพ OCT" เพื่ออัปโหลดภาพ
4. ปรับค่าเกณฑ์การตรวจสอบ (ถ้าต้องการ)
5. กดปุ่ม "ตรวจสอบ"

### การจัดการระบบ

1. ไปที่แท็บ "จัดการระบบ"
2. ดูสถานะระบบและจำนวนผู้ใช้
3. เลือกผู้ใช้เพื่อลบหรือเทรนใหม่
4. ทดสอบการเชื่อมต่อ Arduino

## โครงสร้างโฟลเดอร์ B-scan

ระบบรองรับโครงสร้างโฟลเดอร์แบบต่างๆ:

```
B-scan-folder/
  ├── image1.png
  ├── image2.jpg
  ├── subfolder1/
  │   ├── image3.png
  │   └── image4.jpg
  └── subfolder2/
      └── image5.png
```

ไฟล์ที่รองรับ: .png, .jpg, .jpeg, .bmp, .tiff, .tif

## การทำงานของระบบ

### โหมด Basic (ไม่ต้องใช้ ONNX)
- ใช้ฟีเจอร์ Computer Vision แบบดั้งเดิม
- Histogram features
- Local Binary Pattern (LBP)
- Edge features
- Statistical features

### โหมด Deep Learning (ต้องมี ONNX และ VGG19)
- ใช้โมเดล VGG19 สกัดฟีเจอร์
- ฟีเจอร์ที่ได้มีคุณภาพสูงกว่า
- ความแม่นยำในการตรวจสอบดีกว่า

## การตั้งค่า Arduino

ใช้ไฟล์ `arduino_door_controller.ino` สำหรับควบคุมประตู:

```cpp
// Code จาก arduino_door_controller.ino
void setup() {
  Serial.begin(9600);
  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, LOW);
}

void loop() {
  if (Serial.available()) {
    String command = Serial.readString();
    command.trim();
    
    if (command == "TEST") {
      Serial.println("OK");
    } else if (command == "OPEN") {
      digitalWrite(RELAY_PIN, HIGH);
      Serial.println("OPENED");
      delay(3000);  // เปิด 3 วินาที
      digitalWrite(RELAY_PIN, LOW);
    }
  }
}
```

## การแก้ปัญหา

### ปัญหาที่พบบ่อย

1. **NumPy version conflict**
   - แก้ไข: `pip install "numpy<2.0"`

2. **ไม่พบโมเดล VGG19**
   - ระบบจะใช้โหมด Basic แทน
   - ดาวน์โหลดโมเดลตามขั้นตอนที่ 1

3. **Arduino ไม่เชื่อมต่อ**
   - ตรวจสอบการต่อสาย USB
   - ติดตั้ง driver CH340/CH341 (ถ้าจำเป็น)
   - อัปโหลดโค้ด Arduino ที่ถูกต้อง

4. **ไม่พบภาพ B-scan**
   - ตรวจสอบโครงสร้างโฟลเดอร์
   - ตรวจสอบนามสกุลไฟล์ที่รองรับ

## การพัฒนาเพิ่มเติม

### เพิ่มโมเดล Deep Learning อื่นๆ
- ResNet50
- MobileNetV2
- EfficientNet

### เพิ่มฟีเจอร์อื่นๆ
- Real-time capture จากกล้อง
- Multi-user verification
- Web interface
- API endpoints

## ข้อมูลเพิ่มเติม

- โครงการนี้พัฒนาต่อจากระบบ OCT fingerprint เดิม
- รองรับการทำงานทั้งแบบออนไลน์และออฟไลน์
- สามารถขยายได้ตามความต้องการ

## ไฟล์ที่สำคัญ

- `oct_deep_simple.py`: ระบบหลัก (ใช้งานง่าย)
- `oct_deep_fingerprint.py`: ระบบเต็มรูปแบบ (ต้องการ dependencies เพิ่ม)
- `arduino_door_controller.ino`: โค้ด Arduino
- `requirements_deep.txt`: รายการ Python packages

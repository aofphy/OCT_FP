#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCT Deep Learning System Demo
ตัวอย่างการใช้งานระบบตรวจสอบลายนิ้วมือ OCT แบบ Deep Learning
"""

import os
import sys
import time
from datetime import datetime

def print_banner():
    """Print system banner"""
    print("=" * 60)
    print("🔬 OCT Deep Learning Fingerprint System Demo")
    print("ระบบตรวจสอบลายนิ้วมือ OCT แบบ Deep Learning")
    print("=" * 60)
    print()

def check_system_requirements():
    """Check system requirements and capabilities"""
    print("🔍 ตรวจสอบความพร้อมของระบบ...")
    print()
    
    requirements = {
        'Basic Requirements': [
            ('Python 3', sys.version_info >= (3, 6)),
            ('OpenCV', check_package('cv2')),
            ('NumPy', check_package('numpy')),
            ('Pillow', check_package('PIL')),
            ('PySerial', check_package('serial')),
            ('Tkinter', check_package('tkinter'))
        ],
        'Advanced Features': [
            ('ONNX Runtime', check_package('onnxruntime')),
            ('Scikit-learn', check_package('sklearn')),
            ('VGG19 Model', os.path.exists('models/vgg19-caffe2-9.onnx'))
        ]
    }
    
    for category, reqs in requirements.items():
        print(f"📋 {category}:")
        for name, available in reqs:
            status = "✅" if available else "❌"
            print(f"   {status} {name}")
        print()
    
    # Determine system mode
    has_onnx = check_package('onnxruntime')
    has_vgg19 = os.path.exists('models/vgg19-caffe2-9.onnx')
    has_sklearn = check_package('sklearn')
    
    if has_onnx and has_vgg19:
        mode = "🚀 Deep Learning Mode (Full Features)"
        mode_desc = "ใช้โมเดล VGG19 สำหรับสกัดฟีเจอร์ + Advanced similarity"
    elif has_sklearn:
        mode = "🔧 Computer Vision Mode (Advanced)"
        mode_desc = "ใช้ Computer Vision + Advanced similarity calculation"
    else:
        mode = "⚙️  Basic Mode"
        mode_desc = "ใช้ Computer Vision พื้นฐาน + Basic similarity"
    
    print(f"🎯 ระบบจะทำงานในโหมด: {mode}")
    print(f"   {mode_desc}")
    print()
    
    return mode

def check_package(package_name):
    """Check if a package is available"""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def show_file_structure():
    """Show important files in the system"""
    print("📁 โครงสร้างไฟล์สำคัญ:")
    print()
    
    important_files = [
        ('oct_deep_simple.py', 'ระบบหลัก (แนะนำ)'),
        ('oct_deep_fingerprint.py', 'ระบบเต็มรูปแบบ'),
        ('README_DEEP_LEARNING.md', 'คู่มือการใช้งาน'),
        ('SYSTEM_COMPARISON.md', 'เปรียบเทียบระบบเดิม/ใหม่'),
        ('setup_deep_system.sh', 'สคริปต์ติดตั้ง'),
        ('models/', 'โฟลเดอร์โมเดล Deep Learning'),
        ('user_models/', 'โฟลเดอร์โมเดลผู้ใช้'),
        ('oct_fingerprint_deep.sqlite', 'ฐานข้อมูลระบบใหม่')
    ]
    
    for filename, description in important_files:
        exists = os.path.exists(filename)
        status = "✅" if exists else "📝"
        print(f"   {status} {filename:<30} - {description}")
    
    print()

def show_usage_examples():
    """Show usage examples"""
    print("💡 ตัวอย่างการใช้งาน:")
    print()
    
    examples = [
        {
            'title': '1. เริ่มใช้งานระบบ',
            'command': 'python3 oct_deep_simple.py',
            'description': 'เปิดโปรแกรมหลัก (GUI)'
        },
        {
            'title': '2. ติดตั้ง Deep Learning (ถ้าต้องการ)',
            'command': 'pip install onnxruntime scikit-learn',
            'description': 'ติดตั้งแพ็กเกจเพิ่มเติม'
        },
        {
            'title': '3. ดาวน์โหลดโมเดล VGG19',
            'command': 'curl -L -o models/vgg19-caffe2-9.onnx https://github.com/onnx/models/raw/main/vision/classification/vgg/model/vgg19-caffe2-9.onnx',
            'description': 'ดาวน์โหลดโมเดล Deep Learning'
        },
        {
            'title': '4. รันสคริปต์ติดตั้งอัตโนมัติ',
            'command': './setup_deep_system.sh',
            'description': 'ติดตั้งทุกอย่างแบบอัตโนมัติ'
        }
    ]
    
    for example in examples:
        print(f"📌 {example['title']}:")
        print(f"   $ {example['command']}")
        print(f"   {example['description']}")
        print()

def show_workflow():
    """Show typical workflow"""
    print("🔄 ขั้นตอนการใช้งานปกติ:")
    print()
    
    steps = [
        "1️⃣  เปิดโปรแกรม oct_deep_simple.py",
        "2️⃣  ไปที่แท็บ 'ลงทะเบียนผู้ใช้'",
        "3️⃣  ใส่ชื่อผู้ใช้ใหม่",
        "4️⃣  เลือกโฟลเดอร์ B-scan (ที่มีภาพ OCT หลายๆ ภาพ)",
        "5️⃣  กดปุ่ม 'สร้างผู้ใช้และเทรนโมเดล'",
        "6️⃣  รอให้ระบบเทรนโมเดลเสร็จ",
        "7️⃣  ไปที่แท็บ 'ตรวจสอบผู้ใช้'",
        "8️⃣  เลือกผู้ใช้ที่ต้องการตรวจสอบ",
        "9️⃣  อัปโหลดภาพ OCT สำหรับตรวจสอบ",
        "🔟 กดปุ่ม 'ตรวจสอบ' และดูผล"
    ]
    
    for step in steps:
        print(f"   {step}")
    
    print()

def show_troubleshooting():
    """Show common issues and solutions"""
    print("🛠️  การแก้ปัญหาที่พบบ่อย:")
    print()
    
    issues = [
        {
            'problem': 'NumPy version conflict',
            'solution': 'pip install "numpy<2.0"',
            'description': 'แก้ปัญหาความขัดแย้งเวอร์ชัน NumPy'
        },
        {
            'problem': 'Missing serial module', 
            'solution': 'pip install pyserial',
            'description': 'ติดตั้งไลบรารีสำหรับ Arduino'
        },
        {
            'problem': 'ONNX Runtime ไม่ทำงาน',
            'solution': 'ระบบจะใช้ Basic mode อัตโนมัติ',
            'description': 'ยังใช้งานได้ปกติ'
        },
        {
            'problem': 'ไม่พบภาพ B-scan',
            'solution': 'ตรวจสอบโครงสร้างโฟลเดอร์และนามสกุลไฟล์',
            'description': 'รองรับ: .png, .jpg, .jpeg, .bmp, .tiff'
        }
    ]
    
    for issue in issues:
        print(f"❗ ปัญหา: {issue['problem']}")
        print(f"   💡 แก้ไข: {issue['solution']}")
        print(f"   📝 หมายเหตุ: {issue['description']}")
        print()

def interactive_demo():
    """Interactive demo menu"""
    while True:
        print("🎮 เมนูการสาธิต:")
        print("   1. ตรวจสอบระบบและความพร้อม")
        print("   2. แสดงโครงสร้างไฟล์")
        print("   3. ตัวอย่างคำสั่งการใช้งาน")
        print("   4. ขั้นตอนการใช้งาน")
        print("   5. การแก้ปัญหา")
        print("   6. เริ่มใช้งานระบบจริง")
        print("   0. ออกจากโปรแกรม")
        print()
        
        try:
            choice = input("👉 เลือกตัวเลข (0-6): ").strip()
            print()
            
            if choice == '0':
                print("👋 ขอบคุณที่ใช้งาน OCT Deep Learning System!")
                break
            elif choice == '1':
                check_system_requirements()
            elif choice == '2':
                show_file_structure()
            elif choice == '3':
                show_usage_examples()
            elif choice == '4':
                show_workflow()
            elif choice == '5':
                show_troubleshooting()
            elif choice == '6':
                print("🚀 กำลังเริ่มระบบหลัก...")
                print("   กำลังรัน: python3 oct_deep_simple.py")
                print()
                try:
                    os.system('python3 oct_deep_simple.py')
                except KeyboardInterrupt:
                    print("\n⏹️  หยุดการทำงานของระบบหลัก")
                    print()
            else:
                print("❌ กรุณาเลือกตัวเลข 0-6 เท่านั้น")
                
        except KeyboardInterrupt:
            print("\n\n👋 ขอบคุณที่ใช้งาน!")
            break
        
        input("🔄 กด Enter เพื่อกลับไปยังเมนูหลัก...")
        print("\n" + "="*60)

def main():
    """Main demo function"""
    print_banner()
    
    # Quick system check
    mode = check_system_requirements()
    
    print("📖 ข้อมูลเพิ่มเติม:")
    print("   📄 อ่านคู่มือเต็ม: README_DEEP_LEARNING.md")
    print("   🔄 เปรียบเทียบระบบ: SYSTEM_COMPARISON.md")
    print("   🎉 สรุปความสำเร็จ: SUCCESS_DEEP_LEARNING_SYSTEM.md")
    print()
    
    # Ask user if they want interactive demo
    try:
        response = input("🤖 ต้องการใช้งานเมนูสาธิตแบบโต้ตอบหรือไม่? (y/n): ").strip().lower()
        print()
        
        if response in ['y', 'yes', 'ใช่', '1']:
            interactive_demo()
        else:
            print("🚀 สำหรับการใช้งานจริง ให้รันคำสั่ง:")
            print("   python3 oct_deep_simple.py")
            print()
            print("📚 หรือดูคู่มือเพิ่มเติมในไฟล์ README_DEEP_LEARNING.md")
            
    except KeyboardInterrupt:
        print("\n\n👋 ขอบคุณที่ใช้งาน OCT Deep Learning System!")

if __name__ == "__main__":
    main()

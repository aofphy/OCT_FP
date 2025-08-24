# -*- coding: utf-8 -*-
"""
Simple OCT Deep Learning Fingerprint System
ระบบตรวจสอบลายนิ้วมือ OCT แบบ Deep Learning (Simple Version)
Using basic feature extraction and similarity matching
"""

import os
import shutil
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Toplevel
import cv2
import numpy as np
from PIL import Image, ImageTk
import sqlite3
from datetime import datetime
import warnings
import time
import serial
import serial.tools.list_ports
import threading
import glob
import json
import pickle

# Try to import advanced packages, fallback to basic implementation if not available
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
    print("ONNX Runtime available")
except ImportError:
    ONNX_AVAILABLE = False
    print("ONNX Runtime not available - using basic feature extraction")

try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
    print("Scikit-learn available")
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Scikit-learn not available - using basic similarity calculation")

class BasicFeatureExtractor:
    """Basic feature extractor using OpenCV and traditional computer vision methods"""
    
    def __init__(self):
        self.model_loaded = True
        print("Basic feature extractor initialized")
    
    def extract_features(self, image):
        """Extract features using traditional computer vision methods"""
        if image is None:
            return None
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Resize to standard size
        gray = cv2.resize(gray, (224, 224), interpolation=cv2.INTER_LANCZOS4)
        
        # Extract multiple types of features
        features = []
        
        # 1. Histogram features
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        hist = hist.flatten()
        hist = hist / (hist.sum() + 1e-7)  # Normalize
        features.extend(hist)
        
        # 2. Texture features using Local Binary Pattern (simplified)
        lbp_features = self._extract_lbp_features(gray)
        features.extend(lbp_features)
        
        # 3. Edge features
        edges = cv2.Canny(gray, 50, 150)
        edge_hist = cv2.calcHist([edges], [0], None, [32], [0, 256])
        edge_hist = edge_hist.flatten()
        edge_hist = edge_hist / (edge_hist.sum() + 1e-7)
        features.extend(edge_hist)
        
        # 4. Statistical features
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        features.extend([mean_val/255.0, std_val/255.0])
        
        # Convert to numpy array and normalize
        features = np.array(features, dtype=np.float32)
        
        # L2 normalization
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return features
    
    def _extract_lbp_features(self, gray):
        """Extract simplified Local Binary Pattern features"""
        features = []
        
        # Simple LBP-like features
        h, w = gray.shape
        for i in range(1, h-1, 4):  # Sample every 4 pixels
            for j in range(1, w-1, 4):
                center = gray[i, j]
                pattern = 0
                
                # 8-neighborhood
                neighbors = [
                    gray[i-1, j-1], gray[i-1, j], gray[i-1, j+1],
                    gray[i, j+1], gray[i+1, j+1], gray[i+1, j],
                    gray[i+1, j-1], gray[i, j-1]
                ]
                
                for idx, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        pattern |= (1 << idx)
                
                features.append(pattern / 255.0)  # Normalize to [0, 1]
        
        return features[:128]  # Limit to 128 features

class ONNXFeatureExtractor(BasicFeatureExtractor):
    """ONNX-based deep learning feature extractor"""
    
    def __init__(self, model_path="models/vgg19-caffe2-9.onnx"):
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.output_name = None
        self.model_loaded = False
        
        # ImageNet normalization parameters
        self.mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        
        if ONNX_AVAILABLE:
            self.load_model()
        else:
            print("ONNX not available, falling back to basic features")
            super().__init__()
    
    def load_model(self):
        """Load ONNX model"""
        if not os.path.exists(self.model_path):
            print(f"Model file not found: {self.model_path}")
            print("Using basic feature extraction instead")
            super().__init__()
            return False
        
        try:
            self.session = ort.InferenceSession(self.model_path)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            print(f"ONNX model loaded successfully: {self.model_path}")
            self.model_loaded = True
            return True
            
        except Exception as e:
            print(f"Error loading ONNX model: {e}")
            print("Using basic feature extraction instead")
            super().__init__()
            return False
    
    def preprocess_image(self, image, target_size=(224, 224)):
        """Preprocess image for VGG19 input"""
        if image is None:
            return None
        
        # Convert to RGB if grayscale
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)
        
        # Convert to float and normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Transpose to CHW format and add batch dimension
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        
        # Apply ImageNet normalization
        image = (image - self.mean) / self.std
        
        return image
    
    def extract_features(self, image):
        """Extract deep learning features from image"""
        # If ONNX model is not loaded, use basic features
        if not self.model_loaded or not ONNX_AVAILABLE:
            return super().extract_features(image)
        
        # Preprocess image
        processed_image = self.preprocess_image(image)
        if processed_image is None:
            return None
        
        try:
            # Run inference
            outputs = self.session.run([self.output_name], {self.input_name: processed_image})
            features = outputs[0]
            
            # Flatten and normalize features
            features = features.flatten()
            
            # L2 normalization
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm
            
            return features
            
        except Exception as e:
            print(f"Error extracting ONNX features: {e}")
            print("Falling back to basic features")
            return super().extract_features(image)

class OCTUserManager:
    """Manage OCT user training and verification"""
    
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor
        self.models_dir = "user_models"
        os.makedirs(self.models_dir, exist_ok=True)
    
    def load_bscan_images(self, folder_path):
        """Load all B-scan images from folder"""
        if not os.path.exists(folder_path):
            return []
        
        # Supported image extensions
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif']
        images = []
        
        for ext in extensions:
            pattern = os.path.join(folder_path, '**', ext)
            files = glob.glob(pattern, recursive=True)
            for file_path in files:
                try:
                    img = cv2.imread(file_path)
                    if img is not None:
                        images.append((file_path, img))
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        
        return images
    
    def extract_features_from_folder(self, folder_path, user_id, progress_callback=None):
        """Extract features from all images in B-scan folder"""
        images = self.load_bscan_images(folder_path)
        
        if not images:
            return None, []
        
        features_list = []
        file_paths = []
        
        total_images = len(images)
        for i, (file_path, img) in enumerate(images):
            if progress_callback:
                progress_callback(i + 1, total_images, f"Processing {os.path.basename(file_path)}")
            
            features = self.feature_extractor.extract_features(img)
            if features is not None:
                features_list.append(features)
                file_paths.append(file_path)
        
        if not features_list:
            return None, []
        
        # Convert to numpy array
        features_array = np.array(features_list)
        
        return features_array, file_paths
    
    def train_user_model(self, user_id, user_name, bscan_folder, progress_callback=None):
        """Train a user-specific model from B-scan folder"""
        
        # Extract features from user's B-scan folder
        user_features, user_files = self.extract_features_from_folder(
            bscan_folder, user_id, progress_callback
        )
        
        if user_features is None or len(user_features) == 0:
            raise Exception("No valid B-scan images found or feature extraction failed")
        
        # Create user model data
        user_model_data = {
            'user_id': user_id,
            'user_name': user_name,
            'features': user_features,
            'file_paths': user_files,
            'mean_features': np.mean(user_features, axis=0),
            'std_features': np.std(user_features, axis=0),
            'num_samples': len(user_features),
            'created_at': datetime.now().isoformat(),
            'bscan_folder': bscan_folder,
            'feature_extractor_type': 'ONNX' if ONNX_AVAILABLE and self.feature_extractor.model_loaded else 'Basic'
        }
        
        # Save user model
        model_filename = f"user_{user_id}_{user_name.replace(' ', '_')}_model.pkl"
        model_path = os.path.join(self.models_dir, model_filename)
        
        with open(model_path, 'wb') as f:
            pickle.dump(user_model_data, f)
        
        print(f"User model saved: {model_path}")
        print(f"Features extracted: {len(user_features)} samples")
        
        return model_path, user_model_data
    
    def load_user_model(self, user_id):
        """Load user model by ID"""
        pattern = os.path.join(self.models_dir, f"user_{user_id}_*_model.pkl")
        model_files = glob.glob(pattern)
        
        if not model_files:
            return None
        
        # Load the most recent model file
        model_file = max(model_files, key=os.path.getctime)
        
        try:
            with open(model_file, 'rb') as f:
                user_model_data = pickle.load(f)
            return user_model_data
        except Exception as e:
            print(f"Error loading user model: {e}")
            return None
    
    def calculate_similarity(self, features1, features2):
        """Calculate similarity between two feature vectors"""
        if SKLEARN_AVAILABLE:
            # Use scikit-learn's cosine similarity
            similarity = cosine_similarity([features1], [features2])[0][0]
        else:
            # Basic cosine similarity calculation
            dot_product = np.dot(features1, features2)
            norm1 = np.linalg.norm(features1)
            norm2 = np.linalg.norm(features2)
            
            if norm1 == 0 or norm2 == 0:
                similarity = 0.0
            else:
                similarity = dot_product / (norm1 * norm2)
        
        return float(similarity)
    
    def verify_user(self, test_image, user_id, threshold=0.7):
        """Verify user identity using trained model"""
        
        # Load user model
        user_model = self.load_user_model(user_id)
        if user_model is None:
            return False, 0.0, "User model not found"
        
        # Extract features from test image
        test_features = self.feature_extractor.extract_features(test_image)
        if test_features is None:
            return False, 0.0, "Feature extraction failed"
        
        # Calculate similarity with user's training features
        user_features = user_model['features']
        similarities = []
        
        for train_features in user_features:
            similarity = self.calculate_similarity(test_features, train_features)
            similarities.append(similarity)
        
        # Get best similarity
        max_similarity = np.max(similarities)
        avg_similarity = np.mean(similarities)
        
        # Verification decision
        is_verified = max_similarity >= threshold
        
        result_info = f"Max: {max_similarity:.3f}, Avg: {avg_similarity:.3f}, Samples: {len(similarities)}"
        
        return is_verified, max_similarity, result_info

class OCTDeepFingerprintSystem:
    """Main OCT Deep Learning Fingerprint System"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("ระบบตรวจสอบลายนิ้วมือ OCT - Deep Learning (v6.0 - Simple)")
        self.root.geometry("1400x900")
        
        # Initialize feature extractor
        if ONNX_AVAILABLE:
            self.feature_extractor = ONNXFeatureExtractor()
        else:
            self.feature_extractor = BasicFeatureExtractor()
        
        # Initialize user manager
        self.user_manager = OCTUserManager(self.feature_extractor)
        
        # Database setup
        self.db_path = 'oct_fingerprint_deep.sqlite'
        self.init_db()
        
        # Current state
        self.current_user = None
        self.current_scan = None
        
        # Arduino components
        self.arduino_port = None
        self.arduino_serial = None
        self.arduino_connected = False
        self.relay_open_duration = 3
        self.arduino_status_var = tk.StringVar(value="Arduino: ไม่เชื่อมต่อ")
        
        # UI styling
        self.setup_styles()
        
        # Create UI
        self.create_ui()
        
        # Initialize Arduino after delay
        self.root.after(1000, self.init_arduino)
    
    def init_db(self):
        """Initialize database tables"""
        conn = self.get_db_connection()
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                bscan_folder TEXT,
                model_path TEXT,
                num_training_samples INTEGER DEFAULT 0,
                feature_extractor_type TEXT DEFAULT 'Basic'
            )
        ''')
        
        # Verification logs
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS verification_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                verified BOOLEAN NOT NULL,
                similarity_score REAL NOT NULL,
                verification_time TEXT NOT NULL,
                details TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_db_connection(self):
        """Get a new database connection for the current thread"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn
    
    def setup_styles(self):
        """Setup UI styles"""
        self.style = ttk.Style(self.root)
        
        try:
            bg = self.style.lookup('TFrame', 'background')
            self.style.configure('Normal.TLabel', foreground='black', background=bg, font=('Arial', 12))
            self.style.configure('Success.TLabel', foreground='green', background='#d4ffcc', font=('Arial', 14, 'bold'))
            self.style.configure('Failure.TLabel', foreground='#cc0000', background='#ffcccc', font=('Arial', 14, 'bold'))
            self.style.configure('Info.TLabel', foreground='blue', background='#cceeff', font=('Arial', 12))
        except tk.TclError:
            print("Using fallback styling")
    
    def create_ui(self):
        """Create main user interface"""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Registration tab
        self.reg_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.reg_frame, text="ลงทะเบียนผู้ใช้")
        
        # Verification tab
        self.verify_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.verify_frame, text="ตรวจสอบผู้ใช้")
        
        # Admin tab
        self.admin_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.admin_frame, text="จัดการระบบ")
        
        # Create tab contents
        self.create_registration_panel()
        self.create_verification_panel()
        self.create_admin_panel()
        self.create_status_bar()
        
        # Initialize displays
        self.refresh_all_data()
    
    def create_registration_panel(self):
        """Create user registration panel"""
        # User creation section
        user_frame = ttk.LabelFrame(self.reg_frame, text="สร้างผู้ใช้ใหม่", padding=10)
        user_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(user_frame, text="ชื่อผู้ใช้:").grid(row=0, column=0, sticky='w', padx=5)
        self.username_entry = ttk.Entry(user_frame, width=30)
        self.username_entry.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(user_frame, text="โฟลเดอร์ B-scan:").grid(row=1, column=0, sticky='w', padx=5)
        self.bscan_folder_var = tk.StringVar(value="ยังไม่ได้เลือกโฟลเดอร์")
        ttk.Label(user_frame, textvariable=self.bscan_folder_var, width=50).grid(row=1, column=1, sticky='w', padx=5)
        ttk.Button(user_frame, text="เลือกโฟลเดอร์", command=self.select_bscan_folder).grid(row=1, column=2, padx=5)
        
        ttk.Button(user_frame, text="สร้างผู้ใช้และเทรนโมเดล", 
                  command=self.create_user_with_training).grid(row=2, column=1, pady=10)
        
        # Progress section
        progress_frame = ttk.LabelFrame(self.reg_frame, text="ความคืบหน้า", padding=10)
        progress_frame.pack(fill='x', padx=5, pady=5)
        
        self.training_progress = ttk.Progressbar(progress_frame, length=400, mode='determinate')
        self.training_progress.pack(pady=5)
        
        self.training_status = ttk.Label(progress_frame, text="พร้อม", style='Info.TLabel')
        self.training_status.pack(pady=5)
        
        # User list section
        list_frame = ttk.LabelFrame(self.reg_frame, text="ผู้ใช้ในระบบ", padding=10)
        list_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # User listbox with scrollbar
        list_container = ttk.Frame(list_frame)
        list_container.pack(fill='both', expand=True)
        
        self.user_listbox = tk.Listbox(list_container, height=8)
        scrollbar = ttk.Scrollbar(list_container, orient='vertical', command=self.user_listbox.yview)
        self.user_listbox.configure(yscrollcommand=scrollbar.set)
        
        self.user_listbox.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        self.user_listbox.bind('<<ListboxSelect>>', self.on_user_select)
        
        # User details section
        details_frame = ttk.LabelFrame(self.reg_frame, text="รายละเอียดผู้ใช้", padding=10)
        details_frame.pack(fill='x', padx=5, pady=5)
        
        self.user_details_var = tk.StringVar(value="เลือกผู้ใช้เพื่อดูรายละเอียด")
        ttk.Label(details_frame, textvariable=self.user_details_var, wraplength=600).pack()
    
    def create_verification_panel(self):
        """Create verification panel"""
        # Control section
        control_frame = ttk.LabelFrame(self.verify_frame, text="ตรวจสอบผู้ใช้", padding=10)
        control_frame.pack(fill='x', padx=5, pady=5)
        
        # User selection
        selection_frame = ttk.Frame(control_frame)
        selection_frame.pack(fill='x', pady=5)
        
        ttk.Label(selection_frame, text="เลือกผู้ใช้:").pack(side='left')
        self.verify_user_var = tk.StringVar()
        self.verify_user_combo = ttk.Combobox(selection_frame, textvariable=self.verify_user_var, 
                                             state='readonly', width=30)
        self.verify_user_combo.pack(side='left', padx=10)
        
        # Image upload
        upload_frame = ttk.Frame(control_frame)
        upload_frame.pack(fill='x', pady=10)
        
        ttk.Button(upload_frame, text="เลือกภาพ OCT", 
                  command=self.upload_verification_image).pack(side='left')
        
        self.verify_image_path_var = tk.StringVar(value="ยังไม่ได้เลือกภาพ")
        ttk.Label(upload_frame, textvariable=self.verify_image_path_var).pack(side='left', padx=10)
        
        # Verification threshold
        threshold_frame = ttk.Frame(control_frame)
        threshold_frame.pack(fill='x', pady=5)
        
        ttk.Label(threshold_frame, text="เกณฑ์การตรวจสอบ:").pack(side='left')
        self.threshold_var = tk.DoubleVar(value=0.7)
        threshold_scale = ttk.Scale(threshold_frame, from_=0.5, to=0.9, variable=self.threshold_var, 
                                   orient='horizontal', length=200)
        threshold_scale.pack(side='left', padx=10)
        
        self.threshold_label = ttk.Label(threshold_frame, text="0.70")
        self.threshold_label.pack(side='left', padx=5)
        
        threshold_scale.configure(command=self.update_threshold_label)
        
        # Verify button
        ttk.Button(control_frame, text="ตรวจสอบ", command=self.verify_identity).pack(pady=10)
        
        # Results section
        result_frame = ttk.LabelFrame(self.verify_frame, text="ผลการตรวจสอบ", padding=10)
        result_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.verification_result_var = tk.StringVar(value="ยังไม่ได้ตรวจสอบ")
        self.verification_result_label = ttk.Label(result_frame, textvariable=self.verification_result_var, 
                                                  style='Normal.TLabel', font=('Arial', 16, 'bold'))
        self.verification_result_label.pack(pady=10)
        
        self.verification_details_var = tk.StringVar(value="")
        ttk.Label(result_frame, textvariable=self.verification_details_var, wraplength=600).pack()
        
        # Image display
        image_frame = ttk.Frame(result_frame)
        image_frame.pack(fill='both', expand=True, pady=10)
        
        self.verify_image_canvas = tk.Canvas(image_frame, bg='white', height=200)
        self.verify_image_canvas.pack(fill='both', expand=True)
    
    def create_admin_panel(self):
        """Create admin panel"""
        # System status
        status_frame = ttk.LabelFrame(self.admin_frame, text="สถานะระบบ", padding=10)
        status_frame.pack(fill='x', padx=5, pady=5)
        
        # Feature extractor info
        extractor_type = "ONNX Deep Learning" if (ONNX_AVAILABLE and self.feature_extractor.model_loaded) else "Basic Computer Vision"
        ttk.Label(status_frame, text=f"Feature Extractor: {extractor_type}").pack(anchor='w')
        
        if ONNX_AVAILABLE and hasattr(self.feature_extractor, 'model_path'):
            ttk.Label(status_frame, text=f"Model Path: {self.feature_extractor.model_path}").pack(anchor='w')
        
        # Package availability
        ttk.Label(status_frame, text=f"ONNX Runtime: {'Available' if ONNX_AVAILABLE else 'Not Available'}").pack(anchor='w')
        ttk.Label(status_frame, text=f"Scikit-learn: {'Available' if SKLEARN_AVAILABLE else 'Not Available'}").pack(anchor='w')
        
        # Arduino status
        ttk.Label(status_frame, textvariable=self.arduino_status_var).pack(anchor='w')
        
        # Database info
        conn = self.get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        conn.close()
        ttk.Label(status_frame, text=f"จำนวนผู้ใช้: {user_count}").pack(anchor='w')
        
        # Controls
        control_frame = ttk.LabelFrame(self.admin_frame, text="การควบคุม", padding=10)
        control_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(control_frame, text="รีเฟรชข้อมูล", command=self.refresh_all_data).pack(side='left', padx=5)
        ttk.Button(control_frame, text="ทดสอบ Arduino", command=self.test_arduino).pack(side='left', padx=5)
        ttk.Button(control_frame, text="ล้างล็อก", command=self.clear_logs).pack(side='left', padx=5)
        
        # User management
        user_mgmt_frame = ttk.LabelFrame(self.admin_frame, text="จัดการผู้ใช้", padding=10)
        user_mgmt_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # User tree view
        columns = ('ID', 'Name', 'Created', 'Samples', 'Type')
        self.users_tree = ttk.Treeview(user_mgmt_frame, columns=columns, show='headings', height=8)
        
        for col in columns:
            self.users_tree.heading(col, text=col)
            self.users_tree.column(col, width=120)
        
        tree_scroll = ttk.Scrollbar(user_mgmt_frame, orient='vertical', command=self.users_tree.yview)
        self.users_tree.configure(yscrollcommand=tree_scroll.set)
        
        self.users_tree.pack(side='left', fill='both', expand=True)
        tree_scroll.pack(side='right', fill='y')
        
        # User actions
        actions_frame = ttk.Frame(user_mgmt_frame)
        actions_frame.pack(fill='x', pady=5)
        
        ttk.Button(actions_frame, text="ลบผู้ใช้", command=self.delete_selected_user).pack(side='left', padx=5)
        ttk.Button(actions_frame, text="เทรนใหม่", command=self.retrain_selected_user).pack(side='left', padx=5)
    
    def create_status_bar(self):
        """Create status bar"""
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill='x', side='bottom')
        
        self.status_var = tk.StringVar(value=f"ระบบพร้อม - {self.get_feature_extractor_name()}")
        ttk.Label(status_frame, textvariable=self.status_var).pack(side='left', padx=5)
        
        ttk.Label(status_frame, textvariable=self.arduino_status_var).pack(side='right', padx=5)
    
    def get_feature_extractor_name(self):
        """Get current feature extractor name"""
        if ONNX_AVAILABLE and hasattr(self.feature_extractor, 'model_loaded') and self.feature_extractor.model_loaded:
            return "Deep Learning Mode"
        else:
            return "Basic Mode"
    
    # ========== Core Functions ==========
    
    def select_bscan_folder(self):
        """Select B-scan folder for user training"""
        folder = filedialog.askdirectory(title="เลือกโฟลเดอร์ B-scan")
        if folder:
            self.bscan_folder_var.set(folder)
            # Count images in folder
            images = self.user_manager.load_bscan_images(folder)
            self.training_status.config(text=f"พบภาพ B-scan: {len(images)} ไฟล์")
    
    def create_user_with_training(self):
        """Create user and train model from B-scan folder"""
        name = self.username_entry.get().strip()
        bscan_folder = self.bscan_folder_var.get()
        
        if not name:
            messagebox.showwarning("ข้อผิดพลาด", "กรุณาใส่ชื่อผู้ใช้")
            return
        
        if bscan_folder == "ยังไม่ได้เลือกโฟลเดอร์":
            messagebox.showwarning("ข้อผิดพลาด", "กรุณาเลือกโฟลเดอร์ B-scan")
            return
        
        # Run training in thread
        thread = threading.Thread(target=self._train_user_thread, args=(name, bscan_folder))
        thread.daemon = True
        thread.start()
    
    def _train_user_thread(self, name, bscan_folder):
        """Training thread"""
        try:
            # Create user in database first
            cursor = self.conn.cursor()
            
            # Check if user already exists
            cursor.execute("SELECT id FROM users WHERE name = ?", (name,))
            existing = cursor.fetchone()
            if existing:
                self.root.after(0, lambda: messagebox.showerror("ข้อผิดพลาด", f"ผู้ใช้ '{name}' มีอยู่แล้ว"))
                return
            
            feature_type = self.get_feature_extractor_name()
            cursor.execute("INSERT INTO users (name, created_at, bscan_folder, feature_extractor_type) VALUES (?, ?, ?, ?)",
                         (name, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), bscan_folder, feature_type))
            user_id = cursor.lastrowid
            self.conn.commit()
            
            # Update UI
            self.root.after(0, lambda: self.training_progress.configure(value=0))
            self.root.after(0, lambda: self.training_status.config(text="เริ่มการเทรนโมเดล..."))
            
            def progress_callback(current, total, message):
                progress = (current / total) * 100
                self.root.after(0, lambda: self.training_progress.configure(value=progress))
                self.root.after(0, lambda: self.training_status.config(text=f"{message} ({current}/{total})"))
            
            # Train model
            model_path, model_data = self.user_manager.train_user_model(
                user_id, name, bscan_folder, progress_callback
            )
            
            # Update database with model info
            cursor.execute("UPDATE users SET model_path = ?, num_training_samples = ? WHERE id = ?",
                         (model_path, model_data['num_samples'], user_id))
            self.conn.commit()
            
            # Update UI
            self.root.after(0, lambda: self.training_progress.configure(value=100))
            self.root.after(0, lambda: self.training_status.config(
                text=f"เทรนเสร็จแล้ว! ใช้ {model_data['num_samples']} ภาพ"))
            self.root.after(0, self.refresh_all_data)
            self.root.after(0, lambda: messagebox.showinfo(
                "สำเร็จ", f"สร้างผู้ใช้ '{name}' และเทรนโมเดลเสร็จแล้ว\nใช้ข้อมูล: {model_data['num_samples']} ภาพ"))
            
            # Clear form
            self.root.after(0, lambda: self.username_entry.delete(0, tk.END))
            self.root.after(0, lambda: self.bscan_folder_var.set("ยังไม่ได้เลือกโฟลเดอร์"))
            
        except Exception as e:
            error_msg = f"เกิดข้อผิดพลาด: {str(e)}"
            self.root.after(0, lambda: self.training_status.config(text=error_msg))
            self.root.after(0, lambda: messagebox.showerror("ข้อผิดพลาด", error_msg))
            
            # Clean up on error
            try:
                if 'user_id' in locals():
                    cursor = self.conn.cursor()
                    cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
                    self.conn.commit()
            except:
                pass
    
    def upload_verification_image(self):
        """Upload image for verification"""
        file_path = filedialog.askopenfilename(
            title="เลือกภาพ OCT",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif")]
        )
        
        if file_path:
            self.verify_image_path_var.set(os.path.basename(file_path))
            self.current_verify_image_path = file_path
            
            # Display image
            try:
                img = cv2.imread(file_path)
                if img is not None:
                    self.display_image_on_canvas(img, self.verify_image_canvas)
            except Exception as e:
                print(f"Error displaying image: {e}")
    
    def update_threshold_label(self, value):
        """Update threshold label"""
        self.threshold_label.config(text=f"{float(value):.2f}")
    
    def verify_identity(self):
        """Verify user identity"""
        if not hasattr(self, 'current_verify_image_path'):
            messagebox.showwarning("ข้อผิดพลาด", "กรุณาเลือกภาพสำหรับตรวจสอบ")
            return
        
        selected_user = self.verify_user_var.get()
        if not selected_user:
            messagebox.showwarning("ข้อผิดพลาด", "กรุณาเลือกผู้ใช้ที่ต้องการตรวจสอบ")
            return
        
        try:
            # Get user ID from selection
            user_id = int(selected_user.split(':')[0])
            threshold = self.threshold_var.get()
            
            # Load test image
            test_image = cv2.imread(self.current_verify_image_path)
            if test_image is None:
                messagebox.showerror("ข้อผิดพลาด", "ไม่สามารถโหลดภาพได้")
                return
            
            # Perform verification
            is_verified, similarity_score, details = self.user_manager.verify_user(
                test_image, user_id, threshold=threshold
            )
            
            # Update UI
            if is_verified:
                self.verification_result_var.set("✓ ตรวจสอบผ่าน")
                self.verification_result_label.config(style='Success.TLabel')
                
                # Open Arduino door if connected
                if self.arduino_connected:
                    self.open_door()
                    
            else:
                self.verification_result_var.set("✗ ตรวจสอบไม่ผ่าน")
                self.verification_result_label.config(style='Failure.TLabel')
            
            self.verification_details_var.set(
                f"คะแนนความคล้าย: {similarity_score:.3f}\nเกณฑ์: {threshold:.2f}\n{details}"
            )
            
            # Log verification
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO verification_logs (user_id, verified, similarity_score, verification_time, details) VALUES (?, ?, ?, ?, ?)",
                (user_id, is_verified, similarity_score, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), details)
            )
            self.conn.commit()
            
        except Exception as e:
            messagebox.showerror("ข้อผิดพลาด", f"เกิดข้อผิดพลาดในการตรวจสอบ: {str(e)}")
    
    def display_image_on_canvas(self, img, canvas, max_size=(300, 200)):
        """Display image on canvas"""
        try:
            if len(img.shape) == 3:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = img
            
            # Resize image to fit canvas
            h, w = img_rgb.shape[:2]
            max_w, max_h = max_size
            
            scale = min(max_w/w, max_h/h, 1.0)
            new_w, new_h = int(w * scale), int(h * scale)
            
            img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            # Convert to PIL and display
            pil_img = Image.fromarray(img_resized)
            photo = ImageTk.PhotoImage(pil_img)
            
            canvas.delete("all")
            canvas_w = canvas.winfo_width()
            canvas_h = canvas.winfo_height()
            
            if canvas_w > 1 and canvas_h > 1:
                x = max(0, (canvas_w - new_w) // 2)
                y = max(0, (canvas_h - new_h) // 2)
                canvas.create_image(x, y, anchor='nw', image=photo)
                canvas.image = photo  # Keep reference
                
        except Exception as e:
            print(f"Error displaying image: {e}")
    
    # ========== UI Management Functions ==========
    
    def refresh_all_data(self):
        """Refresh all UI data"""
        self.refresh_user_list()
        self.refresh_verify_combo()
        self.refresh_admin_tree()
    
    def refresh_user_list(self):
        """Refresh user listbox"""
        self.user_listbox.delete(0, tk.END)
        
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, name, num_training_samples FROM users ORDER BY name")
        users = cursor.fetchall()
        
        for user_id, name, samples in users:
            samples = samples or 0
            self.user_listbox.insert(tk.END, f"{user_id}: {name} ({samples} ภาพ)")
    
    def refresh_verify_combo(self):
        """Refresh verification combobox"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, name FROM users WHERE model_path IS NOT NULL ORDER BY name")
        users = cursor.fetchall()
        
        values = [f"{user_id}: {name}" for user_id, name in users]
        self.verify_user_combo['values'] = values
    
    def refresh_admin_tree(self):
        """Refresh admin tree view"""
        # Clear existing items
        for item in self.users_tree.get_children():
            self.users_tree.delete(item)
        
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, name, created_at, num_training_samples, feature_extractor_type FROM users ORDER BY id")
        users = cursor.fetchall()
        
        for user_id, name, created_at, samples, extractor_type in users:
            samples = samples or 0
            extractor_type = extractor_type or "Basic"
            self.users_tree.insert('', tk.END, values=(user_id, name, created_at, samples, extractor_type))
    
    def on_user_select(self, event):
        """Handle user selection"""
        try:
            selection = self.user_listbox.curselection()
            if not selection:
                return
            
            selected_text = self.user_listbox.get(selection[0])
            user_id = int(selected_text.split(':')[0])
            self.current_user = user_id
            
            # Update user details
            cursor = self.conn.cursor()
            cursor.execute("SELECT name, created_at, bscan_folder, model_path, num_training_samples, feature_extractor_type FROM users WHERE id = ?", (user_id,))
            user_data = cursor.fetchone()
            
            if user_data:
                name, created_at, bscan_folder, model_path, samples, extractor_type = user_data
                samples = samples or 0
                extractor_type = extractor_type or "Basic"
                
                details = f"ชื่อ: {name}\n"
                details += f"สร้างเมื่อ: {created_at}\n"
                details += f"โฟลเดอร์ B-scan: {bscan_folder or 'ไม่ระบุ'}\n"
                details += f"จำนวนภาพเทรน: {samples}\n"
                details += f"ประเภทฟีเจอร์: {extractor_type}\n"
                details += f"สถานะโมเดล: {'พร้อมใช้งาน' if model_path else 'ยังไม่เทรน'}"
                
                self.user_details_var.set(details)
            
        except Exception as e:
            print(f"Error in user selection: {e}")
    
    def delete_selected_user(self):
        """Delete selected user"""
        selection = self.users_tree.selection()
        if not selection:
            messagebox.showwarning("ข้อผิดพลาด", "กรุณาเลือกผู้ใช้ที่ต้องการลบ")
            return
        
        item = self.users_tree.item(selection[0])
        user_id = item['values'][0]
        user_name = item['values'][1]
        
        if messagebox.askyesno("ยืนยัน", f"ต้องการลบผู้ใช้ '{user_name}' หรือไม่?"):
            try:
                cursor = self.conn.cursor()
                cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
                self.conn.commit()
                
                # Delete model file if exists
                pattern = os.path.join(self.user_manager.models_dir, f"user_{user_id}_*_model.pkl")
                for model_file in glob.glob(pattern):
                    try:
                        os.remove(model_file)
                        print(f"Deleted model file: {model_file}")
                    except Exception as e:
                        print(f"Error deleting model file: {e}")
                
                self.refresh_all_data()
                messagebox.showinfo("สำเร็จ", f"ลบผู้ใช้ '{user_name}' แล้ว")
                
            except Exception as e:
                messagebox.showerror("ข้อผิดพลาด", f"ไม่สามารถลบผู้ใช้ได้: {str(e)}")
    
    def retrain_selected_user(self):
        """Retrain model for selected user"""
        selection = self.users_tree.selection()
        if not selection:
            messagebox.showwarning("ข้อผิดพลาด", "กรุณาเลือกผู้ใช้ที่ต้องการเทรนใหม่")
            return
        
        item = self.users_tree.item(selection[0])
        user_id = item['values'][0]
        user_name = item['values'][1]
        
        # Get user's bscan folder
        cursor = self.conn.cursor()
        cursor.execute("SELECT bscan_folder FROM users WHERE id = ?", (user_id,))
        result = cursor.fetchone()
        
        if not result or not result[0]:
            messagebox.showerror("ข้อผิดพลาด", "ไม่พบโฟลเดอร์ B-scan สำหรับผู้ใช้นี้")
            return
        
        bscan_folder = result[0]
        
        if messagebox.askyesno("ยืนยัน", f"ต้องการเทรนโมเดลใหม่สำหรับ '{user_name}' หรือไม่?"):
            # Run retraining in thread
            thread = threading.Thread(target=self._retrain_user_thread, args=(user_id, user_name, bscan_folder))
            thread.daemon = True
            thread.start()
    
    def _retrain_user_thread(self, user_id, user_name, bscan_folder):
        """Retraining thread"""
        try:
            self.root.after(0, lambda: self.training_progress.configure(value=0))
            self.root.after(0, lambda: self.training_status.config(text=f"เทรนใหม่สำหรับ {user_name}..."))
            
            def progress_callback(current, total, message):
                progress = (current / total) * 100
                self.root.after(0, lambda: self.training_progress.configure(value=progress))
                self.root.after(0, lambda: self.training_status.config(text=f"{message} ({current}/{total})"))
            
            # Delete old model
            pattern = os.path.join(self.user_manager.models_dir, f"user_{user_id}_*_model.pkl")
            for model_file in glob.glob(pattern):
                try:
                    os.remove(model_file)
                    print(f"Deleted old model: {model_file}")
                except Exception as e:
                    print(f"Error deleting old model: {e}")
            
            # Train new model
            model_path, model_data = self.user_manager.train_user_model(
                user_id, user_name, bscan_folder, progress_callback
            )
            
            # Update database
            cursor = self.conn.cursor()
            feature_type = self.get_feature_extractor_name()
            cursor.execute("UPDATE users SET model_path = ?, num_training_samples = ?, feature_extractor_type = ? WHERE id = ?",
                         (model_path, model_data['num_samples'], feature_type, user_id))
            self.conn.commit()
            
            # Update UI
            self.root.after(0, lambda: self.training_progress.configure(value=100))
            self.root.after(0, lambda: self.training_status.config(
                text=f"เทรนใหม่เสร็จแล้ว! ใช้ {model_data['num_samples']} ภาพ"))
            self.root.after(0, self.refresh_all_data)
            self.root.after(0, lambda: messagebox.showinfo(
                "สำเร็จ", f"เทรนโมเดลใหม่สำหรับ '{user_name}' เสร็จแล้ว\nใช้ข้อมูล: {model_data['num_samples']} ภาพ"))
            
        except Exception as e:
            error_msg = f"เกิดข้อผิดพลาดในการเทรนใหม่: {str(e)}"
            self.root.after(0, lambda: self.training_status.config(text=error_msg))
            self.root.after(0, lambda: messagebox.showerror("ข้อผิดพลาด", error_msg))
    
    def clear_logs(self):
        """Clear verification logs"""
        if messagebox.askyesno("ยืนยัน", "ต้องการล้างล็อกการตรวจสอบทั้งหมดหรือไม่?"):
            try:
                cursor = self.conn.cursor()
                cursor.execute("DELETE FROM verification_logs")
                self.conn.commit()
                messagebox.showinfo("สำเร็จ", "ล้างล็อกเสร็จแล้ว")
            except Exception as e:
                messagebox.showerror("ข้อผิดพลาด", f"ไม่สามารถล้างล็อกได้: {str(e)}")
    
    # ========== Arduino Functions ==========
    
    def init_arduino(self):
        """Initialize Arduino connection"""
        try:
            self.connect_arduino()
        except Exception as e:
            print(f"Arduino initialization error: {e}")
    
    def find_arduino_ports(self):
        """Find available Arduino ports"""
        arduino_ports = []
        try:
            ports = serial.tools.list_ports.comports()
            for port in ports:
                if any(keyword in port.description.lower() for keyword in ['arduino', 'ch340', 'ch341', 'cp210', 'ftdi']):
                    arduino_ports.append(port.device)
        except Exception as e:
            print(f"Error finding Arduino ports: {e}")
        return arduino_ports
    
    def connect_arduino(self):
        """Connect to Arduino"""
        arduino_ports = self.find_arduino_ports()
        
        if not arduino_ports:
            self.arduino_status_var.set("Arduino: ไม่พบพอร์ต")
            return False
        
        for port in arduino_ports:
            try:
                self.arduino_serial = serial.Serial(port, 9600, timeout=2)
                time.sleep(2)  # Wait for Arduino reset
                
                # Test communication
                self.arduino_serial.write(b'TEST\n')
                response = self.arduino_serial.readline().decode('utf-8').strip()
                
                if 'OK' in response:
                    self.arduino_port = port
                    self.arduino_connected = True
                    self.arduino_status_var.set(f"Arduino: เชื่อมต่อ ({port})")
                    print(f"Arduino connected on {port}")
                    return True
                else:
                    self.arduino_serial.close()
                    
            except Exception as e:
                print(f"Error connecting to {port}: {e}")
                if hasattr(self, 'arduino_serial') and self.arduino_serial:
                    try:
                        self.arduino_serial.close()
                    except:
                        pass
        
        self.arduino_status_var.set("Arduino: เชื่อมต่อไม่ได้")
        return False
    
    def open_door(self):
        """Open door via Arduino"""
        if not self.arduino_connected or not self.arduino_serial:
            print("Arduino not connected")
            return False
        
        try:
            self.arduino_serial.write(b'OPEN\n')
            response = self.arduino_serial.readline().decode('utf-8').strip()
            
            if 'OPENED' in response:
                print(f"Door opened for {self.relay_open_duration} seconds")
                self.status_var.set(f"ประตูเปิด {self.relay_open_duration} วินาที")
                return True
            else:
                print(f"Unexpected Arduino response: {response}")
                return False
                
        except Exception as e:
            print(f"Error opening door: {e}")
            self.arduino_connected = False
            self.arduino_status_var.set("Arduino: การเชื่อมต่อหลุด")
            return False
    
    def test_arduino(self):
        """Test Arduino connection"""
        if self.arduino_connected:
            success = self.open_door()
            if success:
                messagebox.showinfo("ทดสอบ Arduino", "ทดสอบสำเร็จ! ประตูเปิดแล้ว")
            else:
                messagebox.showerror("ทดสอบ Arduino", "ทดสอบไม่สำเร็จ")
        else:
            messagebox.showwarning("ทดสอบ Arduino", "Arduino ไม่ได้เชื่อมต่อ")
    
    def close_app(self):
        """Close application"""
        try:
            if hasattr(self, 'arduino_serial') and self.arduino_serial:
                self.arduino_serial.close()
            self.conn.close()
        except Exception as e:
            print(f"Error during cleanup: {e}")
        
        self.root.destroy()

def main():
    """Main function"""
    print("=== OCT Deep Learning Fingerprint System ===")
    print("Starting system...")
    
    # Check VGG19 model availability
    model_path = "models/vgg19-caffe2-9.onnx"
    if os.path.exists(model_path):
        print(f"VGG19 model found: {model_path}")
    else:
        print(f"VGG19 model not found: {model_path}")
        print("System will use basic feature extraction")
    
    print(f"ONNX Runtime: {'Available' if ONNX_AVAILABLE else 'Not Available'}")
    print(f"Scikit-learn: {'Available' if SKLEARN_AVAILABLE else 'Not Available'}")
    
    # Create and run application
    root = tk.Tk()
    app = OCTDeepFingerprintSystem(root)
    
    # Handle window close
    root.protocol("WM_DELETE_WINDOW", app.close_app)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("Application interrupted")
        app.close_app()

if __name__ == "__main__":
    main()

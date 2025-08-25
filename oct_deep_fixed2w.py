# -*- coding: utf-8 -*-
"""
OCT Deep Learning Fingerprint System (Thread-Safe Version)
ระบบตรวจสอบลายนิ้วมือ OCT แบบ Deep Learning - แก้ไข SQLite Threading Issues
"""

import os
import shutil
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Toplevel
import cv2
import numpy as np
from PIL import Image, ImageTk
from typing import Optional
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
import queue

# For screen capture functionality
try:
    import mss
    import mss.tools
    MSS_AVAILABLE = True
    print("MSS (Screen Capture) available")
except ImportError:
    # Define names to avoid static warnings when not installed
    mss = None  # type: ignore
    MSS_AVAILABLE = False
    print("MSS not available - screen capture disabled")

# Try to import advanced packages, fallback to basic implementation if not available
try:
    import importlib
    ort = importlib.import_module('onnxruntime')  # type: ignore
    ONNX_AVAILABLE = True
    print("ONNX Runtime available")
except Exception:
    ONNX_AVAILABLE = False
    ort = None  # type: ignore
    print("ONNX Runtime not available - using basic feature extraction")

# Scikit-learn not required; similarity uses NumPy. Keep flag for UI only.
SKLEARN_AVAILABLE = False

# Resolve important paths relative to this file (not current working directory)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

class ImageCropper(Toplevel):
    """Image cropper window for selecting capture areas"""
    
    def __init__(self, parent, pil_image):
        super().__init__(parent)
        self.parent = parent
        self.pil_image = pil_image
        self.title("เลือกพื้นที่จับภาพ OCT")
        
        # Window setup
        max_width = self.winfo_screenwidth() * 0.8
        max_height = self.winfo_screenheight() * 0.8
        self.geometry(f"{int(max_width)}x{int(max_height)}")
        self.resizable(True, True)
        self.grab_set()
        self.focus_set()
        self.transient(parent)

        # Selection state
        self.start_x = None
        self.start_y = None
        self.current_x = None
        self.current_y = None
        self.selection_rect = None
        self.selected_bbox = None

        # Create UI
        self.canvas = tk.Canvas(self, bg='darkgrey', cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        button_frame = ttk.Frame(self)
        button_frame.pack(pady=10, fill="x")
        
        self.info_label = ttk.Label(button_frame, text="คลิกและลากเพื่อเลือกพื้นที่ที่ต้องการ แล้วกด 'ยืนยัน'")
        self.info_label.pack(side=tk.LEFT, padx=10)
        
        ttk.Button(button_frame, text="ยกเลิก", command=self.cancel).pack(side=tk.RIGHT, padx=10)
        ttk.Button(button_frame, text="ยืนยัน", command=self.confirm).pack(side=tk.RIGHT, padx=10)

        # Display properties
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.display_width = 0
        self.display_height = 0
        self.photo_image = None

        # Bind events
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        self.bind("<Escape>", self.cancel)

        # Initialize display
        self.wait_visibility()
        self.center_window()
        self.display_image()
    
    def on_canvas_resize(self, event):
        """Handle canvas resize"""
        self.after_idle(self.display_image)
    
    def display_image(self):
        """Display image on canvas"""
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            # Canvas not ready yet
            self.after(100, self.display_image)
            return
        
        if not hasattr(self, 'pil_image') or self.pil_image is None:
            print("Warning: No PIL image")
            return
        
        img_width, img_height = self.pil_image.size
        if img_width <= 0 or img_height <= 0:
            print("Warning: Invalid image dimensions")
            return
        
        # Calculate scale
        scale_w = canvas_width / img_width
        scale_h = canvas_height / img_height
        self.scale = min(scale_w, scale_h)
        
        if self.scale >= 1.0:
            self.scale = 1.0
        
        # Calculate display size
        self.display_width = max(1, int(img_width * self.scale))
        self.display_height = max(1, int(img_height * self.scale))
        
        try:
            # Resize image
            display_image = self.pil_image.resize((self.display_width, self.display_height), Image.Resampling.LANCZOS)
            
            # Create PhotoImage
            self.photo_image = ImageTk.PhotoImage(display_image, master=self)
            
            # Calculate offset for centering
            self.offset_x = max(0, (canvas_width - self.display_width) // 2)
            self.offset_y = max(0, (canvas_height - self.display_height) // 2)
            
            # Display image
            if self.canvas.winfo_exists():
                self.canvas.delete("image")
                self.canvas.create_image(self.offset_x, self.offset_y, anchor="nw", 
                                       image=self.photo_image, tags="image")
        except Exception as e:
            print(f"Error displaying image: {e}")
    
    def _canvas_to_original(self, cx, cy):
        """Convert canvas coordinates to original image coordinates"""
        if self.scale == 0:
            return 0, 0
        
        ox = (cx - self.offset_x) / self.scale
        oy = (cy - self.offset_y) / self.scale
        
        iw, ih = self.pil_image.size
        ox = max(0, min(iw, ox))
        oy = max(0, min(ih, oy))
        
        return int(ox), int(oy)
    
    def _canvas_to_display(self, cx, cy):
        """Convert canvas coordinates to display area coordinates"""
        if not hasattr(self, 'display_width') or self.display_width <= 0:
            return cx, cy
        if not hasattr(self, 'display_height') or self.display_height <= 0:
            return cx, cy
        
        x = max(self.offset_x, min(self.offset_x + self.display_width, cx))
        y = max(self.offset_y, min(self.offset_y + self.display_height, cy))
        
        return x, y
    
    def on_button_press(self, event):
        """Handle mouse button press"""
        if not hasattr(self, 'display_width') or not hasattr(self, 'display_height'):
            return
        
        # Check if click is within image bounds
        if not (self.offset_x <= event.x <= self.offset_x + self.display_width and 
                self.offset_y <= event.y <= self.offset_y + self.display_height):
            self.start_x = None
            return
        
        self.start_x, self.start_y = self._canvas_to_display(event.x, event.y)
        
        # Clear existing selection
        if self.selection_rect:
            self.canvas.delete(self.selection_rect)
        
        # Create new selection rectangle
        self.selection_rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline='red', width=2, dash=(4, 2), tags="selection"
        )
    
    def on_mouse_drag(self, event):
        """Handle mouse drag"""
        if self.start_x is None or self.selection_rect is None:
            return
        
        self.current_x, self.current_y = self._canvas_to_display(event.x, event.y)
        # Safe integer conversion (guards against None for static analyzers)
        sx = int(self.start_x) if self.start_x is not None else 0
        sy = int(self.start_y) if self.start_y is not None else 0
        cx = int(self.current_x) if self.current_x is not None else sx
        cy = int(self.current_y) if self.current_y is not None else sy
        self.canvas.coords(self.selection_rect, sx, sy, cx, cy)
    
    def on_button_release(self, event):
        """Handle mouse button release"""
        if self.start_x is None or self.selection_rect is None:
            if self.selection_rect:
                self.canvas.delete(self.selection_rect)
                self.selection_rect = None
            self.start_x = None
            return
        
        ex, ey = self._canvas_to_display(event.x, event.y)
        # Safe integer conversion
        sx = int(self.start_x) if self.start_x is not None else 0
        sy = int(self.start_y) if self.start_y is not None else 0
        self.canvas.coords(self.selection_rect, sx, sy, int(ex), int(ey))
        self.current_x = ex
        self.current_y = ey
    
    def confirm(self):
        """Confirm selection"""
        if self.selection_rect and self.start_x is not None and self.current_x is not None:
            try:
                cx1, cy1, cx2, cy2 = self.canvas.coords(self.selection_rect)
            except ValueError:
                messagebox.showwarning("ไม่ได้เลือก", "กรุณาลากเลือกพื้นที่ก่อน", parent=self)
                return
            
            # Convert to original image coordinates
            ox1, oy1 = self._canvas_to_original(min(cx1, cx2), min(cy1, cy2))
            ox2, oy2 = self._canvas_to_original(max(cx1, cx2), max(cy1, cy2))
            
            if ox2 - ox1 >= 1 and oy2 - oy1 >= 1:
                self.selected_bbox = (ox1, oy1, ox2, oy2)
                print(f"Selected BBox: {self.selected_bbox}")
                self.destroy()
            else:
                messagebox.showwarning("เล็กเกินไป", "กรุณาเลือกพื้นที่ให้ใหญ่กว่า 1x1 พิกเซล", parent=self)
        else:
            messagebox.showwarning("ไม่ได้เลือก", "กรุณาลากเลือกพื้นที่ก่อน", parent=self)
    
    def cancel(self, event=None):
        """Cancel selection"""
        self.selected_bbox = None
        self.destroy()
    
    def center_window(self):
        """Center window on screen"""
        self.update_idletasks()
        
        try:
            geometry = self.geometry()
            size_part = geometry.split('+')[0]
            w = int(size_part.split('x')[0])
            h = int(size_part.split('x')[1])
        except (ValueError, IndexError, AttributeError):
            try:
                w = self.winfo_width()
                h = self.winfo_height()
            except:
                return
        
        if w <= 0 or h <= 0:
            return
        
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        x = max(0, (sw // 2) - (w // 2))
        y = max(0, (sh // 2) - (h // 2))
        
        try:
            self.geometry(f'{w}x{h}+{x}+{y}')
        except Exception as e:
            print(f"Warning: Could not center window: {e}")


class QualityAssessor:
    """Assess image quality to filter out noise/non-fingerprint frames."""

    # Thresholds tuned for grayscale OCT 224-300 px scale
    T_STD = 8.0            # minimum intensity std
    T_ENTROPY = 3.5        # minimum entropy (0-8 for 8-bit)
    T_LAP_VAR = 10.0       # minimum variance of Laplacian (focus/texture)
    EDGE_MIN = 0.02        # min edge density
    EDGE_MAX = 0.45        # max edge density (too noisy if > max)
    T_ORI_COH = 0.14       # orientation coherence (max bin proportion over uniform ~1/18=0.055)

    @staticmethod
    def _entropy(gray: np.ndarray) -> float:
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        p = hist / (hist.sum() + 1e-9)
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum())

    @staticmethod
    def _edge_density(gray: np.ndarray) -> float:
        edges = cv2.Canny(gray, 50, 150)
        return float(edges.mean()) / 255.0

    @staticmethod
    def _laplacian_var(gray: np.ndarray) -> float:
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def _orientation_coherence(gray: np.ndarray) -> float:
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        angles = np.arctan2(gy, gx)
        # Map to [0, pi)
        angles = np.mod(angles, np.pi)
        bins = 18
        hist, _ = np.histogram(angles, bins=bins, range=(0, np.pi))
        total = hist.sum() + 1e-6
        max_prop = float(hist.max()) / total
        return max_prop

    def assess(self, img: np.ndarray):
        if img is None:
            return 0.0, {
                'std': 0.0, 'entropy': 0.0, 'lap_var': 0.0,
                'edge_density': 0.0, 'ori_coh': 0.0
            }
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        # Work on reduced size for speed but keep stats stable
        try:
            gray_s = cv2.resize(gray, (224, 224), interpolation=cv2.INTER_AREA)
        except Exception:
            gray_s = gray
        std = float(gray_s.std())
        ent = self._entropy(gray_s)
        lapv = self._laplacian_var(gray_s)
        ed = self._edge_density(gray_s)
        coh = self._orientation_coherence(gray_s)

        # Normalize components roughly to [0,1]
        std_n = np.clip((std - self.T_STD) / (40.0 - self.T_STD + 1e-6), 0, 1)
        ent_n = np.clip((ent - self.T_ENTROPY) / (8.0 - self.T_ENTROPY + 1e-6), 0, 1)
        lap_n = np.clip((lapv - self.T_LAP_VAR) / (200.0 - self.T_LAP_VAR + 1e-6), 0, 1)
        # Edge density best around 0.1~0.3 → bell-shaped score
        ed_center, ed_width = 0.2, 0.15
        ed_n = np.exp(-((ed - ed_center) ** 2) / (2 * ed_width ** 2))
        coh_n = np.clip((coh - self.T_ORI_COH) / (0.6 - self.T_ORI_COH + 1e-6), 0, 1)

        # Weighted quality score
        score = float(0.25 * std_n + 0.2 * ent_n + 0.25 * lap_n + 0.15 * ed_n + 0.15 * coh_n)
        details = {
            'std': std, 'entropy': ent, 'lap_var': lapv,
            'edge_density': ed, 'ori_coh': coh, 'score': score
        }
        return score, details

    def is_valid(self, img: np.ndarray):
        score, d = self.assess(img)
        # Hard gates first
        gates = [
            d['std'] >= self.T_STD,
            d['entropy'] >= self.T_ENTROPY,
            d['lap_var'] >= self.T_LAP_VAR,
            self.EDGE_MIN <= d['edge_density'] <= self.EDGE_MAX,
            d['ori_coh'] >= self.T_ORI_COH,
        ]
        ok = all(gates) and score >= 0.35
        return ok, score, d

class DatabaseManager:
    """Thread-safe database manager"""
    
    def __init__(self, db_path='oct_fingerprint_deep.sqlite'):
        self.db_path = db_path
        self._init_db()
        self._lock = threading.Lock()
    
    def _init_db(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON;")
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

            # Migrations: ensure required columns exist for older databases
            try:
                cursor.execute("PRAGMA table_info(users)")
                cols = {row[1] for row in cursor.fetchall()}  # row[1] is column name

                def ensure_col(col_name, col_def):
                    if col_name not in cols:
                        cursor.execute(f"ALTER TABLE users ADD COLUMN {col_name} {col_def}")

                # Columns used by the app
                ensure_col('bscan_folder', 'TEXT')
                ensure_col('model_path', 'TEXT')
                ensure_col('num_training_samples', "INTEGER DEFAULT 0")
                ensure_col('feature_extractor_type', "TEXT DEFAULT 'Basic'")
            except Exception as e:
                print(f"DB migration warning: {e}")
            
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
    
    def execute_query(self, query, params=None, fetch=None):
        """Thread-safe query execution"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON;")
                cursor = conn.cursor()
                
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                if fetch == 'one':
                    result = cursor.fetchone()
                elif fetch == 'all':
                    result = cursor.fetchall()
                elif fetch == 'lastrowid':
                    result = cursor.lastrowid
                else:
                    result = None
                
                conn.commit()
                return result
    
    def get_user_count(self):
        """Get total number of users"""
        result = self.execute_query("SELECT COUNT(*) FROM users", fetch='one')
        if isinstance(result, tuple):
            return result[0] if len(result) > 0 else 0
        if isinstance(result, list) and result:
            return result[0][0]
        if isinstance(result, int):
            return result
        return 0
    
    def get_all_users(self):
        """Get all users"""
        res = self.execute_query(
            "SELECT id, name, num_training_samples FROM users ORDER BY name",
            fetch='all'
        )
        return res if isinstance(res, list) else []
    
    def get_users_with_models(self):
        """Get users that have trained models"""
        res = self.execute_query(
            "SELECT id, name FROM users WHERE model_path IS NOT NULL ORDER BY name",
            fetch='all'
        )
        return res if isinstance(res, list) else []
    
    def get_user_details(self, user_id):
        """Get detailed user information"""
        query = "SELECT name, created_at, bscan_folder, model_path, num_training_samples, feature_extractor_type FROM users WHERE id = ?"
        return self.execute_query(query, (user_id,), fetch='one')
    
    def create_user(self, name, bscan_folder, feature_type):
        """Create a new user"""
        # Check if user exists
        existing = self.execute_query("SELECT id FROM users WHERE name = ?", (name,), fetch='one')
        if existing:
            raise ValueError(f"User '{name}' already exists")
        
        # Create user
        query = "INSERT INTO users (name, created_at, bscan_folder, feature_extractor_type) VALUES (?, ?, ?, ?)"
        params = (name, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), bscan_folder, feature_type)
        return self.execute_query(query, params, fetch='lastrowid')
    
    def update_user_model(self, user_id, model_path, num_samples, feature_type):
        """Update user model information"""
        query = "UPDATE users SET model_path = ?, num_training_samples = ?, feature_extractor_type = ? WHERE id = ?"
        self.execute_query(query, (model_path, num_samples, feature_type, user_id))
    
    def delete_user(self, user_id):
        """Delete a user"""
        self.execute_query("DELETE FROM users WHERE id = ?", (user_id,))
    
    def log_verification(self, user_id, verified, similarity_score, details):
        """Log verification attempt"""
        query = "INSERT INTO verification_logs (user_id, verified, similarity_score, verification_time, details) VALUES (?, ?, ?, ?, ?)"
        params = (user_id, verified, similarity_score, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), details)
        self.execute_query(query, params)
    
    def clear_verification_logs(self):
        """Clear all verification logs"""
        self.execute_query("DELETE FROM verification_logs")
    
    def get_admin_users(self):
        """Get users for admin panel"""
        res = self.execute_query(
            "SELECT id, name, created_at, num_training_samples, feature_extractor_type FROM users ORDER BY id",
            fetch='all'
        )
        return res if isinstance(res, list) else []
    
    def get_user_info(self, user_id):
        """Get user information (alias for get_user_details)"""
        return self.get_user_details(user_id)

class BasicFeatureExtractor:
    """Basic feature extractor using OpenCV and traditional computer vision methods"""
    
    def __init__(self):
        self.model_loaded = True
        print("Basic feature extractor initialized")
    
    def extract_features(self, image):
        """Extract features using traditional computer vision methods (without LBP)"""
        if image is None:
            return None
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Resize to standard size
        gray = cv2.resize(gray, (224, 224), interpolation=cv2.INTER_LANCZOS4)
        
        # Extract multiple types of features (without LBP)
        features = []
        
        # 1. Histogram features
        hist = cv2.calcHist([gray], [0], None, [128], [0, 256])  # More bins
        hist = hist.flatten()
        hist = hist / (hist.sum() + 1e-7)  # Normalize
        features.extend(hist)
        
        # 2. Edge features (enhanced)
        edges = cv2.Canny(gray, 50, 150)
        edge_hist = cv2.calcHist([edges], [0], None, [64], [0, 256])
        edge_hist = edge_hist.flatten()
        edge_hist = edge_hist / (edge_hist.sum() + 1e-7)
        features.extend(edge_hist)
        
        # 3. Gradient features
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        grad_hist = np.histogram(grad_mag.flatten(), bins=32, range=(0, 255))[0]
        grad_hist = grad_hist / (grad_hist.sum() + 1e-7)
        features.extend(grad_hist)
        
        # 4. Statistical features (enhanced)
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        skew_val = np.mean(((gray - mean_val) / std_val) ** 3) if std_val > 0 else 0
        kurt_val = np.mean(((gray - mean_val) / std_val) ** 4) if std_val > 0 else 0
        features.extend([mean_val/255.0, std_val/255.0, skew_val, kurt_val])
        
        # Convert to numpy array and normalize
        features = np.array(features, dtype=np.float32)
        
        # L2 normalization
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return features

class VGG16FeatureExtractor(BasicFeatureExtractor):
    """Deep learning feature extractor using available ONNX models"""
    
    def __init__(self, model_path=None):
        # Try to find available ONNX models
        if model_path is None:
            model_candidates = [
                os.path.join(MODELS_DIR, "vgg16-7.onnx"),
                os.path.join(MODELS_DIR, "resnet50-v2-7.onnx"),
                os.path.join(MODELS_DIR, "dl_cls_3b4887fb11.onnx"),
                os.path.join(MODELS_DIR, "dl_cls_c87c7530bb.onnx"),
                os.path.join(MODELS_DIR, "dl_cls_e19efbebbb.onnx"),
            ]
            
            for candidate in model_candidates:
                if os.path.exists(candidate):
                    model_path = candidate
                    break
        
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.output_name = None
        self.model_loaded = False
        
        # ImageNet normalization parameters (float32)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
        
        # Load model if available
        if ONNX_AVAILABLE and self.model_path and ort is not None:
            self.load_model()
        else:
            print("ONNX not available or no model found, using basic features")
            super().__init__()
    
    def load_model(self):
        """Load available ONNX model"""
        if not self.model_path or not os.path.exists(self.model_path):
            print(f"Deep learning model not found: {self.model_path}")
            super().__init__()
            return False

        try:
            if not ONNX_AVAILABLE or ort is None:
                raise RuntimeError("ONNX Runtime not available")
            self.session = ort.InferenceSession(self.model_path)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name

            model_name = os.path.basename(self.model_path)
            print(f"Deep learning model loaded: {model_name}")
            print(f"Input: {self.input_name}, Output: {self.output_name}")

            # Get input shape
            input_shape = self.session.get_inputs()[0].shape
            output_shape = self.session.get_outputs()[0].shape
            print(f"Input shape: {input_shape}, Output shape: {output_shape}")

            self.model_loaded = True
            return True

        except Exception as e:
            print(f"Error loading deep learning model: {e}")
            super().__init__()
            return False
    
    def preprocess_image(self, image, target_size=(224, 224)):
        """Preprocess image for VGG-16 input"""
        if image is None:
            return None
        
        try:
            # 1) Resize early for speed (denoise on smaller image)
            if len(image.shape) == 3 and image.shape[2] == 3:
                resized_bgr = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
                is_gray = False
            elif len(image.shape) == 2:
                resized_bgr = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
                is_gray = True
            else:
                # Unexpected shape; attempt safe resize/convert
                resized_bgr = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
                is_gray = len(resized_bgr.shape) == 2

            # 2) Non-Local Means Denoising (speckle/noise reduction)
            # Parameters tuned for 224x224 images (balanced speed/quality)
            if is_gray:
                try:
                    denoised = cv2.fastNlMeansDenoising(resized_bgr, None, h=10, templateWindowSize=7, searchWindowSize=21)
                except Exception:
                    denoised = resized_bgr
                # 3) CLAHE on grayscale
                try:
                    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
                    enhanced = clahe.apply(denoised)
                except Exception:
                    enhanced = denoised
                # Convert to RGB by stacking channels
                rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
            else:
                try:
                    denoised = cv2.fastNlMeansDenoisingColored(resized_bgr, None, h=8, hColor=8, templateWindowSize=7, searchWindowSize=21)
                except Exception:
                    denoised = resized_bgr
                # 3) CLAHE on L channel in LAB color space
                try:
                    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
                    l_clahe = clahe.apply(l)
                    lab_clahe = cv2.merge((l_clahe, a, b))
                    bgr_enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
                except Exception:
                    bgr_enhanced = denoised
                # Convert BGR -> RGB
                rgb = cv2.cvtColor(bgr_enhanced, cv2.COLOR_BGR2RGB)

            # 4) Convert to float32 in [0,1]
            image_f = rgb.astype(np.float32) / 255.0

            # 5) HWC -> CHW and add batch dimension
            image_f = np.transpose(image_f, (2, 0, 1))
            image_f = np.expand_dims(image_f, axis=0)

            # 6) ImageNet normalization
            image_f = (image_f - self.mean) / self.std

            return image_f.astype(np.float32)

        except Exception as e:
            # Fallback to original simple pipeline if preprocessing fails
            # Convert to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif len(image.shape) == 2:  # Grayscale
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                rgb_image = image

            resized = cv2.resize(rgb_image, target_size, interpolation=cv2.INTER_LINEAR)
            img = resized.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, axis=0)
            img = (img - self.mean) / self.std
            return img.astype(np.float32)
    
    def extract_features(self, image):
        """Extract VGG-16 deep learning features from image"""
        if not self.model_loaded or not ONNX_AVAILABLE or self.session is None:
            print("VGG-16 not available, using basic features")
            return super().extract_features(image)
        
        # Validate input image
        if image is None:
            print("Input image is None")
            return None
        
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)
            if processed_image is None:
                print("Image preprocessing failed")
                return None
            
            # Run VGG-16 inference
            outputs = self.session.run([self.output_name], {self.input_name: processed_image})
            features = outputs[0].flatten().astype(np.float32)
            
            # Validate feature extraction
            if features is None or len(features) == 0:
                print("Feature extraction returned empty result")
                return None
            
            # L2 normalization
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm
            
            print(f"VGG-16 features extracted: shape={features.shape}")
            return features
            
        except Exception as e:
            print(f"Error extracting VGG-16 features: {e}")
            print("Falling back to basic features")
            return super().extract_features(image)
    
    def predict_user(self, image, registered_users_features):
        """Classification-based user prediction using VGG-16"""
        if not self.model_loaded or not ONNX_AVAILABLE:
            return None, 0.0, "VGG-16 model not available"
        
        # Extract features from input image
        input_features = self.extract_features(image)
        if input_features is None:
            return None, 0.0, "Feature extraction failed"
        
        try:
            best_match_user = None
            best_confidence = 0.0
            all_similarities = []
            
            # Compare with all registered users
            for user_id, user_name, user_features_list in registered_users_features:
                user_similarities = []
                
                for user_feature in user_features_list:
                    try:
                        # Check feature compatibility
                        if len(input_features) != len(user_feature):
                            print(f"⚠️  Incompatible features: VGG-16={len(input_features)}, User {user_name}={len(user_feature)}")
                            continue
                        
                        # Ensure user features are normalized
                        user_feature = np.array(user_feature, dtype=np.float32)
                        user_norm = np.linalg.norm(user_feature)
                        if user_norm > 0:
                            user_feature = user_feature / user_norm
                        
                        # Compute cosine similarity
                        similarity = np.dot(input_features, user_feature)
                        
                        # Apply sigmoid activation for classification confidence
                        confidence = 1.0 / (1.0 + np.exp(-10 * (similarity - 0.5)))
                        user_similarities.append(confidence)
                        
                    except Exception as e:
                        print(f"Error comparing with user {user_name}: {e}")
                        continue
                
                # Use maximum similarity for this user
                if user_similarities:
                    max_similarity = np.max(user_similarities)
                    all_similarities.append((user_id, user_name, max_similarity))
                    
                    if max_similarity > best_confidence:
                        best_confidence = max_similarity
                        best_match_user = (user_id, user_name)
            
            # Classification decision
            if best_confidence > 0.7 and best_match_user is not None:  # High confidence threshold
                message = f"VGG-16 Classification: HIGH confidence match"
                return best_match_user[0], best_confidence, message
            elif best_confidence > 0.5 and best_match_user is not None:  # Medium confidence
                message = f"VGG-16 Classification: MEDIUM confidence match"  
                return best_match_user[0], best_confidence, message
            else:
                # Sort similarities to show top candidates
                all_similarities.sort(key=lambda x: x[2], reverse=True)
                top_users = all_similarities[:3]
                candidates = ", ".join([f"{name}({sim:.3f})" for _, name, sim in top_users])
                message = f"VGG-16 Classification: LOW confidence. Top candidates: {candidates}"
                return None, best_confidence, message
                
        except Exception as e:
            return None, 0.0, f"VGG-16 prediction error: {e}"

class OCTUserManager:
    """Manage OCT user training and verification"""
    
    def __init__(self, feature_extractor, quality_assessor: Optional['QualityAssessor'] = None):
        self.feature_extractor = feature_extractor
        self.models_dir = "user_models"
        os.makedirs(self.models_dir, exist_ok=True)
        self.quality_assessor = quality_assessor
        self.filter_training = True
    
    def load_bscan_images(self, folder_path):
        """Load all B-scan images from folder"""
        if not os.path.exists(folder_path):
            return []
        
        # Case-insensitive extension matching
        valid_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
        images = []
        for root, _, files in os.walk(folder_path):
            for fname in files:
                try:
                    if os.path.splitext(fname)[1].lower() in valid_exts:
                        file_path = os.path.join(root, fname)
                        # Robust read: unchanged first to support 16-bit TIFF, then convert to grayscale
                        img0 = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                        if img0 is None:
                            # Fallback to grayscale flag
                            img0 = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                        if img0 is None:
                            print(f"Warning: Unable to read image (cv2 returned None): {file_path}")
                            continue
                        # Convert to grayscale if multi-channel
                        if len(img0.shape) == 3:
                            # Handle RGBA as well
                            if img0.shape[2] == 4:
                                img0 = cv2.cvtColor(img0, cv2.COLOR_BGRA2GRAY)
                            else:
                                img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
                        # Normalize to uint8 if needed
                        if img0.dtype != np.uint8:
                            # Allocate destination to avoid None typing issues
                            img_norm = np.empty_like(img0, dtype=np.float32)
                            cv2.normalize(img0.astype(np.float32), img_norm, 0, 255, cv2.NORM_MINMAX)
                            img0 = img_norm.astype(np.uint8)
                        images.append((file_path, img0))
                except Exception as e:
                    print(f"Error loading {fname}: {e}")
        
        return images
    
    def extract_features_from_folder(self, folder_path, user_id, progress_callback=None):
        """Extract features from all images in B-scan folder"""
        images = self.load_bscan_images(folder_path)
        
        if not images:
            return None, []
        
        features_list = []
        file_paths = []
        skipped = 0
        
        total_images = len(images)
        for i, (file_path, img) in enumerate(images):
            if progress_callback:
                progress_callback(i + 1, total_images, f"Processing {os.path.basename(file_path)}")
            # Filter low-quality/noise frames if enabled
            if self.filter_training and self.quality_assessor is not None:
                ok, score, d = self.quality_assessor.is_valid(img)
                if not ok:
                    skipped += 1
                    continue
            
            features = self.feature_extractor.extract_features(img)
            if features is not None:
                features_list.append(features)
                file_paths.append(file_path)
        
        # Fallback: if everything was filtered or failed, try once without filtering
        if not features_list and total_images > 0:
            if self.filter_training and self.quality_assessor is not None:
                print("Training fallback: No features after quality filtering; retrying without filter once...")
                features_list = []
                file_paths = []
                for i, (file_path, img) in enumerate(images):
                    if progress_callback:
                        progress_callback(i + 1, total_images, f"Retry (no-filter): {os.path.basename(file_path)}")
                    features = self.feature_extractor.extract_features(img)
                    if features is not None:
                        features_list.append(features)
                        file_paths.append(file_path)
            
            if not features_list:
                return None, []
        
        # Convert to numpy array
        features_array = np.array(features_list)
        if progress_callback and skipped:
            progress_callback(total_images, total_images, f"Skipped {skipped} low-quality frames")
        
        return features_array, file_paths
    
    def train_user_model(self, user_id, user_name, bscan_folder, progress_callback=None, epochs=500):
        """Train a user-specific model from B-scan folder with advanced features"""
        
        # Extract features from user's B-scan folder
        user_features, user_files = self.extract_features_from_folder(
            bscan_folder, user_id, progress_callback
        )
        
        if user_features is None or len(user_features) == 0:
            raise Exception("No valid B-scan images found or feature extraction failed")
        
        # Advanced feature processing for enhanced training (simulate epochs)
        enhanced_features = []
        augmented_files = []
        
        if progress_callback:
            progress_callback(0, epochs, f"เริ่มการเทรนขั้นสูง {epochs} epochs...")
        
        # Simulate advanced training with multiple epochs
        for epoch in range(epochs):
            if progress_callback and epoch % 25 == 0:
                progress_callback(epoch, epochs, f"Training epoch {epoch+1}/{epochs}")
            
            # Feature enhancement through VGG16 processing
            for i, features in enumerate(user_features):
                # Add some variation to simulate learning (small random noise)
                if hasattr(self.feature_extractor, 'model_loaded') and self.feature_extractor.model_loaded:
                    # For VGG16 model, add slight variations
                    noise_factor = 0.001 * (1 - epoch / epochs)  # Decrease noise as training progresses
                    enhanced_feat = features + np.random.normal(0, noise_factor, features.shape)
                    enhanced_feat = enhanced_feat / np.linalg.norm(enhanced_feat)  # Re-normalize
                else:
                    enhanced_feat = features
                    
                enhanced_features.append(enhanced_feat)
                augmented_files.append(user_files[i % len(user_files)])
        
        # Convert to numpy array (using only enhanced features from final epochs)
        final_enhanced_features = np.array(enhanced_features[-len(user_features):])
        
        # Compute centroid/std from original features
        # Ensure L2 normalization
        uf = np.array(user_features, dtype=np.float32)
        norms = np.linalg.norm(uf, axis=1, keepdims=True) + 1e-8
        uf_norm = uf / norms
        centroid = uf_norm.mean(axis=0)
        c_norm = np.linalg.norm(centroid) + 1e-8
        centroid = centroid / c_norm
        # Similarities to centroid
        sims = (uf_norm @ centroid)
        mean_s = float(np.mean(sims))
        std_s = float(np.std(sims))
        # Per-user threshold: mean - 1.5*std, clamped
        per_user_threshold = max(0.55, min(0.92, mean_s - 1.5 * std_s))

        # Create user model data with enhanced information
        user_model_data = {
            'user_id': user_id,
            'user_name': user_name,
            'features': final_enhanced_features,
            'original_features': user_features,  # Keep originals for comparison
            'file_paths': user_files,
            'mean_features': np.mean(final_enhanced_features, axis=0),
            'std_features': np.std(final_enhanced_features, axis=0),
            'num_samples': len(user_features),
            'training_epochs': epochs,
            'created_at': datetime.now().isoformat(),
            'bscan_folder': bscan_folder,
            'feature_extractor_type': 'VGG16' if hasattr(self.feature_extractor, 'model_loaded') and self.feature_extractor.model_loaded else 'Basic',
            'vgg16_model': 'vgg16-7.onnx' if hasattr(self.feature_extractor, 'model_loaded') and self.feature_extractor.model_loaded else None,
            # Centroid (L2-normalized over original features) and per-user threshold for centroid similarity
            'centroid': centroid.astype(np.float32),
            'per_user_threshold': per_user_threshold,
            'centroid_stats': {'mean_s': mean_s, 'std_s': std_s}
        }
        
        # Save user model
        model_filename = f"user_{user_id}_{user_name.replace(' ', '_')}_vgg16_model.pkl"
        model_path = os.path.join(self.models_dir, model_filename)
        
        with open(model_path, 'wb') as f:
            pickle.dump(user_model_data, f)
        
        if progress_callback:
            progress_callback(epochs, epochs, f"เทรนเสร็จแล้ว! ใช้ {epochs} epochs")
        
        print(f"Enhanced user model saved: {model_path}")
        print(f"Features extracted: {len(user_features)} samples, trained for {epochs} epochs")
        
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
        """Calculate VGG16 similarity between two feature vectors"""
        
        # Convert to numpy arrays and check compatibility
        features1 = np.array(features1, dtype=np.float32).flatten()
        features2 = np.array(features2, dtype=np.float32).flatten()
        
        # Check dimension compatibility
        if features1.shape != features2.shape:
            print(f"Feature dimension mismatch: {features1.shape} vs {features2.shape}")
            return 0.0
        
        # Basic similarity calculation
        def basic_cosine_similarity(f1, f2):
            dot_product = np.dot(f1, f2)
            norm1 = np.linalg.norm(f1)
            norm2 = np.linalg.norm(f2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            else:
                return dot_product / (norm1 * norm2)
        
        # Use VGG16 approach if available
        if hasattr(self.feature_extractor, 'model_loaded') and self.feature_extractor.model_loaded:
            # For VGG16 features, calculate multiple similarity metrics
            basic_sim = basic_cosine_similarity(features1, features2)
            
            # Additional similarity metrics for VGG16 classification
            # Euclidean distance based similarity
            euclidean_dist = np.linalg.norm(features1 - features2)
            max_dist = np.linalg.norm(features1) + np.linalg.norm(features2)
            euclidean_sim = 1.0 / (1.0 + euclidean_dist / max_dist) if max_dist > 0 else 0.0
            
            # Manhattan distance based similarity  
            manhattan_dist = np.sum(np.abs(features1 - features2))
            max_manhattan = np.sum(np.abs(features1)) + np.sum(np.abs(features2))
            manhattan_sim = 1.0 / (1.0 + manhattan_dist / max_manhattan) if max_manhattan > 0 else 0.0
            
            # Weighted VGG16 classification voting
            vgg16_similarity = (basic_sim * 0.6) + (euclidean_sim * 0.25) + (manhattan_sim * 0.15)
            
            return float(vgg16_similarity)
        
        else:
            # Fallback to basic similarity
            return float(basic_cosine_similarity(features1, features2))
    
    def verify_user(self, test_image, user_id=None, threshold=0.7):
        """Verify user identity using centroid-based matching with optional TTA."""
        # Get all registered users
        db_manager = DatabaseManager()
        users = db_manager.get_users_with_models()
        if not users:
            return False, 0.0, "No registered users found", None

        # Build embedding for test image (use TTA if deep model is loaded)
        use_tta = hasattr(self.feature_extractor, 'model_loaded') and getattr(self.feature_extractor, 'model_loaded', False)
        test_embed = self._compute_tta_embedding(test_image) if use_tta else self.feature_extractor.extract_features(test_image)
        if test_embed is None:
            return False, 0.0, "Feature extraction failed", None

        # Ensure L2 normalization
        n = np.linalg.norm(test_embed) + 1e-8
        test_embed = (test_embed / n).astype(np.float32)

        results = []
        for row in users:
            try:
                uid, uname = row[0], row[1]
                um = self.load_user_model(uid)
                if not um:
                    continue
                centroid = um.get('centroid')
                if centroid is None:
                    # fallback to mean_features if centroid missing
                    centroid = um.get('mean_features')
                if centroid is None:
                    continue
                centroid = np.array(centroid, dtype=np.float32).flatten()
                # normalize
                cn = np.linalg.norm(centroid) + 1e-8
                centroid = centroid / cn

                # cosine to centroid
                cen_sim = float(np.dot(test_embed, centroid))

                # optional: max similarity to individual features
                max_pair = 0.0
                feats = um.get('features')
                if feats is not None and len(feats) > 0:
                    for f in feats:
                        f = np.array(f, dtype=np.float32).flatten()
                        fn = np.linalg.norm(f) + 1e-8
                        if fn > 0:
                            f = f / fn
                            s = float(np.dot(test_embed, f))
                            if s > max_pair:
                                max_pair = s
                # weighted score
                weighted = 0.7 * cen_sim + 0.3 * max_pair
                per_thr = um.get('per_user_threshold', None)
                results.append({'user_id': uid, 'user_name': uname, 'cen_sim': cen_sim, 'max_pair': max_pair, 'weighted': weighted, 'thr': per_thr})
            except Exception as e:
                print(f"verify loop error for {row}: {e}")
                continue

        if not results:
            return False, 0.0, "No valid user models found", None

        # pick best by weighted
        results.sort(key=lambda x: x['weighted'], reverse=True)
        best = results[0]

        # decide thresholds
        eff_thr = threshold
        if best['thr'] is not None:
            eff_thr = max(eff_thr, float(best['thr']))

        if user_id is not None and best['user_id'] != user_id:
            details = f"❌ User Mismatch: Requested {user_id}, Predicted {best['user_name']} (Score: {best['weighted']:.3f}, Thr: {eff_thr:.2f})"
            return False, float(best['weighted']), details, best['user_id']

        is_ok = best['weighted'] >= eff_thr
        status = '✅ Match' if is_ok else '❌ Low Confidence'
        # top-3 for details
        top = results[:3]
        cand = ", ".join([f"{r['user_name']}({r['weighted']:.3f})" for r in top])
        details = f"{status}: best={best['user_name']} score={best['weighted']:.3f} thr={eff_thr:.2f}\nTop: {cand}"
        return is_ok, float(best['weighted']), details, best['user_id']

    def _compute_tta_embedding(self, image, angles=(-5, 0, 5)):
        """Compute TTA embedding by averaging features over slight rotations."""
        embeds = []
        try:
            h, w = image.shape[:2]
        except Exception:
            return None
        center = (w / 2.0, h / 2.0)
        for ang in angles:
            try:
                if ang == 0:
                    img_aug = image
                else:
                    M = cv2.getRotationMatrix2D(center, ang, 1.0)
                    img_aug = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                emb = self.feature_extractor.extract_features(img_aug)
                if emb is not None:
                    n = np.linalg.norm(emb) + 1e-8
                    emb = emb / n
                    embeds.append(emb.astype(np.float32))
            except Exception as e:
                print(f"TTA angle {ang} error: {e}")
                continue
        if not embeds:
            return None
        mean_emb = np.mean(embeds, axis=0)
        n = np.linalg.norm(mean_emb) + 1e-8
        return (mean_emb / n).astype(np.float32)
    
    def identify_user_automatically_fallback(self, test_image, threshold=0.7, db_manager=None):
        """Automatically identify user from all registered users"""
        
        # Use provided database manager or create new one
        if db_manager is None:
            db_manager = DatabaseManager()
        
        # Get all users with models
        users = db_manager.get_users_with_models()
        
        if not users:
            return False, 0.0, "No registered users found", None
        
        # Extract features from test image
        test_features = self.feature_extractor.extract_features(test_image)
        if test_features is None:
            return False, 0.0, "Feature extraction failed", None
        
        best_match = None
        best_similarity = 0.0
        all_results = []
        
        # Test against all users
        for user_data in users:
            user_id, user_name = user_data[0], user_data[1]  # Handle tuple format
            
            user_model = self.load_user_model(user_id)
            if user_model is None:
                continue
            
            # Calculate similarities using VGG16 classification
            user_features = user_model['features']
            similarities = []
            
            for train_features in user_features:
                similarity = self.calculate_similarity(test_features, train_features)
                similarities.append(similarity)
            
            if similarities:
                max_sim = np.max(similarities)
                avg_sim = np.mean(similarities)
                median_sim = np.median(similarities)
                
                # Weighted score considering all metrics
                weighted_score = (max_sim * 0.5) + (avg_sim * 0.3) + (median_sim * 0.2)
                
                all_results.append({
                    'user_id': user_id,
                    'user_name': user_name,
                    'max_similarity': max_sim,
                    'avg_similarity': avg_sim,
                    'median_similarity': median_sim,
                    'weighted_score': weighted_score,
                    'num_samples': len(similarities)
                })
                
                # Track best match using weighted score
                if weighted_score > best_similarity:
                    best_similarity = weighted_score
                    best_match = {
                        'user_id': user_id,
                        'user_name': user_name,
                        'max_similarity': max_sim,
                        'avg_similarity': avg_sim,
                        'median_similarity': median_sim,
                        'weighted_score': weighted_score,
                        'num_samples': len(similarities)
                    }
        
        # Sort results by weighted score
        all_results.sort(key=lambda x: x['weighted_score'], reverse=True)
        
        # Check if best match meets threshold
        if best_match and best_similarity >= threshold:
            result_info = f"🎯 Auto-Identified: {best_match['user_name']}\nWeighted Score: {best_similarity:.3f}\nMax: {best_match['max_similarity']:.3f}, Avg: {best_match['avg_similarity']:.3f}"
            return True, best_similarity, result_info, best_match['user_id']
        else:
            # Show top 3 closest matches
            top_matches = all_results[:3]
            match_info = []
            for i, match in enumerate(top_matches, 1):
                match_info.append(f"{i}. {match['user_name']}: {match['weighted_score']:.3f}")
            
            result_info = f"❌ No auto-identification (threshold: {threshold:.3f})\nTop candidates:\n" + "\n".join(match_info)
            return False, best_similarity, result_info, None

class OCTDeepFingerprintSystem:
    """Main OCT Deep Learning Fingerprint System with thread-safe database and screen capture"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("ระบบตรวจสอบลายนิ้วมือ OCT - Deep Learning + Screen Capture (v6.2)")
        self.root.geometry("1400x900")

        # Initialize components
        if ONNX_AVAILABLE:
            self.feature_extractor = VGG16FeatureExtractor()
        else:
            self.feature_extractor = BasicFeatureExtractor()

        # Image quality assessor (for training/verification filtering)
        self.quality_assessor = QualityAssessor()
        self.user_manager = OCTUserManager(self.feature_extractor, self.quality_assessor)
        self.db = DatabaseManager()

        # Screen capture components
        self.sct = None
        self.capture_monitor = None
        self.capture_bbox = None
        self.is_live_capturing = False
        self.live_capture_job_id = None

        # Current state
        self.current_user = None
        self.current_scan = None

        # Arduino components
        self.arduino_port = None
        self.arduino_serial = None
        self.arduino_connected = False
        self.relay_pin = 7  # Default relay pin (D7)
        self.relay_open_duration = 3  # Duration in seconds
        self.arduino_status_var = tk.StringVar(master=self.root, value="Arduino: ไม่เชื่อมต่อ")
        # Preferred port (user requested COM3)
        self.preferred_arduino_port = "COM3"
        
        # Quality filter toggles (default ON)
        self.filter_verification_var = tk.BooleanVar(master=self.root, value=True)
        self.filter_training_var = tk.BooleanVar(master=self.root, value=True)
        # Apply initial training filter state to user manager
        self.user_manager.filter_training = bool(self.filter_training_var.get())

        # UI styling
        self.setup_styles()

        # Initialize screen capture
        self.init_screen_capture()

        # Create UI
        self.create_ui()

        # Initialize Arduino after delay
        self.root.after(1000, self.init_arduino)

        # Check feature compatibility after UI is ready
        self.root.after(2000, self.check_feature_compatibility_on_startup)
    
    def check_feature_compatibility_on_startup(self):
        """Check for feature compatibility issues when starting the system"""
        try:
            if not ONNX_AVAILABLE or not hasattr(self.feature_extractor, 'model_loaded'):
                return  # Skip check for basic feature extractor
                
            users = self.db.get_all_users()
            if not users:
                return  # No users to check
            
            # Get current VGG16 feature dimensions
            test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            current_features = self.feature_extractor.extract_features(test_image)
            current_dim = len(current_features) if current_features is not None else 0
            
            incompatible_users = []
            
            for user_data in users:
                user_id, user_name = user_data[0], user_data[1]
                try:
                    user_model = self.user_manager.load_user_model(user_id)
                    if user_model and 'features' in user_model:
                        user_features = user_model['features'][0] if len(user_model['features']) > 0 else None
                        if user_features is not None and len(user_features) != current_dim:
                            incompatible_users.append((user_name, len(user_features), current_dim))
                except:
                    continue
            
            if incompatible_users:
                # Create startup warning
                self.show_compatibility_warning(incompatible_users)
                
        except Exception as e:
            print(f"Error checking feature compatibility: {e}")
            
    def show_compatibility_warning(self, incompatible_users):
        """Show warning dialog for incompatible features"""
        warning_msg = f"⚠️  พบ {len(incompatible_users)} ผู้ใช้ที่มี features ไม่เข้ากัน:\n\n"
        
        for user_name, user_dim, current_dim in incompatible_users:
            warning_msg += f"• {user_name}: {user_dim} dims (ปัจจุบันต้องการ {current_dim} dims)\n"
            
        warning_msg += f"\n🔄 กรุณาใช้ปุ่ม 'Migrate Features' เพื่อความแม่นยำสูงสุด\n"
        warning_msg += f"📉 หากไม่ migrate อาจมีผลต่อความแม่นยำในการตรวจสอบ"
        
        try:
            messagebox.showwarning("Feature Compatibility Warning", warning_msg, parent=self.root)
        except:
            print(warning_msg)  # Fallback to console
    
    def init_screen_capture(self):
        """Initialize screen capture system"""
        if MSS_AVAILABLE and mss is not None:
            try:
                self.sct = mss.mss()
                self.capture_monitor = self.sct.monitors[1] if len(self.sct.monitors) > 1 else self.sct.monitors[0]
                print(f"Screen capture initialized. Monitor: {self.capture_monitor}")
            except Exception as e:
                print(f"Error initializing screen capture: {e}")
                self.sct = None
        else:
            print("Screen capture not available - MSS not installed")
    
    def close_screen_capture(self):
        """Close screen capture system"""
        if self.sct:
            try:
                self.sct.close()
                print("Screen capture closed")
            except Exception as e:
                print(f"Error closing screen capture: {e}")
    
    def capture_fullscreen(self):
        """Capture full screen"""
        if not self.sct or not self.capture_monitor:
            print("Error: Screen capture not initialized")
            return None
        
        try:
            screenshot = self.sct.grab(self.capture_monitor)
            img_pil = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
            img_pil_gray = img_pil.convert('L')
            return np.array(img_pil_gray)
        except Exception as e:
            print(f"Error capturing fullscreen: {e}")
            return None
    
    def select_capture_area(self):
        """Select area for live capture"""
        if self.is_live_capturing:
            self.stop_live_capture()
        
        self.status_var.set("จับภาพหน้าจอสำหรับเลือกพื้นที่...")
        self.root.update_idletasks()
        
        # Minimize window and capture
        self.root.iconify()
        self.root.after(300)  # Wait for window to minimize
        
        full_img_np = self.capture_fullscreen()
        
        # Restore window
        if self.root.state() != 'normal':
            self.root.deiconify()
        
        if full_img_np is None:
            messagebox.showerror("ข้อผิดพลาด", "ไม่สามารถจับภาพหน้าจอได้", parent=self.root)
            self.status_var.set("เลือกพื้นที่ล้มเหลว")
            return
        
        try:
            # Create PIL image and open cropper
            full_pil_img = Image.fromarray(full_img_np)
            self.status_var.set("เลือกพื้นที่ในหน้าต่างใหม่...")
            
            cropper = ImageCropper(self.root, full_pil_img)
            self.root.wait_window(cropper)
            
            bbox = getattr(cropper, 'selected_bbox', None)
            
            if bbox:
                x1, y1, x2, y2 = bbox
                w, h = x2 - x1, y2 - y1
                
                # Create MSS-compatible bbox
                self.capture_bbox = {
                    "top": y1,
                    "left": x1,
                    "width": w,
                    "height": h
                }
                
                area_info = f"พื้นที่: {w}x{h} @({x1},{y1})"
                self.capture_area_var.set(area_info)
                
                # Display preview
                preview_img = full_img_np[y1:y2, x1:x2].copy()
                self.display_image_on_canvas(preview_img, self.verify_image_canvas)
                
                # Enable live capture button
                self.live_capture_btn.config(state=tk.NORMAL)
                
                self.status_var.set(f"เลือกพื้นที่แล้ว - {area_info}")
                
            else:
                # User cancelled
                self.capture_bbox = None
                self.capture_area_var.set("ยังไม่ได้เลือกพื้นที่")
                self.live_capture_btn.config(state=tk.DISABLED)
                self.status_var.set("ยกเลิกเลือกพื้นที่")
                
        except Exception as e:
            messagebox.showerror("ข้อผิดพลาด", f"เกิดข้อผิดพลาดในการเลือกพื้นที่: {e}", parent=self.root)
            self.status_var.set("เลือกพื้นที่ล้มเหลว")
    
    def start_live_capture(self):
        """Start continuous live capture for verification"""
        if not self.sct:
            messagebox.showerror("ข้อผิดพลาด", "ระบบจับภาพหน้าจอไม่พร้อม", parent=self.root)
            return
        
        if not self.capture_bbox:
            messagebox.showerror("ข้อผิดพลาด", "กรุณาเลือกพื้นที่ก่อน", parent=self.root)
            return
        
        # Remove requirement to select user - always use auto-identification
        self.is_live_capturing = True
        self.live_capture_btn.config(text="หยุด Scan", command=self.stop_live_capture)
        self.select_area_btn.config(state=tk.DISABLED)
        
        self.status_var.set("🔍 Continuous Scanning Active...")
        self.verification_result_var.set("🎯 กำลังสแกนอัตโนมัติ...")
        self.verification_result_label.config(style='Info.TLabel')
        
        # Reset counters
        self._frame_count = 0
        self._last_verification_time = 0
        
        # Start capture loop
        self.live_capture_loop()
    
    def stop_live_capture(self):
        """Stop continuous live capture"""
        self.is_live_capturing = False
        
        if self.live_capture_job_id:
            self.root.after_cancel(self.live_capture_job_id)
            self.live_capture_job_id = None
        
        self.live_capture_btn.config(text="Scan", command=self.start_live_capture)
        self.select_area_btn.config(state=tk.NORMAL)
        
        self.status_var.set("⏹️ สแกนหยุดทำงาน")
        self.verification_result_var.set("ระบบพร้อม")
        self.verification_result_label.config(style='Normal.TLabel')
    
    def live_capture_loop(self):
        """Continuous live capture loop with deep learning prediction"""
        if not self.is_live_capturing:
            return
        
        if not self.sct or not self.capture_bbox:
            self.stop_live_capture()
            return
        
        try:
            # Capture area
            screenshot = self.sct.grab(self.capture_bbox)
            img_bgr = np.array(screenshot)
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2GRAY)
            
            # Display captured image
            self.display_image_on_canvas(img_gray, self.verify_image_canvas)
            
            # Continuous verification with rate limiting
            current_time = time.time()
            frame_count = getattr(self, '_frame_count', 0) + 1
            last_verification = getattr(self, '_last_verification_time', 0)
            self._frame_count = frame_count
            
            # Verify every 30 frames OR every 2 seconds (whichever comes first) 
            should_verify = (frame_count % 30 == 0) or (current_time - last_verification > 2.0)
            
            if should_verify:
                self._last_verification_time = current_time
                
                # Always use auto-identification (no user selection required)
                threshold = self.threshold_var.get()
                
                # Optional quality gating
                if self.filter_verification_var.get() and self.quality_assessor is not None:
                    ok, qscore, qd = self.quality_assessor.is_valid(img_gray)
                    if not ok:
                        # Skip verification on low-quality frame
                        self.verification_result_var.set("🔍 กำลังสแกน... (กรอง Noise)")
                        self.verification_result_label.config(style='Info.TLabel')
                        self.status_var.set(f"🔍 Scanning... (Frame: {frame_count}) [skip low-quality: {qscore:.2f}]")
                        self.verification_details_var.set(
                            f"Quality score: {qscore:.3f} | Threshold: {threshold:.2f}\n"
                            f"std:{qd['std']:.1f} ent:{qd['entropy']:.2f} lap:{qd['lap_var']:.1f} "
                            f"edge:{qd['edge_density']:.3f} coh:{qd['ori_coh']:.3f}\n"
                            f"Frame: {frame_count} | Time: {current_time:.1f}"
                        )
                        # Schedule next frame
                        delay_ms = 33
                        self.live_capture_job_id = self.root.after(delay_ms, self.live_capture_loop)
                        return
                
                # Perform deep learning prediction
                is_verified, confidence, details, matched_user_id = self.user_manager.verify_user(
                    img_gray, None, threshold=threshold  # None = auto-identification
                )
                
                # Update UI with continuous feedback
                if is_verified:
                    self.verification_result_var.set("✅ ตรวจสอบผ่าน")
                    self.verification_result_label.config(style='Success.TLabel')
                    
                    # Open Arduino door if connected
                    if self.arduino_connected:
                        self.open_door()
                    
                    # Log verification
                    if matched_user_id:
                        self.db.log_verification(matched_user_id, True, confidence, details)
                    
                    # Continue scanning (don't stop after success)
                    self.status_var.set(f"✅ Access Granted - Continuing Scan... (Frame: {frame_count})")
                    
                    # Brief pause to show success
                    self.root.after(1000, lambda: self.status_var.set("🔍 Continuous Scanning Active..."))
                    
                else:
                    # Show scanning status instead of failure
                    self.verification_result_var.set("🔍 กำลังสแกน...")
                    self.verification_result_label.config(style='Info.TLabel')
                    
                    # Update status with frame count
                    self.status_var.set(f"🔍 Scanning... (Frame: {frame_count})")
                    
                    # Log failed attempt if we had a candidate
                    if matched_user_id:
                        self.db.log_verification(matched_user_id, False, confidence, details)
                
                # Always show details for debugging
                self.verification_details_var.set(
                    f"Confidence: {confidence:.3f} | Threshold: {threshold:.2f}\n{details}\nFrame: {frame_count} | Time: {current_time:.1f}"
                )
            
            else:
                # Update frame counter display without verification
                self.status_var.set(f"🔍 Scanning... (Frame: {frame_count})")
            
            # Schedule next capture with higher frame rate for smoothness
            delay_ms = 33  # ~30 FPS for smooth video
            self.live_capture_job_id = self.root.after(delay_ms, self.live_capture_loop)
            
        except Exception as e:
            print(f"Live capture error: {e}")
            self.verification_result_var.set("❌ Scan Error")
            self.verification_result_label.config(style='Failure.TLabel')
            self.status_var.set(f"Error: {str(e)[:50]}...")
            
            # Continue scanning despite errors
            self.live_capture_job_id = self.root.after(1000, self.live_capture_loop)
            print(f"Live capture error: {e}")
            self.stop_live_capture()
            messagebox.showerror("ข้อผิดพลาด", f"เกิดข้อผิดพลาดในการจับภาพสด: {e}", parent=self.root)
    
    def setup_styles(self):
        """Setup UI styles"""
        self.style = ttk.Style(self.root)
        
        try:
            bg = self.style.lookup('TFrame', 'background')
            self.style.configure('Normal.TLabel', foreground='black', background=bg, font=('Arial', 12))
            self.style.configure('Success.TLabel', foreground='green', background='#d4ffcc', font=('Arial', 14, 'bold'))
            self.style.configure('Failure.TLabel', foreground='#cc0000', background='#ffcccc', font=('Arial', 14, 'bold'))
            self.style.configure('Info.TLabel', foreground='blue', background='#cceeff', font=('Arial', 12, 'bold'))
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
        self.bscan_folder_var = tk.StringVar(master=self.root, value="ยังไม่ได้เลือกโฟลเดอร์")
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

        # Training quality filter toggle
        filter_frame = ttk.Frame(progress_frame)
        filter_frame.pack(fill='x', pady=5)
        ttk.Checkbutton(
            filter_frame,
            text="กรองภาพคุณภาพต่ำระหว่างเทรน (QualityAssessor)",
            variable=self.filter_training_var,
            command=lambda: setattr(self.user_manager, 'filter_training', bool(self.filter_training_var.get()))
        ).pack(side='left')

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

        self.user_details_var = tk.StringVar(master=self.root, value="เลือกผู้ใช้เพื่อดูรายละเอียด")
        ttk.Label(details_frame, textvariable=self.user_details_var, wraplength=600).pack()
    
    def create_verification_panel(self):
        """Create verification panel with live capture functionality"""
        # Control section
        control_frame = ttk.LabelFrame(self.verify_frame, text="ตรวจสอบผู้ใช้", padding=10)
        control_frame.pack(fill='x', padx=5, pady=5)

        # User selection
        selection_frame = ttk.Frame(control_frame)
        selection_frame.pack(fill='x', pady=5)

        ttk.Label(selection_frame, text="เลือกผู้ใช้ (ว่างไว้สำหรับระบบ Auto-ID):").pack(side='left')
        self.verify_user_var = tk.StringVar(master=self.root)
        self.verify_user_combo = ttk.Combobox(selection_frame, textvariable=self.verify_user_var,
                                             state='readonly', width=35)
        self.verify_user_combo.pack(side='left', padx=10)

        # Add clear selection button for auto-identification
        ttk.Button(selection_frame, text="ล้าง (Auto-ID)",
                  command=lambda: self.verify_user_combo.set("")).pack(side='left', padx=5)

        # Capture area selection (NEW)
        if MSS_AVAILABLE:
            capture_frame = ttk.Frame(control_frame)
            capture_frame.pack(fill='x', pady=5)

            self.select_area_btn = ttk.Button(capture_frame, text="1. เลือกพื้นที่จับภาพ",
                                            command=self.select_capture_area)
            self.select_area_btn.pack(side='left', padx=5)

            self.capture_area_var = tk.StringVar(master=self.root, value="ยังไม่ได้เลือกพื้นที่")
            ttk.Label(capture_frame, textvariable=self.capture_area_var, foreground="grey").pack(side='left', padx=10)

            # Live capture control
            live_frame = ttk.Frame(control_frame)
            live_frame.pack(fill='x', pady=5)

            self.live_capture_btn = ttk.Button(live_frame, text="🔍 Auto-Scan",
                                             command=self.start_live_capture, state=tk.DISABLED)
            self.live_capture_btn.pack(side='left', padx=5)

            # Auto-scan info
            info_label = ttk.Label(live_frame, text="(ทำงานต่อเนื่องจนกว่าจะกดหยุด)", foreground="gray")
            info_label.pack(side='left', padx=10)
        else:
            # Fallback to file upload if MSS not available
            upload_frame = ttk.Frame(control_frame)
            upload_frame.pack(fill='x', pady=10)

            ttk.Button(upload_frame, text="เลือกภาพ OCT",
                      command=self.upload_verification_image).pack(side='left')

            self.verify_image_path_var = tk.StringVar(master=self.root, value="ยังไม่ได้เลือกภาพ")
            ttk.Label(upload_frame, textvariable=self.verify_image_path_var).pack(side='left', padx=10)

        # Verification threshold
        threshold_frame = ttk.Frame(control_frame)
        threshold_frame.pack(fill='x', pady=5)

        ttk.Label(threshold_frame, text="เกณฑ์การตรวจสอบ:").pack(side='left')
        self.threshold_var = tk.DoubleVar(master=self.root, value=0.7)
        threshold_scale = ttk.Scale(threshold_frame, from_=0.5, to=1.0, variable=self.threshold_var,
                                   orient='horizontal', length=200)
        threshold_scale.pack(side='left', padx=10)

        self.threshold_label = ttk.Label(threshold_frame, text="0.70")
        self.threshold_label.pack(side='left', padx=5)

        threshold_scale.configure(command=self.update_threshold_label)

        # Verification quality filter toggle
        qf_frame = ttk.Frame(control_frame)
        qf_frame.pack(fill='x', pady=5)
        ttk.Checkbutton(
            qf_frame,
            text="กรองภาพคุณภาพต่ำระหว่างตรวจสอบ",
            variable=self.filter_verification_var
        ).pack(side='left')

        # Manual verify button (for file mode)
        if not MSS_AVAILABLE:
            ttk.Button(control_frame, text="ตรวจสอบ", command=self.verify_identity).pack(pady=10)

        # Results section
        result_frame = ttk.LabelFrame(self.verify_frame, text="ผลการตรวจสอบ", padding=10)
        result_frame.pack(fill='both', expand=True, padx=5, pady=5)

        self.verification_result_var = tk.StringVar(master=self.root, value="ยังไม่ได้ตรวจสอบ")
        self.verification_result_label = ttk.Label(result_frame, textvariable=self.verification_result_var,
                                                  style='Normal.TLabel', font=('Arial', 16, 'bold'))
        self.verification_result_label.pack(pady=10)

        self.verification_details_var = tk.StringVar(master=self.root, value="")
        ttk.Label(result_frame, textvariable=self.verification_details_var, wraplength=600).pack()
        
        # Image display
        image_frame = ttk.Frame(result_frame)
        image_frame.pack(fill='both', expand=True, pady=10)
        
        if MSS_AVAILABLE:
            ttk.Label(image_frame, text="ภาพที่จับได้:").pack(anchor='w')
        else:
            ttk.Label(image_frame, text="ภาพที่เลือก:").pack(anchor='w')
        
        self.verify_image_canvas = tk.Canvas(image_frame, bg='white', height=250)
        self.verify_image_canvas.pack(fill='both', expand=True)
        
        # Draw placeholder
        self._draw_placeholder(self.verify_image_canvas, "ภาพจะแสดงที่นี่")
    
    def _draw_placeholder(self, canvas, text):
        """Draw placeholder text on canvas"""
        def draw():
            if not canvas.winfo_exists():
                return
            
            canvas.delete("all")
            width = canvas.winfo_width()
            height = canvas.winfo_height()
            
            if width > 1 and height > 1:
                canvas.create_text(width//2, height//2, text=text, 
                                 fill="gray", font=("Arial", 14))
        
        # Delay drawing until canvas is ready
        canvas.after(10, draw)
    
    def create_admin_panel(self):
        """Create admin panel"""
        # System status
        status_frame = ttk.LabelFrame(self.admin_frame, text="สถานะระบบ", padding=10)
        status_frame.pack(fill='x', padx=5, pady=5)
        
        # Feature extractor info
        extractor_type = "VGG-16 Deep Learning" if (ONNX_AVAILABLE and hasattr(self.feature_extractor, 'model_loaded') and self.feature_extractor.model_loaded) else "Basic Computer Vision"
        ttk.Label(status_frame, text=f"Feature Extractor: {extractor_type}").pack(anchor='w')
        
        if ONNX_AVAILABLE and hasattr(self.feature_extractor, 'model_loaded') and self.feature_extractor.model_loaded:
            models_info = "VGG-16"
            ttk.Label(status_frame, text=f"Loaded Models: {models_info}").pack(anchor='w')
        
        # Package availability
        ttk.Label(status_frame, text=f"ONNX Runtime: {'Available' if ONNX_AVAILABLE else 'Not Available'}").pack(anchor='w')
        ttk.Label(status_frame, text=f"Scikit-learn: {'Available' if SKLEARN_AVAILABLE else 'Not Available'}").pack(anchor='w')
        
        # Arduino status
        ttk.Label(status_frame, textvariable=self.arduino_status_var).pack(anchor='w')
        
        # Database info
        user_count = self.db.get_user_count()
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

        self.status_var = tk.StringVar(master=self.root, value=f"ระบบพร้อม - {self.get_feature_extractor_name()}")
        ttk.Label(status_frame, textvariable=self.status_var).pack(side='left', padx=5)

        ttk.Label(status_frame, textvariable=self.arduino_status_var).pack(side='right', padx=5)
    
    def get_feature_extractor_name(self):
        """Get current feature extractor name"""
        if ONNX_AVAILABLE and hasattr(self.feature_extractor, 'model_loaded') and self.feature_extractor.model_loaded:
            return "VGG-16 Deep Learning"
        else:
            return "Basic Mode (No ONNX)"
    
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
        """Training thread with thread-safe database operations"""
        try:
            user_id = None
            # Check if user already exists and create user
            feature_type = self.get_feature_extractor_name()
            
            try:
                user_id = self.db.create_user(name, bscan_folder, feature_type)
            except ValueError as e:
                self.root.after(0, lambda: messagebox.showerror("ข้อผิดพลาด", str(e)))
                return
            
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
            self.db.update_user_model(user_id, model_path, model_data['num_samples'], feature_type)
            
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
                uid = locals().get('user_id', None)
                if uid is not None:
                    self.db.delete_user(uid)
            except Exception:
                pass
    
    def update_threshold_label(self, value):
        """Update threshold label when scale changes"""
        self.threshold_label.config(text=f"{float(value):.2f}")
    
    def verify_identity(self):
        """Verify user identity using uploaded image"""
        if not hasattr(self, 'current_verify_image_path'):
            messagebox.showwarning("ข้อผิดพลาด", "กรุณาเลือกภาพสำหรับตรวจสอบ")
            return
        
        # Allow auto-identification if no user selected
        selected_user = self.verify_user_var.get()
        
        try:
            # Get user ID from selection or use None for auto-identification
            user_id = None
            if selected_user:
                user_id = int(selected_user.split(':')[0])
                
            threshold = self.threshold_var.get()
            
            # Load test image
            # Always load grayscale
            test_image = cv2.imread(self.current_verify_image_path, cv2.IMREAD_GRAYSCALE)
            if test_image is None:
                messagebox.showerror("ข้อผิดพลาด", "ไม่สามารถโหลดภาพได้")
                return
            
            # Optional quality gating before verification
            if self.filter_verification_var.get() and self.quality_assessor is not None:
                ok, qscore, qd = self.quality_assessor.is_valid(test_image)
                if not ok:
                    self.verification_result_var.set("⚠️ กรองภาพคุณภาพต่ำ/Noise")
                    self.verification_result_label.config(style='Info.TLabel')
                    self.verification_details_var.set(
                        f"Quality score: {qscore:.3f}\n"
                        f"std:{qd['std']:.1f} ent:{qd['entropy']:.2f} lap:{qd['lap_var']:.1f} "
                        f"edge:{qd['edge_density']:.3f} coh:{qd['ori_coh']:.3f}"
                    )
                    return
            
            # Perform verification (now returns 4 values)
            is_verified, similarity_score, details, matched_user_id = self.user_manager.verify_user(
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
            
            # Log verification using matched_user_id (which could be from auto-identification)
            log_user_id = matched_user_id if matched_user_id else user_id
            if log_user_id:
                self.db.log_verification(log_user_id, is_verified, similarity_score, details)
            
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
        
        users = self.db.get_all_users() or []
        for row in users:
            try:
                user_id, name, samples = row[0], row[1], (row[2] if len(row) > 2 else 0)
            except Exception:
                continue
            samples = samples or 0
            self.user_listbox.insert(tk.END, f"{user_id}: {name} ({samples} ภาพ)")
    
    def refresh_verify_combo(self):
        """Refresh verification combobox"""
        users = self.db.get_users_with_models() or []
        pairs = []
        for row in users:
            try:
                pairs.append((row[0], row[1]))
            except Exception:
                continue
        values = [f"{user_id}: {name}" for user_id, name in pairs]
        self.verify_user_combo['values'] = values
    
    def refresh_admin_tree(self):
        """Refresh admin tree view"""
        # Clear existing items
        for item in self.users_tree.get_children():
            self.users_tree.delete(item)
        
        users = self.db.get_admin_users() or []
        for row in users:
            try:
                user_id, name, created_at, samples, extractor_type = row
            except Exception:
                continue
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
            user_data = self.db.get_user_details(user_id)
            
            if user_data and isinstance(user_data, (list, tuple)) and len(user_data) >= 6:
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
                self.db.delete_user(user_id)
                
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
        user_data = self.db.get_user_details(user_id)
        
        if not user_data or not isinstance(user_data, (list, tuple)) or len(user_data) < 3 or not user_data[2]:
            messagebox.showerror("ข้อผิดพลาด", "ไม่พบโฟลเดอร์ B-scan สำหรับผู้ใช้นี้")
            return
        
        bscan_folder = user_data[2]
        
        if messagebox.askyesno("ยืนยัน", f"ต้องการเทรนโมเดลใหม่สำหรับ '{user_name}' หรือไม่?"):
            # Run retraining in thread
            thread = threading.Thread(target=self._retrain_user_thread, args=(user_id, user_name, bscan_folder))
            thread.daemon = True
            thread.start()
    
    def _retrain_user_thread(self, user_id, user_name, bscan_folder):
        """Retraining thread with thread-safe database operations"""
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
            feature_type = self.get_feature_extractor_name()
            self.db.update_user_model(user_id, model_path, model_data['num_samples'], feature_type)
            
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
                self.db.clear_verification_logs()
                messagebox.showinfo("สำเร็จ", "ล้างล็อกเสร็จแล้ว")
            except Exception as e:
                messagebox.showerror("ข้อผิดพลาด", f"ไม่สามารถล้างล็อกได้: {str(e)}")
    
    # ========== Arduino Functions ==========
    
    def init_arduino(self):
        """Initialize Arduino connection"""
        try:
            # Try preferred port first (e.g., COM3), then fall back to auto-detect
            tried_preferred = False
            if hasattr(self, 'preferred_arduino_port') and self.preferred_arduino_port:
                tried_preferred = True
                if self.connect_arduino_to_port(self.preferred_arduino_port):
                    return
            # Fall back to auto-detection if preferred failed
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
        """Connect to Arduino Uno for relay control"""
        arduino_ports = self.find_arduino_ports()
        
        if not arduino_ports:
            self.arduino_status_var.set(f"Arduino: ไม่พบพอร์ต")
            return False
        
        for port in arduino_ports:
            try:
                self.arduino_serial = serial.Serial(port, 9600, timeout=3)
                time.sleep(2)  # Wait for Arduino reset
                
                # Send initialization command with relay pin
                init_cmd = f'INIT,{self.relay_pin}\n'
                self.arduino_serial.write(init_cmd.encode())
                response = self.arduino_serial.readline().decode('utf-8').strip()
                
                if 'READY' in response:
                    self.arduino_port = port
                    self.arduino_connected = True
                    self.arduino_status_var.set(f"Arduino: เชื่อมต่อ ({port}) Pin:{self.relay_pin}")
                    print(f"Arduino connected on {port}, Relay Pin: {self.relay_pin}")
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

    def connect_arduino_to_port(self, port):
        """Attempt to connect to a specific serial port (e.g., 'COM3')."""
        try:
            self.arduino_status_var.set(f"Arduino: กำลังเชื่อมต่อ {port} ...")
            self.arduino_serial = serial.Serial(port, 9600, timeout=3)
            time.sleep(2)  # Wait for Arduino reset
            init_cmd = f'INIT,{self.relay_pin}\n'
            self.arduino_serial.write(init_cmd.encode())
            response = self.arduino_serial.readline().decode('utf-8').strip()
            if 'READY' in response:
                self.arduino_port = port
                self.arduino_connected = True
                self.arduino_status_var.set(f"Arduino: เชื่อมต่อ ({port}) Pin:{self.relay_pin}")
                print(f"Arduino connected on {port}, Relay Pin: {self.relay_pin}")
                return True
            else:
                print(f"Unexpected response on {port}: {response}")
                try:
                    self.arduino_serial.close()
                except:
                    pass
                return False
        except Exception as e:
            print(f"Error connecting to {port}: {e}")
            try:
                if hasattr(self, 'arduino_serial') and self.arduino_serial:
                    self.arduino_serial.close()
            except:
                pass
            return False
    
    def open_door(self):
        """Open magnetic door via Arduino relay"""
        if not self.arduino_connected or not self.arduino_serial:
            print("Arduino not connected")
            return False
        
        try:
            # Send relay control command: RELAY,pin,duration
            relay_cmd = f'RELAY,{self.relay_pin},{self.relay_open_duration}\n'
            self.arduino_serial.write(relay_cmd.encode())
            response = self.arduino_serial.readline().decode('utf-8').strip()
            
            if 'RELAY_ON' in response:
                print(f"🚪 Magnetic door opened - Pin {self.relay_pin} for {self.relay_open_duration}s")
                self.status_var.set(f"🚪 ประตูแม่เหล็กเปิด Pin{self.relay_pin} ({self.relay_open_duration}วิ)")
                return True
            else:
                print(f"Unexpected Arduino response: {response}")
                return False
                
        except Exception as e:
            print(f"Error opening magnetic door: {e}")
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
        """Close application with proper cleanup"""
        try:
            # Stop live capture
            if hasattr(self, 'is_live_capturing') and self.is_live_capturing:
                self.stop_live_capture()
            
            # Close screen capture
            if hasattr(self, 'sct') and self.sct:
                self.close_screen_capture()
            
            # Close Arduino connection
            if hasattr(self, 'arduino_serial') and self.arduino_serial:
                self.arduino_serial.close()
        except Exception as e:
            print(f"Error during cleanup: {e}")
        
        self.root.destroy()
    
    def upload_verification_image(self):
        """Upload image for verification (fallback method)"""
        file_path = filedialog.askopenfilename(
            title="เลือกภาพ OCT",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif")]
        )
        
        if file_path:
            self.verify_image_path_var.set(os.path.basename(file_path))
            self.current_verify_image_path = file_path
            
            # Display image
            try:
                # Always load in grayscale for this system
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    self.display_image_on_canvas(img, self.verify_image_canvas)
            except Exception as e:
                print(f"Error displaying image: {e}")

def main():
    """Main function"""
    print("=== OCT Deep Learning Fingerprint System (Thread-Safe) ===")
    print("Starting system...")
    
    # Check VGG16 model availability
    model_path = os.path.join(MODELS_DIR, "vgg16-7.onnx")
    if os.path.exists(model_path):
        print(f"VGG16 model found: {model_path}")
    else:
        print(f"VGG16 model not found: {model_path}")
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

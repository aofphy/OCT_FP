import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageGrab
import sqlite3
from datetime import datetime
from skimage.feature import hog
from skimage import exposure

class ScreenCaptureArea:
    """Class for capturing a selected area of the screen"""
    def __init__(self, parent):
        self.parent = parent
        self.root = tk.Toplevel(parent)
        self.root.attributes('-fullscreen', True)
        self.root.attributes('-alpha', 0.3)
        self.root.configure(bg='grey')
        
        # Variables for area selection
        self.start_x = None
        self.start_y = None
        self.current_x = None
        self.current_y = None
        self.selection_rect = None
        self.canvas = tk.Canvas(self.root, bg='grey', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bind events
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        
        # Instructions text
        self.canvas.create_text(
            self.root.winfo_screenwidth() // 2, 
            self.root.winfo_screenheight() // 2, 
            text="คลิกและลากเพื่อเลือกพื้นที่ - กด ESC เพื่อยกเลิก", 
            font=('Arial', 18), 
            fill='white'
        )
        
        # ESC key to cancel
        self.root.bind("<Escape>", self.cancel)
        
        # Result
        self.selected_area = None
        
    def on_button_press(self, event):
        # Save mouse drag start position
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        
        # Create rectangle if not yet exist
        if self.selection_rect is None:
            self.selection_rect = self.canvas.create_rectangle(
                self.start_x, self.start_y, self.start_x, self.start_y, 
                outline='red', width=2
            )
    
    def on_mouse_drag(self, event):
        self.current_x = self.canvas.canvasx(event.x)
        self.current_y = self.canvas.canvasy(event.y)
        
        # Update rectangle size
        self.canvas.coords(
            self.selection_rect, 
            self.start_x, self.start_y, 
            self.current_x, self.current_y
        )
    
    def on_button_release(self, event):
        # Get the final coordinates
        self.current_x = self.canvas.canvasx(event.x)
        self.current_y = self.canvas.canvasy(event.y)
        
        # Ensure coordinates are correctly ordered (top-left to bottom-right)
        x1 = min(self.start_x, self.current_x)
        y1 = min(self.start_y, self.current_y)
        x2 = max(self.start_x, self.current_x)
        y2 = max(self.start_y, self.current_y)
        
        # Save the selected area
        self.selected_area = (int(x1), int(y1), int(x2), int(y2))
        
        # Close the window
        self.root.destroy()
    
    def cancel(self, event=None):
        self.selected_area = None
        self.root.destroy()

class FingerprintSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("ระบบตรวจสอบลายนิ้วมือด้วย OCT")
        self.root.geometry("1200x700")
        
        # Database initialization
        self.conn = sqlite3.connect('fingerprint_db.sqlite')
        self.create_tables()
        
        # Current user and scan data
        self.current_user = None
        self.current_scan = None
        self.last_comparison_result = None
        
        # Screen capture mode
        self.capture_mode = "area"  # Can be "area" or "window"
        
        self.create_ui()
    
    def create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS fingerprints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            image_path TEXT NOT NULL,
            scan_date TEXT NOT NULL,
            hog_features BLOB,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        ''')
        self.conn.commit()
    
    def create_ui(self):
        # Create tab control
        self.tab_control = ttk.Notebook(self.root)
        
        # Create tabs
        self.register_tab = ttk.Frame(self.tab_control)
        self.verify_tab = ttk.Frame(self.tab_control)
        self.admin_tab = ttk.Frame(self.tab_control)
        
        self.tab_control.add(self.register_tab, text="ลงทะเบียนผู้ใช้")
        self.tab_control.add(self.verify_tab, text="ตรวจสอบ")
        self.tab_control.add(self.admin_tab, text="ผู้ดูแลระบบ")
        
        self.tab_control.pack(expand=1, fill="both")
        
        # Set up each tab
        self.setup_register_tab()
        self.setup_verify_tab()
        self.setup_admin_tab()
    
    def setup_register_tab(self):
        # Left frame for user details
        left_frame = ttk.LabelFrame(self.register_tab, text="ข้อมูลผู้ใช้")
        left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        ttk.Label(left_frame, text="ชื่อผู้ใช้:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.username_entry = ttk.Entry(left_frame, width=30)
        self.username_entry.grid(row=0, column=1, padx=10, pady=10)
        
        ttk.Button(left_frame, text="สร้างผู้ใช้ใหม่", command=self.create_user).grid(row=1, column=0, columnspan=2, padx=10, pady=10)
        
        # User list
        ttk.Label(left_frame, text="ผู้ใช้ที่มีอยู่:").grid(row=2, column=0, padx=10, pady=(20,5), sticky="w")
        
        self.user_listbox = tk.Listbox(left_frame, width=40, height=15)
        self.user_listbox.grid(row=3, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")
        self.user_listbox.bind('<<ListboxSelect>>', self.on_user_select)
        
        # Populate user list
        self.refresh_user_list()
        
        # Right frame for fingerprint scan
        right_frame = ttk.LabelFrame(self.register_tab, text="สแกนลายนิ้วมือ")
        right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        # Scan controls
        scan_controls_frame = ttk.Frame(right_frame)
        scan_controls_frame.pack(pady=10, fill="x")
        
        # Capture mode selection
        ttk.Label(scan_controls_frame, text="โหมดจับภาพ:").grid(row=0, column=0, padx=5, pady=5)
        self.capture_mode_var = tk.StringVar(value="area")
        ttk.Radiobutton(scan_controls_frame, text="เลือกพื้นที่", variable=self.capture_mode_var, value="area").grid(row=0, column=1, padx=5, pady=5)
        ttk.Radiobutton(scan_controls_frame, text="ทั้งหน้าจอ", variable=self.capture_mode_var, value="full").grid(row=0, column=2, padx=5, pady=5)
        
        ttk.Button(scan_controls_frame, text="จับภาพหน้าจอ", command=self.start_screen_capture).grid(row=0, column=3, padx=5, pady=5)
        ttk.Button(scan_controls_frame, text="บันทึกสแกน", command=self.save_scan).grid(row=0, column=4, padx=5, pady=5)
        
        # OCT parameter controls (optional, can be used for simulating OCT settings)
        param_frame = ttk.LabelFrame(right_frame, text="พารามิเตอร์ OCT (สำหรับจำลอง)")
        param_frame.pack(pady=10, fill="x")
        
        ttk.Label(param_frame, text="X-scan (mm):").grid(row=0, column=0, padx=5, pady=5)
        self.x_scan_entry = ttk.Entry(param_frame, width=10)
        self.x_scan_entry.grid(row=0, column=1, padx=5, pady=5)
        self.x_scan_entry.insert(0, "0.00")
        
        ttk.Label(param_frame, text="Y-scan (mm):").grid(row=0, column=2, padx=5, pady=5)
        self.y_scan_entry = ttk.Entry(param_frame, width=10)
        self.y_scan_entry.grid(row=0, column=3, padx=5, pady=5)
        self.y_scan_entry.insert(0, "0.00")
        
        ttk.Label(param_frame, text="ช่วงความลึก:").grid(row=1, column=0, padx=5, pady=5)
        self.depth_entry = ttk.Entry(param_frame, width=10)
        self.depth_entry.grid(row=1, column=1, padx=5, pady=5)
        self.depth_entry.insert(0, "400")
        
        # Scan display
        self.scan_canvas = tk.Canvas(right_frame, bg="white", width=600, height=400, bd=2, relief="sunken")
        self.scan_canvas.pack(padx=10, pady=10, fill="both", expand=True)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("พร้อมใช้งาน")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w")
        status_bar.pack(side="bottom", fill="x")
        
        # Configure grid weights
        self.register_tab.columnconfigure(0, weight=1)
        self.register_tab.columnconfigure(1, weight=2)
        self.register_tab.rowconfigure(0, weight=1)
    
    def setup_verify_tab(self):
        # Left frame for scan
        left_frame = ttk.LabelFrame(self.verify_tab, text="สแกนปัจจุบัน")
        left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        # Scan controls
        scan_controls_frame = ttk.Frame(left_frame)
        scan_controls_frame.pack(pady=10, fill="x")
        
        # Capture mode for verification
        ttk.Label(scan_controls_frame, text="โหมดจับภาพ:").grid(row=0, column=0, padx=5, pady=5)
        self.verify_capture_mode_var = tk.StringVar(value="area")
        ttk.Radiobutton(scan_controls_frame, text="เลือกพื้นที่", variable=self.verify_capture_mode_var, value="area").grid(row=0, column=1, padx=5, pady=5)
        ttk.Radiobutton(scan_controls_frame, text="ทั้งหน้าจอ", variable=self.verify_capture_mode_var, value="full").grid(row=0, column=2, padx=5, pady=5)
        
        ttk.Button(scan_controls_frame, text="จับภาพสำหรับตรวจสอบ", command=self.start_verification_capture).grid(row=0, column=3, padx=5, pady=5)
        
        # Scan display
        self.verify_canvas = tk.Canvas(left_frame, bg="white", width=600, height=400, bd=2, relief="sunken")
        self.verify_canvas.pack(padx=10, pady=10, fill="both", expand=True)
        
        # Right frame for results
        right_frame = ttk.LabelFrame(self.verify_tab, text="ผลการตรวจสอบ")
        right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        # Results display
        self.results_frame = ttk.Frame(right_frame)
        self.results_frame.pack(padx=10, pady=10, fill="both", expand=True)
        
        ttk.Label(self.results_frame, text="ผู้ใช้:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.result_user_var = tk.StringVar()
        ttk.Label(self.results_frame, textvariable=self.result_user_var, font=("Arial", 12, "bold")).grid(row=0, column=1, padx=10, pady=10, sticky="w")
        
        ttk.Label(self.results_frame, text="คะแนนความเหมือน:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.match_score_var = tk.StringVar()
        ttk.Label(self.results_frame, textvariable=self.match_score_var, font=("Arial", 12, "bold")).grid(row=1, column=1, padx=10, pady=10, sticky="w")
        
        ttk.Label(self.results_frame, text="สถานะ:").grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.verification_status_var = tk.StringVar()
        ttk.Label(self.results_frame, textvariable=self.verification_status_var, font=("Arial", 14, "bold")).grid(row=2, column=1, padx=10, pady=10, sticky="w")
        
        # HOG visualization canvas
        ttk.Label(self.results_frame, text="การเปรียบเทียบ HOG:").grid(row=3, column=0, columnspan=2, padx=10, pady=5, sticky="w")
        self.hog_canvas = tk.Canvas(self.results_frame, bg="white", width=500, height=200, bd=2, relief="sunken")
        self.hog_canvas.grid(row=4, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")
        
        ttk.Button(self.results_frame, text="ตรวจสอบ", command=self.verify_fingerprint).grid(row=5, column=0, columnspan=2, padx=10, pady=20)
        
        # Configure grid weights
        self.verify_tab.columnconfigure(0, weight=1)
        self.verify_tab.columnconfigure(1, weight=1)
        self.verify_tab.rowconfigure(0, weight=1)
        
        # Configure results frame grid weights
        self.results_frame.rowconfigure(4, weight=1)
        self.results_frame.columnconfigure(0, weight=1)
        self.results_frame.columnconfigure(1, weight=1)
    
    def setup_admin_tab(self):
        # Admin panel for managing users and fingerprints
        controls_frame = ttk.Frame(self.admin_tab)
        controls_frame.pack(padx=10, pady=10, fill="x")
        
        ttk.Button(controls_frame, text="รีเฟรชฐานข้อมูล", command=self.refresh_user_list).pack(side="left", padx=5)
        ttk.Button(controls_frame, text="ลบผู้ใช้ที่เลือก", command=self.delete_user).pack(side="left", padx=5)
        ttk.Button(controls_frame, text="ส่งออกฐานข้อมูล", command=self.export_database).pack(side="left", padx=5)
        ttk.Button(controls_frame, text="นำเข้าฐานข้อมูล", command=self.import_database).pack(side="left", padx=5)
        
        # Database view
        data_frame = ttk.LabelFrame(self.admin_tab, text="ข้อมูลในฐานข้อมูล")
        data_frame.pack(padx=10, pady=10, fill="both", expand=True)
        
        # Users table
        users_frame = ttk.LabelFrame(data_frame, text="ผู้ใช้")
        users_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        self.users_treeview = ttk.Treeview(users_frame, columns=("id", "name", "created_at"), show="headings")
        self.users_treeview.heading("id", text="ID")
        self.users_treeview.heading("name", text="ชื่อ")
        self.users_treeview.heading("created_at", text="สร้างเมื่อ")
        
        self.users_treeview.column("id", width=50)
        self.users_treeview.column("name", width=200)
        self.users_treeview.column("created_at", width=150)
        
        self.users_treeview.pack(fill="both", expand=True)
        self.users_treeview.bind('<<TreeviewSelect>>', self.on_admin_user_select)
        
        # Fingerprints table
        fp_frame = ttk.LabelFrame(data_frame, text="ลายนิ้วมือ")
        fp_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        self.fp_treeview = ttk.Treeview(fp_frame, columns=("id", "user_id", "path", "date"), show="headings")
        self.fp_treeview.heading("id", text="ID")
        self.fp_treeview.heading("user_id", text="รหัสผู้ใช้")
        self.fp_treeview.heading("path", text="ที่เก็บไฟล์")
        self.fp_treeview.heading("date", text="วันที่สแกน")
        
        self.fp_treeview.column("id", width=50)
        self.fp_treeview.column("user_id", width=50)
        self.fp_treeview.column("path", width=250)
        self.fp_treeview.column("date", width=150)
        
        self.fp_treeview.pack(fill="both", expand=True)
        
        # Configure grid weights
        data_frame.columnconfigure(0, weight=1)
        data_frame.columnconfigure(1, weight=1)
        data_frame.rowconfigure(0, weight=1)
        
        # Initial population
        self.refresh_admin_view()
    
    def refresh_user_list(self):
        # Clear the current list
        self.user_listbox.delete(0, tk.END)
        
        # Get users from database
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, name FROM users ORDER BY name")
        users = cursor.fetchall()
        
        # Populate listbox
        for user_id, name in users:
            self.user_listbox.insert(tk.END, f"{user_id}: {name}")
    
    def refresh_admin_view(self):
        # Clear current views
        for item in self.users_treeview.get_children():
            self.users_treeview.delete(item)
        
        for item in self.fp_treeview.get_children():
            self.fp_treeview.delete(item)
        
        # Get users
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, name, created_at FROM users ORDER BY id")
        users = cursor.fetchall()
        
        for user in users:
            self.users_treeview.insert("", "end", values=user)
        
        # Get fingerprints
        cursor.execute("SELECT id, user_id, image_path, scan_date FROM fingerprints ORDER BY user_id, scan_date")
        fingerprints = cursor.fetchall()
        
        for fp in fingerprints:
            self.fp_treeview.insert("", "end", values=fp)
    
    def on_user_select(self, event):
        # Get selected user
        selection = self.user_listbox.curselection()
        if not selection:
            return
        
        item = self.user_listbox.get(selection[0])
        user_id = int(item.split(":")[0])
        
        # Set current user
        self.current_user = user_id
        self.status_var.set(f"เลือกผู้ใช้: {item}")
        
        # Load user's fingerprints
        self.load_user_fingerprints(user_id)
    
    def on_admin_user_select(self, event):
        selection = self.users_treeview.selection()
        if not selection:
            return
        
        item = self.users_treeview.item(selection[0])
        user_id = item['values'][0]
        
        # Clear fingerprint view
        for item in self.fp_treeview.get_children():
            self.fp_treeview.delete(item)
        
        # Load fingerprints for this user
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, user_id, image_path, scan_date FROM fingerprints WHERE user_id = ? ORDER BY scan_date", (user_id,))
        fingerprints = cursor.fetchall()
        
        for fp in fingerprints:
            self.fp_treeview.insert("", "end", values=fp)
    
    def load_user_fingerprints(self, user_id):
        # Show latest scan if available
        cursor = self.conn.cursor()
        cursor.execute("SELECT image_path FROM fingerprints WHERE user_id = ? ORDER BY scan_date DESC LIMIT 1", (user_id,))
        result = cursor.fetchone()
        
        if result:
            image_path = result[0]
            try:
                # Load and display the fingerprint image
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                self.display_scan(img, canvas=self.scan_canvas)
                self.status_var.set(f"โหลดลายนิ้วมือสำหรับผู้ใช้ {user_id} จาก {image_path}")
            except Exception as e:
                self.status_var.set(f"เกิดข้อผิดพลาดในการโหลดลายนิ้วมือ: {str(e)}")
    
    def create_user(self):
        name = self.username_entry.get().strip()
        if not name:
            messagebox.showerror("ข้อผิดพลาด", "กรุณาใส่ชื่อผู้ใช้")
            return
        
        # Insert user into database
        cursor = self.conn.cursor()
        cursor.execute("INSERT INTO users (name, created_at) VALUES (?, ?)", 
                       (name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        self.conn.commit()
        
        # Clear entry and refresh list
        self.username_entry.delete(0, tk.END)
        self.refresh_user_list()
        self.refresh_admin_view()
        
        messagebox.showinfo("สำเร็จ", f"สร้างผู้ใช้ '{name}' เรียบร้อยแล้ว")
    
    def delete_user(self):
        selection = self.users_treeview.selection()
        if not selection:
            messagebox.showwarning("ต้องเลือกก่อน", "กรุณาเลือกผู้ใช้ที่ต้องการลบ")
            return
        
        item = self.users_treeview.item(selection[0])
        user_id = item['values'][0]
        user_name = item['values'][1]
        
        # Confirm deletion
        confirm = messagebox.askyesno("ยืนยันการลบ", 
                                      f"คุณแน่ใจหรือไม่ว่าต้องการลบผู้ใช้ '{user_name}' และลายนิ้วมือทั้งหมดที่เกี่ยวข้อง?")
        if not confirm:
            return
        
        # Delete fingerprints first (foreign key constraint)
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM fingerprints WHERE user_id = ?", (user_id,))
        cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
        self.conn.commit()
        
        # Refresh views
        self.refresh_user_list()
        self.refresh_admin_view()
        
        messagebox.showinfo("สำเร็จ", f"ลบผู้ใช้ '{user_name}' เรียบร้อยแล้ว")
    
    def start_screen_capture(self):
        # Check if user is selected
        if self.current_user is None:
            messagebox.showwarning("ต้องเลือกผู้ใช้", "กรุณาเลือกผู้ใช้ก่อน")
            return
        
        # Minimize the application window to make it easier to capture
        self.root.iconify()
        
        # Wait a moment for window to minimize
        self.root.after(500, self.perform_screen_capture)
    
    def perform_screen_capture(self):
        capture_mode = self.capture_mode_var.get()
        
        try:
            if capture_mode == "area":
                # Create a screen capture selection tool
                self.root.withdraw()  # Hide main window completely
                
                # Wait a moment before showing selection window
                self.root.after(200, self._show_area_selector)
            else:  # Full screen
                # Capture entire screen
                screenshot = ImageGrab.grab()
                img_np = np.array(screenshot.convert('L'))  # Convert to grayscale
                self.current_scan = img_np
                self.display_scan(img_np, canvas=self.scan_canvas)
                self.status_var.set("จับภาพหน้าจอเรียบร้อยแล้ว")
                
                # Restore main window
                self.root.deiconify()
        except Exception as e:
            self.root.deiconify()
            messagebox.showerror("ข้อผิดพลาด", f"เกิดข้อผิดพลาดในการจับภาพหน้าจอ: {str(e)}")
    
    def _show_area_selector(self):
        try:
            # Create screen capture area selector
            selector = ScreenCaptureArea(self.root)
            
            # Wait for the selector window to close
            self.root.wait_window(selector.root)
            
            # Process selection
            if selector.selected_area:
                x1, y1, x2, y2 = selector.selected_area
                
                # Capture the selected area
                screenshot = ImageGrab.grab(bbox=(x1, y1, x2, y2))
                img_np = np.array(screenshot.convert('L'))  # Convert to grayscale
                
                # Apply preprocessing
                img_np = self.preprocess_fingerprint(img_np)
                
                self.current_scan = img_np
                self.display_scan(img_np, canvas=self.scan_canvas)
                self.status_var.set(f"จับภาพพื้นที่ที่เลือกเรียบร้อยแล้ว ({x2-x1}x{y2-y1} พิกเซล)")
        except Exception as e:
            messagebox.showerror("ข้อผิดพลาด", f"เกิดข้อผิดพลาดในการจับภาพหน้าจอ: {str(e)}")
        finally:
            # Restore main window
            self.root.deiconify()
    
    def start_verification_capture(self):
        # Minimize the application window to make it easier to capture
        self.root.iconify()
        
        # Wait a moment for window to minimize
        self.root.after(500, self.perform_verification_capture)
    
    def perform_verification_capture(self):
        capture_mode = self.verify_capture_mode_var.get()
        
        try:
            if capture_mode == "area":
                # Create a screen capture selection tool
                self.root.withdraw()  # Hide main window completely
                
                # Wait a moment before showing selection window
                self.root.after(200, self._show_verification_area_selector)
            else:  # Full screen
                # Capture entire screen
                screenshot = ImageGrab.grab()
                img_np = np.array(screenshot.convert('L'))  # Convert to grayscale
                self.current_scan = img_np
                self.display_scan(img_np, canvas=self.verify_canvas)
                self.status_var.set("จับภาพหน้าจอสำหรับตรวจสอบเรียบร้อยแล้ว")
                
                # Restore main window
                self.root.deiconify()
        except Exception as e:
            self.root.deiconify()
            messagebox.showerror("ข้อผิดพลาด", f"เกิดข้อผิดพลาดในการจับภาพหน้าจอ: {str(e)}")
    
    def _show_verification_area_selector(self):
        try:
            # Create screen capture area selector
            selector = ScreenCaptureArea(self.root)
            
            # Wait for the selector window to close
            self.root.wait_window(selector.root)
            
            # Process selection
            if selector.selected_area:
                x1, y1, x2, y2 = selector.selected_area
                
                # Capture the selected area
                screenshot = ImageGrab.grab(bbox=(x1, y1, x2, y2))
                img_np = np.array(screenshot.convert('L'))  # Convert to grayscale
                
                # Apply preprocessing
                img_np = self.preprocess_fingerprint(img_np)
                
                self.current_scan = img_np
                self.display_scan(img_np, canvas=self.verify_canvas)
                self.status_var.set(f"จับภาพพื้นที่ที่เลือกสำหรับตรวจสอบเรียบร้อยแล้ว ({x2-x1}x{y2-y1} พิกเซล)")
        except Exception as e:
            messagebox.showerror("ข้อผิดพลาด", f"เกิดข้อผิดพลาดในการจับภาพหน้าจอ: {str(e)}")
        finally:
            # Restore main window
            self.root.deiconify()
    
    def preprocess_fingerprint(self, img):
        """
        ประมวลผลภาพลายนิ้วมือเพื่อเพิ่มคุณภาพและความชัดเจน
        """
        # ปรับขนาดภาพให้เป็นมาตรฐาน
        img_resized = cv2.resize(img, (300, 300))
        
        # ปรับความเข้มของภาพ (Contrast enhancement)
        img_eq = cv2.equalizeHist(img_resized)
        
        # ลดสัญญาณรบกวน (Noise reduction)
        img_blur = cv2.GaussianBlur(img_eq, (5, 5), 0)
        
        # ปรับความคมชัด
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        img_sharp = cv2.filter2D(img_blur, -1, kernel)
        
        # ปรับเทรชโฮลด์ (Optional - อาจไม่จำเป็นในบางกรณี)
        _, img_thresh = cv2.threshold(img_sharp, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        return img_thresh
    
    def extract_hog_features(self, img):
        """
        สกัดคุณลักษณะ HOG (Histogram of Oriented Gradients) จากภาพ
        """
        # ปรับขนาดภาพเพื่อความสม่ำเสมอ
        img_resized = cv2.resize(img, (128, 128))
        
        # คำนวณคุณลักษณะ HOG
        # orientations: จำนวนช่อง (bins) ในฮิสโตแกรม
        # pixels_per_cell: ขนาดของเซลล์ในพิกเซล
        # cells_per_block: จำนวนเซลล์ในแต่ละบล็อก
        fd, hog_image = hog(
            img_resized, 
            orientations=9, 
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2), 
            visualize=True, 
            block_norm='L2-Hys'
        )
        
        # ทำให้ภาพ HOG มองเห็นได้ชัดเจนขึ้น
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        
        return fd, hog_image_rescaled
    
    def save_scan(self):
        if self.current_scan is None:
            messagebox.showwarning("ไม่มีภาพสแกน", "กรุณาทำการสแกนก่อน")
            return
        
        if self.current_user is None:
            messagebox.showwarning("ไม่ได้เลือกผู้ใช้", "กรุณาเลือกผู้ใช้ก่อน")
            return
        
        # สร้างไดเรกทอรีถ้ายังไม่มี
        os.makedirs("fingerprints", exist_ok=True)
        
        # บันทึกภาพสแกนลงไฟล์
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fingerprints/user_{self.current_user}_{timestamp}.png"
        
        cv2.imwrite(filename, self.current_scan)
        
        # สกัดคุณลักษณะ HOG และแปลงเป็น binary สำหรับเก็บในฐานข้อมูล
        hog_features, _ = self.extract_hog_features(self.current_scan)
        hog_features_binary = hog_features.tobytes()
        
        # บันทึกลงฐานข้อมูล
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO fingerprints (user_id, image_path, scan_date, hog_features) VALUES (?, ?, ?, ?)",
            (
                self.current_user, 
                filename, 
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                hog_features_binary
            )
        )
        self.conn.commit()
        
        self.status_var.set(f"บันทึกภาพสแกนไปยัง {filename}")
        self.refresh_admin_view()
        
        messagebox.showinfo("สำเร็จ", "บันทึกภาพสแกนลายนิ้วมือเรียบร้อยแล้ว")
    
    def verify_fingerprint(self):
        if self.current_scan is None:
            messagebox.showwarning("ไม่มีภาพสแกน", "กรุณาทำการสแกนยืนยันก่อน")
            return
        
        # สกัดคุณลักษณะ HOG จากภาพปัจจุบัน
        current_hog_features, current_hog_image = self.extract_hog_features(self.current_scan)
        
        # แสดงภาพ HOG ของภาพปัจจุบัน
        self.display_hog_image(current_hog_image)
        
        # ดึงข้อมูลลายนิ้วมือทั้งหมดจากฐานข้อมูล
        cursor = self.conn.cursor()
        cursor.execute("""
        SELECT f.id, f.user_id, f.image_path, f.hog_features, u.name 
        FROM fingerprints f
        JOIN users u ON f.user_id = u.id
        ORDER BY f.scan_date DESC
        """)
        fingerprints = cursor.fetchall()
        
        if not fingerprints:
            messagebox.showinfo("ไม่พบข้อมูล", "ไม่พบลายนิ้วมือในฐานข้อมูล")
            return
        
        # เปรียบเทียบด้วย HOG features
        best_match = None
        best_score = 0
        
        for fp_id, user_id, image_path, hog_features_binary, user_name in fingerprints:
            try:
                if hog_features_binary is None:
                    # สำหรับข้อมูลเก่าที่อาจไม่มี HOG features
                    stored_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    if stored_img is None:
                        continue
                    stored_hog_features, _ = self.extract_hog_features(stored_img)
                else:
                    # แปลง binary กลับเป็น numpy array
                    stored_hog_features = np.frombuffer(hog_features_binary, dtype=np.float64)
                
                # คำนวณค่าความเหมือน (เช่น cosine similarity)
                score = self.calculate_similarity(current_hog_features, stored_hog_features)
                
                if score > best_score:
                    best_score = score
                    best_match = (fp_id, user_id, user_name, score)
            except Exception as e:
                print(f"เกิดข้อผิดพลาดในการเปรียบเทียบกับ {image_path}: {str(e)}")
        
        # แสดงผลลัพธ์
        if best_match:
            fp_id, user_id, user_name, score = best_match
            
            self.result_user_var.set(user_name)
            self.match_score_var.set(f"{score:.2f}%")
            
            if score >= 85:
                self.verification_status_var.set("ยืนยันตัวตนสำเร็จ")
                self.results_frame.configure(background="#d4ffcc")  # สีเขียวอ่อน
            else:
                self.verification_status_var.set("ยืนยันตัวตนไม่สำเร็จ")
                self.results_frame.configure(background="#ffcccc")  # สีแดงอ่อน
                
            self.status_var.set(f"ผลลัพธ์ที่ดีที่สุด: ผู้ใช้ {user_name} ด้วยคะแนน {score:.2f}%")
            self.last_comparison_result = best_match
        else:
            self.result_user_var.set("ไม่พบการจับคู่")
            self.match_score_var.set("0.00%")
            self.verification_status_var.set("ยืนยันตัวตนไม่สำเร็จ")
            self.results_frame.configure(background="#ffcccc")  # สีแดงอ่อน
            self.status_var.set("ไม่พบการจับคู่ในฐานข้อมูล")
    
    def calculate_similarity(self, hog_features1, hog_features2):
        """
        คำนวณค่าความเหมือนระหว่างคุณลักษณะ HOG สองชุด โดยใช้ cosine similarity
        """
        # ตรวจสอบความยาวของ features
        if len(hog_features1) != len(hog_features2):
            # ปรับขนาดถ้าความยาวไม่เท่ากัน
            min_length = min(len(hog_features1), len(hog_features2))
            hog_features1 = hog_features1[:min_length]
            hog_features2 = hog_features2[:min_length]
        
        # คำนวณ cosine similarity
        dot_product = np.dot(hog_features1, hog_features2)
        norm1 = np.linalg.norm(hog_features1)
        norm2 = np.linalg.norm(hog_features2)
        
        if norm1 == 0 or norm2 == 0:
            return 0  # หลีกเลี่ยงการหารด้วยศูนย์
        
        cosine_similarity = dot_product / (norm1 * norm2)
        
        # แปลงเป็นเปอร์เซ็นต์
        percentage = (cosine_similarity + 1) / 2 * 100
        
        return percentage
    
    def display_hog_image(self, hog_image):
        """
        แสดงภาพ HOG บน canvas
        """
        # แปลงภาพ numpy เป็น PIL Image
        img_pil = Image.fromarray((hog_image * 255).astype(np.uint8))
        
        # ปรับขนาดให้พอดีกับ canvas
        canvas_width = self.hog_canvas.winfo_width()
        canvas_height = self.hog_canvas.winfo_height()
        
        # ตรวจสอบมิติที่ถูกต้อง
        if canvas_width <= 1:
            canvas_width = 500
        if canvas_height <= 1:
            canvas_height = 200
        
        # ปรับขนาดโดยรักษาอัตราส่วน
        img_width, img_height = img_pil.size
        scale = min(canvas_width/img_width, canvas_height/img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        img_pil = img_pil.resize((new_width, new_height), Image.LANCZOS)
        
        # แปลงเป็น PhotoImage
        img_tk = ImageTk.PhotoImage(img_pil)
        
        # เก็บอ้างอิงเพื่อป้องกันการเก็บขยะ
        self.hog_canvas.image = img_tk
        
        # ล้าง canvas และแสดงภาพตรงกลาง
        self.hog_canvas.delete("all")
        x_pos = (canvas_width - new_width) // 2
        y_pos = (canvas_height - new_height) // 2
        self.hog_canvas.create_image(x_pos, y_pos, anchor="nw", image=img_tk)
    
    def display_scan(self, img, canvas):
        """
        แสดงภาพสแกนบน canvas
        """
        # แปลงภาพจากรูปแบบ OpenCV เป็นรูปแบบ PIL
        img_pil = Image.fromarray(img)
        
        # ปรับขนาดให้พอดีกับ canvas
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        # ตรวจสอบมิติที่ถูกต้อง
        if canvas_width <= 1:
            canvas_width = 600
        if canvas_height <= 1:
            canvas_height = 400
        
        # ปรับขนาดโดยรักษาอัตราส่วน
        img_width, img_height = img_pil.size
        scale = min(canvas_width/img_width, canvas_height/img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        img_pil = img_pil.resize((new_width, new_height), Image.LANCZOS)
        
        # แปลงเป็น PhotoImage
        img_tk = ImageTk.PhotoImage(img_pil)
        
        # เก็บอ้างอิงเพื่อป้องกันการเก็บขยะ
        canvas.image = img_tk
        
        # ล้าง canvas และแสดงภาพตรงกลาง
        canvas.delete("all")
        x_pos = (canvas_width - new_width) // 2
        y_pos = (canvas_height - new_height) // 2
        canvas.create_image(x_pos, y_pos, anchor="nw", image=img_tk)
    
    def export_database(self):
        # ส่งออกฐานข้อมูลไปยังไฟล์
        file_path = filedialog.asksaveasfilename(
            defaultextension=".db",
            filetypes=[("SQLite Database", "*.db"), ("All Files", "*.*")],
            title="ส่งออกฐานข้อมูล"
        )
        
        if not file_path:
            return
        
        try:
            # ปิดการเชื่อมต่อปัจจุบัน
            self.conn.close()
            
            # คัดลอกฐานข้อมูล
            import shutil
            shutil.copy2("fingerprint_db.sqlite", file_path)
            
            # เปิดการเชื่อมต่อใหม่
            self.conn = sqlite3.connect('fingerprint_db.sqlite')
            
            messagebox.showinfo("สำเร็จ", f"ส่งออกฐานข้อมูลไปยัง {file_path} เรียบร้อยแล้ว")
        except Exception as e:
            messagebox.showerror("ข้อผิดพลาดในการส่งออก", f"เกิดข้อผิดพลาดในการส่งออกฐานข้อมูล: {str(e)}")
            # ตรวจสอบว่าเปิดการเชื่อมต่อใหม่
            self.conn = sqlite3.connect('fingerprint_db.sqlite')
    
    def import_database(self):
        # นำเข้าฐานข้อมูลจากไฟล์
        file_path = filedialog.askopenfilename(
            filetypes=[("SQLite Database", "*.db"), ("All Files", "*.*")],
            title="นำเข้าฐานข้อมูล"
        )
        
        if not file_path:
            return
        
        try:
            # ปิดการเชื่อมต่อปัจจุบัน
            self.conn.close()
            
            # สำรองฐานข้อมูลปัจจุบัน
            import shutil
            backup_path = "fingerprint_db_backup.sqlite"
            shutil.copy2("fingerprint_db.sqlite", backup_path)
            
            # คัดลอกฐานข้อมูลที่นำเข้า
            shutil.copy2(file_path, "fingerprint_db.sqlite")
            
            # เปิดการเชื่อมต่อใหม่
            self.conn = sqlite3.connect('fingerprint_db.sqlite')
            
            # รีเฟรชมุมมอง
            self.refresh_user_list()
            self.refresh_admin_view()
            
            messagebox.showinfo("สำเร็จ", f"นำเข้าฐานข้อมูลจาก {file_path} เรียบร้อยแล้ว")
        except Exception as e:
            messagebox.showerror("ข้อผิดพลาดในการนำเข้า", f"เกิดข้อผิดพลาดในการนำเข้าฐานข้อมูล: {str(e)}")
            # คืนค่าสำรองและเปิดการเชื่อมต่อใหม่
            try:
                shutil.copy2(backup_path, "fingerprint_db.sqlite")
            except:
                pass
            self.conn = sqlite3.connect('fingerprint_db.sqlite')

if __name__ == "__main__":
    root = tk.Tk()
    app = FingerprintSystem(root)
    root.mainloop()
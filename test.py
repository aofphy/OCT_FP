import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageGrab, ImageDraw
import sqlite3
from datetime import datetime
import pyautogui
from skimage.feature import hog
from skimage import exposure

class AreaSelector:
    def __init__(self, root, callback):
        self.root = root
        self.callback = callback
        
        # Create a toplevel window
        self.top = tk.Toplevel(root)
        self.top.attributes('-fullscreen', True)
        self.top.attributes('-alpha', 0.3)  # Semi-transparent
        self.top.configure(background='grey')
        
        # Make the window stay on top
        self.top.attributes('-topmost', True)
        
        # Binding mouse events
        self.top.bind("<ButtonPress-1>", self.on_button_press)
        self.top.bind("<B1-Motion>", self.on_mouse_drag)
        self.top.bind("<ButtonRelease-1>", self.on_button_release)
        self.top.bind("<Escape>", self.on_escape)
        
        # Create a canvas for drawing
        self.canvas = tk.Canvas(self.top, cursor="cross", bg="grey")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Variables for rectangle drawing
        self.rect = None
        self.start_x = None
        self.start_y = None
        
        # Instructions
        self.canvas.create_text(
            self.top.winfo_screenwidth() // 2,
            50,
            text="Click and drag to select an area for capture. Press Escape to cancel.",
            fill="white",
            font=("Arial", 16, "bold")
        )
    
    def on_button_press(self, event):
        # Save the starting position
        self.start_x = event.x
        self.start_y = event.y
        
        # Create a rectangle if not yet created
        if self.rect is None:
            self.rect = self.canvas.create_rectangle(
                self.start_x, self.start_y, self.start_x, self.start_y,
                outline='red', width=2
            )
    
    def on_mouse_drag(self, event):
        # Update rectangle as mouse is dragged
        if self.rect:
            self.canvas.coords(self.rect, self.start_x, self.start_y, event.x, event.y)
    
    def on_button_release(self, event):
        # Get final rectangle coordinates
        if self.rect:
            x1, y1, x2, y2 = self.canvas.coords(self.rect)
            # Sort coordinates to ensure x1,y1 is top-left and x2,y2 is bottom-right
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            # Destroy the selector window
            self.top.destroy()
            
            # Capture the selected area
            if x2 - x1 > 10 and y2 - y1 > 10:  # Ensure area is reasonable size
                self.callback(int(x1), int(y1), int(x2), int(y2))
    
    def on_escape(self, event):
        # Cancel the selection process
        self.top.destroy()
        self.callback(None, None, None, None)

class FingerprintSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Fingerprint Verification System")
        self.root.geometry("1200x700")
        
        # Database initialization
        self.conn = sqlite3.connect('fingerprint_db.sqlite')
        self.create_tables()
        
        # Current user and scan data
        self.current_user = None
        self.current_scan = None
        self.last_comparison_result = None
        
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
        
        self.tab_control.add(self.register_tab, text="User Registration")
        self.tab_control.add(self.verify_tab, text="Verification")
        self.tab_control.add(self.admin_tab, text="Admin Panel")
        
        self.tab_control.pack(expand=1, fill="both")
        
        # Set up each tab
        self.setup_register_tab()
        self.setup_verify_tab()
        self.setup_admin_tab()
    
    def setup_register_tab(self):
        # Left frame for user details
        left_frame = ttk.LabelFrame(self.register_tab, text="User Details")
        left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        ttk.Label(left_frame, text="User Name:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.username_entry = ttk.Entry(left_frame, width=30)
        self.username_entry.grid(row=0, column=1, padx=10, pady=10)
        
        ttk.Button(left_frame, text="Create User", command=self.create_user).grid(row=1, column=0, columnspan=2, padx=10, pady=10)
        
        # User list
        ttk.Label(left_frame, text="Existing Users:").grid(row=2, column=0, padx=10, pady=(20,5), sticky="w")
        
        self.user_listbox = tk.Listbox(left_frame, width=40, height=15)
        self.user_listbox.grid(row=3, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")
        self.user_listbox.bind('<<ListboxSelect>>', self.on_user_select)
        
        # Populate user list
        self.refresh_user_list()
        
        # Right frame for fingerprint scan
        right_frame = ttk.LabelFrame(self.register_tab, text="Fingerprint Scan")
        right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        # Scan controls
        scan_controls_frame = ttk.Frame(right_frame)
        scan_controls_frame.pack(pady=10, fill="x")
        
        ttk.Button(scan_controls_frame, text="Capture Screen Area", command=self.start_screen_capture_registration).grid(row=0, column=0, padx=5, pady=5, columnspan=2)
        ttk.Button(scan_controls_frame, text="Save Scan", command=self.save_scan).grid(row=0, column=2, padx=5, pady=5, columnspan=2)
        
        # Original parameters (kept for compatibility)
        ttk.Label(scan_controls_frame, text="X-scan (mm):").grid(row=1, column=0, padx=5, pady=5)
        self.x_scan_entry = ttk.Entry(scan_controls_frame, width=10)
        self.x_scan_entry.grid(row=1, column=1, padx=5, pady=5)
        self.x_scan_entry.insert(0, "0.00")
        
        ttk.Label(scan_controls_frame, text="Y-scan (mm):").grid(row=1, column=2, padx=5, pady=5)
        self.y_scan_entry = ttk.Entry(scan_controls_frame, width=10)
        self.y_scan_entry.grid(row=1, column=3, padx=5, pady=5)
        self.y_scan_entry.insert(0, "0.00")
        
        ttk.Label(scan_controls_frame, text="Depth range:").grid(row=2, column=0, padx=5, pady=5)
        self.depth_entry = ttk.Entry(scan_controls_frame, width=10)
        self.depth_entry.grid(row=2, column=1, padx=5, pady=5)
        self.depth_entry.insert(0, "400")
        
        # HOG parameters
        ttk.Label(scan_controls_frame, text="HOG Cell Size:").grid(row=2, column=2, padx=5, pady=5)
        self.hog_cell_entry = ttk.Entry(scan_controls_frame, width=10)
        self.hog_cell_entry.grid(row=2, column=3, padx=5, pady=5)
        self.hog_cell_entry.insert(0, "8")
        
        # Preview HOG features button
        ttk.Button(scan_controls_frame, text="Preview HOG Features", command=self.preview_hog_features).grid(row=3, column=0, padx=5, pady=5, columnspan=4)
        
        # Scan display
        self.scan_canvas = tk.Canvas(right_frame, bg="white", width=600, height=400, bd=2, relief="sunken")
        self.scan_canvas.pack(padx=10, pady=10, fill="both", expand=True)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w")
        status_bar.pack(side="bottom", fill="x")
        
        # Configure grid weights
        self.register_tab.columnconfigure(0, weight=1)
        self.register_tab.columnconfigure(1, weight=2)
        self.register_tab.rowconfigure(0, weight=1)
    
    def setup_verify_tab(self):
        # Left frame for scan
        left_frame = ttk.LabelFrame(self.verify_tab, text="Live Scan")
        left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        # Scan controls
        scan_controls_frame = ttk.Frame(left_frame)
        scan_controls_frame.pack(pady=10, fill="x")
        
        ttk.Button(scan_controls_frame, text="Capture Screen Area for Verification", 
                  command=self.start_screen_capture_verification).pack(pady=10)
        
        # HOG parameters for verification
        hog_frame = ttk.Frame(left_frame)
        hog_frame.pack(pady=5, fill="x")
        
        ttk.Label(hog_frame, text="Verification Threshold (%):").grid(row=0, column=0, padx=5, pady=5)
        self.verification_threshold_entry = ttk.Entry(hog_frame, width=10)
        self.verification_threshold_entry.grid(row=0, column=1, padx=5, pady=5)
        self.verification_threshold_entry.insert(0, "75")
        
        # Scan display
        self.verify_canvas = tk.Canvas(left_frame, bg="white", width=600, height=400, bd=2, relief="sunken")
        self.verify_canvas.pack(padx=10, pady=10, fill="both", expand=True)
        
        # Right frame for results
        right_frame = ttk.LabelFrame(self.verify_tab, text="Verification Results")
        right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        # Results display
        self.results_frame = ttk.Frame(right_frame)
        self.results_frame.pack(padx=10, pady=10, fill="both", expand=True)
        
        ttk.Label(self.results_frame, text="User:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.result_user_var = tk.StringVar()
        ttk.Label(self.results_frame, textvariable=self.result_user_var, font=("Arial", 12, "bold")).grid(row=0, column=1, padx=10, pady=10, sticky="w")
        
        ttk.Label(self.results_frame, text="Match Score:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.match_score_var = tk.StringVar()
        ttk.Label(self.results_frame, textvariable=self.match_score_var, font=("Arial", 12, "bold")).grid(row=1, column=1, padx=10, pady=10, sticky="w")
        
        ttk.Label(self.results_frame, text="Status:").grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.verification_status_var = tk.StringVar()
        ttk.Label(self.results_frame, textvariable=self.verification_status_var, font=("Arial", 14, "bold")).grid(row=2, column=1, padx=10, pady=10, sticky="w")
        
        # Advanced results section
        ttk.Label(self.results_frame, text="Algorithm:").grid(row=3, column=0, padx=10, pady=10, sticky="w")
        self.algorithm_var = tk.StringVar(value="HOG (Histogram of Oriented Gradients)")
        ttk.Label(self.results_frame, textvariable=self.algorithm_var, font=("Arial", 11)).grid(row=3, column=1, padx=10, pady=10, sticky="w")
        
        ttk.Button(self.results_frame, text="Verify", command=self.verify_fingerprint).grid(row=4, column=0, columnspan=2, padx=10, pady=20)
        
        # HOG visualization frame
        self.hog_frame = ttk.LabelFrame(right_frame, text="HOG Visualization")
        self.hog_frame.pack(padx=10, pady=10, fill="both", expand=True)
        
        self.hog_canvas = tk.Canvas(self.hog_frame, bg="white", width=300, height=200, bd=2, relief="sunken")
        self.hog_canvas.pack(padx=10, pady=10, fill="both", expand=True)
        
        # Configure grid weights
        self.verify_tab.columnconfigure(0, weight=1)
        self.verify_tab.columnconfigure(1, weight=1)
        self.verify_tab.rowconfigure(0, weight=1)
    
    def setup_admin_tab(self):
        # Admin panel for managing users and fingerprints
        controls_frame = ttk.Frame(self.admin_tab)
        controls_frame.pack(padx=10, pady=10, fill="x")
        
        ttk.Button(controls_frame, text="Refresh Database", command=self.refresh_user_list).pack(side="left", padx=5)
        ttk.Button(controls_frame, text="Delete Selected User", command=self.delete_user).pack(side="left", padx=5)
        ttk.Button(controls_frame, text="Export Database", command=self.export_database).pack(side="left", padx=5)
        ttk.Button(controls_frame, text="Import Database", command=self.import_database).pack(side="left", padx=5)
        
        # Database view
        data_frame = ttk.LabelFrame(self.admin_tab, text="Database Contents")
        data_frame.pack(padx=10, pady=10, fill="both", expand=True)
        
        # Users table
        users_frame = ttk.LabelFrame(data_frame, text="Users")
        users_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        self.users_treeview = ttk.Treeview(users_frame, columns=("id", "name", "created_at"), show="headings")
        self.users_treeview.heading("id", text="ID")
        self.users_treeview.heading("name", text="Name")
        self.users_treeview.heading("created_at", text="Created At")
        
        self.users_treeview.column("id", width=50)
        self.users_treeview.column("name", width=200)
        self.users_treeview.column("created_at", width=150)
        
        self.users_treeview.pack(fill="both", expand=True)
        self.users_treeview.bind('<<TreeviewSelect>>', self.on_admin_user_select)
        
        # Fingerprints table
        fp_frame = ttk.LabelFrame(data_frame, text="Fingerprints")
        fp_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        self.fp_treeview = ttk.Treeview(fp_frame, columns=("id", "user_id", "path", "date"), show="headings")
        self.fp_treeview.heading("id", text="ID")
        self.fp_treeview.heading("user_id", text="User ID")
        self.fp_treeview.heading("path", text="Path")
        self.fp_treeview.heading("date", text="Scan Date")
        
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
        self.status_var.set(f"Selected user: {item}")
        
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
        # This would typically show fingerprints in a gallery
        # For simplicity, just show latest scan if available
        cursor = self.conn.cursor()
        cursor.execute("SELECT image_path FROM fingerprints WHERE user_id = ? ORDER BY scan_date DESC LIMIT 1", (user_id,))
        result = cursor.fetchone()
        
        if result:
            image_path = result[0]
            try:
                # Load and display the fingerprint image
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                self.current_scan = img
                self.display_scan(img, canvas=self.scan_canvas)
                self.status_var.set(f"Loaded fingerprint for user {user_id} from {image_path}")
            except Exception as e:
                self.status_var.set(f"Error loading fingerprint: {str(e)}")
    
    def create_user(self):
        name = self.username_entry.get().strip()
        if not name:
            messagebox.showerror("Input Error", "Please enter a user name")
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
        
        messagebox.showinfo("Success", f"User '{name}' created successfully")
    
    def delete_user(self):
        selection = self.users_treeview.selection()
        if not selection:
            messagebox.showwarning("Selection Required", "Please select a user to delete")
            return
        
        item = self.users_treeview.item(selection[0])
        user_id = item['values'][0]
        user_name = item['values'][1]
        
        # Confirm deletion
        confirm = messagebox.askyesno("Confirm Deletion", 
                                      f"Are you sure you want to delete user '{user_name}' and all associated fingerprints?")
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
        
        messagebox.showinfo("Success", f"User '{user_name}' deleted successfully")
    
    def screen_capture_callback(self, x1, y1, x2, y2):
        if None in (x1, y1, x2, y2):
            self.status_var.set("Screen capture cancelled")
            return
        
        try:
            # Capture the selected region
            screenshot = ImageGrab.grab(bbox=(x1, y1, x2, y2))
            img_np = np.array(screenshot.convert('L'))  # Convert to grayscale
            
            # Apply preprocessing for fingerprint enhancement
            img_np = cv2.GaussianBlur(img_np, (5, 5), 0)
            img_np = cv2.equalizeHist(img_np)
            
            # Store the scan
            self.current_scan = img_np
            
            # Display the scan
            self.display_scan(img_np, canvas=self.scan_canvas)
            
            # Update status
            self.status_var.set(f"Screen area captured: {x2-x1}x{y2-y1} pixels")
        except Exception as e:
            self.status_var.set(f"Error capturing screen: {str(e)}")
    
    def start_screen_capture_registration(self):
        # Check if user is selected
        if self.current_user is None:
            messagebox.showwarning("User Required", "Please select a user first")
            return
            
        # Minimize the application window to avoid capturing itself
        self.root.iconify()
        
        # Give time for window to minimize
        self.root.after(500, lambda: self.start_area_selector("registration"))
    
    def start_screen_capture_verification(self):
        # Minimize the application window to avoid capturing itself
        self.root.iconify()
        
        # Give time for window to minimize
        self.root.after(500, lambda: self.start_area_selector("verification"))
    
    def start_area_selector(self, mode):
        if mode == "registration":
            callback = self.screen_capture_callback
        else:  # verification
            callback = self.verification_capture_callback
            
        area_selector = AreaSelector(self.root, callback)
        
        # Restore main window when area selector is closed
        self.root.after(100, lambda: self.root.after(100, self.root.deiconify))
    
    def verification_capture_callback(self, x1, y1, x2, y2):
        if None in (x1, y1, x2, y2):
            self.status_var.set("Verification capture cancelled")
            return
        
        try:
            # Capture the selected region
            screenshot = ImageGrab.grab(bbox=(x1, y1, x2, y2))
            img_np = np.array(screenshot.convert('L'))  # Convert to grayscale
            
            # Apply preprocessing for fingerprint enhancement
            img_np = cv2.GaussianBlur(img_np, (5, 5), 0)
            img_np = cv2.equalizeHist(img_np)
            
            # Store the scan
            self.current_scan = img_np
            
            # Display the scan
            self.display_scan(img_np, canvas=self.verify_canvas)
            
            # Update status
            self.status_var.set(f"Verification image captured: {x2-x1}x{y2-y1} pixels")
            
            # Show HOG features
            self.show_hog_features(img_np, self.hog_canvas)
        except Exception as e:
            self.status_var.set(f"Error capturing verification image: {str(e)}")
    
    def save_scan(self):
        if self.current_scan is None:
            messagebox.showwarning("No Scan", "Please perform a scan first")
            return
        
        if self.current_user is None:
            messagebox.showwarning("User Required", "Please select a user first")
            return
        
        # Create directories if they don't exist
        os.makedirs("fingerprints", exist_ok=True)
        
        # Save the scan to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fingerprints/user_{self.current_user}_{timestamp}.png"
        
        cv2.imwrite(filename, self.current_scan)
        
        # Calculate HOG features
        try:
            hog_features = self.calculate_hog_features(self.current_scan)
            # Serialize HOG features to bytes for storage
            hog_bytes = hog_features.tobytes()
            
            # Save to database with HOG features
            cursor = self.conn.cursor()
            cursor.execute("INSERT INTO fingerprints (user_id, image_path, scan_date, hog_features) VALUES (?, ?, ?, ?)",
                          (self.current_user, filename, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), hog_bytes))
            self.conn.commit()
            
            self.status_var.set(f"Scan saved to {filename} with HOG features")
        except Exception as e:
            # Save without HOG features if there's an error
            cursor = self.conn.cursor()
            cursor.execute("INSERT INTO fingerprints (user_id, image_path, scan_date) VALUES (?, ?, ?)",
                          (self.current_user, filename, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            self.conn.commit()
            
            self.status_var.set(f"Scan saved to {filename} without HOG features: {str(e)}")
        
        self.refresh_admin_view()
        messagebox.showinfo("Success", "Fingerprint scan saved successfully")
    
    def calculate_hog_features(self, image):
        try:
            cell_size = int(self.hog_cell_entry.get())
        except ValueError:
            cell_size = 8  # Default if invalid input
        
        # Resize for consistent HOG calculation
        resized = cv2.resize(image, (128, 128))
        
        # Calculate HOG features
        features, hog_image = hog(
            resized, 
            orientations=9, 
            pixels_per_cell=(cell_size, cell_size),
            cells_per_block=(2, 2), 
            block_norm='L2-Hys',
            visualize=True
        )
        
        return features
    
    def show_hog_features(self, image, canvas):
        try:
            # Calculate HOG features and visualization
            cell_size = int(self.hog_cell_entry.get())
            resized = cv2.resize(image, (128, 128))
            
            # Calculate HOG features with visualization
            features, hog_image = hog(
                resized, 
                orientations=9, 
                pixels_per_cell=(cell_size, cell_size),
                cells_per_block=(2, 2), 
                block_norm='L2-Hys',
                visualize=True
            )
            
            # Enhance visualization
            hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
            
            # Convert to PIL Image
            hog_pil = Image.fromarray((hog_image_rescaled * 255).astype(np.uint8))
            
            # Display on canvas
            self.display_image_on_canvas(hog_pil, canvas)
            
            return features
        except Exception as e:
            self.status_var.set(f"Error calculating HOG features: {str(e)}")
            return None
            
    def preview_hog_features(self):
        if self.current_scan is None:
            messagebox.showwarning("No Scan", "Please perform a scan first")
            return
        
        try:
            # Create a new window for HOG preview
            preview_window = tk.Toplevel(self.root)
            preview_window.title("HOG Features Preview")
            preview_window.geometry("800x400")
            
            # Create frames for original and HOG images
            original_frame = ttk.LabelFrame(preview_window, text="Original Image")
            original_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
            
            hog_frame = ttk.LabelFrame(preview_window, text="HOG Features")
            hog_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
            
            # Create canvases
            original_canvas = tk.Canvas(original_frame, bg="white", width=350, height=350)
            original_canvas.pack(padx=5, pady=5, fill="both", expand=True)
            
            hog_canvas = tk.Canvas(hog_frame, bg="white", width=350, height=350)
            hog_canvas.pack(padx=5, pady=5, fill="both", expand=True)
            
            # Display original image
            img_pil = Image.fromarray(self.current_scan)
            self.display_image_on_canvas(img_pil, original_canvas)
            
            # Calculate and display HOG features
            self.show_hog_features(self.current_scan, hog_canvas)
            
            # Configure grid weights
            preview_window.columnconfigure(0, weight=1)
            preview_window.columnconfigure(1, weight=1)
            preview_window.rowconfigure(0, weight=1)
            
            # Make window modal
            preview_window.transient(self.root)
            preview_window.grab_set()
            self.root.wait_window(preview_window)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate HOG preview: {str(e)}")
            
    def display_scan(self, img, canvas):
        # Convert the image from OpenCV format to PIL format
        img_pil = Image.fromarray(img)
        self.display_image_on_canvas(img_pil, canvas)
    
    def display_image_on_canvas(self, img_pil, canvas):
        # Resize to fit canvas
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        # Ensure we have valid dimensions
        if canvas_width <= 1:
            canvas_width = 600
        if canvas_height <= 1:
            canvas_height = 400
        
        # Resize while maintaining aspect ratio
        img_width, img_height = img_pil.size
        scale = min(canvas_width/img_width, canvas_height/img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        img_pil = img_pil.resize((new_width, new_height), Image.LANCZOS)
        
        # Convert to PhotoImage
        img_tk = ImageTk.PhotoImage(img_pil)
        
        # Save reference to prevent garbage collection
        canvas.image = img_tk
        
        # Clear canvas and display image centered
        canvas.delete("all")
        x_pos = (canvas_width - new_width) // 2
        y_pos = (canvas_height - new_height) // 2
        canvas.create_image(x_pos, y_pos, anchor="nw", image=img_tk)
        
    def verify_fingerprint(self):
        if self.current_scan is None:
            messagebox.showwarning("No Scan", "Please perform a verification scan first")
            return
        
        # Get all fingerprints from database
        cursor = self.conn.cursor()
        cursor.execute("""
        SELECT f.id, f.user_id, f.image_path, u.name, f.hog_features
        FROM fingerprints f
        JOIN users u ON f.user_id = u.id
        ORDER BY f.scan_date DESC
        """)
        fingerprints = cursor.fetchall()
        
        if not fingerprints:
            messagebox.showinfo("No Data", "No fingerprints found in database")
            return
        
        # Calculate HOG features for current scan
        current_hog_features = self.calculate_hog_features(self.current_scan)
        
        # Compare the current scan with all stored fingerprints
        best_match = None
        best_score = 0
        verification_results = []
        
        # Get threshold
        try:
            threshold = float(self.verification_threshold_entry.get())
        except ValueError:
            threshold = 75.0  # Default if invalid input
        
        for fp_id, user_id, image_path, user_name, stored_hog_bytes in fingerprints:
            try:
                # If we have HOG features stored, use them
                if stored_hog_bytes is not None:
                    # Deserialize HOG features from database
                    stored_hog_features = np.frombuffer(stored_hog_bytes, dtype=np.float64)
                    
                    # Make sure shapes match
                    if len(stored_hog_features) == len(current_hog_features):
                        # Calculate similarity score using HOG features
                        score = self.compare_hog_features(current_hog_features, stored_hog_features)
                        verification_results.append((fp_id, user_id, user_name, score, "HOG"))
                else:
                    # Fallback to image comparison if no HOG features
                    stored_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    if stored_img is None:
                        continue
                    
                    # Calculate similarity score
                    score = self.compare_fingerprints(self.current_scan, stored_img)
                    verification_results.append((fp_id, user_id, user_name, score, "Image"))
                
            except Exception as e:
                print(f"Error comparing with {image_path}: {str(e)}")
        
        # Find best match
        for result in verification_results:
            fp_id, user_id, user_name, score, method = result
            if score > best_score:
                best_score = score
                best_match = result
        
        # Show results
        if best_match:
            fp_id, user_id, user_name, score, method = best_match
            
            self.result_user_var.set(user_name)
            self.match_score_var.set(f"{score:.2f}%")
            self.algorithm_var.set(f"{method} (Histogram of Oriented Gradients)" if method == "HOG" else "Image Comparison")
            
            if score >= threshold:
                self.verification_status_var.set("VERIFIED")
                self.results_frame.configure(background="#d4ffcc")  # Light green
            else:
                self.verification_status_var.set("NOT VERIFIED")
                self.results_frame.configure(background="#ffcccc")  # Light red
                
            self.status_var.set(f"Best match: User {user_name} with score {score:.2f}% using {method}")
            self.last_comparison_result = best_match
            
            # Show detailed comparison if it's an important user
            if score >= threshold - 10:  # Show details for close matches too
                self.show_detailed_comparison(user_id)
        else:
            self.result_user_var.set("No match found")
            self.match_score_var.set("0.00%")
            self.verification_status_var.set("NOT VERIFIED")
            self.algorithm_var.set("N/A")
            self.results_frame.configure(background="#ffcccc")  # Light red
            self.status_var.set("No match found in database")
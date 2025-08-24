import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import sqlite3
from datetime import datetime

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
        
        ttk.Label(scan_controls_frame, text="X-scan (mm):").grid(row=0, column=0, padx=5, pady=5)
        self.x_scan_entry = ttk.Entry(scan_controls_frame, width=10)
        self.x_scan_entry.grid(row=0, column=1, padx=5, pady=5)
        self.x_scan_entry.insert(0, "0.00")
        
        ttk.Label(scan_controls_frame, text="Y-scan (mm):").grid(row=0, column=2, padx=5, pady=5)
        self.y_scan_entry = ttk.Entry(scan_controls_frame, width=10)
        self.y_scan_entry.grid(row=0, column=3, padx=5, pady=5)
        self.y_scan_entry.insert(0, "0.00")
        
        ttk.Label(scan_controls_frame, text="Depth range:").grid(row=1, column=0, padx=5, pady=5)
        self.depth_entry = ttk.Entry(scan_controls_frame, width=10)
        self.depth_entry.grid(row=1, column=1, padx=5, pady=5)
        self.depth_entry.insert(0, "400")
        
        ttk.Button(scan_controls_frame, text="Start Scan", command=self.start_scan).grid(row=1, column=2, padx=5, pady=5)
        ttk.Button(scan_controls_frame, text="Save Scan", command=self.save_scan).grid(row=1, column=3, padx=5, pady=5)
        
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
        
        ttk.Button(scan_controls_frame, text="Start Verification Scan", command=self.start_verification_scan).pack(pady=10)
        
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
        
        ttk.Button(self.results_frame, text="Verify", command=self.verify_fingerprint).grid(row=3, column=0, columnspan=2, padx=10, pady=20)
        
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
    
    def start_scan(self):
        # Check if user is selected
        if self.current_user is None:
            messagebox.showwarning("User Required", "Please select a user first")
            return
        
        # In a real application, this would interact with hardware
        # Here we simulate a fingerprint scan with a pattern
        self.status_var.set("Scanning fingerprint...")
        
        # Generate simulated fingerprint scan
        img = self.generate_simulated_fingerprint()
        self.current_scan = img
        
        # Display the scan
        self.display_scan(img, canvas=self.scan_canvas)
        
        self.status_var.set("Scan complete")
    
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
        
        # Save to database
        cursor = self.conn.cursor()
        cursor.execute("INSERT INTO fingerprints (user_id, image_path, scan_date) VALUES (?, ?, ?)",
                      (self.current_user, filename, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        self.conn.commit()
        
        self.status_var.set(f"Scan saved to {filename}")
        self.refresh_admin_view()
        
        messagebox.showinfo("Success", "Fingerprint scan saved successfully")
    
    def start_verification_scan(self):
        self.status_var.set("Starting verification scan...")
        
        # Generate a simulated fingerprint scan
        img = self.generate_simulated_fingerprint()
        self.current_scan = img
        
        # Display the scan
        self.display_scan(img, canvas=self.verify_canvas)
        
        self.status_var.set("Verification scan complete")
    
    def verify_fingerprint(self):
        if self.current_scan is None:
            messagebox.showwarning("No Scan", "Please perform a verification scan first")
            return
        
        # Get all fingerprints from database
        cursor = self.conn.cursor()
        cursor.execute("""
        SELECT f.id, f.user_id, f.image_path, u.name 
        FROM fingerprints f
        JOIN users u ON f.user_id = u.id
        ORDER BY f.scan_date DESC
        """)
        fingerprints = cursor.fetchall()
        
        if not fingerprints:
            messagebox.showinfo("No Data", "No fingerprints found in database")
            return
        
        # Compare the current scan with all stored fingerprints
        best_match = None
        best_score = 0
        
        for fp_id, user_id, image_path, user_name in fingerprints:
            try:
                stored_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if stored_img is None:
                    continue
                
                # Calculate similarity score
                score = self.compare_fingerprints(self.current_scan, stored_img)
                
                if score > best_score:
                    best_score = score
                    best_match = (fp_id, user_id, user_name, score)
            except Exception as e:
                print(f"Error comparing with {image_path}: {str(e)}")
        
        # Show results
        if best_match:
            fp_id, user_id, user_name, score = best_match
            
            self.result_user_var.set(user_name)
            self.match_score_var.set(f"{score:.2f}%")
            
            if score >= 85:
                self.verification_status_var.set("VERIFIED")
                self.results_frame.configure(background="#d4ffcc")  # Light green
            else:
                self.verification_status_var.set("NOT VERIFIED")
                self.results_frame.configure(background="#ffcccc")  # Light red
                
            self.status_var.set(f"Best match: User {user_name} with score {score:.2f}%")
            self.last_comparison_result = best_match
        else:
            self.result_user_var.set("No match found")
            self.match_score_var.set("0.00%")
            self.verification_status_var.set("NOT VERIFIED")
            self.results_frame.configure(background="#ffcccc")  # Light red
            self.status_var.set("No match found in database")
    
    def export_database(self):
        # Export the database to a file
        file_path = filedialog.asksaveasfilename(
            defaultextension=".db",
            filetypes=[("SQLite Database", "*.db"), ("All Files", "*.*")],
            title="Export Database"
        )
        
        if not file_path:
            return
        
        try:
            # Close the current connection
            self.conn.close()
            
            # Copy the database
            import shutil
            shutil.copy2("fingerprint_db.sqlite", file_path)
            
            # Reopen the connection
            self.conn = sqlite3.connect('fingerprint_db.sqlite')
            
            messagebox.showinfo("Success", f"Database exported to {file_path}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Error exporting database: {str(e)}")
            # Ensure connection is reopened
            self.conn = sqlite3.connect('fingerprint_db.sqlite')
    
    def import_database(self):
        # Import database from a file
        file_path = filedialog.askopenfilename(
            filetypes=[("SQLite Database", "*.db"), ("All Files", "*.*")],
            title="Import Database"
        )
        
        if not file_path:
            return
        
        try:
            # Close the current connection
            self.conn.close()
            
            # Backup current database
            import shutil
            backup_path = "fingerprint_db_backup.sqlite"
            shutil.copy2("fingerprint_db.sqlite", backup_path)
            
            # Copy the imported database
            shutil.copy2(file_path, "fingerprint_db.sqlite")
            
            # Reopen the connection
            self.conn = sqlite3.connect('fingerprint_db.sqlite')
            
            # Refresh views
            self.refresh_user_list()
            self.refresh_admin_view()
            
            messagebox.showinfo("Success", f"Database imported from {file_path}")
        except Exception as e:
            messagebox.showerror("Import Error", f"Error importing database: {str(e)}")
            # Restore backup and reopen connection
            try:
                shutil.copy2(backup_path, "fingerprint_db.sqlite")
            except:
                pass
            self.conn = sqlite3.connect('fingerprint_db.sqlite')
    
    def generate_simulated_fingerprint(self):
        # Create a simulated fingerprint pattern
        height, width = 400, 600
        img = np.zeros((height, width), dtype=np.uint8)
        
        # Generate horizontal lines with increasing darkness and noise
        for y in range(height):
            # Increase darkness as we go down
            base_value = max(200 - y // 2, 0)
            
            # Add more noise as we go down
            noise_factor = min(y / 100, 4)
            
            line = np.ones(width) * base_value
            noise = np.random.normal(0, noise_factor * 10, width)
            line = np.clip(line + noise, 0, 255).astype(np.uint8)
            
            img[y] = line
        
        # Add pattern variations to make it look more like a fingerprint
        if y > height // 2:
            # Add ridge-like patterns in lower half
            ridge_freq = 5 + (y % 10)
            ridge_pattern = np.sin(np.arange(width) / ridge_freq) * 20
            img[y] = np.clip(img[y] + ridge_pattern, 0, 255).astype(np.uint8)
        
        return img
    
    def display_scan(self, img, canvas):
        # Convert the image from OpenCV format to PIL format
        img_pil = Image.fromarray(img)
        
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
    
    def compare_fingerprints(self, img1, img2):
        """
        Compare two fingerprint images and return a similarity score (0-100%)
        
        In a real application, this would use specialized fingerprint matching algorithms.
        This is a simplified version using basic image comparison.
        """
        # Resize images to same dimensions
        h, w = 200, 300
        img1_resized = cv2.resize(img1, (w, h))
        img2_resized = cv2.resize(img2, (w, h))
        
        # Calculate structural similarity index
        try:
            # For OpenCV 4.x+
            from skimage.metrics import structural_similarity as ssim
            score, _ = ssim(img1_resized, img2_resized, full=True)
        except ImportError:
            # Fallback to simpler comparison
            # Normalize images
            img1_norm = img1_resized.astype('float') / 255.0
            img2_norm = img2_resized.astype('float') / 255.0
            
            # Calculate the absolute difference
            diff = np.abs(img1_norm - img2_norm)
            
            # Calculate similarity score (inverted difference)
            score = 1.0 - np.mean(diff)
        
        # Convert to percentage
        return score * 100

if __name__ == "__main__":
    root = tk.Tk()
    app = FingerprintSystem(root)
    root.mainloop()
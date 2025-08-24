# -*- coding: utf-8 -*-
import os
import shutil
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Toplevel
import cv2
import numpy as np
from PIL import Image, ImageTk
import sqlite3
from datetime import datetime
from skimage.feature import hog
from skimage import exposure
import mss
import mss.tools
import warnings
import time
import serial
import serial.tools.list_ports
import threading
import re
from typing import Optional, List

# --- Class ImageCropper ---
class ImageCropper(Toplevel):
    # (โค้ด ImageCropper เหมือนเดิมจาก v4.3 ไม่เปลี่ยนแปลง)
    def __init__(self, parent, pil_image):
        super().__init__(parent)
        self.parent = parent
        self.pil_image = pil_image
        self.title("เลือกพื้นที่ที่ต้องการ")
        max_width = self.winfo_screenwidth() * 0.8
        max_height = self.winfo_screenheight() * 0.8
        self.geometry(f"{int(max_width)}x{int(max_height)}")
        self.resizable(True, True)
        self.grab_set()
        self.focus_set()
        self.transient(parent)

        # State for selection
        self.start_x = None
        self.start_y = None
        self.current_x = None
        self.current_y = None
        self.selection_rect = None

        # Canvas and controls
        self.canvas = tk.Canvas(self, bg='darkgrey', cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        button_frame = ttk.Frame(self)
        button_frame.pack(pady=10, fill="x")
        self.info_label = ttk.Label(button_frame, text="คลิกและลากเพื่อเลือกพื้นที่ แล้วกด 'ยืนยัน'")
        self.info_label.pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="ยกเลิก", command=self.cancel).pack(side=tk.RIGHT, padx=10)
        ttk.Button(button_frame, text="ยืนยัน", command=self.confirm).pack(side=tk.RIGHT, padx=10)

        # Image render state
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
        self.selected_bbox = None

        # Defer initial draw until window is visible
        self.wait_visibility()
        self.center_window()
        self.display_image()
    def on_canvas_resize(self, event): self.after_idle(self.display_image)
    def display_image(self):
        canvas_width = self.canvas.winfo_width(); canvas_height = self.canvas.winfo_height()
        if canvas_width <= 1 or canvas_height <= 1:
            try: g=self.geometry(); sp=g.split('+')[0]; canvas_width=int(sp.split('x')[0])-20; canvas_height=int(sp.split('x')[1])-60;
            except (ValueError,IndexError,TypeError,AttributeError,tk.TclError): print("Warn: Geometry fallback."); canvas_width=780; canvas_height=540
        if canvas_width <= 1 or canvas_height <=1 : 
            # Canvas not ready yet, try again later
            print("Warn: Invalid canvas dims - retrying...")
            self.after(100, self.display_image)
            return
        if not hasattr(self,'pil_image') or self.pil_image is None: print("Warn: No pil_image."); return
        img_width, img_height = self.pil_image.size
        if img_width <= 0 or img_height <= 0: print("Warn: Invalid image dims."); return
        scale_w=canvas_width/img_width; scale_h=canvas_height/img_height; self.scale=min(scale_w,scale_h)
        if self.scale>=1.0 or self.scale<=0: self.scale=1.0
        self.display_width=max(1,int(img_width*self.scale)); self.display_height=max(1,int(img_height*self.scale))
        try: dip=self.pil_image.resize((self.display_width,self.display_height),Image.Resampling.LANCZOS)
        except ValueError: print("Error: Resize failed."); return
        try: 
            # IMPORTANT: Keep a reference to the image to prevent garbage collection
            # Bind the image to this Toplevel to ensure the same Tcl interpreter/root
            self.photo_image = ImageTk.PhotoImage(dip, master=self)
        except Exception as e: 
            print(f"Error PhotoImage: {e}")
            # Try again after a short delay if it's a timing issue
            if "no default root window" in str(e).lower():
                self.after(100, self.display_image)
            return
        self.offset_x=max(0,(canvas_width-self.display_width)//2); self.offset_y=max(0,(canvas_height-self.display_height)//2)
        try:
             if not self.canvas.winfo_exists(): return
             self.canvas.delete("image"); self.canvas.create_image(self.offset_x,self.offset_y,anchor="nw",image=self.photo_image,tags="image")
        except tk.TclError as e: print(f"Error drawing image: {e}")
    def _canvas_to_original(self, cx, cy):
        if self.scale==0: return 0,0
        ox=(cx-self.offset_x)/self.scale; oy=(cy-self.offset_y)/self.scale; iw,ih=self.pil_image.size
        ox=max(0,min(iw,ox)); oy=max(0,min(ih,oy)); return int(ox),int(oy)
    def _canvas_to_display(self, cx, cy):
         if not hasattr(self,'display_width') or self.display_width<=0 or not hasattr(self,'display_height') or self.display_height<=0: return cx,cy
         x=max(self.offset_x,min(self.offset_x+self.display_width,cx)); y=max(self.offset_y,min(self.offset_y+self.display_height,cy)); return x,y
    def on_button_press(self, event):
         if not hasattr(self,'display_width') or not hasattr(self,'display_height'): return
         if not (self.offset_x<=event.x<=self.offset_x+self.display_width and self.offset_y<=event.y<=self.offset_y+self.display_height): self.start_x=None; return
         self.start_x,self.start_y=self._canvas_to_display(event.x,event.y)
         if self.selection_rect: self.canvas.delete(self.selection_rect)
         self.selection_rect=self.canvas.create_rectangle(self.start_x,self.start_y,self.start_x,self.start_y,outline='red',width=2,dash=(4,2),tags="selection")
    def on_mouse_drag(self, event):
       if self.start_x is None or self.start_y is None or self.selection_rect is None:
           return
       cx, cy = self._canvas_to_display(event.x, event.y)
       try:
           self.canvas.coords(self.selection_rect, int(self.start_x), int(self.start_y), int(cx), int(cy))
           self.current_x, self.current_y = cx, cy
       except Exception:
           # Ignore transient type issues
           pass
    def on_button_release(self, event):
       if self.start_x is None or self.selection_rect is None:
           if self.selection_rect: self.canvas.delete(self.selection_rect); self.selection_rect=None
           self.start_x=None; return
       ex,ey=self._canvas_to_display(event.x,event.y)
        # Ensure start coords are valid before drawing
       if self.start_x is None or self.start_y is None:
           return
       try:
           self.canvas.coords(self.selection_rect,int(self.start_x),int(self.start_y),int(ex),int(ey))
           self.current_x=ex; self.current_y=ey
       except Exception:
           pass
    def confirm(self):
        if self.selection_rect and self.start_x is not None and self.current_x is not None:
            try: cx1,cy1,cx2,cy2=self.canvas.coords(self.selection_rect)
            except ValueError: messagebox.showwarning("ไม่ได้เลือก","ลากเลือกพื้นที่ก่อน",parent=self); return
            ox1,oy1=self._canvas_to_original(min(cx1,cx2),min(cy1,cy2)); ox2,oy2=self._canvas_to_original(max(cx1,cx2),max(cy1,cy2))
            if ox2-ox1>=1 and oy2-oy1>=1: self.selected_bbox=(ox1,oy1,ox2,oy2); print(f"Selected BBox: {self.selected_bbox}"); self.destroy()
            else: messagebox.showwarning("เล็กไป","เลือกพื้นที่อย่างน้อย 1x1",parent=self)
        else: messagebox.showwarning("ไม่ได้เลือก","ลากเลือกพื้นที่ก่อน",parent=self)
    def cancel(self, event=None): self.selected_bbox=None; self.destroy()
    def center_window(self):
        self.update_idletasks(); w=0; h=0
        try: g=self.geometry(); sp=g.split('+')[0]; w=int(sp.split('x')[0]); h=int(sp.split('x')[1])
        except (ValueError,IndexError,TypeError,AttributeError,tk.TclError) as e:
            print(f"Warn: Parse geometry failed '{self.geometry()}': {e}. Using winfo.")
            try: w=self.winfo_width(); h=self.winfo_height()
            except tk.TclError: print("Warn: winfo failed."); return
        if w<=0 or h<=0: print("Warn: Invalid dimensions."); return
        sw=self.winfo_screenwidth(); sh=self.winfo_screenheight(); x=max(0,(sw//2)-(w//2)); y=max(0,(sh//2)-(h//2))
        try: self.geometry(f'{w}x{h}+{x}+{y}')
        except tk.TclError as e: print(f"Warn: set geometry failed: {e}")

# --- Class FingerprintSystem ---
class FingerprintSystem:
    def __init__(self, root):
        self.root = root; self.root.title("ระบบตรวจสอบลายนิ้วมือ OCT (v5.2 - Arduino Control)"); self.root.geometry("1250x750") # Updated title
        self.is_capturing_register=False; self.is_capturing_verify=False; self.capture_job_id_register=None; self.capture_job_id_verify=None; self.sct=None; self.capture_monitor=None; self.capture_bbox_register=None; self.capture_bbox_verify=None; self.last_frame_time=0
        self.is_verifying_live = False # สถานะการตรวจสอบสด
        self.live_verify_update_interval = 5 # ตรวจสอบทุกๆ กี่เฟรม (ปรับได้)
        self.live_verify_frame_count = 0
        
        # Arduino initialization
        self.arduino_port = None
        self.arduino_serial = None
        self.arduino_connected = False
        self.relay_open_duration = 3  # seconds to keep door open
        self.arduino_status_var = tk.StringVar(value="Arduino: ไม่เชื่อมต่อ")
        
        self.conn=sqlite3.connect('fingerprint_db.sqlite'); self.conn.execute("PRAGMA foreign_keys = ON;"); self.init_db()
        self.current_user=None; self.current_scan=None; self.last_comparison_result=None
        self.style=ttk.Style(self.root)
        try: dbg=self.style.lookup('TFrame','background'); self.style.configure('Normal.TLabel',foreground='black',background=dbg,font=('Arial',14,'bold')); self.style.configure('Success.TLabel',foreground='green',background='#d4ffcc',font=('Arial',14,'bold')); self.style.configure('Failure.TLabel',foreground='#cc0000',background='#ffcccc',font=('Arial',14,'bold')); self.style.configure('Error.TLabel',foreground='orange',background='#fff0cc',font=('Arial',14,'bold'))
        except tk.TclError: print("Warn: Fallback styling."); self.style.configure('Normal.TLabel',foreground='black',font=('Arial',14,'bold')); self.style.configure('Success.TLabel',foreground='green',font=('Arial',14,'bold')); self.style.configure('Failure.TLabel',foreground='#cc0000',font=('Arial',14,'bold')); self.style.configure('Error.TLabel',foreground='orange',font=('Arial',14,'bold'))
        self.create_ui(); self.init_mss(); 
        # Delay Arduino initialization to avoid startup issues
        self.root.after(1000, self.init_arduino)

    def init_mss(self):
         try: self.sct=mss.mss(); self.capture_monitor = self.sct.monitors[1] if len(self.sct.monitors)>1 else self.sct.monitors[0]; print(f"Monitor: {self.capture_monitor}")
         except Exception as e: messagebox.showerror("Error",f"mss init failed: {e}",parent=self.root); self.sct=None

    def close_mss(self):
        if self.sct: self.sct.close(); print("mss closed.")
    
    # --- Arduino Connection Methods ---
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
                # Look for common Arduino identifiers
                if any(keyword in port.description.lower() for keyword in ['arduino', 'ch340', 'ch341', 'cp210', 'ftdi']):
                    arduino_ports.append(port.device)
                elif any(keyword in port.manufacturer.lower() if port.manufacturer else '' for keyword in ['arduino', 'ch340', 'ch341']):
                    arduino_ports.append(port.device)
            
            # If no specific Arduino found, include all available ports
            if not arduino_ports:
                arduino_ports = [port.device for port in ports]
                
        except Exception as e:
            print(f"Error finding Arduino ports: {e}")
        
        return arduino_ports
    
    def connect_arduino(self, port=None):
        """Connect to Arduino"""
        try:
            if self.arduino_serial and self.arduino_serial.is_open:
                self.arduino_serial.close()
                
            if port is None:
                # Try to find Arduino automatically
                ports = self.find_arduino_ports()
                if not ports:
                    raise Exception("ไม่พบพอร์ต Arduino")
                port = ports[0]  # Use first available port
            
            self.arduino_serial = serial.Serial(
                port=port,
                baudrate=9600,
                timeout=2,
                write_timeout=2
            )
            
            # Wait for Arduino to initialize
            time.sleep(2)
            
            # Test connection by sending a test command
            self.send_arduino_command("TEST")
            
            self.arduino_port = port
            self.arduino_connected = True
            self.arduino_status_var.set(f"Arduino: เชื่อมต่อ ({port})")
            self.status_var.set(f"เชื่อมต่อ Arduino สำเร็จ: {port}")
            print(f"Arduino connected on {port}")
            
        except Exception as e:
            self.arduino_connected = False
            self.arduino_status_var.set("Arduino: ไม่เชื่อมต่อ")
            self.status_var.set(f"เชื่อมต่อ Arduino ไม่สำเร็จ: {e}")
            print(f"Arduino connection failed: {e}")
    
    def disconnect_arduino(self):
        """Disconnect from Arduino"""
        try:
            if self.arduino_serial and self.arduino_serial.is_open:
                self.arduino_serial.close()
            self.arduino_connected = False
            self.arduino_status_var.set("Arduino: ไม่เชื่อมต่อ")
            self.status_var.set("ยกเลิกการเชื่อมต่อ Arduino")
            print("Arduino disconnected")
        except Exception as e:
            print(f"Error disconnecting Arduino: {e}")
    
    def send_arduino_command(self, command):
        """Send command to Arduino"""
        if not self.arduino_connected or not self.arduino_serial:
            print("Arduino not connected")
            return False
            
        try:
            self.arduino_serial.write(f"{command}\n".encode())
            self.arduino_serial.flush()
            
            # Wait for response (optional)
            time.sleep(0.1)
            if self.arduino_serial.in_waiting > 0:
                response = self.arduino_serial.readline().decode().strip()
                print(f"Arduino response: {response}")
                
            return True
        except Exception as e:
            print(f"Error sending Arduino command: {e}")
            return False
    
    def open_door(self, test: bool = False):
        """Open magnetic door by controlling relay"""
        if not self.arduino_connected:
            messagebox.showwarning("Arduino ไม่เชื่อมต่อ", 
                                 "กรุณาเชื่อมต่อ Arduino ก่อนใช้งาน", 
                                 parent=self.root)
            return
        
        try:
            # Send command to open relay (turn on)
            success = self.send_arduino_command("OPEN_DOOR")
            if success:
                self.status_var.set("เปิดประตูแม่เหล็ก...")
                
                # Use threading to close door after specified duration
                def close_door_after_delay():
                    time.sleep(self.relay_open_duration)
                    self.send_arduino_command("CLOSE_DOOR")
                    self.status_var.set("ปิดประตูแม่เหล็ก")
                
                door_thread = threading.Thread(target=close_door_after_delay)
                door_thread.daemon = True
                door_thread.start()
                
                return True
            else:
                messagebox.showerror("ข้อผิดพลาด", 
                                   "ไม่สามารถส่งคำสั่งไปยัง Arduino ได้", 
                                   parent=self.root)
                return False
        except Exception as e:
            messagebox.showerror("ข้อผิดพลาด", 
                               f"เกิดข้อผิดพลาดในการเปิดประตู: {e}", 
                               parent=self.root)
            return False
    
    def test_door(self):
        """Test door opening function"""
        self.open_door()
    
    def manual_connect_arduino(self):
        """Manually connect to selected Arduino port"""
        selected_port = self.port_var.get()
        if not selected_port:
            messagebox.showwarning("ไม่ได้เลือกพอร์ต", "กรุณาเลือกพอร์ตก่อน", parent=self.root)
            return
        
        self.connect_arduino(selected_port)
    
    def refresh_ports(self):
        """Refresh available ports list"""
        try:
            ports = self.find_arduino_ports()
            self.port_combo['values'] = ports
            if ports:
                self.port_combo.current(0)  # Select first port
                self.port_var.set(ports[0])
        except Exception as e:
            print(f"Error refreshing ports: {e}")
    
    def update_door_duration(self):
        """Update door opening duration"""
        try:
            new_duration = float(self.door_duration_var.get())
            if 1 <= new_duration <= 10:
                self.relay_open_duration = new_duration
                self.status_var.set(f"อัพเดทเวลาเปิดประตู: {new_duration} วินาที")
            else:
                messagebox.showwarning("ค่าไม่ถูกต้อง", "เวลาเปิดประตูต้องอยู่ระหว่าง 1-10 วินาที", parent=self.root)
                self.door_duration_var.set(str(self.relay_open_duration))  # Reset to previous value
        except ValueError:
            messagebox.showerror("ค่าไม่ถูกต้อง", "กรุณาใส่ตัวเลขที่ถูกต้อง", parent=self.root)
            self.door_duration_var.set(str(self.relay_open_duration))  # Reset to previous value

    def init_db(self):
        try:
            cur = self._get_db_cursor()
            if not cur:
                messagebox.showerror("Database Error", "Could not get DB cursor for initialization.", parent=self.root)
                return

            cur.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL
                )
            ''')
            cur.execute('''
                CREATE TABLE IF NOT EXISTS fingerprints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    image_path TEXT NOT NULL,
                    hog_features BLOB,
                    minutiae BLOB,
                    scan_date TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                )
            ''')
            self._commit_db()
        except Exception as e:
            print(f"Error initializing DB: {e}")
            messagebox.showerror("Database Error", f"Failed to initialize database: {e}", parent=self.root)

    def create_tables(self):
        cur = self._get_db_cursor()
        if not cur:
            return
        cur.execute('CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL)')
        cur.execute('CREATE TABLE IF NOT EXISTS fingerprints (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, image_path TEXT NOT NULL, scan_date TEXT NOT NULL, hog_features BLOB, FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE)')
        self._commit_db()

    def create_ui(self):
        self.tab_control=ttk.Notebook(self.root); self.register_tab=ttk.Frame(self.tab_control); self.verify_tab=ttk.Frame(self.tab_control); self.admin_tab=ttk.Frame(self.tab_control); self.tab_control.add(self.register_tab,text="ลงทะเบียน"); self.tab_control.add(self.verify_tab,text="ตรวจสอบ"); self.tab_control.add(self.admin_tab,text="ดูแลระบบ"); self.tab_control.pack(expand=1,fill="both",padx=5,pady=5)
        self.setup_register_tab(); self.setup_verify_tab(); self.setup_admin_tab()
        self.status_var=tk.StringVar(value="พร้อมใช้งาน"); sb=ttk.Label(self.root,textvariable=self.status_var,relief="sunken",anchor="w",padding=(5,2)); sb.pack(side="bottom",fill="x")

    def setup_register_tab(self):
        lf=ttk.LabelFrame(self.register_tab,text="ข้อมูลผู้ใช้",padding=(10,5)); lf.grid(row=0,column=0,padx=10,pady=10,sticky="nsew")
        ttk.Label(lf,text="ชื่อผู้ใช้:").grid(row=0,column=0,padx=5,pady=5,sticky="w"); self.username_entry=ttk.Entry(lf,width=30); self.username_entry.grid(row=0,column=1,padx=5,pady=5,sticky="ew")
        ttk.Button(lf,text="สร้างผู้ใช้ใหม่",command=self.create_user).grid(row=1,column=0,columnspan=2,padx=5,pady=10)
        ttk.Label(lf,text="ผู้ใช้ที่มีอยู่:").grid(row=2,column=0,columnspan=2,padx=5,pady=(15,2),sticky="w")
        ulf=ttk.Frame(lf); ulf.grid(row=3,column=0,columnspan=2,padx=5,pady=2,sticky="nsew")
        usc=ttk.Scrollbar(ulf,orient="vertical"); self.user_listbox=tk.Listbox(ulf,width=40,height=15,exportselection=False,yscrollcommand=usc.set); usc.config(command=self.user_listbox.yview); usc.pack(side="right",fill="y"); self.user_listbox.pack(side="left",fill="both",expand=True); self.user_listbox.bind('<<ListboxSelect>>',self.on_user_select_register)
        lf.columnconfigure(1,weight=1); lf.rowconfigure(3,weight=1); ulf.columnconfigure(0,weight=1); ulf.rowconfigure(0,weight=1); self.refresh_user_list()
        rf=ttk.LabelFrame(self.register_tab,text="สแกนลายนิ้วมือ",padding=(10,5)); rf.grid(row=0,column=1,padx=10,pady=10,sticky="nsew")
        ccf=ttk.Frame(rf); ccf.pack(pady=5,fill="x")
        self.select_area_reg_btn=ttk.Button(ccf,text="1. เลือกพื้นที่",command=lambda: self.select_capture_area("register")); self.select_area_reg_btn.pack(side=tk.LEFT,padx=5)
        self.start_capture_reg_btn=ttk.Button(ccf,text="2. เริ่มสด",command=self.start_capture_register,state=tk.DISABLED); self.start_capture_reg_btn.pack(side=tk.LEFT,padx=5)
        self.stop_capture_reg_btn=ttk.Button(ccf,text="หยุด",command=self.stop_capture_register,state=tk.DISABLED); self.stop_capture_reg_btn.pack(side=tk.LEFT,padx=5)
        self.capture_frame_reg_btn=ttk.Button(ccf,text="3. จับภาพ",command=self.capture_current_frame_register,state=tk.DISABLED); self.capture_frame_reg_btn.pack(side=tk.LEFT,padx=5)
        # New: import from B-scan folder
        ttk.Button(ccf,text="นำเข้าโฟลเดอร์ B-scan",command=self.action_register_from_bscan_folder).pack(side=tk.LEFT,padx=10)
        self.save_scan_reg_btn=ttk.Button(ccf,text="บันทึก",command=self.save_scan,state=tk.DISABLED); self.save_scan_reg_btn.pack(side=tk.LEFT,padx=10)
        self.area_info_reg_var=tk.StringVar(value="พื้นที่: -"); ttk.Label(ccf,textvariable=self.area_info_reg_var,foreground="grey").pack(side=tk.LEFT,padx=5)
        pf=ttk.LabelFrame(rf,text="พารามิเตอร์ OCT (จำลอง)"); pf.pack(pady=10,fill="x",padx=5)
        ttk.Label(pf,text="X(mm):").grid(row=0,column=0,padx=5,pady=5,sticky="w"); self.x_scan_entry=ttk.Entry(pf,width=10); self.x_scan_entry.grid(row=0,column=1,padx=5,pady=5); self.x_scan_entry.insert(0,"0.00")
        ttk.Label(pf,text="Y(mm):").grid(row=0,column=2,padx=5,pady=5,sticky="w"); self.y_scan_entry=ttk.Entry(pf,width=10); self.y_scan_entry.grid(row=0,column=3,padx=5,pady=5); self.y_scan_entry.insert(0,"0.00")
        ttk.Label(pf,text="ลึก:").grid(row=1,column=0,padx=5,pady=5,sticky="w"); self.depth_entry=ttk.Entry(pf,width=10); self.depth_entry.grid(row=1,column=1,padx=5,pady=5); self.depth_entry.insert(0,"400")
        cfr=ttk.Frame(rf); cfr.pack(padx=5,pady=5,fill="both",expand=True)
        self.scan_canvas_register=tk.Canvas(cfr,bg="black",bd=2,relief="sunken"); self.scan_canvas_register.pack(fill="both",expand=True)
        self.scan_canvas_register.bind("<Configure>",lambda e: self._draw_placeholder(self.scan_canvas_register,"1. เลือกพื้นที่จับภาพ")); self._draw_placeholder(self.scan_canvas_register,"1. เลือกพื้นที่จับภาพ")
        self.register_tab.columnconfigure(0,weight=1); self.register_tab.columnconfigure(1,weight=3); self.register_tab.rowconfigure(0,weight=1); rf.rowconfigure(3,weight=1)

    def setup_verify_tab(self):
        lf=ttk.LabelFrame(self.verify_tab,text="จอแสดงผล / ตรวจสอบสด",padding=(10,5)); lf.grid(row=0,column=0,padx=10,pady=10,sticky="nsew")
        ccf=ttk.Frame(lf); ccf.pack(pady=5,fill="x")
        self.select_area_verify_btn=ttk.Button(ccf,text="1. เลือกพื้นที่",command=lambda: self.select_capture_area("verify")); self.select_area_verify_btn.pack(side=tk.LEFT,padx=5)
        self.toggle_live_verify_btn=ttk.Button(ccf,text="2. เริ่มตรวจสอบสด",command=self.toggle_live_verify, state=tk.DISABLED); self.toggle_live_verify_btn.pack(side=tk.LEFT,padx=10)
        # New: identify from B-scan folder
        ttk.Button(ccf,text="ตรวจจากโฟลเดอร์ B-scan",command=self.action_identify_from_bscan_folder).pack(side=tk.LEFT,padx=10)
        self.area_info_verify_var=tk.StringVar(value="พื้นที่: -"); ttk.Label(ccf,textvariable=self.area_info_verify_var,foreground="grey").pack(side=tk.LEFT,padx=5)
        cfv=ttk.Frame(lf); cfv.pack(padx=5,pady=5,fill="both",expand=True)
        self.scan_canvas_verify=tk.Canvas(cfv,bg="black",bd=2,relief="sunken"); self.scan_canvas_verify.pack(fill="both",expand=True)
        self.scan_canvas_verify.bind("<Configure>",lambda e: self._draw_placeholder(self.scan_canvas_verify,"1. เลือกพื้นที่จับภาพ")); self._draw_placeholder(self.scan_canvas_verify,"1. เลือกพื้นที่จับภาพ")
        rf=ttk.LabelFrame(self.verify_tab,text="ผลการตรวจสอบ",padding=(10,5)); rf.grid(row=0,column=1,padx=10,pady=10,sticky="nsew")
        self.results_frame=ttk.Frame(rf,padding=(5,5)); self.results_frame.pack(fill="both",expand=True)
        ttk.Label(self.results_frame,text="ผู้ใช้ที่ตรงกัน:").grid(row=0,column=0,padx=5,pady=5,sticky="w"); self.result_user_var=tk.StringVar(value="-"); ttk.Label(self.results_frame,textvariable=self.result_user_var,font=("Arial",12,"bold")).grid(row=0,column=1,padx=5,pady=5,sticky="w")
        ttk.Label(self.results_frame,text="คะแนนความเหมือน:").grid(row=1,column=0,padx=5,pady=5,sticky="w"); self.match_score_var=tk.StringVar(value="-"); ttk.Label(self.results_frame,textvariable=self.match_score_var,font=("Arial",12,"bold")).grid(row=1,column=1,padx=5,pady=5,sticky="w")
        ttk.Label(self.results_frame,text="สถานะ:").grid(row=2,column=0,padx=5,pady=5,sticky="w"); self.verification_status_var=tk.StringVar(value="-")
        self.verification_status_label=ttk.Label(self.results_frame,textvariable=self.verification_status_var,style='Normal.TLabel'); self.verification_status_label.grid(row=2,column=1,padx=5,pady=5,sticky="w")

        # --- Comment out HOG display ---
        # ttk.Label(self.results_frame,text="HOG (ภาพที่จับ):").grid(row=3,column=0,columnspan=2,padx=5,pady=(15,2),sticky="w")
        # hcf=ttk.Frame(self.results_frame); hcf.grid(row=4,column=0,columnspan=2,padx=5,pady=2,sticky="nsew")
        # self.hog_canvas=tk.Canvas(hcf,bg="white",bd=2,relief="sunken"); self.hog_canvas.pack(fill="both",expand=True)
        # self.hog_canvas.bind("<Configure>",lambda e: self._draw_placeholder(self.hog_canvas,"HOG Visualization")); self._draw_placeholder(self.hog_canvas,"HOG Visualization")
        # --- End of Comment out ---

        # Adjust grid weights if HOG is removed
        self.verify_tab.columnconfigure(0,weight=1); self.verify_tab.columnconfigure(1,weight=1); self.verify_tab.rowconfigure(0,weight=1); lf.rowconfigure(1,weight=1); self.results_frame.columnconfigure(1,weight=1);
        # self.results_frame.rowconfigure(4,weight=1) # Remove or adjust if HOG canvas is gone

    def setup_admin_tab(self):
        cf=ttk.Frame(self.admin_tab,padding=(0,5)); cf.pack(padx=10,pady=5,fill="x"); ttk.Button(cf,text="รีเฟรช",command=self.refresh_admin_view).pack(side=tk.LEFT,padx=5); ttk.Button(cf,text="ลบผู้ใช้",command=self.delete_user).pack(side=tk.LEFT,padx=5); ttk.Button(cf,text="ส่งออก",command=self.export_database).pack(side=tk.LEFT,padx=5); ttk.Button(cf,text="นำเข้า",command=self.import_database).pack(side=tk.LEFT,padx=5)
        
        # Arduino control panel
        arduino_frame = ttk.LabelFrame(self.admin_tab, text="การควบคุม Arduino", padding=(10,5))
        arduino_frame.pack(padx=10, pady=5, fill="x")
        
        # Arduino status
        status_frame = ttk.Frame(arduino_frame)
        status_frame.pack(fill="x", pady=(0,5))
        ttk.Label(status_frame, text="สถานะ:").pack(side=tk.LEFT, padx=(0,5))
        self.arduino_status_label = ttk.Label(status_frame, textvariable=self.arduino_status_var, style='Normal.TLabel')
        self.arduino_status_label.pack(side=tk.LEFT, padx=5)
        
        # Arduino controls
        control_frame = ttk.Frame(arduino_frame)
        control_frame.pack(fill="x", pady=5)
        
        ttk.Button(control_frame, text="เชื่อมต่อ", command=self.manual_connect_arduino).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="ยกเลิกการเชื่อมต่อ", command=self.disconnect_arduino).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="ทดสอบเปิดประตู", command=lambda: self.open_door(test=True), state=tk.DISABLED).pack(side=tk.LEFT, padx=5)
        
        # Port selection
        port_frame = ttk.Frame(arduino_frame)
        port_frame.pack(fill="x", pady=5)
        ttk.Label(port_frame, text="พอร์ต:").pack(side=tk.LEFT, padx=(0,5))
        self.port_var = tk.StringVar()
        self.port_combo = ttk.Combobox(port_frame, textvariable=self.port_var, width=15, state="readonly")
        self.port_combo.pack(side=tk.LEFT, padx=5)
        ttk.Button(port_frame, text="ค้นหาพอร์ต", command=self.refresh_ports).pack(side=tk.LEFT, padx=5)
        
        # Door control settings
        settings_frame = ttk.Frame(arduino_frame)
        settings_frame.pack(fill="x", pady=5)
        ttk.Label(settings_frame, text="เวลาเปิดประตู (วินาที):").pack(side=tk.LEFT, padx=(0,5))
        self.door_duration_var = tk.StringVar(value=str(self.relay_open_duration))
        door_spin = tk.Spinbox(settings_frame, from_=1, to=10, textvariable=self.door_duration_var, width=5)
        door_spin.pack(side=tk.LEFT, padx=5)
        ttk.Button(settings_frame, text="อัพเดท", command=self.update_door_duration).pack(side=tk.LEFT, padx=5)
        
        # Refresh ports on startup
        self.refresh_ports()
        
        dp=ttk.PanedWindow(self.admin_tab,orient=tk.HORIZONTAL); dp.pack(padx=10,pady=5,fill="both",expand=True); uf=ttk.LabelFrame(dp,text="ผู้ใช้",padding=(10,5)); dp.add(uf,weight=1); self.users_treeview=ttk.Treeview(uf,columns=("id","name","created_at"),show="headings"); us=ttk.Scrollbar(uf,orient="vertical",command=self.users_treeview.yview); self.users_treeview.configure(yscrollcommand=us.set); self.users_treeview.heading("id",text="ID",anchor="center"); self.users_treeview.heading("name",text="ชื่อ"); self.users_treeview.heading("created_at",text="สร้างเมื่อ"); self.users_treeview.column("id",width=50,anchor="center",stretch=False); self.users_treeview.column("name",width=200); self.users_treeview.column("created_at",width=150); us.pack(side="right",fill="y"); self.users_treeview.pack(fill="both",expand=True); self.users_treeview.bind('<<TreeviewSelect>>',self.on_admin_user_select); fpf=ttk.Frame(dp); dp.add(fpf,weight=2); fpfr=ttk.LabelFrame(fpf,text="ลายนิ้วมือ",padding=(10,5)); fpfr.pack(padx=0,pady=0,fill="both",expand=True); self.fp_treeview=ttk.Treeview(fpfr,columns=("id","user_id","path","date"),show="headings"); fps=ttk.Scrollbar(fpfr,orient="vertical",command=self.fp_treeview.yview); self.fp_treeview.configure(yscrollcommand=fps.set); self.fp_treeview.heading("id",text="FP ID",anchor="center"); self.fp_treeview.heading("user_id",text="User ID",anchor="center"); self.fp_treeview.heading("path",text="ที่เก็บไฟล์"); self.fp_treeview.heading("date",text="วันที่สแกน"); self.fp_treeview.column("id",width=60,anchor="center",stretch=False); self.fp_treeview.column("user_id",width=60,anchor="center",stretch=False); self.fp_treeview.column("path",width=250); self.fp_treeview.column("date",width=150); fps.pack(side="right",fill="y"); self.fp_treeview.pack(fill="both",expand=True); self.fp_treeview.bind('<<TreeviewSelect>>',self.on_admin_fp_select); self.admin_preview_frame=ttk.LabelFrame(fpf,text="ภาพตัวอย่าง",padding=(5,5)); self.admin_preview_frame.pack(padx=0,pady=(10,0),fill="x",expand=False); self.admin_preview_canvas=tk.Canvas(self.admin_preview_frame,bg="lightgrey",height=150); self.admin_preview_canvas.pack(fill="x",expand=True); self.admin_preview_canvas.bind("<Configure>",lambda e: self._draw_placeholder(self.admin_preview_canvas,"ภาพตัวอย่าง")); self._draw_placeholder(self.admin_preview_canvas,"ภาพตัวอย่าง"); self.refresh_admin_view()

    def _capture_fullscreen(self):
        if not self.sct or not self.capture_monitor: print("Error: mss not initialized."); return None
        try:
            with mss.mss(display=getattr(self.sct,'display',None)) as sct_fs: sct_img = sct_fs.grab(self.capture_monitor)
            img_pil = Image.frombytes("RGB", sct_img.size, sct_img.rgb); img_pil_gray = img_pil.convert('L'); return np.array(img_pil_gray)
        except Exception as e: print(f"ERROR: Capture fullscreen failed: {e}"); return None

    def select_capture_area(self, mode):
        if mode=="register" and self.is_capturing_register: self.stop_capture_register()
        elif mode=="verify" and self.is_verifying_live: self.stop_live_verify()
        self.status_var.set("จับภาพสำหรับเลือกพื้นที่..."); self.root.update_idletasks(); self.root.iconify(); self.root.after(300)
        full_img_np=self._capture_fullscreen();
        if self.root.state()!='normal': self.root.deiconify()
        if full_img_np is None: messagebox.showerror("ผิดพลาด","จับภาพหน้าจอไม่ได้",parent=self.root); self.status_var.set("เลือกพื้นที่ล้มเหลว"); return
        try:
            fpi=Image.fromarray(full_img_np); self.status_var.set("เลือกพื้นที่ในหน้าต่างใหม่..."); crp=ImageCropper(self.root,fpi); self.root.wait_window(crp)
            bbox=getattr(crp,'selected_bbox',None)
            if bbox:
                x1,y1,x2,y2=bbox; w=x2-x1; h=y2-y1; bd={"top":y1,"left":x1,"width":w,"height":h}; ainfo=f"พื้นที่:{w}x{h} @({x1},{y1})"
                cp=full_img_np[y1:y2,x1:x2].copy()
                if mode=="register":
                    self.capture_bbox_register=bd; self.area_info_reg_var.set(ainfo); self.start_capture_reg_btn.config(state=tk.NORMAL); self.display_scan(cp,self.scan_canvas_register)
                    self.stop_capture_reg_btn.config(state=tk.DISABLED); self.capture_frame_reg_btn.config(state=tk.DISABLED); self.save_scan_reg_btn.config(state=tk.DISABLED); self.current_scan = None
                else: # verify mode
                    self.capture_bbox_verify=bd; self.area_info_verify_var.set(ainfo); self.toggle_live_verify_btn.config(state=tk.NORMAL)
                    self.display_scan(cp,self.scan_canvas_verify); self._reset_verification_results(); self.current_scan = None
                self.status_var.set(f"เลือกพื้นที่แล้ว {ainfo}")
            else: # User cancelled Cropper
                if mode=="register": self.capture_bbox_register=None; self.start_capture_reg_btn.config(state=tk.DISABLED); self.area_info_reg_var.set("พื้นที่: -")
                else: self.capture_bbox_verify=None; self.toggle_live_verify_btn.config(state=tk.DISABLED); self.area_info_verify_var.set("พื้นที่: -")
                self.status_var.set("ยกเลิกเลือกพื้นที่")
        except Exception as e: messagebox.showerror("ผิดพลาด",f"เลือกพื้นที่ผิดพลาด: {e}",parent=self.root); self.status_var.set("เลือกพื้นที่ล้มเหลว")

    def start_capture_register(self):
        if self.is_capturing_register or self.is_verifying_live: messagebox.showwarning("กำลังทำงาน","แสดงผลสดอยู่แล้ว",parent=self.root); return
        if not self.sct: messagebox.showerror("Error","mss ไม่พร้อม",parent=self.root); return
        if not self.capture_bbox_register: messagebox.showerror("Error","เลือกพื้นที่ก่อน",parent=self.root); return
        self.is_capturing_register=True; self.select_area_reg_btn.config(state=tk.DISABLED); self.start_capture_reg_btn.config(state=tk.DISABLED); self.stop_capture_reg_btn.config(state=tk.NORMAL); self.capture_frame_reg_btn.config(state=tk.NORMAL); self.save_scan_reg_btn.config(state=tk.DISABLED);
        self.select_area_verify_btn.config(state=tk.DISABLED); self.toggle_live_verify_btn.config(state=tk.DISABLED)
        self.status_var.set("แสดงผลสด (Register Area)..."); self.capture_loop(self.scan_canvas_register,"register")

    def stop_capture_register(self):
        if self.capture_job_id_register: self.root.after_cancel(self.capture_job_id_register); self.capture_job_id_register=None
        self.is_capturing_register=False; self.select_area_reg_btn.config(state=tk.NORMAL); self.start_capture_reg_btn.config(state=tk.NORMAL if self.capture_bbox_register else tk.DISABLED); self.stop_capture_reg_btn.config(state=tk.DISABLED); self.capture_frame_reg_btn.config(state=tk.DISABLED)
        self.select_area_verify_btn.config(state=tk.NORMAL); self.toggle_live_verify_btn.config(state=tk.NORMAL if self.capture_bbox_verify else tk.DISABLED)
        self.status_var.set("หยุดแสดงผลสด (Register)")
        chs = self.current_scan is not None and hasattr(self.scan_canvas_register, f"img_ref_{self.scan_canvas_register.winfo_name()}") and getattr(self.scan_canvas_register, f"img_ref_{self.scan_canvas_register.winfo_name()}", None) is not None
        if not chs: self._draw_placeholder(self.scan_canvas_register, "1. เลือกพื้นที่จับภาพ")

    def capture_current_frame_register(self):
        if not self.is_capturing_register: messagebox.showinfo("ข้อมูล", "ต้องกด 'เริ่มแสดงผลสด' ก่อน", parent=self.root); return
        self._capture_frame_action(self.scan_canvas_register, "register")

    def _capture_frame_action(self, canvas, mode):
        try:
            bbox = self.capture_bbox_register if mode == "register" else self.capture_bbox_verify
            if not bbox or not self.sct:
                return
            sct_img = self.sct.grab(bbox)
            img_bgr = np.array(sct_img)
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2GRAY)
            processed = self.preprocess_fingerprint(img_gray)
            self.current_scan = processed
            self.display_scan(processed, canvas)
            if mode == "register":
                self.save_scan_reg_btn.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("ผิดพลาด", f"จับเฟรมไม่ได้: {e}", parent=self.root)

    def toggle_live_verify(self):
        if self.is_verifying_live: self.stop_live_verify()
        else: self.start_live_verify()

    def start_live_verify(self):
        if self.is_capturing_register or self.is_verifying_live: messagebox.showwarning("กำลังทำงาน","แสดงผล/ตรวจสอบสดอยู่แล้ว",parent=self.root); return
        if not self.sct: messagebox.showerror("Error","mss ไม่พร้อม",parent=self.root); return
        if not self.capture_bbox_verify: messagebox.showerror("Error","เลือกพื้นที่ก่อน",parent=self.root); return
        self.is_verifying_live = True; self.live_verify_frame_count = 0; self.toggle_live_verify_btn.config(text="หยุดตรวจสอบสด")
        self.select_area_verify_btn.config(state=tk.DISABLED)
        self.select_area_reg_btn.config(state=tk.DISABLED); self.start_capture_reg_btn.config(state=tk.DISABLED); self.stop_capture_reg_btn.config(state=tk.DISABLED); self.capture_frame_reg_btn.config(state=tk.DISABLED); self.save_scan_reg_btn.config(state=tk.DISABLED)
        self.status_var.set("กำลังตรวจสอบสด..."); self._reset_verification_results(); self.capture_loop(self.scan_canvas_verify, "verify")

    def stop_live_verify(self):
        if self.capture_job_id_verify: self.root.after_cancel(self.capture_job_id_verify); self.capture_job_id_verify = None
        self.is_verifying_live = False; self.toggle_live_verify_btn.config(text="เริ่มตรวจสอบสด")
        self.select_area_verify_btn.config(state=tk.NORMAL)
        self.select_area_reg_btn.config(state=tk.NORMAL); self.start_capture_reg_btn.config(state=tk.NORMAL if self.capture_bbox_register else tk.DISABLED)
        self.status_var.set("หยุดตรวจสอบสด")
        # Don't clear canvas, keep last frame or result

    def capture_loop(self, target_canvas, mode):
        is_live_verify_mode = (mode == "verify" and self.is_verifying_live)
        is_live_register_mode = (mode == "register" and self.is_capturing_register)
        if not is_live_verify_mode and not is_live_register_mode: return
        if not self.sct: print("Error: mss not ready"); return
        current_bbox = self.capture_bbox_register if mode == "register" else self.capture_bbox_verify
        if not current_bbox: print(f"Error: Area not set for {mode}"); self.stop_capture_register() if mode=="register" else self.stop_live_verify(); return
        try:
            sct_img = self.sct.grab(current_bbox); img_bgr = np.array(sct_img); img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2RGB)
            self.display_scan(img_rgb, canvas=target_canvas)
            if is_live_verify_mode:
                self.live_verify_frame_count += 1
                if self.live_verify_frame_count % self.live_verify_update_interval == 0:
                    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2GRAY) # Convert frame to gray for processing
                    processed_img = self.preprocess_fingerprint(img_gray)
                    current_hog_features, _ = self.extract_hog_features(processed_img)
                    if current_hog_features is not None:
                        match_result = self._compare_features(current_hog_features)
                        self._update_verification_ui(match_result)
                    else: self._update_verification_ui(None, error_message="HOG Error")
            delay_ms = 30
            job_id = self.root.after(delay_ms, lambda: self.capture_loop(target_canvas, mode))
            if mode == "register": self.capture_job_id_register = job_id
            else: self.capture_job_id_verify = job_id
        except Exception as e:
            print(f"Capture loop error ({mode}, bbox={current_bbox}): {e}")
            if mode == "register": self.stop_capture_register()
            else: self.stop_live_verify()
            if time.time() - getattr(self, f'_{mode}_loop_error_time', 0) > 5: messagebox.showerror("Capture Error", f"แสดงผลสดผิดพลาด: {e}", parent=self.root); setattr(self, f'_{mode}_loop_error_time', time.time())

    def _compare_features(self, current_hog_features):
        """Enhanced comparison using both HOG and minutiae features"""
        best_match_user_id = None; best_match_user_name = None; highest_score = -1.0
        bmuid = None
        bmun = None
        try:
            cur = self._get_db_cursor()
            if not cur:
                return None
            cur.execute("SELECT f.id,f.user_id,f.image_path,f.hog_features,u.name FROM fingerprints f JOIN users u ON f.user_id=u.id WHERE f.hog_features IS NOT NULL")
            fdb = cur.fetchall()
            if not fdb:
                return None

            # Extract minutiae from current scan if available
            current_minutiae = []
            if hasattr(self, 'current_scan') and self.current_scan is not None:
                current_minutiae = self.extract_minutiae_features(self.current_scan)

            for fid, uid, ip, hb, un in fdb:
                try:
                    sh = np.frombuffer(hb, dtype=np.float64)
                except ValueError:
                    print(f"Warn: Invalid HOG data for FP ID {fid}")
                    continue

                if current_hog_features.shape != sh.shape:
                    continue

                # Calculate HOG similarity
                hog_score = self.calculate_similarity(current_hog_features, sh)

                # Calculate minutiae similarity if both images available
                minutiae_score = 0.0
                if current_minutiae and os.path.exists(ip):
                    try:
                        stored_img = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
                        if stored_img is not None:
                            stored_processed = self.preprocess_fingerprint(stored_img)
                            stored_minutiae = self.extract_minutiae_features(stored_processed)
                            minutiae_score = self.compare_minutiae(current_minutiae, stored_minutiae)
                    except Exception as e:
                        print(f"Minutiae comparison error for FP {fid}: {e}")
                        minutiae_score = 0.0

                # Combine HOG and minutiae scores (HOG weighted higher for robustness)
                if minutiae_score > 0:
                    combined_score = 0.7 * hog_score + 0.3 * minutiae_score
                else:
                    combined_score = hog_score

                if combined_score > highest_score:
                    highest_score = combined_score
                    bmuid = uid
                    bmun = un

            if bmun is not None:
                return (bmuid, bmun, highest_score)
            else:
                return None
        except sqlite3.Error as e:
            print(f"DB Error during comparison: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error during comparison: {e}")
            return None

    def compare_hog(self, hog1, hog2):
        """Compare HOG features using cosine similarity"""
        try:
            # Normalize the feature vectors
            norm_hog1 = hog1 / np.linalg.norm(hog1)
            norm_hog2 = hog2 / np.linalg.norm(hog2)
            
            # Compute the cosine similarity
            similarity = np.dot(norm_hog1, norm_hog2)
            
            # Convert to a percentage
            return similarity * 100.0
        except Exception as e:
            print(f"Error comparing HOG features: {e}")
            return 0.0

    def preprocess_fingerprint(self, img):
        if img is None: raise ValueError("Input image is None")
        if len(img.shape)!=2:
             if len(img.shape)==3: img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
             else: raise ValueError(f"Bad shape: {img.shape}")
        
        # Enhanced preprocessing for OCT fingerprint images
        ts=(300,300);
        try: img_r=cv2.resize(img,ts,interpolation=cv2.INTER_AREA)
        except cv2.error as e: print(f"Resize error: {e}"); img_r=img
        
        # 1. Noise reduction with bilateral filter (preserves edges better than median blur)
        img_denoised = cv2.bilateralFilter(img_r, 9, 75, 75)
        
        # 2. Enhanced contrast using CLAHE with optimized parameters for OCT
        cl=cv2.createCLAHE(clipLimit=3.0,tileGridSize=(8,8)); 
        img_c=cl.apply(img_denoised)
        
        # 3. Speckle noise reduction specific to OCT images
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        img_opened = cv2.morphologyEx(img_c, cv2.MORPH_OPEN, kernel)
        img_closed = cv2.morphologyEx(img_opened, cv2.MORPH_CLOSE, kernel)
        
        # 4. Histogram equalization for better contrast
        img_eq = cv2.equalizeHist(img_closed)
        
        # 5. Gabor filter bank for ridge enhancement (key improvement for fingerprints)
        img_gabor = self.apply_gabor_filter_bank(img_eq)
        
        # 6. Final thresholding with optimized parameters
        img_t=cv2.adaptiveThreshold(img_gabor,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,15,3)
        
        # 7. Ridge thinning for better minutiae detection
        img_thinned = self.zhang_suen_thinning(img_t)
        
        return img_thinned

    def apply_gabor_filter_bank(self, img):
        """Apply Gabor filter bank for ridge enhancement"""
        try:
            # Define Gabor parameters optimized for fingerprint ridges
            angles = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5]  # 8 orientations
            frequency = 0.1  # Spatial frequency
            sigma_x = 2
            sigma_y = 2
            
            gabor_responses = []
            
            for angle in angles:
                theta = np.pi * angle / 180
                kernel = cv2.getGaborKernel((21, 21), sigma_x, theta, 2*np.pi*frequency, 0.5, 0, ktype=cv2.CV_32F)
                filtered = cv2.filter2D(img, cv2.CV_8UC3, kernel)
                gabor_responses.append(filtered)
            
            # Combine responses using maximum response
            if gabor_responses:
                combined = np.maximum.reduce(gabor_responses)
                return combined.astype(np.uint8)
            else:
                return img
        except Exception as e:
            print(f"Gabor filter error: {e}")
            return img

    def zhang_suen_thinning(self, img):
        """Apply Zhang-Suen thinning algorithm for ridge thinning"""
        try:
            # Convert to binary if not already
            if img.max() > 1:
                img = (img > 127).astype(np.uint8)
            
            # Zhang-Suen thinning implementation
            def get_neighbors(img, x, y):
                """Get 8-connected neighbors in order"""
                h, w = img.shape
                neighbors = []
                for dx, dy in [(-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w:
                        neighbors.append(img[nx, ny])
                    else:
                        neighbors.append(0)
                return neighbors

            def count_transitions(neighbors):
                """Count 0->1 transitions in neighbors"""
                transitions = 0
                for i in range(8):
                    if neighbors[i] == 0 and neighbors[(i+1) % 8] == 1:
                        transitions += 1
                return transitions

            img_copy = img.copy()
            changing = True
            
            while changing:
                changing = False
                
                # Step 1
                to_delete = []
                h, w = img_copy.shape
                
                for x in range(1, h-1):
                    for y in range(1, w-1):
                        if img_copy[x, y] == 1:
                            neighbors = get_neighbors(img_copy, x, y)
                            
                            # Conditions for deletion
                            if (2 <= sum(neighbors) <= 6 and
                                count_transitions(neighbors) == 1 and
                                neighbors[0] * neighbors[2] * neighbors[4] == 0 and
                                neighbors[2] * neighbors[4] * neighbors[6] == 0):
                                to_delete.append((x, y))
                
                for x, y in to_delete:
                    img_copy[x, y] = 0
                    changing = True
                
                # Step 2
                to_delete = []
                
                for x in range(1, h-1):
                    for y in range(1, w-1):
                        if img_copy[x, y] == 1:
                            neighbors = get_neighbors(img_copy, x, y)
                            
                            # Conditions for deletion
                            if (2 <= sum(neighbors) <= 6 and
                                count_transitions(neighbors) == 1 and
                                neighbors[0] * neighbors[2] * neighbors[6] == 0 and
                                neighbors[0] * neighbors[4] * neighbors[6] == 0):
                                to_delete.append((x, y))
                
                for x, y in to_delete:
                    img_copy[x, y] = 0
                    changing = True
            
            # Convert back to 0-255 range
            return (img_copy * 255).astype(np.uint8)
            
        except Exception as e:
            print(f"Thinning error: {e}")
            return img

    def extract_hog_features(self, img):
        """Enhanced HOG feature extraction with multiple scales and orientations"""
        if img is None: return None,None
        if len(img.shape)!=2:
             if len(img.shape)==3: img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
             else: print(f"Bad shape HOG: {img.shape}"); return None,None
        
        # Multi-scale HOG feature extraction
        features = []
        
        try:
            # Scale 1: 128x128 - fine details
            ths1=(128,128)
            img_r1=cv2.resize(img,ths1,interpolation=cv2.INTER_AREA)
            fd1,hi1=hog(img_r1,orientations=12,pixels_per_cell=(8,8),cells_per_block=(2,2),
                        visualize=True,block_norm='L2-Hys',feature_vector=True)
            features.append(fd1)
            
            # Scale 2: 64x64 - medium details  
            ths2=(64,64)
            img_r2=cv2.resize(img,ths2,interpolation=cv2.INTER_AREA)
            fd2,hi2=hog(img_r2,orientations=9,pixels_per_cell=(4,4),cells_per_block=(2,2),
                        visualize=True,block_norm='L2-Hys',feature_vector=True)
            features.append(fd2)
            
            # Scale 3: 32x32 - global structure
            ths3=(32,32)
            img_r3=cv2.resize(img,ths3,interpolation=cv2.INTER_AREA)
            fd3,hi3=hog(img_r3,orientations=6,pixels_per_cell=(4,4),cells_per_block=(1,1),
                        visualize=True,block_norm='L2-Hys',feature_vector=True)
            features.append(fd3)
            
            # Combine all features
            combined_features = np.concatenate(features)
            
            # Use the finest scale for visualization
            hir = None
            if hi1 is not None: 
                hir = exposure.rescale_intensity(hi1, out_range='uint8').astype(np.uint8)
            
            return combined_features, hir
            
        except Exception as e: 
            print(f"HOG error: {e}")
            # Fallback to original method
            try:
                ths=(128,128)
                img_r=cv2.resize(img,ths,interpolation=cv2.INTER_AREA)
                fd,hi=hog(img_r,orientations=9,pixels_per_cell=(8,8),cells_per_block=(2,2),
                         visualize=True,block_norm='L2-Hys',feature_vector=True)
                hir=None
                if hi is not None:
                    hir = exposure.rescale_intensity(hi, out_range='uint8').astype(np.uint8)
                return fd,hir
            except Exception as e2:
                print(f"Fallback HOG error: {e2}")
                return None,None

    def save_scan(self):
        if self.current_scan is None: messagebox.showwarning("ไม่มีภาพ","'จับภาพเฟรมนี้' ก่อน",parent=self.root); return
        if self.current_user is None: messagebox.showwarning("ไม่ได้เลือก","เลือกผู้ใช้ก่อน",parent=self.root); return
        save_dir = "fingerprints"; os.makedirs(save_dir, exist_ok=True); timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f"); user_name = "unknown"; filename = None
        try:
            cur=self._get_db_cursor()
            if cur:
                cur.execute("SELECT name FROM users WHERE id=?",(self.current_user,))
                res=cur.fetchone()
                user_name = res[0].replace(" ","_") if res else user_name
        except Exception as name_e:
            print(f"Warn: get user name failed: {name_e}")
        filename_base = f"user_{self.current_user}_{user_name}_{timestamp}.png"; filename = os.path.abspath(os.path.join(save_dir,filename_base))
        try:
            success = cv2.imwrite(filename,self.current_scan);
            if not success: raise IOError(f"imwrite failed: {filename}")
            hog_features, _ = self.extract_hog_features(self.current_scan);
            if hog_features is None:
                 messagebox.showerror("HOG Error","สกัด HOG ไม่ได้ ไม่บันทึก",parent=self.root)
                 if filename and os.path.exists(filename):
                     try: os.remove(filename); print(f"Removed HOG error file: {filename}")
                     except OSError as rem_e: print(f"Warn: Cannot remove HOG error file {filename}: {rem_e}")
                     except Exception as rem_gen_e: print(f"Warn: Unexpected error removing HOG error file {filename}: {rem_gen_e}")
                 return
            hog_features_binary = hog_features.astype(np.float64).tobytes(); cursor = self._get_db_cursor()
            if not cursor:
                raise sqlite3.Error("No DB cursor available")
            cursor.execute("INSERT INTO fingerprints (user_id,image_path,scan_date,hog_features) VALUES (?,?,?,?)",(self.current_user,filename,datetime.now().strftime("%Y-%m-%d %H:%M:%S"),hog_features_binary)); self._commit_db()
            self.status_var.set(f"บันทึก: {filename_base}"); self.refresh_admin_view(); self.load_user_fingerprints(self.current_user,self.scan_canvas_register); messagebox.showinfo("สำเร็จ","บันทึกแล้ว",parent=self.root)
            self.save_scan_reg_btn.config(state=tk.DISABLED)
        except (sqlite3.Error,IOError,Exception) as e:
            messagebox.showerror("ผิดพลาด",f"บันทึกไม่ได้: {str(e)}",parent=self.root); self._commit_db()
            if filename and os.path.exists(filename):
                try: os.remove(filename); print(f"Removed orphaned file due to DB error: {filename}")
                except OSError as rem_e: print(f"Warning: Could not remove orphaned file {filename}: {rem_e}")
                except Exception as rem_gen_e: print(f"Warning: Unexpected error removing orphaned file {filename}: {rem_gen_e}")

    # verify_fingerprint is no longer needed for a button
    # def verify_fingerprint(self): ...

    def calculate_similarity(self, hog1, hog2):
        """Enhanced similarity calculation using multiple metrics"""
        f1=np.asarray(hog1).flatten(); f2=np.asarray(hog2).flatten();
        if f1.shape!=f2.shape: return 0.0;
        
        try:
            # 1. Cosine Similarity (original method)
            n1=np.linalg.norm(f1); n2=np.linalg.norm(f2);
            if n1==0 or n2==0: 
                cosine_sim = 100.0 if n1==0 and n2==0 else 0.0;
            else:
                eps=1e-9; cs=np.dot(f1,f2)/((n1*n2)+eps); cs=np.clip(cs,-1.0,1.0);
                cosine_sim = (cs+1.0)/2.0*100.0
            
            # 2. Correlation coefficient
            if len(f1) > 1:
                correlation = np.corrcoef(f1, f2)[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
                correlation_sim = (correlation + 1.0) / 2.0 * 100.0
            else:
                correlation_sim = cosine_sim
            
            # 3. Chi-square distance (converted to similarity)
            # Add small epsilon to avoid division by zero
            f1_norm = f1 + 1e-10
            f2_norm = f2 + 1e-10
            chi_square = np.sum((f1_norm - f2_norm) ** 2 / (f1_norm + f2_norm))
            chi_square_sim = max(0, 100.0 - chi_square)
            
            # 4. Histogram intersection
            intersection = np.sum(np.minimum(f1, f2))
            union = np.sum(np.maximum(f1, f2))
            if union > 0:
                intersection_sim = (intersection / union) * 100.0
            else:
                intersection_sim = 100.0 if np.array_equal(f1, f2) else 0.0
            
            # 5. Weighted combination of all metrics
            # Cosine similarity gets highest weight for OCT fingerprints
            weights = [0.4, 0.25, 0.15, 0.2]  # cosine, correlation, chi-square, intersection
            similarities = [cosine_sim, correlation_sim, chi_square_sim, intersection_sim]
            
            final_similarity = sum(w * s for w, s in zip(weights, similarities))
            
            # Apply adaptive thresholding based on feature vector statistics
            feature_std = np.std(f1) + np.std(f2)
            if feature_std < 0.1:  # Low variance features might need boost
                final_similarity *= 1.1
            
            return np.clip(final_similarity, 0.0, 100.0)
            
        except Exception as e:
            print(f"Similarity calculation error: {e}")
            # Fallback to original cosine similarity
            n1=np.linalg.norm(f1); n2=np.linalg.norm(f2);
            if n1==0 or n2==0: return 100.0 if n1==0 and n2==0 else 0.0;
            eps=1e-9; cs=np.dot(f1,f2)/((n1*n2)+eps); cs=np.clip(cs,-1.0,1.0);
            return (cs+1.0)/2.0*100.0

    def extract_minutiae_features(self, img):
        """Extract minutiae points (ridge endings and bifurcations) for enhanced matching"""
        try:
            if img is None or len(img.shape) != 2:
                return []
            
            # Ensure binary image
            if img.max() > 1:
                binary_img = (img > 127).astype(np.uint8)
            else:
                binary_img = img
            
            minutiae_points = []
            h, w = binary_img.shape
            
            # Scan for minutiae using crossing number method
            for i in range(1, h-1):
                for j in range(1, w-1):
                    if binary_img[i, j] == 1:  # Ridge pixel
                        # Get 8-connected neighbors
                        neighbors = [
                            binary_img[i-1, j], binary_img[i-1, j+1], binary_img[i, j+1],
                            binary_img[i+1, j+1], binary_img[i+1, j], binary_img[i+1, j-1],
                            binary_img[i, j-1], binary_img[i-1, j-1]
                        ]
                        
                        # Calculate crossing number
                        cn = 0
                        for k in range(8):
                            cn += abs(neighbors[k] - neighbors[(k+1) % 8])
                        cn = cn // 2
                        
                        # Classify minutiae
                        if cn == 1:  # Ridge ending
                            minutiae_points.append((i, j, 'ending'))
                        elif cn == 3:  # Ridge bifurcation
                            minutiae_points.append((i, j, 'bifurcation'))
            
            return minutiae_points
            
        except Exception as e:
            print(f"Minutiae extraction error: {e}")
            return []

    def compare_minutiae(self, minutiae1, minutiae2, tolerance=10):
        """Compare minutiae points between two fingerprints"""
        try:
            if not minutiae1 or not minutiae2:
                return 0.0
            
            matches = 0
            total_minutiae = max(len(minutiae1), len(minutiae2))
            
            for m1 in minutiae1:
                x1, y1, type1 = m1
                for m2 in minutiae2:
                    x2, y2, type2 = m2
                    
                    # Check if minutiae are close and of same type
                    distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                    if distance <= tolerance and type1 == type2:
                        matches += 1
                        break  # Count each minutiae only once
            
            if total_minutiae > 0:
                return (matches / total_minutiae) * 100.0
            else:
                return 0.0
                
        except Exception as e:
            print(f"Minutiae comparison error: {e}")
            return 0.0

    def display_hog_image(self, hog_image_np):
        # HOG display disabled in this UI variant
        return

    def display_scan(self, img_np, canvas, max_width=None, max_height=None, is_hog=False):
        if img_np is None: self._clear_canvas(canvas,"ไม่มีภาพ"); return
        cname = "unknown_canvas"; nw, nh = 0, 0
        try:
            if not canvas or not canvas.winfo_exists(): print("Warn: Invalid canvas."); return
            cname=canvas.winfo_name()
            if len(img_np.shape)==2: img_pil=Image.fromarray(img_np)
            elif len(img_np.shape)==3 and img_np.shape[2]==3: img_pil=Image.fromarray(img_np)
            elif len(img_np.shape)==3 and img_np.shape[2]==1: img_pil=Image.fromarray(img_np.squeeze())
            else: raise ValueError(f"Bad shape: {img_np.shape}")
            cw=canvas.winfo_width(); ch=canvas.winfo_height();
            if cw<=1: cw=max_width if max_width else 600
            if ch<=1: ch=max_height if max_height else 400
            tw=max_width if max_width else cw; th=max_height if max_height else ch; iw,ih=img_pil.size;
            if iw<=0 or ih<=0: print(f"Warn: Invalid image dims {cname}: {iw}x{ih}"); self._clear_canvas(canvas,"ขนาดภาพผิด"); return
            if tw<=0 or th<=0: print(f"Warn: Invalid target dims {cname}: {tw}x{th}"); self._clear_canvas(canvas,"ขนาด Canvas ผิด"); return
            scale=min(tw/iw,th/ih,1.0);
            if scale <= 0: scale = 1.0
            nw=max(1,int(iw*scale)); nh=max(1,int(ih*scale));
            
            # Create and store image reference properly
            img_r=img_pil.resize((nw,nh),Image.Resampling.LANCZOS)
            # Bind PhotoImage to the target canvas to ensure same master/interpreter
            img_tk=ImageTk.PhotoImage(img_r, master=canvas)
            
            # Store image reference to prevent garbage collection
            if not hasattr(self, '_canvas_images'):
                self._canvas_images = {}
            self._canvas_images[id(canvas)] = img_tk
            
            canvas.delete("all")
            xp=max(0,(cw-nw)//2); yp=max(0,(ch-nh)//2)
            canvas.create_image(xp,yp,anchor="nw",image=img_tk,tags="image")
            
            # if is_hog: canvas.create_text(5,5,anchor="nw",text="HOG",fill="blue",font=("Arial",8),tags="hog_text") # Don't draw HOG text if hidden
        except Exception as e:
            print(f"ERROR display {cname}: {e}")
            self._clear_canvas(canvas,"แสดงไม่ได้")

    # --- B-scan Folder Ingestion and En-face Reconstruction ---
    def action_register_from_bscan_folder(self):
        """เลือกโฟลเดอร์ B-scan แล้วสร้างภาพ En-face เพื่อใช้ลงทะเบียน"""
        if self.current_user is None:
            messagebox.showwarning("ไม่ได้เลือกผู้ใช้","กรุณาเลือกผู้ใช้ก่อน",parent=self.root)
            return
        folder = filedialog.askdirectory(title="เลือกโฟลเดอร์ B-scan ของผู้ใช้", parent=self.root)
        if not folder:
            return
        try:
            enface = self.build_enface_from_bscans(folder, method='mip')
            if enface is None:
                messagebox.showerror("ผิดพลาด","ไม่พบภาพ B-scan ที่อ่านได้ในโฟลเดอร์นี้", parent=self.root)
                return
            processed = self.preprocess_fingerprint(enface)
            self.current_scan = processed
            self.display_scan(processed, self.scan_canvas_register)
            self.save_scan_reg_btn.config(state=tk.NORMAL)
            self.status_var.set(f"สร้าง En-face จากโฟลเดอร์: {os.path.basename(folder)}")
        except Exception as e:
            messagebox.showerror("ผิดพลาด", f"สร้าง En-face ไม่สำเร็จ: {e}", parent=self.root)

    def action_identify_from_bscan_folder(self):
        """เลือกโฟลเดอร์ B-scan แล้วสร้างภาพ En-face เพื่อใช้ตรวจสอบ Identify"""
        folder = filedialog.askdirectory(title="เลือกโฟลเดอร์ B-scan สำหรับตรวจสอบ", parent=self.root)
        if not folder:
            return
        try:
            enface = self.build_enface_from_bscans(folder, method='mip')
            if enface is None:
                messagebox.showerror("ผิดพลาด","ไม่พบภาพ B-scan ที่อ่านได้ในโฟลเดอร์นี้", parent=self.root)
                return
            processed = self.preprocess_fingerprint(enface)
            self.current_scan = processed
            self.display_scan(processed, self.scan_canvas_verify)
            cur_hog, _ = self.extract_hog_features(processed)
            if cur_hog is None:
                self._update_verification_ui(None, error_message="HOG Error")
                return
            result = self._compare_features(cur_hog)
            self._update_verification_ui(result)
            self.status_var.set(f"ตรวจจาก En-face โฟลเดอร์: {os.path.basename(folder)}")
        except Exception as e:
            messagebox.showerror("ผิดพลาด", f"สร้าง/ตรวจสอบ En-face ไม่สำเร็จ: {e}", parent=self.root)

    def build_enface_from_bscans(self, folder: str, method: str = 'mip') -> Optional[np.ndarray]:
        """
        สร้างภาพ En-face จากโฟลเดอร์ B-scan ตามแนวทางในเปเปอร์ (Draft7)
        - method: 'mip' (maximum intensity projection) หรือ 'mean'
        Steps (assumed): per-slice normalization + optional CLAHE, then projection.
        """
        files = self._list_image_files(folder)
        if not files:
            return None
        first = cv2.imread(files[0], cv2.IMREAD_GRAYSCALE)
        if first is None:
            return None
        h, w = first.shape
        if method not in ('mip','mean'):
            method = 'mip'
        if method == 'mip':
            acc = np.zeros((h, w), dtype=np.uint8)
        else:
            acc = np.zeros((h, w), dtype=np.float32)

        # Pre-allocate CLAHE for consistency across slices
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

        count = 0
        for p in files:
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if img.shape != (h, w):
                try:
                    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
                except Exception:
                    continue
            # Normalize per slice to reduce illumination variation
            # Normalize to 0-255 using NumPy to avoid cv2 signature issues
            imin = float(img.min())
            imax = float(img.max())
            if imax > imin:
                img = ((img.astype(np.float32) - imin) / (imax - imin) * 255.0).astype(np.uint8)
            else:
                img = np.zeros_like(img, dtype=np.uint8)
            img = clahe.apply(img)
            if method == 'mip':
                acc = np.maximum(acc, img)
            else:  # mean
                acc += img.astype(np.float32)
            count += 1

        if count == 0:
            return None
        if method == 'mean':
            acc = (acc / float(count)).astype(np.uint8)

        enface = self._postprocess_enface(acc)
        return enface

    def _postprocess_enface(self, enface: np.ndarray) -> np.ndarray:
        """Post-process En-face to stabilize contrast and reduce speckle"""
        try:
            # Gentle denoise and contrast boost
            blur = cv2.GaussianBlur(enface, (3,3), 0)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            out = clahe.apply(blur)
            return out
        except Exception:
            return enface

    def _list_image_files(self, folder: str) -> List[str]:
        exts = {'.png','.jpg','.jpeg','.tif','.tiff','.bmp'}
        paths = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in exts]
        if not paths:
            return []
        # Natural sort by embedded numbers
        def _natural_key(s: str):
            b = os.path.basename(s)
            return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', b)]
        paths.sort(key=_natural_key)
        return paths

    # --- Utility and UI helper methods added to stabilize the app ---
    def _clear_canvas(self, canvas, placeholder_text=""):
        try:
            if not canvas or not canvas.winfo_exists():
                return
            canvas.delete("all")
            if placeholder_text:
                w = max(1, canvas.winfo_width())
                h = max(1, canvas.winfo_height())
                canvas.create_text(w//2, h//2, text=placeholder_text, fill="white", font=("Arial", 12), anchor="center")
        except tk.TclError:
            pass

    def _draw_placeholder(self, canvas, text):
        self._clear_canvas(canvas, text)

    def _get_db_cursor(self):
        try:
            if self.conn is None:
                self.conn = sqlite3.connect('fingerprint_db.sqlite')
                self.conn.execute("PRAGMA foreign_keys = ON;")
            return self.conn.cursor()
        except sqlite3.Error as e:
            print(f"DB cursor error: {e}")
            return None

    def _commit_db(self):
        try:
            if self.conn:
                self.conn.commit()
        except sqlite3.Error as e:
            print(f"DB commit error: {e}")

    def refresh_user_list(self):
        try:
            self.user_listbox.delete(0, tk.END)
            cur = self._get_db_cursor()
            if not cur:
                return
            cur.execute("SELECT id, name FROM users ORDER BY name")
            for uid, name in cur.fetchall():
                self.user_listbox.insert(tk.END, f"{uid}: {name}")
        except Exception as e:
            print(f"Refresh user list error: {e}")

    def refresh_admin_view(self):
        try:
            # Users treeview
            for item in self.users_treeview.get_children():
                self.users_treeview.delete(item)
            cur = self._get_db_cursor()
            if not cur:
                return
            cur.execute("SELECT id, name, created_at FROM users ORDER BY id DESC")
            for row in cur.fetchall():
                self.users_treeview.insert('', 'end', values=row)

            # Fingerprints list depends on selected user; clear for now
            for item in self.fp_treeview.get_children():
                self.fp_treeview.delete(item)
        except Exception as e:
            print(f"Refresh admin view error: {e}")

    def create_user(self):
        name = self.username_entry.get().strip()
        if not name:
            messagebox.showwarning("ชื่อว่าง", "กรุณาใส่ชื่อผู้ใช้", parent=self.root)
            return
        try:
            cur = self._get_db_cursor()
            if not cur:
                return
            cur.execute("INSERT INTO users (name, created_at) VALUES (?, ?)", (name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            self._commit_db()
            self.refresh_user_list()
            self.status_var.set(f"สร้างผู้ใช้: {name}")
        except sqlite3.IntegrityError:
            messagebox.showerror("ซ้ำ", "มีชื่อผู้ใช้นี้แล้ว", parent=self.root)
        except Exception as e:
            messagebox.showerror("ผิดพลาด", f"เพิ่มผู้ใช้ไม่ได้: {e}", parent=self.root)

    def on_user_select_register(self, event):
        try:
            sel = self.user_listbox.curselection()
            if not sel:
                self.current_user = None
                return
            text = self.user_listbox.get(sel[0])
            uid = int(text.split(':', 1)[0])
            self.current_user = uid
            self.load_user_fingerprints(uid, self.scan_canvas_register)
        except Exception as e:
            print(f"User select error: {e}")

    def on_admin_user_select(self, event):
        try:
            sel = self.users_treeview.selection()
            if not sel:
                return
            values = self.users_treeview.item(sel[0], 'values')
            uid = int(values[0])
            self.load_user_fingerprints(uid, self.admin_preview_canvas)
        except Exception as e:
            print(f"Admin user select error: {e}")

    def on_admin_fp_select(self, event):
        try:
            sel = self.fp_treeview.selection()
            if not sel:
                return
            values = self.fp_treeview.item(sel[0], 'values')
            path = values[2]
            if os.path.exists(path):
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    self.display_scan(img, self.admin_preview_canvas)
        except Exception as e:
            print(f"Admin fp select error: {e}")

    def load_user_fingerprints(self, user_id, preview_canvas=None):
        try:
            # Populate fingerprints table for selected user
            for item in self.fp_treeview.get_children():
                self.fp_treeview.delete(item)
            cur = self._get_db_cursor()
            if not cur:
                return
            cur.execute("SELECT id, user_id, image_path, scan_date FROM fingerprints WHERE user_id=? ORDER BY id DESC", (user_id,))
            rows = cur.fetchall()
            for row in rows:
                self.fp_treeview.insert('', 'end', values=row)
            # Preview first image
            if rows and preview_canvas is not None:
                path = rows[0][2]
                if os.path.exists(path):
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        self.display_scan(img, preview_canvas)
        except Exception as e:
            print(f"Load user fingerprints error: {e}")

    def delete_user(self):
        try:
            sel = self.users_treeview.selection()
            if not sel:
                messagebox.showinfo("เลือกผู้ใช้", "กรุณาเลือกผู้ใช้ก่อน", parent=self.root)
                return
            values = self.users_treeview.item(sel[0], 'values')
            uid = int(values[0])
            if not messagebox.askyesno("ยืนยัน", "ต้องการลบผู้ใช้นี้และลายนิ้วมือทั้งหมดหรือไม่?", parent=self.root):
                return
            cur = self._get_db_cursor()
            if not cur:
                return
            cur.execute("DELETE FROM users WHERE id=?", (uid,))
            self._commit_db()
            self.refresh_admin_view()
            self.refresh_user_list()
        except Exception as e:
            messagebox.showerror("ผิดพลาด", f"ลบผู้ใช้ไม่ได้: {e}", parent=self.root)

    def _reset_verification_results(self):
        try:
            self.result_user_var.set("-")
            self.match_score_var.set("-")
            self.verification_status_var.set("-")
            self.verification_status_label.configure(style='Normal.TLabel')
        except Exception:
            pass

    def _update_verification_ui(self, match_result, error_message=None):
        if error_message:
            self.result_user_var.set("-")
            self.match_score_var.set("-")
            self.verification_status_var.set(error_message)
            self.verification_status_label.configure(style='Error.TLabel')
            return
        if not match_result:
            self.result_user_var.set("ไม่พบ")
            self.match_score_var.set("0.0%")
            self.verification_status_var.set("ไม่ตรง")
            self.verification_status_label.configure(style='Failure.TLabel')
            return
        uid, name, score = match_result
        self.result_user_var.set(name)
        self.match_score_var.set(f"{score:.2f}%")
        if score >= 75.0:
            self.verification_status_var.set("ตรง")
            self.verification_status_label.configure(style='Success.TLabel')
            # Optional: open door on success
            try:
                self.open_door()
            except Exception:
                pass
        else:
            self.verification_status_var.set("ไม่ตรง")
            self.verification_status_label.configure(style='Failure.TLabel')

    def export_database(self):
        current_db_path="fingerprint_db.sqlite";
        try:
            if not self.conn:
                self.conn = sqlite3.connect('fingerprint_db.sqlite')
            res=self.conn.execute("PRAGMA database_list;").fetchone(); current_db_path=res[2] if res and len(res)>2 and res[2] else current_db_path
        except Exception as e: print(f"Warn: PRAGMA error: {e}. Using default.")
        try:
            dfn=f"fp_db_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sqlite"; fp=filedialog.asksaveasfilename(defaultextension=".sqlite",filetypes=[("SQLite","*.sqlite"),("DB","*.db"),("All","*.*")],title="ส่งออก...",initialfile=dfn,parent=self.root)
            if not fp: self.status_var.set("ยกเลิกส่งออก"); return
            shutil.copy2(current_db_path,fp); messagebox.showinfo("สำเร็จ",f"ส่งออกไป\n{fp}\nแล้ว",parent=self.root); self.status_var.set(f"ส่งออก: {os.path.basename(fp)}")
        except Exception as e: messagebox.showerror("ผิดพลาด",f"ส่งออกไม่สำเร็จ: {str(e)}",parent=self.root); self.status_var.set("ส่งออกล้มเหลว")

    def import_database(self):
        fp=filedialog.askopenfilename(filetypes=[("SQLite","*.sqlite"),("DB","*.db"),("All","*.*")],title="เลือกไฟล์นำเข้า",parent=self.root)
        if not fp: self.status_var.set("ยกเลิกนำเข้า"); return
        cfm=messagebox.askyesno("ยืนยัน","**คำเตือน:** เขียนทับ DB ปัจจุบัน!\nแนะนำให้ส่งออกก่อน\n\nดำเนินการต่อ?",icon='warning',parent=self.root)
        if not cfm: self.status_var.set("ยกเลิกนำเข้า"); return
        cdb="fingerprint_db.sqlite"; bp=None; cwo=False
        try:
            if not self.conn:
                self.conn = sqlite3.connect('fingerprint_db.sqlite')
            res=self.conn.execute("PRAGMA database_list;").fetchone(); cdb=res[2] if res and len(res)>2 and res[2] else cdb; self.conn.close(); cwo=True; print(f"Current DB: {cdb}")
            bp=cdb+f".backup_{datetime.now().strftime('%Y%m%d%H%M%S')}"; shutil.copy2(cdb,bp); print(f"Backed up: {bp}")
            shutil.copy2(fp,cdb); print(f"Copied {fp} over {cdb}")
            self.conn=sqlite3.connect(cdb); self.conn.execute("PRAGMA foreign_keys = ON;")
            cur=self.conn.cursor(); cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('users','fingerprints');"); tables={r[0] for r in cur.fetchall()}
            if 'users' not in tables or 'fingerprints' not in tables: raise ValueError("DB ไม่มีตารางที่จำเป็น")
            self.refresh_user_list(); self.refresh_admin_view(); self._clear_ui_state()
            messagebox.showinfo("สำเร็จ",f"นำเข้าจาก\n{fp}\nแล้ว",parent=self.root); self.status_var.set(f"นำเข้า: {os.path.basename(fp)}")
        except Exception as e:
            messagebox.showerror("ผิดพลาด",f"นำเข้าไม่ได้: {str(e)}\nกำลังกู้คืน...",parent=self.root)
            try:
                if self.conn: self.conn.close()
                if bp and os.path.exists(bp): shutil.copy2(bp,cdb); print(f"Restored: {bp}"); self.conn=sqlite3.connect(cdb); self.conn.execute("PRAGMA foreign_keys = ON;"); self.refresh_user_list(); self.refresh_admin_view(); self._clear_ui_state(); messagebox.showinfo("กู้คืนสำเร็จ","กู้คืน DB เดิมแล้ว",parent=self.root); self.status_var.set("กู้คืน DB เดิม")
                else: messagebox.showerror("กู้คืนล้มเหลว","ไม่พบไฟล์สำรอง",parent=self.root); self.status_var.set("นำเข้าล้มเหลว กู้คืนไม่ได้"); self.conn=None
            except Exception as rse: messagebox.showerror("ผิดพลาดร้ายแรง",f"กู้คืน DB ไม่ได้: {rse}",parent=self.root); self.conn=None; self.status_var.set("ข้อผิดพลาดร้ายแรง! กู้คืน DB ไม่ได้")
        finally:
             if self.conn is None and cwo:
                 try: print("Final reconnect..."); self.conn=sqlite3.connect(cdb); self.conn.execute("PRAGMA foreign_keys = ON;")
                 except Exception as final_e: messagebox.showerror("ผิดพลาดร้ายแรง",f"เปิด DB ไม่ได้: {final_e}",parent=self.root); self.conn=None

    def _clear_ui_state(self):
         self.current_user=None; self.current_scan=None; self.last_comparison_result=None; self.capture_bbox_register=None; self.capture_bbox_verify=None;
         self._clear_canvas(self.scan_canvas_register,"1. เลือกพื้นที่จับภาพ"); self._clear_canvas(self.scan_canvas_verify,"1. เลือกพื้นที่จับภาพ"); self._clear_canvas(self.admin_preview_canvas,"ภาพตัวอย่าง")
         # Don't clear HOG canvas if hidden
         # self._clear_canvas(self.hog_canvas,"HOG Visualization")
         self.user_listbox.selection_clear(0,tk.END)
         if self.users_treeview.selection(): self.users_treeview.selection_remove(self.users_treeview.selection())
         if self.fp_treeview.selection(): self.fp_treeview.selection_remove(self.fp_treeview.selection())
         self._reset_verification_results()
         self.save_scan_reg_btn.config(state=tk.DISABLED); # verify_btn removed
         self.select_area_reg_btn.config(state=tk.NORMAL); self.start_capture_reg_btn.config(state=tk.DISABLED); self.stop_capture_reg_btn.config(state=tk.DISABLED); self.capture_frame_reg_btn.config(state=tk.DISABLED)
         self.select_area_verify_btn.config(state=tk.NORMAL); self.toggle_live_verify_btn.config(state=tk.DISABLED); # Adjusted verify buttons
         self.area_info_reg_var.set("พื้นที่: -"); self.area_info_verify_var.set("พื้นที่: -")

# --- Main Execution ---
if __name__ == "__main__":
    root = None
    try:
        from ttkthemes import ThemedTk  # type: ignore
        # Use a preferred theme directly if available
        root = ThemedTk(theme='arc')
        print("Theme: arc")
    except ImportError:
        print("ttkthemes not found, using default.")
        root = tk.Tk()

    app = FingerprintSystem(root)

    def on_closing():
        if app.is_capturing_register:
            app.stop_capture_register()
        if app.is_verifying_live:
            app.stop_live_verify()  # Use stop_live_verify
        if messagebox.askokcancel("ปิดโปรแกรม", "ต้องการปิดโปรแกรมหรือไม่?"):
            try:
                app.close_mss()
                app.disconnect_arduino()  # Disconnect Arduino before closing
                if app.conn:
                    app.conn.close()
                    print("DB closed.")
            except Exception as e:
                print(f"Cleanup error: {e}")
            finally:
                try:
                    if root is not None:
                        root.destroy()
                except Exception:
                    pass

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

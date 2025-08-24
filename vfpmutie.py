import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Toplevel
import cv2
import numpy as np
from PIL import Image, ImageTk
import sqlite3
from datetime import datetime
# from skimage.feature import hog # ไม่ใช้ HOG แล้ว
# from skimage import exposure   # ไม่ใช้ HOG แล้ว
from skimage.morphology import skeletonize # อาจจำเป็นสำหรับ Minutiae Preprocessing
import json # สำหรับเก็บ Minutiae ใน DB
import mss
import mss.tools
import warnings
import time

# --- Class ImageCropper ---
# (เหมือนเดิม ไม่เปลี่ยนแปลง)
class ImageCropper(Toplevel):
    # ... (โค้ดเดิม) ...
    def __init__(self, parent, pil_image):
        super().__init__(parent)
        self.parent = parent; self.pil_image = pil_image; self.title("เลือกพื้นที่ที่ต้องการ")
        max_width = self.winfo_screenwidth()*0.8; max_height = self.winfo_screenheight()*0.8; self.geometry(f"{int(max_width)}x{int(max_height)}")
        self.resizable(True, True); self.grab_set(); self.focus_set(); self.transient(parent)
        self.start_x=None; self.start_y=None; self.current_x=None; self.current_y=None; self.selection_rect=None
        self.canvas = tk.Canvas(self, bg='darkgrey', cursor="cross"); self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        button_frame=ttk.Frame(self); button_frame.pack(pady=10, fill="x")
        self.info_label=ttk.Label(button_frame, text="คลิกและลากเพื่อเลือกพื้นที่ แล้วกด 'ยืนยัน'"); self.info_label.pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="ยกเลิก", command=self.cancel).pack(side=tk.RIGHT, padx=10); ttk.Button(button_frame, text="ยืนยัน", command=self.confirm).pack(side=tk.RIGHT, padx=10)
        self.scale = 1.0; self.offset_x = 0; self.offset_y = 0; self.display_width = 0; self.display_height = 0; self.photo_image = None
        self.display_image()
        self.canvas.bind("<Configure>", self.on_canvas_resize); self.canvas.bind("<ButtonPress-1>", self.on_button_press); self.canvas.bind("<B1-Motion>", self.on_mouse_drag); self.canvas.bind("<ButtonRelease-1>", self.on_button_release); self.bind("<Escape>", self.cancel)
        self.selected_bbox = None; self.wait_visibility(); self.center_window()
    def on_canvas_resize(self, event): self.after_idle(self.display_image)
    def display_image(self):
        canvas_width = self.canvas.winfo_width(); canvas_height = self.canvas.winfo_height()
        if canvas_width <= 1 or canvas_height <= 1:
            try: g=self.geometry(); sp=g.split('+')[0]; canvas_width=int(sp.split('x')[0])-20; canvas_height=int(sp.split('x')[1])-60;
            except (ValueError,IndexError,TypeError,AttributeError,tk.TclError): print("Warn: Geometry fallback."); canvas_width=780; canvas_height=540
        if canvas_width <= 1 or canvas_height <=1 : print("Warn: Invalid canvas dims."); return
        if not hasattr(self,'pil_image') or self.pil_image is None: print("Warn: No pil_image."); return
        img_width, img_height = self.pil_image.size
        if img_width <= 0 or img_height <= 0: print("Warn: Invalid image dims."); return
        scale_w=canvas_width/img_width; scale_h=canvas_height/img_height; self.scale=min(scale_w,scale_h)
        if self.scale>=1.0 or self.scale<=0: self.scale=1.0
        self.display_width=max(1,int(img_width*self.scale)); self.display_height=max(1,int(img_height*self.scale))
        try: dip=self.pil_image.resize((self.display_width,self.display_height),Image.Resampling.LANCZOS)
        except ValueError: print("Error: Resize failed."); return
        try: self.photo_image=ImageTk.PhotoImage(dip)
        except Exception as e: print(f"Error PhotoImage: {e}"); return
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
        if self.start_x is None or self.selection_rect is None: return
        self.current_x,self.current_y=self._canvas_to_display(event.x,event.y); self.canvas.coords(self.selection_rect,self.start_x,self.start_y,self.current_x,self.current_y)
    def on_button_release(self, event):
        if self.start_x is None or self.selection_rect is None:
             if self.selection_rect: self.canvas.delete(self.selection_rect); self.selection_rect=None
             self.start_x=None; return
        ex,ey=self._canvas_to_display(event.x,event.y); self.canvas.coords(self.selection_rect,self.start_x,self.start_y,ex,ey); self.current_x=ex; self.current_y=ey
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

# --- Class FingerprintSystem (Modified for Minutiae - Conceptual) ---
class FingerprintSystem:
    def __init__(self, root):
        self.root = root
        # --- ปรับ Title ---
        self.root.title("ระบบตรวจสอบลายนิ้วมือ OCT (v6.0 - Minutiae Concept)")
        self.root.geometry("1250x750")
        self.is_capturing_register=False; self.is_capturing_verify=False; self.capture_job_id_register=None; self.capture_job_id_verify=None; self.sct=None; self.capture_monitor=None; self.capture_bbox_register=None; self.capture_bbox_verify=None; self.last_frame_time=0
        self.is_verifying_live = False # สถานะการตรวจสอบสด
        self.live_verify_update_interval = 10 # ตรวจสอบทุกๆ กี่เฟรม (อาจต้องเพิ่มเพราะ Minutiae ช้ากว่า HOG)
        self.live_verify_frame_count = 0
        self.conn=sqlite3.connect('fingerprint_db_minutiae.sqlite'); # --- เปลี่ยนชื่อ DB ---
        self.conn.execute("PRAGMA foreign_keys = ON;"); self.create_tables()
        self.current_user=None; self.current_scan_processed=None # เก็บภาพที่ประมวลผลแล้วสำหรับบันทึก
        self.current_minutiae = None # เก็บ Minutiae ที่สกัดได้ล่าสุด
        self.last_comparison_result=None
        self.style=ttk.Style(self.root)
        try: dbg=self.style.lookup('TFrame','background'); self.style.configure('Normal.TLabel',foreground='black',background=dbg,font=('Arial',14,'bold')); self.style.configure('Success.TLabel',foreground='green',background='#d4ffcc',font=('Arial',14,'bold')); self.style.configure('Failure.TLabel',foreground='#cc0000',background='#ffcccc',font=('Arial',14,'bold')); self.style.configure('Error.TLabel',foreground='orange',background='#fff0cc',font=('Arial',14,'bold'))
        except tk.TclError: print("Warn: Fallback styling."); self.style.configure('Normal.TLabel',foreground='black',font=('Arial',14,'bold')); self.style.configure('Success.TLabel',foreground='green',font=('Arial',14,'bold')); self.style.configure('Failure.TLabel',foreground='#cc0000',font=('Arial',14,'bold')); self.style.configure('Error.TLabel',foreground='orange',font=('Arial',14,'bold'))
        self.create_ui(); self.init_mss()
        # --- Placeholder for Minutiae Extractor/Matcher (แนะนำให้ใช้ Library เช่น SourceAFIS) ---
        self.minutiae_extractor = self._placeholder_minutiae_extractor
        self.minutiae_matcher = self._placeholder_minutiae_matcher
        # Example using sourceafis (if installed):
        # try:
        #     from sourceafis import SourceAFIS
        #     self.minutiae_extractor = SourceAFIS.extract # หรือวิธีเรียกใช้ที่ถูกต้อง
        #     self.minutiae_matcher = SourceAFIS.verify # หรือวิธีเรียกใช้ที่ถูกต้อง
        #     print("Using SourceAFIS library.")
        # except ImportError:
        #     print("SourceAFIS not found, using placeholders.")
        #     self.minutiae_extractor = self._placeholder_minutiae_extractor
        #     self.minutiae_matcher = self._placeholder_minutiae_matcher
        # --- End Placeholder ---

    def init_mss(self):
         # ... (โค้ดเดิม) ...
         try: self.sct=mss.mss(); self.capture_monitor = self.sct.monitors[1] if len(self.sct.monitors)>1 else self.sct.monitors[0]; print(f"Monitor: {self.capture_monitor}")
         except Exception as e: messagebox.showerror("Error",f"mss init failed: {e}",parent=self.root); self.sct=None

    def close_mss(self):
        # ... (โค้ดเดิม) ...
        if self.sct: self.sct.close(); print("mss closed.")

    def create_tables(self):
        cur=self.conn.cursor()
        cur.execute('CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL)')
        # --- เปลี่ยน schema ตาราง fingerprints ---
        cur.execute('''CREATE TABLE IF NOT EXISTS fingerprints (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        image_path TEXT NOT NULL, -- ยังคงเก็บ path ภาพต้นฉบับ (หลัง preprocess)
                        scan_date TEXT NOT NULL,
                        minutiae_data TEXT, -- เก็บ Minutiae เป็น JSON Text
                        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                     )''')
        self.conn.commit()

    def create_ui(self):
        # ... (ส่วนใหญ่เหมือนเดิม ยกเว้น HOG display ที่ถูกเอาออกไปแล้ว) ...
        self.tab_control=ttk.Notebook(self.root); self.register_tab=ttk.Frame(self.tab_control); self.verify_tab=ttk.Frame(self.tab_control); self.admin_tab=ttk.Frame(self.tab_control); self.tab_control.add(self.register_tab,text="ลงทะเบียน"); self.tab_control.add(self.verify_tab,text="ตรวจสอบ"); self.tab_control.add(self.admin_tab,text="ดูแลระบบ"); self.tab_control.pack(expand=1,fill="both",padx=5,pady=5)
        self.setup_register_tab(); self.setup_verify_tab(); self.setup_admin_tab()
        self.status_var=tk.StringVar(value="พร้อมใช้งาน"); sb=ttk.Label(self.root,textvariable=self.status_var,relief="sunken",anchor="w",padding=(5,2)); sb.pack(side="bottom",fill="x")

    def setup_register_tab(self):
        # ... (เหมือนเดิม) ...
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
        self.save_scan_reg_btn=ttk.Button(ccf,text="4. บันทึก",command=self.save_scan,state=tk.DISABLED); self.save_scan_reg_btn.pack(side=tk.LEFT,padx=10)
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
        # ... (เหมือนเดิม - ไม่มี HOG display แล้ว) ...
        lf=ttk.LabelFrame(self.verify_tab,text="จอแสดงผล / ตรวจสอบสด",padding=(10,5)); lf.grid(row=0,column=0,padx=10,pady=10,sticky="nsew")
        ccf=ttk.Frame(lf); ccf.pack(pady=5,fill="x")
        self.select_area_verify_btn=ttk.Button(ccf,text="1. เลือกพื้นที่",command=lambda: self.select_capture_area("verify")); self.select_area_verify_btn.pack(side=tk.LEFT,padx=5)
        self.toggle_live_verify_btn=ttk.Button(ccf,text="2. เริ่มตรวจสอบสด",command=self.toggle_live_verify, state=tk.DISABLED); self.toggle_live_verify_btn.pack(side=tk.LEFT,padx=10)
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
        self.verify_tab.columnconfigure(0,weight=1); self.verify_tab.columnconfigure(1,weight=1); self.verify_tab.rowconfigure(0,weight=1); lf.rowconfigure(1,weight=1); self.results_frame.columnconfigure(1,weight=1);

    def setup_admin_tab(self):
        # --- ปรับ Treeview สำหรับ Fingerprints ---
        cf=ttk.Frame(self.admin_tab,padding=(0,5)); cf.pack(padx=10,pady=5,fill="x"); ttk.Button(cf,text="รีเฟรช",command=self.refresh_admin_view).pack(side=tk.LEFT,padx=5); ttk.Button(cf,text="ลบผู้ใช้",command=self.delete_user).pack(side=tk.LEFT,padx=5); ttk.Button(cf,text="ส่งออก",command=self.export_database).pack(side=tk.LEFT,padx=5); ttk.Button(cf,text="นำเข้า",command=self.import_database).pack(side=tk.LEFT,padx=5); dp=ttk.PanedWindow(self.admin_tab,orient=tk.HORIZONTAL); dp.pack(padx=10,pady=5,fill="both",expand=True); uf=ttk.LabelFrame(dp,text="ผู้ใช้",padding=(10,5)); dp.add(uf,weight=1); self.users_treeview=ttk.Treeview(uf,columns=("id","name","created_at"),show="headings"); us=ttk.Scrollbar(uf,orient="vertical",command=self.users_treeview.yview); self.users_treeview.configure(yscrollcommand=us.set); self.users_treeview.heading("id",text="ID",anchor="center"); self.users_treeview.heading("name",text="ชื่อ"); self.users_treeview.heading("created_at",text="สร้างเมื่อ"); self.users_treeview.column("id",width=50,anchor="center",stretch=False); self.users_treeview.column("name",width=200); self.users_treeview.column("created_at",width=150); us.pack(side="right",fill="y"); self.users_treeview.pack(fill="both",expand=True); self.users_treeview.bind('<<TreeviewSelect>>',self.on_admin_user_select); fpf=ttk.Frame(dp); dp.add(fpf,weight=2); fpfr=ttk.LabelFrame(fpf,text="ลายนิ้วมือ",padding=(10,5)); fpfr.pack(padx=0,pady=0,fill="both",expand=True);
        # --- เปลี่ยนคอลัมน์ fp_treeview ---
        self.fp_treeview=ttk.Treeview(fpfr,columns=("id","user_id","path","date", "minutiae_count"),show="headings");
        fps=ttk.Scrollbar(fpfr,orient="vertical",command=self.fp_treeview.yview); self.fp_treeview.configure(yscrollcommand=fps.set);
        self.fp_treeview.heading("id",text="FP ID",anchor="center"); self.fp_treeview.heading("user_id",text="User ID",anchor="center"); self.fp_treeview.heading("path",text="ที่เก็บไฟล์"); self.fp_treeview.heading("date",text="วันที่สแกน"); self.fp_treeview.heading("minutiae_count", text="จุด Minutiae", anchor="center") # คอลัมน์ใหม่
        self.fp_treeview.column("id",width=60,anchor="center",stretch=False); self.fp_treeview.column("user_id",width=60,anchor="center",stretch=False); self.fp_treeview.column("path",width=250); self.fp_treeview.column("date",width=150); self.fp_treeview.column("minutiae_count", width=80, anchor="center", stretch=False) # คอลัมน์ใหม่
        fps.pack(side="right",fill="y"); self.fp_treeview.pack(fill="both",expand=True); self.fp_treeview.bind('<<TreeviewSelect>>',self.on_admin_fp_select); self.admin_preview_frame=ttk.LabelFrame(fpf,text="ภาพตัวอย่าง",padding=(5,5)); self.admin_preview_frame.pack(padx=0,pady=(10,0),fill="x",expand=False); self.admin_preview_canvas=tk.Canvas(self.admin_preview_frame,bg="lightgrey",height=150); self.admin_preview_canvas.pack(fill="x",expand=True); self.admin_preview_canvas.bind("<Configure>",lambda e: self._draw_placeholder(self.admin_preview_canvas,"ภาพตัวอย่าง")); self._draw_placeholder(self.admin_preview_canvas,"ภาพตัวอย่าง"); self.refresh_admin_view()

    def _capture_fullscreen(self):
        # ... (โค้ดเดิม) ...
        if not self.sct or not self.capture_monitor: print("Error: mss not initialized."); return None
        try:
            with mss.mss(display=getattr(self.sct,'display',None)) as sct_fs: sct_img = sct_fs.grab(self.capture_monitor)
            img_pil = Image.frombytes("RGB", sct_img.size, sct_img.rgb); img_pil_gray = img_pil.convert('L'); return np.array(img_pil_gray)
        except Exception as e: print(f"ERROR: Capture fullscreen failed: {e}"); return None

    def select_capture_area(self, mode):
        # ... (โค้ดเดิม - ส่วนจับภาพหน้าจอและ Cropper) ...
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
                # --- แสดงภาพสีเทาตั้งต้น ---
                cp_gray=full_img_np[y1:y2,x1:x2].copy()
                if mode=="register":
                    self.capture_bbox_register=bd; self.area_info_reg_var.set(ainfo); self.start_capture_reg_btn.config(state=tk.NORMAL); self.display_scan(cp_gray,self.scan_canvas_register) # แสดงภาพ Gray
                    self.stop_capture_reg_btn.config(state=tk.DISABLED); self.capture_frame_reg_btn.config(state=tk.DISABLED); self.save_scan_reg_btn.config(state=tk.DISABLED);
                    self.current_scan_processed = None; self.current_minutiae = None # รีเซ็ตค่า
                else: # verify mode
                    self.capture_bbox_verify=bd; self.area_info_verify_var.set(ainfo); self.toggle_live_verify_btn.config(state=tk.NORMAL)
                    self.display_scan(cp_gray,self.scan_canvas_verify); self._reset_verification_results(); # แสดงภาพ Gray
                    self.current_scan_processed = None; self.current_minutiae = None # รีเซ็ตค่า
                self.status_var.set(f"เลือกพื้นที่แล้ว {ainfo}")
            else: # User cancelled Cropper
                if mode=="register": self.capture_bbox_register=None; self.start_capture_reg_btn.config(state=tk.DISABLED); self.area_info_reg_var.set("พื้นที่: -")
                else: self.capture_bbox_verify=None; self.toggle_live_verify_btn.config(state=tk.DISABLED); self.area_info_verify_var.set("พื้นที่: -")
                self.status_var.set("ยกเลิกเลือกพื้นที่")
        except Exception as e: messagebox.showerror("ผิดพลาด",f"เลือกพื้นที่ผิดพลาด: {e}",parent=self.root); self.status_var.set("เลือกพื้นที่ล้มเหลว")


    def start_capture_register(self):
        # ... (โค้ด UI ควบคุมเหมือนเดิม) ...
        if self.is_capturing_register or self.is_verifying_live: messagebox.showwarning("กำลังทำงาน","แสดงผลสดอยู่แล้ว",parent=self.root); return
        if not self.sct: messagebox.showerror("Error","mss ไม่พร้อม",parent=self.root); return
        if not self.capture_bbox_register: messagebox.showerror("Error","เลือกพื้นที่ก่อน",parent=self.root); return
        self.is_capturing_register=True; self.select_area_reg_btn.config(state=tk.DISABLED); self.start_capture_reg_btn.config(state=tk.DISABLED); self.stop_capture_reg_btn.config(state=tk.NORMAL); self.capture_frame_reg_btn.config(state=tk.NORMAL); self.save_scan_reg_btn.config(state=tk.DISABLED);
        self.select_area_verify_btn.config(state=tk.DISABLED); self.toggle_live_verify_btn.config(state=tk.DISABLED)
        self.status_var.set("แสดงผลสด (Register Area)..."); self.capture_loop(self.scan_canvas_register,"register")

    def stop_capture_register(self):
        # ... (โค้ด UI ควบคุมเหมือนเดิม) ...
        if self.capture_job_id_register: self.root.after_cancel(self.capture_job_id_register); self.capture_job_id_register=None
        self.is_capturing_register=False; self.select_area_reg_btn.config(state=tk.NORMAL); self.start_capture_reg_btn.config(state=tk.NORMAL if self.capture_bbox_register else tk.DISABLED); self.stop_capture_reg_btn.config(state=tk.DISABLED); self.capture_frame_reg_btn.config(state=tk.DISABLED)
        self.select_area_verify_btn.config(state=tk.NORMAL); self.toggle_live_verify_btn.config(state=tk.NORMAL if self.capture_bbox_verify else tk.DISABLED)
        self.status_var.set("หยุดแสดงผลสด (Register)")
        # เช็คว่ามีภาพที่ถูก capture ค้างไว้หรือไม่ ถ้าไม่มีให้แสดง placeholder
        has_captured_image = self.current_scan_processed is not None
        if not has_captured_image and hasattr(self.scan_canvas_register, f"img_ref_{self.scan_canvas_register.winfo_name()}"):
             # ตรวจสอบเพิ่มเติมว่า canvas มี image tag อยู่จริงหรือไม่
             if not self.scan_canvas_register.find_withtag("image"):
                  self._draw_placeholder(self.scan_canvas_register, "1. เลือกพื้นที่จับภาพ")
        elif not has_captured_image:
             self._draw_placeholder(self.scan_canvas_register, "1. เลือกพื้นที่จับภาพ")


    def capture_current_frame_register(self):
        # ... (โค้ดเดิม) ...
        if not self.is_capturing_register: messagebox.showinfo("ข้อมูล", "ต้องกด 'เริ่มแสดงผลสด' ก่อน", parent=self.root); return
        self._capture_frame_action(self.scan_canvas_register, "register")

    def toggle_live_verify(self):
        # ... (โค้ดเดิม) ...
        if self.is_verifying_live: self.stop_live_verify()
        else: self.start_live_verify()

    def start_live_verify(self):
        # ... (โค้ด UI ควบคุมเหมือนเดิม) ...
        if self.is_capturing_register or self.is_verifying_live: messagebox.showwarning("กำลังทำงาน","แสดงผล/ตรวจสอบสดอยู่แล้ว",parent=self.root); return
        if not self.sct: messagebox.showerror("Error","mss ไม่พร้อม",parent=self.root); return
        if not self.capture_bbox_verify: messagebox.showerror("Error","เลือกพื้นที่ก่อน",parent=self.root); return
        self.is_verifying_live = True; self.live_verify_frame_count = 0; self.toggle_live_verify_btn.config(text="หยุดตรวจสอบสด")
        self.select_area_verify_btn.config(state=tk.DISABLED)
        self.select_area_reg_btn.config(state=tk.DISABLED); self.start_capture_reg_btn.config(state=tk.DISABLED); self.stop_capture_reg_btn.config(state=tk.DISABLED); self.capture_frame_reg_btn.config(state=tk.DISABLED); self.save_scan_reg_btn.config(state=tk.DISABLED)
        self.status_var.set("กำลังตรวจสอบสด..."); self._reset_verification_results(); self.capture_loop(self.scan_canvas_verify, "verify")

    def stop_live_verify(self):
        # ... (โค้ด UI ควบคุมเหมือนเดิม) ...
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
            sct_img = self.sct.grab(current_bbox)
            img_bgr = np.array(sct_img)
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2GRAY) # ใช้ภาพ grayscale

            # --- แสดงภาพ Gray ที่จับได้สดๆ ---
            self.display_scan(img_gray, canvas=target_canvas)

            if is_live_verify_mode:
                self.live_verify_frame_count += 1
                if self.live_verify_frame_count % self.live_verify_update_interval == 0:
                    try:
                        # --- ประมวลผลและสกัด Minutiae สำหรับการตรวจสอบสด ---
                        processed_img = self.preprocess_fingerprint(img_gray)
                        current_minutiae = self.extract_minutiae(processed_img)

                        if current_minutiae is not None and len(current_minutiae) > 0: # ตรวจสอบว่ามี Minutiae หรือไม่
                            match_result = self._compare_features(current_minutiae)
                            self._update_verification_ui(match_result)
                        elif current_minutiae is None:
                             self._update_verification_ui(None, error_message="Minutiae Error")
                        else: # current_minutiae is empty list
                             self._update_verification_ui(None, error_message="No Minutiae")

                    except Exception as proc_e:
                        print(f"Error during live verify processing: {proc_e}")
                        self._update_verification_ui(None, error_message="Processing Error")

            delay_ms = 50 # อาจจะต้องเพิ่ม delay เพราะการประมวลผล Minutiae ช้ากว่า
            job_id = self.root.after(delay_ms, lambda: self.capture_loop(target_canvas, mode))
            if mode == "register": self.capture_job_id_register = job_id
            else: self.capture_job_id_verify = job_id

        except Exception as e:
            print(f"Capture loop error ({mode}, bbox={current_bbox}): {e}")
            if mode == "register": self.stop_capture_register()
            else: self.stop_live_verify()
            if time.time() - getattr(self, f'_{mode}_loop_error_time', 0) > 5: messagebox.showerror("Capture Error", f"แสดงผลสดผิดพลาด: {e}", parent=self.root); setattr(self, f'_{mode}_loop_error_time', time.time())

    # --- ฟังก์ชัน Core ที่เปลี่ยนแปลง ---

    def preprocess_fingerprint(self, img):
        """ปรับปรุงการ Preprocess สำหรับ Minutiae (อาจต้องปรับเพิ่ม)"""
        if img is None: raise ValueError("Input image is None")
        if len(img.shape) != 2:
            if len(img.shape) == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else: raise ValueError(f"Bad shape: {img.shape}")

        # 1. Resize (อาจจะไม่จำเป็นเสมอไป ขึ้นอยู่กับคุณภาพต้นฉบับ)
        # ts = (300, 300)
        # try: img_r = cv2.resize(img, ts, interpolation=cv2.INTER_AREA)
        # except cv2.error as e: print(f"Resize error: {e}"); img_r = img
        img_r = img # ลองไม่ resize ก่อน

        # 2. Contrast Enhancement (CLAHE ยังคงมีประโยชน์)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_c = clahe.apply(img_r)

        # 3. Binarization (Adaptive Threshold ยังใช้ได้ดี)
        img_b = cv2.adaptiveThreshold(img_c, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        # 4. (Optional but Recommended) Morphological Operations (Noise Reduction)
        # kernel = np.ones((3,3),np.uint8)
        # img_b = cv2.morphologyEx(img_b, cv2.MORPH_CLOSE, kernel)
        # img_b = cv2.morphologyEx(img_b, cv2.MORPH_OPEN, kernel)

        # 5. (Important for some extractors) Thinning/Skeletonization
        # ต้องการภาพที่ Binarize เป็น 0 (background) และ 1 (foreground)
        img_b_norm = img_b // 255
        with warnings.catch_warnings(): # ปิด warning จาก skimage ถ้ามี
            warnings.simplefilter("ignore")
            img_skel = skeletonize(img_b_norm).astype(np.uint8) * 255
        return img_skel # คืนภาพที่ทำ Thinning แล้ว

    def extract_minutiae(self, processed_img):
        """
        สกัด Minutiae จากภาพที่ผ่าน Preprocessing แล้ว (Conceptual)
        คืนค่า list ของ minutiae points: [(x, y, type, angle), ...]
        *** แนะนำให้ใช้ Library เช่น SourceAFIS แทนการเขียนเอง ***
        """
        if processed_img is None: return None
        # ใช้ Placeholder หรือเรียก Library
        return self.minutiae_extractor(processed_img)


    def _placeholder_minutiae_extractor(self, processed_img):
        """Placeholder สำหรับการสกัด Minutiae (คืนค่าจำลอง/ว่าง)"""
        print("--- Using Placeholder Minutiae Extractor ---")
        # การ Implement จริงต้อง:
        # 1. หา Candidate points โดยใช้ Crossing Number (CN) บนภาพ skeleton
        #    - วนรอบ 3x3 ของแต่ละ pixel บนสันลาย (ค่า=255)
        #    - คำนวณ CN = 0.5 * sum(|P[i] - P[i+1]|) for i=1 to 8 (P9=P1)
        #    - CN=1 -> Ending, CN=3 -> Bifurcation
        # 2. คำนวณ Angle/Orientation ของแต่ละ Minutia
        # 3. Filter out spurious/false minutiae (เช่น ที่ขอบภาพ, ใกล้กันเกินไป)
        # ตัวอย่างข้อมูลคืนค่าที่ควรจะเป็น (แต่ต้องคำนวณจริง):
        # dummy_minutiae = [
        #     {'x': 50, 'y': 100, 'type': 'ending', 'angle': np.pi / 2},
        #     {'x': 150, 'y': 120, 'type': 'bifurcation', 'angle': -np.pi / 4},
        # ]
        # return dummy_minutiae
        h, w = processed_img.shape[:2]
        # คืนค่าเป็น list ว่าง หรือ list จำลองเล็กน้อย เพื่อให้โค้ดส่วนอื่นทำงานต่อได้
        # การคืนค่า list ว่าง จะทำให้ไม่เกิดการ match
        # อาจจะใส่จุดจำลอง 2-3 จุด เพื่อทดสอบ flow
        if np.sum(processed_img > 0) > 100 : # ถ้ามี pixel สีขาวพอสมควร
             return [{'x': w//3, 'y': h//3, 'type': 'ending', 'angle': 0.0},
                     {'x': 2*w//3, 'y': 2*h//3, 'type': 'bifurcation', 'angle': 1.57}]
        return [] # คืนค่า list ว่างถ้าภาพไม่น่าจะมี minutiae


    def match_minutiae(self, minutiae_set1, minutiae_set2):
        """
        เปรียบเทียบชุด Minutiae สองชุด (Conceptual)
        คืนค่าคะแนนความเหมือน (Score) ซึ่งนิยามขึ้นกับอัลกอริธึมที่ใช้
        *** แนะนำให้ใช้ Library เช่น SourceAFIS แทนการเขียนเอง ***
        """
        # ใช้ Placeholder หรือเรียก Library
        return self.minutiae_matcher(minutiae_set1, minutiae_set2)


    def _placeholder_minutiae_matcher(self, minutiae_set1, minutiae_set2):
        """Placeholder สำหรับการเปรียบเทียบ Minutiae (คืนค่าจำลอง)"""
        print("--- Using Placeholder Minutiae Matcher ---")
        if not minutiae_set1 or not minutiae_set2:
            return 0.0 # ไม่มีจุดให้เทียบ

        # การ Implement จริงต้องซับซ้อนกว่านี้มาก เช่น:
        # 1. Alignment: หาคู่ reference และปรับแนว minutiae_set2 ให้เข้ากับ minutiae_set1
        # 2. Pairing: หาคู่ minutiae ที่ใกล้เคียงกันหลัง alignment
        # 3. Scoring: คำนวณคะแนนจากจำนวนคู่ที่ match
        # --- ค่าจำลองง่ายๆ ---
        # ลองนับจำนวน minutiae ที่ใกล้เคียงกันแบบง่ายๆ (ไม่แม่นยำจริง)
        matched_count = 0
        threshold_dist = 10 # ระยะห่างสูงสุดที่ยอมรับ (pixel)
        threshold_angle = np.pi / 6 # มุมต่างสูงสุดที่ยอมรับ (radian)

        set2_used = [False] * len(minutiae_set2)

        for m1 in minutiae_set1:
            best_match_idx = -1
            min_dist_sq = threshold_dist**2

            for i, m2 in enumerate(minutiae_set2):
                 if not set2_used[i] and m1['type'] == m2['type']:
                     dist_sq = (m1['x'] - m2['x'])**2 + (m1['y'] - m2['y'])**2
                     if dist_sq < min_dist_sq:
                         angle_diff = abs(m1['angle'] - m2['angle'])
                         # Normalize angle difference (handle wrap around pi)
                         angle_diff = min(angle_diff, 2*np.pi - angle_diff)
                         if angle_diff < threshold_angle:
                             min_dist_sq = dist_sq
                             best_match_idx = i

            if best_match_idx != -1:
                matched_count += 1
                set2_used[best_match_idx] = True

        # คำนวณ score แบบง่ายๆ (เปอร์เซ็นต์ของจุดที่ match เทียบกับค่าเฉลี่ยจำนวนจุด)
        # อาจจะไม่ใช่ score ที่ดีนักสำหรับการใช้งานจริง
        avg_len = (len(minutiae_set1) + len(minutiae_set2)) / 2.0
        score = (matched_count / avg_len) * 100.0 if avg_len > 0 else 0.0
        return min(score, 100.0) # จำกัดไม่ให้เกิน 100

    def _compare_features(self, current_minutiae):
        """เปรียบเทียบ Minutiae ที่สกัดได้กับข้อมูลในฐานข้อมูล"""
        best_match_user_id = None
        best_match_user_name = None
        highest_score = -1.0
        best_match_fp_id = None # เก็บ ID ของ fingerprint ที่ match ดีสุด

        if current_minutiae is None or len(current_minutiae) == 0:
             print("No current minutiae to compare.")
             return None # ไม่มีข้อมูลปัจจุบันให้เปรียบเทียบ

        try:
            cur = self.conn.cursor()
            # --- ดึงข้อมูล minutiae_data ---
            cur.execute("""
                SELECT f.id, f.user_id, f.minutiae_data, u.name
                FROM fingerprints f
                JOIN users u ON f.user_id = u.id
                WHERE f.minutiae_data IS NOT NULL AND f.minutiae_data != ''
            """)
            fingerprint_db = cur.fetchall()

            if not fingerprint_db:
                print("No fingerprints with minutiae data in DB.")
                return None # ไม่มีข้อมูลใน DB ให้เปรียบเทียบ

            for fp_id, user_id, minutiae_json, user_name in fingerprint_db:
                try:
                    # --- แปลง JSON Text กลับเป็น List ---
                    stored_minutiae = json.loads(minutiae_json)
                    if not stored_minutiae: # ถ้าข้อมูลที่เก็บเป็น list ว่าง
                        continue

                    # --- เรียกใช้ Minutiae Matcher ---
                    # ควรส่งข้อมูลในรูปแบบที่ Matcher ต้องการ (อาจจะเป็น list of dicts หรือ object เฉพาะ)
                    score = self.match_minutiae(current_minutiae, stored_minutiae)

                    # print(f"Compared with FP ID {fp_id} (User: {user_name}), Score: {score:.2f}") # Debug

                    # --- ใช้เกณฑ์คะแนนที่ได้จาก Matcher ---
                    # คะแนนจาก Minutiae Matcher อาจจะมีความหมายต่างจาก Cosine Similarity
                    # ค่า 0-100 อาจจะเหมาะสม แต่ threshold อาจจะต้องปรับ
                    if score > highest_score:
                        highest_score = score
                        best_match_user_id = user_id
                        best_match_user_name = user_name
                        best_match_fp_id = fp_id

                except json.JSONDecodeError:
                    print(f"Warn: Invalid JSON minutiae data for FP ID {fp_id}")
                    continue
                except Exception as match_err:
                    print(f"Error matching with FP ID {fp_id}: {match_err}")
                    continue # เปรียบเทียบรายการถัดไป

            if best_match_user_name is not None:
                 print(f"Best Match: User {best_match_user_name} (ID:{best_match_user_id}), FP ID: {best_match_fp_id}, Score: {highest_score:.2f}")
                 return (best_match_user_id, best_match_user_name, highest_score)
            else:
                 print("No suitable match found.")
                 return None

        except sqlite3.Error as e:
            print(f"DB Error during comparison: {e}")
            return None
        except Exception as e:
            print(f"Error during comparison: {e}")
            return None

    def _update_verification_ui(self, match_result, error_message=None):
        if error_message:
            self.result_user_var.set("-")
            self.match_score_var.set("-")
            self.verification_status_var.set(error_message)
            self.verification_status_label.configure(style='Error.TLabel')
            return

        # --- ปรับ Threshold สำหรับ Minutiae Matching Score ---
        # ค่านี้ต้องปรับจูนอย่างระมัดระวัง ขึ้นอยู่กับ Matcher ที่ใช้
        # อาจจะต้องทดลองหาค่าที่เหมาะสม
        verification_threshold = 30.0 # ** ค่าสมมติ ต้องปรับแก้ **

        if match_result:
            bmuid, bmun, hs = match_result
            dn = f"{bmun} (ID: {bmuid})"
            self.result_user_var.set(dn)
            self.match_score_var.set(f"{hs:.2f}") # แสดงคะแนนดิบจาก Matcher

            if hs >= verification_threshold:
                self.verification_status_var.set("ยืนยันสำเร็จ")
                self.verification_status_label.configure(style='Success.TLabel')
            else:
                self.verification_status_var.set("ยืนยันไม่สำเร็จ (ต่ำกว่าเกณฑ์)")
                self.verification_status_label.configure(style='Failure.TLabel')
        else:
            self.result_user_var.set("ไม่พบการจับคู่")
            self.match_score_var.set("N/A")
            self.verification_status_var.set("ไม่พบการจับคู่")
            self.verification_status_label.configure(style='Failure.TLabel')


    def _capture_frame_action(self, source_canvas, mode):
        """จับภาพเฟรมปัจจุบัน, ประมวลผล, และสกัด Minutiae (สำหรับ Register)"""
        if not self.sct: messagebox.showerror("Error","ไม่พร้อมจับภาพ",parent=self.root); return
        current_bbox = self.capture_bbox_register if mode=="register" else self.capture_bbox_verify
        if not current_bbox: messagebox.showerror("Error","ยังไม่ได้เลือกพื้นที่",parent=self.root); return

        was_capturing_reg = False
        if mode == "register" and self.is_capturing_register:
            was_capturing_reg = True
            self.stop_capture_register() # หยุด live feed ชั่วคราวเพื่อจับภาพนิ่ง

        # ไม่หยุด live verify ตอนจับภาพ (เพราะ verify ทำงานบน live feed อยู่แล้ว)
        if mode == "register":
            self.root.after(100) # Delay ให้ UI อัปเดต

        try:
            sct_img = self.sct.grab(current_bbox)
            img_bgr = np.array(sct_img)
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2GRAY)

            # --- ประมวลผลภาพและสกัด Minutiae ---
            self.current_scan_processed = self.preprocess_fingerprint(img_gray)
            self.current_minutiae = self.extract_minutiae(self.current_scan_processed)

            minutiae_count = len(self.current_minutiae) if self.current_minutiae else 0
            self.status_var.set(f"จับภาพเฟรม: {current_bbox['width']}x{current_bbox['height']}, พบ {minutiae_count} Minutiae")

            # --- แสดงภาพที่ *ผ่านการประมวลผลแล้ว* บน Canvas ---
            # อาจจะแสดงภาพ skeleton หรือภาพ binarized ก็ได้
            self.display_scan(self.current_scan_processed, canvas=source_canvas)

            if mode == "register":
                # เปิดปุ่ม Save ถ้ามี Minutiae และเลือก User แล้ว
                can_save = self.current_user is not None and self.current_minutiae is not None and minutiae_count > 0
                self.save_scan_reg_btn.config(state=tk.NORMAL if can_save else tk.DISABLED)
                # ถ้าเคยเปิด live feed ไว้ ให้เปิดกลับ (อาจจะไม่จำเป็น ถ้า user กด stop เอง)
                # if was_capturing_reg:
                #     self.start_capture_register()

        except Exception as e:
            messagebox.showerror("Capture Error", f"จับภาพเฟรมไม่ได้: {e}", parent=self.root)
            self.current_scan_processed = None
            self.current_minutiae = None
            if mode == "register":
                self.save_scan_reg_btn.config(state=tk.DISABLED)
            self._clear_canvas(source_canvas, "จับภาพเฟรมล้มเหลว")
            # Restart live feed if it was stopped? Maybe not, let user restart.

    def _reset_verification_results(self):
         # ... (เหมือนเดิม) ...
         self.result_user_var.set("-"); self.match_score_var.set("-"); self.verification_status_var.set("-")
         if hasattr(self,'verification_status_label'): self.verification_status_label.configure(style='Normal.TLabel')

    def _draw_placeholder(self, canvas, text):
         # ... (เหมือนเดิม) ...
         try:
             if not canvas.winfo_exists(): return
             canvas.delete("placeholder"); width=canvas.winfo_width(); height=canvas.winfo_height()
             if width>1 and height>1:
                 is_capturing=(canvas==self.scan_canvas_register and self.is_capturing_register)or(canvas==self.scan_canvas_verify and self.is_verifying_live)
                 if not canvas.find_withtag("image") and not is_capturing: canvas.create_text(width/2,height/2,text=text,fill="darkgrey",font=("Arial",10),tags="placeholder",width=width*0.9)
         except tk.TclError: pass

    def refresh_user_list(self):
        # ... (เหมือนเดิม) ...
        self.user_listbox.delete(0,tk.END); self.users_data={}
        try: cur=self.conn.cursor(); cur.execute("SELECT id, name FROM users ORDER BY name"); users=cur.fetchall()
        except sqlite3.Error as e: messagebox.showerror("DB Error",f"Load users failed: {e}"); return
        for uid,name in users: dt=f"{name} (ID: {uid})"; self.user_listbox.insert(tk.END,dt); self.users_data[dt]=uid

    def refresh_admin_view(self):
        # ... (เหมือนเดิม แต่เพิ่มข้อมูล minutiae count) ...
        for item in self.users_treeview.get_children(): self.users_treeview.delete(item)
        for item in self.fp_treeview.get_children(): self.fp_treeview.delete(item)
        self._clear_canvas(self.admin_preview_canvas,"ภาพตัวอย่าง")
        try: cur=self.conn.cursor(); cur.execute("SELECT id, name, created_at FROM users ORDER BY id"); users=cur.fetchall()
        except sqlite3.Error as e: messagebox.showerror("DB Error",f"Load admin users failed: {e}"); return
        for user in users: self.users_treeview.insert("","end",values=user)
        # โหลดข้อมูล FP ทีหลัง เมื่อเลือก User

    def on_user_select_register(self, event):
        # ... (เหมือนเดิม แต่ปรับการ enable ปุ่ม save) ...
        sel=self.user_listbox.curselection()
        if not sel:
             self.current_user=None; self.status_var.set("ไม่มีผู้ใช้");
             self._clear_canvas(self.scan_canvas_register,"1. เลือกพื้นที่จับภาพ");
             self.save_scan_reg_btn.config(state=tk.DISABLED);
             self.current_scan_processed = None; self.current_minutiae = None; return
        st=self.user_listbox.get(sel[0]); uid=self.users_data.get(st)
        if uid:
             self.current_user=uid; self.status_var.set(f"เลือก: {st}");
             self.load_user_fingerprints(uid,self.scan_canvas_register); # โหลดภาพตัวอย่างล่าสุด
             # Enable ปุ่ม Save ถ้ามีภาพที่ Capture ไว้แล้ว และมี Minutiae
             can_save = self.current_minutiae is not None and len(self.current_minutiae) > 0
             self.save_scan_reg_btn.config(state=tk.NORMAL if can_save else tk.DISABLED)
        else:
             self.current_user=None; self.status_var.set("เลือกผิดพลาด");
             self._clear_canvas(self.scan_canvas_register,"1. เลือกพื้นที่จับภาพ");
             self.save_scan_reg_btn.config(state=tk.DISABLED)
             self.current_scan_processed = None; self.current_minutiae = None;

    def on_admin_user_select(self, event):
        # ... (เหมือนเดิม แต่เพิ่มการดึง minutiae_data และแสดงจำนวน) ...
        sel=self.users_treeview.selection();
        for item in self.fp_treeview.get_children(): self.fp_treeview.delete(item); self._clear_canvas(self.admin_preview_canvas,"ภาพตัวอย่าง")
        if not sel: return
        item=self.users_treeview.item(sel[0]); uid=item['values'][0]
        try:
            cur=self.conn.cursor()
            # --- ดึง minutiae_data มาด้วย ---
            cur.execute("SELECT id, user_id, image_path, scan_date, minutiae_data FROM fingerprints WHERE user_id = ? ORDER BY scan_date DESC",(uid,))
            fps=cur.fetchall()
        except sqlite3.Error as e: messagebox.showerror("DB Error",f"Load FPs failed (User {uid}): {e}",parent=self.root); return

        for fp in fps:
            fp_id, user_id, img_path, scan_date, minutiae_json = fp
            minutiae_count_str = "N/A"
            if minutiae_json:
                try:
                    m_data = json.loads(minutiae_json)
                    minutiae_count_str = str(len(m_data))
                except json.JSONDecodeError:
                    minutiae_count_str = "Error"
                except TypeError: # Handle if m_data is not list/dict
                    minutiae_count_str = "Invalid"

            # --- เพิ่ม minutiae_count ใน values ---
            self.fp_treeview.insert("","end",values=(fp_id, user_id, img_path, scan_date, minutiae_count_str))


    def on_admin_fp_select(self, event):
        # ... (เหมือนเดิม - แสดงภาพ Preview จาก image_path) ...
        sel=self.fp_treeview.selection();
        if not sel: self._clear_canvas(self.admin_preview_canvas,"ภาพตัวอย่าง"); return
        item=self.fp_treeview.item(sel[0]); ip=item['values'][2] # index 2 คือ image_path
        try:
            if ip and os.path.exists(ip):
                 # โหลดภาพสีเทาจาก path ที่บันทึกไว้
                 img=cv2.imread(ip,cv2.IMREAD_GRAYSCALE);
                 self.display_scan(img,canvas=self.admin_preview_canvas) if img is not None else self._clear_canvas(self.admin_preview_canvas,f"โหลดไม่ได้:\n{os.path.basename(ip)}")
            elif not ip: self._clear_canvas(self.admin_preview_canvas,"ไม่มี Path")
            else: self._clear_canvas(self.admin_preview_canvas,f"ไม่พบ:\n{os.path.basename(ip)}")
        except Exception as e: print(f"Preview error: {e}"); self._clear_canvas(self.admin_preview_canvas,"ข้อผิดพลาด")


    def _clear_canvas(self, canvas, placeholder_text):
         # ... (เหมือนเดิม) ...
         if canvas and canvas.winfo_exists(): canvas.delete("all"); self._draw_placeholder(canvas,placeholder_text)

    def load_user_fingerprints(self, user_id, target_canvas):
        """โหลดภาพตัวอย่างล่าสุดของผู้ใช้ (จาก image_path)"""
        placeholder="1. เลือกพื้นที่จับภาพ"
        self._clear_canvas(target_canvas,"กำลังโหลด...")
        try:
            cur=self.conn.cursor()
            # --- ดึง image_path ล่าสุด ---
            cur.execute("SELECT image_path FROM fingerprints WHERE user_id=? ORDER BY scan_date DESC LIMIT 1",(user_id,))
            res=cur.fetchone()
        except sqlite3.Error as e: self.status_var.set(f"DB Error: {e}"); self._clear_canvas(target_canvas,"DB Error"); return
        except Exception as e: self.status_var.set(f"Error: {e}"); self._clear_canvas(target_canvas,"Error"); return

        if res and res[0]: ip=res[0]
        else: self.status_var.set(f"ไม่พบข้อมูลลายนิ้วมือสำหรับ ID {user_id}"); self._clear_canvas(target_canvas,placeholder); return

        if os.path.exists(ip):
            try:
                # --- โหลดภาพสีเทา ---
                img=cv2.imread(ip,cv2.IMREAD_GRAYSCALE)
            except Exception as load_e:
                print(f"Error loading image {ip}: {load_e}")
                img = None
        else:
            self.status_var.set(f"ไม่พบไฟล์ภาพ: {os.path.basename(ip)}"); self._clear_canvas(target_canvas,placeholder); return

        if img is not None:
             self.display_scan(img,canvas=target_canvas); self.status_var.set(f"แสดงตัวอย่างล่าสุด ID {user_id}")
        else:
             self.status_var.set(f"โหลดภาพตัวอย่างไม่ได้: {os.path.basename(ip)}"); self._clear_canvas(target_canvas,placeholder); return

    def create_user(self):
        # ... (เหมือนเดิม) ...
        name=self.username_entry.get().strip();
        if not name: messagebox.showerror("ผิดพลาด","ใส่ชื่อผู้ใช้",parent=self.root); return
        try: cur=self.conn.cursor(); cur.execute("INSERT INTO users (name,created_at) VALUES (?,?)",(name,datetime.now().strftime("%Y-%m-%d %H:%M:%S"))); self.conn.commit(); self.username_entry.delete(0,tk.END); self.refresh_user_list(); self.refresh_admin_view(); messagebox.showinfo("สำเร็จ",f"สร้าง '{name}' แล้ว",parent=self.root)
        except sqlite3.IntegrityError: messagebox.showwarning("ชื่อซ้ำ",f"'{name}' มีอยู่แล้ว",parent=self.root); self.conn.rollback()
        except sqlite3.Error as e: messagebox.showerror("DB Error",f"สร้างไม่ได้: {e}",parent=self.root); self.conn.rollback()

    def delete_user(self):
        # ... (เหมือนเดิม แต่ต้องลบไฟล์ภาพที่เกี่ยวข้องด้วย) ...
        sel=self.users_treeview.selection();
        if not sel: messagebox.showwarning("เลือกก่อน","เลือกผู้ใช้",parent=self.root); return
        item=self.users_treeview.item(sel[0]); uid=item['values'][0]; uname=item['values'][1]
        cfm=messagebox.askyesno("ยืนยัน",f"ลบ '{uname}' (ID:{uid}) และข้อมูลลายนิ้วมือทั้งหมด?\n**ไม่สามารถย้อนกลับได้**",icon='warning',parent=self.root);
        if not cfm: return
        try:
             cur=self.conn.cursor();
             # --- ดึง path ไฟล์ภาพที่จะลบ ---
             cur.execute("SELECT image_path FROM fingerprints WHERE user_id=?",(uid,));
             image_paths=[r[0] for r in cur.fetchall() if r[0]] # เก็บ path ทั้งหมด

             # --- ลบข้อมูลจาก DB (FK constraint จะลบ fingerprints ที่เกี่ยวข้องด้วย) ---
             cur.execute("DELETE FROM users WHERE id=?",(uid,));
             deleted_rows=cur.rowcount;
             self.conn.commit()

             if deleted_rows > 0:
                 # --- ลบไฟล์ภาพที่เก็บไว้ ---
                 for img_path in image_paths:
                     try:
                         if os.path.exists(img_path):
                              os.remove(img_path); print(f"Deleted image file: {img_path}")
                     except OSError as e:
                          print(f"Warn: Cannot delete image file {img_path}: {e}")

                 self.refresh_user_list(); self.refresh_admin_view()
                 if self.current_user==uid:
                      self.current_user=None; self.user_listbox.selection_clear(0,tk.END);
                      self._clear_canvas(self.scan_canvas_register,"1. เลือกพื้นที่จับภาพ");
                      self.status_var.set("ผู้ใช้ถูกลบแล้ว")
                      self.current_scan_processed = None; self.current_minutiae = None;
                      self.save_scan_reg_btn.config(state=tk.DISABLED)
                 messagebox.showinfo("สำเร็จ",f"ลบ '{uname}' และข้อมูลที่เกี่ยวข้องแล้ว",parent=self.root)
             else: messagebox.showerror("ผิดพลาด",f"ไม่พบผู้ใช้ ID {uid} ที่จะลบ",parent=self.root)
        except sqlite3.Error as e: messagebox.showerror("DB Error",f"ลบผู้ใช้ไม่ได้: {e}",parent=self.root); self.conn.rollback()
        except Exception as e: messagebox.showerror("ผิดพลาด", f"เกิดข้อผิดพลาดในการลบ: {e}", parent=self.root); self.conn.rollback()

    def save_scan(self):
        """บันทึกภาพที่ Process แล้ว และ Minutiae ที่สกัดได้"""
        # --- ตรวจสอบว่ามีข้อมูล Minutiae และภาพที่ Process แล้ว ---
        if self.current_minutiae is None or len(self.current_minutiae) == 0:
            messagebox.showwarning("ไม่มีข้อมูล", "ต้อง 'จับภาพ' และสกัด Minutiae ให้ได้ก่อน", parent=self.root)
            return
        if self.current_scan_processed is None:
             messagebox.showwarning("ไม่มีภาพ", "ไม่พบภาพที่ผ่านการประมวลผล", parent=self.root)
             return
        if self.current_user is None:
            messagebox.showwarning("ไม่ได้เลือก", "เลือกผู้ใช้ก่อนบันทึก", parent=self.root)
            return

        save_dir = "fingerprints_processed"; os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f"); user_name = "unknown"; filename = None

        try:
            cur=self.conn.cursor(); cur.execute("SELECT name FROM users WHERE id=?",(self.current_user,)); res=cur.fetchone(); user_name = res[0].replace(" ","_") if res else user_name
        except Exception as name_e: print(f"Warn: get user name failed: {name_e}")

        # --- บันทึกภาพที่ Processed (เช่น ภาพ Skeleton) ---
        filename_base = f"user_{self.current_user}_{user_name}_{timestamp}_processed.png";
        filename = os.path.abspath(os.path.join(save_dir,filename_base))

        try:
            # --- บันทึกภาพที่ผ่าน Preprocessing ---
            success = cv2.imwrite(filename, self.current_scan_processed);
            if not success: raise IOError(f"imwrite failed for processed image: {filename}")

            # --- แปลง Minutiae เป็น JSON ---
            # ตรวจสอบให้แน่ใจว่าข้อมูล Minutiae สามารถ Serialize เป็น JSON ได้
            # ถ้าใช้ Object พิเศษจาก Library อาจจะต้องแปลงเป็น Dict มาตรฐานก่อน
            try:
                 minutiae_serializable = []
                 for m in self.current_minutiae:
                      # แปลง numpy ค่า (ถ้ามี) เป็น float/int ปกติ
                      serializable_m = {}
                      for key, value in m.items():
                          if isinstance(value, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
                              serializable_m[key] = int(value)
                          elif isinstance(value, (np.float_, np.float16, np.float32, np.float64)):
                              serializable_m[key] = float(value)
                          elif isinstance(value, (np.ndarray,)): # Handle numpy arrays if any (e.g., orientation vector) - convert to list
                              serializable_m[key] = value.tolist()
                          else:
                              serializable_m[key] = value # Assume standard types (str, int, float, list, dict)
                      minutiae_serializable.append(serializable_m)

                 minutiae_json = json.dumps(minutiae_serializable)
            except TypeError as json_err:
                 raise TypeError(f"Minutiae data not JSON serializable: {json_err}")


            # --- บันทึกข้อมูลลง DB ---
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO fingerprints (user_id, image_path, scan_date, minutiae_data)
                VALUES (?, ?, ?, ?)
            """, (self.current_user, filename, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), minutiae_json))
            self.conn.commit()

            self.status_var.set(f"บันทึก: {filename_base}");
            self.refresh_admin_view(); # อัปเดต Admin view
            # โหลดภาพตัวอย่างล่าสุดหลังจากบันทึก
            self.load_user_fingerprints(self.current_user,self.scan_canvas_register);
            messagebox.showinfo("สำเร็จ","บันทึกลายนิ้วมือ (Minutiae) แล้ว",parent=self.root)

            # รีเซ็ตสถานะหลังบันทึกสำเร็จ และ disable ปุ่ม save
            self.current_scan_processed = None
            self.current_minutiae = None
            self.save_scan_reg_btn.config(state=tk.DISABLED)

        except (sqlite3.Error, IOError, TypeError, Exception) as e:
            messagebox.showerror("ผิดพลาด", f"บันทึกลายนิ้วมือไม่ได้: {str(e)}", parent=self.root)
            self.conn.rollback()
            # พยายามลบไฟล์ภาพที่อาจถูกสร้างขึ้นแต่บันทึก DB ไม่สำเร็จ
            if filename and os.path.exists(filename):
                try: os.remove(filename); print(f"Removed orphaned processed file due to save error: {filename}")
                except OSError as rem_e: print(f"Warning: Could not remove orphaned file {filename}: {rem_e}")


    # --- display_scan, export, import, clear_ui ไม่ต้องแก้มากนัก ---
    def display_scan(self, img_np, canvas, max_width=None, max_height=None):
        # ... (เหมือนเดิม - แสดงภาพ numpy array บน canvas) ...
        if img_np is None: self._clear_canvas(canvas,"ไม่มีภาพ"); return
        cname = "unknown_canvas"; nw, nh = 0, 0
        try:
            if not canvas or not canvas.winfo_exists(): print("Warn: Invalid canvas."); return
            cname=canvas.winfo_name()
            # --- จัดการ Input Image (Grayscale หรือ RGB) ---
            if len(img_np.shape)==2: # Grayscale
                 img_pil=Image.fromarray(img_np)
            elif len(img_np.shape)==3 and img_np.shape[2]==3: # RGB
                 img_pil=Image.fromarray(img_np)
            elif len(img_np.shape)==3 and img_np.shape[2]==1: # Grayscale with channel dim
                 img_pil=Image.fromarray(img_np.squeeze(), 'L')
            elif len(img_np.shape)==3 and img_np.shape[2]==4: # BGRA from MSS
                 img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGRA2RGB)
                 img_pil = Image.fromarray(img_rgb)
            else: raise ValueError(f"Unsupported image shape for display: {img_np.shape}")

            cw=canvas.winfo_width(); ch=canvas.winfo_height();
            if cw<=1: cw=max_width if max_width else 600
            if ch<=1: ch=max_height if max_height else 400
            tw=max_width if max_width else cw; th=max_height if max_height else ch; iw,ih=img_pil.size;
            if iw<=0 or ih<=0: print(f"Warn: Invalid image dims {cname}: {iw}x{ih}"); self._clear_canvas(canvas,"ขนาดภาพผิด"); return
            if tw<=0 or th<=0: print(f"Warn: Invalid target dims {cname}: {tw}x{th}"); self._clear_canvas(canvas,"ขนาด Canvas ผิด"); return

            scale=min(tw/iw,th/ih)
            # --- ปรับปรุง: อนุญาตให้ขยายภาพเล็กน้อยได้ ถ้า scale < 1 ---
            if scale < 1.0 : # Only downscale or keep original size
                scale = min(scale, 1.0)
            # Ensure scale is positive
            if scale <= 0: scale = 1.0

            nw=max(1,int(iw*scale)); nh=max(1,int(ih*scale));
            img_r=img_pil.resize((nw,nh),Image.Resampling.LANCZOS); img_tk=ImageTk.PhotoImage(img_r); ref=f"img_ref_{cname}"; setattr(canvas,ref,img_tk)
            canvas.delete("all"); xp=max(0,(cw-nw)//2); yp=max(0,(ch-nh)//2); canvas.create_image(xp,yp,anchor="nw",image=img_tk,tags="image")

        except Exception as e:
            print(f"ERROR display {cname} (shape: {img_np.shape if img_np is not None else 'None'}): {e}")
            import traceback
            traceback.print_exc() # พิมพ์ Traceback เพื่อดูรายละเอียดข้อผิดพลาด
            self._clear_canvas(canvas,"แสดงภาพไม่ได้")


    def export_database(self):
        # --- ควรเปลี่ยน default filename ---
        current_db_path="fingerprint_db_minutiae.sqlite";
        try: res=self.conn.execute("PRAGMA database_list;").fetchone(); current_db_path=res[2] if res and len(res)>2 and res[2] else current_db_path
        except Exception as e: print(f"Warn: PRAGMA error: {e}. Using default.")
        try:
            dfn=f"fp_db_minutiae_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sqlite"; fp=filedialog.asksaveasfilename(defaultextension=".sqlite",filetypes=[("SQLite","*.sqlite"),("DB","*.db"),("All","*.*")],title="ส่งออกฐานข้อมูล Minutiae...",initialfile=dfn,parent=self.root)
            if not fp: self.status_var.set("ยกเลิกส่งออก"); return
            import shutil; shutil.copy2(current_db_path,fp); messagebox.showinfo("สำเร็จ",f"ส่งออกฐานข้อมูลไปที่\n{fp}\nแล้ว",parent=self.root); self.status_var.set(f"ส่งออก: {os.path.basename(fp)}")
        except Exception as e: messagebox.showerror("ผิดพลาด",f"ส่งออกฐานข้อมูลไม่สำเร็จ: {str(e)}",parent=self.root); self.status_var.set("ส่งออกล้มเหลว")

    def import_database(self):
        # --- ควรเปลี่ยน default filename ---
        fp=filedialog.askopenfilename(filetypes=[("SQLite Minutiae DB","*.sqlite"),("DB","*.db"),("All","*.*")],title="เลือกไฟล์ฐานข้อมูล Minutiae ที่จะนำเข้า",parent=self.root)
        if not fp: self.status_var.set("ยกเลิกนำเข้า"); return
        cfm=messagebox.askyesno("ยืนยันการนำเข้า","**คำเตือน:** การดำเนินการนี้จะเขียนทับฐานข้อมูลปัจจุบัน!\nข้อมูลเก่าทั้งหมดจะสูญหาย แนะนำให้ส่งออกข้อมูลปัจจุบันก่อน\n\nคุณต้องการดำเนินการนำเข้าต่อหรือไม่?",icon='warning',parent=self.root)
        if not cfm: self.status_var.set("ยกเลิกนำเข้า"); return

        current_db_file="fingerprint_db_minutiae.sqlite"; backup_file=None; was_conn_open=False

        try:
            # ตรวจสอบ path ปัจจุบัน และปิด connection ก่อน copy
            try:
                res=self.conn.execute("PRAGMA database_list;").fetchone();
                current_db_file=res[2] if res and len(res)>2 and res[2] else current_db_file;
            except Exception as e: print(f"Warn: PRAGMA error getting db path: {e}. Using default.")

            if self.conn:
                 self.conn.close(); was_conn_open = True; print(f"Current DB closed: {current_db_file}")

            import shutil
            # สร้าง Backup
            backup_file=current_db_file+f".backup_{datetime.now().strftime('%Y%m%d%H%M%S')}";
            shutil.copy2(current_db_file,backup_file); print(f"Backed up current DB to: {backup_file}")

            # Copy ไฟล์ที่เลือกทับไฟล์ปัจจุบัน
            shutil.copy2(fp, current_db_file); print(f"Copied imported file {fp} over {current_db_file}")

            # เชื่อมต่อ DB ใหม่ และตรวจสอบ Schema เบื้องต้น
            self.conn=sqlite3.connect(current_db_file); self.conn.execute("PRAGMA foreign_keys = ON;")
            cur=self.conn.cursor()
            # --- ตรวจสอบตาราง fingerprints และคอลัมน์ minutiae_data ---
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='fingerprints';");
            if not cur.fetchone(): raise ValueError("ฐานข้อมูลที่นำเข้าไม่มีตาราง 'fingerprints'")
            cur.execute("PRAGMA table_info(fingerprints);")
            columns = [info[1] for info in cur.fetchall()]
            if 'minutiae_data' not in columns:
                 raise ValueError("ตาราง 'fingerprints' ในฐานข้อมูลที่นำเข้าไม่มีคอลัมน์ 'minutiae_data'")
            if 'users' not in [tbl[0] for tbl in cur.execute("SELECT name FROM sqlite_master WHERE type='table';")]:
                 raise ValueError("ฐานข้อมูลที่นำเข้าไม่มีตาราง 'users'")


            # ถ้าผ่านการตรวจสอบ -> Refresh UI
            self.refresh_user_list(); self.refresh_admin_view(); self._clear_ui_state()
            messagebox.showinfo("สำเร็จ",f"นำเข้าข้อมูลจาก\n{fp}\nแล้ว",parent=self.root); self.status_var.set(f"นำเข้าสำเร็จ: {os.path.basename(fp)}")

        except Exception as e:
            messagebox.showerror("ผิดพลาด",f"นำเข้าฐานข้อมูลไม่ได้: {str(e)}\nกำลังพยายามกู้คืนจากไฟล์สำรอง...",parent=self.root)
            try:
                # ปิด connection ถ้ายังเปิดอยู่
                if self.conn: self.conn.close(); self.conn = None;

                # กู้คืนจาก Backup ถ้ามี
                if backup_file and os.path.exists(backup_file):
                    import shutil
                    shutil.copy2(backup_file, current_db_file); print(f"Restored DB from backup: {backup_file}")
                    # เชื่อมต่อ DB ที่กู้คืนมาใหม่
                    self.conn=sqlite3.connect(current_db_file); self.conn.execute("PRAGMA foreign_keys = ON;")
                    # Refresh UI อีกครั้ง
                    self.refresh_user_list(); self.refresh_admin_view(); self._clear_ui_state();
                    messagebox.showinfo("กู้คืนสำเร็จ","กู้คืนฐานข้อมูลเดิมจากไฟล์สำรองแล้ว",parent=self.root); self.status_var.set("กู้คืน DB เดิมสำเร็จ")
                else:
                    messagebox.showerror("กู้คืนล้มเหลว","ไม่พบไฟล์สำรอง ไม่สามารถกู้คืนฐานข้อมูลได้",parent=self.root);
                    self.status_var.set("นำเข้าล้มเหลว และกู้คืนไม่ได้");
                    # อาจจะต้องปิดโปรแกรม หรือสร้าง DB ใหม่
                    self.conn=None # ตั้งเป็น None เพื่อบ่งชี้ว่า DB ไม่พร้อมใช้งาน
            except Exception as rse:
                messagebox.critical("ผิดพลาดร้ายแรง",f"กู้คืนฐานข้อมูลจาก Backup ไม่สำเร็จ: {rse}",parent=self.root);
                self.conn=None; self.status_var.set("ข้อผิดพลาดร้ายแรง! กู้คืน DB ไม่ได้")
        finally:
             # ถ้า conn เป็น None หลังจากพยายามทั้งหมด ให้ลองเชื่อมต่อใหม่เผื่อกรณีฉุกเฉิน
             if self.conn is None and was_conn_open:
                 try:
                     print("Attempting final reconnect to DB...");
                     self.conn=sqlite3.connect(current_db_file); self.conn.execute("PRAGMA foreign_keys = ON;")
                     print("Final reconnect successful.")
                 except Exception as final_e:
                     messagebox.critical("ผิดพลาดร้ายแรง",f"ไม่สามารถเปิดการเชื่อมต่อฐานข้อมูลสุดท้ายได้: {final_e}",parent=self.root);
                     self.conn=None # ยืนยันว่าเปิดไม่ได้

    def _clear_ui_state(self):
         # ... (เหมือนเดิม แต่ปรับตาม state ที่เปลี่ยนไป) ...
         self.current_user=None;
         self.current_scan_processed = None; self.current_minutiae = None; # รีเซ็ต state ใหม่
         self.last_comparison_result=None; self.capture_bbox_register=None; self.capture_bbox_verify=None;
         self._clear_canvas(self.scan_canvas_register,"1. เลือกพื้นที่จับภาพ"); self._clear_canvas(self.scan_canvas_verify,"1. เลือกพื้นที่จับภาพ"); self._clear_canvas(self.admin_preview_canvas,"ภาพตัวอย่าง")
         self.user_listbox.selection_clear(0,tk.END)
         if self.users_treeview.selection(): self.users_treeview.selection_remove(self.users_treeview.selection())
         if self.fp_treeview.selection(): self.fp_treeview.selection_remove(self.fp_treeview.selection())
         self._reset_verification_results()
         self.save_scan_reg_btn.config(state=tk.DISABLED);
         self.select_area_reg_btn.config(state=tk.NORMAL); self.start_capture_reg_btn.config(state=tk.DISABLED); self.stop_capture_reg_btn.config(state=tk.DISABLED); self.capture_frame_reg_btn.config(state=tk.DISABLED)
         self.select_area_verify_btn.config(state=tk.NORMAL); self.toggle_live_verify_btn.config(state=tk.DISABLED);
         self.area_info_reg_var.set("พื้นที่: -"); self.area_info_verify_var.set("พื้นที่: -")


# --- Main Execution ---
if __name__ == "__main__":
    root = None
    try:
        from ttkthemes import ThemedTk
        available_themes = ThemedTk().get_themes(); preferred_themes = ['arc', 'plastik', 'adapta', 'aqua']; chosen_theme = 'default'
        for theme in preferred_themes:
            if theme in available_themes: chosen_theme = theme; break
        root = ThemedTk(theme=chosen_theme); print(f"Theme: {chosen_theme}")
    except ImportError: print("ttkthemes not found, using default."); root = tk.Tk()

    app = FingerprintSystem(root)

    def on_closing():
        if app.is_capturing_register: app.stop_capture_register()
        if app.is_verifying_live: app.stop_live_verify()
        if messagebox.askokcancel("ปิดโปรแกรม", "ต้องการปิดโปรแกรมหรือไม่?", parent=root):
            try:
                app.close_mss();
                if app.conn: app.conn.close(); print("DB closed.")
            except Exception as e: print(f"Cleanup error: {e}")
            finally: root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()
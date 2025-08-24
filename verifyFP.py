import os
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

# --- Class ImageCropper ---
class ImageCropper(Toplevel):
    # (โค้ด ImageCropper เหมือนเดิมจาก v4.3 ไม่เปลี่ยนแปลง)
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

# --- Class FingerprintSystem ---
class FingerprintSystem:
    def __init__(self, root):
        self.root = root; self.root.title("ระบบตรวจสอบลายนิ้วมือ OCT (v5.2 - Hide HOG)"); self.root.geometry("1250x750") # Updated title
        self.is_capturing_register=False; self.is_capturing_verify=False; self.capture_job_id_register=None; self.capture_job_id_verify=None; self.sct=None; self.capture_monitor=None; self.capture_bbox_register=None; self.capture_bbox_verify=None; self.last_frame_time=0
        self.is_verifying_live = False # สถานะการตรวจสอบสด
        self.live_verify_update_interval = 5 # ตรวจสอบทุกๆ กี่เฟรม (ปรับได้)
        self.live_verify_frame_count = 0
        self.conn=sqlite3.connect('fingerprint_db.sqlite'); self.conn.execute("PRAGMA foreign_keys = ON;"); self.create_tables()
        self.current_user=None; self.current_scan=None; self.last_comparison_result=None
        self.style=ttk.Style(self.root)
        try: dbg=self.style.lookup('TFrame','background'); self.style.configure('Normal.TLabel',foreground='black',background=dbg,font=('Arial',14,'bold')); self.style.configure('Success.TLabel',foreground='green',background='#d4ffcc',font=('Arial',14,'bold')); self.style.configure('Failure.TLabel',foreground='#cc0000',background='#ffcccc',font=('Arial',14,'bold')); self.style.configure('Error.TLabel',foreground='orange',background='#fff0cc',font=('Arial',14,'bold'))
        except tk.TclError: print("Warn: Fallback styling."); self.style.configure('Normal.TLabel',foreground='black',font=('Arial',14,'bold')); self.style.configure('Success.TLabel',foreground='green',font=('Arial',14,'bold')); self.style.configure('Failure.TLabel',foreground='#cc0000',font=('Arial',14,'bold')); self.style.configure('Error.TLabel',foreground='orange',font=('Arial',14,'bold'))
        self.create_ui(); self.init_mss()

    def init_mss(self):
         try: self.sct=mss.mss(); self.capture_monitor = self.sct.monitors[1] if len(self.sct.monitors)>1 else self.sct.monitors[0]; print(f"Monitor: {self.capture_monitor}")
         except Exception as e: messagebox.showerror("Error",f"mss init failed: {e}",parent=self.root); self.sct=None

    def close_mss(self):
        if self.sct: self.sct.close(); print("mss closed.")

    def create_tables(self):
        cur=self.conn.cursor(); cur.execute('CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL)'); cur.execute('CREATE TABLE IF NOT EXISTS fingerprints (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, image_path TEXT NOT NULL, scan_date TEXT NOT NULL, hog_features BLOB, FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE)'); self.conn.commit()

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
        cf=ttk.Frame(self.admin_tab,padding=(0,5)); cf.pack(padx=10,pady=5,fill="x"); ttk.Button(cf,text="รีเฟรช",command=self.refresh_admin_view).pack(side=tk.LEFT,padx=5); ttk.Button(cf,text="ลบผู้ใช้",command=self.delete_user).pack(side=tk.LEFT,padx=5); ttk.Button(cf,text="ส่งออก",command=self.export_database).pack(side=tk.LEFT,padx=5); ttk.Button(cf,text="นำเข้า",command=self.import_database).pack(side=tk.LEFT,padx=5); dp=ttk.PanedWindow(self.admin_tab,orient=tk.HORIZONTAL); dp.pack(padx=10,pady=5,fill="both",expand=True); uf=ttk.LabelFrame(dp,text="ผู้ใช้",padding=(10,5)); dp.add(uf,weight=1); self.users_treeview=ttk.Treeview(uf,columns=("id","name","created_at"),show="headings"); us=ttk.Scrollbar(uf,orient="vertical",command=self.users_treeview.yview); self.users_treeview.configure(yscrollcommand=us.set); self.users_treeview.heading("id",text="ID",anchor="center"); self.users_treeview.heading("name",text="ชื่อ"); self.users_treeview.heading("created_at",text="สร้างเมื่อ"); self.users_treeview.column("id",width=50,anchor="center",stretch=False); self.users_treeview.column("name",width=200); self.users_treeview.column("created_at",width=150); us.pack(side="right",fill="y"); self.users_treeview.pack(fill="both",expand=True); self.users_treeview.bind('<<TreeviewSelect>>',self.on_admin_user_select); fpf=ttk.Frame(dp); dp.add(fpf,weight=2); fpfr=ttk.LabelFrame(fpf,text="ลายนิ้วมือ",padding=(10,5)); fpfr.pack(padx=0,pady=0,fill="both",expand=True); self.fp_treeview=ttk.Treeview(fpfr,columns=("id","user_id","path","date"),show="headings"); fps=ttk.Scrollbar(fpfr,orient="vertical",command=self.fp_treeview.yview); self.fp_treeview.configure(yscrollcommand=fps.set); self.fp_treeview.heading("id",text="FP ID",anchor="center"); self.fp_treeview.heading("user_id",text="User ID",anchor="center"); self.fp_treeview.heading("path",text="ที่เก็บไฟล์"); self.fp_treeview.heading("date",text="วันที่สแกน"); self.fp_treeview.column("id",width=60,anchor="center",stretch=False); self.fp_treeview.column("user_id",width=60,anchor="center",stretch=False); self.fp_treeview.column("path",width=250); self.fp_treeview.column("date",width=150); fps.pack(side="right",fill="y"); self.fp_treeview.pack(fill="both",expand=True); self.fp_treeview.bind('<<TreeviewSelect>>',self.on_admin_fp_select); self.admin_preview_frame=ttk.LabelFrame(fpf,text="ภาพตัวอย่าง",padding=(5,5)); self.admin_preview_frame.pack(padx=0,pady=(10,0),fill="x",expand=False); self.admin_preview_canvas=tk.Canvas(self.admin_preview_frame,bg="lightgrey",height=150); self.admin_preview_canvas.pack(fill="x",expand=True); self.admin_preview_canvas.bind("<Configure>",lambda e: self._draw_placeholder(self.admin_preview_canvas,"ภาพตัวอย่าง")); self._draw_placeholder(self.admin_preview_canvas,"ภาพตัวอย่าง"); self.refresh_admin_view()

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
        best_match_user_id = None; best_match_user_name = None; highest_score = -1.0
        try:
            cur=self.conn.cursor(); cur.execute("SELECT f.id,f.user_id,f.image_path,f.hog_features,u.name FROM fingerprints f JOIN users u ON f.user_id=u.id WHERE f.hog_features IS NOT NULL"); fdb=cur.fetchall()
            if not fdb: return None
            for fid,uid,ip,hb,un in fdb:
                 try: sh=np.frombuffer(hb,dtype=np.float64);
                 except ValueError: print(f"Warn: Invalid HOG data for FP ID {fid}"); continue
                 if current_hog_features.shape!=sh.shape: continue
                 scr=self.calculate_similarity(current_hog_features,sh)
                 if scr>highest_score: hs=scr; bmuid=uid; bmun=un; highest_score = hs
            if bmun is not None: return (bmuid, bmun, highest_score)
            else: return None
        except sqlite3.Error as e: print(f"DB Error during comparison: {e}"); return None
        except Exception as e: print(f"Error during comparison: {e}"); return None

    def _update_verification_ui(self, match_result, error_message=None):
        if error_message:
             self.result_user_var.set("-"); self.match_score_var.set("-"); self.verification_status_var.set(error_message); self.verification_status_label.configure(style='Error.TLabel')
             return
        verification_threshold = 80.0 # Define threshold here
        if match_result:
            bmuid, bmun, hs = match_result
            dn=f"{bmun} (ID: {bmuid})"; self.result_user_var.set(dn); self.match_score_var.set(f"{hs:.2f}%")
            # --- แก้ไข: ใช้ verification_threshold ---
            if hs>=verification_threshold:
                self.verification_status_var.set("ยืนยันสำเร็จ"); self.verification_status_label.configure(style='Success.TLabel')
            else:
                self.verification_status_var.set("ยืนยันไม่สำเร็จ"); self.verification_status_label.configure(style='Failure.TLabel')
            # --- สิ้นสุดการแก้ไข ---
        else:
            self.result_user_var.set("ไม่พบการจับคู่"); self.match_score_var.set("N/A"); self.verification_status_var.set("ไม่พบการจับคู่"); self.verification_status_label.configure(style='Failure.TLabel')

    def _capture_frame_action(self, source_canvas, mode):
        if not self.sct: messagebox.showerror("Error","ไม่พร้อมจับภาพ",parent=self.root); return
        current_bbox = self.capture_bbox_register if mode=="register" else self.capture_bbox_verify
        if not current_bbox: messagebox.showerror("Error","ยังไม่ได้เลือกพื้นที่",parent=self.root); return
        was_capturing = False
        if mode=="register" and self.is_capturing_register: was_capturing=True; self.stop_capture_register()
        # Don't stop live verify just to capture
        self.root.after(50) # Short delay might still be good
        try:
             sct_img=self.sct.grab(current_bbox); img_bgr=np.array(sct_img); img_gray=cv2.cvtColor(img_bgr,cv2.COLOR_BGRA2GRAY)
             processed_img=self.preprocess_fingerprint(img_gray); self.current_scan=processed_img
             self.status_var.set(f"จับภาพเฟรม: {current_bbox['width']}x{current_bbox['height']}")
             self.display_scan(self.current_scan,canvas=source_canvas)
             if mode=="register": self.save_scan_reg_btn.config(state=tk.NORMAL if self.current_user else tk.DISABLED)
             # No separate verify button for live mode
        except Exception as e:
             messagebox.showerror("Capture Error",f"จับภาพเฟรมไม่ได้: {e}",parent=self.root); self.current_scan=None
             if mode=="register": self.save_scan_reg_btn.config(state=tk.DISABLED)
             self._clear_canvas(source_canvas,"จับภาพเฟรมล้มเหลว")

    def _reset_verification_results(self):
         self.result_user_var.set("-"); self.match_score_var.set("-"); self.verification_status_var.set("-")
         if hasattr(self,'verification_status_label'): self.verification_status_label.configure(style='Normal.TLabel')
         # Don't clear HOG canvas if it's hidden
         # self._clear_canvas(self.hog_canvas,"HOG Visualization")

    def _draw_placeholder(self, canvas, text):
         try:
             if not canvas.winfo_exists(): return
             canvas.delete("placeholder"); width=canvas.winfo_width(); height=canvas.winfo_height()
             if width>1 and height>1:
                 is_capturing=(canvas==self.scan_canvas_register and self.is_capturing_register)or(canvas==self.scan_canvas_verify and self.is_verifying_live)
                 if not canvas.find_withtag("image") and not is_capturing: canvas.create_text(width/2,height/2,text=text,fill="darkgrey",font=("Arial",10),tags="placeholder",width=width*0.9)
         except tk.TclError: pass

    def refresh_user_list(self):
        self.user_listbox.delete(0,tk.END); self.users_data={}
        try: cur=self.conn.cursor(); cur.execute("SELECT id, name FROM users ORDER BY name"); users=cur.fetchall()
        except sqlite3.Error as e: messagebox.showerror("DB Error",f"Load users failed: {e}"); return
        for uid,name in users: dt=f"{name} (ID: {uid})"; self.user_listbox.insert(tk.END,dt); self.users_data[dt]=uid

    def refresh_admin_view(self):
        for item in self.users_treeview.get_children(): self.users_treeview.delete(item)
        for item in self.fp_treeview.get_children(): self.fp_treeview.delete(item)
        self._clear_canvas(self.admin_preview_canvas,"ภาพตัวอย่าง")
        try: cur=self.conn.cursor(); cur.execute("SELECT id, name, created_at FROM users ORDER BY id"); users=cur.fetchall()
        except sqlite3.Error as e: messagebox.showerror("DB Error",f"Load admin users failed: {e}"); return
        for user in users: self.users_treeview.insert("","end",values=user)

    def on_user_select_register(self, event):
        sel=self.user_listbox.curselection()
        if not sel: self.current_user=None; self.status_var.set("ไม่มีผู้ใช้"); self._clear_canvas(self.scan_canvas_register,"1. เลือกพื้นที่จับภาพ"); self.save_scan_reg_btn.config(state=tk.DISABLED); return
        st=self.user_listbox.get(sel[0]); uid=self.users_data.get(st)
        if uid: self.current_user=uid; self.status_var.set(f"เลือก: {st}"); self.load_user_fingerprints(uid,self.scan_canvas_register); self.save_scan_reg_btn.config(state=tk.NORMAL if self.current_scan else tk.DISABLED)
        else: self.current_user=None; self.status_var.set("เลือกผิดพลาด"); self._clear_canvas(self.scan_canvas_register,"1. เลือกพื้นที่จับภาพ"); self.save_scan_reg_btn.config(state=tk.DISABLED)

    def on_admin_user_select(self, event):
        sel=self.users_treeview.selection();
        for item in self.fp_treeview.get_children(): self.fp_treeview.delete(item); self._clear_canvas(self.admin_preview_canvas,"ภาพตัวอย่าง")
        if not sel: return
        item=self.users_treeview.item(sel[0]); uid=item['values'][0]
        try: cur=self.conn.cursor(); cur.execute("SELECT id, user_id, image_path, scan_date FROM fingerprints WHERE user_id = ? ORDER BY scan_date DESC",(uid,)); fps=cur.fetchall()
        except sqlite3.Error as e: messagebox.showerror("DB Error",f"Load FPs failed (User {uid}): {e}",parent=self.root); return
        for fp in fps: self.fp_treeview.insert("","end",values=(fp[0],fp[1],fp[2],fp[3]))

    def on_admin_fp_select(self, event):
        sel=self.fp_treeview.selection();
        if not sel: self._clear_canvas(self.admin_preview_canvas,"ภาพตัวอย่าง"); return
        item=self.fp_treeview.item(sel[0]); ip=item['values'][2]
        try:
            if ip and os.path.exists(ip): img=cv2.imread(ip,cv2.IMREAD_GRAYSCALE); self.display_scan(img,canvas=self.admin_preview_canvas) if img is not None else self._clear_canvas(self.admin_preview_canvas,f"โหลดไม่ได้:\n{os.path.basename(ip)}")
            elif not ip: self._clear_canvas(self.admin_preview_canvas,"ไม่มี Path")
            else: self._clear_canvas(self.admin_preview_canvas,f"ไม่พบ:\n{os.path.basename(ip)}")
        except Exception as e: print(f"Preview error: {e}"); self._clear_canvas(self.admin_preview_canvas,"ข้อผิดพลาด")

    def _clear_canvas(self, canvas, placeholder_text):
         if canvas and canvas.winfo_exists(): canvas.delete("all"); self._draw_placeholder(canvas,placeholder_text)

    def load_user_fingerprints(self, user_id, target_canvas):
        placeholder="1. เลือกพื้นที่จับภาพ"
        self._clear_canvas(target_canvas,"กำลังโหลด...")
        try: cur=self.conn.cursor(); cur.execute("SELECT image_path FROM fingerprints WHERE user_id=? ORDER BY scan_date DESC LIMIT 1",(user_id,)); res=cur.fetchone()
        except sqlite3.Error as e: self.status_var.set(f"DB Error: {e}"); self._clear_canvas(target_canvas,"DB Error"); return
        except Exception as e: self.status_var.set(f"Error: {e}"); self._clear_canvas(target_canvas,"Error"); return
        if res and res[0]: ip=res[0]
        else: self.status_var.set(f"ไม่พบข้อมูล ID {user_id}"); self._clear_canvas(target_canvas,placeholder); return
        if os.path.exists(ip): img=cv2.imread(ip,cv2.IMREAD_GRAYSCALE)
        else: self.status_var.set(f"ไม่พบไฟล์: {os.path.basename(ip)}"); self._clear_canvas(target_canvas,placeholder); return
        if img is not None: self.display_scan(img,canvas=target_canvas); self.status_var.set(f"แสดงตัวอย่างล่าสุด ID {user_id}")
        else: self.status_var.set(f"โหลดภาพไม่ได้: {os.path.basename(ip)}"); self._clear_canvas(target_canvas,placeholder); return

    def create_user(self):
        name=self.username_entry.get().strip();
        if not name: messagebox.showerror("ผิดพลาด","ใส่ชื่อผู้ใช้",parent=self.root); return
        try: cur=self.conn.cursor(); cur.execute("INSERT INTO users (name,created_at) VALUES (?,?)",(name,datetime.now().strftime("%Y-%m-%d %H:%M:%S"))); self.conn.commit(); self.username_entry.delete(0,tk.END); self.refresh_user_list(); self.refresh_admin_view(); messagebox.showinfo("สำเร็จ",f"สร้าง '{name}' แล้ว",parent=self.root)
        except sqlite3.IntegrityError: messagebox.showwarning("ชื่อซ้ำ",f"'{name}' มีอยู่แล้ว",parent=self.root); self.conn.rollback()
        except sqlite3.Error as e: messagebox.showerror("DB Error",f"สร้างไม่ได้: {e}",parent=self.root); self.conn.rollback()

    def delete_user(self):
        sel=self.users_treeview.selection();
        if not sel: messagebox.showwarning("เลือกก่อน","เลือกผู้ใช้",parent=self.root); return
        item=self.users_treeview.item(sel[0]); uid=item['values'][0]; uname=item['values'][1]
        cfm=messagebox.askyesno("ยืนยัน",f"ลบ '{uname}' (ID:{uid}) และข้อมูลทั้งหมด?\n**ไม่สามารถย้อนกลับได้**",icon='warning',parent=self.root);
        if not cfm: return
        try:
             cur=self.conn.cursor(); cur.execute("SELECT image_path FROM fingerprints WHERE user_id=?",(uid,)); ips=[r[0] for r in cur.fetchall() if r[0]]
             cur.execute("DELETE FROM users WHERE id=?",(uid,)); dr=cur.rowcount; self.conn.commit()
             if dr>0:
                 for ip in ips:
                     try:
                         if os.path.exists(ip): os.remove(ip); print(f"Deleted: {ip}")
                     except OSError as e: print(f"Warn: Cannot delete {ip}: {e}")
                 self.refresh_user_list(); self.refresh_admin_view()
                 if self.current_user==uid: self.current_user=None; self.user_listbox.selection_clear(0,tk.END); self._clear_canvas(self.scan_canvas_register,"1. เลือกพื้นที่จับภาพ"); self.status_var.set("ผู้ใช้ถูกลบแล้ว")
                 messagebox.showinfo("สำเร็จ",f"ลบ '{uname}' แล้ว",parent=self.root)
             else: messagebox.showerror("ผิดพลาด",f"ไม่พบ ID {uid}",parent=self.root)
        except sqlite3.Error as e: messagebox.showerror("DB Error",f"ลบไม่ได้: {e}",parent=self.root); self.conn.rollback()

    def preprocess_fingerprint(self, img):
        if img is None: raise ValueError("Input image is None")
        if len(img.shape)!=2:
             if len(img.shape)==3: img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
             else: raise ValueError(f"Bad shape: {img.shape}")
        ts=(300,300);
        try: img_r=cv2.resize(img,ts,interpolation=cv2.INTER_AREA)
        except cv2.error as e: print(f"Resize error: {e}"); img_r=img
        cl=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)); img_c=cl.apply(img_r); img_b=cv2.medianBlur(img_c,5)
        img_t=cv2.adaptiveThreshold(img_b,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
        return img_t

    def extract_hog_features(self, img):
        if img is None: return None,None
        if len(img.shape)!=2:
             if len(img.shape)==3: img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
             else: print(f"Bad shape HOG: {img.shape}"); return None,None
        ths=(128,128);
        try: img_r=cv2.resize(img,ths,interpolation=cv2.INTER_AREA)
        except cv2.error as e: print(f"Resize HOG error: {e}"); return None,None
        try: fd,hi=hog(img_r,orientations=9,pixels_per_cell=(8,8),cells_per_block=(2,2),visualize=True,block_norm='L2-Hys',feature_vector=True); hir=None
        except Exception as e: print(f"HOG error: {e}"); return None,None
        if hi is not None: hir=exposure.rescale_intensity(hi,out_range=(0,255)).astype(np.uint8)
        return fd,hir

    def save_scan(self):
        if self.current_scan is None: messagebox.showwarning("ไม่มีภาพ","'จับภาพเฟรมนี้' ก่อน",parent=self.root); return
        if self.current_user is None: messagebox.showwarning("ไม่ได้เลือก","เลือกผู้ใช้ก่อน",parent=self.root); return
        save_dir = "fingerprints"; os.makedirs(save_dir, exist_ok=True); timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f"); user_name = "unknown"; filename = None
        try: cur=self.conn.cursor(); cur.execute("SELECT name FROM users WHERE id=?",(self.current_user,)); res=cur.fetchone(); user_name = res[0].replace(" ","_") if res else user_name
        except Exception as name_e: print(f"Warn: get user name failed: {name_e}")
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
            hog_features_binary = hog_features.astype(np.float64).tobytes(); cursor = self.conn.cursor()
            cursor.execute("INSERT INTO fingerprints (user_id,image_path,scan_date,hog_features) VALUES (?,?,?,?)",(self.current_user,filename,datetime.now().strftime("%Y-%m-%d %H:%M:%S"),hog_features_binary)); self.conn.commit()
            self.status_var.set(f"บันทึก: {filename_base}"); self.refresh_admin_view(); self.load_user_fingerprints(self.current_user,self.scan_canvas_register); messagebox.showinfo("สำเร็จ","บันทึกแล้ว",parent=self.root)
            self.save_scan_reg_btn.config(state=tk.DISABLED)
        except (sqlite3.Error,IOError,Exception) as e:
            messagebox.showerror("ผิดพลาด",f"บันทึกไม่ได้: {str(e)}",parent=self.root); self.conn.rollback();
            if filename and os.path.exists(filename):
                try: os.remove(filename); print(f"Removed orphaned file due to DB error: {filename}")
                except OSError as rem_e: print(f"Warning: Could not remove orphaned file {filename}: {rem_e}")
                except Exception as rem_gen_e: print(f"Warning: Unexpected error removing orphaned file {filename}: {rem_gen_e}")

    # verify_fingerprint is no longer needed for a button
    # def verify_fingerprint(self): ...

    def calculate_similarity(self, hog1, hog2):
        f1=np.asarray(hog1).flatten(); f2=np.asarray(hog2).flatten();
        if f1.shape!=f2.shape: return 0.0;
        n1=np.linalg.norm(f1); n2=np.linalg.norm(f2);
        if n1==0 or n2==0: return 100.0 if n1==0 and n2==0 else 0.0;
        eps=1e-9; cs=np.dot(f1,f2)/((n1*n2)+eps); cs=np.clip(cs,-1.0,1.0);
        return (cs+1.0)/2.0*100.0

    def display_hog_image(self, hog_image_np):
        if hog_image_np is None: self._clear_canvas(self.hog_canvas,"ไม่มี HOG"); return;
        # Don't display HOG if it's hidden
        # self.display_scan(hog_image_np,canvas=self.hog_canvas,is_hog=True)
        pass

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
            img_r=img_pil.resize((nw,nh),Image.Resampling.LANCZOS); img_tk=ImageTk.PhotoImage(img_r); ref=f"img_ref_{cname}"; setattr(canvas,ref,img_tk)
            canvas.delete("all"); xp=max(0,(cw-nw)//2); yp=max(0,(ch-nh)//2); canvas.create_image(xp,yp,anchor="nw",image=img_tk,tags="image")
            # if is_hog: canvas.create_text(5,5,anchor="nw",text="HOG",fill="blue",font=("Arial",8),tags="hog_text") # Don't draw HOG text if hidden
        except Exception as e:
            print(f"ERROR display {cname}: {e}")
            self._clear_canvas(canvas,"แสดงไม่ได้")

    def export_database(self):
        current_db_path="fingerprint_db.sqlite";
        try: res=self.conn.execute("PRAGMA database_list;").fetchone(); current_db_path=res[2] if res and len(res)>2 and res[2] else current_db_path
        except Exception as e: print(f"Warn: PRAGMA error: {e}. Using default.")
        try:
            dfn=f"fp_db_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sqlite"; fp=filedialog.asksaveasfilename(defaultextension=".sqlite",filetypes=[("SQLite","*.sqlite"),("DB","*.db"),("All","*.*")],title="ส่งออก...",initialfile=dfn,parent=self.root)
            if not fp: self.status_var.set("ยกเลิกส่งออก"); return
            import shutil; shutil.copy2(current_db_path,fp); messagebox.showinfo("สำเร็จ",f"ส่งออกไป\n{fp}\nแล้ว",parent=self.root); self.status_var.set(f"ส่งออก: {os.path.basename(fp)}")
        except Exception as e: messagebox.showerror("ผิดพลาด",f"ส่งออกไม่สำเร็จ: {str(e)}",parent=self.root); self.status_var.set("ส่งออกล้มเหลว")

    def import_database(self):
        fp=filedialog.askopenfilename(filetypes=[("SQLite","*.sqlite"),("DB","*.db"),("All","*.*")],title="เลือกไฟล์นำเข้า",parent=self.root)
        if not fp: self.status_var.set("ยกเลิกนำเข้า"); return
        cfm=messagebox.askyesno("ยืนยัน","**คำเตือน:** เขียนทับ DB ปัจจุบัน!\nแนะนำให้ส่งออกก่อน\n\nดำเนินการต่อ?",icon='warning',parent=self.root)
        if not cfm: self.status_var.set("ยกเลิกนำเข้า"); return
        cdb="fingerprint_db.sqlite"; bp=None; cwo=False
        try:
            res=self.conn.execute("PRAGMA database_list;").fetchone(); cdb=res[2] if res and len(res)>2 and res[2] else cdb; self.conn.close(); cwo=True; print(f"Current DB: {cdb}")
            import shutil; bp=cdb+f".backup_{datetime.now().strftime('%Y%m%d%H%M%S')}"; shutil.copy2(cdb,bp); print(f"Backed up: {bp}")
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
            except Exception as rse: messagebox.critical("ผิดพลาดร้ายแรง",f"กู้คืน DB ไม่ได้: {rse}",parent=self.root); self.conn=None; self.status_var.set("ข้อผิดพลาดร้ายแรง! กู้คืน DB ไม่ได้")
        finally:
             if self.conn is None and cwo:
                 try: print("Final reconnect..."); self.conn=sqlite3.connect(cdb); self.conn.execute("PRAGMA foreign_keys = ON;")
                 except Exception as final_e: messagebox.critical("ผิดพลาดร้ายแรง",f"เปิด DB ไม่ได้: {final_e}",parent=self.root); self.conn=None

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
        from ttkthemes import ThemedTk
        available_themes = ThemedTk().get_themes(); preferred_themes = ['arc', 'plastik', 'adapta', 'aqua']; chosen_theme = 'default'
        for theme in preferred_themes:
            if theme in available_themes: chosen_theme = theme; break
        root = ThemedTk(theme=chosen_theme); print(f"Theme: {chosen_theme}")
    except ImportError: print("ttkthemes not found, using default."); root = tk.Tk()

    app = FingerprintSystem(root)

    def on_closing():
        if app.is_capturing_register: app.stop_capture_register()
        if app.is_verifying_live: app.stop_live_verify() # Use stop_live_verify
        if messagebox.askokcancel("ปิดโปรแกรม", "ต้องการปิดโปรแกรมหรือไม่?", parent=root):
            try:
                app.close_mss();
                if app.conn: app.conn.close(); print("DB closed.")
            except Exception as e: print(f"Cleanup error: {e}")
            finally: root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

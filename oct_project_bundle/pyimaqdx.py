
import ctypes
import numpy as np

class PyIMAQdx:
    def __init__(self, camera_name="cam0", width=640, height=480):
        self.imaqdx = ctypes.windll.LoadLibrary("C:\\Windows\\System32\\imaqdx.dll")
        self.session = ctypes.c_uint32()
        self.camera_name = camera_name.encode('utf-8')
        self.width = width
        self.height = height
        self.frame = np.zeros((self.height, self.width), dtype=np.uint8)
        
    def open(self):
        status = self.imaqdx.IMAQdxOpenCamera(self.camera_name, 0, ctypes.byref(self.session))
        if status != 0:
            raise Exception(f"Failed to open camera {self.camera_name.decode()} with status {status}")

    def configure(self, continuous=True):
        mode = 1 if continuous else 0
        self.imaqdx.IMAQdxConfigureAcquisition(self.session, mode)

    def start(self):
        self.imaqdx.IMAQdxStartAcquisition(self.session)

    def stop(self):
        self.imaqdx.IMAQdxStopAcquisition(self.session)

    def close(self):
        self.imaqdx.IMAQdxCloseCamera(self.session)

    def get_frame(self):
        status = self.imaqdx.IMAQdxGetImage(self.session, self.frame.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)), 0x00000008)
        if status != 0:
            raise Exception(f"Failed to get image, status {status}")
        return self.frame.copy()

    def set_attribute(self, attr_name, value):
        attr = attr_name.encode('utf-8')
        if isinstance(value, int):
            self.imaqdx.IMAQdxSetAttributeI32(self.session, attr, value)
        elif isinstance(value, float):
            self.imaqdx.IMAQdxSetAttributeF32(self.session, attr, value)
        else:
            raise ValueError("Unsupported attribute type")

    def software_trigger_once(self):
        self.imaqdx.IMAQdxSoftwareTrigger(self.session)

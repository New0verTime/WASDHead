import cv2, numpy as np
import threading
import time
from threading import Thread, Event

class CameraThread:
    
    def __init__(self, frame_callback=None):
        self.lock = threading.Lock()
        self.frame_callback = frame_callback
        self.cap = None
        self.is_running = False
        self.stop_flag = Event()
        self.camera_thread = None
        self.current_frame = None
        self.frame_width = 640
        self.frame_height = 480
    
    def start(self):
        if not self.is_running:
            self.is_running = True
            self.stop_flag.clear()
            self.camera_thread = Thread(target=self.camera_loop, daemon=True)
            self.camera_thread.start()
            print("Camera thread started.")
    
    def camera_loop(self):
        try:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            
            if not self.cap.isOpened():
                print("Can't open camera!")
                self.is_running = False
                return
        except Exception as e:
            print(f"Error camera init: {e}")
            self.is_running = False
            return
        
        failure_count = 0 
        
        while not self.stop_flag.is_set():
            try:
                start_time = time.time()
                ret, frame = self.cap.read()
                # print(f"frame time: {time.time() - start_time:.4f} giây")
                if not ret:
                    failure_count += 1
                    print(f"Lỗi đọc frame từ camera (lần {failure_count})")
                    
                    if failure_count > 5:
                        print("Thử khởi động lại camera...")
                        self.cap.release()
                        time.sleep(1)
                        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
                        failure_count = 0
                    
                    time.sleep(0.1)
                    continue
                failure_count = 0
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                with self.lock:
                    self.current_frame = frame_rgb.copy()
                
                if self.frame_callback:
                    self.frame_callback(frame_rgb)
                
            except Exception as e:
                print(f"Camera_loop bug: {e}")
                time.sleep(0.1)
            
            try:
                cv2.waitKey(1)
            except:
                pass
    
    def set_frame_callback(self, callback):
        self.frame_callback = callback

    def get_frame(self):
        with self.lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
            return None
    
    def __del__(self):
        if hasattr(self, "is_running") and self.is_running:
            self.stop_flag.set()  
        if hasattr(self, "cap") and self.cap:
            self.cap.release()
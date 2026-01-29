import tkinter as tk
import customtkinter as ctk
import PIL.Image, PIL.ImageTk
import cv2
from src.pipeline import Pipeline
from src.gui.profile_manager_ui import ProfileManagerUI
from src.gui.mouse_settings_ui import MouseSettingsUI
from src.gui.blendshape_ui import BlendshapeSettingsUI

from src.gui.overlay import Overlay

class MainWindow(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.root = self
            
        self.title("WASDHead Mouse Controller")
        self.resizable(False, False)
        
        self.pipeline = Pipeline()
        if not self.pipeline.is_started:
            self.pipeline.start()

        self.face_processor = self.pipeline.get_face_processor()
        self.mouse_controller = self.pipeline.get_mouse_controller()
        self.blendshape_processor = self.pipeline.get_blendshape_processor()
        self.profile_manager = self.pipeline.get_profile_manager()
        self.current_settings = self.profile_manager.get_profile_settings()
        self.ovl = Overlay(self.mouse_controller)

        self.blendshape_options = {
            "browInnerUp": {"name": "browInnerUp", "index": 3},
            "jawOpen": {"name": "jawOpen", "index": 25},
            "mouthSmileLeft": {"name": "mouthSmileLeft", "index": 44},
            "mouthRollUpper": {"name": "mouthRollUpper", "index": 40},
            "mouthFunnel": {"name": "mouthFunnel", "index": 36},
            "mouthLeft": {"name": "mouthLeft", "index": 48},
            "mouthRight": {"name": "mouthRight", "index": 54},
        }
        self._create_main_layout()
        
        self.update_interval = 10
        self.update_frame()
    
    def _create_main_layout(self):
        # Main container
        main_container = ctk.CTkFrame(self)
        main_container.pack(fill="both", expand=True, padx=5, pady=12)
        
        self._create_left_frame(main_container)
        
        self._create_right_frame(main_container)
    
    def _create_left_frame(self, parent):
        left_frame = ctk.CTkFrame(parent)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Video frame
        self.video_frame = ctk.CTkFrame(left_frame, width=640, height=480)
        self.video_frame.pack(padx=0, pady=5)
        
        self.canvas = ctk.CTkCanvas(self.video_frame)
        self.canvas.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Control frame with buttons
        control_frame = ctk.CTkFrame(left_frame, fg_color="transparent")
        control_frame.pack(fill="x", padx=0, pady=5)
        
        control_frame.columnconfigure(0, weight=1)
        control_frame.columnconfigure(1, weight=1)
        control_frame.columnconfigure(2, weight=1)

        # Mouse control button
        self.mouse_active = False 
        self.mouse_button = ctk.CTkButton(
            control_frame, 
            text="Mouse Control: OFF", 
            command=self.toggle_mouse_control,
            width=150,
            height=30
        )
        self.mouse_button.grid(row=0, column=0, padx=5, pady=5)
        # Blendshape control button
        self.blendshape_processor.enable()
        current_blendshape = self.current_settings.get("state_machine_blendshape", {
            "name": "browInnerUp",
            "index": 3,
            "display": "Brow Raise",
            "threshold": 0.5
        })

        self.blendshape_var = ctk.StringVar(value=current_blendshape.get("display", "Brow Raise"))
        self.blendshape_dropdown = ctk.CTkOptionMenu(
            control_frame,
            variable=self.blendshape_var,
            values=list(self.blendshape_options.keys()),
            command=self.on_blendshape_change,
            width=150,
            height=30,
        )
        self.blendshape_dropdown.grid(row=0, column=1, padx=5, pady=5)
        
        blendshape_status_frame = ctk.CTkFrame(control_frame, fg_color="transparent")
        blendshape_status_frame.grid(row=0, column=2, padx=5, pady=1, sticky="ew")

        blendshape_status_frame.columnconfigure(1, weight=1)
        
        self.blendshape_progress_bar = ctk.CTkProgressBar(blendshape_status_frame, height=12, width=150)
        self.blendshape_progress_bar.grid(row=0, column=0, padx=2, pady=1, sticky="ew")
        self.blendshape_progress_bar.set(0.0)
        
        self.blendshape_value_label = ctk.CTkLabel(blendshape_status_frame, text="0.00", width=40)
        self.blendshape_value_label.grid(row=0, column=1, padx=(2, 5), pady=1, sticky="w")
        
        self.blendshape_threshold_var = ctk.DoubleVar(value=current_blendshape.get("threshold", 0.5))
        self.blendshape_threshold_slider = ctk.CTkSlider(
            blendshape_status_frame,
            from_=0.01,
            to=1.0,
            variable=self.blendshape_threshold_var,
            number_of_steps=100,
            command=self.on_threshold_change,
            width=150,
            height=12
        )
        self.blendshape_threshold_slider.grid(row=1, column=0, padx=2, pady=1, sticky="ew")
        
        self.blendshape_threshold_label = ctk.CTkLabel(
            blendshape_status_frame, 
            text=f"{current_blendshape.get('threshold', 0.5):.2f}", 
            width=40
        )
        self.blendshape_threshold_label.grid(row=1, column=1, padx=(2, 5), pady=1, sticky="w")
        
        # Áp dụng blendshape hiện tại
        self.apply_blendshape_to_controller(current_blendshape)
        
        # Bắt đầu update progress bar
        self.update_blendshape_display()
    
    def update_blendshape_display(self):
        try:
            if hasattr(self, 'blendshape_progress_bar') and hasattr(self.face_processor, 'result'):
                result = self.face_processor.result
                if result and result.face_blendshapes and len(result.face_blendshapes) > 0:
                    blendshapes = result.face_blendshapes[0]
                    current_index = self.mouse_controller.state_machine_blendshape_index
                    
                    if current_index < len(blendshapes):
                        current_value = blendshapes[current_index].score
                        self.blendshape_progress_bar.set(current_value)
                        self.blendshape_value_label.configure(text=f"{current_value:.2f}")
        except Exception as e:
            pass
        
        # Gọi lại sau 33ms (khoảng 30fps)
        self.after(33, self.update_blendshape_display)

    def on_threshold_change(self, value):
        """Callback khi user thay đổi threshold slider"""
        value = float(value)
        self.blendshape_threshold_label.configure(text=f"{value:.2f}")
        self.mouse_controller.trigger_threshold = value
        # Cập nhật vào current_settings
        if "state_machine_blendshape" in self.current_settings:
            self.current_settings["state_machine_blendshape"]["threshold"] = value
            self.profile_manager.update_profile_settings(self.current_settings)

    def on_blendshape_change(self, display_name):
        """Callback khi user chọn blendshape khác từ dropdown"""
        selected = self.blendshape_options[display_name]
        
        # Giữ nguyên threshold cũ hoặc dùng mặc định
        current_threshold = self.blendshape_threshold_var.get()
        
        blendshape_config = {
            "name": selected["name"],
            "index": selected["index"],
            "display": display_name,
            "threshold": current_threshold
        }
        
        # Lưu vào profile
        self.current_settings["state_machine_blendshape"] = blendshape_config
        self.profile_manager.update_profile_settings(self.current_settings)
        
        # Áp dụng cho controller
        self.apply_blendshape_to_controller(blendshape_config)
        
        # Reset progress bar về 0
        self.blendshape_progress_bar.set(0.0)
        self.blendshape_value_label.configure(text="0.00")
        
        print(f"Blendshape changed to: {display_name} (index: {selected['index']})")

    def apply_blendshape_to_controller(self, blendshape_config):
        """Áp dụng blendshape index cho mouse controller"""
        if hasattr(self.mouse_controller, 'state_machine_blendshape_index') and hasattr(self.mouse_controller, 'trigger_threshold'):
            self.mouse_controller.state_machine_blendshape_index = blendshape_config["index"]
            self.mouse_controller.trigger_threshold = blendshape_config.get("threshold", 0.5)
        else:
            # Nếu chưa có thuộc tính, thêm vào
            self.mouse_controller.state_machine_blendshape_index = blendshape_config["index"]
            self.mouse_controller.trigger_threshold = blendshape_config.get("threshold", 0.5)


    def _create_right_frame(self, parent):
        right_frame = ctk.CTkFrame(parent, width=410, fg_color="transparent")
        right_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=5)
        right_frame.grid_columnconfigure(0, weight=1) 
        right_frame.grid_propagate(False)
        
        profile_container = ctk.CTkFrame(right_frame, fg_color="transparent")
        profile_container.grid(row=0, column=0, sticky="new", padx=50, pady=0)
        profile_container.grid_columnconfigure(0, weight=1)

        # Create profile section
        self.profile_ui = ProfileManagerUI(
            profile_container, 
            self.profile_manager,
            self.mouse_controller,
            self.on_profile_change
        )
        self.profile_ui.grid(row=0, column=0, sticky="new", padx=10, pady=5)
        # Create settings tabview
        settings_frame = ctk.CTkTabview(right_frame,fg_color="transparent")
        settings_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        
        # Tab Mouse
        tab1 = settings_frame.add("Mouse")
        self.mouse_settings = MouseSettingsUI(
            tab1, 
            self.current_settings, 
            self.mouse_controller
        )
        self.mouse_settings.set_face_processor(self.face_processor)
        self.mouse_settings.set_profile_manager(self.profile_manager)

        # Tab Blendshape
        tab2 = settings_frame.add("Gesture Shortcuts")
        self.blendshape_settings = BlendshapeSettingsUI(
            tab2, 
            self.blendshape_processor, 
            self.profile_manager,
            self.current_settings
        )

    
    def toggle_mouse_control(self):
        self.mouse_active = not self.mouse_active
        
        if self.mouse_active:
            self.mouse_controller.start_tracking()
            self.mouse_button.configure(text="Mouse Control: ON")
        else:
            self.mouse_controller.stop_tracking()
            self.mouse_button.configure(text="Mouse Control: OFF")


    
    def on_profile_change(self, profile_name):
        try:
            self.current_settings = self.profile_manager.load_profile(profile_name)
            
            # Update mouse settings UI
            self.mouse_settings.update_from_profile(self.current_settings)
            blendshape_config = self.current_settings.get("state_machine_blendshape", {
                "name": "browInnerUp",
                "index": 3,
                "display": "Brow Raise",
                "threshold": 0.5
            })
            self.blendshape_var.set(blendshape_config.get("display", "Brow Raise"))
            self.blendshape_threshold_var.set(blendshape_config.get("threshold", 0.5))
            self.blendshape_threshold_label.configure(text=f"{blendshape_config.get('threshold', 0.5):.2f}")
            self.apply_blendshape_to_controller(blendshape_config)
            
                
        except Exception as e:
            print(f"Error loading profile: {e}")
    
    def update_frame(self):
        frame = self.face_processor.get_processed_frame()
        frame = cv2.flip(frame, 1) if frame is not None else None
        if frame is not None:
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            
            self.canvas.config(width=frame.shape[1], height=frame.shape[0])
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.after(self.update_interval, self.update_frame)
        if hasattr(self, "ovl") and self.ovl is not None:
            self.ovl.update_once()
    
    def __del__(self):
        if hasattr(self, 'camera_thread') and self.camera_thread:
            self.camera_thread.stop_flag.set()
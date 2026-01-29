import tkinter as tk
import customtkinter as ctk
from src.gui.dialogs.profile_dialog import ProfileDialog

class ProfileManagerUI(ctk.CTkFrame):
    
    def __init__(self, parent, profile_manager, mouse_controller, on_profile_change_callback):
        super().__init__(parent, fg_color=("#CFCFCF", "#333333"))
        
        self.profile_manager = profile_manager
        self.mouse_controller = mouse_controller
        self.on_profile_change_callback = on_profile_change_callback
        
        self._create_profile_ui()
    
    def _create_profile_ui(self):
        # Profile label
        profile_label = ctk.CTkLabel(self, text="Profile:")
        profile_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        # Profile dropdown
        self.profiles = self.profile_manager.list_profiles()
        self.profile_var = tk.StringVar(value=self.profile_manager.get_current_profile_name())
        
        self.profile_dropdown = ctk.CTkOptionMenu(
            self,
            values=self.profiles,
            variable=self.profile_var,
            command=self.on_profile_change,
            width=150
        )
        self.profile_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky="e")
        
        # Profile buttons
        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.grid(row=1, column=0, columnspan=2, pady=2, sticky="ew")

        self.grid_columnconfigure(0, weight=0)  # Label không cần mở rộng
        self.grid_columnconfigure(1, weight=1) 

        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        button_frame.columnconfigure(2, weight=1)
        
        # Profile buttons với padding giảm xuống
        self.save_profile_btn = ctk.CTkButton(
            button_frame,
            text="Save",
            command=self.save_current_profile,
            width=70,
            height=28
        )
        self.save_profile_btn.grid(row=0, column=0, padx=2, pady=2, sticky="ew")
        
        self.add_profile_btn = ctk.CTkButton(
            button_frame,
            text="Add",
            command=self.create_new_profile,
            width=70,
            height=28
        )
        self.add_profile_btn.grid(row=0, column=1, padx=2, pady=2, sticky="ew")
        
        self.delete_profile_btn = ctk.CTkButton(
            button_frame,
            text="Delete",
            command=self.delete_current_profile,
            width=70,
            height=28
        )
        self.delete_profile_btn.grid(row=0, column=2, padx=2, pady=2, sticky="ew")
    
    def on_profile_change(self, profile_name):
        if self.on_profile_change_callback:
            self.on_profile_change_callback(profile_name)
    
    def save_current_profile(self):
        current_name = self.profile_var.get()
        
        updated_settings = {
            "mouse_controller": {
                "velocity_scale": getattr(self.mouse_controller, 'velocity_scale', 15.0),
                "mincutoff": getattr(self.mouse_controller, 'mincutoff', 1.0),
                "beta": getattr(self.mouse_controller, 'beta', 0.01)
            },
            "face_processing": {
                "mode": getattr(self, 'face_processing_mode', "LIVE_STREAM")
            }
        }
        
        try:
            self.profile_manager.update_profile_settings(updated_settings, current_name)
            print(f"Profile '{current_name}' saved successfully")
        except Exception as e:
            print(f"Error saving profile: {e}")
    
    def create_new_profile(self):
        dialog = ProfileDialog(self)
        new_name = dialog.get_input()
        
        try:
            success = self.profile_manager.create_profile(new_name)
            
            if success:
                self.refresh_profile_list()
                self.profile_dropdown.set(new_name)
                
                if self.on_profile_change_callback:
                    self.on_profile_change_callback(new_name)
                
                print(f"Profile '{new_name}' created successfully with full default settings")
            else:
                print(f"Profile '{new_name}' already exists")
                
        except Exception as e:
            print(f"Error creating profile '{new_name}': {e}")
    
    def delete_current_profile(self):
        current_name = self.profile_var.get()
        
        if current_name == "default":
            print("Cannot delete default profile")
            return
        
        try:
            success = self.profile_manager.delete_profile(current_name)
            
            if success:
                self.refresh_profile_list()
                self.profile_dropdown.set("default")
                
                if self.on_profile_change_callback:
                    self.on_profile_change_callback("default")
                
                print(f"Profile '{current_name}' deleted successfully")
            else:
                print(f"Failed to delete profile '{current_name}'")
                
        except Exception as e:
            print(f"Error deleting profile '{current_name}': {e}")

    def refresh_profile_list(self):
        self.profiles = self.profile_manager.list_profiles()
        self.profile_dropdown.configure(values=self.profiles)
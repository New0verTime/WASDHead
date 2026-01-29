import os
import json
from typing import Dict, Any, List, Optional

class ProfileManager:
    def __init__(self, profiles_dir="profiles"):
        self.profiles_dir = profiles_dir
        self.current_profile = "default"
        
        if not os.path.exists(self.get_profile_path("default")):
            self.create_default_profile()
    
    def get_profile_path(self, profile_name: str) -> str:
        return os.path.join(self.profiles_dir, f"{profile_name}.json")
    
    def get_default_profile_template(self) -> Dict[str, Any]:
        return{
            "mouse_controller": {
                "velocity_scale": 15.0,
                "mincutoff": 1.5,
                "beta": 0.1,
                "accel_on": True,
                "toggle": True,
            },
            "voice_processor": {
                "selected_microphone": "Default Microphone",
                "commands": [
                    {
                        "command": "c",
                        "action": "key_c"
                    }
                ]
            },
            "blendshape_bindings": {
                "bindings": [
                    {
                        "blendshape": "mouthSmileLeft",
                        "action": "mouse_click",
                        "threshold": 0.5,
                        "mode": "hold"
                    },
                    {
                        "blendshape": "jawOpen",
                        "action": "mouse_right_click",
                        "threshold": 0.5,
                        "mode": "hold"
                    }
                ],
                "threshold": 0.5
            },
            "face_processing": {
                "mode": "LIVE_STREAM"
            }
        }

    def create_default_profile(self):
        default_settings = self.get_default_profile_template()
        self.save_profile("default", default_settings)
        
    def list_profiles(self) -> List[str]:
        if not os.path.exists(self.profiles_dir):
            return []
        
        profiles = []
        
        for file in os.listdir(self.profiles_dir):
            if file.endswith(".json"):
                profile_name = os.path.splitext(file)[0]
                profiles.append(profile_name)
        
        return profiles
    
    def create_profile(self, profile_name: str) -> bool:
        try:
            if self.profile_exists(profile_name):
                return False
            
            default_settings = self.get_default_profile_template()
            
            self.save_profile(profile_name, default_settings)
            return True
            
        except Exception as e:
            print(f"Error creating profile '{profile_name}': {e}")
            return False
        
    def profile_exists(self, profile_name: str) -> bool:
        return os.path.exists(self.get_profile_path(profile_name))
    
    def load_profile(self, profile_name: str) -> Dict[str, Any]:
        if not self.profile_exists(profile_name):
            print(f"Profile '{profile_name}' not found, creating with defaults")
            self.create_profile(profile_name)
        
        try:
            with open(self.get_profile_path(profile_name), 'r') as f:
                profile_data = json.load(f)
            
            self.current_profile = profile_name
            return profile_data
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in profile '{profile_name}'")
    
    def save_profile(self, profile_name: str, profile_data: Dict[str, Any]) -> None:
        with open(self.get_profile_path(profile_name), 'w') as f:
            json.dump(profile_data, f, indent=4)
        
        self.current_profile = profile_name
    
    def delete_profile(self, profile_name: str) -> bool:
        if profile_name == "default":
            raise ValueError("Cannot delete the default profile")
        
        if self.profile_exists(profile_name):
            os.remove(self.get_profile_path(profile_name))
            
            if self.current_profile == profile_name:
                self.current_profile = "default"
            
            return True
        
        return False
    
    def get_current_profile_name(self) -> str:
        return self.current_profile
    
    def get_profile_settings(self, profile_name: Optional[str] = None) -> Dict[str, Any]:
        profile = profile_name or self.current_profile
        return self.load_profile(profile)
    
    def update_profile_settings(self, new_settings: Dict[str, Any], profile_name: Optional[str] = None) -> None:
        profile = profile_name or self.current_profile
        
        if not self.profile_exists(profile):
            self.save_profile(profile, new_settings)
            return
        
        current_settings = self.load_profile(profile)
        
        def update_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    d[k] = update_dict(d[k], v)
                else:
                    d[k] = v
            return d
        
        updated_settings = update_dict(current_settings, new_settings)
        self.save_profile(profile, updated_settings)
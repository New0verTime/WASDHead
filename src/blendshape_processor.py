import time
import pyautogui
from collections import deque

class BlendshapeProcessor:    
    def __init__(self, profile_manager=None):
        self.profile_manager = profile_manager
        
        self.default_threshold = 0.5
        self.bindings = []
        
        self.active_key = None 
        self.active_action = None

        self.pressed_keys = set()
        self.last_press_time = {}
        self.press_cooldown = 1

        self.is_enabled = False

        self.blendshapes = [
            "browInnerUp",
            "mouthLeft", "mouthRight", "mouthSmileLeft", 
            "mouthFunnel", "mouthRollUpper", "jawOpen"
            # "eyeBlinkLeft", "eyeBlinkRight"
        ]
        self.mouth_priority = ["mouthShrugLower", "mouthLeft", "mouthRight", 
                               "mouthRollUpper", "mouthFunnel", "mouthSmileLeft", "jawOpen"]
        self.eye_priority = ["eyeLookInLeft", "eyeLookOutLeft", "eyeLookUpLeft", "eyeLookDownLeft"]
        self.brow_priority = ["browDownLeft", "browInnerUp"]

        self.actions = {
            "mouse": [
                "mouse_click", 
                "mouse_right_click", 
                "mouse_middle_click",
                "mouse_double_click",
                "scroll_up", 
                "scroll_down"
            ],
            "basic_keys": [
                "key_space", 
                "key_enter", 
                "key_tab", 
                "key_escape", 
                "key_backspace", 
                "key_delete"
            ],
            "arrow_keys": [
                "key_up", 
                "key_down", 
                "key_left", 
                "key_right"
            ],
            "number_keys": [
                "key_0", "key_1", "key_2", "key_3", "key_4", 
                "key_5", "key_6", "key_7", "key_8", "key_9"
            ],
            "letter_keys": [
                "key_a", "key_b", "key_c", "key_d", "key_e", "key_f", "key_g",
                "key_h", "key_i", "key_j", "key_k", "key_l", "key_m", "key_n",
                "key_o", "key_p", "key_q", "key_r", "key_s", "key_t", "key_u",
                "key_v", "key_w", "key_x", "key_y", "key_z"
            ],
            "function_keys": [
                "key_f1", "key_f2", "key_f3", "key_f4", "key_f5", "key_f6",
                "key_f7", "key_f8", "key_f9", "key_f10", "key_f11", "key_f12"
            ]
        }

        self.jaw_open_counter = 0
        self.jaw_open_threshold = 0.1
        self.jaw_open_frame_count = 50

        if profile_manager:
            self.load_from_profile()
    
    def load_from_profile(self):
        profile_settings = self.profile_manager.get_profile_settings()
        bs_settings = profile_settings.get("blendshape_bindings", {})

        if not bs_settings:
            profile_settings["blendshape_bindings"] = {
                "bindings": [
                    {"blendshape": "mouthSmileLeft", "action": "mouse_click", "threshold": 0.5, "mode": "hold"}
                ],
                "threshold": 0.5
            }
            self.update_profile(self.profile_manager)
        
        self.bindings = bs_settings.get("bindings", [])
        self.default_threshold = bs_settings.get("threshold", 0.5)

    def enable(self):
        self.is_enabled = True
        
    def disable(self):
        self.is_enabled = False
        if self.active_key:
            self._release_key()

    def update_profile(self, profile_manager):
        try:
            profile_name = self.profile_manager.get_current_profile_name()
            profile_settings = self.profile_manager.load_profile(profile_name)
            
            if "blendshape_bindings" not in profile_settings:
                profile_settings["blendshape_bindings"] = {}
                
            profile_settings['blendshape_bindings']['bindings'] = self.bindings
            profile_settings['blendshape_bindings']['threshold'] = self.default_threshold
            
            self.profile_manager.save_profile(profile_name, profile_settings)
            return True
        except Exception as e:
            print(f"Profile update error: {e}")
            return False
        
    def save_to_profile(self):
        if not self.profile_manager:
            return
            
        try:
            profile_name = self.profile_manager.get_current_profile_name()
            profile_settings = self.profile_manager.load_profile(profile_name)
            
            if "blendshape_bindings" not in profile_settings:
                profile_settings["blendshape_bindings"] = {}
                
            profile_settings["blendshape_bindings"]["bindings"] = self.bindings
            profile_settings["blendshape_bindings"]["threshold"] = self.default_threshold
            
            self.profile_manager.save_profile(profile_name, profile_settings)
            print("Saved blendshape settings to profile")
        except Exception as e:
            print(f"Error saving blendshape settings: {e}")
    
    def update_blendshape(self, blendshapes):
        self.current_blendshapes = blendshapes

        if not self.is_enabled:
            if self.active_key:
                self._release_key()
            return None, 0

        if not blendshapes:
            if self.active_key:
                self._release_key()
            return None, 0
        
        action, value = self.process_blendshapes(blendshapes)

        return action, value

    def process_blendshapes(self, blendshapes):
        if not blendshapes:
            if hasattr(self, 'active_categories'):
                for category in self.active_categories:
                    self._release_category(category)
            elif self.active_key:
                self._release_key()
            return None, 0
        
        current_time = time.time()
        blendshape_values = {}

        jaw_open_value = 0.0
        
        for blendshape in blendshapes:
            name = blendshape.category_name
            value = blendshape.score
            blendshape_values[name] = value

            if name == "jawOpen":
                jaw_open_value = value

        if jaw_open_value > self.jaw_open_threshold:
            self.jaw_open_counter = 0
        else:
            self.jaw_open_counter += 1
            
        self._process_hold_mode(blendshape_values)

        action, value = self._process_press_mode(blendshape_values, current_time)

        if self.active_action == "scroll_up":
            pyautogui.scroll(5)  
        elif self.active_action == "scroll_down":
            pyautogui.scroll(-5)

        return action, value
    
    def _process_hold_mode(self, blendshape_values):
        if not hasattr(self, 'active_categories'):
            self.active_categories = {'mouth': None, 'eye': None, 'brow': None}
        
        self._process_category('mouth', self.mouth_priority, blendshape_values)
        self._process_category('eye', self.eye_priority, blendshape_values)
        self._process_category('brow', self.brow_priority, blendshape_values)

    def _process_category(self, category, priority_list, blendshape_values):
        current_active = self.active_categories[category]
        
        if current_active:
            binding = self._find_binding(current_active)
            if (binding and binding.get("mode", "hold") == "hold" and
                current_active in blendshape_values and
                blendshape_values[current_active] >= binding.get("threshold", self.default_threshold)):
                return  
            else:
                self._release_category(category)
        
        best_candidate = None
        best_priority = float('inf')
        
        for name in priority_list:
            if name in blendshape_values:
                binding = self._find_binding(name)
                if (binding and binding.get("mode", "hold") == "hold" and 
                    blendshape_values[name] >= binding.get("threshold", self.default_threshold)):
                    
                    priority = priority_list.index(name)
                    if priority < best_priority:
                        best_priority = priority
                        best_candidate = name
        
        if best_candidate and not current_active:
            self._hold_category(category, best_candidate)

    def _hold_category(self, category, blendshape_name):
        self.active_categories[category] = blendshape_name
        binding = self._find_binding(blendshape_name)
        action = binding["action"]
        
        if category == 'mouth' and not self.active_key:
            self.active_key = blendshape_name
            self.active_action = action
        
        try:
            if action in self.actions["mouse"]:
                if action in ["mouse_click", "mouse_left_click"]:
                    pyautogui.mouseDown(button="left")
                elif action == "mouse_right_click":
                    pyautogui.mouseDown(button="right")
                elif action == "mouse_middle_click":
                    pyautogui.mouseDown(button="middle")
                    
            elif action.startswith("key_"):
                key = action[4:]  
                pyautogui.keyDown(key)
                
            print(f"[{category}] Key Down: {blendshape_name} -> {action}")
        except Exception as e:
            print(f"Error in {category}: {e}")
            self.active_categories[category] = None

    def _release_category(self, category):
        blendshape_name = self.active_categories[category]
        if not blendshape_name:
            return
            
        binding = self._find_binding(blendshape_name)
        action = binding["action"]

        if category == 'mouth':
            self.active_key = None
            self.active_action = None
        
        try:
            if action in self.actions["mouse"]:
                if action in ["mouse_click", "mouse_left_click"]:
                    pyautogui.mouseUp(button="left")
                elif action == "mouse_right_click":
                    pyautogui.mouseUp(button="right")
                elif action == "mouse_middle_click":
                    pyautogui.mouseUp(button="middle")
                    
            elif action.startswith("key_"):
                key = action[4:] 
                pyautogui.keyUp(key)
                
            print(f"[{category}] Key Up: {blendshape_name} -> {action}")
        except Exception as e:
            print(f"Error releasing {category}: {e}")
        finally:
            self.active_categories[category] = None

    def _process_press_mode(self, blendshape_values, current_time):
        mouth_candidates = []
        eye_candidates = []
        brow_candidates = []
        
        for name, value in blendshape_values.items():
            binding = self._find_binding(name)
            if (binding and binding.get("mode", "hold") == "press" and 
                value >= binding.get("threshold", self.default_threshold)):
                
                last_press = self.last_press_time.get(name, 0)
                if current_time - last_press < self.press_cooldown:
                    continue
                
                candidate = {
                    "name": name,
                    "value": value,
                    "action": binding["action"],
                    "binding": binding
                }
                
                if name in self.mouth_priority:
                    candidate["priority"] = self.mouth_priority.index(name)
                    mouth_candidates.append(candidate)
                elif name in self.eye_priority:
                    candidate["priority"] = self.eye_priority.index(name)
                    eye_candidates.append(candidate)
                elif name in self.brow_priority:
                    candidate["priority"] = self.brow_priority.index(name)
                    brow_candidates.append(candidate)
        
        actions_to_press = []
        
        if mouth_candidates:
            mouth_candidates.sort(key=lambda x: x["priority"])
            actions_to_press.append(mouth_candidates[0])
        
        if eye_candidates:
            eye_candidates.sort(key=lambda x: x["priority"])
            actions_to_press.append(eye_candidates[0])

        if brow_candidates:
            brow_candidates.sort(key=lambda x: x["priority"])
            actions_to_press.append(brow_candidates[0])
        
        for candidate in actions_to_press:
            self._execute_press_action(candidate["name"], candidate["action"])
            self.last_press_time[candidate["name"]] = current_time

            if candidate == actions_to_press[-1]:
                return candidate["action"], candidate["value"]
        
        return None, 0
    
    def _execute_press_action(self, blendshape_name, action):
        try:
            if action in self.actions["mouse"]:
                if action in ["mouse_click", "mouse_left_click"]:
                    pyautogui.click()
                elif action == "mouse_right_click":
                    pyautogui.click(button="right")
                elif action == "mouse_middle_click":
                    pyautogui.click(button="middle")
                elif action == "mouse_double_click":
                    pyautogui.doubleClick()
                elif action == "scroll_up":
                    pyautogui.scroll(3)
                elif action == "scroll_down":
                    pyautogui.scroll(-3)
                    
            elif action.startswith("key_"):
                key = action[4:]  
                pyautogui.press(key)
                
            print(f"Press Action: {blendshape_name} -> {action}")
        except Exception as e:
            print(f"Error executing press action: {e}")

    def is_mouth_recently_open(self):
        if self.jaw_open_counter > 50:
            return False
        return True
    
    def _find_binding(self, blendshape_name):
        for binding in self.bindings:
            if binding["blendshape"] == blendshape_name:
                return binding
        return None

    def _get_threshold(self, blendshape_name):
        binding = self._find_binding(blendshape_name)
        if binding:
            return binding.get("threshold", self.default_threshold)
        return self.default_threshold
    
    def get_blendshape_value(self, blendshape_name):
        if hasattr(self, 'current_blendshapes') and self.current_blendshapes:
            for blendshape in self.current_blendshapes:
                if blendshape.category_name == blendshape_name:
                    return blendshape.score
        return 0.0

    def _hold_key(self, blendshape_name, action):
        self.active_key = blendshape_name
        self.active_action = action
        
        try:
            if action in self.actions["mouse"]:
                if action in ["mouse_click", "mouse_left_click"]:
                    pyautogui.mouseDown(button="left")
                elif action == "mouse_right_click":
                    pyautogui.mouseDown(button="right")
                elif action == "mouse_middle_click":
                    pyautogui.mouseDown(button="middle")
                elif action in ["scroll_up", "scroll_down"]:
                    pass 
                    
            elif action.startswith("key_"):
                key = action[4:]  
                pyautogui.keyDown(key)
                
            print(f"Key Down: {blendshape_name} -> {action}")
        except Exception as e:
            print(f"Error pressing key: {e}")
            self.active_key = None
            self.active_action = None

    def _release_key(self):
        if not self.active_key or not self.active_action:
            return
            
        try:
            action = self.active_action
            
            if action in self.actions["mouse"]:
                if action in ["mouse_click", "mouse_left_click"]:
                    pyautogui.mouseUp(button="left")
                elif action == "mouse_right_click":
                    pyautogui.mouseUp(button="right")
                elif action == "mouse_middle_click":
                    pyautogui.mouseUp(button="middle")
                elif action in ["scroll_up", "scroll_down"]:
                    pass
                    
            elif action.startswith("key_"):
                key = action[4:] 
                pyautogui.keyUp(key)
                
            print(f"Key Up: {self.active_key} -> {action}")
        except Exception as e:
            print(f"Error releasing key: {e}")
        finally:
            self.active_key = None
            self.active_action = None

    def on_profile_change(self):
        self.load_from_profile()

    def get_available_blendshapes(self):
        return self.blendshapes
    
    def get_available_actions(self, category=None):
        if category and category in self.actions:
            return self.actions[category]
        elif category:
            return []  
        else:
            return self.actions
        
    def get_action_category(self, action):
        for category, actions in self.actions.items():
            if action in actions:
                return category
        return None
    
    def add_binding(self, blendshape, action, threshold=None, mode="hold"):
        if threshold is None:
            threshold = self.default_threshold
            
        for binding in self.bindings:
            if binding["blendshape"] == blendshape:
                binding["action"] = action
                binding["threshold"] = threshold
                binding["mode"] = mode
                self.save_to_profile()
                return True
        
        self.bindings.append({
            "blendshape": blendshape,
            "action": action,
            "threshold": threshold,
            "mode": mode
        })
        
        self.save_to_profile()
        return True
    
    def update_binding_mode(self, blendshape, mode):
        for binding in self.bindings:
            if binding["blendshape"] == blendshape:
                binding["mode"] = mode
                self.save_to_profile()
                return True
        return False
    
    def get_binding_mode(self, blendshape):
        binding = self._find_binding(blendshape)
        return binding.get("mode", "hold") if binding else "hold"
    
    def remove_binding(self, blendshape):
        for i, binding in enumerate(self.bindings):
            if binding["blendshape"] == blendshape:
                self.bindings.pop(i)
                    
                self.save_to_profile()
                return True
        
        return False

    def set_bindings(self, bindings):
        self.bindings = bindings
        self.save_to_profile()
    
    def cleanup(self):
        if self.active_key:
            self._release_key()
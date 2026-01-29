import numpy as np
import numpy.typing as npt
import pyautogui
from src.accel import SigmoidAccel
import time
from src.modified_oneEuroFilter import OneEuroFilter
import threading
import queue
import keyboard
import math

class MouseController:
    def __init__(self):
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0
        pyautogui.MINIMUM_DURATION = 0
        pyautogui.MINIMUM_SLEEP = 0.0049
        self.mincutoff = 0.5
        self.beta = 0.07
        self.vx = 0
        self.vy = 0
        config = {
            'freq': 30,      
            'mincutoff': self.mincutoff, 
            'beta': self.beta,       
            'dcutoff': 1.0    
            }

        self.f1 = OneEuroFilter(**config)
        self.prev_smooth_position = None
        self.velocity_scale = 35
        self.accel = SigmoidAccel()
        self.get_cursor = None
        self.checkk = False
        self.state_machine = True
        self.sending_simulated_key = False
        self.pressed_mouse_keys = set()
        self.mouse_buttons_held = {'left': False, 'middle': False, 'right': False}
        self.setup_keyboard_listeners()
        self.tracking_active = False
        self.update_thread = None
        self.lock = threading.Lock()
        self.tmp = time.time()
        self.x_now = 0
        self.y_now = 0
        self.state_machine_blendshape_index = 3
        self.delay = 0
        self.trigger_threshold = 0.5
        self.toggle = True
        self.accel_on = True
        self.is_recent_typing = False
        self.last_typing_time = 0
        self.warning_duration = 0.1 
        self.control_keys = {
            'w', 'a', 's', 'd', 'j', 'k', 'l', 'e',
            'ctrl', 'right ctrl', 'left ctrl',
            'alt', 'right alt', 'left alt', 
            'shift', 'right shift', 'left shift',
            'win', 'right win', 'left win',
            'enter', 'tab', 'caps lock', 'esc', 'menu', 
            'num lock', 'scroll lock', 'print screen', 'pause', 'insert',
            'backspace', 'delete', 
            'up', 'down', 'left', 'right',
            'home', 'end', 'page up', 'page down',
            'volume up', 'volume down', 'volume mute', 
            'play/pause', 'stop media', 'next track', 'previous track'
        }
        for i in range(1, 13):
            self.control_keys.add(f'f{i}')
        self.setup_global_listener()
    def setup_global_listener(self):
        keyboard.hook(self._on_any_key_event)
    @property
    def should_show_warning(self):
        """
        Overlay gọi biến này.
        Trả về True nếu người dùng vừa gõ phím văn bản trong khi đang bật chế độ chuột.
        """
        self.is_recent_typing = (time.time() - self.last_typing_time) < self.warning_duration
        
        return self.tracking_active and self.state_machine and self.is_recent_typing
    def _on_any_key_event(self, e):
        if not self.tracking_active or not self.state_machine:
            return
        if e.event_type != 'down':
            return
        if keyboard.is_pressed('ctrl') or keyboard.is_pressed('alt') or keyboard.is_pressed('win'):
            return

        try:
            key_name = e.name.lower()
            if key_name in self.control_keys:
                return
            self.last_typing_time = time.time()
        except Exception:
            pass
    def reset(self):
        config = {
            'freq': 120,      
            'mincutoff': self.mincutoff,  
            'beta': self.beta,       
            'dcutoff': 1.0    
            }
        self.f1 = OneEuroFilter(**config)
        self.prev_smooth_position = None

    def set_get_cursor(self, get_cursor_func):
        self.get_cursor = get_cursor_func
        print("Get cursor function set successfully")
    
    def apply_smoothing(self, point):
        current_time = time.time()
        return self.f1(math.sqrt(point[0]**2+point[1]**2), current_time)
    
    def move(self, landmark):
        
        _, alpha = self.apply_smoothing(landmark)
        
        if self.prev_smooth_position is not None:
            self.vx, self.vy = ((landmark - self.prev_smooth_position) * alpha + 
                                (1 - alpha) * (np.array([self.vx, self.vy])))
            
            self.prev_smooth_position = landmark
            if self.accel_on:
                vx = -self.vx * self.accel(self.vx * self.velocity_scale) * self.velocity_scale
                vy = self.vy * self.accel(self.vy * self.velocity_scale) * self.velocity_scale
            else:
                vx = -self.vx * self.velocity_scale
                vy = self.vy * self.velocity_scale
            pyautogui.moveRel(vx/2, vy/2, duration=0)
            time.sleep(0.01)
            pyautogui.moveRel(vx/2, vy/2, duration=0)
        else:
            self.prev_smooth_position = landmark
            
        return landmark

    def update_loop(self, cursor_pos=None, blendshape=None):
        try:
            if blendshape is not None:
                trigger_blendshape = blendshape[self.state_machine_blendshape_index]

                if self.toggle and self.tracking_active:
                    if trigger_blendshape > self.trigger_threshold and self.checkk == False:
                        self.checkk = True
                        self.state_machine = not self.state_machine
                        print(f"Keyboard Mode: {'ON' if self.state_machine else 'OFF'}")
                    elif trigger_blendshape <= self.trigger_threshold:
                        self.checkk = False  
                elif self.tracking_active:
                    if (not self.state_machine) and trigger_blendshape > self.trigger_threshold:
                        self.state_machine = True
                    elif self.state_machine and trigger_blendshape <= self.trigger_threshold * 0.5:
                        self.state_machine = False
            if self.tracking_active and cursor_pos is not None and time.time() - self.delay > 0.15:
                self.move(cursor_pos)

        except Exception as e:
            print(f"Error in mouse update loop: {e}")
    def setup_keyboard_listeners(self):
        def handle_key_logic(e, action_type, param):            
            if self.sending_simulated_key:
                return
            is_shortcut = keyboard.is_pressed('ctrl') or keyboard.is_pressed('alt') or keyboard.is_pressed('win')
            should_bypass = not self.tracking_active or is_shortcut or self.state_machine == False

            if should_bypass:
                self.sending_simulated_key = True
                if e.event_type == 'down':
                    keyboard.press(e.name)
                else:
                    keyboard.release(e.name)
                self.sending_simulated_key = False
                return
            
            if action_type == 'mouse':
                mouse_button = param['btn']
                
                if e.event_type == 'down':
                    if not self.mouse_buttons_held[mouse_button]:
                        pyautogui.mouseDown(button=mouse_button)
                        self.mouse_buttons_held[mouse_button] = True
                        self.delay = time.time()
                        print(f"Mouse {mouse_button} down")
                elif e.event_type == 'up':
                    if self.mouse_buttons_held[mouse_button]:
                        pyautogui.mouseUp(button=mouse_button)
                        self.mouse_buttons_held[mouse_button] = False
                        print(f"Mouse {mouse_button} up")
        def handle_move_logic(e):
            if self.sending_simulated_key: return
            is_shortcut = keyboard.is_pressed('ctrl') or keyboard.is_pressed('alt') or keyboard.is_pressed('win')
            should_bypass = not self.tracking_active or is_shortcut or self.state_machine == False

            if should_bypass:
                self.sending_simulated_key = True
                if e.event_type == 'down': keyboard.press(e.name)
                else: keyboard.release(e.name)
                self.sending_simulated_key = False
                return

            arrow_key_map = {
                'w': 'up',
                'a': 'left',
                's': 'down',
                'd': 'right'
            }

            key_char = e.name.lower()
            arrow_key = arrow_key_map.get(key_char)

            if arrow_key:
                self.sending_simulated_key = True
                if e.event_type == 'down':
                    keyboard.press(arrow_key)
                else:
                    keyboard.release(arrow_key)
                self.sending_simulated_key = False
                

        try:
            keyboard.hook_key('j', lambda e: handle_key_logic(e, 'mouse', {'btn': 'left'}), suppress=True)
            keyboard.hook_key('k', lambda e: handle_key_logic(e, 'mouse', {'btn': 'middle'}), suppress=True)
            keyboard.hook_key('l', lambda e: handle_key_logic(e, 'mouse', {'btn': 'right'}), suppress=True)
            keyboard.hook_key('w', lambda e: handle_move_logic(e), suppress=True)
            keyboard.hook_key('a', lambda e: handle_move_logic(e), suppress=True)
            keyboard.hook_key('s', lambda e: handle_move_logic(e), suppress=True)
            keyboard.hook_key('d', lambda e: handle_move_logic(e), suppress=True)
            print("Keyboard listeners setup complete. Press 'E' to switch click modes.")
        except Exception as e:
            print(f"Error setting up keyboard listeners: {e}")
    def start_tracking(self):
        with self.lock:
            self.tracking_active = True
            self.prev_smooth_position = None
            print("Mouse tracking started")
    def stop_tracking(self):
        self.tracking_active = False
        print("Mouse tracking stopped")
    def click(self):
        pyautogui.click()
    def increase_speed(self, step=5):
        try:
            current = self.velocity_scale
            new_speed = min(50, current + step) 
            self.velocity_scale = new_speed
            return True
        except Exception as e:
            print(f"Error increasing mouse speed: {e}")
            return False
    def decrease_speed(self, step=5):
        try:
            current = self.velocity_scale
            new_speed = max(1, current - step) 
            self.velocity_scale = new_speed
            return True
        except Exception as e:
            print(f"Error decreasing mouse speed: {e}")
            return False
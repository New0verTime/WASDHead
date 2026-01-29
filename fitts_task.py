import pygame
import sys
import math
import random
import time
import pandas as pd
from datetime import datetime
import cv2
import mediapipe as mp
import numpy as np

# --- CẤU HÌNH THÍ NGHIỆM ---
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
# --- DANH SÁCH TỪ VỰNG ---
COMMON_WORDS = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
    "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
    "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
    "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
    "even", "new", "want", "because", "any", "these", "give", "day", "most", "us"
]
# Biến độc lập (Independent Variables)
TARGET_DISTANCES = [1000, 600] # px (Khoảng cách từ tâm ra target)
TARGET_WIDTHS = [60, 100]      # px (Đường kính target)
REPETITIONS = 10              # Số lần lặp lại (Total = 2*2*12 = 48 trials)

# Thông tin lưu Log (Chỉ để ghi vào file CSV)
PARTICIPANT_ID = "P01"
CONTROL_METHOD = "WASDHead"   # Đặt tên này để phân biệt file log (ví dụ: "Mouse" hoặc "WASDHead")

# Text yêu cầu

# Màu sắc
COLOR_BG = (20, 20, 20)
COLOR_TEXT = (255, 255, 255)
COLOR_TARGET = (0, 200, 100)
COLOR_CENTER = (100, 100, 100)
COLOR_PROMPT_1 = (100, 255, 100)
COLOR_PROMPT_2 = (100, 100, 255)

# --- KHỞI TẠO PYGAME ---
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption(f"Fitts Task - Condition: {CONTROL_METHOD}")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Consolas", 24)
font_large = pygame.font.SysFont("Consolas", 36)

# Đảm bảo con trỏ chuột hệ thống luôn hiển thị
pygame.mouse.set_visible(True)

# --- CLASS QUẢN LÝ DỮ LIỆU ---
class DataLogger:
    def __init__(self):
        self.data = []

    def log_trial(self, trial_info, metrics):
        entry = {**trial_info, **metrics}
        self.data.append(entry)

    def save_to_csv(self):
        if not self.data:
            return
        filename = f"ExpData_{PARTICIPANT_ID}_{CONTROL_METHOD}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df = pd.DataFrame(self.data)
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")
class HandTracker:
    def __init__(self, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Khởi tạo module theo dõi tay Mediapipe.
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Khởi tạo model Hands
        self.hands = self.mp_hands.Hands(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            max_num_hands=1  # Chỉ track 1 tay
        )
        
        # Các biến trạng thái
        self.home_wrist_pos = None  # Tọa độ gốc (x, y)
        self.max_displacement = 0.0 # Khoảng cách lớn nhất ghi nhận được
        self.current_displacement = 0.0

    def _calculate_distance(self, p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def reset(self):
        """
        Reset trạng thái: Xóa điểm gốc và reset max distance về 0.
        Điểm gốc mới sẽ được thiết lập tự động ở frame tiếp theo tìm thấy tay.
        """
        self.home_wrist_pos = None
        self.max_displacement = 0.0
        self.current_displacement = 0.0
        print(">> HandTracker: Đã Reset. Đang chờ vị trí tay mới...")

    def get_max_displacement(self):
        """Trả về độ lệch lớn nhất tính từ lần reset cuối cùng."""
        return self.max_displacement

    def process_frame(self, image):
        # 1. Chuẩn bị ảnh (MediaPipe cần RGB)
        # Lưu ý: Không flip ảnh ở đây để giữ nguyên tọa độ gốc của input. 
        # Việc flip nên làm ở ngoài trước khi truyền vào nếu cần.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 2. Xử lý
        results = self.hands.process(image_rgb)
        h, w, _ = image.shape
        found_hand =False
        if results.multi_hand_landmarks:
            found_hand = True
            for hand_landmarks in results.multi_hand_landmarks:
                # Lấy tọa độ cổ tay (Landmark 0)
                wrist = hand_landmarks.landmark[0]
                cx, cy = int(wrist.x * w), int(wrist.y * h)
                if self.home_wrist_pos is None:
                    self.home_wrist_pos = (cx, cy)
                    print(f">> HandTracker: Đã set Home Position tại ({cx}, {cy})")
                self.current_displacement = self._calculate_distance(self.home_wrist_pos, (cx, cy))
                if self.current_displacement > self.max_displacement:
                    self.max_displacement = self.current_displacement
        return found_hand
    def close(self):
        self.hands.close()
# --- CLASS XỬ LÝ LOGIC THÍ NGHIỆM ---
class Experiment:
    def __init__(self):
        self.trials = self.generate_trials()
        self.current_trial_idx = -1
        self.logger = DataLogger()
        self.state = "START_SCREEN" 
        self.hand_tracker = HandTracker(model_complexity=1)
        # Target info
        self.target_pos = (0, 0)
        self.target_radius = 0
        self.start_pos = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
        self.current_text_prior = "start" 
        self.current_text_posterior = "end"        # Typing info
        self.input_buffer = ""
        
        # Timestamps
        self.t_last_keystroke_prior = 0
        self.t_start_movement = 0
        self.t_target_selected = 0
        self.t_first_keystroke_posterior = 0
        
        # Misc
        self.attempts = 0
        self.movement_started = False

    def generate_trials(self):
        trials = []
        for d in TARGET_DISTANCES:
            for w in TARGET_WIDTHS:
                for _ in range(REPETITIONS):
                    trials.append({'distance': d, 'width': w})
        random.shuffle(trials)
        return trials

    def next_trial(self):
        self.current_trial_idx += 1
        if self.current_trial_idx >= len(self.trials):
            self.state = "FINISHED"
            self.logger.save_to_csv()
            return
        self.current_text_prior = random.choice(COMMON_WORDS)
        self.current_text_posterior = random.choice(COMMON_WORDS)
        trial_setup = self.trials[self.current_trial_idx]
        
        # Tính vị trí Target ngẫu nhiên theo hình tròn Fitts
        angle = random.uniform(0, 2 * math.pi)
        dist = trial_setup['distance']
        
        cx, cy = self.start_pos
        tx = cx + int(dist/2 * math.cos(angle)) 
        ty = cy + int(dist/2 * math.sin(angle))
        
        # Giới hạn trong màn hình
        tx = max(50, min(SCREEN_WIDTH-50, tx))
        ty = max(50, min(SCREEN_HEIGHT-50, ty))

        self.target_pos = (tx, ty)
        self.target_radius = trial_setup['width'] // 2
        
        # Reset biến
        self.input_buffer = ""
        self.state = "PRIOR_TYPING"
        self.attempts = 0
        self.movement_started = False
        
        # Reset chuột về giữa màn hình để chuẩn bị cho lượt mới
        # Điều này áp dụng cho cả Mouse thường và WASDHead (nếu nó điều khiển cursor hệ thống)
        pygame.mouse.set_pos(self.start_pos)

    def handle_input(self, event, frame=None):
        if frame is None:
            frame = np.full((640, 480, 3), 255, dtype=np.uint8)  # Dummy white frame for hand tracker
        if self.state == "START_SCREEN":
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.next_trial()
        
        elif self.state == "PRIOR_TYPING":
            if event.type == pygame.KEYDOWN:
                self.process_typing(event, target_word=self.current_text_prior, next_state="MOVING", frame=frame)

        elif self.state == "MOVING":
            # Xử lý Click: Hỗ trợ cả chuột trái và phím Space/Enter (phòng trường hợp WASDHead dùng phím để click)
            clicked = False
            click_pos = (0,0)
            self.hand_tracker.process_frame(frame)  # Cập nhật vị trí tay liên tục
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # Left Click
                    clicked = True
                    click_pos = event.pos
            

            if clicked:
                self.check_target_hit(click_pos)

        elif self.state == "POSTERIOR_TYPING":
            if event.type == pygame.KEYDOWN:
                self.process_typing(event, target_word=self.current_text_posterior, next_state="NEXT_TRIAL")

    def check_start_movement(self):
        # Hàm này kiểm tra xem con trỏ đã rời khỏi vị trí trung tâm chưa
        if self.state == "MOVING" and not self.movement_started:
            curr = pygame.mouse.get_pos()
            # Tính khoảng cách từ tâm
            dist = math.hypot(curr[0] - self.start_pos[0], curr[1] - self.start_pos[1])
            
            # Deadzone 10px để tránh rung nhẹ
            if dist > 10: 
                self.movement_started = True
                self.t_start_movement = time.time()

    def process_typing(self, event, target_word, next_state, frame=None):
        if event.unicode.isalpha():
            self.input_buffer += event.unicode
        elif event.key == pygame.K_BACKSPACE:
            self.input_buffer = self.input_buffer[:-1]
        
        # Capture timestamp ký tự đầu tiên của giai đoạn sau (Switching cost part 2)
        if self.state == "POSTERIOR_TYPING" and len(self.input_buffer) == 1 and self.t_first_keystroke_posterior == 0:
            self.t_first_keystroke_posterior = time.time()

        # Kiểm tra hoàn thành từ
        if self.input_buffer == target_word:
            now = time.time()
            if next_state == "MOVING":
                self.hand_tracker.reset()
                self.hand_tracker.process_frame(frame)  # Gọi ngay để set home position
                # Kết thúc gõ 'hello'
                self.t_last_keystroke_prior = now
                self.state = "MOVING"
                # Reset cursor lần nữa để chắc chắn
                pygame.mouse.set_pos(self.start_pos)
                self.input_buffer = ""
                
            elif next_state == "NEXT_TRIAL":
                # Kết thúc gõ 'goodbye'
                self.calculate_metrics()
                self.next_trial()
                self.t_first_keystroke_posterior = 0

    def check_target_hit(self, pos):
        self.attempts += 1
        dist = math.hypot(pos[0] - self.target_pos[0], pos[1] - self.target_pos[1])
        
        if dist <= self.target_radius:
            self.t_target_selected = time.time()
            self.state = "POSTERIOR_TYPING"
            self.input_buffer = ""
        else:
            # Miss click
            pass

    def calculate_metrics(self):
        # Nếu người dùng click quá nhanh hoặc movement tracking bị trễ, lấy t_start = t_last_keystroke (worst case)
        if self.t_start_movement == 0:
            self.t_start_movement = self.t_last_keystroke_prior

        # 1. Acquisition Time
        acq_time = self.t_target_selected - self.t_start_movement
        
        # 2. Switching Time
        switch_1 = self.t_start_movement - self.t_last_keystroke_prior
        switch_2 = self.t_first_keystroke_posterior - self.t_target_selected
        
        # Sanity check cho số âm
        if switch_1 < 0: switch_1 = 0
        if switch_2 < 0: switch_2 = 0
        
        switching_time = switch_1 + switch_2
        
        # Lưu Log
        trial_info = self.trials[self.current_trial_idx]
        maximum_displacement = self.hand_tracker.get_max_displacement()
        metrics = {
            "Control": CONTROL_METHOD,
            "AcquisitionTime": round(acq_time, 4),
            "SwitchingTime": round(switching_time, 4),
            "Homing_In": round(switch_1, 4),
            "Homing_Out": round(switch_2, 4),
            "SuccessRate": 1 if self.attempts == 1 else 0,
            "Attempts": self.attempts,
            "MaxDisplacement": round(maximum_displacement, 2)
        }
        self.logger.log_trial(trial_info, metrics)
        print(f"Trial {self.current_trial_idx+1}: Acq={acq_time:.3f}s, Switch={switching_time:.3f}s")

    def draw(self, screen):
        screen.fill(COLOR_BG)
        
        # Thông tin Header
        status = f"Trial: {self.current_trial_idx + 1}/{len(self.trials)}"
        screen.blit(font.render(status, True, COLOR_CENTER), (10, 10))

        if self.state == "START_SCREEN":
            txt1 = font_large.render("Fitts' Law Experiment", True, COLOR_TEXT)
            txt2 = font.render(f"Condition: {CONTROL_METHOD} | Press SPACE to Start", True, COLOR_TEXT)
            screen.blit(txt1, (SCREEN_WIDTH//2 - txt1.get_width()//2, SCREEN_HEIGHT//2 - 40))
            screen.blit(txt2, (SCREEN_WIDTH//2 - txt2.get_width()//2, SCREEN_HEIGHT//2 + 20))
            
        elif self.state == "FINISHED":
            txt = font_large.render("DONE! Log saved.", True, COLOR_TARGET)
            screen.blit(txt, (SCREEN_WIDTH//2 - txt.get_width()//2, SCREEN_HEIGHT//2))

        elif self.state == "PRIOR_TYPING":
            prompt = font.render(f"Type '{self.current_text_prior}'", True, COLOR_TEXT)
            inp = font_large.render(self.input_buffer, True, COLOR_PROMPT_1)
            screen.blit(prompt, (SCREEN_WIDTH//2 - prompt.get_width()//2, SCREEN_HEIGHT//2 - 60))
            screen.blit(inp, (SCREEN_WIDTH//2 - inp.get_width()//2, SCREEN_HEIGHT//2))

        elif self.state == "POSTERIOR_TYPING":
            prompt = font.render(f"Type '{self.current_text_posterior}'", True, COLOR_TEXT)
            inp = font_large.render(self.input_buffer, True, COLOR_PROMPT_2)
            screen.blit(prompt, (SCREEN_WIDTH//2 - prompt.get_width()//2, SCREEN_HEIGHT//2 - 60))
            screen.blit(inp, (SCREEN_WIDTH//2 - inp.get_width()//2, SCREEN_HEIGHT//2))
        elif self.state == "MOVING":
            # Vẽ Target
            pygame.draw.circle(screen, COLOR_TARGET, self.target_pos, self.target_radius)
            # Vẽ điểm xuất phát (để user biết phải di chuyển từ đâu)
            pygame.draw.circle(screen, COLOR_CENTER, self.start_pos, 5)

# --- MAIN LOOP ---
if __name__ == "__main__":
    experiment = Experiment()
    CAMERA_INDEX = 0 # Đổi thành 0 nếu dùng webcam laptop
    cap = cv2.VideoCapture(CAMERA_INDEX)
    while True:
        success, frame = cap.read()
        # 1. Xử lý sự kiện (Gõ phím, Click, Quit)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                experiment.logger.save_to_csv()
                pygame.quit()
                sys.exit()
            experiment.handle_input(event,frame=frame)

        # 2. Kiểm tra con trỏ hệ thống có bắt đầu di chuyển chưa
        experiment.check_start_movement()

        # 3. Vẽ giao diện
        experiment.draw(screen)
        
        pygame.display.flip()
        clock.tick(60)
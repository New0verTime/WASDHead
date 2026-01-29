import tkinter as tk
import pyautogui

class Overlay:
    def __init__(self, mouse_controller):
        self.radius = 65
        self.color = "#FFD700"
        self.update_interval = 20 
        self.hide_distance = 0
        self.fade_distance = self.hide_distance * 1.5
        self.mouse_controller = mouse_controller
        self.root = tk.Toplevel()
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-transparentcolor", "white")
        self.root.config(bg="white")
        self.size = self.radius * 2 + 10
        self.canvas = tk.Canvas(self.root, width=self.size, height=self.size, bg="white", highlightthickness=0)
        self.canvas.pack()
        self.notification_window = None
        self.running = True
        self.temp = True
    def _draw_dot(self, center_x, center_y):
        dot_radius = 12  # Bán kính chấm (có thể điều chỉnh)
        self.canvas.create_oval(
            center_x - dot_radius, center_y - dot_radius,
            center_x + dot_radius, center_y + dot_radius,
            fill="red", outline="", tags="gaze_circle", stipple="gray50"
        )

    def show_message(self, text, duration=1000):
        """
        Hiển thị thông báo có khung, nền xám nhạt, chữ xám đậm, trong suốt 0.5
        """
        try:
            # Hủy thông báo cũ nếu đang hiện
            if self.notification_window is not None:
                self.notification_window.destroy()

            # Tạo cửa sổ mới
            win = tk.Toplevel()
            win.overrideredirect(True)      # Mất thanh tiêu đề
            win.attributes("-topmost", True) # Luôn nổi lên trên
            win.attributes("-alpha", 0.5)    # Độ trong suốt 50% cho cả cửa sổ

            # --- Cấu hình màu sắc ---
            bg_color = "#E0E0E0"  # Màu xám nhạt (Light Gray)
            text_color = "#333333" # Màu xám đậm (Dark Gray)
            
            win.config(bg=bg_color)

            # Thêm viền (Frame) để trông đẹp hơn (Optional)
            frame = tk.Frame(win, bg=bg_color, bd=2, relief="ridge")
            frame.pack(fill="both", expand=True)

            # Tạo Label chứa chữ
            # padx, pady ở đây tạo khoảng cách từ chữ ra mép khung
            label = tk.Label(
                frame, 
                text=text, 
                font=("Arial", 24, "bold"), # Giảm size font một chút cho tinh tế
                fg=text_color, 
                bg=bg_color,
                padx=20, 
                pady=10
            )
            label.pack()

            # Tính toán vị trí giữa màn hình
            win.update_idletasks() 
            width = win.winfo_width()
            height = win.winfo_height()
            screen_width = win.winfo_screenwidth()
            
            # Canh giữa theo chiều ngang, cách top 100px
            x = (screen_width - width) // 2
            y = 100 
            
            win.geometry(f"{width}x{height}+{x}+{y}")

            # Lưu tham chiếu và hẹn giờ tắt
            self.notification_window = win
            win.after(duration, self._destroy_notification)
            
        except Exception as e:
            print(f"Error showing message: {e}")
    def _destroy_notification(self):
        """Hàm phụ để tắt thông báo an toàn"""
        if self.notification_window:
            try:
                self.notification_window.destroy()
            except:
                pass
            self.notification_window = None
    def update_once(self):
        self.canvas.delete("gaze_circle")
        if self.mouse_controller.should_show_warning:
            self.show_message("Please Change to keyboard mode!", duration=5000)
            self.mouse_controller.is_recent_typing = False
        if self.temp != self.mouse_controller.state_machine:
            if not self.mouse_controller.state_machine:
                self.show_message("WASDHead mode off", duration=5000)
            else:
                self.show_message("WASDHead mode on", duration=5000)
            self.temp = self.mouse_controller.state_machine
        # if not self.running or not self.mouse_controller.tracking_active or self.mouse_controller.state_machine:
        #     return
        # x, y = self.mouse_controller.x_now, self.mouse_controller.y_now
        # if x is not None and y is not None:
        #     x, y = int(x), int(y)

        #     self.root.deiconify()
        #     self.root.attributes("-alpha", 0.7)
        #     self.root.geometry(f"{self.size}x{self.size}+{x - self.radius}+{y - self.radius}")
        #     self._draw_dot(
        #         center_x=self.size // 2,
        #         center_y=self.size // 2
        #     )

    def close(self):
        self.running = False
        try:
            self.root.destroy()
        except tk.TclError:
            pass
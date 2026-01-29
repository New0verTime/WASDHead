import tkinter as tk
import customtkinter as ctk

class SubmenuDropdown(ctk.CTkFrame):
    def __init__(self, parent, actions_dict, current_action="", callback=None):
        super().__init__(parent)
        
        self.actions_dict = actions_dict
        self.callback = callback
        self.current_action = current_action
        
        self.main_button = ctk.CTkButton(
            self,
            text=current_action if current_action else "Select Action...",
            command=self.show_menu
        )
        self.main_button.pack(fill="x")
    
    def show_menu(self):
        menu = tk.Menu(self, tearoff=0)
        
        for category, actions in self.actions_dict.items():
            submenu = tk.Menu(menu, tearoff=0)
            
            for action in actions:
                submenu.add_command(
                    label=action.replace('_', ' ').title(),
                    command=lambda a=action: self.select_action(a)
                )
            
            menu.add_cascade(
                label=category.replace('_', ' ').title(),
                menu=submenu
            )
        
        try:
            x = self.main_button.winfo_rootx()
            y = self.main_button.winfo_rooty() + self.main_button.winfo_height()
            menu.tk_popup(x, y)
        finally:
            menu.grab_release()
    
    def select_action(self, action):
        self.current_action = action
        self.main_button.configure(text=action)
        if self.callback:
            self.callback(action)

    def get_selected_action(self):
        return self.current_action if self.current_action else None
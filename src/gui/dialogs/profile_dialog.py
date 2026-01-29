import tkinter as tk
import customtkinter as ctk

class ProfileDialog:
    def __init__(self, parent):
        self.parent = parent
        self.result = None
        
        self.dialog = ctk.CTkInputDialog(text="Enter new profile name:", title="New Profile")
    
    def get_input(self):
        return self.dialog.get_input()
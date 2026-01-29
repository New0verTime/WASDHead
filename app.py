import tkinter as tk
import customtkinter as ctk
from src.pipeline import Pipeline
from src.gui.main_window import MainWindow

def main():
    pipeline = Pipeline()
    pipeline.start()

    app = MainWindow()
    app.mainloop()

if __name__ == "__main__":
    main()
    

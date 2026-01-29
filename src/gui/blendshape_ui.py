import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk
import os
import sys
from src.gui.submenu import SubmenuDropdown

class BlendshapeSettingsUI(ctk.CTkFrame):
    def __init__(self, parent, blendshape_processor, profile_manager, current_settings):
        super().__init__(parent)
        self.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.blendshape_processor = blendshape_processor
        self.profile_manager = profile_manager
        self.current_settings = current_settings
        
        self.available_blendshapes = self.blendshape_processor.get_available_blendshapes()
        self.available_actions = self.blendshape_processor.get_available_actions()
        
        self.threshold_vars = {}
        self.threshold_labels = {}
        self.blendshape_bars = {}
        self.blendshape_value_labels = {}

        self._create_blendshape_ui()

        self.update_bars()
    
    def _create_blendshape_ui(self):
        # Button to add binding
        add_btn = ctk.CTkButton(self, text="Add Binding", command=self._add_binding)
        add_btn.pack(fill="x", padx=5, pady=5)
        
        # Bindings section
        self._load_bindings()
    
    def update_bars(self):
        for blendshape_name, progress_bar in self.blendshape_bars.items():
            current_value = self.blendshape_processor.get_blendshape_value(blendshape_name)
            
            if progress_bar:
                progress_bar.set(current_value)
            
            if blendshape_name in self.blendshape_value_labels:
                self.blendshape_value_labels[blendshape_name].configure(text=f"{current_value:.2f}")
        
        self.after(33, self.update_bars)

    def _update_threshold(self, value):
        value = float(value)
        self.threshold_label.configure(text=f"{value:.2f}")
        self.blendshape_processor.set_threshold(value)
    
    def _load_bindings(self):
        if hasattr(self, 'bindings_frame'):
            self.bindings_frame.destroy()
            
        self.bindings_frame = ctk.CTkFrame(self)
        self.bindings_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        bindings_label = ctk.CTkLabel(self.bindings_frame, text="Bindings:")
        bindings_label.pack(anchor="w", padx=10, pady=5)
        
        bindings = self.blendshape_processor.bindings
        
        self.blendshape_vars = {}
        self.action_vars = {}
        self.blendshape_progress_bars = {}
        self.slider_frames = {}
        
        self.blendshape_bars = {}
        self.blendshape_value_labels = {}

        if not bindings:
            no_bindings = ctk.CTkLabel(self.bindings_frame, text="No bindings defined", text_color="gray")
            no_bindings.pack(pady=10)
        else:
            bindings_scroll = ctk.CTkScrollableFrame(self.bindings_frame, height=120)
            bindings_scroll.pack(fill="both", expand=True, padx=10, pady=5)
            
            for i, binding in enumerate(bindings):
                binding_container = ctk.CTkFrame(bindings_scroll, fg_color=("gray90", "#2B2B2B"))
                binding_container.pack(fill="x", padx=2, pady=5, anchor="w")
                
                row1 = ctk.CTkFrame(binding_container, fg_color="transparent")
                row1.pack(fill="x", padx=2, pady=(5, 0))
                row1.columnconfigure(0, weight=1) 
                row1.columnconfigure(1, weight=1) 
                
                blendshape = binding.get("blendshape", "unknown")
                blendshape_var = tk.StringVar(value=blendshape)
                self.blendshape_vars[i] = blendshape_var
                
                blendshape_dropdown = ctk.CTkOptionMenu(
                    row1,
                    values=self.available_blendshapes,
                    variable=blendshape_var,
                    command=lambda v, idx=i, old_bs=blendshape: self._update_blendshape(idx, old_bs, v),
                    width=150,
                )
                blendshape_dropdown.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
                
                action_var = binding.get("action", "unknown")
                self.action_vars[i] = action_var
                
                action_dropdown = SubmenuDropdown(
                    row1,
                    self.available_actions,
                    action_var,
                    callback=lambda v, idx=i, bs=blendshape: self._update_action(idx, bs, v)
                )
                # action_dropdown = ctk.CTkOptionMenu(
                #     row1,
                #     values=self.available_actions,
                #     variable=action_var,
                #     command=lambda v, idx=i, bs=blendshape: self._update_action(idx, bs, v),
                #     width=150,
                # )
                action_dropdown.grid(row=0, column=3, padx=5, pady=2, sticky="ew")
                
                status_row = ctk.CTkFrame(binding_container, fg_color="transparent")
                status_row.pack(fill="x", padx=2, pady=(0, 0))
                status_row.columnconfigure(1, weight=1)
                
                status_label = ctk.CTkLabel(status_row, text="Current:", anchor="w", width=70)
                status_label.grid(row=0, column=0, padx=5, pady=2, sticky="w")
                
                progress_bar = ctk.CTkProgressBar(status_row, height=15)
                progress_bar.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
                progress_bar.set(0.0) 
                self.blendshape_bars[blendshape] = progress_bar
                
                value_label = ctk.CTkLabel(status_row, text="0.00", width=40)
                value_label.grid(row=0, column=2, padx=5, pady=2, sticky="w")
                self.blendshape_value_labels[blendshape] = value_label
                
                row2 = ctk.CTkFrame(binding_container, fg_color="transparent")
                row2.pack(fill="x", padx=2, pady=(0, 5))
                row2.columnconfigure(1, weight=1) 
                
                threshold = binding.get("threshold", 0.5)
                threshold_var = ctk.DoubleVar(value=threshold)
                self.threshold_vars[blendshape] = threshold_var
                
                threshold_label = ctk.CTkLabel(row2, text="Threshold:", anchor="w", width=70)
                threshold_label.grid(row=0, column=0, padx=5, pady=2, sticky="w")
                
                threshold_slider = ctk.CTkSlider(
                    row2, 
                    from_=0.01, 
                    to=1.0, 
                    variable=threshold_var, number_of_steps=100,
                    command=lambda v, bs=blendshape: self._update_blendshape_threshold(bs, v)
                )
                threshold_slider.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
                
                threshold_value = ctk.CTkLabel(row2, text=f"{threshold:.2f}", width=40)
                threshold_slider.configure(
                    command=lambda v, lbl=threshold_value, bs=blendshape: self._update_threshold_label(v, lbl, bs)
                )
                threshold_value.grid(row=0, column=2, padx=5, pady=2, sticky="w")
                
                mode_btn = ctk.CTkButton(
                    binding_container,
                    text=binding.get("mode", "hold").upper(),
                    command=lambda bs=blendshape: self._toggle_mode(bs),
                    width=60,
                    height=24
                )
                mode_btn.pack(side="left", padx=5, pady=2)

                delete_btn = ctk.CTkButton(
                    binding_container, 
                    text="Del", 
                    command=lambda bs=blendshape: self._delete_binding(bs),
                    width=50, 
                    height=24,
                    fg_color="#FF5555",
                    hover_color="#DD4444"
                )
                delete_btn.pack(side="right", padx=5, pady=2)

    def _update_blendshape(self, index, old_blendshape, new_blendshape):
        for binding in self.blendshape_processor.bindings:
            if binding.get("blendshape") == old_blendshape:
                action = binding.get("action")
                threshold = binding.get("threshold", 0.5)
                
                self.blendshape_processor.remove_binding(old_blendshape)
                
                self.blendshape_processor.add_binding(new_blendshape, action, threshold)
                
                self._load_bindings()
                break

    def _update_action(self, index, blendshape, new_action):
        for binding in self.blendshape_processor.bindings:
            if binding.get("blendshape") == blendshape:
                binding["action"] = new_action
                self.blendshape_processor.save_to_profile()
                break

    def _update_threshold_label(self, value, label, blendshape):
        value = float(value)
        label.configure(text=f"{value:.2f}")
        self._update_blendshape_threshold(blendshape, value)

    def _delete_binding(self, blendshape):
        self.blendshape_processor.remove_binding(blendshape)
        self._load_bindings()

    def _update_blendshape_threshold(self, blendshape, value):
        value = float(value)
        
        for binding in self.blendshape_processor.bindings:
            if binding.get("blendshape") == blendshape:
                binding["threshold"] = value
                break
        
        self.blendshape_processor.save_to_profile()

    def _toggle_mode(self, blendshape):
        for binding in self.blendshape_processor.bindings:
            if binding.get("blendshape") == blendshape:
                current_mode = binding.get("mode", "hold")
                new_mode = "press" if current_mode == "hold" else "hold"
                binding["mode"] = new_mode
                self.blendshape_processor.save_to_profile()
                print(f"Mode toggled: {blendshape} -> {new_mode}")
                self._load_bindings() 
                break

    def _add_binding(self):
        self._show_binding_dialog()
    
    def _edit_binding(self, blendshape_name):
        binding = None
        for b in self.blendshape_processor.bindings:
            if b["blendshape"] == blendshape_name:
                binding = b
                break
                
        if binding:
            self._show_binding_dialog(binding)
    
    def _show_binding_dialog(self, binding=None):
        dialog = ctk.CTkToplevel(self)
        dialog.title("Blendshape Binding")
        dialog.geometry("380x260")
        dialog.grab_set()
        
        frame = ctk.CTkFrame(dialog)
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        frame.columnconfigure(1, weight=1)
        
        # Blendshape selection
        blendshape_label = ctk.CTkLabel(frame, text="Blendshape:")
        blendshape_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        blendshape_var = tk.StringVar(value=binding["blendshape"] if binding else "")
        blendshape_dropdown = ctk.CTkOptionMenu(
            frame, values=self.available_blendshapes, variable=blendshape_var
        )
        blendshape_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        # Action selection
        action_label = ctk.CTkLabel(frame, text="Action:")
        action_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        
        actions_dict = self.blendshape_processor.get_available_actions()
        action_var = ""
        action_dropdown = SubmenuDropdown(
            frame,
            actions_dict,
            action_var
        )
        action_dropdown.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        
        #Current value
        status_label = ctk.CTkLabel(frame, text="Current Value:")
        status_label.grid(row=2, column=0, padx=5, pady=5, sticky="w")
        
        # status_frame = ctk.CTkFrame(frame, fg_color="transparent")
        # status_frame.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        # status_frame.columnconfigure(0, weight=1)
        
        dialog.status_progress_bar = ctk.CTkProgressBar(frame, height=15)
        dialog.status_progress_bar.grid(row=2, column=1, padx=0, pady=0, sticky="ew")
        dialog.status_progress_bar.set(0.0)
        
        dialog.status_value_label = ctk.CTkLabel(frame, text="0.00", width=50)
        dialog.status_value_label.grid(row=2, column=2, padx=0, pady=0, sticky="w")

        # Threshold
        threshold_label = ctk.CTkLabel(frame, text="Threshold:", anchor="w", width=50)
        threshold_label.grid(row=3, column=0, padx=0, pady=5, sticky="w")
        
        threshold_var = ctk.DoubleVar(value=binding["threshold"] if binding else 0.5)
        threshold_slider = ctk.CTkSlider(
            frame, from_=0.01, to=1.0, number_of_steps=100, variable=threshold_var
        )
        threshold_slider.grid(row=3, column=1, padx=5, pady=5, sticky="ew")
        
        threshold_value = ctk.CTkLabel(frame, text=f"{threshold_var.get():.2f}", width=40)
        threshold_slider.configure(command=lambda v: threshold_value.configure(text=f"{float(v):.2f}"))
        threshold_value.grid(row=3, column=2, padx=5, pady=5, sticky="e")
        
        mode_label = ctk.CTkLabel(frame, text="Mode:")
        mode_label.grid(row=4, column=0, padx=5, pady=5, sticky="w")

        mode_var = tk.StringVar(value=binding.get("mode", "hold") if binding else "hold")
        mode_dropdown = ctk.CTkOptionMenu(
            frame, 
            values=["hold", "press"], 
            variable=mode_var,
            width=100
        )
        mode_dropdown.grid(row=4, column=1, padx=5, pady=5, sticky="w")

        # Buttons
        btn_frame = ctk.CTkFrame(frame, fg_color="transparent")
        btn_frame.grid(row=5, column=0, columnspan=3, padx=5, pady=10, sticky="ew")
        
        def update_dialog_status():
            try:
                current_blendshape = blendshape_var.get()
                if current_blendshape and hasattr(dialog, 'status_progress_bar'):
                    current_value = self.blendshape_processor.get_blendshape_value(current_blendshape)
                    dialog.status_progress_bar.set(current_value)
                    dialog.status_value_label.configure(text=f"{current_value:.2f}")
                
                if dialog.winfo_exists():
                    dialog.after(33, update_dialog_status)
            except:
                pass  
        
        def _update_dialog_blendshape_status(selected_blendshape):
            dialog.status_progress_bar.set(0.0)
            dialog.status_value_label.configure(text="0.00")
        
        blendshape_dropdown.configure(command=_update_dialog_blendshape_status)
        
        if binding:
            current_value = self.blendshape_processor.get_blendshape_value(binding["blendshape"])
            dialog.status_progress_bar.set(current_value)
            dialog.status_value_label.configure(text=f"{current_value:.2f}")
        
        update_dialog_status()

        def save_binding():
            blendshape = blendshape_var.get()
            action = action_dropdown.get_selected_action()
            threshold = threshold_var.get()
            mode = mode_var.get() 
            
            if not blendshape or not action:
                return
                
            if binding:
                self.blendshape_processor.remove_binding(binding["blendshape"])
                
            self.blendshape_processor.add_binding(blendshape, action, threshold)
            
            # Reload UI
            self._load_bindings()
            
            dialog.destroy()
            
        save_btn = ctk.CTkButton(
            btn_frame, text="Save", command=save_binding, width=80
        )
        save_btn.pack(side="left", padx=10)
        
        if binding:
            def delete_binding():
                self.blendshape_processor.remove_binding(binding["blendshape"])
                self._load_bindings()
                dialog.destroy()
                
            delete_btn = ctk.CTkButton(
                btn_frame, text="Delete", command=delete_binding, 
                width=80, fg_color="#FF5555", hover_color="#DD4444"
            )
            delete_btn.pack(side="left", padx=10)
            
        cancel_btn = ctk.CTkButton(
            btn_frame, text="Cancel", command=dialog.destroy, width=80
        )
        cancel_btn.pack(side="left", padx=10)
    
    def update_from_profile(self):
        self.blendshape_processor.load_from_profile()
        self.current_settings = self.profile_manager.get_profile_settings()
        self._load_bindings()
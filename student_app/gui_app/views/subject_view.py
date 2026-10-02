import customtkinter as ctk

class SubjectView(ctk.CTkFrame):
    def __init__(self, master, subjects, on_select, on_back, display_names=None, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.on_select = on_select
        self.on_back = on_back
        self.subjects = subjects
        # 英文数据键 -> 中文显示名（可选）。按钮显示中文，但回调仍传英文数据键。
        self.display_names = display_names or {}

        self.header_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.header_frame.pack(pady=(40, 30))
        
        self.label = ctk.CTkLabel(
            self.header_frame, 
            text="选择你的学科", 
            font=ctk.CTkFont(family="Segoe UI", size=28, weight="bold")
        )
        self.label.pack()
        
        ctk.CTkLabel(
            self.header_frame,
            text="选择一个学科开始学习",
            font=ctk.CTkFont(family="Segoe UI", size=14),
            text_color="gray70"
        ).pack(pady=(5, 0))

        self.scrollable_frame = ctk.CTkScrollableFrame(self, width=650, height=400, fg_color="transparent")
        self.scrollable_frame.pack(pady=(0, 20), padx=20, fill='both', expand=True)
        self.scrollable_frame.grid_columnconfigure(0, weight=1)
        self.scrollable_frame.grid_columnconfigure(1, weight=1)

        self.subject_buttons = []
        for i, subj in enumerate(subjects):
            btn = ctk.CTkButton(
                self.scrollable_frame, 
                text=self.display_names.get(subj, subj), 
                font=ctk.CTkFont(family="Segoe UI", size=16, weight="bold"),
                height=60,
                corner_radius=12,
                fg_color="#3949AB", # Indigo 600
                hover_color="#303F9F", # Indigo 700
                command=lambda s=subj: self.on_select(s)
            )
            # 2 columns
            btn.grid(row=i//2, column=i%2, padx=10, pady=10, sticky="ew")
            self.subject_buttons.append(btn)

        self.back_btn = ctk.CTkButton(
            self, 
            text="返回主菜单", 
            command=self.on_back, 
            fg_color="transparent", 
            border_width=1, 
            border_color="#bdbdbd",
            text_color=("gray10", "#bdbdbd"),
            hover_color=("gray90", "gray20")
        )
        self.back_btn.pack(pady=(10, 30)) 
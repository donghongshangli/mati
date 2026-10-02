import customtkinter as ctk

class ProgressOpsView(ctk.CTkFrame):
    def __init__(self, master, on_export, on_import, on_reset, on_back, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.on_export = on_export
        self.on_import = on_import
        self.on_reset = on_reset
        self.on_back = on_back

        self.COLOR_PRIMARY = "#1A237E"    # Deep Indigo
        self.COLOR_ACCENT = "#283593"     # Lighter Indigo
        self.COLOR_SUCCESS = "#2E7D32"    # Emerald Green
        self.COLOR_WARNING = "#C62828"    # Deep Red
        self.COLOR_NEUTRAL = "#424242"    # Dark Grey
        self.COLOR_BG_CARD = "#FFFFFF"    # White
        self.COLOR_TEXT_LIGHT = "#757575" # Light Grey Text

        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.pack(fill='x', padx=40, pady=(30, 20))
        
        ctk.CTkLabel(
            header_frame, 
            text="数据管理 💾", 
            font=ctk.CTkFont(family="Segoe UI", size=28, weight="bold"),
            text_color=self.COLOR_PRIMARY
        ).pack(anchor='w')
        
        ctk.CTkLabel(
            header_frame, 
            text="备份你的学习进度，或重新开始。", 
            font=ctk.CTkFont(family="Segoe UI", size=14),
            text_color=self.COLOR_TEXT_LIGHT
        ).pack(anchor='w', pady=(2, 0))

        self.content_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.content_frame.pack(fill='both', expand=True, padx=40, pady=20)

        self._create_action_card(
            self.content_frame,
            icon="📤",
            title="备份学习进度",
            desc="将你的统计与掌握情况保存为 JSON 文件。",
            btn_text="导出数据",
            btn_color=self.COLOR_PRIMARY,
            command=self.on_export
        )

        self._create_action_card(
            self.content_frame,
            icon="📥",
            title="恢复学习进度",
            desc="从之前的备份中加载你的统计数据。",
            btn_text="导入数据",
            btn_color=self.COLOR_ACCENT,
            command=self.on_import
        )

        self._create_action_card(
            self.content_frame,
            icon="🗑️",
            title="清除所有数据",
            desc="永久删除所有学习进度并重新开始。",
            btn_text="重置学习进度",
            btn_color=self.COLOR_WARNING,
            command=self.on_reset,
            hover_color="#B71C1C"
        )

        footer_frame = ctk.CTkFrame(self, fg_color="transparent")
        footer_frame.pack(fill='x', padx=40, pady=30, side='bottom')
        
        self.back_btn = ctk.CTkButton(
            footer_frame, 
            text="← 返回主菜单", 
            command=self.on_back,
            font=ctk.CTkFont(family="Segoe UI", size=14),
            height=45,
            fg_color="transparent",
            border_width=1,
            border_color="#BDBDBD",
            text_color="#616161",
            hover_color="#F5F5F5"
        )
        self.back_btn.pack(side='left')

    def _create_action_card(self, parent, icon, title, desc, btn_text, btn_color, command, hover_color=None):
        card = ctk.CTkFrame(parent, fg_color=self.COLOR_BG_CARD, corner_radius=15, border_width=1, border_color="#E0E0E0")
        card.pack(fill='x', pady=10)
        
        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(fill='both', expand=True, padx=20, pady=20)
        
        ctk.CTkLabel(inner, text=icon, font=ctk.CTkFont(size=32)).pack(side='left', padx=(0, 20))
        
        text_frame = ctk.CTkFrame(inner, fg_color="transparent")
        text_frame.pack(side='left', fill='both', expand=True)
        
        ctk.CTkLabel(
            text_frame, 
            text=title, 
            font=ctk.CTkFont(family="Segoe UI", size=18, weight="bold"),
            text_color="#212121"
        ).pack(anchor='w')
        
        ctk.CTkLabel(
            text_frame, 
            text=desc, 
            font=ctk.CTkFont(family="Segoe UI", size=13),
            text_color=self.COLOR_TEXT_LIGHT
        ).pack(anchor='w')
        
        ctk.CTkButton(
            inner,
            text=btn_text,
            command=command,
            fg_color=btn_color,
            hover_color=hover_color if hover_color else btn_color,
            font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"),
            height=40,
            width=150
        ).pack(side='right') 
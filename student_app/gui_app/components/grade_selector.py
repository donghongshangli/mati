import customtkinter as ctk

class GradeSelector(ctk.CTkOptionMenu):
    def __init__(self, master, command=None, **kwargs):
        values = ["初一", "初二", "初三", "高一", "高二", "高三"]
        super().__init__(master, values=values, command=command, **kwargs)
        self.set("高一") # 默认

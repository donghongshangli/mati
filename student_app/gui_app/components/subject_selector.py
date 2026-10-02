import customtkinter as ctk


class SubjectSelector(ctk.CTkOptionMenu):
    """
    顶栏学科选择器。

    「科学」是**家族**学科（检索物理 + 化学 + 生物）；
    物理 / 化学 / 生物 是**细分学科**，用于把检索收窄到单科，
    避免问生物时召回物理内容（集合已按 discipline 拆分）。
    """

    VALUES = ["科学", "物理", "化学", "生物", "数学", "语文",
              "英语语法", "计算机科学", "社会科学"]

    # 细分学科取值（用于推导 current_discipline_filter）
    DISCIPLINES = ("物理", "化学", "生物")

    def __init__(self, master, command=None, **kwargs):
        super().__init__(master, values=list(self.VALUES), command=command, **kwargs)
        self.set("科学")  # 默认

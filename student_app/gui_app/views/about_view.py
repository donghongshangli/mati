import customtkinter as ctk

class AboutView(ctk.CTkFrame):
    def __init__(self, master, on_back, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.on_back = on_back

        self.label = ctk.CTkLabel(self, text="关于 Mati", font=ctk.CTkFont(size=22, weight="bold"))
        self.label.pack(pady=(30, 20))

        about_text = """
        欢迎使用 Mati，您的个人 AI 学习助手！

        Mati 旨在以有趣、互动的方式帮助您掌握新概念并追踪学习进度。无论您是在探索新的学科，还是在备战考试，Mati 都会在您的学习之旅中一路相伴。

        主要功能：
        - 浏览学科：探索丰富多样的学科与主题，每个主题都配有详细的概念和学习资料。
        - 互动问答：通过互动题目检验知识掌握情况，并获得即时反馈。
        - 咨询 AI：有疑问？向我们的 AI 助手提问，即可获得基于先进语言模型的详细解答。
        - 学习进度追踪：监控您的学习进度，识别优势与薄弱环节，并获取下一步学习建议。
        - 离线优先：Mati 支持离线使用，让您随时随地都能学习。

        祝您使用 Mati 学习愉快！
        """

        self.textbox = ctk.CTkTextbox(self, width=600, height=300, font=ctk.CTkFont(size=16), wrap='word')
        self.textbox.insert('1.0', about_text)
        self.textbox.configure(state="disabled")
        self.textbox.pack(pady=10, padx=20, fill="both", expand=True)

        self.back_btn = ctk.CTkButton(self, text="返回", command=self.on_back)
        self.back_btn.pack(pady=20)

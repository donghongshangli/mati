import customtkinter as ctk
import threading

class AskQuestionView(ctk.CTkScrollableFrame):
    def __init__(self, master, on_submit, on_back, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.on_submit = on_submit
        self.on_back = on_back
        self.spinner_animation_running = False

        self.label = ctk.CTkLabel(self, text="提问", font=ctk.CTkFont(size=22, weight="bold"))
        self.label.pack(pady=(30, 5))

        self.guide_frame = ctk.CTkFrame(self, fg_color="#E8EAF6", corner_radius=12)
        self.guide_frame.pack(pady=(0, 25), padx=40, fill='x')
        
        ctk.CTkLabel(
            self.guide_frame, 
            text="提问小贴士", 
            font=ctk.CTkFont(size=15, weight="bold"), 
            text_color="#3949AB"
        ).pack(pady=(12, 5))
        
        ctk.CTkLabel(
            self.guide_frame, 
            text="• 直接输入问题，AI 会结合知识库为你解答\n• 尽量具体，并使用课本术语\n• 支持中文与英文提问，解答以中文呈现", 
            font=ctk.CTkFont(size=13), 
            text_color="#5C6BC0", # Indigo-400
            justify="center"
        ).pack(pady=(0, 12))

        self.entry = ctk.CTkEntry(self, placeholder_text="在此输入你的问题...", width=400, font=ctk.CTkFont(size=16))
        self.entry.pack(pady=10)
        self.entry.bind('<Return>', lambda e: self.submit())

        self.submit_btn = ctk.CTkButton(self, text="提问", command=self.submit)
        self.submit_btn.pack(pady=10)

        self.result_frame = ctk.CTkFrame(self)
        self.result_frame.pack(pady=20, fill='x', expand=True)

        self.spinner = None
        self.answer_box = None
        self.answer_frame = None
        self.streaming_answer = ""
        self.status_label = None  # 独立运行状态标签（方案A：状态不流入答案框）

        self.back_btn = ctk.CTkButton(self, text="返回", command=self.on_back, fg_color="#bdbdbd", hover_color="#757575")
        self.back_btn.pack(pady=(30, 0))

    def submit(self):
        question = self.entry.get().strip()
        if question:
            self.set_loading(True)    
            self.on_submit(question, "medium")
            
    def set_loading(self, loading):
        if loading:
            self.entry.configure(state="disabled")
            self.submit_btn.configure(state="disabled")
            for widget in self.result_frame.winfo_children():
                widget.destroy()
            self.answer_box = None
            self.answer_frame = None
            self.streaming_answer = ""
            self.status_label = None
            self.spinner = ctk.CTkLabel(self.result_frame, text="", font=ctk.CTkFont(size=16))
            self.spinner.pack(pady=20)
            self.spinner_animation_running = True
            self._animate_spinner(0)
        else:
            self.entry.configure(state="normal")
            self.submit_btn.configure(state="normal")
            if self.spinner:
                self.spinner.destroy()
                self.spinner = None
            self.spinner_animation_running = False

    def set_status(self, msg):
        """在独立状态标签显示运行状态（如“正在搜索…”“Qwen 正在生成…”），不流入答案框。"""
        if self.status_label is None or not self.status_label.winfo_exists():
            self.status_label = ctk.CTkLabel(
                self.result_frame,
                text=msg,
                font=ctk.CTkFont(size=12, slant="italic"),
                text_color="#78909c",
                wraplength=580,
                justify="left"
            )
            self.status_label.pack(pady=(0, 8), padx=10, anchor='w')
        else:
            self.status_label.configure(text=msg)
        self.update_idletasks()

    def _clear_status(self):
        """清除运行状态标签。"""
        if self.status_label is not None:
            try:
                if self.status_label.winfo_exists():
                    self.status_label.destroy()
            except Exception:
                pass
            self.status_label = None

    def _animate_spinner(self, index):
        if self.spinner_animation_running and self.spinner and self.spinner.winfo_exists():
            chars = ["|", "/", "-", "\\"]
            self.spinner.configure(text=f"思考中... {chars[index]}")
            next_index = (index + 1) % len(chars)
            self.after(100, self._animate_spinner, next_index)

    def _calculate_fluid_height(self, text):
        """Calculate height needed for text content."""
        if not text:
            return 80
        
        chars_per_line = 70
        total_lines = 0
        
        for paragraph in text.split('\n'):
            if not paragraph.strip():
                total_lines += 1
            else:
                total_lines += max(1, (len(paragraph) + chars_per_line - 1) // chars_per_line)
        
        line_height = 22
        padding = 40
        calculated_height = (total_lines * line_height) + padding
        
        return max(100, min(calculated_height, 600))
    
    def append_answer_token(self, token):
        """Append a token to the streaming answer display."""
        if self.answer_box is None:
            self.streaming_answer = ""
            for widget in self.result_frame.winfo_children():
                widget.destroy()
            
            self.answer_frame = ctk.CTkFrame(self.result_frame, fg_color="#e3f2fd", corner_radius=8)
            self.answer_frame.pack(pady=(0, 10), padx=10, fill='x', expand=False)
            
            starting_height = 120
            self.answer_box = ctk.CTkTextbox(
                self.answer_frame, 
                width=600, 
                height=starting_height, 
                font=ctk.CTkFont(size=14), 
                wrap='word'
            )
            self.answer_box.pack(pady=(10, 10), padx=10, fill='x', expand=False)
        
        self.streaming_answer += token
        self.answer_box.insert('end', token)
        self.answer_box.see('end')
        
        if len(self.streaming_answer) % 100 == 0:
            new_height = self._calculate_fluid_height(self.streaming_answer)
            if new_height > self.answer_box.cget("height"):
                self.answer_box.configure(height=new_height)
                
        self.update_idletasks()
    
    def _sanitize_streamed_answer(self):
        """
        把已流式写入答案框的文本整体规整一次。

        为什么不在写入时逐 token 规整：流式 token 是任意切分的，
        一个 LaTeX 命令（如 `\\frac`）可能被拆成几个 token，
        逐段替换会误判。等生成结束后整体替换是最可靠的做法。

        规整内容见 system/rag/text_cleanup.py：
        · 修 PDF 字形错映射（狓/狔 → x/y 等）
        · 把 LaTeX 标记降级为纯文本数学（答案框是纯文本控件，渲染不了 LaTeX）
        """
        shown = self.streaming_answer or ""
        if not shown:
            return
        try:
            from system.rag.text_cleanup import clean_text
            cleaned = clean_text(shown)
        except Exception:  # noqa: BLE001 - 规整失败不能影响正常显示
            return
        if cleaned == shown:
            return
        self.streaming_answer = cleaned
        if self.answer_box is not None:
            try:
                self.answer_box.delete("1.0", "end")
                self.answer_box.insert("1.0", cleaned)
            except Exception:  # noqa: BLE001
                pass

    def finalize_answer(self, confidence, hints=None, related=None, source_info=None, question=None, grade=None, subject=None):
        self.set_loading(False)

        # 先规整答案再做后续渲染：图表生成线程也会读 self.streaming_answer，
        # 让它拿到的是干净文本。
        self._sanitize_streamed_answer()

        self._current_grade = grade
        self._current_subject = subject
        
        if question and self.streaming_answer:
            try:
                self.analyzing_label = ctk.CTkLabel(
                    self.answer_frame if self.answer_frame else self.result_frame,
                    text="⚡ 正在分析并生成可视化图表...",
                    font=ctk.CTkFont(size=12, slant="italic"),
                    text_color="#78909c"
                )
                self.analyzing_label.pack(pady=(5, 10), padx=10, anchor='w')
                
                
                threading.Thread(
                    target=self._generate_diagram_background,
                    args=(question, self.streaming_answer, grade, subject),
                    daemon=True
                ).start()
                
            except Exception:
                pass

        if self.answer_box is None:
            self._clear_status()
            self._render_complete_answer(
                self.streaming_answer, 
                confidence, 
                hints, 
                related, 
                source_info
            )
            return
        
        self.answer_box.update_idletasks()
        line_count = int(self.answer_box.index('end-1c').split('.')[0])
        
        line_height = 22
        padding = 40
        final_height = max(100, min((line_count * line_height) + padding, 600))
        
        self.answer_box.configure(height=final_height, state="disabled")
        self.answer_frame.update_idletasks()
        
        if source_info and self.answer_frame:
            source_exists = any(
                isinstance(w, ctk.CTkLabel) and "来源：" in w.cget("text")
                for w in self.answer_frame.winfo_children()
            )
            
            if not source_exists:
                source_label = ctk.CTkLabel(
                    self.answer_frame, 
                    text=f"来源：{source_info}", 
                    font=ctk.CTkFont(size=12), 
                    text_color="#1976d2"
                )
                source_label.pack(pady=(5, 0), padx=10, anchor='w', before=self.answer_box)
        
        self._add_metadata_sections(confidence, hints, related)

    def _generate_diagram_background(self, question, answer, grade, subject):
        """Runs diagram generation in a separate thread."""
        try:
            from system.diagrams import generate_diagram_content
            result = generate_diagram_content(
                question, 
                answer,
                grade=grade,
                subject=subject
            )
            self.after(0, self._on_diagram_generated, result)
        except Exception:
            self.after(0, self._on_diagram_generated, None)

    def _on_diagram_generated(self, result):
        """Callback to update UI with generated diagram."""
        if hasattr(self, 'analyzing_label') and self.analyzing_label:
            self.analyzing_label.destroy()
            
        if result:
            formatted_diagram, diagram_type = result
            
            type_name = diagram_type.value if hasattr(diagram_type, 'value') else str(diagram_type)
            type_name_lower = type_name.lower()
            
            from student_app.gui_app.components.diagram_viewer import get_diagram_style, DiagramViewer
            
            style = get_diagram_style(type_name_lower)
            
            self.diagram_frame = ctk.CTkFrame(
                self.result_frame, 
                fg_color=style['bg_color'], 
                corner_radius=8
            )
            self.diagram_frame.pack(pady=(0, 10), padx=10, fill='x', expand=False)
            
            ctk.CTkLabel(
                self.diagram_frame, 
                text=f"{style['icon']} {style['label']}", 
                font=ctk.CTkFont(size=14, weight="bold"), 
                text_color=style['header_color']
            ).pack(pady=(10, 5), padx=10, anchor='w')
            
            line_count = formatted_diagram.count('\n') + 1
            line_height = 18
            padding = 40
            calculated_height = max(150, min((line_count * line_height) + padding, 450))
            
            diagram_box = ctk.CTkTextbox(
                self.diagram_frame, 
                width=600, 
                height=calculated_height, 
                font=ctk.CTkFont(family="Consolas", size=12), 
                wrap='none',  # Preserve ASCII alignment
                fg_color='#ffffff'  # White background
            )
            diagram_box.insert("1.0", formatted_diagram)
            diagram_box.configure(state="disabled")
            diagram_box.pack(pady=(0, 10), padx=10, fill='x', expand=False)
    
    def set_result(self, answer, confidence=None, hints=None, related=None, source_info=None):
        """Complete answer result (non-streaming fallback)."""
        self.set_loading(False)
        self.streaming_answer = answer
        for widget in self.result_frame.winfo_children():
            widget.destroy()
        self.status_label = None
        self._render_complete_answer(answer, confidence, hints, related, source_info)
    
    def _render_complete_answer(self, answer, confidence, hints, related, source_info):
        """Render complete answer with all metadata."""
        height = self._calculate_fluid_height(answer)
        
        if source_info:
            source_frame = ctk.CTkFrame(self.result_frame, fg_color="#e3f2fd", corner_radius=8)
            source_frame.pack(pady=(0, 10), padx=10, fill='x')
            ctk.CTkLabel(
                source_frame, 
                text=f"来源：{source_info}", 
                font=ctk.CTkFont(size=12), 
                text_color="#1976d2"
            ).pack(pady=5, padx=10, anchor='w')
        
        if confidence is None or confidence < 0.3:
            warn_frame = ctk.CTkFrame(self.result_frame, fg_color="#fffde7", corner_radius=8)
            warn_frame.pack(pady=(0, 10), padx=10, fill='x', expand=True)
            
            ctk.CTkLabel(
                warn_frame, 
                text="我不太确定这个答案，让我帮你找到正确的信息：", 
                font=ctk.CTkFont(size=15, weight="bold"), 
                text_color="#fbc02d"
            ).pack(pady=(10, 0), padx=10, anchor='w')
            
            answer_box = ctk.CTkTextbox(warn_frame, width=600, height=height, font=ctk.CTkFont(size=14), wrap='word')
            answer_box.insert('1.0', answer)
            answer_box.configure(state="disabled")
            answer_box.pack(pady=(5, 10), padx=10, fill='x', expand=True)
        else:
            self.answer_box = ctk.CTkTextbox(
                self.result_frame, 
                width=600, 
                height=height, 
                font=ctk.CTkFont(size=14), 
                wrap='word'
            )
            self.answer_box.insert('1.0', answer)
            
            self.answer_box.configure(state="disabled")
            self.answer_box.pack(pady=(0, 10), padx=10, fill='x', expand=True)
        
        self._add_metadata_sections(confidence, hints, related)
    
    def _add_metadata_sections(self, confidence, hints, related):
        """Add confidence indicator, hints, and related concepts sections."""
        if confidence and confidence < 0.7:
            if confidence > 0.4:
                conf_color = "#fff9c4"
                conf_text_color = "#f57c00"
                conf_text = "⚠️ 中等置信度"
            else:
                conf_color = "#ffebee"
                conf_text_color = "#c62828"
                conf_text = "⚠️ 低置信度 - 请核实"
            
            conf_frame = ctk.CTkFrame(self.result_frame, fg_color=conf_color, corner_radius=6)
            conf_frame.pack(pady=(0, 5), padx=10, anchor='e')
            ctk.CTkLabel(
                conf_frame, 
                text=f"{conf_text} ({confidence*100:.0f}%)", 
                font=ctk.CTkFont(size=11), 
                text_color=conf_text_color
            ).pack(pady=3, padx=8)
        
        if hints:
            hints_frame = ctk.CTkFrame(self.result_frame, fg_color="#f3e5f5", corner_radius=8)
            hints_frame.pack(pady=(0, 10), padx=10, fill='x')
            ctk.CTkLabel(
                hints_frame, 
                text="💡 提示：", 
                font=ctk.CTkFont(size=15, weight="bold"), 
                text_color="#7b1fa2"
            ).pack(pady=(10, 5), padx=10, anchor='w')
            
            for hint in hints:
                ctk.CTkLabel(
                    hints_frame, 
                    text=f"• {hint}", 
                    font=ctk.CTkFont(size=14), 
                    wraplength=580, 
                    justify='left'
                ).pack(anchor='w', padx=20, pady=2)
        

        if related:
            related_frame = ctk.CTkFrame(self.result_frame, fg_color="#e8f5e8", corner_radius=8)
            related_frame.pack(pady=(0, 10), padx=10, fill='x')
            ctk.CTkLabel(
                related_frame, 
                text="🔗 相关概念：", 
                font=ctk.CTkFont(size=15, weight="bold"), 
                text_color="#388e3c"
            ).pack(pady=(10, 5), padx=10, anchor='w')
            
            for rel in related:
                ctk.CTkLabel(
                    related_frame, 
                    text=f"• {rel}", 
                    font=ctk.CTkFont(size=14)
                ).pack(anchor='w', padx=20, pady=2)
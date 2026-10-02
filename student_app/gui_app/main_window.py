import customtkinter as ctk
from student_app.gui_app.views import WelcomeView, GradeView, SubjectView, TopicView, ConceptView, ConceptDetailView, QuestionView, ProgressView, AskQuestionView, ProgressOpsView, AboutView, UserGuideView, TeacherView
from student_app.gui_app.tk_safety import (
    install_shutdown_safety, stop_scaling_tracker, safe_after,
)
from system.data_manager.content_manager import ContentManager
from system.rag.rag_retrieval_engine import RAGRetrievalEngine
from student_app.progress import progress_manager
from ai_model.model_utils.model_handler import ModelHandler
from system.utils.resource_path import resolve_model_dir, resolve_content_dir, resolve_chroma_db_dir
from student_app.gui_app.components.grade_selector import GradeSelector
from student_app.gui_app.components.subject_selector import SubjectSelector
from system.security.security_utils import (
    validate_content_input,
    log_security_event,
)
from system.performance.performance_utils import timeit, log_resource_usage

import difflib
import logging
import re
import tkinter.filedialog as fd
import tkinter.messagebox as mb
import os
import ctypes
import json
from PIL import Image
import threading
import atexit
import time

# `logger` 在本模块里被多处使用（内容加载告警、模型释放等），
# 但此前从未定义 —— 一旦走到那些分支就会 NameError，且多半被外层
# try/except 吞掉，表现为"告警莫名其妙不出现"。
logger = logging.getLogger(__name__)

# ============================================================
# 本地化辅助：年级 / 学科 中文显示值 <-> 英文数据键
# （下拉组件的显示值已改为中文，此处保证数据逻辑仍使用英文键）
# ============================================================
_GRADE_CN = {7: "初一", 8: "初二", 9: "初三", 10: "高一", 11: "高二", 12: "高三"}
_GRADE_CN_TO_INT = {v: k for k, v in _GRADE_CN.items()}

_SUBJECT_EN_TO_CN = {
    "Science": "科学",
    "Math": "数学",
    "Chinese": "语文",
    "English Grammar": "英语语法",
    "Computer Science": "计算机科学",
    "Social Studies": "社会科学",
}
_SUBJECT_ALIAS_TO_CN = {
    "Mathematics": "数学",
    "English": "英语",
}
_SUBJECT_CN_TO_EN = {v: k for k, v in _SUBJECT_EN_TO_CN.items()}


def _subject_to_cn(subject):
    """把内容数据中的英文学科键转换为下拉框显示用的中文名。"""
    return _SUBJECT_EN_TO_CN.get(subject) or _SUBJECT_ALIAS_TO_CN.get(subject) or subject


def _subject_to_en(value):
    """把下拉框的中文学科显示值转换回英文数据键。"""
    return _SUBJECT_CN_TO_EN.get(value, value)


def _format_sources(sources, limit=2, citations=None):
    """
    把检索层返回的 `sources`（chunk 元数据列表）渲染成给人看的溯源字符串。

    此前检索层算好了书名/出版社/年级，但界面调用 finalize_answer 时**从不传**
    source_info，导致「来源：…」标签一次也没显示过。这里补上。

    去重规则：按「书名（或文件名）」去重，最多展示 limit 本，
    顺序即证据排序（第一条最相关）。

    Args:
        citations: 检索层生成的来源编号表（P2）。给了就**优先用它** ——
            编号与来源描述由检索层生成（不是模型编的），
            因此能给出「[1] 书名 · 第2章 · 第53页」这样可核对的溯源，
            而不只是书名。
    """
    if citations:
        labels = []
        for c in citations[:limit]:
            label = str(c.get("label") or "").strip()
            if not label:
                title = c.get("book_title") or c.get("id") or ""
                if not title:
                    continue
                bits = [str(title)]
                if c.get("chapter"):
                    bits.append(str(c["chapter"]))
                if c.get("page"):
                    bits.append(f"第{c['page']}页")
                label = " · ".join(bits)
            labels.append(f"[{c.get('index', len(labels) + 1)}] {label}")
        if labels:
            joined = "；".join(labels)
            if len(citations) > len(labels):
                joined += f" 等 {len(citations)} 条"
            return joined

    if not sources:
        return None
    seen, labels = set(), []
    for meta in sources:
        if not isinstance(meta, dict):
            continue
        title = meta.get("book_title") or meta.get("source") or meta.get("source_file")
        if not title or title in seen:
            continue
        seen.add(title)
        grade = str(meta.get("grade_label") or "")
        if grade.lower() in ("unknown", "none", "null"):
            grade = ""
        labels.append(f"{title}（{grade}）" if grade else str(title))
        if len(labels) >= limit:
            break
    if not labels:
        return None
    joined = "；".join(labels)
    if len(seen) > len(labels):
        joined += f" 等 {len(seen)} 本"
    return joined


class NEBeduApp(ctk.CTk):
    """Main application window for Mati Learning System."""
    
    def __init__(self):
        super().__init__()
        
        try:
            myappid = 'mati.learning.companion.1.0' 
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
        except Exception:
            pass
            
        self.title('Mati：智能学习助手')
        self.geometry('1200x750')
        self.minsize(900, 600)
        
        self.withdraw()
        
        ctk.set_appearance_mode('light')
        ctk.set_default_color_theme('green')
        
        self._cache = {}
        self._loading = False
        self._update_timer = None
        self._initialization_done = False
        
        # 关闭时序标志：工作线程的回调据此判断"是否还要往界面上写"
        self._closing = False
        self.username = None
        self.content_manager = None
        
        self.rag_engine = None
        self._rag_initialized = False
        
        self.model_handler = None
        self.model_path = None

        atexit.register(self.cleanup_model)
        self.selected_subject = None
        self.selected_topic = None
        self.selected_concept = None
        self.selected_concept_data = None
        self.question_index = 0
        
        self.current_grade_filter = "10"
        self.current_subject_filter = "Science"
        # 细分学科过滤器（物理/化学/生物）；None = 检索整个学科家族
        self.current_discipline_filter = None
        self.selected_grade = None

        self._set_window_icon()
        self._create_ui_structure()
        self._eager_init_all_models()

        # ── 关闭时序保护（详见 student_app/gui_app/tk_safety.py）
        # customtkinter 的 DPI 缩放轮询是个挂在窗口上的 100ms `after` 定时器，
        # 库自身在 destroy 时不会停它；销毁后那一次回调会调用 winfo_exists()
        # 并抛出 TclError，控制台刷一屏红。这里：
        #   · on_close → 先置 _closing、停轮询、释放模型，再销毁
        #   · install_shutdown_safety 补上库函数保护与兜底异常钩子
        # 关闭回调**交给 install_shutdown_safety 统一注册**：
        # Tk 的 protocol(name) 未注册时也返回真值，自己先注册再调 install
        # 会让"是否已注册"的判断失效（详见 tk_safety 的说明）。
        install_shutdown_safety(self, on_close=self.on_close)

    def _set_window_icon(self):
        def _apply_icon():
            try:
                logo_path = os.path.join(os.path.dirname(__file__), "images", "logo.png")
                
                if not os.path.exists(logo_path):
                    logo_path = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "logo.png")
                
                if os.path.exists(logo_path):
                    try:
                        ico_path = os.path.splitext(logo_path)[0] + ".ico"
                        
                        if not os.path.exists(ico_path):
                            img = Image.open(logo_path)
                            img.save(ico_path, format='ICO', sizes=[(32, 32), (64, 64), (128, 128)])
                        
                        self.iconbitmap(ico_path)
                    except Exception:
                        icon_image = Image.open(logo_path)
                        self.app_icon = ImageTk.PhotoImage(icon_image)
                        self.wm_iconphoto(True, self.app_icon)
            except Exception as e:
                print(f"Could not set window icon: {e}")
        
        self.after(200, _apply_icon)
    
    def _create_ui_structure(self):
        self.container = ctk.CTkFrame(self, fg_color="transparent")
        self.container.pack(fill='both', expand=True)

        self.sidebar = ctk.CTkFrame(
            self.container,
            width=220,
            corner_radius=0,
            fg_color="#F5F5F5"
        )
        self.sidebar.grid(row=0, column=0, sticky='ns')
        self.sidebar.grid_propagate(False)
        self.sidebar.grid_remove()

        self._create_sidebar()

        self.main_content = ctk.CTkFrame(
            self.container,
            corner_radius=0,
            fg_color="#FFFFFF"
        )
        self.main_content.grid(row=0, column=1, sticky='nsew', padx=0, pady=0)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(1, weight=1)

        self.top_bar = ctk.CTkFrame(
            self.main_content,
            height=60,
            corner_radius=0,
            fg_color="#FFFFFF",
            border_width=0,
            border_color="#E0E0E0"
        )
        self.top_bar.pack(side="top", fill="x")
        self.top_bar.pack_propagate(False)

        self.sidebar_toggle_btn = ctk.CTkButton(
            self.top_bar,
            text="☰",
            width=40,
            height=40,
            corner_radius=8,
            command=self.toggle_sidebar,
            fg_color="#E8F5E9",
            hover_color="#C8E6C9",
            text_color="#2E7D32"
        )
        self.sidebar_toggle_btn.pack(side="left", padx=15, pady=10)
        
        self.title_label = ctk.CTkLabel(
            self.top_bar,
            text="Mati 智能学习系统",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color="#2E7D32"
        )
        self.title_label.pack(side="left", padx=15)
        
        self.grade_selector = GradeSelector(self.top_bar, command=self.on_grade_change, width=120)
        self.grade_selector.pack(side="right", padx=5)

        self.subject_selector = SubjectSelector(self.top_bar, command=self.on_subject_change, width=140)
        self.subject_selector.pack(side="right", padx=5)
        
        self.current_subject_filter = _subject_to_en(self.subject_selector.get())
        self.current_discipline_filter = self._discipline_from_label(self.subject_selector.get())
        self.current_grade_filter = str(_GRADE_CN_TO_INT.get(self.grade_selector.get(), 10))

        self.sidebar_shown = False

        self.main_frame = ctk.CTkFrame(
            self.main_content,
            corner_radius=0,
            fg_color="#FAFAFA"
        )
        self.main_frame.pack(side="bottom", fill="both", expand=True, padx=0, pady=0)

        self.welcome_view = WelcomeView(self.main_frame, self.on_login)
        self.welcome_view.pack(fill='both', expand=True)

    def on_grade_change(self, value):
        try:
            self.current_grade_filter = str(_GRADE_CN_TO_INT.get(value, 10))
            self.selected_grade = int(self.current_grade_filter)
        except Exception:
            pass

    def _discipline_from_label(self, label):
        """
        从顶栏学科标签推导细分学科过滤器。

        「科学」→ None（检索物理+化学+生物全家族）
        「物理」/「化学」/「生物」→ 对应 discipline，只检索该科
        """
        try:
            if label in SubjectSelector.DISCIPLINES:
                return label
        except Exception:
            pass
        return None

    def on_subject_change(self, value):
        self.current_subject_filter = _subject_to_en(value)
        self.selected_subject = _subject_to_en(value)
        self.current_discipline_filter = self._discipline_from_label(value)

    def _create_sidebar(self):
        logo_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        logo_frame.pack(pady=(25, 20), fill="x")
        
        try:
            logo_path = os.path.join(os.path.dirname(__file__), "images", "logo.png")
            if os.path.exists(logo_path):
                logo_img = ctk.CTkImage(light_image=Image.open(logo_path), size=(100, 100))
                self.logo_image_label = ctk.CTkLabel(logo_frame, image=logo_img, text="")
                self.logo_image_label.pack()
            else:
                ctk.CTkLabel(
                    logo_frame,
                    text="🌟 Mati",
                    font=ctk.CTkFont(size=24, weight="bold"),
                    text_color="#2E7D32"
                ).pack()
        except Exception:
            ctk.CTkLabel(
                logo_frame,
                text="🌟 Mati",
                font=ctk.CTkFont(size=24, weight="bold"),
                text_color="#2E7D32"
            ).pack()

        nav_buttons = [
            ('📚 浏览学科', self.show_browse, "#2E7D32"),
            ('❓ 提问', self.show_ask, "#1976D2"),
            ('🧑‍🏫 教师工作台', self.show_teacher, "#00695C"),
            ('📊 学习进度', self.show_progress, "#7B1FA2"),
            ('⚙️ 进度管理', self.show_progress_ops, "#F57C00"),
            ('ℹ️ 关于', self.show_about, "#616161"),
            ('📘 使用指南', self.show_user_guide, "#00796B"),
        ]
        
        for text, command, color in nav_buttons:
            btn = ctk.CTkButton(
                self.sidebar,
                text=text,
                command=command,
                height=45,
                font=ctk.CTkFont(size=15, weight="normal"),
                corner_radius=10,
                fg_color=color,
                hover_color=self._darken_color(color),
                anchor="w"
            )
            btn.pack(pady=8, fill='x', padx=15)

        self.btn_exit = ctk.CTkButton(
            self.sidebar,
            text='🚪 退出',
            command=self.quit,
            height=45,
            font=ctk.CTkFont(size=15, weight="normal"),
            corner_radius=10,
            fg_color='#E53935',
            hover_color='#C62828',
            anchor="w"
        )
        self.btn_exit.pack(pady=(30, 0), fill='x', padx=15)

        self.model_info_frame = ctk.CTkFrame(
            self.sidebar,
            fg_color="#E8F5E9",
            corner_radius=10
        )
        self.model_info_frame.pack(side="bottom", pady=15, padx=15, fill="x")
        
        self.model_info_label = ctk.CTkLabel(
            self.model_info_frame,
            text="正在初始化...",
            font=ctk.CTkFont(size=11),
            text_color="#424242",
            wraplength=180
        )
        self.model_info_label.pack(pady=10, padx=10)

    def _darken_color(self, hex_color):
        if hex_color.startswith('#'):
            hex_color = hex_color[1:]
        r = max(0, int(hex_color[0:2], 16) - 30)
        g = max(0, int(hex_color[2:4], 16) - 30)
        b = max(0, int(hex_color[4:6], 16) - 30)
        return f"#{r:02x}{g:02x}{b:02x}"
    
    def _eager_init_all_models(self):
        from student_app.gui_app.startup_loader import StartupLoader
        
        loader = StartupLoader() 
        loader.update()
        loader.update_idletasks()
        
        try:
            loader.update_status("正在加载内容...", "正在初始化内容管理器", 0.1)
            loader.update_idletasks()
            self.content_manager = ContentManager(
                str(resolve_content_dir("mati_data/content"))
            )
            self._warn_content_load_issues()
            
            loader.update_status("正在定位模型...", "正在查找 Qwen2.5 模型文件", 0.2)
            loader.update_idletasks()
            self.model_path = str(resolve_model_dir("mati_data/models/qwen2_5"))
            
            loader.update_status("正在加载 Qwen2.5...", "正在加载并预热模型", 0.3)
            loader.update_idletasks()
            self.model_handler = ModelHandler(self.model_path)
            
            loader.update_status("正在加载 RAG 检索引擎...", "正在初始化向量与数据库", 0.6)
            loader.update_idletasks()
            chroma_db_path = str(resolve_model_dir("mati_data/chroma_db"))
            self.rag_engine = RAGRetrievalEngine(
                chroma_db_path=chroma_db_path,
                llm_handler=self.model_handler
            )
            
            loader.update_status("正在预热 RAG...", "正在预加载向量与数据库", 0.8)
            loader.update_idletasks()
            self.rag_engine.warm_up()  
            self._rag_initialized = True
            
            loader.update_status("就绪！", "所有系统已加载就绪", 1.0)
            loader.update_idletasks()
            time.sleep(0.5)
            
            self._update_model_info()
            self._update_status("就绪！")
            self._initialization_done = True
            log_resource_usage("初始化完成")
            
            loader.destroy()
            self.deiconify()
            self.update()
            
        except Exception as e:
            loader.destroy()
            self._show_model_error(f"初始化错误：{e}")
            self.deiconify()
            self.update()

    def _update_status(self, message):
        if hasattr(self, 'model_info_label'):
            self.model_info_label.configure(text=message)
    
    def _update_model_info(self):
        try:
            if self.model_handler:
                model_info = self.model_handler.get_model_info()
                if model_info:
                    info_text = f"🤖 {model_info.get('name', 'AI 模型')}\n"
                    info_text += f"上下文：{model_info.get('context_size', 'N/A')}"
                    self.model_info_label.configure(text=info_text)
        except Exception:
            self.model_info_label.configure(text="模型已加载")
    
    def _warn_content_load_issues(self):
        """
        把内容加载问题提示给用户（P2-6）。

        content JSON 校验失败时过去只写后台日志，用户看到的现象是
        「某个学科莫名其妙不见了」。这里在启动后弹一次提示，把出问题的
        文件名与原因讲清楚；纯 warning（已自动忽略的未知字段）只写日志，
        不打扰用户。
        """
        try:
            errors = [e for e in self.content_manager.get_load_errors()
                      if e.get("level") == "error"]
            warns = [e for e in self.content_manager.get_load_errors()
                     if e.get("level") == "warn"]
            for w in warns:
                logger.warning(f"内容文件提示：{w['file']} — {w['message']}")
            if not errors:
                return
            detail = "\n".join(f"· {e['file']}：{e['message']}" for e in errors[:5])
            if len(errors) > 5:
                detail += f"\n… 另有 {len(errors) - 5} 个文件"
            msg = ("以下内容文件未能加载，对应学科在界面上不会出现：\n\n"
                   f"{detail}\n\n请检查 JSON 是否为合法格式，以及是否包含必填字段"
                   "（subject / grade / topics）。")
            logger.error(f"内容加载失败 {len(errors)} 个文件")
            # 延迟到主窗口显示后弹出，避免卡在启动画面
            self.after(800, lambda: mb.showwarning("内容加载提示", msg))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"展示内容加载问题失败（忽略）：{e}")

    def _show_model_error(self, error_msg):
        self.model_info_label.configure(
            text=f"⚠️ 错误：{error_msg[:50]}...",
            text_color="#E53935"
        )

    def toggle_sidebar(self):
        if self.sidebar_shown:
            self.sidebar.grid_remove()
            self.sidebar_shown = False
        else:
            self.sidebar.grid()
            self.sidebar_shown = True

    def cleanup_model(self):
        """
        退出前释放大模型。

        原实现查的是 `bitnet_handler` —— 本项目早就换成 Qwen2.5 了，
        `ModelHandler` 上根本没有这个属性，于是 `hasattr` 恒为 False，
        **1GB 的模型从来没被显式释放过**，只靠进程退出时 GC 兜底。
        `ModelHandler.cleanup()` 才是现成的正确入口。
        """
        try:
            if self.model_handler is not None and hasattr(self.model_handler, 'cleanup'):
                self.model_handler.cleanup()
        except Exception as e:  # noqa: BLE001 - 退出路径上的异常不该影响退出
            logger.warning(f"模型释放失败（忽略）：{e}")

    def on_close(self):
        """
        统一的窗口关闭入口（绑定到 WM_DELETE_WINDOW）。

        顺序很重要：
          1. 置 `_closing` —— 让还在跑的工作线程停止往界面投递
          2. 停掉 customtkinter 的 DPI 轮询 —— 它是挂在窗口上的 `after`
             定时器，窗口一销毁就再也没机会摘掉（详见 tk_safety.py）
          3. 释放模型
          4. 销毁窗口
        """
        if getattr(self, "_closing", False):
            return
        self._closing = True

        # 先取消自己挂着的定时器（防抖刷新用），再停库的轮询
        try:
            if self._update_timer:
                self.after_cancel(self._update_timer)
                self._update_timer = None
        except Exception:  # noqa: BLE001
            pass

        try:
            stop_scaling_tracker()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.cleanup_model()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.destroy()
        except Exception as e:  # noqa: BLE001
            logger.debug(f"destroy 异常（忽略）：{e}")

    def _lazy_init_rag(self):
        if not self._rag_initialized:
            try:
                chroma_db_path = str(resolve_chroma_db_dir("mati_data/chroma_db"))
                # 方案B：懒加载时也传入已加载的模型处理器，避免提问退回纯检索
                self.rag_engine = RAGRetrievalEngine(
                    chroma_db_path=chroma_db_path,
                    llm_handler=self.model_handler
                )
                self._rag_initialized = True
            except Exception as e:
                self.rag_engine = None
                self._rag_initialized = True

    def _debounced_update(self, func, delay=50):
        if self._update_timer:
            self.after_cancel(self._update_timer)
        self._update_timer = self.after(delay, func)

    def _safe_destroy_widgets(self):
        try:
            for widget in self.main_frame.winfo_children():
                widget.destroy()
        except Exception:
            pass

    def on_login(self, username):
        self.username = username
        self.welcome_view.pack_forget()
        self.sidebar.grid()
        self.sidebar_shown = True
        self.show_main_menu()
    
    def show_main_menu(self):
        if self._loading: return
        self._safe_destroy_widgets()
        dashboard_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        dashboard_frame.pack(fill='both', expand=True, padx=40, pady=40)
        welcome_header = ctk.CTkFrame(dashboard_frame, fg_color="transparent")
        welcome_header.pack(fill='x', pady=(0, 30))
        welcome_title = ctk.CTkLabel(welcome_header, text=f"欢迎回来，{self.username}！👋", font=ctk.CTkFont(size=36, weight="bold"), text_color="#2E7D32")
        welcome_title.pack(anchor='w')
        welcome_subtitle = ctk.CTkLabel(welcome_header, text="您的个人学习主页", font=ctk.CTkFont(size=18), text_color="#666666")
        welcome_subtitle.pack(anchor='w', pady=(5, 0))
        features_frame = ctk.CTkFrame(dashboard_frame, fg_color="transparent")
        features_frame.pack(fill='both', expand=True)
        features = [("📚", "浏览学科", "探索丰富的学科与主题内容", self.show_browse, "#2E7D32"), ("❓", "提问", "从 AI 助手获取即时解答", self.show_ask, "#1976D2"), ("📊", "查看学习进度", "了解您的学习情况与改进方向", self.show_progress, "#7B1FA2")]
        for icon, title, desc, command, color in features:
            card = ctk.CTkFrame(features_frame, corner_radius=15, fg_color="#FFFFFF", border_width=1, border_color="#E0E0E0")
            card.pack(side='left', fill='both', expand=True, padx=10, pady=10)
            icon_label = ctk.CTkLabel(card, text=icon, font=ctk.CTkFont(size=48))
            icon_label.pack(pady=(20, 10))
            title_label = ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=18, weight="bold"), text_color=color)
            title_label.pack(pady=(0, 10))
            desc_label = ctk.CTkLabel(card, text=desc, font=ctk.CTkFont(size=14), text_color="#666666", wraplength=200, justify='center')
            desc_label.pack(pady=(0, 20), padx=20)
            action_btn = ctk.CTkButton(card, text="开始学习", command=command, fg_color=color, hover_color=self._darken_color(color), width=150, height=35)
            action_btn.pack(pady=(0, 20))

    def show_browse(self):
        """Show grade selection first, then proceed to subjects."""
        if self._loading: return
        self._safe_destroy_widgets()
        
        view = GradeView(
            self.main_frame, 
            on_select=self.on_grade_selected, 
            on_back=self.show_main_menu
        )
        view.pack(fill='both', expand=True)
    
    def on_grade_selected(self, grade: int):
        """Handle grade selection, then show subjects."""
        self.current_grade_filter = str(grade)
        self.selected_grade = grade
        
        if hasattr(self, 'grade_selector'):
            self.grade_selector.set(_GRADE_CN.get(grade, f"第 {grade} 年级"))
        
        if not self.content_manager:
            self._show_loading_message("正在初始化...请稍候。")
            self.after(100, lambda: self.on_grade_selected(grade))
            return
        
        self._loading = True
        self._safe_destroy_widgets()
        
        loading_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        loading_frame.pack(fill='both', expand=True)
        loading_icon = ctk.CTkLabel(loading_frame, text="⏳", font=ctk.CTkFont(size=48))
        loading_icon.pack(pady=40)
        loading_label = ctk.CTkLabel(loading_frame, text="正在加载学科...", font=ctk.CTkFont(size=18), text_color="#666666")
        loading_label.pack(pady=10)
        self.update()
        
        def load_subjects():
            try:
                subjects = self.content_manager.get_all_subjects()
                def show_subjects():
                    self._safe_destroy_widgets()
                    # 学科按钮显示中文名，但 on_subject_selected 回调仍收到英文数据键
                    display = {**_SUBJECT_EN_TO_CN, **_SUBJECT_ALIAS_TO_CN}
                    view = SubjectView(self.main_frame, subjects, self.on_subject_selected, self.show_browse, display_names=display)
                    view.pack(fill='both', expand=True)
                    self._loading = False
                safe_after(self, 0, show_subjects)
            except Exception as e:
                def show_error():
                    self._safe_destroy_widgets()
                    mb.showerror("错误", f"加载学科失败：{e}")
                    self.show_main_menu()
                    self._loading = False
                safe_after(self, 0, show_error)
        threading.Thread(target=load_subjects, daemon=True).start()


    def _show_loading_message(self, message):
         self._safe_destroy_widgets()
         loading_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
         loading_frame.pack(fill='both', expand=True)
         ctk.CTkLabel(loading_frame, text=message, font=ctk.CTkFont(size=16), text_color="#666666").pack(pady=40)

    def on_subject_selected(self, subject):
        if self._loading: return
        self._loading = True
        self.selected_subject = subject
        self.subject_selector.set(_subject_to_cn(subject))
        self.current_subject_filter = subject
        # 从「浏览学科」进入时，学科是内容库给的家族键（如 Science），
        # 不存在细分收窄，需清掉可能残留的 discipline 过滤。
        self.current_discipline_filter = self._discipline_from_label(
            self.subject_selector.get())
        self._safe_destroy_widgets()
        loading = ctk.CTkLabel(self.main_frame, text="正在加载主题...", font=ctk.CTkFont(size=16))
        loading.pack(pady=40)
        self.main_frame.update()
        def load_topics():
            try:
                topics = self.content_manager.list_browseable_topics(subject)
                def show_topics():
                    self._safe_destroy_widgets()
                    view = TopicView(self.main_frame, topics, self.on_topic_selected, self.show_browse)
                    view.pack(fill='both', expand=True)
                    self._loading = False
                safe_after(self, 0, show_topics)
            except Exception as e:
                 def show_error():
                    self._safe_destroy_widgets()
                    mb.showerror("错误", f"加载主题失败：{e}")
                    self.show_browse()
                    self._loading = False
                 safe_after(self, 0, show_error)
        threading.Thread(target=load_topics, daemon=True).start()

    def on_topic_selected(self, topic):
        if self._loading: return
        self._loading = True
        if isinstance(topic, dict) and "topic" in topic:
            self.selected_topic = topic["topic"]
            self.selected_subtopic_path = topic.get("subtopic_path", [])
            display_label = topic.get("label", self.selected_topic)
        else:
            self.selected_topic = topic.get("name") if isinstance(topic, dict) else topic
            self.selected_subtopic_path = []
            display_label = self.selected_topic
        self._safe_destroy_widgets()
        loading = ctk.CTkLabel(self.main_frame, text=f"正在加载 {display_label} 的概念...", font=ctk.CTkFont(size=16))
        loading.pack(pady=40)
        self.main_frame.update()
        def load_concepts():
            try:
                concepts = self.content_manager.get_concepts_at_path(self.selected_subject, self.selected_topic, self.selected_subtopic_path,)
                def show_concepts():
                    self._safe_destroy_widgets()
                    view = ConceptView(self.main_frame, concepts, self.on_concept_selected, lambda: self.on_subject_selected(self.selected_subject))
                    view.pack(fill='both', expand=True)
                    self._loading = False
                safe_after(self, 0, show_concepts)
            except Exception as e:
                def show_error():
                    self._safe_destroy_widgets()
                    mb.showerror("错误", f"加载概念失败：{e}")
                    self.on_subject_selected(self.selected_subject)
                    self._loading = False
                safe_after(self, 0, show_error)
        threading.Thread(target=load_concepts, daemon=True).start()

    def on_concept_selected(self, concept_name):
        if self._loading: return
        self._loading = True
        self.selected_concept = concept_name
        self._safe_destroy_widgets()
        loading = ctk.CTkLabel(self.main_frame, text="正在加载概念详情...", font=ctk.CTkFont(size=16))
        loading.pack(pady=40)
        self.main_frame.update()
        def load_concept():
            try:
                concept = self.content_manager.get_concept(self.selected_subject, self.selected_topic, concept_name)
                self.selected_concept_data = concept
                def show_concept():
                    self._safe_destroy_widgets()
                    view = ConceptDetailView(self.main_frame, concept, self.start_questions, lambda: self.on_topic_selected(self.selected_topic))
                    view.pack(fill='both', expand=True)
                    self._loading = False
                safe_after(self, 0, show_concept)
            except Exception as e:
                def show_error():
                    self._safe_destroy_widgets()
                    mb.showerror("错误", f"加载概念失败：{e}")
                    self.on_topic_selected(self.selected_topic)
                    self._loading = False
                safe_after(self, 0, show_error)
        threading.Thread(target=load_concept, daemon=True).start()

    def start_questions(self):
        self.question_index = 0
        self.show_question()

    def show_question(self):
        if self._loading: return
        self._loading = True
        concept = self.selected_concept_data
        questions = concept.get('questions', [])
        if self.question_index >= len(questions):
            self.show_concept_complete()
            return
        q = questions[self.question_index]
        self._safe_destroy_widgets()
        view = QuestionView(self.main_frame, q['question'], lambda ans: self.on_answer_submitted(ans, q), lambda: self.on_concept_selected(self.selected_concept))
        view.pack(fill='both', expand=True)
        self._loading = False

    def on_answer_submitted(self, answer, question):
        if self._loading: return
        self._loading = True
        
        self._safe_destroy_widgets()
        loading = ctk.CTkLabel(self.main_frame, text="🤖 AI 正在批改您的答案...", font=ctk.CTkFont(size=18))
        loading.pack(pady=40)
        
        def worker():
            user_ans = answer.strip()
            correct_ans = question.get('answer', '').strip()
            
            is_correct, explanation = self._grade_answer_with_ai(question['question'], user_ans, correct_ans)
            
            def show_result():
                progress_manager.update_progress(self.username, self.selected_subject, self.selected_topic, self.selected_concept, question['question'], is_correct)
                
                self._safe_destroy_widgets()
                if is_correct:
                    msg = "✅ 回答正确！"
                    color = "#43a047"
                else:
                    msg = "❌ 回答错误"
                    color = "#e53935"
                
                result_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
                result_frame.pack(fill='both', expand=True, pady=40)
                
                label = ctk.CTkLabel(result_frame, text=msg, font=ctk.CTkFont(size=22, weight="bold"), text_color=color, wraplength=600)
                label.pack(pady=(40, 10))
                
                exp_text = explanation if explanation else question.get('explanation', f"正确答案是：{correct_ans}")
                
                exp_frame = ctk.CTkFrame(result_frame, fg_color="#f5f5f5", corner_radius=10)
                exp_frame.pack(pady=20, padx=40, fill='x')
                ctk.CTkLabel(exp_frame, text="反馈：", font=ctk.CTkFont(size=16, weight="bold"), text_color="#424242").pack(anchor='w', padx=15, pady=(10, 5))
                ctk.CTkLabel(exp_frame, text=exp_text, font=ctk.CTkFont(size=14), text_color="#616161", wraplength=500, justify='left').pack(anchor='w', padx=15, pady=(0, 10))
                
                ctk.CTkButton(result_frame, text="下一题", command=self.next_question, width=200, height=40).pack(pady=30)
                self._loading = False
                
            safe_after(self, 0, show_result)
            
        threading.Thread(target=worker, daemon=True).start()

    def next_question(self):
        if self._loading: return
        self.question_index += 1
        self.show_question()

    def show_concept_complete(self):
        if self._loading: return
        self._loading = True
        self._safe_destroy_widgets()
        complete_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        complete_frame.pack(fill='both', expand=True, pady=60)
        ctk.CTkLabel(complete_frame, text="🎉 恭喜您！", font=ctk.CTkFont(size=32, weight="bold"), text_color="#2E7D32").pack(pady=(40, 10))
        ctk.CTkLabel(complete_frame, text="您已完成本概念的全部题目！", font=ctk.CTkFont(size=18), text_color="#666666").pack(pady=10)
        ctk.CTkButton(complete_frame, text="返回概念列表", command=lambda: self.on_topic_selected(self.selected_topic), width=200, height=40).pack(pady=30)
        self._loading = False

    def show_ask(self):
        if self._loading: return
        self._loading = True
        self._safe_destroy_widgets()
        self.ask_view = AskQuestionView(self.main_frame, self.on_ask_submit, self.show_main_menu)
        self.ask_view.pack(fill='both', expand=True)
        self._loading = False

    
    @timeit
    def _grade_answer_with_ai(self, question_text, user_answer, correct_answer):
        """Using the model to grade the answer relative to the correct answer."""
        if not self.model_handler:
            is_correct = (user_answer.lower() == correct_answer.lower()) or (difflib.SequenceMatcher(None, user_answer.lower(), correct_answer.lower()).ratio() > 0.8)
            return is_correct, f"正确答案是：{correct_answer}"

        prompt = (
            f"任务：根据「正确答案」评判学生答案是否正确，并用简体中文给出具体、有帮助的反馈。\n\n"
            f"示例1：\n"
            f"问题：线粒体的功能是什么？\n"
            f"正确答案：它通过细胞呼吸为细胞产生能量。\n"
            f"学生答案：它保护细胞核。\n"
            f"VERDICT: [INCORRECT]\n"
            f"FEEDBACK: 这个回答不正确。细胞核由核膜保护；线粒体是细胞的“动力车间”，负责通过细胞呼吸产生能量（ATP）。\n\n"
            f"示例2：\n"
            f"问题：2 + 2 等于多少？\n"
            f"正确答案：4\n"
            f"学生答案：等于四。\n"
            f"VERDICT: [CORRECT]\n"
            f"FEEDBACK: 回答正确！你答对了。\n\n"
            f"当前任务：\n"
            f"问题：{question_text}\n"
            f"正确答案：{correct_answer}\n"
            f"学生答案：{user_answer}\n"
            f"VERDICT: "
        )
        
        try:

            response = self.model_handler.generate_response(prompt, max_tokens=100)
            
            response_upper = response.upper()
            is_correct = "[CORRECT]" in response_upper
            explanation = response
            
            explanation = re.sub(r'^\[.*?\]', '', explanation).strip()
            
            if explanation.upper().startswith("VERDICT:"):
                explanation = explanation[8:].strip()
                explanation = re.sub(r'^\[.*?\]', '', explanation).strip()
                
            if explanation.upper().startswith("FEEDBACK:"):
                explanation = explanation[9:].strip()
            
            explanation = explanation.strip()
            
            if not is_correct and "[INCORRECT]" not in response_upper:
                 
                 is_correct = (user_answer.lower() == correct_answer.lower())
            
            return is_correct, explanation
            
        except Exception as e:
            print(f"AI Grading Error: {e}")
            is_correct = (user_answer.lower() == correct_answer.lower()) or (difflib.SequenceMatcher(None, user_answer.lower(), correct_answer.lower()).ratio() > 0.8)
            return is_correct, f"正确答案是：{correct_answer}"

    def on_ask_submit(self, question, answer_length="medium"):
        if self._loading:
            return
        self._loading = True
        
        self.ask_view.set_loading(True)
        
        def worker():
            try:
                normalized_question = question.strip()
                if not normalized_question:
                    def show_error():
                        mb.showerror("错误", "请输入问题")
                        self.ask_view.set_loading(False)
                        self._loading = False
                    safe_after(self, 0, show_error)
                    return
                
                if not self.rag_engine:
                    def show_error():
                        self.ask_view.set_result(
                            "RAG 检索引擎未初始化，请重启应用。"
                        )
                        self._loading = False
                    safe_after(self, 0, show_error)
                    return
                
                first_token = [True]
                
                def on_token(token):
                    def display():
                        if first_token[0]:
                            self.ask_view.set_loading(False)
                            first_token[0] = False
                        self.ask_view.append_answer_token(token)
                    safe_after(self, 0, display)
                
                # 方案A：运行状态消息显示在独立标签，不流入答案框
                def on_status(msg):
                    def display():
                        self.ask_view.set_status(msg)
                    safe_after(self, 0, display)
                
                # 方案C：多轮对话历史（最近 6 轮），用于追问承接
                history = getattr(self, '_ask_history', [])
                
                try:
                    response = self.rag_engine.query(
                        query_text=normalized_question,
                        subject=self.current_subject_filter,
                        grade=self.current_grade_filter,
                        discipline=getattr(self, 'current_discipline_filter', None),
                        stream_callback=on_token,
                        status_callback=on_status,
                        conversation=history,
                        already_normalized=True,
                    )
                    
                    confidence = response.get('confidence', 0.5)
                    diagram = response.get('diagram')
                    llm_used = response.get('llm_used', False)
                    # 溯源：优先用检索层生成的编号化来源（P2），
                    # 退回按书名去重的旧格式
                    source_info = _format_sources(
                        response.get('sources'),
                        citations=response.get('citations'),
                    )
                    
                    def finalize():
                        self.ask_view.finalize_answer(
                            confidence, 
                            question=normalized_question,
                            grade=self.selected_grade,
                            subject=self.current_subject_filter,
                            source_info=source_info
                        )
                        
                        if diagram:
                            try:
                                self.ask_view.show_diagram(diagram)
                            except AttributeError:
                                pass
                        
                        # 记录本轮问答用于后续追问（仅记录大模型参与的有效回答）
                        if llm_used and response.get('answer'):
                            self._ask_history = getattr(self, '_ask_history', [])
                            self._ask_history.append((normalized_question, response['answer']))
                            self._ask_history = self._ask_history[-6:]

                        log_resource_usage("问答完成")
                        self._loading = False
                    
                    safe_after(self, 0, finalize)
                    
                except Exception as e:
                    error_msg = str(e)
                    def show_error():
                        mb.showerror("RAG 错误", f"获取答案失败：{error_msg}")
                        self.ask_view.set_loading(False)
                        self._loading = False
                    safe_after(self, 0, show_error)
                    
            except Exception as ex:
                error_msg = str(ex)
                def show_error():
                    mb.showerror("错误", f"发生错误：{error_msg}")
                    self.ask_view.set_loading(False)
                    self._loading = False
                safe_after(self, 0, show_error)
        
        threading.Thread(target=worker, daemon=True).start()

    def _calculate_progress_stats(self):
        progress_data = progress_manager.load_progress(self.username)
        
        total_q = 0
        total_correct = 0
        mastered = []
        weak = []
        subject_stats = {}
        
        for subject, topics in progress_data.items():
            subj_total = 0
            subj_correct = 0
            
            for topic, concepts in topics.items():
                for concept, data in concepts.items():
                    questions = data.get('questions', [])
                    c_total = sum(q['attempts'] for q in questions)
                    c_correct = sum(q['correct'] for q in questions)
                    
                    if c_total >= 5:
                        score = (c_correct / c_total) * 100
                        if score >= 80:
                            mastered.append(f"{concept} ({topic})")
                        elif score < 50:
                            weak.append(f"{concept} ({topic})")
                    
                    subj_total += c_total
                    subj_correct += c_correct
            
            if subj_total > 0:
                pct = (subj_correct / subj_total) * 100
                subject_stats[subject] = {
                    'pct': pct,
                    'correct': subj_correct,
                    'total': subj_total
                }
            
            total_q += subj_total
            total_correct += subj_correct
            
        overall_score = (total_correct / total_q * 100) if total_q > 0 else 0
        
        stats = {
            'total': total_q,
            'correct': total_correct,
            'score': overall_score
        }
        
        next_concept = None
        if weak:
            next_concept = f"建议复习：{weak[0]}"
        elif mastered:
             next_concept = "做得好！试试探索新的学科。"
        else:
             next_concept = "先从浏览学科开始您的学习之旅吧！"

        return stats, mastered, weak, subject_stats, next_concept

    def _export_progress(self):
        src = progress_manager.get_progress_path(self.username)
        if not os.path.exists(src):
            mb.showinfo("提示", "没有可导出的学习进度数据。")
            return
            
        dest = fd.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON 文件", "*.json")],
            initialfile=f"mati_progress_{self.username}.json",
            title="导出学习进度"
        )
        if dest:
            import shutil
            try:
                shutil.copy2(src, dest)
                mb.showinfo("成功", "学习进度导出成功！")
            except Exception as e:
                log_security_event(f"GUI progress export failed: {e}")
                mb.showerror("错误", f"导出失败：{e}")

    def _import_progress(self):
        src = fd.askopenfilename(
            filetypes=[("JSON 文件", "*.json")],
            title="导入学习进度"
        )
        if src:
            dest = progress_manager.get_progress_path(self.username)
            import shutil
            try:
                with open(src, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                log_security_event(f"GUI progress import rejected (unreadable): {src} ({e})")
                mb.showerror("错误", f"导入失败：文件不是有效的 JSON 进度文件。\n{e}")
                return
            if not validate_content_input(data):
                log_security_event(f"GUI progress import rejected (invalid content): {src}")
                mb.showerror("错误", "进度文件无效或过大，已拒绝导入。")
                return
            try:
                shutil.copy2(src, dest)
                mb.showinfo("成功", "学习进度导入成功！请重启应用以查看更改。")
                self.show_progress_ops()
            except Exception as e:
                log_security_event(f"GUI progress import failed: {e}")
                mb.showerror("错误", f"导入失败：{e}")

    def _reset_progress(self):
        if mb.askyesno("确认重置", "您确定要删除所有学习进度吗？此操作无法撤销。"):
            try:
                path = progress_manager.get_progress_path(self.username)
                if os.path.exists(path):
                    os.remove(path)
                mb.showinfo("成功", "学习进度已重置。")
                self.show_progress_ops()
            except Exception as e:
                mb.showerror("错误", f"重置失败：{e}")

    def show_progress(self):
        if self._loading: return
        self._loading = True
        self._safe_destroy_widgets()
        
        stats, mastered, weak, subject_stats, next_concept = self._calculate_progress_stats()
        
        view = ProgressView(
            self.main_frame, 
            stats=stats,
            mastered=mastered,
            weak=weak,
            subject_stats=subject_stats,
            next_concept=next_concept, 
            on_back=self.show_main_menu
        )
        view.pack(fill='both', expand=True)
        self._loading = False

    def show_progress_ops(self):
        if self._loading: return
        self._loading = True
        self._safe_destroy_widgets()
        
        view = ProgressOpsView(
            self.main_frame, 
            on_export=self._export_progress,
            on_import=self._import_progress,
            on_reset=self._reset_progress,
            on_back=self.show_main_menu
        )
        view.pack(fill='both', expand=True)
        self._loading = False

    def show_teacher(self):
        if self._loading: return
        self._loading = True
        self._safe_destroy_widgets()
        view = TeacherView(self.main_frame, self.show_main_menu,
                           on_content_changed=self._on_content_changed)
        view.pack(fill='both', expand=True)
        self._loading = False

    def _on_content_changed(self):
        """教师导入/生成内容后：刷新内容管理器并提示。"""
        try:
            if self.content_manager is not None:
                self.content_manager.reload()
            mb.showinfo("内容已更新", "内容已刷新，可到“浏览学科”查看新的章节与试题。")
        except Exception as e:
            mb.showerror("刷新失败", f"刷新内容失败：{e}")

    def show_user_guide(self):
        if self._loading: return
        self._loading = True
        self._safe_destroy_widgets()
        view = UserGuideView(self.main_frame, self.show_main_menu)
        view.pack(fill='both', expand=True)
        self._loading = False

    def show_about(self):
        if self._loading: return
        self._loading = True
        self._safe_destroy_widgets()
        view = AboutView(self.main_frame, self.show_main_menu)
        view.pack(fill='both', expand=True)
        self._loading = False

if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except Exception:
            pass
            
    app = NEBeduApp()
    app.mainloop()
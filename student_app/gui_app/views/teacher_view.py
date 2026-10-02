"""
教师工作台（TeacherView）

为教师提供图形化前端，无需使用命令行即可：
  1. 导入教材（PDF/TXT/MD）到 RAG 知识库（学生可对课本提问）
  2. 从课本 PDF 自动生成章节、概念与测试题（调用 Qwen 模型）

依赖（项目已具备）：chromadb、sentence-transformers、llama-cpp-python、pymupdf
模型：mati_data/models/qwen2_5/*.gguf
"""

import shutil
import sys
import threading
import tkinter.filedialog as fd
from pathlib import Path

import customtkinter as ctk

# 项目根目录（本文件位于 student_app/gui_app/views/）
PROJECT_ROOT = Path(__file__).resolve().parents[3]
TOOLS_DIR = PROJECT_ROOT / "tools"
CONTENT_DIR = PROJECT_ROOT / "scripts" / "data_collection" / "data" / "content"
CHROMA_DB = PROJECT_ROOT / "mati_data" / "chroma_db"

# 学科/年级：中文显示值 -> 英文/数字数据键
_SUBJECT_OPTIONS = {
    "科学": "Science",
    "数学": "Math",
    "英语语法": "English Grammar",
    "计算机科学": "Computer Science",
    "社会科学": "Social Studies",
}
_GRADE_OPTIONS = {"初一": 7, "初二": 8, "初三": 9, "高一": 10, "高二": 11, "高三": 12}
_GRADE_INT_TO_CN = {v: k for k, v in _GRADE_OPTIONS.items()}

# 「导入知识库」专用的学科/年级选项：多一个「自动识别」默认项。
# 学科可选到细分（物理/化学/生物），因为 science 家族下多个细分学科会分集合存放，
# 选到细分才能让教材落到 neb_physics_* 而不是家族级的 neb_science_*。
_IMPORT_SUBJECT_CHOICES = {
    "自动识别（按书名）": None,
    "科学（不细分）": "Science",
    "物理": "物理",
    "化学": "化学",
    "生物": "生物",
    "数学": "Math",
    "语文": "Chinese",
    "英语语法": "English Grammar",
    "计算机科学": "Computer Science",
    "社会科学": "Social Studies",
}
_IMPORT_GRADE_CHOICES = {
    "自动识别（按书名/目录）": None,
    "高一": 10,
    "高二": 11,
    "高三": 12,
    "初一": 7,
    "初二": 8,
    "初三": 9,
}


class TeacherView(ctk.CTkFrame):
    """教师工作台：前端导入教材 + 生成测试题。"""

    def __init__(self, master, on_back, on_content_changed=None, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.on_back = on_back
        self.on_content_changed = on_content_changed
        self._busy = False
        self._import_files = []
        self._gen_pdf = None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI 构建
    # ------------------------------------------------------------------
    def _build_ui(self):
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill='x', padx=40, pady=(30, 20))
        ctk.CTkLabel(
            header, text="🧑‍🏫 教师工作台",
            font=ctk.CTkFont(family="Segoe UI", size=28, weight="bold"),
            text_color="#1A237E").pack(anchor='w')
        ctk.CTkLabel(
            header, text="把教材 PDF 导入知识库，并自动生成章节与测试题（全程图形界面，无需命令行）。",
            font=ctk.CTkFont(family="Segoe UI", size=14),
            text_color="#757575").pack(anchor='w', pady=(2, 0))

        body = ctk.CTkScrollableFrame(self, fg_color="transparent")
        body.pack(fill='both', expand=True, padx=40, pady=(0, 10))

        # ---- 分区 1：导入教材到知识库 ----
        card1 = self._make_card(
            body, "📚", "导入教材到知识库",
            "选择 PDF / TXT / MD 教材，处理后学生即可对课本内容提问（RAG 问答）。")
        row1 = ctk.CTkFrame(card1, fg_color="transparent")
        row1.pack(fill='x', pady=(6, 0))
        ctk.CTkButton(row1, text="选择教材文件", command=self._choose_import_files,
                      fg_color="#3949AB", hover_color="#303F9F", width=180).pack(side='left', padx=(0, 10))
        ctk.CTkButton(row1, text="导入知识库", command=self._do_import,
                      fg_color="#2E7D32", hover_color="#1B5E20", width=160).pack(side='left')
        self.import_files_label = ctk.CTkLabel(
            card1, text="未选择文件", text_color="#757575", anchor='w')
        self.import_files_label.pack(fill='x', pady=(6, 0))

        # 学科 / 年级：自编讲义的文件名推不出学科年级，可在此手动指定；
        # 留「自动识别」时行为与从前一致（按书名 + 路径推断）。
        row1b = ctk.CTkFrame(card1, fg_color="transparent")
        row1b.pack(fill='x', pady=(8, 0))
        ctk.CTkLabel(row1b, text="学科：", font=ctk.CTkFont(size=14)).pack(side='left')
        self.import_subject_menu = ctk.CTkOptionMenu(
            row1b, values=list(_IMPORT_SUBJECT_CHOICES.keys()), width=180)
        self.import_subject_menu.set("自动识别（按书名）")
        self.import_subject_menu.pack(side='left', padx=(0, 16))
        ctk.CTkLabel(row1b, text="年级：", font=ctk.CTkFont(size=14)).pack(side='left')
        self.import_grade_menu = ctk.CTkOptionMenu(
            row1b, values=list(_IMPORT_GRADE_CHOICES.keys()), width=170)
        self.import_grade_menu.set("自动识别（按书名/目录）")
        self.import_grade_menu.pack(side='left')
        ctk.CTkLabel(
            card1,
            text="提示：不指定时可留「自动识别」；自编讲义建议手动指定，否则会落到通用集合。",
            font=ctk.CTkFont(size=12), text_color="#9E9E9E",
            anchor='w', justify='left').pack(fill='x', pady=(4, 0))

        # 导入结果：逐文件展示，0 块必须显式报警（不再笼统说"导入完成"）
        self.import_result_box = ctk.CTkTextbox(
            card1, height=110, fg_color="#FAFAFA", text_color="#212121",
            corner_radius=8, font=ctk.CTkFont(family="Consolas", size=12))
        self.import_result_box.pack(fill='x', pady=(8, 0))
        self.import_result_box.configure(state="disabled")

        # ---- 分区 2：从 PDF 生成测试题 ----
        card2 = self._make_card(
            body, "🧩", "从 PDF 生成章节与测试题",
            "选择一本课本 PDF，系统用 AI 自动生成章节、概念与简答题（含参考答案），"
            "生成后可在“浏览学科”中查看与做题。")
        row2 = ctk.CTkFrame(card2, fg_color="transparent")
        row2.pack(fill='x', pady=(6, 0))
        ctk.CTkButton(row2, text="选择 PDF", command=self._choose_gen_pdf,
                      fg_color="#3949AB", hover_color="#303F9F", width=150).pack(side='left', padx=(0, 12))
        self.gen_pdf_label = ctk.CTkLabel(row2, text="未选择 PDF", text_color="#757575", anchor='w')
        self.gen_pdf_label.pack(side='left', fill='x', expand=True)

        row2b = ctk.CTkFrame(card2, fg_color="transparent")
        row2b.pack(fill='x', pady=(8, 0))
        ctk.CTkLabel(row2b, text="学科：", font=ctk.CTkFont(size=14)).pack(side='left')
        self.subject_menu = ctk.CTkOptionMenu(row2b, values=list(_SUBJECT_OPTIONS.keys()), width=150)
        self.subject_menu.set("科学")
        self.subject_menu.pack(side='left', padx=(0, 20))
        ctk.CTkLabel(row2b, text="年级：", font=ctk.CTkFont(size=14)).pack(side='left')
        self.grade_menu = ctk.CTkOptionMenu(row2b, values=list(_GRADE_OPTIONS.keys()), width=120)
        self.grade_menu.set("高一")
        self.grade_menu.pack(side='left')
        ctk.CTkButton(row2b, text="生成测试题", command=self._do_generate,
                      fg_color="#1976D2", hover_color="#0D47A1", width=160).pack(side='right')

        # ---- 状态日志 ----
        ctk.CTkLabel(body, text="运行状态：", font=ctk.CTkFont(size=14, weight="bold"),
                     text_color="#212121").pack(anchor='w', pady=(12, 4))
        self.log_box = ctk.CTkTextbox(
            body, height=180, fg_color="#F5F5F5", text_color="#212121",
            corner_radius=8, font=ctk.CTkFont(family="Consolas", size=12))
        self.log_box.pack(fill='x')
        self.log_box.configure(state="disabled")

        # 返回
        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.pack(fill='x', padx=40, pady=(10, 30), side='bottom')
        ctk.CTkButton(
            footer, text="← 返回主菜单", command=self.on_back,
            font=ctk.CTkFont(size=14), height=45, fg_color="transparent",
            border_width=1, border_color="#BDBDBD", text_color="#616161",
            hover_color="#F5F5F5").pack(side='left')

    def _make_card(self, parent, icon, title, desc):
        card = ctk.CTkFrame(parent, fg_color="#FFFFFF", corner_radius=15,
                            border_width=1, border_color="#E0E0E0")
        card.pack(fill='x', pady=10)
        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(fill='both', expand=True, padx=20, pady=18)
        ctk.CTkLabel(inner, text=icon, font=ctk.CTkFont(size=30)).pack(side='left', padx=(0, 18))
        tf = ctk.CTkFrame(inner, fg_color="transparent")
        tf.pack(side='left', fill='both', expand=True)
        ctk.CTkLabel(tf, text=title, font=ctk.CTkFont(family="Segoe UI", size=18, weight="bold"),
                     text_color="#212121").pack(anchor='w')
        ctk.CTkLabel(tf, text=desc, font=ctk.CTkFont(family="Segoe UI", size=13),
                     text_color="#757575").pack(anchor='w')
        return inner

    # ------------------------------------------------------------------
    # 日志（线程安全）
    # ------------------------------------------------------------------
    def _notify_content_changed(self):
        """通知主窗口：内容已更新，请刷新（线程安全）。"""
        if self.on_content_changed:
            try:
                self.after(0, self.on_content_changed)
            except Exception:
                pass

    def _append_log(self, msg):
        def do():
            self.log_box.configure(state="normal")
            self.log_box.insert("end", msg + "\n")
            self.log_box.see("end")
            self.log_box.configure(state="disabled")
        try:
            self.after(0, do)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 分区 1：导入教材到知识库
    # ------------------------------------------------------------------
    def _choose_import_files(self):
        files = fd.askopenfilenames(
            title="选择教材文件（PDF/TXT/MD）",
            filetypes=[("教材文件", "*.pdf *.txt *.md"), ("所有文件", "*.*")])
        if files:
            self._import_files = list(files)
            names = "、".join(Path(f).name for f in files[:3])
            if len(files) > 3:
                names += f" 等 {len(files)} 个文件"
            self.import_files_label.configure(text=f"已选择：{names}")

    def _set_import_result(self, text):
        """把导入结果写进结果框（线程安全）。"""
        def do():
            self.import_result_box.configure(state="normal")
            self.import_result_box.delete("1.0", "end")
            self.import_result_box.insert("end", text)
            self.import_result_box.configure(state="disabled")
        try:
            self.after(0, do)
        except Exception:
            pass

    def _do_import(self):
        if not self._import_files:
            self._append_log("⚠️ 请先选择教材文件。")
            return
        if self._busy:
            self._append_log("⏳ 正在处理中，请稍候……")
            return
        self._busy = True
        subject = _IMPORT_SUBJECT_CHOICES.get(self.import_subject_menu.get())
        grade = _IMPORT_GRADE_CHOICES.get(self.import_grade_menu.get())
        hint = []
        if subject:
            hint.append(f"学科={self.import_subject_menu.get()}")
        if grade:
            hint.append(f"年级={_GRADE_INT_TO_CN.get(grade, grade)}")
        self._append_log("📥 开始导入教材到知识库……"
                         + ("（" + "，".join(hint) + "）" if hint else "（自动识别）"))
        self._set_import_result("处理中……")
        files = list(self._import_files)
        threading.Thread(target=self._import_worker, args=(files, subject, grade),
                         daemon=True).start()

    def _import_worker(self, files, subject, grade):
        """
        复制文件 → 摄取 → **逐文件回报结果**。

        这里刻意不再无条件打印「✅ 导入完成」：摄取返回的结构化摘要里
        每个文件都带状态与原因，没有任何文件成功入库时必须明确报错
        （扫描件过去就是这样静默变成 0 块却显示成功的）。
        """
        try:
            dest = PROJECT_ROOT / "textbooks"
            dest.mkdir(exist_ok=True)
            for f in files:
                shutil.copy2(f, dest / Path(f).name)
            self._append_log(f"已复制 {len(files)} 个文件到 {dest}")

            sys.path.insert(0, str(PROJECT_ROOT))
            from scripts.ingest_content import (
                UniversalContentIngester, format_ingest_summary,
            )
            ingester = UniversalContentIngester(
                str(CHROMA_DB), "auto",
                force_subject=subject, force_grade=grade,
            )
            summary = ingester.ingest_directory(str(dest), progress=False)
            report = format_ingest_summary(summary)
            self._set_import_result(report)

            if summary["ok"] == 0:
                self._append_log("❌ 导入失败：没有任何文件成功入库，请查看右侧明细。")
            elif summary["skipped"] or summary["failed"]:
                self._append_log(
                    f"⚠️ 部分成功：成功 {summary['ok']}，跳过 {summary['skipped']}，"
                    f"失败 {summary['failed']}（共 {summary['chunks']} 块）。"
                    "未入库的文件学生查不到，请查看明细。")
            else:
                self._append_log(
                    f"✅ 导入完成：{summary['ok']} 个文件、{summary['chunks']} 块已进入知识库。")
            self._notify_content_changed()
        except Exception as e:
            self._append_log(f"❌ 导入失败：{e}")
            self._set_import_result(f"❌ 导入失败：{e}")
            self._append_log("   请确认已安装 chromadb 与 sentence-transformers"
                             "（首次使用会下载 bge 嵌入模型）。")
        finally:
            self._busy = False

    # ------------------------------------------------------------------
    # 分区 2：从 PDF 生成测试题
    # ------------------------------------------------------------------
    def _choose_gen_pdf(self):
        f = fd.askopenfilename(
            title="选择课本 PDF",
            filetypes=[("PDF 文件", "*.pdf"), ("所有文件", "*.*")])
        if f:
            self._gen_pdf = f
            self.gen_pdf_label.configure(text=f"已选择：{Path(f).name}")

    def _do_generate(self):
        if not self._gen_pdf:
            self._append_log("⚠️ 请先选择 PDF。")
            return
        if self._busy:
            self._append_log("⏳ 正在处理中，请稍候……")
            return
        subject = _SUBJECT_OPTIONS[self.subject_menu.get()]
        grade = _GRADE_OPTIONS[self.grade_menu.get()]
        self._busy = True
        self._append_log(f"🧩 开始用 AI 生成测试题（{subject}，{_GRADE_INT_TO_CN.get(grade, str(grade))}，约需几分钟）……")
        threading.Thread(
            target=self._generate_worker, args=(self._gen_pdf, subject, grade),
            daemon=True).start()

    def _generate_worker(self, pdf, subject, grade):
        try:
            sys.path.insert(0, str(TOOLS_DIR))
            from pdf_to_content import generate_from_pdf
            result = generate_from_pdf(
                pdf_path=Path(pdf), subject=subject, grade=grade,
                out_dir=str(CONTENT_DIR), n_threads=4,
            )
            if result["ok"]:
                self._append_log(
                    f"✅ 生成完成：{result['topics']} 章 / {result['concepts']} 概念"
                    f" / {result['questions']} 题")
                if result.get("out_file"):
                    self._append_log(f"   已保存：{result['out_file']}")
                self._append_log("   内容已自动刷新，回到“浏览学科”即可查看章节与做题。")
                self._notify_content_changed()
            else:
                self._append_log(f"❌ 生成失败：{result.get('error')}")
        except Exception as e:
            self._append_log(f"❌ 生成失败：{e}")
            self._append_log("   请确认已下载 Qwen 模型（mati_data/models/qwen2_5/*.gguf）。")
        finally:
            self._busy = False

# GUI 关闭时序与模型上下文：两条控制台信息的说明与修复

> 环境：Windows / Anaconda `autoresearch` / Python 3.12.13 / tkinter 8.6 / **customtkinter 5.2.2** / Qwen2.5-1.5B-Instruct (GGUF)

---

## 1. 第 ① 条：llama.cpp 的上下文提示

```
llama_context: n_ctx_per_seq (4096) < n_ctx_train (32768)
               -- the full capacity of the model will not be utilized
```

### 是警告还是错误？

**是 info 级提示，不是错误，也不是警告。** llama.cpp 只是如实告诉你：
"这个模型训练时用的是 32768 的上下文，你现在只开了 4096，没用满它的能力上限。"

### 会不会影响推理效果？

**不会。** 关键在于"没用满"和"不够用"是两回事：

- 质量掉档发生在 **prompt + 输出超过 n_ctx 被截断** 的时候；
- 我们是 **远小于** 训练长度（4096 ≪ 32768），模型处在完全正常的区间内，
  位置编码、注意力都按原生方式工作，没有任何降级。

换句话说：**只有被截断才会有损，缩着用不会。**

### 那 4096 是怎么定的、能不能调？

Qwen2.5-1.5B 的实测元数据（`qwen2.*`）：

| 项 | 值 |
|---|---|
| `block_count`（层数） | 28 |
| `attention.head_count` / `head_count_kv` | 12 / 2 |
| `embedding_length` | 1536 → head_dim = 128 |
| `context_length`（训练长度） | 32768 |

本项目 `f16_kv=False`，KV cache 存 fp32（比 f16 稳，但占两倍）：

```
KV/token = 2(K,V) × 28层 × 2KV头 × 128维 × 2字节 ≈ 28 KB   （f16 时减半 ≈ 14 KB）
```

| n_ctx | KV cache（fp32 / f16） |
|---|---|
| 2048 | 57 MB / 29 MB |
| **4096** | **114 MB / 57 MB** ← 当前 |
| 8192 | 229 MB / 114 MB |
| 32768 | 917 MB / 458 MB |

4096 的依据（不是拍脑袋）：`system prompt ≈ 300 + 证据上限 2000 + 3 轮历史 + 输出 768`。
原先 2048 在 3 轮历史（约 660 token）时会把证据预算挤到只剩 280 —— 等于没有证据。

**调整方式**：改 `ai_model/model_utils/qwen_handler.py` 的 `n_ctx`，
**并同步** `mati_data/config/retrieval.yaml → context_budget.n_ctx`
（检索引擎按 n_ctx 反推证据预算，两处必须一致，否则会溢出）。
若只是想省内存，改 `f16_kv=True` 即可让 KV cache 减半。

---

## 2. 第 ② 条：Tkinter 回调异常

```
Exception in Tkinter callback
  File ".../tkinter/__init__.py", line ..., in callit
    func(*args)
  File ".../customtkinter/windows/widgets/scaling/scaling_tracker.py",
       line 178, in check_dpi_scaling
    if window.winfo_exists() and not window.state() == "iconic":
_tkinter.TclError: can't invoke "winfo" command: application has been destroyed
```

### 根因（逐条核过库源码，customtkinter 5.2.2）

1. **轮询启动后永不停止。**
   `ScalingTracker.add_widget()`（`scaling_tracker.py:82-84`）在注册第一个
   CTk 控件时起了一个 **100ms 的 `after` 轮询**，且只在 `update_loop_running`
   为 False 时启动 —— 之后**再没有任何地方会停它**。

2. **库自己的 `destroy()` 不注销窗口。**
   `CTk.destroy()`（`windows/ctk_tk.py:83-89`）只调了 `tkinter.Tk.destroy()`
   和两个基类的 destroy。5.2.2 里 `ScalingTracker.remove_window()` 确实存在
   （`scaling_tracker.py:94`），但**库内部没有任何调用点** ——
   窗口销毁后仍然留在 `window_widgets_dict` 里。

3. **销毁后已排队的回调仍会执行一次。**
   窗口销毁 → Tcl 解释器开始拆除 → 那个已经排队的 `after` 回调照常触发，
   此时 `window.winfo_exists()` 面对的是已销毁的 application → 抛 TclError。

4. **第一段循环没有异常保护。**
   `check_dpi_scaling` 的第一段遍历（`line 177-178`）**没有 try/except**；
   倒是第二段重新排 `after` 的地方（`line 196-204`）包了 try/except。
   于是异常一路冒到 tkinter 的 `callit`，被 `report_callback_exception` 打印成红栈。

补充两个**本项目放大该问题的因素**：

- 没有注册 `WM_DELETE_WINDOW` —— 关闭窗口时没有任何机会停轮询、停线程。
- 6 个工作线程直接调 `self.after(0, ...)` 投递 UI 更新。
  Tkinter **不是线程安全的**，跨线程 `after` 本身就已越界，
  在关闭阶段更是直接踩中 TclError。

### 这是 customtkinter 的缺陷，不是本项目的用法错误；但无害 ≠ 该忍：
每次关闭窗口刷一屏红，会掩盖真正的 UI 异常。

---

## 3. 修复方案（三层，已实现）

新增模块 **`student_app/gui_app/tk_safety.py`**。

| 层 | 手段 | 作用 |
|---|---|---|
| ① 治本 | `stop_scaling_tracker()` | 关闭前把轮询依赖的两个状态清掉 |
| ② 兜底 | `_guard_scaling_tracker()` | 给库函数套 TclError 保护，拦住"已排队、停不掉"的那一次 |
| ③ 兜底 | `_install_callback_exception_hook()` | 只吞"已销毁"类异常，其余照常抛出 |

**② 为什么必要**：即使 ① 停掉了轮询，也可能有**一个已经排队**的 `after` 回调
在窗口销毁后被执行 —— 那一次仍会炸。

**③ 的边界（重要）**：只吞 `application has been destroyed` /
`invalid command name` 这类；`bad screen distance`、`unknown option -foo`
这类真实错误**必须照原样打印**。已有测试锁定这个边界，
避免"为了消红字把 UI bug 一起藏了"。

### 关键代码位置

| 文件 | 位置 | 改动 |
|---|---|---|
| `student_app/gui_app/tk_safety.py` | 新增 | 三层安全网 + `safe_after()` |
| `student_app/gui_app/main_window.py` | `__init__` 末尾 | `install_shutdown_safety(self, on_close=self.on_close)` |
| 同上 | `on_close()`（新增） | 置 `_closing` → 取消 `_update_timer` → 停轮询 → 释放模型 → `destroy()` |
| 同上 | `cleanup_model()` | 修掉只查 `bitnet_handler` 的死代码 → 改调 `ModelHandler.cleanup()` |
| 同上 | 16 处 `self.after(0, …)` | 改为 `safe_after(self, 0, …)` |
| 同上 | 模块级 | 补 `logger = logging.getLogger(__name__)` |
| `ai_model/model_utils/qwen_handler.py` | `Llama(...)` | n_ctx 取舍与内存账写进注释 |

### 正确的窗口关闭写法

```python
# 构造之后、mainloop 之前
install_shutdown_safety(self, on_close=self.on_close)

def on_close(self):
    if getattr(self, "_closing", False):      # 防重入
        return
    self._closing = True                      # 让工作线程停止投递
    try:
        if self._update_timer:                # 先取消自己的定时器
            self.after_cancel(self._update_timer)
            self._update_timer = None
    except Exception:
        pass
    stop_scaling_tracker()                    # 再停库的轮询（必须在 destroy 前）
    self.cleanup_model()                      # 释放模型
    self.destroy()
```

**顺序不能反**：轮询是挂在窗口上的 `after` 定时器，窗口一销毁就再也没机会摘掉。

### 跨线程投递

```python
from student_app.gui_app.tk_safety import safe_after

# 工作线程里：替代 self.after(0, display)
safe_after(self, 0, display)
```

`safe_after` 会在窗口已销毁时静默丢弃；其它异常照抛。
> 更彻底的做法是"线程写队列 + 主线程 `after` 轮询队列"，
> 那才是真正线程安全的写法 —— 但那要改 6 个工作线程的结构，
> 本次先用最小改动把崩溃面收住。

### 一个容易写错的点（我第一版就写错了）

想"先查一下有没有注册过 `protocol` 再决定要不要注册"是**行不通的**：
Tk 的 `protocol(name)` 即使没注册过也会返回真值字符串
（实测返回 `'2331855702592destroy'`，那是 Tk 内置默认处理器）。
所以本模块改成：调用方显式传入 `on_close`，由 `install_shutdown_safety`
统一注册，并用实例标记 `_mati_close_installed` 保证只注册一次。
**不要**自己先 `self.protocol(...)` 再调 `install_shutdown_safety`。

---

## 4. 是否需要升级 customtkinter？

**不需要为了这个问题升级。**

- 当前安装 **5.2.2**；PyPI 上最新是 **6.0.0**（大版本跨度）。
- 我没有在 6.0.0 上验证过该缺陷是否已修复 —— 用一个未经验证的假设去动
  整个 GUI 的依赖，风险远大于收益：本项目**没有任何 GUI 自动化测试**
  （这是已知缺口），一次大版本升级的 API 变更只能靠人工点一遍，代价高。
- 本方案的三层防护**与版本无关**：即使 6.0.0 把 `ScalingTracker` 挪了位置，
  ①②会优雅降级（返回 False，不抛异常），③ 的异常钩子依然生效，
  红字照样消掉。

**建议**：保持 5.2.2 + 本防护。若确实想升级，做法是
**在独立虚拟环境里装 6.0.0 跑一遍 GUI 冒烟**，确认无回归再动生产环境 ——
可以单独安排，不要在修 bug 时顺手升。

---

## 5. 验证

`tests/test_tk_safety.py`（9 项，全通过），其中一项是**真窗口复现**：
创建 CTk 窗口 → 装保护 → `destroy()` → 直接调
`ScalingTracker.check_dpi_scaling()`，不再抛 TclError —— 这正是主人
控制台里那条红栈的触发路径。

另跑了一次端到端模拟（模拟 `NEBeduApp.on_close` 的完整流程：
取消定时器 → 停轮询 → 销毁 → `update()` 让排队回调有机会执行）：
**stderr 无输出 ✓**

全量：13 + 9 + 23 + 45 + 22 + 9 + 6 + 6 = **133 项测试通过**。

---

## 6. 顺带修掉的一个潜在 bug

`main_window.py` 多处使用 `logger`（内容加载告警等），但模块里**从未定义**
`logger`，也没有 `import logging`。一旦走到那些分支就是 `NameError`，
且多半被外层 `try/except` 吞掉 —— 表现为"告警莫名其妙不出现"。
已补上模块级 `logger = logging.getLogger(__name__)`。

同时 `cleanup_model()` 原来查的是 `bitnet_handler`（本项目早已换成 Qwen2.5，
`ModelHandler` 上根本没有这个属性，`hasattr` 恒为 False），
**1GB 的模型从来没被显式释放过**，只靠进程退出时 GC 兜底。
已改为调用现成的 `ModelHandler.cleanup()`。

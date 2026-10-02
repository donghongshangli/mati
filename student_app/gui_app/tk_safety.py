"""
Tk / customtkinter 关闭时序安全网。

背景（本项目实测，customtkinter 5.2.2 + Python 3.12 + Windows）
------------------------------------------------------------
关闭窗口后控制台会刷一屏红：

    Exception in Tkinter callback
    Traceback (most recent call last):
      File ".../tkinter/__init__.py", line ..., in callit
        func(*args)
      File ".../customtkinter/windows/widgets/scaling/scaling_tracker.py",
           line 178, in check_dpi_scaling
        if window.winfo_exists() and not window.state() == "iconic":
    _tkinter.TclError: can't invoke "winfo" command: application has been destroyed

根因链条（逐条核过库源码）
------------------------
1. `ScalingTracker.add_widget()` 在注册第一个 CTk 控件时用
   `window_root.after(100, cls.check_dpi_scaling)` 起了一个 **100ms 的轮询**，
   而且只在 `update_loop_running` 为 False 时启动 —— 之后**没有任何地方会停它**。
2. customtkinter 自己的 `CTk.destroy()`（`windows/ctk_tk.py:83`）只调用了
   `tkinter.Tk.destroy()` 和两个基类的 destroy，**没有**调用
   `ScalingTracker.remove_window()`。5.2.2 里 `remove_window` 存在，
   但在库内部**没有任何调用点** —— 窗口销毁后仍留在 `window_widgets_dict` 里。
3. 窗口销毁 → Tcl 解释器开始拆除 → 已经排队的 `after` 回调仍会执行一次。
   此时 `window.winfo_exists()` 面对的是已销毁的 application，于是抛 TclError。
4. `check_dpi_scaling` 的**第一段循环没有 try/except**（第二段重新排 `after`
   倒是包了 try/except），异常一路冒到 tkinter 的 `callit`，被
   `report_callback_exception` 打印成红色栈。

这是 customtkinter 的缺陷，不是本项目的用法问题。但它会让每次关闭窗口都
刷一屏红，对使用者是干扰，也会掩盖真正的异常。

本模块按优先级给三层处理：
  ① `stop_scaling_tracker()`  —— 关闭前主动停掉轮询（治本）
  ② `install_shutdown_safety()` —— 给库函数加保护，拦住已排队的回调
  ③ 兜底异常钩子            —— 最后一道，且**只吞"已销毁"这一类**，
                               其它异常照原样抛出，不掩盖真 bug

用法
----
    from student_app.gui_app.tk_safety import install_shutdown_safety, stop_scaling_tracker, safe_after

    app = NEBeduApp()
    install_shutdown_safety(app)     # mainloop() 之前
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()

    # on_close 里：
    def on_close(self):
        self._closing = True
        stop_scaling_tracker()
        self.destroy()

    # 工作线程里投递 UI 更新：
    safe_after(self, 0, display)     # 替代 self.after(0, display)
"""

from __future__ import annotations

import logging
import tkinter
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# TclError 里表示「Tk 应用/控件已销毁」的特征串。
# 只对这些静默；其它 TclError（比如真的用法错误）必须照常抛出。
_DESTROYED_MARKERS = (
    "application has been destroyed",
    "has been destroyed",
    "invalid command name",
    "can't invoke \"winfo\" command",
)


def is_destroyed_error(exc: BaseException) -> bool:
    """判断一个异常是否为「Tk 已销毁」这一类（可以安全忽略）。"""
    if not isinstance(exc, (tkinter.TclError, RuntimeError)):
        return False
    msg = str(exc).lower()
    return any(m.lower() in msg for m in _DESTROYED_MARKERS)


# ═══════════════════════════════════════════════════════════════════
# ① 主动停掉 customtkinter 的 DPI 轮询
# ═══════════════════════════════════════════════════════════════════
def stop_scaling_tracker() -> bool:
    """
    停掉 `ScalingTracker` 的 100ms DPI 缩放轮询。

    必须在 `destroy()` **之前**调用：轮询是挂在窗口上的 `after` 定时器，
    窗口一旦销毁就再也没有机会把它摘掉了。

    做法是把轮询依赖的两个状态清掉：
      · `update_loop_running = False` —— 防止 `add_widget()` 再起一个
      · 清空 `window_widgets_dict` —— 下一次回调遍历不到任何窗口，
        会在末尾自己把 `update_loop_running` 置回 False 并退出

    Returns:
        是否成功（找不到 customtkinter / 版本结构变化时返回 False，不抛异常）
    """
    try:
        from customtkinter.windows.widgets.scaling.scaling_tracker import ScalingTracker
    except Exception as e:  # noqa: BLE001 - customtkinter 缺失时不应影响主流程
        logger.debug(f"未加载 customtkinter ScalingTracker：{e}")
        return False

    try:
        n = len(getattr(ScalingTracker, "window_widgets_dict", {}) or {})
        ScalingTracker.update_loop_running = False
        ScalingTracker.window_widgets_dict.clear()
        ScalingTracker.window_dpi_scaling_dict.clear()
        if n:
            logger.debug(f"已停止 customtkinter DPI 轮询（注销 {n} 个窗口）")
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning(f"停止 DPI 轮询失败（不影响退出）：{e}")
        return False


# ═══════════════════════════════════════════════════════════════════
# ② 给库函数加保护（拦住"已经排队、停不掉"的那一次回调）
# ═══════════════════════════════════════════════════════════════════
def _guard_scaling_tracker() -> bool:
    """
    给 `ScalingTracker.check_dpi_scaling` 套一层 TclError 保护。

    即使 ① 已停掉轮询，也可能有**一个已经排队**的 `after` 回调在窗口销毁后
    被执行 —— 那一次仍会炸。这里把库函数换成带保护的版本：
      · 捕获到「已销毁」类 TclError → 顺手把轮询彻底关掉，静默返回
      · 其它异常原样抛出，绝不掩盖真 bug

    Returns:
        是否成功安装（已安装过 / 库结构变化时返回 False）
    """
    try:
        from customtkinter.windows.widgets.scaling import scaling_tracker as _st
        cls = _st.ScalingTracker
    except Exception as e:  # noqa: BLE001
        logger.debug(f"未加载 ScalingTracker，跳过保护：{e}")
        return False

    if getattr(cls, "_mati_guarded", False):
        return True

    original = cls.check_dpi_scaling      # 已绑定到类的 classmethod

    def patched(*_args, **_kwargs):
        try:
            return original()
        except tkinter.TclError as e:
            if not is_destroyed_error(e):
                raise
            # Tk 已销毁：关掉轮询，避免后续回调继续炸
            try:
                cls.update_loop_running = False
                cls.window_widgets_dict.clear()
                cls.window_dpi_scaling_dict.clear()
            except Exception:  # noqa: BLE001
                pass
            logger.debug("Tk 已销毁，已终止 DPI 缩放轮询（忽略一次 TclError）")
            return None

    # 存成普通函数：通过类访问时不会被二次绑定，调用签名保持零参
    cls.check_dpi_scaling = patched
    cls._mati_guarded = True
    return True


# ═══════════════════════════════════════════════════════════════════
# ③ 兜底异常钩子
# ═══════════════════════════════════════════════════════════════════
def _install_callback_exception_hook() -> bool:
    """
    兜底：让 `report_callback_exception` 忽略「已销毁」类的 TclError。

    改的是 `tkinter.Misc`（所有控件的基类），因此任何控件上的回调异常都会经过。
    **只吞"已销毁"这一类**，其余异常交回原实现打印 —— 我们不能为了消掉
    一屏红字，把真正的 UI bug 也一起藏了。
    """
    # Python 3.12 里 `report_callback_exception` 定义在 **tkinter.Tk** 上，
    # 不在 Misc 上。调用链是
    #   Misc._report_exception() → self._root().report_callback_exception(...)
    # 所以改 Tk 这个类就能覆盖到所有控件（包括 CTk 子窗口）。
    target = tkinter.Tk if hasattr(tkinter.Tk, "report_callback_exception") else tkinter.Misc
    if getattr(target, "_mati_hook_installed", False):
        return True

    original = target.report_callback_exception

    def hook(self, exc, val, tb):
        if is_destroyed_error(val):
            return None
        return original(self, exc, val, tb)

    target.report_callback_exception = hook
    target._mati_hook_installed = True
    target._mati_original_report = original
    return True


def install_shutdown_safety(root: Optional[tkinter.Misc] = None,
                            on_close: Optional[Callable[[], Any]] = None) -> None:
    """
    一次性安装关闭时序保护。**应在构造窗口之后、mainloop() 之前调用。**

    做三件事：
      1. 给 `ScalingTracker.check_dpi_scaling` 加 TclError 保护
      2. 安装兜底的 `report_callback_exception` 钩子
      3. 注册 `WM_DELETE_WINDOW`

    关于 3：**必须显式把关闭回调传进来**（`on_close=`），不要自己先调
    `root.protocol(...)` 再调本函数。
    原因：Tk 的 `protocol(name)` 即使没注册过也会返回一个真值字符串
    （实测返回 `'2331855702592destroy'`，那是 Tk 内置默认处理器），
    所以"查一下有没有处理器再决定要不要注册"这种写法永远不会生效。
    这里改成：调用方给什么就注册什么，并用实例标记保证只注册一次。

    Args:
        root: 主窗口。为 None 时只做 1、2。
        on_close: 关闭窗口时执行的回调。不传则用一个默认实现
                 （停轮询 → 销毁；不做业务清理）。
    """
    _guard_scaling_tracker()
    _install_callback_exception_hook()

    if root is None:
        return
    if getattr(root, "_mati_close_installed", False):
        return

    if on_close is None:
        def on_close():                                # noqa: F811 - 兜底默认实现
            try:
                stop_scaling_tracker()
            finally:
                try:
                    root.destroy()
                except Exception as e:  # noqa: BLE001
                    if not is_destroyed_error(e):
                        logger.warning(f"窗口销毁异常：{e}")

    try:
        root.protocol("WM_DELETE_WINDOW", on_close)
        root._mati_close_installed = True
    except Exception as e:  # noqa: BLE001
        logger.debug(f"注册 WM_DELETE_WINDOW 失败（忽略）：{e}")


# ═══════════════════════════════════════════════════════════════════
# ④ 跨线程投递 UI 更新
# ═══════════════════════════════════════════════════════════════════
def safe_after(widget: Any, ms: int, callback: Callable[[], Any]) -> Optional[str]:
    """
    从工作线程安全地投递一次 UI 更新，替代直接 `widget.after(...)`。

    Tkinter **不是线程安全的**：后台线程直接调 `after` 本身就已越界，
    在窗口关闭阶段更会抛 `TclError: application has been destroyed`。
    这里统一处理：
      · 窗口已销毁 / 取不到解释器 → 静默丢弃，不再投递
      · 捕获到「已销毁」类 TclError → 静默
      · 其它异常照原样抛出，不掩盖真 bug

    Returns:
        `after` 的 timer id；未投递时返回 None。
    """
    if widget is None:
        return None
    try:
        # winfo_exists 在解释器已销毁时会抛 TclError（正是要挡的那种）
        if not widget.winfo_exists():
            return None
        return widget.after(int(ms), callback)
    except Exception as e:  # noqa: BLE001
        if not is_destroyed_error(e):
            raise
        return None

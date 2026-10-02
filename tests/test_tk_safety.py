#!/usr/bin/env python3
"""
tk_safety 回归测试：Tk / customtkinter 关闭时序安全网。

覆盖两类用例：
  A. 纯逻辑（不需要窗口，任何时候都能跑）
       · is_destroyed_error 的判定边界
       · safe_after 在"窗口已销毁"时静默、其它异常照抛
       · stop_scaling_tracker 的降级行为
  B. 真实窗口（能创建 Tk 时跑，不能则跳过）
       · 装好保护后，**在窗口销毁之后**调用 ScalingTracker.check_dpi_scaling()
         不再抛 TclError —— 这正是主人控制台里那条红色栈的复现路径
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tkinter  # noqa: E402

from student_app.gui_app.tk_safety import (  # noqa: E402
    install_shutdown_safety,
    is_destroyed_error,
    safe_after,
    stop_scaling_tracker,
)

_PASS = []
_FAIL = []
_SKIP = []


def check(name, fn):
    try:
        fn()
        _PASS.append(name)
    except AssertionError as e:
        _FAIL.append((name, str(e) or "断言失败"))
    except Exception as e:  # noqa: BLE001
        _FAIL.append((name, f"{type(e).__name__}: {e}"))


# ═══════════════════════════════════════════════════ A. 纯逻辑
def test_is_destroyed_error_true_cases():
    msgs = [
        "can't invoke \"winfo\" command: application has been destroyed",
        "invalid command name \".!ctkframe\"",
        "main thread is not in main loop",
    ]
    for m in msgs[:2]:                       # 前两条必须判定为"已销毁"
        assert is_destroyed_error(tkinter.TclError(m)), m
    # 非空串、非 TclError 一律不算
    assert not is_destroyed_error(RuntimeError("boom")), "普通异常不该被当已销毁"
    assert not is_destroyed_error(ValueError("x")), "普通异常不该被当已销毁"


def test_is_destroyed_error_does_not_hide_real_errors():
    """关键：不能为了消红字把真异常也吞了。"""
    assert not is_destroyed_error(tkinter.TclError("bad screen distance"))
    assert not is_destroyed_error(tkinter.TclError("unknown option -foo"))


class _DeadWidget:
    """模拟"已销毁的窗口"：winfo_exists / after 都会抛 TclError。"""

    def winfo_exists(self):
        raise tkinter.TclError(
            "can't invoke \"winfo\" command: application has been destroyed")

    def after(self, ms, cb):
        raise tkinter.TclError(
            "can't invoke \"after\" command: application has been destroyed")


class _LiveWidget:
    def __init__(self):
        self.scheduled = None

    def winfo_exists(self):
        return True

    def after(self, ms, cb):
        self.scheduled = (ms, cb)
        return "timer-1"


def test_safe_after_silent_on_destroyed_widget():
    assert safe_after(_DeadWidget(), 0, lambda: None) is None, "已销毁时应静默返回 None"


def test_safe_after_none_widget():
    assert safe_after(None, 0, lambda: None) is None


def test_safe_after_passes_through_normally():
    w = _LiveWidget()
    fn = lambda: None  # noqa: E731
    assert safe_after(w, 0, fn) == "timer-1"
    assert w.scheduled[0] == 0 and w.scheduled[1] is fn


def test_safe_after_does_not_swallow_real_errors():
    class _BrokenWidget:
        def winfo_exists(self):
            return True

        def after(self, ms, cb):
            raise tkinter.TclError("unknown option -nope")

    try:
        safe_after(_BrokenWidget(), 0, lambda: None)
    except tkinter.TclError as e:
        assert "unknown option" in str(e), e
    else:
        raise AssertionError("真实 TclError 被吞掉了 —— 这会把 UI bug 藏起来")


def test_stop_scaling_tracker_returns_bool_and_never_raises():
    # 真实环境下返回 True；库结构变化时返回 False，都不该抛异常
    r = stop_scaling_tracker()
    assert isinstance(r, bool), type(r)


# ═══════════════════════════════════════════════════ B. 真实窗口
def _make_root():
    """尝试创建一个真实窗口；无显示环境时返回 None（用例跳过）。"""
    try:
        import customtkinter as ctk
        root = ctk.CTk()
        root.withdraw()
        return root
    except Exception:  # noqa: BLE001
        try:
            root = tkinter.Tk()
            root.withdraw()
            return root
        except Exception:  # noqa: BLE001
            return None


def test_check_dpi_scaling_survives_after_destroy():
    """
    复现主人遇到的那条红色栈：
    窗口销毁后，customtkinter 已排队的 DPI 轮询回调仍会执行一次，
    其中 window.winfo_exists() 抛 TclError。
    装了保护之后，这次调用必须**静默返回**，且轮询被彻底关掉。
    """
    try:
        from customtkinter.windows.widgets.scaling.scaling_tracker import ScalingTracker
    except Exception as e:  # noqa: BLE001
        _SKIP.append(("check_dpi_scaling 存活", f"无 customtkinter：{e}"))
        return

    root = _make_root()
    if root is None:
        _SKIP.append(("check_dpi_scaling 存活", "无法创建窗口（无显示环境）"))
        return

    try:
        install_shutdown_safety(root)
        assert getattr(ScalingTracker, "_mati_guarded", False), "保护未安装"

        # 让 tracker 登记这个窗口（CTk 已经通过控件注册过；保险起见再确认）
        if root not in ScalingTracker.window_widgets_dict:
            ScalingTracker.window_widgets_dict[root] = []
            ScalingTracker.window_dpi_scaling_dict[root] = 1.0

        root.destroy()                       # ← 销毁，Tcl 应用拆除

        # 这一句在未修复时会抛 TclError: application has been destroyed
        ScalingTracker.check_dpi_scaling()

        # 保护生效后应把轮询彻底关掉，避免后续回调继续炸
        assert ScalingTracker.update_loop_running is False, "轮询未停止"
    finally:
        try:
            root.destroy()
        except Exception:  # noqa: BLE001
            pass


def test_global_hook_ignores_only_destroyed():
    """
    兜底钩子只吞"已销毁"，其它异常必须交回原实现（打印到 stderr）。

    注意：钩子在**安装时**就把原实现捕获进闭包了，所以测试要验证真实行为
    （stderr 上有无输出），而不是去替换 `_mati_original_report` —— 那不会影响
    已经装好的钩子（我第一版测试就是在这里写错的）。
    """
    import contextlib
    import io

    install_shutdown_safety(None)
    assert getattr(tkinter.Tk, "_mati_hook_installed", False), "钩子未安装"

    def _report(msg):
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            tkinter.Tk.report_callback_exception(None, tkinter.TclError,
                                                 tkinter.TclError(msg), None)
        return buf.getvalue()

    assert _report("application has been destroyed") == "", \
        "已销毁类异常不应再打印（否则关闭窗口仍会刷红）"

    out = _report("bad screen distance")
    assert "bad screen distance" in out, \
        f"真实 TclError 被吞了 —— 这会把 UI bug 藏起来：{out!r}"


def main():
    checks = [
        ("判定：已销毁类 TclError", test_is_destroyed_error_true_cases),
        ("判定：不吞真实错误", test_is_destroyed_error_does_not_hide_real_errors),
        ("safe_after：已销毁静默", test_safe_after_silent_on_destroyed_widget),
        ("safe_after：None 控件", test_safe_after_none_widget),
        ("safe_after：正常投递", test_safe_after_passes_through_normally),
        ("safe_after：不吞真实错误", test_safe_after_does_not_swallow_real_errors),
        ("stop_scaling_tracker 降级", test_stop_scaling_tracker_returns_bool_and_never_raises),
        ("销毁后 DPI 回调存活", test_check_dpi_scaling_survives_after_destroy),
        ("兜底钩子只吞已销毁", test_global_hook_ignores_only_destroyed),
    ]
    for name, fn in checks:
        check(name, fn)

    print(f"\n{'='*64}\ntk_safety 回归测试\n{'='*64}")
    for name in _PASS:
        print(f"  PASS  {name}")
    for name, reason in _SKIP:
        print(f"  SKIP  {name}  （{reason}）")
    for name, reason in _FAIL:
        print(f"  FAIL  {name}\n        {reason}")
    print("-" * 64)
    print(f"  共 {len(_PASS)} 通过 / {len(_FAIL)} 失败 / {len(_SKIP)} 跳过")
    return 1 if _FAIL else 0


if __name__ == "__main__":
    sys.exit(main())

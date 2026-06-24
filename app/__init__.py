# superquant app — Hikyuu 驱动的量化短线交易系统

import atexit
import signal
import sys

import hikyuu


def _safe_quit():
    """安全退出: 通知 Hikyuu C++ 核心释放线程池等资源。

    Hikyuu 2.8.0 已知问题: GlobalStealThreadPool 在 Python 解释器退出时
    可能访问已释放的互斥锁 → SIGSEGV。hku_cleanup() 主动释放避免此竞态。
    """
    try:
        hikyuu.hku_cleanup()
    except Exception:
        pass


def _sig_handler(sig, frame):
    """兜底: 即使 SIGSEGV 发生，也走 sys.exit 而不是 core dump。"""
    sys.exit(0)


atexit.register(_safe_quit)
signal.signal(signal.SIGSEGV, _sig_handler)

# -*- coding: utf-8 -*-
"""
OpenSees Sandbox Worker for MGHR Reward Evaluation.

Executes LLM-generated OpenSeesPy code in an isolated subprocess with stdout/stderr
capture, error line tracking, and environment cleanup. Used by process_pool.py.
"""

import sys
import io
import math
import traceback
import os

_DEBUG = False
_PRINT_ENV_ONCE = False


def _debug_print(*args, **kwargs):
    if _DEBUG:
        print(*args, **kwargs, file=sys.__stderr__)
        sys.__stderr__.flush()


def _worker_func(code: str) -> dict:
    global _PRINT_ENV_ONCE

    # 诊断信息
    worker_pid = os.getpid()
    
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    real_stdout, real_stderr = sys.stdout, sys.stderr
    sys.stdout = stdout_capture
    sys.stderr = stderr_capture

    result = {
        "success": False,
        "stdout": "",
        "stderr": "",
        "exception_type": None,
        "error_line": 0,
        "total_lines": 0,
        "worker_pid": worker_pid,  # 添加进程ID
    }

    try:
        # 导入依赖
        try:
            import openseespy.opensees as ops
            import numpy as np
            
            if _DEBUG and not _PRINT_ENV_ONCE:
                _debug_print(f"[Worker-{worker_pid}] Imports OK")
                _PRINT_ENV_ONCE = True
                
        except Exception as e:
            result["exception_type"] = "EnvInitError"
            result["stderr"] = f"Import failed in worker {worker_pid}: {e}\n{traceback.format_exc()}"
            return result

        # 清理环境
        try:
            ops.wipe()
            if hasattr(ops, "setNumThread"):
                ops.setNumThread(1)
        except Exception as e:
            if _DEBUG:
                _debug_print(f"[Worker-{worker_pid}] Wipe warning: {e}")

        # 准备代码
        lines = code.strip().splitlines()
        result["total_lines"] = len(lines)
        code_to_run = "# -*- coding: utf-8 -*-\n" + code
        compiled_code = compile(code_to_run, "<string>", "exec")

        exec_globals = {
            "math": math,
            "ops": ops,
            "opensees": ops,
            "np": np,
            "numpy": np,
        }

        # 执行
        exec(compiled_code, exec_globals)
        result["success"] = True

    except Exception as e:
        exc_type, exc_value, exc_tb = sys.exc_info()
        result["exception_type"] = exc_type.__name__ if exc_type else "Exception"
        result["stderr"] = f"[Worker-{worker_pid}] {exc_value}"

        # 定位错误行
        tb = exc_tb
        last_line = 0
        while tb:
            if tb.tb_frame.f_code.co_filename == "<string>":
                last_line = tb.tb_lineno
            tb = tb.tb_next
        result["error_line"] = max(0, last_line - 1)

        if _DEBUG:
            result["stderr"] += "\n" + traceback.format_exc()

    finally:
        try:
            ops.wipe()
        except:
            pass

        result["stdout"] = stdout_capture.getvalue()
        captured_err = stderr_capture.getvalue()
        if captured_err and not result["stderr"]:
            result["stderr"] = captured_err

        sys.stdout = real_stdout
        sys.stderr = real_stderr

    return result

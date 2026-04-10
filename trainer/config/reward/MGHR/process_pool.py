# -*- coding: utf-8 -*-
"""
Process Pool Manager for OpenSees Code Execution.

Manages a ProcessPoolExecutor with spawn context for safe multiprocess execution
of OpenSeesPy code. Features automatic pool health monitoring, error-threshold-based
recreation, and timeout handling.
"""

import os
import sys
import atexit
import multiprocessing
import threading
import time
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError

_THIS_DIR = os.path.dirname(__file__)
if _THIS_DIR not in sys.path:
    sys.path.append(_THIS_DIR)

from opensees_worker import _worker_func

_POOL = None
_POOL_LOCK = threading.Lock()
_POOL_CREATION_TIME = None
_POOL_MAX_AGE = 600
_POOL_ERROR_COUNT = 0
_MAX_ERROR_THRESHOLD = 10


def _create_fresh_pool() -> ProcessPoolExecutor:
    """创建新的进程池"""
    try:
        ctx = multiprocessing.get_context("spawn")
    except ValueError:
        ctx = None
    
    max_workers = min(16, os.cpu_count() or 4)
    pool = ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx)
    
    print(f"[ProcessPool] Created new pool with {max_workers} workers", flush=True)
    
    return pool


def _should_recreate_pool() -> bool:
    """检查是否需要重建进程池"""
    global _POOL_CREATION_TIME, _POOL_ERROR_COUNT
    
    if _POOL is None:
        return True
    
    if _POOL_CREATION_TIME and (time.time() - _POOL_CREATION_TIME) > _POOL_MAX_AGE:
        print("[ProcessPool] Pool expired, will recreate", flush=True)
        return True
    
    if _POOL_ERROR_COUNT >= _MAX_ERROR_THRESHOLD:
        print(f"[ProcessPool] Error threshold ({_POOL_ERROR_COUNT}), will recreate", flush=True)
        return True
    
    return False


def _get_pool() -> ProcessPoolExecutor:
    """获取进程池（带健康检查）"""
    global _POOL, _POOL_CREATION_TIME, _POOL_ERROR_COUNT
    
    with _POOL_LOCK:
        if _should_recreate_pool():
            if _POOL is not None:
                try:
                    _POOL.shutdown(wait=True, cancel_futures=True)
                except Exception as e:
                    print(f"[ProcessPool] Shutdown error: {e}", flush=True)
                _POOL = None
            
            _POOL = _create_fresh_pool()
            _POOL_CREATION_TIME = time.time()
            _POOL_ERROR_COUNT = 0
        
        return _POOL


def _shutdown_pool():
    """清理进程池"""
    global _POOL
    with _POOL_LOCK:
        if _POOL is not None:
            try:
                _POOL.shutdown(wait=True, cancel_futures=True)
                print("[ProcessPool] Shutdown complete", flush=True)
            except Exception as e:
                print(f"[ProcessPool] Shutdown error: {e}", flush=True)
            finally:
                _POOL = None


atexit.register(_shutdown_pool)


def run_opensees_multiprocess(code: str, timeout: float) -> dict:
    """
    在子进程中执行 OpenSees 代码（带重试和错误恢复）
    """
    global _POOL_ERROR_COUNT, _POOL  # ✅ 关键修复：声明 global _POOL
    
    if not code or not code.strip():
        return {
            "success": False,
            "stdout": "",
            "stderr": "Empty code",
            "exception_type": "EmptyCode",
            "error_line": 0,
            "total_lines": 1,
        }

    max_retries = 2
    for attempt in range(max_retries):
        try:
            pool = _get_pool()
            future = pool.submit(_worker_func, code)
            result = future.result(timeout=timeout)
            
            # 成功执行，递减错误计数
            if result.get("success", False):
                with _POOL_LOCK:
                    _POOL_ERROR_COUNT = max(0, _POOL_ERROR_COUNT - 1)
            
            return result
            
        except FuturesTimeoutError:
            with _POOL_LOCK:
                _POOL_ERROR_COUNT += 1
            
            return {
                "success": False,
                "stdout": "",
                "stderr": "Execution Timeout",
                "exception_type": "TimeoutError",
                "error_line": 0,
                "total_lines": len(code.splitlines()),
            }
            
        except Exception as e:
            with _POOL_LOCK:
                _POOL_ERROR_COUNT += 1
            
            error_msg = f"Process Pool Error (attempt {attempt+1}/{max_retries}): {e}"
            
            # 最后一次尝试
            if attempt == max_retries - 1:
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": error_msg,
                    "exception_type": "SystemError",
                    "error_line": 0,
                    "total_lines": len(code.splitlines()),
                }
            
            # 重试逻辑：强制重建进程池
            print(f"[ProcessPool] {error_msg}, forcing pool recreation...", flush=True)
            time.sleep(0.5)
            
            with _POOL_LOCK:
                if _POOL is not None:
                    try:
                        _POOL.shutdown(wait=False, cancel_futures=True)
                    except:
                        pass
                    _POOL = None
                _POOL_ERROR_COUNT = _MAX_ERROR_THRESHOLD  # 强制触发重建

    return {
        "success": False,
        "stdout": "",
        "stderr": "Unknown error after retries",
        "exception_type": "SystemError",
        "error_line": 0,
        "total_lines": 0,
    }

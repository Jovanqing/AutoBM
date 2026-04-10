# -*- coding: utf-8 -*-
"""
Multi-Granularity Hybrid Reward (MGHR) for SPC-GRPO Training.

This module implements the reward function R(o) = w_fmt * r_fmt + w_ast * r_ast + w_exec * r_exec
as described in Section 5.3 of the paper:

    "Rethinking Scientific Modeling: Toward Physically Consistent
     and Simulation-Executable Programmatic Generation" (KDD '26)

Components:
    - Format Compliance Reward (r_fmt, w=0.05): Enforces <think>/<answer> structure with code blocks.
    - Logical Completeness via Hierarchical AST (r_ast, w=0.25): Three-tiered OpenSeesPy API coverage.
    - Sandbox-Based Physical Consistency (r_exec, w=0.70): Execution in OpenSees sandbox with
      progress-based partial credit and period-error-based grading.

The reward function entry point is `compute_score(solution_str, ground_truth, **kwargs)`.
"""

import re
import ast
import hashlib
import builtins
import textwrap
from functools import lru_cache
from typing import Any, Optional
import atexit
import io
import math
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)


# 关键：从独立模块引入多进程执行函数
from process_pool import run_opensees_multiprocess

# =========================================================
# 1. 全局配置 & 正则定义
# =========================================================
REWARD_WEIGHTS = {
    "format": 0.05,
    "ast": 0.25,
    "exec": 0.70,
}

REQUIRED_APIS = {
    "wipe",
    "model",
    "node",
    "fix",
    "mass",
    "nodeDisp",
    "nodeEigenvector",
    "geomTransf",
    "element",
    "eleLoad",
    "eleForce",
    "timeSeries",
    "pattern",
    "load",
    "loadConst",
    "constraints",
    "numberer",
    "system",
    "test",
    "algorithm",
    "integrator",
    "analysis",
    "analyze",
    "eigen",
}

MAX_CODE_LENGTH = 80000
TIMEOUT_OPENSEES = 8.0
RELATIVE_TOLERANCE = 0.10


class CompiledPatterns:
    ANSWER_TAGS = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
    THINK_TAGS = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)
    PYTHON_CODE_BLOCK = re.compile(r"```python\s*(.*?)\s*```", re.DOTALL)
    PYTHON_MARKER = re.compile(r"```(?:python)?", re.IGNORECASE)

    STRICT_FORMAT = re.compile(
        r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$", re.DOTALL
    )
    LOOSE_FORMAT = re.compile(r"<think>.*?</think>.*<answer>.*?</answer>", re.DOTALL)
    HAS_ANSWER = re.compile(r"<answer>.*?</answer>", re.DOTALL)

    METRIC_PATTERNS = [
        re.compile(
            r"(?i)T[1₁]\s*[=＝:：]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        ),
        re.compile(
            r"基本周期[^0-9\n]*?[=＝:：]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        ),
        re.compile(
            r"(?:Mode\s*1|第\s*1\s*阶)[^0-9\n]*?T\s*[=＝]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        ),
        re.compile(
            r"(?i)Period\s*[=＝:：]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        ),
    ]

    COMPLIANCE_KEYWORD = re.compile(
        r"(?<![未不])(?:通过|满足|Satisfied|Pass|Verified)"
    )


# =========================================================
# 2. AST 静态分析与辅助函数
# =========================================================
class AdvancedStaticAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.ops_aliases = {"ops", "opensees"}
        self.found_apis = set()
        self.defined_vars = set(dir(builtins)) | {"np", "numpy", "math"}
        self.local_scopes = []
        self.undefined_errors = []
        self.node_count = 0

    def _is_defined(self, name: str) -> bool:
        if self.local_scopes and name in self.local_scopes[-1]:
            return True
        return name in self.defined_vars

    def visit_Import(self, node):
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name.split(".")[0]
            self.defined_vars.add(name)
            if "openseespy" in alias.name and alias.asname:
                self.ops_aliases.add(alias.asname)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module:
            self.defined_vars.add(node.module.split(".")[0])
        for alias in node.names:
            self.defined_vars.add(alias.asname if alias.asname else alias.name)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self.defined_vars.add(node.name)
        self.local_scopes.append(set(a.arg for a in node.args.args))
        self.generic_visit(node)
        self.local_scopes.pop()

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Store):
            if self.local_scopes:
                self.local_scopes[-1].add(node.id)
            else:
                self.defined_vars.add(node.id)
        elif isinstance(node.ctx, ast.Load):
            if not self._is_defined(node.id):
                self.undefined_errors.append(node.id)

    def visit_Call(self, node):
        self.node_count += 1
        if self.node_count > 20000:
            return
        if isinstance(node.func, ast.Attribute) and isinstance(
            node.func.value, ast.Name
        ):
            if (
                node.func.value.id in self.ops_aliases
                and node.func.attr in REQUIRED_APIS
            ):
                self.found_apis.add(node.func.attr)
        elif isinstance(node.func, ast.Name) and node.func.id in REQUIRED_APIS:
            self.found_apis.add(node.func.id)
        self.generic_visit(node)


@lru_cache(maxsize=1024)
def extract_python_code(text: str) -> str:
    if not text:
        return ""
    raw = text
    match_xml = CompiledPatterns.ANSWER_TAGS.search(raw)
    if match_xml:
        raw = match_xml.group(1)

    match_md = CompiledPatterns.PYTHON_CODE_BLOCK.search(raw)
    if match_md:
        code = match_md.group(1)
    else:
        code = CompiledPatterns.PYTHON_MARKER.sub("", raw).replace("```", "")

    code = textwrap.dedent(code)
    lines = [
        ln
        for ln in code.splitlines()[:5000]
        if not ln.strip().lower().startswith("python")
    ]
    code = "\n".join(lines).replace("：", ":").strip()

    if len(code) > MAX_CODE_LENGTH:
        code = code[:MAX_CODE_LENGTH]
        last_newline = code.rfind("\n")
        if last_newline != -1:
            code = code[:last_newline]

    return code


def extract_metric_from_stdout(stdout: str) -> Optional[float]:
    if not stdout:
        return None
    for p in CompiledPatterns.METRIC_PATTERNS:
        m = p.search(stdout)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
    return None


class RewardManagerV4:
    def __init__(self):
        self._ast_cache = {}
        self.TIER1 = {
            "wipe",
            "model",
            "node",
            "fix",
            "mass",
            "geomTransf",
            "element",
            "timeSeries",
        }
        self.TIER2 = {
            "pattern",
            "load",
            "loadConst",
            "constraints",
            "numberer",
            "system",
            "test",
            "algorithm",
            "integrator",
            "analysis",
            "analyze",
        }
        self.TIER3 = {
            "eigen",
            "nodeEigenvector",
            "eleForce",
            "eleLoad",
            "nodeDisp",
        }
        self.tier_weights = {"t1": 0.40, "t2": 0.40, "t3": 0.20}

    def _calc_detailed_ast_reward(self, code: str) -> float:
        if not code:
            return 0.0
        code_hash = hashlib.md5(code.encode("utf-8")).hexdigest()
        if code_hash in self._ast_cache:
            return self._ast_cache[code_hash]

        try:
            tree = ast.parse(code)
            analyzer = AdvancedStaticAnalyzer()
            analyzer.visit(tree)

            found = analyzer.found_apis
            t1_score = len(self.TIER1 & found) / max(1, len(self.TIER1))
            t2_score = len(self.TIER2 & found) / max(1, len(self.TIER2))
            t3_score = len(self.TIER3 & found) / max(1, len(self.TIER3))

            final_ast = (
                self.tier_weights["t1"] * t1_score
                + self.tier_weights["t2"] * t2_score
                + self.tier_weights["t3"] * t3_score
            )

            if REQUIRED_APIS.issubset(found):
                final_ast = max(final_ast, 0.99)

            quality_deduction = min(len(analyzer.undefined_errors) * 0.05, 0.5)
            final_ast = max(0.0, final_ast - quality_deduction)

        except Exception:
            final_ast = 0.0

        self._ast_cache[code_hash] = final_ast
        return final_ast

    def _calc_execution_reward(
        self, exec_result: dict, gt_val: Optional[float], found_apis: set
    ) -> float:
        success = exec_result.get("success", False)
        stdout = exec_result.get("stdout", "") or ""
        exc_type = exec_result.get("exception_type", "")

        if "Timeout" in str(exc_type) or exc_type == "TimeoutError":
            return 0.20

        if "ImportError" in str(exc_type) or str(exc_type) == "EnvInitError":
            return 0.25

        if not success:
            error_line = exec_result.get("error_line", 0)
            total_lines = max(1, exec_result.get("total_lines", 1))
            progress = min(error_line / total_lines, 1.0)
            return 0.30 + 0.25 * progress

        score = 0.50
        if CompiledPatterns.COMPLIANCE_KEYWORD.search(stdout):
            score += 0.05

        pred_val = extract_metric_from_stdout(stdout)
        if pred_val is None:
            return min(0.65, score)
        else:
            if gt_val is None:
                return 0.80
            try:
                rel_err = abs(pred_val - gt_val) / (abs(gt_val) + 1e-9)
            except Exception:
                rel_err = 1.0

            if rel_err <= RELATIVE_TOLERANCE:
                return 1.0
            elif rel_err <= 0.10:
                return 0.95
            elif rel_err <= 0.20:
                return 0.90
            elif rel_err <= 0.40:
                return 0.80
            else:
                return 0.70


_MANAGER = RewardManagerV4()
atexit.register(lambda: _MANAGER._ast_cache.clear())


# =========================================================
# 3. 主入口函数：compute_score
# =========================================================
def compute_score(solution_str: str, ground_truth: Any, **kwargs) -> float:
    # ground_truth 解析
    try:
        gt_val = (
            float(str(ground_truth).strip("[]'\""))
            if ground_truth is not None and str(ground_truth).strip() != ""
            else None
        )
    except Exception:
        gt_val = None

    # ---------- Format Reward ----------
    has_code_block = bool(
        CompiledPatterns.PYTHON_CODE_BLOCK.search(solution_str)
    )
    code_in_think = False
    think_match = CompiledPatterns.THINK_TAGS.search(solution_str)
    if think_match:
        if "```" in think_match.group(1):
            code_in_think = True

    if code_in_think:
        r_fmt = 0.0
    elif CompiledPatterns.STRICT_FORMAT.match(solution_str):
        r_fmt = 1.0
    elif CompiledPatterns.LOOSE_FORMAT.search(solution_str):
        r_fmt = 0.7
    elif has_code_block:
        r_fmt = 0.4
    elif CompiledPatterns.HAS_ANSWER.search(solution_str):
        r_fmt = 0.2
    else:
        r_fmt = 0.05

    # ---------- AST Reward ----------
    code = extract_python_code(solution_str)
    r_ast = _MANAGER._calc_detailed_ast_reward(code)

    # ---------- Execution Reward ----------
    r_exec = 0.0
    found_apis = set()
    res = {} # 初始化为空，防止空代码导致日志报错
    
    try:
        tree = ast.parse(code) if code else None
        if tree:
            analyzer = AdvancedStaticAnalyzer()
            analyzer.visit(tree)
            found_apis = analyzer.found_apis
    except Exception:
        pass

    if code:
        res = run_opensees_multiprocess(code, TIMEOUT_OPENSEES)
        r_exec = _MANAGER._calc_execution_reward(res, gt_val, found_apis)

        # 如果执行成绩很好，但没用到所有 REQUIRED_APIS，略微降权
        if r_exec >= 0.9 and not REQUIRED_APIS.issubset(found_apis):
            r_exec = min(r_exec, 0.85)
    else:
        r_exec = 0.0

    final = (
        REWARD_WEIGHTS["format"] * r_fmt
        + REWARD_WEIGHTS["ast"] * r_ast
        + REWARD_WEIGHTS["exec"] * r_exec
    )

    # =========================================================
    # [新增] 强制实时打印奖励组成 (带颜色高亮)
    # =========================================================
    try:
        pid = os.getpid()
        # ANSI 颜色: \033[93m=黄, \033[96m=青, \033[92m=绿, \033[91m=红, \033[0m=复位
        separator = "=" * 50
        
        # 尝试提取预测值用于展示
        pred_display = "N/A"
        if res and res.get("stdout"):
             val = extract_metric_from_stdout(res["stdout"])
             if val is not None:
                 pred_display = f"{val:.4f}"
        
        # 提取错误类型
        err_type = res.get("exception_type", "None") if res else "NoCode"
        success_flag = res.get('success', False) if res else False

        log_msg = (
            f"\n\033[93m{separator}\033[0m\n"
            f"\033[96m[GRPO-LOG] PID:{pid}\033[0m\n"
            f"GT: {gt_val} | Pred: {pred_display} | Success: {success_flag} | Err: {err_type}\n"
            f"Scores -> Format: \033[92m{r_fmt:.2f}\033[0m | "
            f"AST: \033[92m{r_ast:.2f}\033[0m | "
            f"Exec: \033[92m{r_exec:.2f}\033[0m\n"
            f"Calc   -> {REWARD_WEIGHTS['format']}*{r_fmt:.2f} + "
            f"{REWARD_WEIGHTS['ast']}*{r_ast:.2f} + "
            f"{REWARD_WEIGHTS['exec']}*{r_exec:.2f}\n"
            f"FINAL  -> \033[91m{final:.5f}\033[0m\n"
            f"\033[93m{separator}\033[0m\n"
        )
        
        # 打印并强制刷新缓冲区
        print(log_msg)
        sys.stdout.flush()
        
    except Exception:
        pass

    return float(final)
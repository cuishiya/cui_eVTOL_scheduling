#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eVTOL调度多目标优化包 - 基于Gurobi
提供基于Gurobi求解器的多目标优化解决方案
"""

from .evtol_scheduling_gurobi_multi import (
    solve_evtol_scheduling_multi_objective,
    solve_evtol_scheduling_with_chains_multi_objective,
    visualize_pareto_front_gurobi_multi,
    visualize_convergence_gurobi_multi
)

# 从原始gurobi模块导入任务链生成函数，确保一致性
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from gurobi.evtol_scheduling_gurobi import generate_task_chains

__version__ = "1.0.0"

__all__ = [
    "solve_evtol_scheduling_multi_objective",
    "solve_evtol_scheduling_with_chains_multi_objective",
    "generate_task_chains", 
    "visualize_pareto_front_gurobi_multi",
    "visualize_convergence_gurobi_multi"
] 
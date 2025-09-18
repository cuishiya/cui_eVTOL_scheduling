"""
eVTOL调度Gurobi优化包
包含基于Gurobi求解器的精确优化算法实现
"""

from .evtol_scheduling_gurobi import (
    generate_task_chains,
    solve_evtol_scheduling_with_chains,
    solve_evtol_scheduling_with_task_chains,
    visualize_schedule_gurobi,
    visualize_schedule_table_gurobi
)

__version__ = "1.0.0"
__author__ = "eVTOL调度优化团队"

__all__ = [
    "generate_task_chains",
    "solve_evtol_scheduling_with_chains", 
    "solve_evtol_scheduling_with_task_chains",
    "visualize_schedule_gurobi",
    "visualize_schedule_table_gurobi"
] 
"""
eVTOL调度遗传算法包
包含NSGA-II多目标优化算法实现
"""

from .evtol_nsga2 import eVTOL_NSGA2, solve_evtol_nsga2, visualize_pareto_front, visualize_evolution_history

__version__ = "1.0.0"
__author__ = "eVTOL调度优化团队"

__all__ = [
    "eVTOL_NSGA2",
    "solve_evtol_nsga2", 
    "visualize_pareto_front",
    "visualize_evolution_history"
] 
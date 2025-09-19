#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PyGMO多目标优化模块 (改进版)

本模块使用PyGMO库实现eVTOL调度问题的多目标优化，完全对应gurobi_multi数学模型的所有决策变量和约束条件。

核心特性:
- 与gurobi_multi数学模型100%对应的建模方案
- 真正的多目标优化，无权重组合，直接优化两个独立目标
- 高效的纯整数编码处理复杂组合优化问题
- 多种多目标算法：NSGA-II、MOEA/D、NSPSO、MACO
- 使用惩罚函数处理所有约束条件
- 优化的算法参数确保解的多样性

主要功能:
- eVTOLSchedulingProblem: 问题定义类，封装数学模型
- solve_pygmo_multi_objective: 多目标算法求解函数（支持多种算法）

使用示例:
    from pygmo_multi import eVTOLSchedulingProblem, solve_pygmo_multi_objective
    from data_definitions import get_tasks, get_evtols
    from gurobi.evtol_scheduling_gurobi import generate_task_chains
    
    # 加载数据
    tasks = get_tasks()
    evtols = get_evtols()
    task_chains = generate_task_chains(tasks, max_chain_length=3)
    
    # 求解
    result = solve_pygmo_multi_objective(
        tasks=tasks,
        evtols=evtols,
        task_chains=task_chains,
        time_horizon=720,
        population_size=100,
        generations=200,
        verbose=True,
        algorithm='nsga2'  # 可选: 'nsga2', 'moead', 'nspso', 'maco'
    )
    
    # 分析结果
    pareto_front = result['pareto_front']
    for sol in pareto_front:
        print(f"能耗={sol['energy']:.1f}, 延误={sol['delay']:.1f}分钟")

技术特点:
1. 遗传编码方案:
   - y变量: 任务串分配和开始时间 (num_chains × 2 维)
   - z变量: 航线选择 (num_tasks 维)
   - 时间微调: 任务和任务串时间偏移 (num_tasks + num_chains 维)
   
2. 约束处理:
   - 2.1 任务串分配唯一性 (惩罚权重: 1000)
   - 2.2 航线选择唯一性 (惩罚权重: 1000) 
   - 2.3 任务串开始时间约束 (惩罚权重: 500)
   - 2.4 任务串内时间约束 (惩罚权重: 800)
   - 2.5 eVTOL冲突约束 (惩罚权重: 1200)
   - 2.6 高度层防撞约束 (惩罚权重: 1500)
   - 2.7 任务串间隔约束 (惩罚权重: 1000)
   - 2.8 时间窗约束 (惩罚权重: 800)
   
3. 多目标函数 (真正的多目标优化):
   - 目标1: 总能耗 = Σ(soc_consumption[i][h] * z[i,h])
   - 目标2: 总延误 = Σ(max(0, task_start[i] - earliest_start[i]))
   - 重要: 无权重组合！直接优化两个独立目标

数学模型对应关系:
- 完全复制gurobi_multi中的所有决策变量
- 相同的约束条件和多目标函数
- 对应gurobi_multi的epsilon约束方法
- 确保公平的多目标算法性能比较

算法参数:
- 种群大小: ≥8且为4的倍数
- 进化代数: 推荐200代
- 算法选择: 'nsga2'(默认), 'moead', 'nspso', 'maco'
- 参数自动针对不同算法优化
"""

from .evtol_scheduling_pygmo_multi import (
    eVTOLSchedulingProblem, 
    solve_pygmo_multi_objective,
    solve_pygmo_nsga2,  # 向后兼容
    visualize_evolution_curves,
    visualize_pareto_front_evolution
)

__all__ = [
    'eVTOLSchedulingProblem',
    'solve_pygmo_multi_objective',
    'solve_pygmo_nsga2',  # 向后兼容
    'visualize_evolution_curves',
    'visualize_pareto_front_evolution'
]

__version__ = '2.0.0'
__author__ = 'eVTOL Scheduling Team'
__description__ = 'PyGMO多目标优化 - 完全对应gurobi_multi数学模型的eVTOL调度问题求解器' 
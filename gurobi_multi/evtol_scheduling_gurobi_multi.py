#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eVTOL调度多目标优化模块 - 基于Gurobi
使用与原始gurobi模块相同的数学模型，通过epsilon约束方法生成帕累托前沿

关键设计决策：
1. 变量、约束结构与原始gurobi完全相同
2. 多目标优化直接使用原始目标函数（能耗、延误），无需基准化处理
3. 通过ε-约束方法生成帕累托前沿
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'FangSong'  # 设置中文字体为仿宋
import pandas as pd
from typing import Dict, List, Tuple, Any
import itertools
from collections import defaultdict
import time
import copy
import sys
import os

# 直接从原始gurobi模块导入任务链生成函数，确保完全一致
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from gurobi.evtol_scheduling_gurobi import generate_task_chains


def solve_pareto_front_optimization(
    tasks: List[Dict], 
    evtols: List[Dict], 
    time_horizon, 
    max_chain_length,
    num_points,
    verbose: bool = False
) -> Dict:
    """
    帕累托前沿生成器 - 生成完整的帕累托最优解集
    主目标：最小化延误时间，约束条件：能耗限制
    使用与原始gurobi模块完全相同的数学模型
    
    参数:
        tasks: 任务列表
        evtols: eVTOL列表
        time_horizon: 调度时间范围
        max_chain_length: 最大任务链长度
        num_points: 帕累托前沿点数
        verbose: 是否详细输出
    
    返回:
        包含帕累托前沿的结果字典
    """
    
    print(f"🎯 开始epsilon约束方法求解")
    print(f"主目标: 最小化延误时间，约束: 能耗限制")
    
    # 生成任务链（使用与原始gurobi完全相同的函数）
    task_chains = generate_task_chains(tasks, max_chain_length)
    
    if not task_chains:
        return {"status": "no_task_chains", "pareto_front": []}
    
    # epsilon约束方法
    pareto_solutions = _generate_pareto_points_with_energy_constraints(
            tasks, evtols, task_chains, time_horizon, num_points, verbose
        )
    
    # 过滤帕累托前沿
    if pareto_solutions:
        pareto_solutions = _filter_pareto_front(pareto_solutions)
        pareto_solutions.sort(key=lambda x: x["total_energy_consumption"])
    
        # 选择一个前沿解进行可视化
        if verbose and pareto_solutions:
            # 选择延误最小的解进行可视化
            best_solution = min(pareto_solutions, key=lambda x: x["total_delay"])
            print(f"\n选择前沿解进行可视化 (能耗={best_solution['total_energy_consumption']:.1f}, 延误={best_solution['total_delay']:.1f})")
            _visualize_gurobi_multi_solution(best_solution)
    
    return {
        "status": "optimal",
        "method": "epsilon_constraint",
        "pareto_front": pareto_solutions,
        "num_solutions": len(pareto_solutions),
        "energy_range": (
            min(sol["total_energy_consumption"] for sol in pareto_solutions) if pareto_solutions else 0,
            max(sol["total_energy_consumption"] for sol in pareto_solutions) if pareto_solutions else 0
        ),
        "delay_range": (
            min(sol["total_delay"] for sol in pareto_solutions) if pareto_solutions else 0,
            max(sol["total_delay"] for sol in pareto_solutions) if pareto_solutions else 0
        )
    }


def _generate_pareto_points_with_energy_constraints(tasks, evtols, task_chains, time_horizon, num_points, verbose):
    """帕累托点生成算法 - 通过能耗约束生成多个帕累托点"""
    
    print(f"📊 使用ε-约束方法生成 {num_points} 个帕累托点")
    print(f"🎯 主目标：最小化延误时间，约束条件：能耗限制")
    
    pareto_solutions = []
    
    # 求解无约束的延误最优解，确定能耗范围
    print("🔍 求解延误最优解...")
    
    delay_opt_result = solve_single_optimization_with_constraint(
        tasks, evtols, task_chains, time_horizon, 
        target_energy=None, verbose=False
    )
    
    if delay_opt_result["status"] not in ["optimal", "time_limit"]:
        print("❌ 延误最优解求解失败")
        return []
    
    # 设定能耗约束范围：从最优解的80%到120%
    base_energy = delay_opt_result["total_energy_consumption"]
    min_energy = base_energy * 0.8
    max_energy = base_energy * 1.2
    
    print(f"📈 能耗约束范围: {min_energy:.1f} - {max_energy:.1f}")
    
    pareto_solutions = []
    
    # 生成帕累托点：在不同能耗约束下最小化延误
    energy_constraints = np.linspace(min_energy, max_energy, num_points)
    
    for i, energy_limit in enumerate(energy_constraints):
        if verbose:
            print(f"🔄 求解点 {i+1}/{num_points}: 能耗约束 ≤ {energy_limit:.1f}")
        
        result = solve_single_optimization_with_constraint(
            tasks, evtols, task_chains, time_horizon,
            target_energy=energy_limit, verbose=False
        )
        
        if result["status"] in ["optimal", "time_limit"]:
            pareto_solutions.append(result)
    
    return pareto_solutions


def _filter_pareto_front(solutions):
    """过滤出真正的帕累托前沿"""
    
    if not solutions:
        return []
    
    pareto_front = []
    
    for i, sol1 in enumerate(solutions):
        is_dominated = False
        
        for j, sol2 in enumerate(solutions):
            if i != j:
                # sol2支配sol1的条件：sol2在所有目标上都不比sol1差，且至少在一个目标上更好
                # 或者两个解完全相同但sol2的索引更小（用于去重）
                if (sol2["total_energy_consumption"] <= sol1["total_energy_consumption"] and 
                    sol2["total_delay"] <= sol1["total_delay"] and 
                    (sol2["total_energy_consumption"] < sol1["total_energy_consumption"] or 
                     sol2["total_delay"] < sol1["total_delay"] or 
                     (sol2["total_energy_consumption"] == sol1["total_energy_consumption"] and 
                      sol2["total_delay"] == sol1["total_delay"] and j < i))):
                    is_dominated = True
                    break
        
        if not is_dominated:
            pareto_front.append(sol1)
    
    return pareto_front

def solve_single_optimization_with_constraint(
    tasks: List[Dict],
    evtols: List[Dict],
    task_chains: List[List[int]],
    time_horizon: int = 720,
    target_energy: float = None,
    verbose: bool = False
) -> Dict:
    """
    单次优化求解器 - 最小化延误，可选能耗约束
    
    参数:
        tasks: 任务列表
        evtols: eVTOL列表
        task_chains: 任务串列表
        time_horizon: 调度时间范围
        target_energy: 能耗约束限制（可选）
        verbose: 是否打印详细信息
    
    返回:
        包含调度结果的字典
    """
    if verbose:
        print("正在构建基于任务串的优化模型...")

    # 创建模型（与原始gurobi完全相同）
    model = gp.Model("eVTOL_Scheduling_with_Chains_Epsilon")

    # 提取基本信息
    num_tasks = len(tasks)
    num_evtols = len(evtols)
    num_chains = len(task_chains)
    num_routes = 3  # 每对起降点之间有3条航线

    # ===== 1. 定义决策变量（与原始gurobi完全相同）=====

    # 主决策变量: y[c,k,t] - eVTOL k在时刻t开始执行任务串c
    y = {}
    for c in range(num_chains):
        for k in range(num_evtols):
            for t in range(time_horizon):
                y[c, k, t] = model.addVar(vtype=GRB.BINARY, name=f"y_{c}_{k}_{t}")

    # 任务-航线选择变量: z[i,h] - 任务i使用航线h
    z = {}
    for i in range(num_tasks):
        for h in range(num_routes):
            z[i, h] = model.addVar(vtype=GRB.BINARY, name=f"z_{i}_{h}")

    # 辅助变量：任务的开始时间和结束时间
    task_start = {}
    task_end = {}
    for i in range(num_tasks):
        task_start[i] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=time_horizon, name=f"task_start_{i}")
        task_end[i] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=time_horizon, name=f"task_end_{i}")

    # 任务串的开始时间
    chain_start = {}
    for c in range(num_chains):
        chain_start[c] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=time_horizon, name=f"chain_start_{c}")

    # 任务串的结束时间 (用于任务串间隔约束)
    chain_end = {}
    for c in range(num_chains):
        chain_end[c] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=time_horizon, name=f"chain_end_{c}")

    # 任务串分配指示变量 (用于任务串间隔约束)
    b_chain_evtol = {}
    for c in range(num_chains):
        for k in range(num_evtols):
            b_chain_evtol[c, k] = model.addVar(vtype=GRB.BINARY, name=f"b_{c}_{k}")

    # 任务串对分配指示变量 (用于任务串间隔约束)
    both_assigned = {}
    for k in range(num_evtols):
        for c1 in range(num_chains):
            for c2 in range(c1 + 1, num_chains):
                both_assigned[c1, c2, k] = model.addVar(vtype=GRB.BINARY, name=f"both_assigned_{c1}_{c2}_{k}")

    # 任务串顺序指示变量 (用于任务串间隔约束)
    chain_order = {}
    for k in range(num_evtols):
        for c1 in range(num_chains):
            for c2 in range(c1 + 1, num_chains):
                chain_order[c1, c2, k] = model.addVar(vtype=GRB.BINARY, name=f"order_{c1}_{c2}_{k}")

    model.update()

    # ===== 2. 定义约束条件（与原始gurobi完全相同）=====

    # 2.1 任务串分配唯一性约束
    for c in range(num_chains):
        model.addConstr(
            gp.quicksum(y[c, k, t] for k in range(num_evtols) for t in range(time_horizon)) == 1,
            f"chain_assignment_{c}"
        )

    # 2.2 每个任务必须选择一条航线
    for i in range(num_tasks):
        model.addConstr(
            gp.quicksum(z[i, h] for h in range(num_routes)) == 1,
            f"task_route_selection_{i}"
        )

    # 2.3 任务串开始时间约束
    for c in range(num_chains):
        model.addConstr(
            chain_start[c] == gp.quicksum(t * y[c, k, t] for k in range(num_evtols) for t in range(time_horizon)),
            f"chain_start_time_{c}"
        )

    # 2.4 任务串内任务的时间约束
    for c, chain in enumerate(task_chains):
        if len(chain) == 1:
            # 单任务串
            task_id = chain[0]
            model.addConstr(task_start[task_id] == chain_start[c], f"single_task_start_{c}_{task_id}")
            model.addConstr(
                task_end[task_id] == task_start[task_id] + gp.quicksum(
                    tasks[task_id]['duration'][h] * z[task_id, h] for h in range(num_routes)
                ),
                f"single_task_end_{c}_{task_id}"
            )
        else:
            # 多任务串 - 确保任务按顺序执行且位置连续
            for i, task_id in enumerate(chain):
                if i == 0:
                    # 第一个任务从任务串开始时间开始
                    model.addConstr(task_start[task_id] == chain_start[c], f"first_task_start_{c}_{task_id}")
                else:
                    # 后续任务在前一个任务结束后立即开始（考虑间隔时间）
                    prev_task_id = chain[i-1]
                    interval_time = 20  # 任务间隔时间（分钟）
                    model.addConstr(
                        task_start[task_id] >= task_end[prev_task_id] + interval_time,
                        f"task_sequence_{c}_{prev_task_id}_{task_id}"
                    )

                # 任务结束时间
                model.addConstr(
                    task_end[task_id] == task_start[task_id] + gp.quicksum(
                        tasks[task_id]['duration'][h] * z[task_id, h] for h in range(num_routes)
                    ),
                    f"task_end_{c}_{task_id}"
                )

    # 2.5 eVTOL同一时刻只能执行一个任务串
    for tau in range(time_horizon):
        for k in range(num_evtols):
            active_chains = []
            for c in range(num_chains):
                # 计算任务串c的最大可能持续时间
                max_chain_duration = sum(max(tasks[task_id]['duration']) for task_id in task_chains[c]) + 20 * (len(task_chains[c]) - 1)

                # 如果任务串在时刻tau可能正在执行
                for t in range(max(0, tau - max_chain_duration + 1), tau + 1):
                    if t < time_horizon:
                        active_chains.append(y[c, k, t])

            if active_chains:
                model.addConstr(
                    gp.quicksum(active_chains) <= 1,
                    f"evtol_single_chain_{tau}_{k}"
                )

    # 2.6 高度层防撞约束 - 基于任务对的时间冲突检测
    for i in range(num_tasks):
        for j in range(i + 1, num_tasks):  # 避免重复检查
            for h in range(num_routes):
                # 引入二进制变量表示两个任务是否都选择航线h
                both_use_route_h = model.addVar(vtype=GRB.BINARY, name=f"both_route_{i}_{j}_{h}")

                # 线性化约束：both_use_route_h = z[i,h] * z[j,h]
                model.addConstr(both_use_route_h <= z[i, h], f"both_route_1_{i}_{j}_{h}")
                model.addConstr(both_use_route_h <= z[j, h], f"both_route_2_{i}_{j}_{h}")
                model.addConstr(both_use_route_h >= z[i, h] + z[j, h] - 1, f"both_route_3_{i}_{j}_{h}")

                # 如果两个任务都使用航线h，则它们不能时间重叠
                i_before_j = model.addVar(vtype=GRB.BINARY, name=f"order_{i}_{j}_{h}")

                # Big-M约束来表示时间顺序
                M = time_horizon  # 足够大的常数
                model.addConstr(
                    task_end[i] <= task_start[j] + M * (1 - i_before_j) + M * (1 - both_use_route_h),
                    f"no_overlap_1_{i}_{j}_{h}"
                )
                model.addConstr(
                    task_end[j] <= task_start[i] + M * i_before_j + M * (1 - both_use_route_h),
                    f"no_overlap_2_{i}_{j}_{h}"
                )

    # 2.7 任务串之间的时间间隔约束
    chain_interval_time = 30  # 任务串之间的最小间隔时间（分钟）

    # 计算每个任务串的结束时间
    for c, chain in enumerate(task_chains):
        last_task_id = chain[-1]
        model.addConstr(chain_end[c] == task_end[last_task_id], f"chain_end_time_{c}")

    # 定义任务串分配指示变量的约束
    for c in range(num_chains):
        for k in range(num_evtols):
            model.addConstr(b_chain_evtol[c, k] == gp.quicksum(y[c, k, t] for t in range(time_horizon)), 
                          f"chain_evtol_assignment_{c}_{k}")

    # 对于每架eVTOL，其执行的任意两个任务串之间必须有时间间隔
    for k in range(num_evtols):
        for c1 in range(num_chains):
            for c2 in range(c1 + 1, num_chains):
                # 定义both_assigned约束: 当且仅当c1和c2都分配给eVTOL k时为1
                model.addConstr(both_assigned[c1, c2, k] <= b_chain_evtol[c1, k], f"both_assigned_1_{c1}_{c2}_{k}")
                model.addConstr(both_assigned[c1, c2, k] <= b_chain_evtol[c2, k], f"both_assigned_2_{c1}_{c2}_{k}")
                model.addConstr(both_assigned[c1, c2, k] >= b_chain_evtol[c1, k] + b_chain_evtol[c2, k] - 1, 
                              f"both_assigned_3_{c1}_{c2}_{k}")

                # Big-M 约束确保任务串间隔
                M = time_horizon
                model.addConstr(
                    chain_end[c1] + chain_interval_time <= chain_start[c2] + M * (1 - chain_order[c1, c2, k]) + M * (1 - both_assigned[c1, c2, k]),
                    f"chain_interval_1_{c1}_{c2}_{k}"
                )
                model.addConstr(
                    chain_end[c2] + chain_interval_time <= chain_start[c1] + M * chain_order[c1, c2, k] + M * (1 - both_assigned[c1, c2, k]),
                    f"chain_interval_2_{c1}_{c2}_{k}"
                )

    # 2.8 任务时间窗约束
    for i in range(num_tasks):
        model.addConstr(
            task_start[i] >= tasks[i]['earliest_start'],
            f"earliest_start_{i}"
        )

    # ===== 3. 定义目标函数（epsilon约束方法）=====
    
    # 计算总能量消耗（与原始gurobi完全相同）
    total_energy_consumption = gp.quicksum(
        tasks[i]['soc_consumption'][h] * z[i, h]
        for i in range(num_tasks)
        for h in range(num_routes)
    )

    # 计算总延误时间
    total_delay = gp.quicksum(task_start[i] - tasks[i]['earliest_start'] for i in range(num_tasks))
    
    # ===== 目标：最小化延误，约束：能耗限制 =====
    
    if target_energy is not None:
        # 设置能耗约束
        model.addConstr(total_energy_consumption <= target_energy, "energy_constraint")
    
    # 目标函数：最小化延误时间
    model.setObjective(total_delay, GRB.MINIMIZE)

    # ===== 4. 求解模型（与原始gurobi相同的参数）=====
    model.setParam('OutputFlag', 1 if verbose else 0)
    model.setParam('MIPGap', 0.05)
    model.setParam('TimeLimit', 1800)
    model.setParam('MIPFocus', 1)
    
    start_time = time.time()
    model.optimize()
    solve_time = time.time() - start_time

    # ===== 5. 提取结果（与原始gurobi完全相同）=====
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        # 计算总能量消耗
        total_energy = sum(
            tasks[i]['soc_consumption'][h] * z[i, h].x
            for i in range(num_tasks)
            for h in range(num_routes)
            if z[i, h].x > 0.5
        )

        # 计算总延误时间
        total_delay_value = sum(task_start[i].x - tasks[i]['earliest_start'] for i in range(num_tasks))

        # 提取结果（epsilon约束方法：直接使用原始目标值）
        result = {
            "status": "optimal" if model.status == GRB.OPTIMAL else "time_limit",
            "objective_value": model.objVal,
            "total_energy_consumption": total_energy,
            "total_delay": total_delay_value,
            "solve_time": solve_time,
            "schedule": [],
            "task_chains": task_chains,
            "chain_assignments": []
        }

        # 提取任务串分配（与原始gurobi完全相同）
        for c in range(num_chains):
            for k in range(num_evtols):
                for t in range(time_horizon):
                    if y[c, k, t].x > 0.5:
                        result["chain_assignments"].append({
                            "chain_id": c,
                            "evtol_id": k,
                            "start_time": t,
                            "tasks": task_chains[c]
                        })

        # 提取任务调度（与原始gurobi完全相同）
        for i in range(num_tasks):
            selected_route = None
            for h in range(num_routes):
                if z[i, h].x > 0.5:
                    selected_route = h
                    break

            # 找到执行此任务的eVTOL
            evtol_id = None
            for assignment in result["chain_assignments"]:
                if i in assignment["tasks"]:
                    evtol_id = assignment["evtol_id"]
                    break

            if selected_route is not None and evtol_id is not None:
                result["schedule"].append({
                    "task_id": i,
                    "evtol_id": evtol_id,
                    "start_time": int(task_start[i].x),
                    "end_time": int(task_end[i].x),
                    "route": selected_route,
                    "from": tasks[i]['from'],
                    "to": tasks[i]['to'],
                    "delay": task_start[i].x - tasks[i]['earliest_start']
                })

        return result
    else:
        return {"status": "infeasible", "solve_time": solve_time}

def _visualize_gurobi_multi_solution(solution):
    """
    可视化Gurobi Multi解
    """
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from visualization import visualize_schedule_table, visualize_schedule_gantt
    
    if "schedule" in solution and solution["schedule"]:
        visualize_schedule_table(solution["schedule"], "Gurobi Multi", "picture_result/evtol_schedule_table_gurobi_multi.png")
        visualize_schedule_gantt(solution["schedule"], "Gurobi Multi", "picture_result/evtol_schedule_gurobi_multi.png")

def visualize_pareto_front_gurobi_epsilon(result: Dict, save_path: str = "picture_result/pareto_front_gurobi_epsilon_constraint.png"):
    """可视化Gurobi epsilon约束方法的帕累托前沿"""
    
    pareto_front = result["pareto_front"]
    
    if not pareto_front:
        print("没有帕累托前沿数据可视化")
        return
    
    # 提取目标函数值
    energies = [sol["total_energy_consumption"] for sol in pareto_front]
    delays = [sol["total_delay"] for sol in pareto_front]
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # 绘制帕累托前沿点
    plt.scatter(energies, delays, c='blue', s=80, alpha=0.7, label=f'Gurobi ε-约束解 ({len(pareto_front)}个)', 
                edgecolors='darkblue', linewidth=1)
    
    # 连接帕累托前沿
    if len(pareto_front) > 1:
        sorted_solutions = sorted(pareto_front, key=lambda x: x["total_energy_consumption"])
        sorted_energies = [sol["total_energy_consumption"] for sol in sorted_solutions]
        sorted_delays = [sol["total_delay"] for sol in sorted_solutions]
        plt.plot(sorted_energies, sorted_delays, 'b--', alpha=0.5, linewidth=1)
    
        
    
    # 设置标签和标题
    plt.xlabel('总能耗')
    plt.ylabel('总延误时间 (分钟)')
    plt.title(f'eVTOL调度问题的帕累托前沿 - Gurobi ε-约束方法\n主目标：最小化延误，约束：能耗')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 添加统计信息
    energy_range = result["energy_range"]
    delay_range = result["delay_range"]
    plt.text(0.02, 0.98, f'解的数量: {len(pareto_front)}\n'
                         f'能耗范围: {energy_range[0]:.1f} - {energy_range[1]:.1f}\n'
                         f'延误范围: {delay_range[0]:.1f} - {delay_range[1]:.1f}分钟',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 保存图形
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"帕累托前沿图已保存到: {save_path}")


def visualize_convergence_gurobi_epsilon(result: Dict, save_path: str = "picture_result/convergence_gurobi_epsilon_constraint.png"):
    """可视化Gurobi epsilon约束方法的收敛历史"""
    
    pareto_front = result["pareto_front"]
    
    if not pareto_front:
        print("没有数据可视化")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 求解时间分布
    solve_times = [sol.get("solve_time", 0) for sol in pareto_front]
    ax1.hist(solve_times, bins=min(10, len(pareto_front)), alpha=0.7, color='blue')
    ax1.set_xlabel('求解时间 (秒)')
    ax1.set_ylabel('解的数量')
    ax1.set_title('求解时间分布')
    ax1.grid(True, alpha=0.3)
    
    # 2. 能耗分布
    energies = [sol["total_energy_consumption"] for sol in pareto_front]
    ax2.hist(energies, bins=min(15, len(pareto_front)), alpha=0.7, color='green')
    ax2.set_xlabel('总能耗')
    ax2.set_ylabel('解的数量')
    ax2.set_title('能耗分布')
    ax2.grid(True, alpha=0.3)
    
    # 3. 延误分布
    delays = [sol["total_delay"] for sol in pareto_front]
    ax3.hist(delays, bins=min(15, len(pareto_front)), alpha=0.7, color='red')
    ax3.set_xlabel('总延误时间 (分钟)')
    ax3.set_ylabel('解的数量')
    ax3.set_title('延误分布')
    ax3.grid(True, alpha=0.3)
    
    # 4. 目标函数空间散点图
    ax4.scatter(energies, delays, alpha=0.7, color='purple')
    ax4.set_xlabel('总能耗')
    ax4.set_ylabel('总延误时间 (分钟)')
    ax4.set_title('目标函数空间')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"收敛历史图已保存到: {save_path}") 
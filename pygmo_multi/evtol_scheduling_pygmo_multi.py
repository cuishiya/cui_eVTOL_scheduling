#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eVTOL调度NSGA-II优化模块 - 基于PyGMO
使用NSGA-II算法求解eVTOL调度多目标优化问题
"""

import pygmo as pg
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_definitions import get_evtols, get_locations, get_tasks
from gurobi.evtol_scheduling_gurobi import generate_task_chains
import itertools
from collections import defaultdict

class eVTOLSchedulingProblem:
    """
    eVTOL调度问题的PyGMO封装类
    
    数学模型完全对应gurobi_multi实现，包含相同的决策变量和约束条件：
    
    决策变量：
    1. y[c,k,t] - eVTOL k在时刻t开始执行任务串c (二进制)
    2. z[i,h] - 任务i使用航线h (二进制)
    3. task_start[i] - 任务i的开始时间 (整数)
    4. task_end[i] - 任务i的结束时间 (整数)
    5. chain_start[c] - 任务串c的开始时间 (整数)
    6. chain_end[c] - 任务串c的结束时间 (整数)
    7. b_chain_evtol[c,k] - 任务串c是否分配给eVTOL k (二进制)
    8. both_assigned[c1,c2,k] - 任务串c1和c2是否都分配给eVTOL k (二进制)
    9. chain_order[c1,c2,k] - 任务串c1是否在c2之前执行 (二进制)
    10. both_use_route_h[i,j,h] - 任务i和j是否都使用航线h (二进制)
    11. i_before_j[i,j,h] - 任务i是否在任务j之前完成 (二进制)
    
    约束条件：
    2.1 任务串分配唯一性约束
    2.2 每个任务必须选择一条航线
    2.3 任务串开始时间约束
    2.4 任务串内任务的时间约束
    2.5 eVTOL同一时刻只能执行一个任务串
    2.6 高度层防撞约束
    2.7 任务串之间的时间间隔约束
    2.8 任务时间窗约束
    
    多目标函数 (对应gurobi_multi的epsilon约束方法)：
    目标1: minimize 总能耗
    目标2: minimize 总延误
    
    注意: 这是真正的多目标优化，无权重组合！
    """
    
    def __init__(self, tasks, evtols, task_chains, time_horizon=720):
        self.tasks = tasks
        self.evtols = evtols
        self.task_chains = task_chains
        self.time_horizon = time_horizon
        
        self.num_tasks = len(tasks)
        self.num_evtols = len(evtols)
        self.num_chains = len(task_chains)
        self.num_routes = 3
        
        # 遗传编码设计
        self._setup_encoding()
        
        # 约束参数
        self.chain_interval_time = 30  # 任务串之间的最小间隔时间
        
        print(f"问题规模: {self.num_tasks}个任务, {self.num_evtols}架eVTOL, {self.num_chains}个任务串")
        print(f"决策变量维度: {self.dimensions} (纯整数编码)")
        print(f"多目标优化: 目标1=总能耗, 目标2=总延误 (无权重组合)")
        print(f"搜索空间: 有限离散整数空间，更符合问题本质")
    
    def _setup_encoding(self):
        """
        设计遗传编码方案 - 纯整数编码
        
        编码结构：
        [y_variables | z_variables | task_start_variables | chain_start_variables]
        
        1. y_variables (任务串分配和开始时间):
           - 每个任务串c对应2个整数：
             * y_evtol[c]: 范围[0, num_evtols-1] -> 直接eVTOL ID
             * y_time[c]: 范围[0, time_horizon-1] -> 直接开始时间
           - 总计: num_chains * 2 个变量
        
        2. z_variables (航线选择):
           - 每个任务i对应1个整数：
             * z_route[i]: 范围[0, num_routes-1] -> 直接航线ID
           - 总计: num_tasks 个变量
        
        3. task_start_variables (任务开始时间微调):
           - 每个任务i对应1个整数：
             * task_start_offset[i]: 范围[0, 59] -> 直接时间偏移(分钟)
           - 总计: num_tasks 个变量
           
        4. chain_start_variables (任务串开始时间微调):
           - 每个任务串c对应1个整数：
             * chain_start_offset[c]: 范围[0, 119] -> 直接时间偏移(分钟)
           - 总计: num_chains 个变量
        
        总维度 = num_chains * 2 + num_tasks + num_tasks + num_chains
                = num_chains * 3 + num_tasks * 2
        """
        # 编码段索引
        self.y_start = 0
        self.y_end = self.num_chains * 2
        self.z_start = self.y_end
        self.z_end = self.z_start + self.num_tasks
        self.task_offset_start = self.z_end
        self.task_offset_end = self.task_offset_start + self.num_tasks
        self.chain_offset_start = self.task_offset_end
        self.chain_offset_end = self.chain_offset_start + self.num_chains
        
        self.dimensions = self.chain_offset_end
        
        # 设置各段的边界
        self._setup_bounds()
    
    def _setup_bounds(self):
        """设置各变量段的边界"""
        self.lower_bounds = []
        self.upper_bounds = []
        
        # 1. y_variables边界 (任务串分配)
        for c in range(self.num_chains):
            self.lower_bounds.append(0)                    # eVTOL ID下界
            self.upper_bounds.append(self.num_evtols - 1)  # eVTOL ID上界
            self.lower_bounds.append(0)                    # 开始时间下界
            self.upper_bounds.append(self.time_horizon - 1) # 开始时间上界
        
        # 2. z_variables边界 (航线选择)
        for i in range(self.num_tasks):
            self.lower_bounds.append(0)                    # 航线ID下界
            self.upper_bounds.append(self.num_routes - 1)  # 航线ID上界
        
        # 3. task_start_variables边界 (任务时间微调)
        for i in range(self.num_tasks):
            self.lower_bounds.append(0)    # 时间偏移下界
            self.upper_bounds.append(59)   # 时间偏移上界 (0-59分钟)
        
        # 4. chain_start_variables边界 (任务串时间微调)
        for c in range(self.num_chains):
            self.lower_bounds.append(0)    # 时间偏移下界
            self.upper_bounds.append(119)  # 时间偏移上界 (0-119分钟)
        
    def get_bounds(self):
        """返回决策变量的边界"""
        return (self.lower_bounds, self.upper_bounds)
    
    def get_nobj(self):
        """返回目标函数数量"""
        return 2  # 能耗 + 延误
    
    def get_nec(self):
        """返回等式约束数量"""
        return 0  # 使用惩罚函数处理所有约束
    
    def get_nic(self):
        """返回不等式约束数量"""
        return 0  # 使用惩罚函数处理所有约束
    
    def _decode_solution(self, x):
        """
        解码遗传个体为调度方案
        
        返回解码后的调度变量字典，对应gurobi中的决策变量
        """
        try:
            # 边界修复：确保所有值都在有效范围内
            x_repaired = self._repair_solution(x)
            
            # 1. 解码y变量 (任务串分配) - 直接使用整数值
            y = {}
            for c in range(self.num_chains):
                evtol_id = int(x_repaired[self.y_start + c * 2])
                start_time = int(x_repaired[self.y_start + c * 2 + 1])
                
                # 初始化y矩阵
                for k in range(self.num_evtols):
                    for t in range(self.time_horizon):
                        y[c, k, t] = 0
                
                # 设置选中的分配
                y[c, evtol_id, start_time] = 1
            
            # 2. 解码z变量 (航线选择) - 直接使用整数值
            z = {}
            for i in range(self.num_tasks):
                route_id = int(x_repaired[self.z_start + i])
                
                # 初始化z矩阵
                for h in range(self.num_routes):
                    z[i, h] = 0
                
                # 设置选中的航线
                z[i, route_id] = 1
            
            # 3. 计算任务开始时间 - 直接使用整数偏移值
            task_start = {}
            for i in range(self.num_tasks):
                # 基础时间：从任务所在的任务串开始时间推导
                base_time = 0
                offset = int(x_repaired[self.task_offset_start + i])  # 直接使用整数偏移(0-59分钟)
                
                # 找到任务i所属的任务串
                for c, chain in enumerate(self.task_chains):
                    if i in chain:
                        # 从任务串开始时间计算
                        for k in range(self.num_evtols):
                            for t in range(self.time_horizon):
                                if y[c, k, t] == 1:
                                    chain_start_time = t
                                    chain_offset = int(x_repaired[self.chain_offset_start + c])  # 直接使用整数偏移(0-119分钟)
                                    
                                    # 计算任务在串中的位置
                                    task_index_in_chain = chain.index(i)
                                    if task_index_in_chain == 0:
                                        base_time = chain_start_time + chain_offset
                                    else:
                                        # 考虑前面任务的执行时间和间隔
                                        prev_duration = 0
                                        for idx in range(task_index_in_chain):
                                            prev_task = chain[idx]
                                            for h in range(self.num_routes):
                                                if z[prev_task, h] == 1:
                                                    prev_duration += self.tasks[prev_task]['duration'][h]
                                        base_time = chain_start_time + chain_offset + prev_duration + task_index_in_chain * 20
                                    break
                        break
                
                task_start[i] = max(base_time + offset, self.tasks[i]['earliest_start'])
            
            # 4. 计算任务结束时间
            task_end = {}
            for i in range(self.num_tasks):
                duration = 0
                for h in range(self.num_routes):
                    if z[i, h] == 1:
                        duration = self.tasks[i]['duration'][h]
                        break
                task_end[i] = task_start[i] + duration
            
            # 5. 计算任务串开始时间
            chain_start = {}
            for c in range(self.num_chains):
                for k in range(self.num_evtols):
                    for t in range(self.time_horizon):
                        if y[c, k, t] == 1:
                            chain_offset_val = x[self.chain_offset_start + c]
                            chain_offset = int(chain_offset_val * 120)
                            chain_start[c] = t + chain_offset
                            break
            
            # 6. 计算任务串结束时间
            chain_end = {}
            for c, chain in enumerate(self.task_chains):
                last_task_id = chain[-1]
                chain_end[c] = task_end[last_task_id]
            
            # 7. 计算辅助变量
            b_chain_evtol = {}
            for c in range(self.num_chains):
                for k in range(self.num_evtols):
                    b_chain_evtol[c, k] = sum(y[c, k, t] for t in range(self.time_horizon))
            
            both_assigned = {}
            for k in range(self.num_evtols):
                for c1 in range(self.num_chains):
                    for c2 in range(c1 + 1, self.num_chains):
                        both_assigned[c1, c2, k] = min(b_chain_evtol[c1, k], b_chain_evtol[c2, k])
            
            return {
                'y': y,
                'z': z,
                'task_start': task_start,
                'task_end': task_end,
                'chain_start': chain_start,
                'chain_end': chain_end,
                'b_chain_evtol': b_chain_evtol,
                'both_assigned': both_assigned
            }
            
        except Exception as e:
            print(f"解码错误: {e}")
            return None
    
    def _repair_solution(self, x):
        """
        修复解向量，确保所有值都在有效边界内
        """
        x_repaired = []
        for i in range(len(x)):
            val = x[i]
            lower = self.lower_bounds[i]
            upper = self.upper_bounds[i]
            
            # 将值限制在边界内并转换为整数
            val = max(lower, min(upper, int(round(val))))
            x_repaired.append(val)
        
        return x_repaired
    
    def _calculate_objectives(self, solution):
        """
        计算目标函数值 - 对应gurobi_multi的两个独立目标
        """
        z = solution['z']
        task_start = solution['task_start']
        
        # 目标1: 总能量消耗 (与gurobi_multi完全相同)
        total_energy = 0
        for i in range(self.num_tasks):
            for h in range(self.num_routes):
                if z[i, h] == 1:
                    total_energy += self.tasks[i]['soc_consumption'][h]
        
        # 目标2: 总延误时间 (与gurobi_multi完全相同)
        total_delay = 0
        for i in range(self.num_tasks):
            delay = max(0, task_start[i] - self.tasks[i]['earliest_start'])
            total_delay += delay
        
        return total_energy, total_delay
    
    def _check_constraints(self, solution):
        """检查约束违反情况"""
        violations = []
        penalty = 0.0
        
        y = solution['y']
        z = solution['z']
        task_start = solution['task_start']
        task_end = solution['task_end']
        chain_start = solution['chain_start']
        chain_end = solution['chain_end']
        b_chain_evtol = solution['b_chain_evtol']
        both_assigned = solution['both_assigned']
        
        # 2.1 任务串分配唯一性约束
        for c in range(self.num_chains):
            assignment_sum = sum(y[c, k, t] for k in range(self.num_evtols) for t in range(self.time_horizon))
            if abs(assignment_sum - 1.0) > 1e-6:
                violations.append(f"任务串{c}分配违反唯一性: {assignment_sum}")
                penalty += 1000
        
        # 2.2 航线选择唯一性约束
        for i in range(self.num_tasks):
            route_sum = sum(z[i, h] for h in range(self.num_routes))
            if abs(route_sum - 1.0) > 1e-6:
                violations.append(f"任务{i}航线选择违反唯一性: {route_sum}")
                penalty += 1000
        
        # 2.3 任务串开始时间约束
        for c in range(self.num_chains):
            calculated_start = sum(t * y[c, k, t] for k in range(self.num_evtols) for t in range(self.time_horizon))
            if abs(chain_start[c] - calculated_start) > 120:  # 允许微调偏差
                violations.append(f"任务串{c}开始时间约束违反")
                penalty += 500
        
        # 2.4 任务串内任务时间约束
        for c, chain in enumerate(self.task_chains):
            if len(chain) > 1:
                for i in range(len(chain) - 1):
                    curr_task = chain[i]
                    next_task = chain[i + 1]
                    if task_start[next_task] < task_end[curr_task] + 20:
                        violations.append(f"任务串{c}内任务{curr_task}->{next_task}时间约束违反")
                        penalty += 800
        
        # 2.5 eVTOL冲突约束
        for k in range(self.num_evtols):
            for tau in range(self.time_horizon):
                active_chains_count = 0
                for c in range(self.num_chains):
                    # 计算任务串c的最大可能持续时间
                    max_duration = sum(max(self.tasks[task_id]['duration']) for task_id in self.task_chains[c]) + 20 * (len(self.task_chains[c]) - 1)
                    
                    # 检查在时刻tau是否可能正在执行
                    for t in range(max(0, tau - max_duration + 1), tau + 1):
                        if t < self.time_horizon and y[c, k, t] == 1:
                            active_chains_count += 1
                            break
                
                if active_chains_count > 1:
                    violations.append(f"eVTOL{k}在时刻{tau}执行多个任务串")
                    penalty += 1200
        
        # 2.6 高度层防撞约束
        for i in range(self.num_tasks):
            for j in range(i + 1, self.num_tasks):
                for h in range(self.num_routes):
                    if z[i, h] == 1 and z[j, h] == 1:
                        # 两个任务使用相同航线，检查时间重叠
                        if not (task_end[i] <= task_start[j] or task_end[j] <= task_start[i]):
                            violations.append(f"任务{i}和{j}在航线{h}上时间重叠")
                            penalty += 1500
        
        # 2.7 任务串间隔约束
        for k in range(self.num_evtols):
            for c1 in range(self.num_chains):
                for c2 in range(c1 + 1, self.num_chains):
                    if both_assigned[c1, c2, k] == 1:
                        interval_satisfied = (chain_end[c1] + self.chain_interval_time <= chain_start[c2] or 
                                            chain_end[c2] + self.chain_interval_time <= chain_start[c1])
                        if not interval_satisfied:
                            violations.append(f"eVTOL{k}的任务串{c1}和{c2}间隔不足")
                            penalty += 1000
        
        # 2.8 任务时间窗约束
        for i in range(self.num_tasks):
            if task_start[i] < self.tasks[i]['earliest_start']:
                violations.append(f"任务{i}违反最早开始时间")
                penalty += 800
        
        return violations, penalty
    
    def fitness(self, x):
        """
        计算适应度函数 - 多目标优化
        
        返回: [目标1, 目标2] = [总能耗, 总延误] + 约束惩罚
        对应gurobi_multi的epsilon约束方法中的两个独立目标
        """
        try:
            # 解码个体
            solution = self._decode_solution(x)
            if solution is None:
                return [50000.0, 50000.0]
            
            # 计算目标函数 (与gurobi_multi相同)
            total_energy, total_delay = self._calculate_objectives(solution)
            
            # 检查约束
            violations, penalty = self._check_constraints(solution)
            
            # 返回带惩罚的原始目标函数值 (无权重组合)
            objective1 = total_energy + penalty      # 目标1: 总能耗
            objective2 = total_delay + penalty       # 目标2: 总延误
            
            return [objective1, objective2]
            
        except Exception as e:
            print(f"适应度计算错误: {e}")
            return [50000.0, 50000.0]

def solve_pygmo_nsga2(tasks, evtols, task_chains, time_horizon=720, 
                     population_size=100, generations=200, verbose=True):
    """
    使用NSGA-II算法求解eVTOL调度问题
    """
    if verbose:
        print("=== PyGMO NSGA-II 多目标优化求解 (纯整数编码) ===")
    
    # 确保population_size符合NSGA-II要求
    if population_size < 8:
        population_size = 8
    if population_size % 4 != 0:
        population_size = ((population_size // 4) + 1) * 4
    
    try:
        # 创建问题实例
        problem = eVTOLSchedulingProblem(tasks, evtols, task_chains, time_horizon)
        
        # 创建PyGMO问题对象
        pg_problem = pg.problem(problem)
        
        # 创建算法 - 针对整数编码优化参数
        nsga2 = pg.nsga2(
            gen=1,  # 每次只进化1代
            cr=0.8,     # 降低交叉率，适应整数编码
            eta_c=10,   # 降低交叉分布指数，增加探索性
            m=2.0/problem.dimensions,  # 提高变异率，适应离散搜索
            eta_m=10    # 降低变异分布指数，增加变异强度
        )
        algo = pg.algorithm(nsga2)
        
        # 创建种群
        pop = pg.population(pg_problem, population_size)
        
        if verbose:
            print(f"初始种群大小: {len(pop)}")
            print(f"决策变量维度: {problem.dimensions}")
            print(f"开始进化 {generations} 代...")
            print("=" * 80)
        
        # 记录进化过程数据
        evolution_data = {
            'generations': [],
            'pareto_count': [],
            'min_energy': [],
            'avg_energy': [],
            'min_delay': [],
            'avg_delay': [],
            'hypervolume': [],
            'pareto_fronts': []  # 存储每代的帕累托前沿
        }
        
        # 逐代进化并打印信息
        for gen in range(generations):
            # 进化一代
            pop = algo.evolve(pop)
            
            if verbose:
                # 获取当前种群的适应度值
                fitness_values = pop.get_f()
                
                # 计算统计信息
                fitness1 = fitness_values[:, 0]  # 适应度1 (能耗+惩罚)
                fitness2 = fitness_values[:, 1]  # 适应度2 (延误+惩罚)
                
                min_fitness1 = np.min(fitness1)
                max_fitness1 = np.max(fitness1)
                avg_fitness1 = np.mean(fitness1)
                
                min_fitness2 = np.min(fitness2)
                max_fitness2 = np.max(fitness2)
                avg_fitness2 = np.mean(fitness2)
                
                # 计算当前帕累托前沿数量 (使用适应度进行筛选用于显示)
                pareto_indices = pg.non_dominated_front_2d(fitness_values)
                pareto_count = len(pareto_indices)
                
                # 计算帕累托前沿的适应度范围 (用于显示)
                if pareto_count > 0:
                    pareto_fitness1 = fitness1[pareto_indices]
                    pareto_fitness2 = fitness2[pareto_indices]
                    pareto_fitness1_range = f"{np.min(pareto_fitness1):.1f}-{np.max(pareto_fitness1):.1f}"
                    pareto_fitness2_range = f"{np.min(pareto_fitness2):.1f}-{np.max(pareto_fitness2):.1f}"
                    
                    # 计算超体积 (Hypervolume) - 使用适应度
                    ref_point = [np.max(fitness1) * 1.1, np.max(fitness2) * 1.1]
                    try:
                        hv = pg.hypervolume(fitness_values[pareto_indices])
                        hypervolume = hv.compute(ref_point)
                    except:
                        hypervolume = 0.0
                    
                    # 计算帕累托前沿的真实目标值 (用于帕累托前沿图)
                    real_pareto_objectives = []
                    current_individuals = pop.get_x()
                    for idx in pareto_indices:
                        individual = current_individuals[idx]
                        solution = problem._decode_solution(individual)
                        if solution is not None:
                            real_energy, real_delay = problem._calculate_objectives(solution)
                            real_pareto_objectives.append((real_energy, real_delay))
                    
                    # 对真实目标值再次进行帕累托筛选
                    if real_pareto_objectives:
                        real_objectives_array = np.array(real_pareto_objectives)
                        real_pareto_indices = pg.non_dominated_front_2d(real_objectives_array)
                        pareto_front_points = [real_pareto_objectives[i] for i in real_pareto_indices]
                    else:
                        pareto_front_points = []
                else:
                    pareto_fitness1_range = "N/A"
                    pareto_fitness2_range = "N/A"
                    hypervolume = 0.0
                    pareto_front_points = []
                
                # 记录进化数据
                evolution_data['generations'].append(gen + 1)
                evolution_data['pareto_count'].append(pareto_count)
                evolution_data['min_energy'].append(min_fitness1)
                evolution_data['avg_energy'].append(avg_fitness1)
                evolution_data['min_delay'].append(min_fitness2)
                evolution_data['avg_delay'].append(avg_fitness2)
                evolution_data['hypervolume'].append(hypervolume)
                evolution_data['pareto_fronts'].append(pareto_front_points)
                
                # 打印当代信息
                print(f"第{gen+1:3d}代 | "
                      f"帕累托解: {pareto_count:2d} | "
                      f"适应度1: {min_fitness1:6.1f}-{max_fitness1:6.1f} (avg:{avg_fitness1:6.1f}) | "
                      f"适应度2: {min_fitness2:6.1f}-{max_fitness2:6.1f} (avg:{avg_fitness2:6.1f}) | "
                      f"前沿适应度1: {pareto_fitness1_range} | "
                      f"前沿适应度2: {pareto_fitness2_range}")
                
                # 每10代或最后一代打印详细信息
                if (gen + 1) % 10 == 0 or gen == generations - 1:
                    print("-" * 80)
                    print(f"第{gen+1}代详细统计:")
                    print(f"  种群大小: {len(pop)}")
                    print(f"  帕累托前沿解数: {pareto_count}")
                    print(f"  适应度1统计: 最小={min_fitness1:.1f}, 最大={max_fitness1:.1f}, 平均={avg_fitness1:.1f}")
                    print(f"  适应度2统计: 最小={min_fitness2:.1f}, 最大={max_fitness2:.1f}, 平均={avg_fitness2:.1f}")
                    
                    if pareto_count > 0:
                        print(f"  帕累托前沿适应度1范围: {pareto_fitness1_range}")
                        print(f"  帕累托前沿适应度2范围: {pareto_fitness2_range}")
                        
                        # 显示帕累托前沿的前3个解
                        print(f"  帕累托前沿解示例 (前3个):")
                        for i, idx in enumerate(pareto_indices[:3]):
                            fitness1_val = fitness1[idx]
                            fitness2_val = fitness2[idx]
                            print(f"    解{i+1}: 适应度1={fitness1_val:.1f}, 适应度2={fitness2_val:.1f}")
                    
                    print("-" * 80)
        
        # 提取最终帕累托前沿 - 使用真实目标值筛选
        final_fitness = pop.get_f()
        final_individuals = pop.get_x()
        
        # 计算所有个体的真实目标值
        real_objectives = []
        valid_solutions = []
        for idx in range(len(final_individuals)):
            individual = final_individuals[idx]
            fitness = final_fitness[idx]
            
            solution = problem._decode_solution(individual)
            if solution is not None:
                total_energy, total_delay = problem._calculate_objectives(solution)
                real_objectives.append([total_energy, total_delay])
                valid_solutions.append({
                    'energy': total_energy,
                    'delay': total_delay,
                    'fitness': fitness,
                    'individual': individual,
                    'idx': idx
                })
        
        # 使用真实目标值进行帕累托前沿筛选
        if real_objectives:
            real_objectives_array = np.array(real_objectives)
            real_pareto_indices = pg.non_dominated_front_2d(real_objectives_array)
            pareto_front = [valid_solutions[i] for i in real_pareto_indices]
        else:
            pareto_front = []
        
        if verbose:
            print("\n🎉 进化完成!")
            print(f"最终帕累托前沿解数量: {len(pareto_front)} (基于真实目标值筛选)")
            if pareto_front:
                energies = [sol['energy'] for sol in pareto_front]
                delays = [sol['delay'] for sol in pareto_front]
                print(f"最终能耗范围: {min(energies):.1f} - {max(energies):.1f}")
                print(f"最终延误范围: {min(delays):.1f} - {max(delays):.1f}")
                print("注: 最终帕累托前沿基于真实目标值筛选，进化过程显示的是包含约束惩罚的适应度值")
        
        # 选择一个前沿解进行可视化
        if pareto_front:
            # 选择能耗最小的解进行可视化
            best_solution = min(pareto_front, key=lambda x: x['energy'])
            selected_schedule = _convert_pygmo_solution_to_schedule(best_solution['individual'], problem, tasks)
            
            if verbose and selected_schedule:
                print(f"\n选择前沿解进行可视化 (能耗={best_solution['energy']:.1f}, 延误={best_solution['delay']:.1f})")
                _visualize_pygmo_solution(selected_schedule)
        
        return {
            'pareto_front': pareto_front,
            'problem': problem,
            'population': pop,
            'algorithm': algo,
            'evolution_data': evolution_data
        }
        
    except Exception as e:
        print(f"NSGA-II求解错误: {e}")
        return None

def _convert_pygmo_solution_to_schedule(individual, problem, tasks):
    """
    将PyGMO解转换为标准调度格式
    """
    try:
        solution = problem._decode_solution(individual)
        if not solution:
            return []
        
        schedule = []
        for i in range(problem.num_tasks):
            # 找到执行此任务的eVTOL
            evtol_id = None
            for c, chain in enumerate(problem.task_chains):
                if i in chain:
                    for k in range(problem.num_evtols):
                        for t in range(problem.time_horizon):
                            if solution['y'][c, k, t] == 1:
                                evtol_id = k
                                break
                    break
            
            # 找到选择的航线
            route_id = None
            for h in range(problem.num_routes):
                if solution['z'][i, h] == 1:
                    route_id = h
                    break
            
            if evtol_id is not None and route_id is not None:
                delay = max(0, solution['task_start'][i] - tasks[i]['earliest_start'])
                schedule.append({
                    "task_id": i,
                    "evtol_id": evtol_id,
                    "start_time": int(solution['task_start'][i]),
                    "end_time": int(solution['task_end'][i]),
                    "route": route_id,
                    "from": tasks[i]['from'],
                    "to": tasks[i]['to'],
                    "delay": delay
                })
        
        return schedule
    except Exception as e:
        print(f"转换PyGMO解失败: {e}")
        return []

def _visualize_pygmo_solution(schedule):
    """
    可视化PyGMO解
    """
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from visualization import visualize_schedule_table, visualize_schedule_gantt
    
    if schedule:
        visualize_schedule_table(schedule, "PyGMO NSGA-II", "picture_result/evtol_schedule_table_pygmo_nsga2.png")
        visualize_schedule_gantt(schedule, "PyGMO NSGA-II", "picture_result/evtol_schedule_pygmo_nsga2.png")


def visualize_evolution_curves(evolution_data, save_path="picture_result/evolution_curves_pygmo_nsga2.png"):
    """
    可视化NSGA-II进化曲线
    
    参数:
        evolution_data: 进化过程数据
        save_path: 保存路径
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.family'] = 'FangSong'
    
    generations = evolution_data['generations']
    
    # 创建子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 帕累托前沿解数量变化
    ax1.plot(generations, evolution_data['pareto_count'], 'b-', linewidth=2, marker='o', markersize=4)
    ax1.set_xlabel('代数')
    ax1.set_ylabel('帕累托前沿解数量')
    ax1.set_title('帕累托前沿解数量进化曲线')
    ax1.grid(True, alpha=0.3)
    
    # 2. 适应度1指标进化 (能耗+惩罚)
    ax2.plot(generations, evolution_data['min_energy'], 'r-', linewidth=2, label='最小适应度1', marker='o', markersize=3)
    ax2.plot(generations, evolution_data['avg_energy'], 'g-', linewidth=2, label='平均适应度1', marker='s', markersize=3)
    ax2.set_xlabel('代数')
    ax2.set_ylabel('适应度1 (能耗+惩罚)')
    ax2.set_title('适应度1指标进化曲线')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 适应度2指标进化 (延误+惩罚)
    ax3.plot(generations, evolution_data['min_delay'], 'purple', linewidth=2, label='最小适应度2', marker='o', markersize=3)
    ax3.plot(generations, evolution_data['avg_delay'], 'orange', linewidth=2, label='平均适应度2', marker='s', markersize=3)
    ax3.set_xlabel('代数')
    ax3.set_ylabel('适应度2 (延误+惩罚)')
    ax3.set_title('适应度2指标进化曲线')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 超体积进化
    ax4.plot(generations, evolution_data['hypervolume'], 'brown', linewidth=2, marker='o', markersize=4)
    ax4.set_xlabel('代数')
    ax4.set_ylabel('超体积')
    ax4.set_title('帕累托前沿质量进化曲线 (超体积)')
    ax4.grid(True, alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"进化曲线已保存到: {save_path}")


def visualize_pareto_front_evolution(evolution_data, save_path="picture_result/pareto_front_evolution_pygmo_nsga2.png", 
                                   show_generations=[1, 10, 50, 100, -1]):
    """
    可视化帕累托前沿的进化过程
    
    参数:
        evolution_data: 进化过程数据
        save_path: 保存路径
        show_generations: 要显示的代数 (-1表示最后一代)
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.family'] = 'FangSong'
    
    plt.figure(figsize=(12, 8))
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    markers = ['o', 's', '^', 'D', 'v']
    
    for i, gen_idx in enumerate(show_generations):
        if gen_idx == -1:
            gen_idx = len(evolution_data['pareto_fronts']) - 1
            label = f"第{len(evolution_data['pareto_fronts'])}代 (最终)"
        else:
            gen_idx = gen_idx - 1  # 转换为数组索引
            label = f"第{gen_idx + 1}代"
        
        if gen_idx < len(evolution_data['pareto_fronts']):
            pareto_points = evolution_data['pareto_fronts'][gen_idx]
            if pareto_points:
                energies, delays = zip(*pareto_points)
                plt.scatter(energies, delays, 
                          c=colors[i % len(colors)], 
                          marker=markers[i % len(markers)],
                          s=60, alpha=0.7, label=label,
                          edgecolors='black', linewidth=0.5)
    
    plt.xlabel('总能耗 (真实目标值)')
    plt.ylabel('总延误时间 (分钟, 真实目标值)')
    plt.title('帕累托前沿进化过程 (基于真实目标值筛选)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存图形
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"帕累托前沿进化图已保存到: {save_path}") 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
改进的eVTOL调度NSGA-II优化模块 - 基于PyGMO
使用改进的NSGA-II算法求解eVTOL调度多目标优化问题

改进内容:
1. 变邻域搜索(VNS)变异算子
2. 基于Q-learning的交叉与变异概率自适应调整
"""

import pygmo as pg
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import random
from collections import defaultdict, deque
import copy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_definitions import get_evtols, get_locations, get_tasks
from gurobi.evtol_scheduling_gurobi import generate_task_chains
import itertools

class QLearningGAController:
    """
    基于Q-learning的遗传算法参数自适应控制器
    
    动态调整交叉概率和变异概率以优化算法性能
    """
    
    def __init__(self, n_actions=9, learning_rate=0.1, epsilon=0.3, 
                 discount_factor=0.95, epsilon_decay=0.99):
        # Q-learning参数
        self.n_actions = n_actions  # 动作空间大小 (3x3的交叉率变异率组合)
        self.learning_rate = learning_rate
        self.epsilon = epsilon  # 探索率
        self.discount_factor = discount_factor
        self.epsilon_decay = epsilon_decay
        
        # 动作空间 (交叉率, 变异率)
        self.crossover_rates = [0.6, 0.8, 0.9]
        self.mutation_rates = [0.05, 0.1, 0.2]
        self.actions = [(cr, mr) for cr in self.crossover_rates for mr in self.mutation_rates]
        
        # Q表和状态
        self.q_table = {}
        self.current_state = None
        self.current_action = 0
        self.previous_state = None
        self.previous_action = 0
        
        # 性能历史
        self.performance_history = deque(maxlen=10)
        self.hypervolume_history = deque(maxlen=5)
        
        print(f"Q-learning控制器初始化: {len(self.actions)}个动作组合")
        for i, (cr, mr) in enumerate(self.actions):
            print(f"  动作{i}: 交叉率={cr}, 变异率={mr}")
    
    def get_state(self, generation, hypervolume, improvement_ratio, diversity_ratio):
        """
        根据当前进化状态获取状态编码
        
        状态特征:
        - 进化阶段 (early/middle/late)
        - 超体积趋势 (improving/stable/declining)
        - 改善率 (high/medium/low)
        - 多样性 (high/medium/low)
        """
        # 进化阶段
        if generation < 50:
            stage = 0  # early
        elif generation < 150:
            stage = 1  # middle
        else:
            stage = 2  # late
        
        # 超体积趋势
        if len(self.hypervolume_history) >= 3:
            recent_trend = np.mean(list(self.hypervolume_history)[-3:]) - np.mean(list(self.hypervolume_history)[-5:-2])
            if recent_trend > 0.01:
                hv_trend = 0  # improving
            elif recent_trend > -0.01:
                hv_trend = 1  # stable
            else:
                hv_trend = 2  # declining
        else:
            hv_trend = 1  # stable (default)
        
        # 改善率级别
        if improvement_ratio > 0.05:
            improvement_level = 0  # high
        elif improvement_ratio > 0.01:
            improvement_level = 1  # medium
        else:
            improvement_level = 2  # low
        
        # 多样性级别
        if diversity_ratio > 0.7:
            diversity_level = 0  # high
        elif diversity_ratio > 0.4:
            diversity_level = 1  # medium
        else:
            diversity_level = 2  # low
        
        state = (stage, hv_trend, improvement_level, diversity_level)
        return state
    
    def get_action(self, state):
        """
        根据当前状态选择动作 (epsilon-greedy策略)
        """
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)
        
        # epsilon-greedy选择
        if random.random() < self.epsilon:
            action = random.randint(0, self.n_actions - 1)
        else:
            action = np.argmax(self.q_table[state])
        
        return action
    
    def update_q_value(self, reward):
        """
        更新Q值
        """
        if self.previous_state is not None and self.current_state is not None:
            if self.previous_state not in self.q_table:
                self.q_table[self.previous_state] = np.zeros(self.n_actions)
            if self.current_state not in self.q_table:
                self.q_table[self.current_state] = np.zeros(self.n_actions)
            
            # Q-learning更新公式
            current_q = self.q_table[self.previous_state][self.previous_action]
            max_future_q = np.max(self.q_table[self.current_state])
            new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_future_q - current_q)
            self.q_table[self.previous_state][self.previous_action] = new_q
    
    def update(self, generation, hypervolume, pareto_count, avg_fitness_improvement):
        """
        更新控制器状态并获取新的参数设置
        """
        # 计算性能指标
        self.hypervolume_history.append(hypervolume)
        
        # 计算改善率
        if len(self.performance_history) > 0:
            improvement_ratio = avg_fitness_improvement / (np.mean(self.performance_history) + 1e-6)
        else:
            improvement_ratio = 0.1
        
        # 计算多样性比例
        diversity_ratio = min(pareto_count / 50.0, 1.0)  # 假设理想帕累托解数为50
        
        # 更新状态
        self.previous_state = self.current_state
        self.previous_action = self.current_action
        self.current_state = self.get_state(generation, hypervolume, improvement_ratio, diversity_ratio)
        
        # 计算奖励
        reward = self._calculate_reward(hypervolume, improvement_ratio, diversity_ratio)
        
        # 更新Q值
        self.update_q_value(reward)
        
        # 选择新动作
        self.current_action = self.get_action(self.current_state)
        
        # 更新探索率
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, 0.05)
        
        # 记录性能
        self.performance_history.append(avg_fitness_improvement)
        
        # 返回新的参数设置
        cr, mr = self.actions[self.current_action]
        return cr, mr
    
    def _calculate_reward(self, hypervolume, improvement_ratio, diversity_ratio):
        """
        计算奖励函数
        """
        # 基础奖励：超体积改善
        hv_reward = hypervolume * 10
        
        # 改善奖励
        improvement_reward = improvement_ratio * 5
        
        # 多样性奖励
        diversity_reward = diversity_ratio * 2
        
        # 综合奖励
        total_reward = hv_reward + improvement_reward + diversity_reward
        
        return total_reward


class VariableNeighborhoodSearch:
    """
    变邻域搜索(VNS)变异算子
    
    使用多种邻域结构进行局部搜索，提高解的质量
    """
    
    def __init__(self, problem):
        self.problem = problem
        self.neighborhood_structures = [
            self._neighborhood_1_swap_evtol,
            self._neighborhood_2_swap_route,
            self._neighborhood_3_adjust_time,
            self._neighborhood_4_swap_chain_order,
            self._neighborhood_5_local_optimization
        ]
        self.max_neighborhoods = len(self.neighborhood_structures)
    
    def apply_vns_mutation(self, individual, mutation_rate=0.1):
        """
        应用变邻域搜索变异
        
        VNS算法流程:
        1. 从第一个邻域结构开始
        2. 在当前邻域中生成邻居解
        3. 如果邻居解更好，接受并回到第一个邻域
        4. 否则，转到下一个邻域结构
        5. 重复直到所有邻域都尝试过
        """
        if random.random() > mutation_rate:
            return individual
        
        current_solution = individual.copy()
        current_fitness = self._evaluate_solution(current_solution)
        
        k = 0  # 当前邻域结构索引
        max_iterations = 10  # 最大迭代次数
        iteration = 0
        
        while k < self.max_neighborhoods and iteration < max_iterations:
            # 在邻域k中生成邻居解
            neighbor = self.neighborhood_structures[k](current_solution.copy())
            neighbor_fitness = self._evaluate_solution(neighbor)
            
            # 如果邻居解更好 (多目标下使用支配关系判断)
            if self._is_better_solution(neighbor_fitness, current_fitness):
                current_solution = neighbor
                current_fitness = neighbor_fitness
                k = 0  # 回到第一个邻域
            else:
                k += 1  # 尝试下一个邻域
            
            iteration += 1
        
        return current_solution
    
    def _neighborhood_1_swap_evtol(self, solution):
        """
        邻域1: 交换两个任务串的eVTOL分配
        """
        if self.problem.num_chains < 2:
            return solution
        
        # 随机选择两个不同的任务串
        c1, c2 = random.sample(range(self.problem.num_chains), 2)
        
        # 交换eVTOL分配
        idx1 = c1 * 2  # y_evtol[c1]的索引
        idx2 = c2 * 2  # y_evtol[c2]的索引
        
        solution[idx1], solution[idx2] = solution[idx2], solution[idx1]
        
        return solution
    
    def _neighborhood_2_swap_route(self, solution):
        """
        邻域2: 改变一个任务的航线选择
        """
        if self.problem.num_tasks == 0:
            return solution
        
        # 随机选择一个任务
        task_idx = random.randint(0, self.problem.num_tasks - 1)
        
        # 改变航线选择
        route_var_idx = self.problem.num_chains * 2 + task_idx
        current_route = solution[route_var_idx]
        new_route = random.randint(0, self.problem.num_routes - 1)
        solution[route_var_idx] = new_route
        
        return solution
    
    def _neighborhood_3_adjust_time(self, solution):
        """
        邻域3: 微调任务串开始时间
        """
        if self.problem.num_chains == 0:
            return solution
        
        # 随机选择一个任务串
        chain_idx = random.randint(0, self.problem.num_chains - 1)
        
        # 微调开始时间
        time_var_idx = chain_idx * 2 + 1  # y_time[c]的索引
        current_time = solution[time_var_idx]
        
        # 在当前时间附近进行小幅调整
        adjustment = random.randint(-30, 30)
        new_time = max(0, min(self.problem.time_horizon - 1, current_time + adjustment))
        solution[time_var_idx] = new_time
        
        return solution
    
    def _neighborhood_4_swap_chain_order(self, solution):
        """
        邻域4: 交换同一eVTOL上两个任务串的执行顺序
        """
        # 找到分配给同一eVTOL的任务串
        evtol_chains = defaultdict(list)
        for c in range(self.problem.num_chains):
            evtol_id = solution[c * 2]
            evtol_chains[evtol_id].append(c)
        
        # 选择有多个任务串的eVTOL
        multi_chain_evtols = [k for k, v in evtol_chains.items() if len(v) >= 2]
        
        if multi_chain_evtols:
            evtol_id = random.choice(multi_chain_evtols)
            chains = evtol_chains[evtol_id]
            c1, c2 = random.sample(chains, 2)
            
            # 交换时间分配
            time_idx1 = c1 * 2 + 1
            time_idx2 = c2 * 2 + 1
            solution[time_idx1], solution[time_idx2] = solution[time_idx2], solution[time_idx1]
        
        return solution
    
    def _neighborhood_5_local_optimization(self, solution):
        """
        邻域5: 对单个任务串进行局部优化
        """
        if self.problem.num_chains == 0:
            return solution
        
        # 随机选择一个任务串进行局部优化
        chain_idx = random.randint(0, self.problem.num_chains - 1)
        
        # 尝试为该任务串找到更好的时间安排
        best_solution = solution.copy()
        best_fitness = self._evaluate_solution(solution)
        
        # 尝试不同的时间窗口
        time_var_idx = chain_idx * 2 + 1
        current_time = solution[time_var_idx]
        
        for time_offset in [-60, -30, 0, 30, 60]:
            new_time = max(0, min(self.problem.time_horizon - 1, current_time + time_offset))
            test_solution = solution.copy()
            test_solution[time_var_idx] = new_time
            
            test_fitness = self._evaluate_solution(test_solution)
            if self._is_better_solution(test_fitness, best_fitness):
                best_solution = test_solution
                best_fitness = test_fitness
        
        return best_solution
    
    def _evaluate_solution(self, individual):
        """
        评估解的适应度
        """
        try:
            solution = self.problem._decode_solution(individual)
            if solution is None:
                return [float('inf'), float('inf')]
            
            energy, delay = self.problem._calculate_objectives(solution)
            penalty = self.problem._calculate_penalty(solution)
            
            return [energy + penalty, delay + penalty]
        except:
            return [float('inf'), float('inf')]
    
    def _is_better_solution(self, fitness1, fitness2):
        """
        判断fitness1是否支配fitness2
        """
        if any(f == float('inf') for f in fitness1):
            return False
        if any(f == float('inf') for f in fitness2):
            return True
        
        # Pareto支配判断
        better_in_all = all(f1 <= f2 for f1, f2 in zip(fitness1, fitness2))
        better_in_some = any(f1 < f2 for f1, f2 in zip(fitness1, fitness2))
        
        return better_in_all and better_in_some


class ImprovedNSGA2:
    """
    改进的NSGA-II算法
    
    整合变邻域搜索变异和Q-learning参数自适应
    """
    
    def __init__(self, problem, population_size=100, generations=200):
        self.problem = problem
        self.population_size = population_size
        self.generations = generations
        
        # 创建PyGMO问题对象
        self.pg_problem = pg.problem(problem)
        
        # 初始化Q-learning控制器
        self.q_controller = QLearningGAController()
        
        # 初始化VNS变异算子
        self.vns_mutation = VariableNeighborhoodSearch(problem)
        
        # 获取初始参数
        self.current_cr = 0.8
        self.current_mr = 0.1
        
        print(f"改进NSGA-II初始化完成")
        print(f"  种群大小: {population_size}")
        print(f"  进化代数: {generations}")
        print(f"  集成组件: Q-learning参数控制 + VNS变异")
    
    def evolve(self, verbose=True):
        """
        执行改进的NSGA-II进化过程
        """
        if verbose:
            print("=== 改进NSGA-II多目标优化求解 ===")
            print("集成变邻域搜索变异 + Q-learning参数自适应")
        
        # 调整种群大小以符合NSGA-II要求
        if self.population_size < 8:
            self.population_size = 8
        if self.population_size % 4 != 0:
            self.population_size = ((self.population_size // 4) + 1) * 4
        
        # 创建初始种群
        pop = pg.population(self.pg_problem, self.population_size)
        
        # 记录进化过程数据
        evolution_data = {
            'generations': [],
            'pareto_count': [],
            'min_energy': [],
            'avg_energy': [],
            'min_delay': [],
            'avg_delay': [],
            'hypervolume': [],
            'pareto_fronts': [],
            'parameter_history': {'crossover_rate': [], 'mutation_rate': []}
        }
        
        if verbose:
            print(f"初始种群大小: {len(pop)}")
            print(f"决策变量维度: {self.problem.dimensions}")
            print("=" * 80)
        
        # 进化循环
        for gen in range(self.generations):
            # 创建NSGA-II算法实例 (使用当前参数)
            nsga2 = pg.nsga2(
                gen=1,
                cr=self.current_cr,
                eta_c=10,
                m=self.current_mr,  # 基础变异率，VNS会额外应用
                eta_m=10
            )
            algo = pg.algorithm(nsga2)
            
            # 标准NSGA-II进化一代
            pop = algo.evolve(pop)
            
            # 应用VNS变异到部分个体
            self._apply_vns_to_population(pop)
            
            # 收集统计信息
            fitness_values = pop.get_f()
            stats = self._calculate_generation_stats(fitness_values, gen)
            
            # 更新Q-learning控制器并获取新参数
            self.current_cr, self.current_mr = self.q_controller.update(
                gen, stats['hypervolume'], stats['pareto_count'], stats['avg_improvement']
            )
            
            # 记录进化数据
            evolution_data['generations'].append(gen)
            evolution_data['pareto_count'].append(stats['pareto_count'])
            evolution_data['min_energy'].append(stats['min_energy'])
            evolution_data['avg_energy'].append(stats['avg_energy'])
            evolution_data['min_delay'].append(stats['min_delay'])
            evolution_data['avg_delay'].append(stats['avg_delay'])
            evolution_data['hypervolume'].append(stats['hypervolume'])
            evolution_data['pareto_fronts'].append(stats['pareto_front'])
            evolution_data['parameter_history']['crossover_rate'].append(self.current_cr)
            evolution_data['parameter_history']['mutation_rate'].append(self.current_mr)
            
            # 打印进度信息
            if verbose and (gen % 20 == 0 or gen == self.generations - 1):
                print(f"代数 {gen:3d}: "
                      f"帕累托解={stats['pareto_count']:2d}, "
                      f"能耗={stats['min_energy']:6.1f}-{stats['max_energy']:6.1f}, "
                      f"延误={stats['min_delay']:6.1f}-{stats['max_delay']:6.1f}, "
                      f"HV={stats['hypervolume']:6.3f}, "
                      f"CR={self.current_cr:.2f}, MR={self.current_mr:.3f}")
        
        # 提取最终帕累托前沿
        final_pareto_front = self._extract_pareto_front(pop)
        
        return {
            'pareto_front': final_pareto_front,
            'problem': self.problem,
            'population': pop,
            'evolution_data': evolution_data,
            'q_controller': self.q_controller
        }
    
    def _apply_vns_to_population(self, pop):
        """
        对种群中的部分个体应用VNS变异
        """
        individuals = pop.get_x()
        n_vns_individuals = max(1, len(individuals) // 10)  # 对10%的个体应用VNS
        
        # 随机选择个体进行VNS变异
        vns_indices = random.sample(range(len(individuals)), n_vns_individuals)
        
        for idx in vns_indices:
            original = individuals[idx].copy()
            mutated = self.vns_mutation.apply_vns_mutation(original, mutation_rate=0.3)
            
            # 检查变异后的个体是否有效
            if self._is_valid_individual(mutated):
                # 用变异后的个体替换原个体
                pop.set_x(idx, mutated)
    
    def _is_valid_individual(self, individual):
        """
        检查个体是否在定义域内
        """
        bounds = self.pg_problem.get_bounds()
        lower_bounds, upper_bounds = bounds
        
        return all(lb <= val <= ub for val, lb, ub in zip(individual, lower_bounds, upper_bounds))
    
    def _calculate_generation_stats(self, fitness_values, generation):
        """
        计算当代统计信息
        """
        fitness1 = fitness_values[:, 0]  # 能耗+惩罚
        fitness2 = fitness_values[:, 1]  # 延误+惩罚
        
        # 基本统计
        min_fitness1 = np.min(fitness1)
        max_fitness1 = np.max(fitness1)
        avg_fitness1 = np.mean(fitness1)
        
        min_fitness2 = np.min(fitness2)
        max_fitness2 = np.max(fitness2)
        avg_fitness2 = np.mean(fitness2)
        
        # 帕累托前沿
        pareto_indices = pg.non_dominated_front_2d(fitness_values)
        pareto_count = len(pareto_indices)
        
        # 超体积计算
        try:
            ref_point = [max_fitness1 * 1.1, max_fitness2 * 1.1]
            hv = pg.hypervolume(fitness_values[pareto_indices])
            hypervolume = hv.compute(ref_point)
        except:
            hypervolume = 0.0
        
        # 改善度计算
        if hasattr(self, 'previous_avg_fitness'):
            avg_improvement = abs(avg_fitness1 - self.previous_avg_fitness[0]) + abs(avg_fitness2 - self.previous_avg_fitness[1])
        else:
            avg_improvement = 1.0
        
        self.previous_avg_fitness = (avg_fitness1, avg_fitness2)
        
        return {
            'pareto_count': pareto_count,
            'min_energy': min_fitness1,
            'max_energy': max_fitness1,
            'avg_energy': avg_fitness1,
            'min_delay': min_fitness2,
            'max_delay': max_fitness2,
            'avg_delay': avg_fitness2,
            'hypervolume': hypervolume,
            'avg_improvement': avg_improvement,
            'pareto_front': pareto_indices
        }
    
    def _extract_pareto_front(self, pop):
        """
        提取最终帕累托前沿
        """
        fitness_values = pop.get_f()
        individuals = pop.get_x()
        pareto_indices = pg.non_dominated_front_2d(fitness_values)
        
        pareto_front = []
        for idx in pareto_indices:
            individual = individuals[idx]
            solution = self.problem._decode_solution(individual)
            
            if solution is not None:
                energy, delay = self.problem._calculate_objectives(solution)
                pareto_front.append({
                    'individual': individual,
                    'energy': energy,
                    'delay': delay,
                    'fitness': fitness_values[idx]
                })
        
        return pareto_front


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


def solve_improved_nsga2(tasks, evtols, task_chains, time_horizon=720, 
                        population_size=100, generations=200, verbose=True):
    """
    使用改进的NSGA-II算法求解eVTOL调度问题
    
    改进包括:
    1. 变邻域搜索(VNS)变异算子
    2. 基于Q-learning的交叉与变异概率自适应调整
    
    参数:
        tasks: 任务列表
        evtols: eVTOL列表
        task_chains: 任务串列表
        time_horizon: 时间范围
        population_size: 种群大小
        generations: 进化代数
        verbose: 是否显示详细信息
    
    返回:
        包含帕累托前沿、进化数据等的结果字典
    """
    if verbose:
        print("=== 改进NSGA-II多目标优化求解 ===")
        print("集成改进: 变邻域搜索变异 + Q-learning参数自适应")
        print(f"问题规模: {len(tasks)}个任务, {len(evtols)}架eVTOL, {len(task_chains)}个任务串")
    
    try:
        # 创建问题实例
        problem = eVTOLSchedulingProblem(tasks, evtols, task_chains, time_horizon)
        
        # 创建改进的NSGA-II算法
        improved_algo = ImprovedNSGA2(problem, population_size, generations)
        
        # 执行优化
        result = improved_algo.evolve(verbose)
        
        # 提取帕累托前沿的真实目标值
        pareto_front_with_objectives = []
        for solution in result['pareto_front']:
            pareto_front_with_objectives.append({
                'individual': solution['individual'],
                'energy': solution['energy'],
                'delay': solution['delay'],
                'fitness': solution['fitness']
            })
        
        # 可视化最佳解
        if verbose and pareto_front_with_objectives:
            best_solution = min(pareto_front_with_objectives, key=lambda x: x['energy'])
            selected_schedule = _convert_pygmo_solution_to_schedule(
                best_solution['individual'], problem, tasks
            )
            
            if selected_schedule:
                print(f"\n选择前沿解进行可视化 (能耗={best_solution['energy']:.1f}, 延误={best_solution['delay']:.1f})")
                _visualize_improved_solution(selected_schedule)
        
        # 更新结果
        result['pareto_front'] = pareto_front_with_objectives
        
        return result
        
    except Exception as e:
        print(f"改进NSGA-II求解错误: {e}")
        import traceback
        traceback.print_exc()
        return None


def visualize_improved_evolution_curves(evolution_data, save_path="picture_result/evolution_curves_improved_nsga2.png"):
    """
    可视化改进NSGA-II的进化曲线
    
    包括参数自适应过程的展示
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.family'] = 'FangSong'
    
    generations = evolution_data['generations']
    
    # 创建子图
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(16, 14))
    
    # 1. 帕累托前沿解数量变化
    ax1.plot(generations, evolution_data['pareto_count'], 'b-', linewidth=2, marker='o', markersize=4)
    ax1.set_xlabel('代数')
    ax1.set_ylabel('帕累托前沿解数量')
    ax1.set_title('帕累托前沿解数量变化')
    ax1.grid(True, alpha=0.3)
    
    # 2. 能耗目标变化
    ax2.plot(generations, evolution_data['min_energy'], 'g-', linewidth=2, label='最小能耗', marker='s', markersize=3)
    ax2.plot(generations, evolution_data['avg_energy'], 'r--', linewidth=2, label='平均能耗', alpha=0.7)
    ax2.set_xlabel('代数')
    ax2.set_ylabel('能耗')
    ax2.set_title('能耗目标进化')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 延误目标变化
    ax3.plot(generations, evolution_data['min_delay'], 'b-', linewidth=2, label='最小延误', marker='^', markersize=3)
    ax3.plot(generations, evolution_data['avg_delay'], 'orange', linestyle='--', linewidth=2, label='平均延误', alpha=0.7)
    ax3.set_xlabel('代数')
    ax3.set_ylabel('延误时间 (分钟)')
    ax3.set_title('延误目标进化')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 超体积变化
    ax4.plot(generations, evolution_data['hypervolume'], 'purple', linewidth=2, marker='d', markersize=3)
    ax4.set_xlabel('代数')
    ax4.set_ylabel('超体积')
    ax4.set_title('超体积指标')
    ax4.grid(True, alpha=0.3)
    
    # 5. 交叉率自适应变化
    ax5.plot(generations, evolution_data['parameter_history']['crossover_rate'], 
             'red', linewidth=2, marker='o', markersize=3, label='交叉率')
    ax5.set_xlabel('代数')
    ax5.set_ylabel('交叉率')
    ax5.set_title('Q-learning交叉率自适应')
    ax5.set_ylim(0.5, 1.0)
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # 6. 变异率自适应变化
    ax6.plot(generations, evolution_data['parameter_history']['mutation_rate'], 
             'blue', linewidth=2, marker='s', markersize=3, label='变异率')
    ax6.set_xlabel('代数')
    ax6.set_ylabel('变异率')
    ax6.set_title('Q-learning变异率自适应')
    ax6.set_ylim(0.0, 0.25)
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"改进NSGA-II进化曲线已保存到: {save_path}")


def visualize_improved_pareto_front(pareto_front, save_path="picture_result/pareto_front_improved_nsga2.png"):
    """
    可视化改进NSGA-II的帕累托前沿
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.family'] = 'FangSong'
    
    if not pareto_front:
        print("警告: 帕累托前沿为空，无法绘制图形")
        return
    
    # 提取目标值
    energies = [sol['energy'] for sol in pareto_front]
    delays = [sol['delay'] for sol in pareto_front]
    
    plt.figure(figsize=(10, 8))
    
    # 绘制帕累托前沿点
    plt.scatter(energies, delays, c='red', s=80, alpha=0.7, 
                edgecolors='black', linewidth=1, label=f'帕累托解 ({len(pareto_front)}个)')
    
    # 连接帕累托前沿
    if len(pareto_front) > 1:
        # 按能耗排序以便连线
        sorted_solutions = sorted(pareto_front, key=lambda x: x['energy'])
        sorted_energies = [sol['energy'] for sol in sorted_solutions]
        sorted_delays = [sol['delay'] for sol in sorted_solutions]
        plt.plot(sorted_energies, sorted_delays, 'b--', alpha=0.5, linewidth=1)
    
    # 标注特殊点
    min_energy_sol = min(pareto_front, key=lambda x: x['energy'])
    min_delay_sol = min(pareto_front, key=lambda x: x['delay'])
    
    plt.scatter(min_energy_sol['energy'], min_energy_sol['delay'], 
                c='green', s=120, marker='*', label='最低能耗解', 
                edgecolors='black', linewidth=1)
    plt.scatter(min_delay_sol['energy'], min_delay_sol['delay'], 
                c='blue', s=120, marker='*', label='最低延误解',
                edgecolors='black', linewidth=1)
    
    plt.xlabel('总能耗')
    plt.ylabel('总延误时间 (分钟)')
    plt.title('改进NSGA-II帕累托前沿\n(VNS变异 + Q-learning参数自适应)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 添加统计信息
    energy_range = max(energies) - min(energies)
    delay_range = max(delays) - min(delays)
    plt.text(0.02, 0.98, f'能耗范围: {min(energies):.1f} - {max(energies):.1f}\n'
                         f'延误范围: {min(delays):.1f} - {max(delays):.1f}\n'
                         f'解的数量: {len(pareto_front)}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"改进NSGA-II帕累托前沿图已保存到: {save_path}")


def analyze_algorithm_improvements(standard_result, improved_result):
    """
    对比分析标准NSGA-II和改进NSGA-II的性能
    
    参数:
        standard_result: 标准NSGA-II结果
        improved_result: 改进NSGA-II结果
    """
    print("=== 算法改进效果分析 ===")
    
    if not standard_result or not improved_result:
        print("缺少对比数据，无法进行分析")
        return
    
    std_pareto = standard_result.get('pareto_front', [])
    imp_pareto = improved_result.get('pareto_front', [])
    
    print(f"\n1. 帕累托前沿质量对比:")
    print(f"   标准NSGA-II: {len(std_pareto)} 个解")
    print(f"   改进NSGA-II: {len(imp_pareto)} 个解")
    
    if std_pareto and imp_pareto:
        # 能耗对比
        std_min_energy = min(sol['energy'] for sol in std_pareto)
        imp_min_energy = min(sol['energy'] for sol in imp_pareto)
        energy_improvement = (std_min_energy - imp_min_energy) / std_min_energy * 100
        
        # 延误对比
        std_min_delay = min(sol['delay'] for sol in std_pareto)
        imp_min_delay = min(sol['delay'] for sol in imp_pareto)
        delay_improvement = (std_min_delay - imp_min_delay) / std_min_delay * 100
        
        print(f"\n2. 目标函数改进:")
        print(f"   最低能耗: {std_min_energy:.1f} → {imp_min_energy:.1f} "
              f"(改进 {energy_improvement:+.1f}%)")
        print(f"   最低延误: {std_min_delay:.1f} → {imp_min_delay:.1f} "
              f"(改进 {delay_improvement:+.1f}%)")
    
    # 收敛性分析
    if 'evolution_data' in improved_result:
        evolution_data = improved_result['evolution_data']
        if 'parameter_history' in evolution_data:
            print(f"\n3. 参数自适应特征:")
            cr_history = evolution_data['parameter_history']['crossover_rate']
            mr_history = evolution_data['parameter_history']['mutation_rate']
            
            print(f"   交叉率变化: {cr_history[0]:.3f} → {cr_history[-1]:.3f}")
            print(f"   变异率变化: {mr_history[0]:.3f} → {mr_history[-1]:.3f}")
            print(f"   参数调整次数: {len(set(cr_history))} (交叉率), {len(set(mr_history))} (变异率)")
    
    print(f"\n4. 算法改进总结:")
    print(f"   ✓ 变邻域搜索变异: 提高局部搜索能力")
    print(f"   ✓ Q-learning参数控制: 自适应调整算法参数")
    print(f"   ✓ 集成优化: 兼顾全局探索和局部开发")


def _visualize_improved_solution(schedule):
    """
    可视化改进算法的解
    """
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from visualization import visualize_schedule_table, visualize_schedule_gantt
    
    if schedule:
        visualize_schedule_table(schedule, "改进NSGA-II", "picture_result/evtol_schedule_table_improved_nsga2.png")
        visualize_schedule_gantt(schedule, "改进NSGA-II", "picture_result/evtol_schedule_improved_nsga2.png")


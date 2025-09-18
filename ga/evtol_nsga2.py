import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'FangSong'
import pandas as pd
from typing import Dict, List, Tuple, Any
import copy
from collections import defaultdict
import time

class Individual:
    """个体类，表示一个解决方案"""
    def __init__(self, chromosome: List[int]):
        self.chromosome = chromosome
        self.objectives = None  # (energy, delay)
        self.rank = None
        self.crowding_distance = 0.0
        self.schedule = None  # 解码后的调度方案
        
    def __lt__(self, other):
        """用于排序的比较函数"""
        if self.rank != other.rank:
            return self.rank < other.rank
        return self.crowding_distance > other.crowding_distance

class eVTOL_NSGA2:
    """eVTOL调度问题的NSGA-II求解器"""
    
    def __init__(self, tasks: List[Dict], evtols: List[Dict], task_chains: List[List[int]], 
                 population_size: int = 100, generations: int = 200, 
                 crossover_prob: float = 0.9, mutation_prob: float = 0.1):
        """
        初始化NSGA-II求解器
        使用与gurobi模块相同的问题建模理念：
        - 基于任务链的分配策略
        - 相同的目标函数：能耗 vs 延误
        - 相同的约束概念：时间窗、防撞、容量等
        
        参数:
            tasks: 任务列表
            evtols: eVTOL列表  
            task_chains: 任务链列表（与gurobi使用相同的任务链生成）
            population_size: 种群大小
            generations: 进化代数
            crossover_prob: 交叉概率
            mutation_prob: 变异概率
        """
        self.tasks = tasks
        self.evtols = evtols
        self.task_chains = task_chains
        self.pop_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        
        # 问题参数
        self.num_tasks = len(tasks)
        self.num_evtols = len(evtols)
        self.num_chains = len(task_chains)
        self.num_routes = 3
        self.time_horizon = 720
        
        # 编码长度 = 任务链分配 + 航线选择
        self.chain_assignment_length = self.num_chains
        self.route_selection_length = self.num_tasks
        self.chromosome_length = self.chain_assignment_length + self.route_selection_length
        
        print(f"初始化NSGA-II求解器:")
        print(f"  任务数: {self.num_tasks}")
        print(f"  eVTOL数: {self.num_evtols}")
        print(f"  任务链数: {self.num_chains}")
        print(f"  染色体长度: {self.chromosome_length}")
        
    def initialize_population(self) -> List[Individual]:
        """初始化种群"""
        population = []
        
        for _ in range(self.pop_size):
            # 任务链分配部分：每个任务链随机分配给一个eVTOL
            chain_assignment = [random.randint(0, self.num_evtols - 1) 
                              for _ in range(self.num_chains)]
            
            # 航线选择部分：每个任务随机选择一条航线
            route_selection = [random.randint(0, self.num_routes - 1) 
                             for _ in range(self.num_tasks)]
            
            chromosome = chain_assignment + route_selection
            individual = Individual(chromosome)
            population.append(individual)
            
        return population
    
    def decode_individual(self, individual: Individual) -> Dict:
        """
        解码个体为具体的调度方案
        
        返回:
            调度方案字典，包含任务分配、时间安排等信息
        """
        chromosome = individual.chromosome
        
        # 分离编码部分
        chain_assignment = chromosome[:self.chain_assignment_length]
        route_selection = chromosome[self.chain_assignment_length:]
        
        # 构建调度方案
        schedule = {
            "chain_assignments": [],
            "task_assignments": {},
            "task_routes": {},
            "task_start_times": {},
            "task_end_times": {},
            "evtol_schedules": {k: [] for k in range(self.num_evtols)}
        }
        
        # 任务链分配
        for chain_id, evtol_id in enumerate(chain_assignment):
            schedule["chain_assignments"].append({
                "chain_id": chain_id,
                "evtol_id": evtol_id,
                "tasks": self.task_chains[chain_id]
            })
            
            # 为任务链中的每个任务记录eVTOL分配
            for task_id in self.task_chains[chain_id]:
                schedule["task_assignments"][task_id] = evtol_id
        
        # 航线选择
        for task_id, route_id in enumerate(route_selection):
            schedule["task_routes"][task_id] = route_id
            
        # 计算任务时间安排
        self._calculate_task_times(schedule)
        
        return schedule
    
    def _calculate_task_times(self, schedule: Dict):
        """计算任务的开始和结束时间"""
        
        # 为每个eVTOL收集其分配的任务链
        evtol_chains = defaultdict(list)
        for assignment in schedule["chain_assignments"]:
            evtol_id = assignment["evtol_id"]
            chain_id = assignment["chain_id"]
            evtol_chains[evtol_id].append((chain_id, assignment["tasks"]))
        
        # 为每个eVTOL安排任务时间
        for evtol_id, chains in evtol_chains.items():
            current_time = 0
            
            # 按任务链的最早任务开始时间排序
            chains.sort(key=lambda x: min(self.tasks[task_id]["earliest_start"] 
                                        for task_id in x[1]))
            
            for chain_id, task_list in chains:
                chain_start_time = current_time
                
                # 确保任务链开始时间不早于第一个任务的最早开始时间
                first_task_earliest = min(self.tasks[task_id]["earliest_start"] 
                                        for task_id in task_list)
                chain_start_time = max(chain_start_time, first_task_earliest)
                
                # 为任务链中的每个任务安排时间
                task_time = chain_start_time
                for i, task_id in enumerate(task_list):
                    # 确保不早于任务的最早开始时间
                    task_time = max(task_time, self.tasks[task_id]["earliest_start"])
                    
                    schedule["task_start_times"][task_id] = task_time
                    
                    # 计算任务持续时间
                    route_id = schedule["task_routes"][task_id]
                    duration = self.tasks[task_id]["duration"][route_id]
                    schedule["task_end_times"][task_id] = task_time + duration
                    
                    # 添加到eVTOL调度中
                    schedule["evtol_schedules"][evtol_id].append({
                        "task_id": task_id,
                        "start_time": task_time,
                        "end_time": task_time + duration,
                        "route": route_id
                    })
                    
                    # 为下一个任务留出间隔时间
                    if i < len(task_list) - 1:
                        task_time = task_time + duration + 20  # 20分钟间隔
                
                # 更新eVTOL的当前时间（任务链结束时间 + 任务链间隔）
                last_task_id = task_list[-1]
                current_time = schedule["task_end_times"][last_task_id] + 30
    
    def evaluate_objectives(self, individual: Individual) -> Tuple[float, float]:
        """
        计算个体的双目标函数值
        
        返回:
            (总能耗, 总延误时间)
        """
        schedule = self.decode_individual(individual)
        individual.schedule = schedule
        
        # 目标1：总能耗
        total_energy = 0
        for task_id in range(self.num_tasks):
            route_id = schedule["task_routes"][task_id]
            energy_consumption = self.tasks[task_id]["soc_consumption"][route_id]
            total_energy += energy_consumption
        
        # 目标2：总延误时间
        total_delay = 0
        for task_id in range(self.num_tasks):
            earliest_start = self.tasks[task_id]["earliest_start"]
            actual_start = schedule["task_start_times"][task_id]
            delay = max(0, actual_start - earliest_start)
            total_delay += delay
        
        # 添加约束违反惩罚
        penalty = self._calculate_constraint_penalty(schedule)
        
        # 将惩罚加到目标函数上
        total_energy += penalty * 1000  # 能耗惩罚
        total_delay += penalty * 100    # 延误惩罚
        
        return total_energy, total_delay
    
    def _calculate_constraint_penalty(self, schedule: Dict) -> float:
        """计算约束违反惩罚"""
        penalty = 0
        
        # 检查时间冲突（同一eVTOL的任务时间重叠）
        for evtol_id, tasks in schedule["evtol_schedules"].items():
            tasks.sort(key=lambda x: x["start_time"])
            for i in range(len(tasks) - 1):
                if tasks[i]["end_time"] > tasks[i+1]["start_time"]:
                    overlap = tasks[i]["end_time"] - tasks[i+1]["start_time"]
                    penalty += overlap * 0.1
        
        # 检查航线冲突（相同时间使用相同航线的任务）
        route_usage = defaultdict(list)
        for task_id in range(self.num_tasks):
            route_id = schedule["task_routes"][task_id]
            start_time = schedule["task_start_times"][task_id]
            end_time = schedule["task_end_times"][task_id]
            route_usage[route_id].append((start_time, end_time, task_id))
        
        for route_id, usage_list in route_usage.items():
            usage_list.sort()
            for i in range(len(usage_list) - 1):
                end1, start2 = usage_list[i][1], usage_list[i+1][0]
                if end1 > start2:
                    overlap = end1 - start2
                    penalty += overlap * 0.1
        
        return penalty
    
    def fast_non_dominated_sort(self, population: List[Individual]) -> List[List[int]]:
        """快速非支配排序"""
        n = len(population)
        
        # 为每个个体计算支配关系
        dominated_solutions = [[] for _ in range(n)]  # 被i支配的解
        domination_count = [0] * n  # 支配i的解的数量
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    if self._dominates(population[i], population[j]):
                        dominated_solutions[i].append(j)
                    elif self._dominates(population[j], population[i]):
                        domination_count[i] += 1
        
        # 分层
        fronts = []
        current_front = []
        
        # 找到第一层（非支配解）
        for i in range(n):
            if domination_count[i] == 0:
                population[i].rank = 0
                current_front.append(i)
        
        fronts.append(current_front)
        
        # 找到后续层
        while len(current_front) > 0:
            next_front = []
            for i in current_front:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        population[j].rank = len(fronts)
                        next_front.append(j)
            current_front = next_front
            if len(current_front) > 0:
                fronts.append(current_front)
        
        return fronts
    
    def _dominates(self, ind1: Individual, ind2: Individual) -> bool:
        """判断ind1是否支配ind2"""
        obj1 = ind1.objectives
        obj2 = ind2.objectives
        
        if obj1 is None or obj2 is None:
            return False
        
        # 至少在一个目标上更好，且在所有目标上不更差
        better = False
        for i in range(len(obj1)):
            if obj1[i] > obj2[i]:  # 最小化问题，值越小越好
                return False
            elif obj1[i] < obj2[i]:
                better = True
        
        return better
    
    def _filter_unique_solutions(self, solutions: List[Individual]) -> List[Individual]:
        """过滤出目标函数值不同的解，去除重复"""
        if not solutions:
            return []
        
        unique_solutions = []
        seen_objectives = set()
        
        for sol in solutions:
            if sol.objectives:
                # 将目标函数值转换为元组作为key（保留1位小数以处理数值误差）
                obj_key = tuple(round(obj, 1) for obj in sol.objectives)
                if obj_key not in seen_objectives:
                    seen_objectives.add(obj_key)
                    unique_solutions.append(sol)
        
        return unique_solutions
    
    def calculate_crowding_distance(self, front: List[int], population: List[Individual]):
        """计算拥挤距离"""
        if len(front) <= 2:
            for i in front:
                population[i].crowding_distance = float('inf')
            return
        
        # 初始化拥挤距离
        for i in front:
            population[i].crowding_distance = 0
        
        # 对每个目标进行排序和距离计算
        num_objectives = len(population[front[0]].objectives)
        
        for obj_idx in range(num_objectives):
            # 按第obj_idx个目标排序
            front.sort(key=lambda x: population[x].objectives[obj_idx])
            
            # 边界解设为无穷大
            population[front[0]].crowding_distance = float('inf')
            population[front[-1]].crowding_distance = float('inf')
            
            # 计算中间解的拥挤距离
            obj_min = population[front[0]].objectives[obj_idx]
            obj_max = population[front[-1]].objectives[obj_idx]
            
            if obj_max - obj_min > 0:
                for i in range(1, len(front) - 1):
                    distance = (population[front[i+1]].objectives[obj_idx] - 
                              population[front[i-1]].objectives[obj_idx]) / (obj_max - obj_min)
                    population[front[i]].crowding_distance += distance
    
    def tournament_selection(self, population: List[Individual], k: int = 3) -> Individual:
        """锦标赛选择"""
        tournament = random.sample(population, k)
        return min(tournament)  # 根据rank和crowding_distance排序
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """交叉算子"""
        if random.random() > self.crossover_prob:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
        
        # 两点交叉
        chromosome1 = parent1.chromosome.copy()
        chromosome2 = parent2.chromosome.copy()
        
        # 为任务链分配和航线选择分别进行交叉
        
        # 任务链分配部分交叉
        if self.chain_assignment_length > 1:
            point1 = random.randint(0, self.chain_assignment_length - 1)
            chromosome1[:point1], chromosome2[:point1] = chromosome2[:point1], chromosome1[:point1]
        
        # 航线选择部分交叉
        if self.route_selection_length > 1:
            start_idx = self.chain_assignment_length
            point2 = random.randint(start_idx, self.chromosome_length - 1)
            chromosome1[point2:], chromosome2[point2:] = chromosome2[point2:], chromosome1[point2:]
        
        offspring1 = Individual(chromosome1)
        offspring2 = Individual(chromosome2)
        
        return offspring1, offspring2
    
    def mutation(self, individual: Individual) -> Individual:
        """变异算子"""
        if random.random() > self.mutation_prob:
            return individual
        
        chromosome = individual.chromosome.copy()
        
        # 随机选择变异位置
        pos = random.randint(0, len(chromosome) - 1)
        
        if pos < self.chain_assignment_length:
            # 任务链分配变异
            chromosome[pos] = random.randint(0, self.num_evtols - 1)
        else:
            # 航线选择变异
            chromosome[pos] = random.randint(0, self.num_routes - 1)
        
        return Individual(chromosome)
    
    def environmental_selection(self, population: List[Individual]) -> List[Individual]:
        """环境选择"""
        # 非支配排序
        fronts = self.fast_non_dominated_sort(population)
        
        # 选择下一代
        next_population = []
        front_idx = 0
        
        while len(next_population) + len(fronts[front_idx]) <= self.pop_size:
            # 计算当前层的拥挤距离
            self.calculate_crowding_distance(fronts[front_idx], population)
            
            # 添加整个层
            for i in fronts[front_idx]:
                next_population.append(population[i])
            
            front_idx += 1
            if front_idx >= len(fronts):
                break
        
        # 如果还需要更多个体，从下一层中选择
        if len(next_population) < self.pop_size and front_idx < len(fronts):
            remaining = self.pop_size - len(next_population)
            self.calculate_crowding_distance(fronts[front_idx], population)
            
            # 按拥挤距离排序，选择拥挤距离大的
            last_front = [population[i] for i in fronts[front_idx]]
            last_front.sort(key=lambda x: x.crowding_distance, reverse=True)
            
            next_population.extend(last_front[:remaining])
        
        return next_population
    
    def evolve(self) -> Dict:
        """主进化循环"""
        print("开始NSGA-II进化...")
        
        # 初始化种群
        population = self.initialize_population()
        
        # 评估初始种群
        for ind in population:
            ind.objectives = self.evaluate_objectives(ind)
        
        # 记录进化过程
        evolution_history = {
            "generations": [],
            "pareto_front_sizes": [],
            "best_energy": [],
            "best_delay": [],
            "hypervolume": []
        }
        
        # 进化循环
        for generation in range(self.generations):
            start_time = time.time()
            
            # 生成子代
            offspring = []
            while len(offspring) < self.pop_size:
                # 选择父代
                parent1 = self.tournament_selection(population)
                parent2 = self.tournament_selection(population)
                
                # 交叉
                child1, child2 = self.crossover(parent1, parent2)
                
                # 变异
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                
                offspring.extend([child1, child2])
            
            # 限制子代数量
            offspring = offspring[:self.pop_size]
            
            # 评估子代
            for ind in offspring:
                ind.objectives = self.evaluate_objectives(ind)
            
            # 合并父子代
            combined_population = population + offspring
            
            # 环境选择
            population = self.environmental_selection(combined_population)
            
            # 记录统计信息
            if generation % 10 == 0:
                fronts = self.fast_non_dominated_sort(population)
                raw_pareto = [population[i] for i in fronts[0]]
                pareto_front = self._filter_unique_solutions(raw_pareto)
                
                best_energy = min(ind.objectives[0] for ind in pareto_front)
                best_delay = min(ind.objectives[1] for ind in pareto_front)
                
                evolution_history["generations"].append(generation)
                evolution_history["pareto_front_sizes"].append(len(pareto_front))
                evolution_history["best_energy"].append(best_energy)
                evolution_history["best_delay"].append(best_delay)
                
                elapsed_time = time.time() - start_time
                print(f"代数 {generation}: 第一层={len(raw_pareto)}个, 独特解={len(pareto_front)}个, "
                      f"最佳能耗={best_energy:.1f}, 最佳延误={best_delay:.1f}, "
                      f"用时={elapsed_time:.2f}s")
        
        # 最终非支配排序，提取帕累托前沿
        final_fronts = self.fast_non_dominated_sort(population)
        raw_pareto_front = [population[i] for i in final_fronts[0]]
        
        # 过滤出目标函数值不同的解（去除重复）
        pareto_front = self._filter_unique_solutions(raw_pareto_front)
        
        print(f"\n进化完成！原始第一层包含 {len(raw_pareto_front)} 个解")
        print(f"过滤后的帕累托前沿包含 {len(pareto_front)} 个独特解")
        
        return {
            "pareto_front": pareto_front,
            "evolution_history": evolution_history,
            "final_population": population
        }

def solve_evtol_nsga2(tasks: List[Dict], evtols: List[Dict], task_chains: List[List[int]],
                     population_size: int = 100, generations: int = 200) -> Dict:
    """
    使用NSGA-II算法求解eVTOL调度多目标优化问题
    
    注意：本函数使用与gurobi模块相同的问题建模：
    - 相同的任务链概念和生成方法
    - 相同的目标函数：最小化能耗 vs 最小化延误
    - 遵循相同的约束逻辑（时间窗、防撞、容量等）
    
    参数:
        tasks: 任务列表
        evtols: eVTOL列表
        task_chains: 任务链列表（应使用gurobi.generate_task_chains()生成）
        population_size: 种群大小
        generations: 进化代数
    
    返回:
        包含帕累托前沿和进化历史的结果字典
    """
    solver = eVTOL_NSGA2(tasks, evtols, task_chains, population_size, generations)
    result = solver.evolve()
    
    return result

def visualize_pareto_front(result: Dict, save_path: str = "picture_result/pareto_front.png"):
    """可视化帕累托前沿"""
    pareto_front = result["pareto_front"]
    
    if not pareto_front:
        print("没有帕累托前沿数据可视化")
        return
    
    # 提取目标函数值
    energies = [ind.objectives[0] for ind in pareto_front]
    delays = [ind.objectives[1] for ind in pareto_front]
    
    # 创建图形
    plt.figure(figsize=(10, 8))
    plt.scatter(energies, delays, c='red', s=50, alpha=0.7, label='帕累托前沿')
    
    # 设置标签和标题
    plt.xlabel('总能耗')
    plt.ylabel('总延误时间 (分钟)')
    plt.title('eVTOL调度问题的帕累托前沿')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 保存图形
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"帕累托前沿图已保存到: {save_path}")

def visualize_evolution_history(result: Dict, save_path: str = "picture_result/evolution_history.png"):
    """可视化进化历史"""
    history = result["evolution_history"]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 帕累托前沿大小变化
    ax1.plot(history["generations"], history["pareto_front_sizes"], 'b-o')
    ax1.set_xlabel('代数')
    ax1.set_ylabel('帕累托前沿大小')
    ax1.set_title('帕累托前沿大小变化')
    ax1.grid(True)
    
    # 最佳能耗变化
    ax2.plot(history["generations"], history["best_energy"], 'g-o')
    ax2.set_xlabel('代数')
    ax2.set_ylabel('最佳能耗')
    ax2.set_title('最佳能耗变化')
    ax2.grid(True)
    
    # 最佳延误变化
    ax3.plot(history["generations"], history["best_delay"], 'r-o')
    ax3.set_xlabel('代数')
    ax3.set_ylabel('最佳延误时间')
    ax3.set_title('最佳延误变化')
    ax3.grid(True)
    
    # 目标函数空间中的进化轨迹
    ax4.plot(history["best_energy"], history["best_delay"], 'purple', marker='o', alpha=0.7)
    ax4.set_xlabel('最佳能耗')
    ax4.set_ylabel('最佳延误时间')
    ax4.set_title('目标函数空间中的进化轨迹')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"进化历史图已保存到: {save_path}") 
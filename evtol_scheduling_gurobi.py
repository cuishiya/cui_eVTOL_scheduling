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

def generate_task_chains(tasks: List[Dict], max_chain_length) -> List[List[int]]:
    """
    生成基于位置连续性的任务串
    前一个任务的终点必须等于后一个任务的起点，确保eVTOL可以连续执行任务

    参数:
        tasks: 任务列表
        max_chain_length: 单个任务串的最大长度

    返回:
        任务串列表，每个任务串是任务ID的列表
    """
    print("正在生成基于位置连续性的任务串...")

    # 按起点分组任务，用于快速查找从某位置出发的任务
    tasks_by_start_location = defaultdict(list)
    for i, task in enumerate(tasks):
        tasks_by_start_location[task['from']].append(i)

    task_chains = []
    used_tasks = set()

    def build_chain_from_task(start_task_id):
        """从指定任务开始构建位置连续的任务串"""
        if start_task_id in used_tasks:
            return []

        chain = [start_task_id]
        used_tasks.add(start_task_id)
        current_location = tasks[start_task_id]['to']  # 当前任务的终点作为下一个任务的起点

        # 继续寻找可以连接的任务（起点 = 当前位置）
        while len(chain) < max_chain_length:
            # 找到所有从当前位置出发且未使用的任务
            candidate_tasks = [
                t for t in tasks_by_start_location[current_location]
                if t not in used_tasks
            ]

            if not candidate_tasks:
                break

            # 选择策略：考虑时间窗兼容性
            valid_candidates = []

            # 估算当前任务串的最早结束时间
            chain_earliest_end = tasks[chain[0]]['earliest_start']
            for task_id in chain:
                chain_earliest_end += min(tasks[task_id]['duration']) + 20  # 加上任务间隔时间

            # 筛选时间窗兼容的候选任务
            for candidate in candidate_tasks:
                # 候选任务的最早开始时间应该允许在当前任务串结束后执行
                if tasks[candidate]['earliest_start'] >= chain_earliest_end - 30:  # 允许30分钟的灵活性
                    valid_candidates.append(candidate)

            # 如果没有时间兼容的任务，选择所有候选任务中最早的
            if not valid_candidates:
                valid_candidates = candidate_tasks

            # 选择最早开始时间的任务
            selected_task = min(valid_candidates, key=lambda t: tasks[t]['earliest_start'])
            chain.append(selected_task)
            used_tasks.add(selected_task)
            current_location = tasks[selected_task]['to']  # 更新当前位置

        return chain

    # 按任务的最早开始时间排序，优先处理早期任务
    sorted_task_ids = sorted(range(len(tasks)), key=lambda i: tasks[i]['earliest_start'])

    # 为每个未使用的任务尝试构建任务串
    for task_id in sorted_task_ids:
        if task_id not in used_tasks:
            chain = build_chain_from_task(task_id)
            if chain:
                task_chains.append(chain)

    print(f"生成了 {len(task_chains)} 个任务串，覆盖 {len(used_tasks)} 个任务")

    # 打印任务串信息并验证位置连续性
    for i, chain in enumerate(task_chains):
        chain_info = []
        is_continuous = True

        for j, task_id in enumerate(chain):
            task = tasks[task_id]
            chain_info.append(f"T{task_id}({task['from']}→{task['to']})")

            # 验证位置连续性（除了第一个任务）
            if j > 0:
                prev_task = tasks[chain[j-1]]
                if prev_task['to'] != task['from']:
                    print(f"❌ 警告：任务串 {i} 中任务 {chain[j-1]} 的终点({prev_task['to']}) != 任务 {task_id} 的起点({task['from']})！")
                    is_continuous = False

        # 显示任务串信息
        status = "✅" if is_continuous else "❌"
        print(f"{status} 任务串 {i}: {' → '.join(chain_info)}")

        # 显示完整的位置路径
        if len(chain) > 1:
            path = [str(tasks[chain[0]]['from'])]
            for task_id in chain:
                path.append(str(tasks[task_id]['to']))
            print(f"   位置路径: {' → '.join(path)}")

        # 显示时间信息
        time_info = []
        for task_id in chain:
            time_info.append(f"T{task_id}(最早:{tasks[task_id]['earliest_start']})")
        print(f"   时间信息: {' → '.join(time_info)}")

    return task_chains

def solve_evtol_scheduling_with_chains(
    tasks: List[Dict],
    evtols: List[Dict],
    task_chains: List[List[int]],
    time_horizon: int = 720,
    verbose: bool = False
) -> Dict:
    """
    基于任务串的eVTOL调度优化模型

    参数:
        tasks: 任务列表
        evtols: eVTOL列表
        task_chains: 任务串列表
        time_horizon: 调度时间范围
        verbose: 是否打印详细信息

    返回:
        包含调度结果的字典
    """
    print("正在构建基于任务串的优化模型...")

    # 创建模型
    model = gp.Model("eVTOL_Scheduling_with_Chains")

    # 提取基本信息
    num_tasks = len(tasks)
    num_evtols = len(evtols)
    num_chains = len(task_chains)
    num_routes = 3  # 每对起降点之间有3条航线

    # ===== 1. 定义决策变量 =====

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

    model.update()

    # ===== 2. 定义约束条件 =====

    # 2.1 任务串分配唯一性约束
    # 每个任务串必须被分配给一个eVTOL在某个时刻执行
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
    # 对于每对任务，如果它们使用相同航线且时间重叠，则不能同时执行

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
                # 时间不重叠的条件：task_end[i] <= task_start[j] OR task_end[j] <= task_start[i]
                # 引入二进制变量表示任务i在任务j之前完成
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

    # 辅助变量：任务串的结束时间
    chain_end = {}
    for c in range(num_chains):
        chain_end[c] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=time_horizon, name=f"chain_end_{c}")

    # 计算每个任务串的结束时间
    for c, chain in enumerate(task_chains):
        last_task_id = chain[-1]
        model.addConstr(chain_end[c] == task_end[last_task_id], f"chain_end_time_{c}")

    # 对于每架eVTOL，其执行的任意两个任务串之间必须有时间间隔
    for k in range(num_evtols):
        for c1 in range(num_chains):
            for c2 in range(c1 + 1, num_chains):
                # b_c1_k = 1 表示任务串c1分配给eVTOL k
                b_c1_k = model.addVar(vtype=GRB.BINARY, name=f"b_{c1}_{k}")
                model.addConstr(b_c1_k == gp.quicksum(y[c1, k, t] for t in range(time_horizon)))

                # b_c2_k = 1 表示任务串c2分配给eVTOL k
                b_c2_k = model.addVar(vtype=GRB.BINARY, name=f"b_{c2}_{k}")
                model.addConstr(b_c2_k == gp.quicksum(y[c2, k, t] for t in range(time_horizon)))

                # both_assigned = 1 表示c1和c2都分配给eVTOL k
                both_assigned = model.addVar(vtype=GRB.BINARY, name=f"both_assigned_{c1}_{c2}_{k}")
                model.addConstr(both_assigned <= b_c1_k)
                model.addConstr(both_assigned <= b_c2_k)
                model.addConstr(both_assigned >= b_c1_k + b_c2_k - 1)

                # c1_before_c2 = 1 表示任务串c1在c2之前执行
                c1_before_c2 = model.addVar(vtype=GRB.BINARY, name=f"order_{c1}_{c2}_{k}")

                # Big-M 约束
                M = time_horizon
                # 如果c1和c2都分配给k，则必须满足时间间隔约束
                # 约束1: chain_end[c1] + interval <= chain_start[c2] OR c2在c1前
                model.addConstr(
                    chain_end[c1] + chain_interval_time <= chain_start[c2] + M * (1 - c1_before_c2) + M * (1 - both_assigned),
                    f"chain_interval_1_{c1}_{c2}_{k}"
                )
                # 约束2: chain_end[c2] + interval <= chain_start[c1] OR c1在c2前
                model.addConstr(
                    chain_end[c2] + chain_interval_time <= chain_start[c1] + M * c1_before_c2 + M * (1 - both_assigned),
                    f"chain_interval_2_{c1}_{c2}_{k}"
                )

    # 2.7 任务时间窗约束
    for i in range(num_tasks):
        model.addConstr(
            task_start[i] >= tasks[i]['earliest_start'],
            f"earliest_start_{i}"
        )
    #     if 'latest_start' in tasks[i]:
    #         model.addConstr(
    #             task_start[i] <= tasks[i]['latest_start'],
    #             f"latest_start_{i}"
    #         )

    # ===== 3. 定义目标函数 =====
    
    # 计算总能量消耗
    total_energy_consumption = gp.quicksum(
        tasks[i]['soc_consumption'][h] * z[i, h]
        for i in range(num_tasks)
        for h in range(num_routes)
    )

    # 计算总延误时间
    total_delay = gp.quicksum(task_start[i] - tasks[i]['earliest_start'] for i in range(num_tasks))
    
    # ===== 目标函数基准化处理 =====
    # 计算理论基准值用于基准化
    
    # 能耗基准：所有任务选择最低能耗航线的总和
    min_energy_baseline = sum(min(tasks[i]['soc_consumption']) for i in range(num_tasks))
    
    # 延误基准：估算合理的延误基准值
    # 方法1：使用任务数量 × 平均任务间隔作为基准
    delay_baseline = num_tasks * 40  # 假设平均每个任务延误40分钟是可接受的
    
    # 基准化目标函数
    benchmarked_energy = total_energy_consumption / min_energy_baseline
    benchmarked_delay = total_delay / delay_baseline
    
    # 权重参数 (现在两个目标在相同量级)
    alpha = 0.3  # 能耗权重
    beta = 0.7   # 延误权重 (alpha + beta = 1.0)
    
    model.setObjective(alpha * benchmarked_energy + beta * benchmarked_delay, GRB.MINIMIZE)

    # ===== 4. 求解模型 =====
    model.setParam('OutputFlag', 1 if verbose else 0)
    model.setParam('MIPGap', 0.2)
    model.setParam('TimeLimit', 1800)
    model.setParam('MIPFocus', 1)
    model.optimize()

    # ===== 5. 提取结果 =====
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
        
        # 计算基准化后的目标值
        benchmarked_energy_value = total_energy / min_energy_baseline
        benchmarked_delay_value = total_delay_value / delay_baseline

        # 提取结果
        result = {
            "status": "optimal" if model.status == GRB.OPTIMAL else "time_limit",
            "objective_value": model.objVal,
            "total_energy_consumption": total_energy,
            "total_delay": total_delay_value,
            "benchmarked_energy": benchmarked_energy_value,
            "benchmarked_delay": benchmarked_delay_value,
            "energy_baseline": min_energy_baseline,
            "delay_baseline": delay_baseline,
            "schedule": [],
            "task_chains": task_chains,
            "chain_assignments": []
        }

        # 提取任务串分配
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

        # 提取任务调度
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
        return {"status": "infeasible"}


def solve_evtol_scheduling_with_task_chains(
    tasks: List[Dict],
    evtols: List[Dict],
    time_horizon: int = 720,
    max_chain_length: int = 10,
    verbose: bool = False
) -> Dict:
    """
    使用任务串方法求解eVTOL调度问题的主函数

    参数:
        tasks: 任务列表
        evtols: eVTOL列表
        time_horizon: 调度时间范围
        max_chain_length: 最大任务串长度
        verbose: 是否打印详细信息

    返回:
        包含调度结果的字典
    """
    print("=== 使用任务串方法求解eVTOL调度问题 ===")

    # 第一步：生成任务串
    task_chains = generate_task_chains(tasks, max_chain_length)

    # 第二步：基于任务串进行优化
    result = solve_evtol_scheduling_with_chains(
        tasks=tasks,
        evtols=evtols,
        task_chains=task_chains,
        time_horizon=time_horizon,
        verbose=verbose
    )

    # 打印任务串分配结果
    if result["status"] in ["optimal", "time_limit"] and verbose:
        print("\n=== 任务串分配结果 ===")
        for assignment in result["chain_assignments"]:
            chain_id = assignment["chain_id"]
            evtol_id = assignment["evtol_id"]
            start_time = assignment["start_time"]
            tasks_in_chain = assignment["tasks"]

            print(f"eVTOL {evtol_id} 在时刻 {start_time} 执行任务串 {chain_id}:")
            for task_id in tasks_in_chain:
                task = tasks[task_id]
                print(f"  - 任务 {task_id}: {task['from']} → {task['to']}")

    return result

def visualize_schedule_table_gurobi(result: Dict) -> None:
    """
    生成任务调度表的可视化图表 (Gurobi版本)

    参数:
        result: solve_evtol_scheduling_gurobi函数返回的结果字典
    """
    if result["status"] not in ["optimal", "time_limit"]:
        print(f"无法可视化：模型求解状态为 {result['status']}，没有可行解")
        return

    # 提取调度信息
    schedule = result["schedule"]

    # 对调度按开始时间排序
    sorted_schedule = sorted(schedule, key=lambda x: x["start_time"])

    # 创建图形
    fig, ax = plt.subplots(figsize=(12, max(6, len(sorted_schedule) * 0.4)))

    # 隐藏坐标轴
    ax.axis('tight')
    ax.axis('off')

    # 准备表格数据
    table_data = []
    headers = ['任务ID', 'eVTOL ID', '起点', '终点', '开始时间', '结束时间', '航线', '持续时间', '延误']

    for task in sorted_schedule:
        duration = task["end_time"] - task["start_time"]
        table_data.append([
            task["task_id"],
            task["evtol_id"],
            task["from"],
            task["to"],
            task["start_time"],
            task["end_time"],
            task["route"],
            duration,
            task["delay"]
        ])

    # 创建表格
    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        loc='center',
        cellLoc='center',
        colColours=['#f2f2f2'] * len(headers)
    )

    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.2)  # 调整表格大小

    # 设置标题
    plt.title('eVTOL 任务调度表', pad=20)

    # 保存图形
    plt.savefig('evtol_schedule_table_gurobi.png', dpi=300, bbox_inches='tight')

    # 显示图形
    plt.show()

def visualize_schedule_gurobi(result: Dict, time_horizon: int = 1440) -> None:
    """
    生成调度结果的可视化甘特图 (Gurobi版本)

    参数:
        result: solve_evtol_scheduling_gurobi函数返回的结果字典
        time_horizon: 调度时间范围（分钟）
    """
    if result["status"] not in ["optimal", "time_limit"]:
        print(f"无法可视化：模型求解状态为 {result['status']}，没有可行解")
        return

    # 提取调度信息
    schedule = result["schedule"]

    # 确定eVTOL数量 - 修复：显示所有eVTOL，包括没有分配任务的
    evtol_ids = set([task["evtol_id"] for task in schedule])
    if evtol_ids:
        num_evtols = max(evtol_ids) + 1  # 使用最大ID+1确保包含所有eVTOL
    else:
        num_evtols = 1  # 至少显示1个eVTOL

    # 创建图形 - 调整高度使任务条更加紧凑
    fig, ax = plt.subplots(figsize=(20, max(6, num_evtols * 1.4)))

    # 定义颜色
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # 绘制任务
    for task in schedule:
        evtol_id = task["evtol_id"]
        start = task["start_time"]
        duration = task["end_time"] - task["start_time"]

        # 绘制任务块 - 减小高度使其更紧凑美观
        task_bar = ax.barh(evtol_id, duration, left=start, height=0.35,
                          color=colors[task["route"] % len(colors)],
                          edgecolor='white', linewidth=1.5, alpha=0.8)

        # 添加任务标签 - 在任务条内显示任务ID和起点→终点
        task_label = f"T{task['task_id']}"
        route_label = f"{task['from']}→{task['to']}"

        # 任务ID显示在任务条上方
        ax.text(start + duration/2, evtol_id + 0.25, task_label,
                ha='center', va='center', color='black', fontsize=9, weight='bold')

        # 起点→终点显示在任务条下方
        ax.text(start + duration/2, evtol_id - 0.25, route_label,
                ha='center', va='center', color='darkblue', fontsize=8)

    # 设置图形属性
    ax.set_xlabel('时间 (分钟)')
    ax.set_ylabel('eVTOL ID')
    ax.set_title('eVTOL 调度甘特图 (Gurobi)')
    ax.set_yticks(range(num_evtols))
    ax.set_yticklabels([f'eVTOL {i}' for i in range(num_evtols)])
    
    # 设置更细致的时间刻度
    # 计算实际使用的时间范围
    if schedule:
        min_time = min(task["start_time"] for task in schedule)
        max_time = max(task["end_time"] for task in schedule)
        time_range = max_time - min_time
        
        # 根据时间范围动态调整刻度间隔
        if time_range <= 120:  # 2小时内，每10分钟一个刻度
            tick_interval = 6
        elif time_range <= 300:  # 5小时内，每20分钟一个刻度
            tick_interval = 12
        elif time_range <= 600:  # 10小时内，每30分钟一个刻度
            tick_interval = 18
        else:  # 10小时以上，每60分钟一个刻度
            tick_interval = 18
        
        # 生成时间刻度
        start_tick = (min_time // tick_interval) * tick_interval
        end_tick = ((max_time // tick_interval) + 1) * tick_interval
        time_ticks = list(range(int(start_tick), int(end_tick) + 1, tick_interval))
        
        ax.set_xticks(time_ticks)
        ax.set_xlim(start_tick - tick_interval, end_tick + tick_interval)
    else:
        # 如果没有任务，使用默认刻度
        ax.set_xticks(range(0, time_horizon + 1, 30))
    
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)

    # 添加高度层颜色说明（航线颜色图例）
    num_routes = 3  # 假设有3条航线
    legend_elements = []
    for h in range(num_routes):
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, color=colors[h % len(colors)],
                              label=f'航线 {h} (高度层 {h})'))

    # 将图例放在右上角且在调度图外面
    # 调整图形边距，为右侧图例留出空间
    plt.subplots_adjust(right=0.85)
    # 使用bbox_to_anchor将图例放在坐标轴外部
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.01, 1),
             title='高度层/航线说明')

    # 保存图形
    plt.savefig('evtol_schedule_gurobi.png', dpi=300, bbox_inches='tight')

    # 显示图形
    plt.show()
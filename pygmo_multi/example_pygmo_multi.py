#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eVTOL调度PyGMO NSGA-II示例
"""

import sys
import os
import time

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入模块
from data_definitions import get_tasks, get_evtols, get_locations
from gurobi.evtol_scheduling_gurobi import generate_task_chains
from pygmo_multi.evtol_scheduling_pygmo_multi import (
    solve_pygmo_nsga2, 
    visualize_evolution_curves, 
    visualize_pareto_front_evolution
)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'FangSong'


def main():
    """
    eVTOL调度问题PyGMO多目标优化示例
    
    展示完全对应gurobi_multi数学模型的PyGMO实现，包括相同的变量和约束
    使用NSGA-II算法进行真正的多目标优化，无权重组合
    """
    print("="*60)
    print("   eVTOL调度问题 - PyGMO NSGA-II 多目标优化")
    print("="*60)
    
    # 加载数据
    print("\n📊 数据加载:")
    tasks = get_tasks()
    evtols = get_evtols()
    locations = get_locations()
    
    print(f"   任务数量: {len(tasks)}")
    print(f"   eVTOL数量: {len(evtols)}")
    print(f"   位置数量: {len(locations)}")
    
    # 生成任务串
    print("\n🔗 任务串生成:")
    task_chains = generate_task_chains(tasks, max_chain_length=int(len(tasks)/len(evtols))+1)
    print(f"   生成任务串: {len(task_chains)}个")
    
    # 展示任务串信息
    for i, chain in enumerate(task_chains[:5]):  # 显示前5个
        locations_info = []
        for task_id in chain:
            from_loc = tasks[task_id]['from']
            to_loc = tasks[task_id]['to']
            locations_info.append(f"{from_loc}→{to_loc}")
        print(f"   任务串{i}: 任务{chain} ({' | '.join(locations_info)})")
    
    # 遗传编码方案详细说明
    print("\n🧬 遗传编码方案说明:")
    print("   这是一个实数编码方案，完全对应gurobi数学模型的决策变量")
    print("")
    print("   决策变量对应关系:")
    print("   ┌─ gurobi_multi变量 ──────────────────────┬─ PyGMO编码 ─────────────────┐")
    print("   │ y[c,k,t] - eVTOL k在时刻t开始执行任务串c │ y_evtol[c] + y_time[c]      │")
    print("   │ z[i,h] - 任务i使用航线h                  │ z_route[i]                  │")
    print("   │ task_start[i] - 任务i开始时间           │ 从任务串时间推导+微调        │")
    print("   │ task_end[i] - 任务i结束时间             │ 从开始时间+持续时间计算      │")
    print("   │ chain_start[c] - 任务串c开始时间        │ 从y变量推导+微调             │")
    print("   │ chain_end[c] - 任务串c结束时间          │ 从最后任务结束时间计算       │")
    print("   │ 其他辅助变量                           │ 从主要变量推导计算           │")
    print("   └─────────────────────────────────────────┴─────────────────────────────┘")
    print("")
    print("   编码结构:")
    num_chains = len(task_chains)
    num_tasks = len(tasks)
    
    y_vars = num_chains * 2
    z_vars = num_tasks
    task_offset_vars = num_tasks
    chain_offset_vars = num_chains
    total_dims = y_vars + z_vars + task_offset_vars + chain_offset_vars
    
    print(f"   • y变量 (任务串分配): {y_vars}维 = {num_chains}串 × 2(eVTOL+时间)")
    print(f"   • z变量 (航线选择): {z_vars}维 = {num_tasks}任务 × 1(航线)")
    print(f"   • 任务时间微调: {task_offset_vars}维 = {num_tasks}任务 × 1(偏移)")
    print(f"   • 任务串时间微调: {chain_offset_vars}维 = {num_chains}串 × 1(偏移)")
    print(f"   • 总维度: {total_dims}维")
    print("")
    print("   编码示例 (前8维):")
    print("   [0.3, 0.7, 0.1, 0.4, 0.8, 0.2, 0.6, 0.9]")
    print("    ├─┬─┘ ├─┬─┘ ├───┘ ├───┘ ├───┘ ├───┘")
    print("    │ │   │ │   │     │     │     └─ 任务2航线: int(0.9*3)=2")
    print("    │ │   │ │   │     │     └─ 任务1航线: int(0.2*3)=0")
    print("    │ │   │ │   │     └─ 任务0航线: int(0.8*3)=2") 
    print("    │ │   │ └─ 串1时间: int(0.4*720)=288分钟")
    print("    │ │   └─ 串1eVTOL: int(0.1*5)=0号eVTOL")
    print("    │ └─ 串0时间: int(0.7*720)=504分钟")
    print("    └─ 串0eVTOL: int(0.3*5)=1号eVTOL")
    
    # 约束条件对应说明
    print("\n📋 约束条件对应关系:")
    print("   gurobi_multi约束 → PyGMO惩罚函数:")
    print("   • 2.1 任务串分配唯一性 → 检查每个串分配给唯一eVTOL和时刻")
    print("   • 2.2 航线选择唯一性 → 检查每个任务选择唯一航线")
    print("   • 2.3 任务串开始时间约束 → 检查串开始时间与y变量一致性")
    print("   • 2.4 任务串内时间约束 → 检查串内任务时间顺序和间隔")
    print("   • 2.5 eVTOL冲突约束 → 检查同一eVTOL不同时执行多串")
    print("   • 2.6 高度层防撞约束 → 检查同航线任务时间不重叠")
    print("   • 2.7 任务串间隔约束 → 检查同eVTOL不同串间30分钟间隔")
    print("   • 2.8 时间窗约束 → 检查任务不早于最早开始时间")
    
    # 目标函数说明
    print("\n🎯 多目标函数对应关系:")
    print("   • 目标1 (总能耗): Σ(soc_consumption[i][h] * z[i,h])")
    print("   • 目标2 (总延误): Σ(max(0, task_start[i] - earliest_start[i]))")
    print("   • 🔥 重要: 这是真正的多目标优化，无权重组合！")
    print("   • 对应gurobi_multi的epsilon约束方法的两个独立目标")
    
    # 运行优化
    print("\n🚀 开始NSGA-II优化:")
    
    # 算法参数
    population_size = 100  # 较小的种群便于观察
    generations = 100      # 较少的代数便于演示
    
    print(f"   种群大小: {population_size}")
    print(f"   进化代数: {generations}")
    print(f"   交叉概率: 0.9")
    print(f"   变异概率: {1.0/total_dims:.4f}")
    print(f"   注意: 每一代的进化信息都会被打印出来")
    
    # 求解
    result = solve_pygmo_nsga2(
        tasks=tasks,
        evtols=evtols, 
        task_chains=task_chains,
        time_horizon=720,
        population_size=population_size,
        generations=generations,
        verbose=True
    )
    
    if result is None:
        print("❌ 优化失败")
        return
    
    # 分析结果
    pareto_front = result['pareto_front']
    evolution_data = result['evolution_data']
    
    print(f"\n📈 优化结果分析:")
    print(f"   帕累托前沿解数量: {len(pareto_front)}")
    
    if pareto_front:
        energies = [sol['energy'] for sol in pareto_front]
        delays = [sol['delay'] for sol in pareto_front]
        
        print(f"   能耗范围: {min(energies):.1f} - {max(energies):.1f}")
        print(f"   延误范围: {min(delays):.1f} - {max(delays):.1f} 分钟")
        
        # 可视化最终帕累托前沿
        plt.figure(figsize=(10, 6))
        plt.scatter(energies, delays, c='red', s=80, alpha=0.7, edgecolors='black', label='NSGA-II解')
        plt.xlabel('总能耗')
        plt.ylabel('总延误时间 (分钟)')
        plt.title('eVTOL调度问题帕累托前沿 - PyGMO NSGA-II\n(完全对应gurobi_multi数学模型，真正多目标优化)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 添加解的索引标注
        for i, (e, d) in enumerate(zip(energies, delays)):
            if i < 10:  # 只标注前10个解
                plt.annotate(f'{i+1}', (e, d), xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('picture_result/pareto_front_pygmo_nsga2_improved.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"   帕累托前沿图已保存到: picture_result/pareto_front_pygmo_nsga2_improved.png")
        
        # 🔥 新增：绘制进化曲线
        print(f"\n📊 生成进化曲线...")
        visualize_evolution_curves(evolution_data)
        
        # 🔥 新增：绘制帕累托前沿进化过程
        print(f"\n🔄 生成帕累托前沿进化图...")
        # 根据总代数调整显示的代数
        if generations >= 100:
            show_gens = [1, 20, 50, 100, generations//2, -1]
        elif generations >= 50:
            show_gens = [1, 10, 25, -1]
        else:
            show_gens = [1, generations//4, generations//2, -1]
        
        visualize_pareto_front_evolution(evolution_data, show_generations=show_gens)
        
        # 显示最优解
        print(f"\n🏆 代表性解分析:")
        
        # 最小能耗解
        min_energy_idx = energies.index(min(energies))
        min_energy_sol = pareto_front[min_energy_idx]
        print(f"   最小能耗解: 能耗={min_energy_sol['energy']:.1f}, 延误={min_energy_sol['delay']:.1f}分钟")
        
        # 最小延误解
        min_delay_idx = delays.index(min(delays))
        min_delay_sol = pareto_front[min_delay_idx]
        print(f"   最小延误解: 能耗={min_delay_sol['energy']:.1f}, 延误={min_delay_sol['delay']:.1f}分钟")
        
        # 均衡解 (根据fitness值选择)
        if len(pareto_front) > 2:
            fitness_sums = [sol['fitness'][0] + sol['fitness'][1] for sol in pareto_front]
            balanced_idx = fitness_sums.index(min(fitness_sums))
            balanced_sol = pareto_front[balanced_idx]
            print(f"   均衡解: 能耗={balanced_sol['energy']:.1f}, 延误={balanced_sol['delay']:.1f}分钟")
    
    print("\n✅ 优化完成!")
    print("\n📝 建模总结:")
    print("   • PyGMO实现与gurobi_multi数学模型完全对应")
    print("   • 相同的决策变量、约束条件、多目标函数")
    print("   • 真正的多目标优化，无权重组合")
    print("   • 实数编码有效处理复杂的组合优化问题")
    print("   • NSGA-II算法成功找到多个帕累托最优解")
    print("   • 可以与gurobi_multi epsilon约束方法进行性能对比分析")
    print("\n📊 可视化输出:")
    print("   • 最终帕累托前沿图")
    print("   • 进化曲线图 (4个子图显示算法收敛过程)")
    print("   • 帕累托前沿进化过程图 (显示不同代数的前沿变化)")

if __name__ == "__main__":
    main() 
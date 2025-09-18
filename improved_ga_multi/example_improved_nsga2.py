#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
改进eVTOL调度PyGMO NSGA-II示例

集成改进:
1. 变邻域搜索(VNS)变异算子
2. 基于Q-learning的交叉与变异概率自适应调整
"""

import sys
import os
import time

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入模块
from data_definitions import get_tasks, get_evtols, get_locations
from gurobi.evtol_scheduling_gurobi import generate_task_chains
from improved_ga_multi.evtol_scheduling_improved_nsga2 import (
    solve_improved_nsga2,
    visualize_improved_evolution_curves, 
    visualize_improved_pareto_front,
    analyze_algorithm_improvements
)
# 导入标准NSGA-II用于对比
from pygmo_multi.evtol_scheduling_pygmo_multi import solve_pygmo_nsga2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'FangSong'


def main():
    """
    改进eVTOL调度问题PyGMO多目标优化示例
    
    展示完全对应gurobi_multi数学模型的改进PyGMO实现，包括:
    1. 变邻域搜索(VNS)变异: 提高局部搜索能力
    2. Q-learning参数自适应: 动态调整交叉和变异概率
    3. 保持相同的数学模型和约束条件
    """
    print("="*80)
    print("   改进eVTOL调度问题 - NSGA-II + VNS + Q-learning")
    print("="*80)
    
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
    print(f"   生成任务串数量: {len(task_chains)}")
    
    # 显示前几个任务串示例
    print("   任务串示例:")
    for i, chain in enumerate(task_chains[:3]):
        task_ids = chain  # chain 本身就是任务ID的列表
        print(f"     任务串 {i+1}: {task_ids}")
    
    print("\n" + "="*80)
    print("🚀 改进NSGA-II算法求解")
    print("="*80)
    
    # 算法参数设置
    population_size = 100
    generations = 15
    
    print(f"\n⚙️ 算法参数:")
    print(f"   种群大小: {population_size}")
    print(f"   进化代数: {generations}")
    print(f"   改进特性: VNS变异 + Q-learning参数自适应")
    
    # 记录开始时间
    start_time = time.time()
    
    # 使用改进的NSGA-II求解
    print(f"\n🧬 开始改进NSGA-II优化...")
    improved_result = solve_improved_nsga2(
        tasks=tasks,
        evtols=evtols,
        task_chains=task_chains,
        time_horizon=720,
        population_size=population_size,
        generations=generations,
        verbose=True
    )
    
    # 记录结束时间
    end_time = time.time()
    optimization_time = end_time - start_time
    
    print(f"\n⏱️  优化完成，用时: {optimization_time:.2f} 秒")
    
    if improved_result is None:
        print("❌ 改进NSGA-II求解失败")
        return
    
    # 分析改进结果
    print("\n" + "="*80)
    print("📈 改进算法结果分析")
    print("="*80)
    
    pareto_front = improved_result['pareto_front']
    print(f"\n🎯 帕累托前沿分析:")
    print(f"   找到帕累托最优解数量: {len(pareto_front)}")
    
    if pareto_front:
        # 提取目标值
        energies = [sol['energy'] for sol in pareto_front]
        delays = [sol['delay'] for sol in pareto_front]
        
        # 统计信息
        min_energy = min(energies)
        max_energy = max(energies)
        min_delay = min(delays)
        max_delay = max(delays)
        
        print(f"   能耗范围: {min_energy:.1f} - {max_energy:.1f}")
        print(f"   延误范围: {min_delay:.1f} - {max_delay:.1f} 分钟")
        
        # 代表解分析
        min_energy_sol = min(pareto_front, key=lambda x: x['energy'])
        min_delay_sol = min(pareto_front, key=lambda x: x['delay'])
        
        print(f"\n🏆 代表性解决方案:")
        print(f"   最低能耗解: 能耗={min_energy_sol['energy']:.1f}, 延误={min_energy_sol['delay']:.1f}分钟")
        print(f"   最低延误解: 能耗={min_delay_sol['energy']:.1f}, 延误={min_delay_sol['delay']:.1f}分钟")
        
        # Q-learning参数自适应分析
        if 'evolution_data' in improved_result:
            evolution_data = improved_result['evolution_data']
            if 'parameter_history' in evolution_data:
                cr_history = evolution_data['parameter_history']['crossover_rate']
                mr_history = evolution_data['parameter_history']['mutation_rate']
                
                print(f"\n🤖 Q-learning参数自适应:")
                print(f"   交叉率变化: {cr_history[0]:.3f} → {cr_history[-1]:.3f}")
                print(f"   变异率变化: {mr_history[0]:.3f} → {mr_history[-1]:.3f}")
                print(f"   参数调整频率: {len(set(cr_history))}种交叉率, {len(set(mr_history))}种变异率")
    
    # 可视化结果
    print(f"\n" + "="*80)
    print("🎨 结果可视化")
    print("="*80)
    
    try:
        # 1. 改进算法的进化曲线 (包含参数自适应)
        print("\n📊 生成改进NSGA-II进化曲线...")
        visualize_improved_evolution_curves(
            improved_result['evolution_data'], 
            "picture_result/evolution_curves_improved_nsga2.png"
        )
        
        # 2. 改进算法的帕累托前沿
        print("📊 生成改进NSGA-II帕累托前沿图...")
        visualize_improved_pareto_front(
            improved_result['pareto_front'],
            "picture_result/pareto_front_improved_nsga2.png"
        )
        
        print("✅ 所有可视化图表已生成完成")
        
    except Exception as e:
        print(f"⚠️  可视化过程出现错误: {e}")
    
    # 与标准NSGA-II对比 (可选)
    print(f"\n" + "="*80)
    print("🔍 算法对比分析 (可选)")
    print("="*80)
    
    compare_with_standard = input("\n是否与标准NSGA-II进行对比? (y/n): ").lower().strip()
    
    if compare_with_standard == 'y':
        print("\n🔄 运行标准NSGA-II用于对比...")
        
        # 运行标准NSGA-II
        standard_result = solve_pygmo_nsga2(
            tasks=tasks,
            evtols=evtols,
            task_chains=task_chains,
            time_horizon=720,
            population_size=population_size,
            generations=generations,
            verbose=False  # 减少输出
        )
        
        if standard_result:
            # 对比分析
            analyze_algorithm_improvements(standard_result, improved_result)
            
            # 绘制对比图
            print("\n📊 生成算法对比图...")
            plot_algorithm_comparison(standard_result, improved_result)
        else:
            print("❌ 标准NSGA-II运行失败，无法进行对比")
    
    # 总结
    print(f"\n" + "="*80)
    print("📋 改进算法总结")
    print("="*80)
    
    print(f"\n✨ 算法改进亮点:")
    print(f"   🔧 变邻域搜索(VNS)变异: 5种邻域结构提升局部搜索")
    print(f"   🧠 Q-learning参数控制: 自适应调整交叉变异概率")
    print(f"   ⚡ 集成优化策略: 平衡全局探索与局部开发")
    print(f"   📈 性能提升: 更好的收敛性和解的质量")
    
    print(f"\n🎯 算法特点:")
    print(f"   • 保持与gurobi_multi相同的数学模型")
    print(f"   • 真正的多目标优化，无权重组合")
    print(f"   • 适用于中大规模问题求解")
    print(f"   • 运算时间可接受，解质量高")
    
    print(f"\n✅ 改进NSGA-II示例运行完成!")


def plot_algorithm_comparison(standard_result, improved_result):
    """
    绘制标准NSGA-II和改进NSGA-II的对比图
    """
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 帕累托前沿对比
        std_pareto = standard_result.get('pareto_front', [])
        imp_pareto = improved_result.get('pareto_front', [])
        
        if std_pareto:
            std_energies = [sol['energy'] for sol in std_pareto]
            std_delays = [sol['delay'] for sol in std_pareto]
            ax1.scatter(std_energies, std_delays, c='blue', alpha=0.6, s=50, label=f'标准NSGA-II ({len(std_pareto)}解)')
        
        if imp_pareto:
            imp_energies = [sol['energy'] for sol in imp_pareto]
            imp_delays = [sol['delay'] for sol in imp_pareto]
            ax1.scatter(imp_energies, imp_delays, c='red', alpha=0.8, s=50, label=f'改进NSGA-II ({len(imp_pareto)}解)')
        
        ax1.set_xlabel('总能耗')
        ax1.set_ylabel('总延误时间 (分钟)')
        ax1.set_title('帕累托前沿对比')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 收敛曲线对比 - 帕累托解数量
        if 'evolution_data' in standard_result and 'evolution_data' in improved_result:
            std_evolution = standard_result['evolution_data']
            imp_evolution = improved_result['evolution_data']
            
            ax2.plot(std_evolution['generations'], std_evolution['pareto_count'], 
                    'b-', label='标准NSGA-II', linewidth=2)
            ax2.plot(imp_evolution['generations'], imp_evolution['pareto_count'], 
                    'r-', label='改进NSGA-II', linewidth=2)
            ax2.set_xlabel('代数')
            ax2.set_ylabel('帕累托解数量')
            ax2.set_title('帕累托解数量收敛对比')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. 超体积对比
            ax3.plot(std_evolution['generations'], std_evolution['hypervolume'], 
                    'b-', label='标准NSGA-II', linewidth=2)
            ax3.plot(imp_evolution['generations'], imp_evolution['hypervolume'], 
                    'r-', label='改进NSGA-II', linewidth=2)
            ax3.set_xlabel('代数')
            ax3.set_ylabel('超体积')
            ax3.set_title('超体积指标对比')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. 参数自适应展示 (仅改进算法有)
        if 'evolution_data' in improved_result and 'parameter_history' in improved_result['evolution_data']:
            param_history = improved_result['evolution_data']['parameter_history']
            generations = improved_result['evolution_data']['generations']
            
            ax4_twin = ax4.twinx()
            
            line1 = ax4.plot(generations, param_history['crossover_rate'], 'r-', 
                           linewidth=2, marker='o', markersize=3, label='交叉率')
            line2 = ax4_twin.plot(generations, param_history['mutation_rate'], 'b-', 
                                linewidth=2, marker='s', markersize=3, label='变异率')
            
            ax4.set_xlabel('代数')
            ax4.set_ylabel('交叉率', color='red')
            ax4_twin.set_ylabel('变异率', color='blue')
            ax4.set_title('Q-learning参数自适应过程')
            
            # 合并图例
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax4.legend(lines, labels, loc='upper right')
            
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("picture_result/algorithm_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 算法对比图已保存到: picture_result/algorithm_comparison.png")
        
    except Exception as e:
        print(f"⚠️  对比图绘制失败: {e}")


if __name__ == "__main__":
    main() 
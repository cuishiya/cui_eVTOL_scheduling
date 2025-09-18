#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eVTOL调度算法比较脚本
比较NSGA-II多目标优化和Gurobi精确优化的性能
"""

import sys
import os
import time
import matplotlib.pyplot as plt
import numpy as np

# 添加子文件夹到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'ga'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'gurobi'))

# 导入数据定义
from data_definitions import get_tasks, get_evtols

def create_test_data():
    """创建测试数据"""
    tasks = get_tasks()
    evtols = get_evtols()
    
    return tasks, evtols

def run_gurobi_algorithm(tasks, evtols):
    """运行Gurobi算法"""
    print("=== 运行Gurobi算法 ===")
    
    try:
        from gurobi.evtol_scheduling_gurobi import solve_evtol_scheduling_with_task_chains
        
        start_time = time.time()
        result = solve_evtol_scheduling_with_task_chains(
            tasks=tasks,
            evtols=evtols,
            time_horizon=720,
            max_chain_length=3,
            verbose=False
        )
        end_time = time.time()
        
        if result["status"] in ["optimal", "time_limit"]:
            return {
                "success": True,
                "runtime": end_time - start_time,
                "energy": result["total_energy_consumption"],
                "delay": result["total_delay"],
                "objective": result["objective_value"],
                "status": result["status"]
            }
        else:
            return {"success": False, "status": result["status"]}
            
    except Exception as e:
        print(f"Gurobi算法运行失败: {e}")
        return {"success": False, "error": str(e)}

def run_nsga2_algorithm(tasks, evtols, task_chains):
    """运行NSGA-II算法"""
    print("=== 运行NSGA-II算法 ===")
    
    try:
        from ga.evtol_nsga2 import solve_evtol_nsga2
        
        start_time = time.time()
        result = solve_evtol_nsga2(
            tasks=tasks,
            evtols=evtols,
            task_chains=task_chains,
            population_size=50,
            generations=100
        )
        end_time = time.time()
        
        pareto_front = result["pareto_front"]
        if len(pareto_front) > 0:
            # 提取帕累托前沿的统计信息
            energies = [ind.objectives[0] for ind in pareto_front]
            delays = [ind.objectives[1] for ind in pareto_front]
            
            return {
                "success": True,
                "runtime": end_time - start_time,
                "pareto_size": len(pareto_front),
                "min_energy": min(energies),
                "max_energy": max(energies),
                "min_delay": min(delays),
                "max_delay": max(delays),
                "mean_energy": np.mean(energies),
                "mean_delay": np.mean(delays),
                "energies": energies,
                "delays": delays
            }
        else:
            return {"success": False, "error": "未找到可行解"}
            
    except Exception as e:
        print(f"NSGA-II算法运行失败: {e}")
        return {"success": False, "error": str(e)}

def visualize_comparison(gurobi_result, nsga2_result):
    """可视化算法比较结果"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 目标函数空间比较
    if gurobi_result["success"] and nsga2_result["success"]:
        # NSGA-II帕累托前沿
        ax1.scatter(nsga2_result["energies"], nsga2_result["delays"], 
                   c='red', s=30, alpha=0.7, label='NSGA-II帕累托前沿')
        
        # Gurobi单点解
        ax1.scatter([gurobi_result["energy"]], [gurobi_result["delay"]], 
                   c='blue', s=100, marker='*', label='Gurobi解')
        
        ax1.set_xlabel('总能耗')
        ax1.set_ylabel('总延误时间 (分钟)')
        ax1.set_title('目标函数空间比较')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. 运行时间比较
    if gurobi_result["success"] and nsga2_result["success"]:
        algorithms = ['Gurobi', 'NSGA-II']
        runtimes = [gurobi_result["runtime"], nsga2_result["runtime"]]
        colors = ['blue', 'red']
        
        bars = ax2.bar(algorithms, runtimes, color=colors, alpha=0.7)
        ax2.set_ylabel('运行时间 (秒)')
        ax2.set_title('算法运行时间比较')
        
        # 添加数值标签
        for bar, runtime in zip(bars, runtimes):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{runtime:.2f}s', ha='center', va='bottom')
    
    # 3. 解的分布（NSGA-II）
    if nsga2_result["success"]:
        ax3.hist(nsga2_result["energies"], bins=10, alpha=0.7, color='red', label='能耗分布')
        ax3.set_xlabel('总能耗')
        ax3.set_ylabel('解的数量')
        ax3.set_title('NSGA-II解的能耗分布')
        ax3.grid(True, alpha=0.3)
    
    # 4. 算法性能统计
    performance_data = []
    labels = []
    
    if gurobi_result["success"]:
        performance_data.append([
            gurobi_result["energy"],
            gurobi_result["delay"],
            gurobi_result["runtime"]
        ])
        labels.append('Gurobi')
    
    if nsga2_result["success"]:
        performance_data.append([
            nsga2_result["min_energy"],
            nsga2_result["min_delay"],
            nsga2_result["runtime"]
        ])
        labels.append('NSGA-II (最优)')
    
    if performance_data:
        # 创建性能对比表
        metrics = ['最小能耗', '最小延误', '运行时间']
        x = np.arange(len(metrics))
        width = 0.35
        
        if len(performance_data) >= 2:
            ax4.bar(x - width/2, performance_data[0], width, label=labels[0], alpha=0.7)
            ax4.bar(x + width/2, performance_data[1], width, label=labels[1], alpha=0.7)
        else:
            ax4.bar(x, performance_data[0], width, label=labels[0], alpha=0.7)
        
        ax4.set_xlabel('性能指标')
        ax4.set_ylabel('数值')
        ax4.set_title('算法性能对比')
        ax4.set_xticks(x)
        ax4.set_xticklabels(metrics)
        ax4.legend()
    
    plt.tight_layout()
    plt.savefig('picture_result/algorithm_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主函数"""
    print("🚁 eVTOL调度算法性能比较")
    print("=" * 60)
    
    # 创建测试数据
    tasks, evtols = create_test_data()
    
    print("测试配置:")
    print(f"  任务数: {len(tasks)}")
    print(f"  eVTOL数: {len(evtols)}")
    
    # 生成任务链（NSGA-II需要）
    try:
        from gurobi.evtol_scheduling_gurobi import generate_task_chains
        task_chains = generate_task_chains(tasks, max_chain_length=3)
        print(f"  任务链数: {len(task_chains)}")
    except Exception as e:
        print(f"任务链生成失败: {e}")
        return
    
    print("\n" + "="*60)
    
    # 运行Gurobi算法
    gurobi_result = run_gurobi_algorithm(tasks, evtols)
    
    if gurobi_result["success"]:
        print(f"✅ Gurobi运行成功")
        print(f"   运行时间: {gurobi_result['runtime']:.2f}秒")
        print(f"   总能耗: {gurobi_result['energy']:.1f}")
        print(f"   总延误: {gurobi_result['delay']:.1f}分钟")
        print(f"   求解状态: {gurobi_result['status']}")
    else:
        print(f"❌ Gurobi运行失败: {gurobi_result.get('status', gurobi_result.get('error'))}")
    
    print("\n" + "="*60)
    
    # 运行NSGA-II算法
    nsga2_result = run_nsga2_algorithm(tasks, evtols, task_chains)
    
    if nsga2_result["success"]:
        print(f"✅ NSGA-II运行成功")
        print(f"   运行时间: {nsga2_result['runtime']:.2f}秒")
        print(f"   帕累托前沿大小: {nsga2_result['pareto_size']}")
        print(f"   能耗范围: {nsga2_result['min_energy']:.1f} - {nsga2_result['max_energy']:.1f}")
        print(f"   延误范围: {nsga2_result['min_delay']:.1f} - {nsga2_result['max_delay']:.1f}分钟")
    else:
        print(f"❌ NSGA-II运行失败: {nsga2_result.get('error')}")
    
    print("\n" + "="*60)
    print("📊 算法比较总结")
    print("="*60)
    
    if gurobi_result["success"] and nsga2_result["success"]:
        print(f"🏃 运行时间对比:")
        print(f"   Gurobi: {gurobi_result['runtime']:.2f}秒")
        print(f"   NSGA-II: {nsga2_result['runtime']:.2f}秒")
        
        print(f"\n🎯 解质量对比:")
        print(f"   Gurobi解: 能耗={gurobi_result['energy']:.1f}, 延误={gurobi_result['delay']:.1f}")
        print(f"   NSGA-II最优解: 能耗={nsga2_result['min_energy']:.1f}, 延误={nsga2_result['min_delay']:.1f}")
        
        print(f"\n🔍 算法特点:")
        print(f"   Gurobi: 精确求解，单一最优解，需要权重设置")
        print(f"   NSGA-II: 近似求解，{nsga2_result['pareto_size']}个帕累托解，无需权重")
        
        # 生成比较图表
        print(f"\n📈 正在生成比较图表...")
        visualize_comparison(gurobi_result, nsga2_result)
        print(f"   算法比较图已保存: picture_result/algorithm_comparison.png")
    
    else:
        print("⚠️  部分算法运行失败，无法进行完整比较")

if __name__ == "__main__":
    main() 
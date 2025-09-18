#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eVTOL调度Gurobi epsilon约束方法示例
使用Gurobi求解器的epsilon约束方法求解eVTOL调度问题
"""

import sys
import os

# 添加项目根目录到路径，以便导入数据定义
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# 导入Gurobi epsilon约束方法模块
from evtol_scheduling_gurobi_multi import (
    solve_pareto_front_optimization,
    visualize_pareto_front_gurobi_epsilon,
    visualize_convergence_gurobi_epsilon
)

# 导入原始gurobi的任务链生成函数，确保一致性
from gurobi.evtol_scheduling_gurobi import generate_task_chains

# 导入数据定义
from data_definitions import get_tasks, get_evtols, get_locations


def main():
    """
    使用Gurobi epsilon约束方法的示例脚本
    """
    print("🚁 eVTOL 调度epsilon约束方法示例 - Gurobi求解器")
    print("=" * 70)
    
    # 获取地点信息
    locations = get_locations()
    
    # 获取任务和eVTOL数据
    tasks = get_tasks()
    evtols = get_evtols()
    
    # 设置时间范围（分钟）
    time_horizon = 720  # 12小时
    
    print("地点信息:")
    for loc_id, loc_name in locations.items():
        print(f"  {loc_id}: {loc_name}")
    
    print(f"\n任务信息:")
    for task in tasks[:10]:  # 显示前10个任务作为示例
        print(f"  任务{task['id']}: {locations[task['from']]}→{locations[task['to']]}, "
              f"最早开始: {task['earliest_start']}分钟")
    if len(tasks) > 10:
        print(f"  ... 共{len(tasks)}个任务")
    
    print(f"\neVTOL机队:")
    for evtol in evtols:
        print(f"  eVTOL{evtol['id']}: 初始位置={locations[evtol['initial_position']]}, "
              f"初始电量={evtol['initial_soc']}%")
    
    print("\n" + "="*70)
    print("🎯 Gurobi epsilon约束方法优化")
    print("="*70)
    
    
    print(f"\n{'='*50}")
    print(f"🔬 开始epsilon约束方法求解")
    print(f"{'='*50}")
    
    try:
        import time
        start_time = time.time()
        
        # 执行epsilon约束优化
        result = solve_pareto_front_optimization(
            tasks=tasks,
            evtols=evtols,
            time_horizon=time_horizon,
            max_chain_length=int(len(tasks)/len(evtols))+1,
            num_points=30,
            verbose=True
        )
        
        total_time = time.time() - start_time
        
        if result["status"] == "optimal":
            pareto_front = result["pareto_front"]
            
            print(f"\n🎯 epsilon约束方法完成!")
            print(f"   总运行时间: {total_time:.2f}秒")
            print(f"   帕累托前沿大小: {len(pareto_front)}个解")
            
            if pareto_front:
                energies = [sol["total_energy_consumption"] for sol in pareto_front]
                delays = [sol["total_delay"] for sol in pareto_front]
                
                print(f"   能耗范围: {min(energies):.1f} - {max(energies):.1f}")
                print(f"   延误范围: {min(delays):.1f} - {max(delays):.1f}分钟")
                print(f"   平均求解时间: {sum(sol.get('solve_time', 0) for sol in pareto_front)/len(pareto_front):.3f}秒/解")
                
                # 显示帕累托前沿解
                print(f"\n📊 帕累托前沿解:")
                print(f"{'序号':<4} {'能耗':<8} {'延误(分钟)':<10} {'执行任务数':<8}")
                print("-" * 35)
                for idx, sol in enumerate(sorted(pareto_front, key=lambda x: x["total_energy_consumption"]), 1):
                    print(f"{idx:<4} {sol['total_energy_consumption']:<8.1f} {sol['total_delay']:<10.1f} {len(sol['schedule']):<8}")
                
                # 显示总体统计
                total_tasks_executed = sum(len(sol['schedule']) for sol in pareto_front) // len(pareto_front)
                print(f"\n📈 统计信息:")
                print(f"   平均执行任务数: {total_tasks_executed:.0f}个")

            
        else:
            print(f"❌ epsilon约束方法求解失败: {result['status']}")
            
    except Exception as e:
        print(f"❌ epsilon约束方法运行出错: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 生成可视化
    print(f"\n📈 正在生成可视化图表...")
    
    try:
        if result["status"] == "optimal":
            # 帕累托前沿图
            front_path = f"picture_result/pareto_front_gurobi_epsilon_constraint.png"
            visualize_pareto_front_gurobi_epsilon(result, front_path)
            
            # 收敛历史图
            convergence_path = f"picture_result/convergence_gurobi_epsilon_constraint.png"
            visualize_convergence_gurobi_epsilon(result, convergence_path)
        
        print(f"\n✅ Gurobi epsilon约束方法示例运行成功！")
        print(f"\n📊 结果文件已生成:")
        if result["status"] == "optimal":
            print(f"   📈 picture_result/pareto_front_gurobi_epsilon_constraint.png - 帕累托前沿图")
            print(f"   📊 picture_result/convergence_gurobi_epsilon_constraint.png - 收敛分析图")
        
        print(f"\n🎉 优化完成！主目标：最小化延误，约束：能耗限制")
        
    except Exception as e:
        print(f"可视化生成失败: {e}")


if __name__ == "__main__":
    main() 
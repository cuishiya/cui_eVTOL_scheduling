#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eVTOL调度NSGA-II多目标优化示例
使用NSGA-II算法求解eVTOL调度的多目标优化问题
"""

import sys
import os

# 添加项目根目录到路径，以便导入数据定义
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# 导入NSGA-II相关模块
from evtol_nsga2 import solve_evtol_nsga2, visualize_pareto_front, visualize_evolution_history

# 导入原始gurobi的任务链生成函数，确保使用相同的建模理念
from gurobi.evtol_scheduling_gurobi import generate_task_chains

# 导入数据定义
from data_definitions import get_tasks, get_evtols, get_locations


def main():
    """
    使用NSGA-II多目标优化的示例脚本
    """
    print("🚁 eVTOL 调度多目标优化示例 - NSGA-II算法")
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
    
    # 生成任务链
    print(f"\n正在生成任务链...")
    try:
        task_chains = generate_task_chains(tasks, max_chain_length=9)
        print(f"成功生成 {len(task_chains)} 个任务链")
    except Exception as e:
        print(f"任务链生成失败: {e}")
        return
    
    print("\n" + "="*70)
    print("🧬 开始NSGA-II多目标优化求解...")
    print("="*70)
    
    # NSGA-II算法参数配置
    nsga2_configs = [
        {
            "name": "快速测试配置",
            "population_size": 30,
            "generations": 50,
            "description": "适合快速验证和调试"
        },
        {
            "name": "标准配置",
            "population_size": 50,
            "generations": 100,
            "description": "平衡求解质量和计算时间"
        },
        {
            "name": "高精度配置",
            "population_size": 100,
            "generations": 200,
            "description": "追求更好的帕累托前沿质量"
        }
    ]
    
    print("可用的算法配置:")
    for i, config in enumerate(nsga2_configs):
        print(f"  {i+1}. {config['name']}: 种群={config['population_size']}, "
              f"代数={config['generations']} - {config['description']}")
    
    # 默认使用标准配置
    selected_config = nsga2_configs[2]  # 标准配置
    print(f"\n使用配置: {selected_config['name']}")
    
    try:
        import time
        start_time = time.time()
        
        # 执行NSGA-II优化
        nsga2_result = solve_evtol_nsga2(
            tasks=tasks,
            evtols=evtols,
            task_chains=task_chains,
            population_size=selected_config['population_size'],
            generations=selected_config['generations']
        )
        
        solve_time = time.time() - start_time
        
        # 分析优化结果
        pareto_front = nsga2_result["pareto_front"]
        print(f"\n🎯 NSGA-II优化完成!")
        print(f"   运行时间: {solve_time:.2f}秒")
        print(f"   帕累托前沿大小: {len(pareto_front)}个非支配解")
        
        if len(pareto_front) > 0:
            # 提取目标函数值
            energies = [ind.objectives[0] for ind in pareto_front]
            delays = [ind.objectives[1] for ind in pareto_front]
            
            print(f"\n📊 帕累托前沿统计:")
            print(f"   能耗范围: {min(energies):.1f} - {max(energies):.1f}")
            print(f"   延误范围: {min(delays):.1f} - {max(delays):.1f}分钟")
            print(f"   平均能耗: {sum(energies)/len(energies):.1f}")
            print(f"   平均延误: {sum(delays)/len(delays):.1f}分钟")
            
            # 展示代表性解决方案
            print(f"\n🌟 代表性解决方案:")
            print("="*50)
            
            # 最低能耗解
            idx_min_energy = min(range(len(energies)), key=lambda i: energies[i])
            print(f"🔋 最低能耗方案:")
            print(f"   总能耗: {energies[idx_min_energy]:.1f}")
            print(f"   总延误: {delays[idx_min_energy]:.1f}分钟")
            
            # 最低延误解
            idx_min_delay = min(range(len(delays)), key=lambda i: delays[i])
            print(f"\n⏰ 最低延误方案:")
            print(f"   总能耗: {energies[idx_min_delay]:.1f}")
            print(f"   总延误: {delays[idx_min_delay]:.1f}分钟")
            
            # 均衡解（如果存在多个解）
            if len(pareto_front) > 2:
                # 计算每个解到理想点的距离（归一化后）
                min_energy_norm = min(energies)
                max_energy_norm = max(energies)
                min_delay_norm = min(delays)
                max_delay_norm = max(delays)
                
                distances = []
                for i in range(len(pareto_front)):
                    if max_energy_norm > min_energy_norm and max_delay_norm > min_delay_norm:
                        energy_norm = (energies[i] - min_energy_norm) / (max_energy_norm - min_energy_norm)
                        delay_norm = (delays[i] - min_delay_norm) / (max_delay_norm - min_delay_norm)
                        distance = (energy_norm**2 + delay_norm**2)**0.5
                        distances.append(distance)
                    else:
                        distances.append(float('inf'))
                
                if distances and min(distances) != float('inf'):
                    idx_balanced = min(range(len(distances)), key=lambda i: distances[i])
                    print(f"\n⚖️  均衡权衡方案:")
                    print(f"   总能耗: {energies[idx_balanced]:.1f}")
                    print(f"   总延误: {delays[idx_balanced]:.1f}分钟")
            
            # 显示最优解的详细调度信息
            best_solution = pareto_front[idx_min_energy]  # 以最低能耗解为例
            schedule = best_solution.schedule
            
            print(f"\n📋 最低能耗方案的详细调度:")
            print("="*60)
            
            # 按开始时间排序显示任务
            task_details = []
            for task_id in range(len(tasks)):
                if task_id < len(schedule["task_assignments"]):
                    evtol_id = schedule["task_assignments"][task_id]
                    route_id = schedule["task_routes"][task_id]
                    start_time = schedule["task_start_times"][task_id]
                    end_time = schedule["task_end_times"][task_id]
                    
                    if evtol_id >= 0:  # 任务被分配
                        delay = start_time - tasks[task_id]["earliest_start"]
                        task_details.append({
                            "task_id": task_id,
                            "evtol_id": evtol_id,
                            "route_id": route_id,
                            "start_time": start_time,
                            "end_time": end_time,
                            "delay": delay,
                            "from": tasks[task_id]["from"],
                            "to": tasks[task_id]["to"]
                        })
            
            # 按开始时间排序
            task_details.sort(key=lambda x: x["start_time"])
            
            for task in task_details[:15]:  # 显示前15个任务
                print(f"任务{task['task_id']:2d}: eVTOL{task['evtol_id']} "
                      f"{locations[task['from']]}→{locations[task['to']]} "
                      f"{task['start_time']:3.0f}-{task['end_time']:3.0f}分钟 "
                      f"航线{task['route_id']} 延误{task['delay']:3.0f}分")
            
            if len(task_details) > 15:
                print(f"... 共{len(task_details)}个被分配的任务")
            
            # 生成可视化图表
            print(f"\n📈 正在生成可视化图表...")
            try:
                visualize_pareto_front(nsga2_result, "picture_result/evtol_pareto_front_example_nsga2.png")
                visualize_evolution_history(nsga2_result, "picture_result/evtol_evolution_history_example_nsga2.png")
                
                print(f"\n✅ NSGA-II优化示例运行成功！")
                print(f"\n📊 结果文件已生成:")
                print(f"   📈 picture_result/evtol_pareto_front_example_nsga2.png - 帕累托前沿图")
                print(f"   📊 picture_result/evtol_evolution_history_example_nsga2.png - 进化历史图")
                
                print(f"\n🎉 多目标优化优势:")
                print(f"   ✓ 无需预设权重，获得多个权衡方案")
                print(f"   ✓ 决策者可根据实际需求选择最适合的方案")
                print(f"   ✓ 帕累托前沿展示了能耗与延误的权衡关系")
                print(f"   ✓ 提供了{len(pareto_front)}个非支配的优化方案")
                
            except Exception as e:
                print(f"可视化生成失败: {e}")
                
        else:
            print("❌ 未找到可行的帕累托解")
            
    except Exception as e:
        print(f"\n❌ NSGA-II求解过程中出现错误：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 
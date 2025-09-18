import sys
import os

# 添加项目根目录到路径，以便导入数据定义
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# 导入任务串方法
from evtol_scheduling_gurobi import solve_evtol_scheduling_with_task_chains
from evtol_scheduling_gurobi import visualize_schedule_gurobi as visualize_gurobi
from evtol_scheduling_gurobi import visualize_schedule_table_gurobi as visualize_table_gurobi

# 导入数据定义
from data_definitions import get_tasks, get_evtols, get_locations


def main():
    """
    使用任务串方法的示例脚本
    """
    print("eVTOL 调度优化示例 - 任务串方法")
    
    # 获取地点信息
    locations = get_locations()
    
        # 获取任务和eVTOL数据
    tasks = get_tasks()
    evtols = get_evtols()
    
    # 设置时间范围（分钟）
    time_horizon = 720  # 12小时
    
    # 求解调度问题
    print("正在使用任务串方法求解eVTOL调度优化问题...")
    
    try:
        result = solve_evtol_scheduling_with_task_chains(
            tasks=tasks,
            evtols=evtols,
            time_horizon=time_horizon,
            max_chain_length=int(len(tasks)/len(evtols))+1,  # 最大任务串长度
            verbose=True
        )
    except Exception as e:
        print(f"求解过程中出现错误：{e}")
        sys.exit(1)
    
    # 检查求解状态
    if result["status"] in ["optimal", "time_limit"]:
        print(f"\n找到解！目标函数值: {result['objective_value']:.4f}")
        print(f"\n=== 原始目标值 ===")
        print(f"总能量消耗: {result['total_energy_consumption']:.2f} SOC")
        print(f"总延误时间: {result['total_delay']:.2f} 分钟")
        
        print(f"\n=== 基准化目标值 ===")
        print(f"基准化能耗: {result['benchmarked_energy']:.4f} (基准值: {result['energy_baseline']:.2f})")
        print(f"基准化延误: {result['benchmarked_delay']:.4f} (基准值: {result['delay_baseline']:.2f})")
        print(f"能耗权重: 0.2, 延误权重: 0.8")
        
        
        # 可视化调度结果
        print("\n生成调度甘特图...")
        visualize_gurobi(result, time_horizon)

        # 生成任务调度表
        print("\n生成任务调度表...")
        visualize_table_gurobi(result)
        
    else:
        print(f"无法找到可行解: {result['status']}")

if __name__ == "__main__":
    main()

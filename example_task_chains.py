import sys

# 导入任务串方法
from evtol_scheduling_gurobi import solve_evtol_scheduling_with_task_chains
from evtol_scheduling_gurobi import visualize_schedule_gurobi as visualize_gurobi
from evtol_scheduling_gurobi import visualize_schedule_table_gurobi as visualize_table_gurobi


def main():
    """
    使用任务串方法的示例脚本
    """
    print("eVTOL 调度优化示例 - 任务串方法")
    
    # 定义地点
    locations = {
        1: "高铁站",
        2: "旅游区", 
        3: "居民区",
        4: "商业区"
    }
    
    # 随机分布的任务 - 起终点和时间都随机混乱
    tasks = [
        {"id": 0, "from": 3, "to": 2, "earliest_start": 45, "latest_start": 180, "duration": [4, 7, 10], "soc_consumption": [19, 32, 50]},
        {"id": 1, "from": 1, "to": 4, "earliest_start": 210, "latest_start": 350, "duration": [5, 8, 11], "soc_consumption": [26, 39, 57]},
        {"id": 2, "from": 4, "to": 3, "earliest_start": 15, "latest_start": 140, "duration": [5, 8, 11], "soc_consumption": [26, 39, 57]},

        {"id": 3, "from": 2, "to": 1, "earliest_start": 380, "latest_start": 520, "duration": [6, 8, 12], "soc_consumption": [28, 42, 61]},
        {"id": 4, "from": 3, "to": 4, "earliest_start": 75, "latest_start": 190, "duration": [4, 7, 10], "soc_consumption": [19, 32, 50]},

        {"id": 5, "from": 1, "to": 3, "earliest_start": 280, "latest_start": 420, "duration": [4, 7, 10], "soc_consumption": [19, 32, 50]},
        {"id": 6, "from": 4, "to": 2, "earliest_start": 125, "latest_start": 260, "duration": [5, 7, 10], "soc_consumption": [23, 36, 53]},
        {"id": 7, "from": 2, "to": 3, "earliest_start": 450, "latest_start": 580, "duration": [6, 8, 12], "soc_consumption": [28, 42, 61]},

        {"id": 8, "from": 3, "to": 1, "earliest_start": 160, "latest_start": 290, "duration": [5, 8, 11], "soc_consumption": [26, 39, 57]},
        {"id": 9, "from": 4, "to": 1, "earliest_start": 320, "latest_start": 480, "duration": [4, 7, 10], "soc_consumption": [19, 32, 50]},

        {"id": 10, "from": 2, "to": 4, "earliest_start": 35, "latest_start": 170, "duration": [5, 7, 10], "soc_consumption": [23, 36, 53]},
        {"id": 11, "from": 1, "to": 2, "earliest_start": 390, "latest_start": 530, "duration": [5, 8, 11], "soc_consumption": [26, 39, 57]},

        {"id": 12, "from": 3, "to": 4, "earliest_start": 220, "latest_start": 360, "duration": [6, 8, 12], "soc_consumption": [28, 42, 61]},
        {"id": 13, "from": 4, "to": 3, "earliest_start": 85, "latest_start": 200, "duration": [5, 8, 11], "soc_consumption": [26, 39, 57]},
        {"id": 14, "from": 1, "to": 4, "earliest_start": 480, "latest_start": 600, "duration": [4, 7, 10], "soc_consumption": [22, 36, 53]},

        {"id": 15, "from": 2, "to": 3, "earliest_start": 110, "latest_start": 240, "duration": [6, 9, 13], "soc_consumption": [30, 45, 65]},
        {"id": 16, "from": 4, "to": 1, "earliest_start": 270, "latest_start": 410, "duration": [5, 8, 12], "soc_consumption": [25, 40, 58]},
        {"id": 17, "from": 1, "to": 3, "earliest_start": 40, "latest_start": 160, "duration": [4, 6, 9], "soc_consumption": [20, 35, 52]},

        {"id": 18, "from": 3, "to": 2, "earliest_start": 350, "latest_start": 490, "duration": [7, 10, 14], "soc_consumption": [32, 48, 68]},
        {"id": 19, "from": 4, "to": 2, "earliest_start": 95, "latest_start": 220, "duration": [3, 5, 8], "soc_consumption": [18, 28, 42]},
        {"id": 20, "from": 2, "to": 1, "earliest_start": 430, "latest_start": 570, "duration": [6, 8, 11], "soc_consumption": [28, 42, 60]},

        # 下午任务 (13:00-16:00, 即420-600分钟)
        {"id": 21, "from": 3, "to": 4, "earliest_start": 220, "latest_start": 540, "duration": [5, 7, 10], "soc_consumption": [24, 38, 55]},
        {"id": 22, "from": 4, "to": 2, "earliest_start": 240, "latest_start": 560, "duration": [4, 6, 9], "soc_consumption": [22, 34, 50]},
        {"id": 23, "from": 2, "to": 1, "earliest_start": 260, "latest_start": 580, "duration": [5, 8, 12], "soc_consumption": [26, 40, 58]},

        #晚高峰任务 (16:00-18:00, 即600-720分钟)
        {"id": 24, "from": 4, "to": 3, "earliest_start": 200, "latest_start": 690, "duration": [6, 9, 13], "soc_consumption": [29, 44, 63]},

        # 新增10个任务 - 时间窗在50-250分钟之间 (7:50-10:10)
        {"id": 25, "from": 3, "to": 4, "earliest_start": 65, "latest_start": 195, "duration": [5, 8, 11], "soc_consumption": [26, 39, 57]},
        {"id": 26, "from": 1, "to": 2, "earliest_start": 340, "latest_start": 470, "duration": [6, 9, 12], "soc_consumption": [28, 42, 60]},
        {"id": 27, "from": 4, "to": 3, "earliest_start": 180, "latest_start": 320, "duration": [4, 7, 10], "soc_consumption": [22, 35, 52]},
        {"id": 28, "from": 2, "to": 4, "earliest_start": 25, "latest_start": 150, "duration": [5, 8, 11], "soc_consumption": [25, 38, 55]},
        {"id": 29, "from": 3, "to": 1, "earliest_start": 420, "latest_start": 560, "duration": [6, 8, 12], "soc_consumption": [30, 44, 62]},
        {"id": 30, "from": 1, "to": 4, "earliest_start": 140, "latest_start": 280, "duration": [4, 6, 9], "soc_consumption": [20, 32, 48]},
        {"id": 31, "from": 4, "to": 1, "earliest_start": 250, "latest_start": 390, "duration": [3, 5, 8], "soc_consumption": [18, 28, 42]},
        {"id": 32, "from": 2, "to": 3, "earliest_start": 55, "latest_start": 185, "duration": [6, 9, 13], "soc_consumption": [29, 43, 61]},
        {"id": 33, "from": 4, "to": 2, "earliest_start": 370, "latest_start": 510, "duration": [7, 10, 14], "soc_consumption": [32, 48, 68]},
        {"id": 34, "from": 3, "to": 1, "earliest_start": 190, "latest_start": 330, "duration": [5, 8, 11], "soc_consumption": [26, 40, 58]},

        {"id": 35, "from": 2, "to": 4, "earliest_start": 460, "latest_start": 590, "duration": [5, 7, 10], "soc_consumption": [24, 36, 52]},
        {"id": 36, "from": 1, "to": 3, "earliest_start": 105, "latest_start": 235, "duration": [4, 6, 9], "soc_consumption": [20, 30, 45]},
        {"id": 37, "from": 4, "to": 1, "earliest_start": 290, "latest_start": 430, "duration": [5, 8, 11], "soc_consumption": [25, 38, 55]},
        {"id": 38, "from": 3, "to": 2, "earliest_start": 15, "latest_start": 145, "duration": [6, 9, 12], "soc_consumption": [28, 42, 60]},
        
        {"id": 39, "from": 1, "to": 4, "earliest_start": 520, "latest_start": 650, "duration": [5, 8, 11], "soc_consumption": [26, 39, 57]},
        {"id": 40, "from": 2, "to": 1, "earliest_start": 170, "latest_start": 300, "duration": [4, 7, 10], "soc_consumption": [22, 35, 50]},
        {"id": 41, "from": 4, "to": 3, "earliest_start": 410, "latest_start": 540, "duration": [6, 8, 12], "soc_consumption": [28, 41, 59]},
        {"id": 42, "from": 3, "to": 4, "earliest_start": 230, "latest_start": 370, "duration": [5, 7, 10], "soc_consumption": [23, 36, 53]},
        
        # 下午晚些时段任务 (15:00-17:00, 即540-660分钟)
        {"id": 43, "from": 3, "to": 2, "earliest_start": 540, "latest_start": 620, "duration": [4, 6, 9], "soc_consumption": [21, 33, 48]},
        {"id": 44, "from": 2, "to": 1, "earliest_start": 560, "latest_start": 640, "duration": [5, 8, 11], "soc_consumption": [25, 38, 56]},
        {"id": 45, "from": 1, "to": 3, "earliest_start": 580, "latest_start": 660, "duration": [7, 10, 13], "soc_consumption": [31, 46, 65]},
        {"id": 46, "from": 4, "to": 2, "earliest_start": 600, "latest_start": 680, "duration": [4, 6, 9], "soc_consumption": [20, 32, 47]},
        
        # 早期灵活时段任务 (6:30-10:30, 即30-270分钟)
        {"id": 47, "from": 2, "to": 3, "earliest_start": 30, "latest_start": 150, "duration": [3, 5, 8], "soc_consumption": [18, 28, 42]},
        {"id": 48, "from": 3, "to": 4, "earliest_start": 70, "latest_start": 190, "duration": [5, 8, 11], "soc_consumption": [25, 38, 55]},
        {"id": 49, "from": 4, "to": 1, "earliest_start": 110, "latest_start": 230, "duration": [6, 9, 12], "soc_consumption": [29, 43, 62]},
        {"id": 50, "from": 1, "to": 2, "earliest_start": 150, "latest_start": 270, "duration": [5, 7, 10], "soc_consumption": [24, 37, 54]},
        
        # 跨时段长时间窗任务 (灵活调度)
        {"id": 51, "from": 2, "to": 4, "earliest_start": 200, "latest_start": 450, "duration": [5, 8, 11], "soc_consumption": [26, 39, 57]},
        {"id": 52, "from": 4, "to": 3, "earliest_start": 250, "latest_start": 500, "duration": [4, 7, 10], "soc_consumption": [22, 35, 52]},
        {"id": 53, "from": 3, "to": 1, "earliest_start": 300, "latest_start": 550, "duration": [6, 8, 11], "soc_consumption": [28, 42, 60]},
        {"id": 54, "from": 1, "to": 3, "earliest_start": 350, "latest_start": 600, "duration": [7, 10, 14], "soc_consumption": [32, 48, 68]},

        # 再新增30个任务 (ID: 55-84) - 更复杂的调度场景
        # 早晨高峰第二波 (7:00-9:00, 即60-180分钟)
        {"id": 55, "from": 3, "to": 4, "earliest_start": 60, "latest_start": 140, "duration": [4, 6, 9], "soc_consumption": [21, 34, 50]},
        {"id": 56, "from": 4, "to": 2, "earliest_start": 80, "latest_start": 160, "duration": [5, 7, 10], "soc_consumption": [24, 37, 54]},
        {"id": 57, "from": 2, "to": 1, "earliest_start": 100, "latest_start": 180, "duration": [6, 8, 11], "soc_consumption": [27, 40, 58]},
        {"id": 58, "from": 1, "to": 3, "earliest_start": 120, "latest_start": 200, "duration": [7, 9, 12], "soc_consumption": [30, 44, 62]},
        
        # 上午工作时段 (9:30-11:30, 即210-330分钟)
        {"id": 59, "from": 2, "to": 4, "earliest_start": 210, "latest_start": 290, "duration": [4, 7, 10], "soc_consumption": [22, 35, 52]},
        {"id": 60, "from": 4, "to": 1, "earliest_start": 230, "latest_start": 310, "duration": [6, 8, 12], "soc_consumption": [28, 42, 60]},
        {"id": 61, "from": 1, "to": 2, "earliest_start": 250, "latest_start": 330, "duration": [5, 8, 11], "soc_consumption": [26, 39, 57]},
        {"id": 62, "from": 3, "to": 1, "earliest_start": 270, "latest_start": 350, "duration": [6, 9, 12], "soc_consumption": [29, 43, 61]},
        
        # # 中午繁忙时段 (12:00-14:00, 即360-480分钟)
        # {"id": 63, "from": 1, "to": 4, "earliest_start": 360, "latest_start": 440, "duration": [5, 7, 10], "soc_consumption": [23, 36, 53]},
        # {"id": 64, "from": 4, "to": 3, "earliest_start": 380, "latest_start": 460, "duration": [4, 6, 9], "soc_consumption": [20, 33, 48]},
        # {"id": 65, "from": 3, "to": 2, "earliest_start": 400, "latest_start": 480, "duration": [3, 5, 8], "soc_consumption": [18, 28, 42]},
        # {"id": 66, "from": 2, "to": 3, "earliest_start": 420, "latest_start": 500, "duration": [4, 6, 9], "soc_consumption": [21, 33, 48]},
        
        # # 下午忙碌时段 (14:30-16:30, 即510-630分钟)
        # {"id": 67, "from": 1, "to": 2, "earliest_start": 510, "latest_start": 590, "duration": [5, 8, 11], "soc_consumption": [26, 39, 57]},
        # {"id": 68, "from": 2, "to": 4, "earliest_start": 530, "latest_start": 610, "duration": [5, 7, 10], "soc_consumption": [24, 37, 54]},
        # {"id": 69, "from": 4, "to": 1, "earliest_start": 550, "latest_start": 630, "duration": [6, 9, 12], "soc_consumption": [28, 42, 60]},
        # {"id": 70, "from": 3, "to": 4, "earliest_start": 570, "latest_start": 650, "duration": [5, 8, 11], "soc_consumption": [25, 38, 55]},
        
        # # 傍晚时段 (17:00-18:30, 即660-750分钟)
        # {"id": 71, "from": 4, "to": 2, "earliest_start": 660, "latest_start": 720, "duration": [4, 6, 9], "soc_consumption": [20, 32, 47]},
        # {"id": 72, "from": 2, "to": 3, "earliest_start": 680, "latest_start": 740, "duration": [3, 5, 8], "soc_consumption": [18, 28, 42]},
        # {"id": 73, "from": 3, "to": 1, "earliest_start": 700, "latest_start": 760, "duration": [6, 8, 11], "soc_consumption": [28, 42, 60]},
        # {"id": 74, "from": 1, "to": 4, "earliest_start": 720, "latest_start": 780, "duration": [5, 7, 10], "soc_consumption": [23, 36, 53]},
        
        # # 早期-中午跨度任务 (长时间窗灵活任务)
        # {"id": 75, "from": 2, "to": 1, "earliest_start": 90, "latest_start": 400, "duration": [5, 8, 11], "soc_consumption": [26, 40, 58]},
        # {"id": 76, "from": 1, "to": 3, "earliest_start": 150, "latest_start": 450, "duration": [7, 10, 14], "soc_consumption": [32, 48, 68]},
        # {"id": 77, "from": 3, "to": 2, "earliest_start": 200, "latest_start": 500, "duration": [4, 6, 9], "soc_consumption": [21, 33, 48]},
        # {"id": 78, "from": 4, "to": 3, "earliest_start": 250, "latest_start": 550, "duration": [4, 7, 10], "soc_consumption": [22, 35, 52]},
        
        # # 中午-下午跨度任务
        # {"id": 79, "from": 1, "to": 2, "earliest_start": 300, "latest_start": 600, "duration": [5, 7, 10], "soc_consumption": [24, 37, 54]},
        # {"id": 80, "from": 2, "to": 4, "earliest_start": 350, "latest_start": 650, "duration": [5, 8, 11], "soc_consumption": [26, 39, 57]},
        # {"id": 81, "from": 4, "to": 1, "earliest_start": 400, "latest_start": 700, "duration": [6, 9, 12], "soc_consumption": [28, 42, 60]},
        # {"id": 82, "from": 3, "to": 4, "earliest_start": 450, "latest_start": 720, "duration": [5, 8, 11], "soc_consumption": [25, 38, 55]},
        
        # # 紧急任务 (短时间窗，高优先级)
        # {"id": 83, "from": 1, "to": 4, "earliest_start": 180, "latest_start": 220, "duration": [5, 7, 10], "soc_consumption": [23, 36, 53]},
        # {"id": 84, "from": 4, "to": 2, "earliest_start": 380, "latest_start": 420, "duration": [4, 6, 9], "soc_consumption": [20, 32, 47]},
 
    ]
    
    # 创建eVTOL列表
    evtols = [
        {"id": 0, "initial_position": 3, "initial_soc": 100, "initial_state": 0},
        {"id": 1, "initial_position": 1, "initial_soc": 100, "initial_state": 0},
        {"id": 2, "initial_position": 2, "initial_soc": 100, "initial_state": 0},
        {"id": 3, "initial_position": 4, "initial_soc": 100, "initial_state": 0},
        {"id": 4, "initial_position": 4, "initial_soc": 100, "initial_state": 0},
        {"id": 5, "initial_position": 1, "initial_soc": 100, "initial_state": 0},
        # {"id": 6, "initial_position": 2, "initial_soc": 100, "initial_state": 0},
        # {"id": 7, "initial_position": 3, "initial_soc": 100, "initial_state": 0},
    ]
    
    # 设置时间范围（分钟）
    time_horizon = 720  # 12小时
    
    # 求解调度问题
    print("正在使用任务串方法求解eVTOL调度优化问题...")
    
    try:
        result = solve_evtol_scheduling_with_task_chains(
            tasks=tasks,
            evtols=evtols,
            time_horizon=time_horizon,
            max_chain_length=9,  # 最大任务串长度
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
        print(f"能耗权重: 0.3, 延误权重: 0.7")
        
        
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

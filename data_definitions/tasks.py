#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eVTOL调度系统任务定义
定义系统中所有的任务信息
"""

# 任务数据定义 - 基于example_task_chains.py
TASKS = [
    # 新增任务 (ID: 1-20) - earliest_start = 0，注释状态
    {"id": 1, "from": 1, "to": 2, "earliest_start": 0, "latest_start": 130, "duration": [15, 17, 19], "soc_consumption": [26, 35, 45]},
    {"id": 2, "from": 2, "to": 3, "earliest_start": 0, "latest_start": 130, "duration": [17, 19, 22], "soc_consumption": [39, 48, 58]},
    {"id": 3, "from": 3, "to": 4, "earliest_start": 0, "latest_start": 130, "duration": [14, 16, 18], "soc_consumption": [23, 32, 42]},
    {"id": 4, "from": 4, "to": 1, "earliest_start": 0, "latest_start": 130, "duration": [15, 17, 19], "soc_consumption": [23, 32, 42]},
    {"id": 5, "from": 1, "to": 3, "earliest_start": 0, "latest_start": 130, "duration": [14, 16, 18], "soc_consumption": [19, 28, 38]},
    
    {"id": 6, "from": 3, "to": 1, "earliest_start": 0, "latest_start": 130, "duration": [14, 16, 18], "soc_consumption": [19, 28, 38]},
    {"id": 7, "from": 1, "to": 4, "earliest_start": 0, "latest_start": 130, "duration": [15, 17, 19], "soc_consumption": [23, 32, 42]},
    {"id": 8, "from": 4, "to": 2, "earliest_start": 0, "latest_start": 130, "duration": [14, 16, 18], "soc_consumption": [32, 41, 51]},
    {"id": 9, "from": 2, "to": 4, "earliest_start": 0, "latest_start": 130, "duration": [14, 16, 18], "soc_consumption": [32, 41, 51]},
    {"id": 10, "from": 4, "to": 3, "earliest_start": 0, "latest_start": 130, "duration": [14, 16, 18], "soc_consumption": [23, 32, 42]},
    
    {"id": 11, "from": 3, "to": 2, "earliest_start": 0, "latest_start": 130, "duration": [17, 19, 22], "soc_consumption": [39, 48, 58]},
    {"id": 12, "from": 2, "to": 1, "earliest_start": 0, "latest_start": 130, "duration": [17, 19, 22], "soc_consumption": [39, 48, 58]},
    {"id": 13, "from": 1, "to": 2, "earliest_start": 0, "latest_start": 130, "duration": [15, 17, 19], "soc_consumption": [26, 35, 45]},
    {"id": 14, "from": 2, "to": 3, "earliest_start": 0, "latest_start": 130, "duration": [17, 19, 22], "soc_consumption": [39, 48, 58]},
    {"id": 15, "from": 3, "to": 4, "earliest_start": 0, "latest_start": 130, "duration": [14, 16, 18], "soc_consumption": [23, 32, 42]},
    
    {"id": 16, "from": 4, "to": 1, "earliest_start": 0, "latest_start": 130, "duration": [15, 17, 19], "soc_consumption": [23, 32, 42]},
    {"id": 17, "from": 1, "to": 3, "earliest_start": 0, "latest_start": 130, "duration": [14, 16, 18], "soc_consumption": [19, 28, 38]},
    {"id": 18, "from": 3, "to": 1, "earliest_start": 0, "latest_start": 130, "duration": [14, 16, 18], "soc_consumption": [19, 28, 38]},
    {"id": 19, "from": 1, "to": 4, "earliest_start": 0, "latest_start": 130, "duration": [15, 17, 19], "soc_consumption": [23, 32, 42]},
    {"id": 20, "from": 4, "to": 2, "earliest_start": 0, "latest_start": 130, "duration": [14, 16, 18], "soc_consumption": [32, 41, 51]},
    
    # 原有任务保持不变
    # {"id": 0, "from": 3, "to": 2, "earliest_start": 45, "latest_start": 180, "duration": [4, 7, 10], "soc_consumption": [19, 32, 50]},
    # {"id": 1, "from": 1, "to": 4, "earliest_start": 210, "latest_start": 350, "duration": [5, 8, 11], "soc_consumption": [26, 39, 57]},
    # {"id": 2, "from": 4, "to": 3, "earliest_start": 15, "latest_start": 140, "duration": [5, 8, 11], "soc_consumption": [26, 39, 57]},

    # {"id": 3, "from": 2, "to": 1, "earliest_start": 380, "latest_start": 520, "duration": [6, 8, 12], "soc_consumption": [28, 42, 61]},
    # {"id": 4, "from": 3, "to": 4, "earliest_start": 75, "latest_start": 190, "duration": [4, 7, 10], "soc_consumption": [19, 32, 50]},

    # {"id": 5, "from": 1, "to": 3, "earliest_start": 280, "latest_start": 420, "duration": [4, 7, 10], "soc_consumption": [19, 32, 50]},
    # {"id": 6, "from": 4, "to": 2, "earliest_start": 125, "latest_start": 260, "duration": [5, 7, 10], "soc_consumption": [23, 36, 53]},
    # {"id": 7, "from": 2, "to": 3, "earliest_start": 450, "latest_start": 580, "duration": [6, 8, 12], "soc_consumption": [28, 42, 61]},

    # {"id": 8, "from": 3, "to": 1, "earliest_start": 160, "latest_start": 290, "duration": [5, 8, 11], "soc_consumption": [26, 39, 57]},
    # {"id": 9, "from": 4, "to": 1, "earliest_start": 320, "latest_start": 480, "duration": [4, 7, 10], "soc_consumption": [19, 32, 50]},

    # {"id": 10, "from": 2, "to": 4, "earliest_start": 35, "latest_start": 170, "duration": [5, 7, 10], "soc_consumption": [23, 36, 53]},
    # {"id": 11, "from": 1, "to": 2, "earliest_start": 390, "latest_start": 530, "duration": [5, 8, 11], "soc_consumption": [26, 39, 57]},

    # {"id": 12, "from": 3, "to": 4, "earliest_start": 220, "latest_start": 360, "duration": [6, 8, 12], "soc_consumption": [28, 42, 61]},
    # {"id": 13, "from": 4, "to": 3, "earliest_start": 85, "latest_start": 200, "duration": [5, 8, 11], "soc_consumption": [26, 39, 57]},
    # {"id": 14, "from": 1, "to": 4, "earliest_start": 480, "latest_start": 600, "duration": [4, 7, 10], "soc_consumption": [22, 36, 53]},

    # {"id": 15, "from": 2, "to": 3, "earliest_start": 110, "latest_start": 240, "duration": [6, 9, 13], "soc_consumption": [30, 45, 65]},
    # {"id": 16, "from": 4, "to": 1, "earliest_start": 270, "latest_start": 410, "duration": [5, 8, 12], "soc_consumption": [25, 40, 58]},
    # {"id": 17, "from": 1, "to": 3, "earliest_start": 40, "latest_start": 160, "duration": [4, 6, 9], "soc_consumption": [20, 35, 52]},

    # {"id": 18, "from": 3, "to": 2, "earliest_start": 350, "latest_start": 490, "duration": [7, 10, 14], "soc_consumption": [32, 48, 68]},
    # {"id": 19, "from": 4, "to": 2, "earliest_start": 95, "latest_start": 220, "duration": [3, 5, 8], "soc_consumption": [18, 28, 42]},
    # {"id": 20, "from": 2, "to": 1, "earliest_start": 430, "latest_start": 570, "duration": [6, 8, 11], "soc_consumption": [28, 42, 60]},

    # # 下午任务 (13:00-16:00, 即420-600分钟)
    # {"id": 21, "from": 3, "to": 4, "earliest_start": 220, "latest_start": 540, "duration": [5, 7, 10], "soc_consumption": [24, 38, 55]},
    # {"id": 22, "from": 4, "to": 2, "earliest_start": 240, "latest_start": 560, "duration": [4, 6, 9], "soc_consumption": [22, 34, 50]},
    # {"id": 23, "from": 2, "to": 1, "earliest_start": 260, "latest_start": 580, "duration": [5, 8, 12], "soc_consumption": [26, 40, 58]},

    # #晚高峰任务 (16:00-18:00, 即600-720分钟)
    # {"id": 24, "from": 4, "to": 3, "earliest_start": 200, "latest_start": 690, "duration": [6, 9, 13], "soc_consumption": [29, 44, 63]},

    # # 新增10个任务 - 时间窗在50-250分钟之间 (7:50-10:10)
    # {"id": 25, "from": 3, "to": 4, "earliest_start": 65, "latest_start": 195, "duration": [5, 8, 11], "soc_consumption": [26, 39, 57]},
    # {"id": 26, "from": 1, "to": 2, "earliest_start": 340, "latest_start": 470, "duration": [6, 9, 12], "soc_consumption": [28, 42, 60]},
    # {"id": 27, "from": 4, "to": 3, "earliest_start": 180, "latest_start": 320, "duration": [4, 7, 10], "soc_consumption": [22, 35, 52]},
    # {"id": 28, "from": 2, "to": 4, "earliest_start": 25, "latest_start": 150, "duration": [5, 8, 11], "soc_consumption": [25, 38, 55]},
    # {"id": 29, "from": 3, "to": 1, "earliest_start": 420, "latest_start": 560, "duration": [6, 8, 12], "soc_consumption": [30, 44, 62]},
    # {"id": 30, "from": 1, "to": 4, "earliest_start": 140, "latest_start": 280, "duration": [4, 6, 9], "soc_consumption": [20, 32, 48]},
    # {"id": 31, "from": 4, "to": 1, "earliest_start": 250, "latest_start": 390, "duration": [3, 5, 8], "soc_consumption": [18, 28, 42]},
    # {"id": 32, "from": 2, "to": 3, "earliest_start": 55, "latest_start": 185, "duration": [6, 9, 13], "soc_consumption": [29, 43, 61]},
    # {"id": 33, "from": 4, "to": 2, "earliest_start": 370, "latest_start": 510, "duration": [7, 10, 14], "soc_consumption": [32, 48, 68]},
    # {"id": 34, "from": 3, "to": 1, "earliest_start": 190, "latest_start": 330, "duration": [5, 8, 11], "soc_consumption": [26, 40, 58]},

    # {"id": 35, "from": 2, "to": 4, "earliest_start": 460, "latest_start": 590, "duration": [5, 7, 10], "soc_consumption": [24, 36, 52]},
    # {"id": 36, "from": 1, "to": 3, "earliest_start": 105, "latest_start": 235, "duration": [4, 6, 9], "soc_consumption": [20, 30, 45]},
    # {"id": 37, "from": 4, "to": 1, "earliest_start": 290, "latest_start": 430, "duration": [5, 8, 11], "soc_consumption": [25, 38, 55]},
    # {"id": 38, "from": 3, "to": 2, "earliest_start": 15, "latest_start": 145, "duration": [6, 9, 12], "soc_consumption": [28, 42, 60]},
    
    # {"id": 39, "from": 1, "to": 4, "earliest_start": 520, "latest_start": 650, "duration": [5, 8, 11], "soc_consumption": [26, 39, 57]},
    # {"id": 40, "from": 2, "to": 1, "earliest_start": 170, "latest_start": 300, "duration": [4, 7, 10], "soc_consumption": [22, 35, 50]},
    # {"id": 41, "from": 4, "to": 3, "earliest_start": 410, "latest_start": 540, "duration": [6, 8, 12], "soc_consumption": [28, 41, 59]},
    # {"id": 42, "from": 3, "to": 4, "earliest_start": 230, "latest_start": 370, "duration": [5, 7, 10], "soc_consumption": [23, 36, 53]},
    
    # # 新增50个随机任务 (ID: 43-92)
    # {"id": 43, "from": 1, "to": 2, "earliest_start": 80, "latest_start": 210, "duration": [5, 7, 9], "soc_consumption": [26, 35, 45]},
    # {"id": 44, "from": 3, "to": 4, "earliest_start": 150, "latest_start": 280, "duration": [4, 6, 8], "soc_consumption": [23, 32, 42]},
    # {"id": 45, "from": 2, "to": 1, "earliest_start": 320, "latest_start": 450, "duration": [7, 9, 12], "soc_consumption": [39, 48, 58]},
    # {"id": 46, "from": 4, "to": 3, "earliest_start": 45, "latest_start": 175, "duration": [4, 6, 8], "soc_consumption": [23, 32, 42]},
    # {"id": 47, "from": 1, "to": 4, "earliest_start": 480, "latest_start": 610, "duration": [5, 7, 9], "soc_consumption": [23, 32, 42]},
    
    # {"id": 48, "from": 2, "to": 4, "earliest_start": 90, "latest_start": 220, "duration": [4, 6, 8], "soc_consumption": [32, 41, 51]},
    # {"id": 49, "from": 4, "to": 2, "earliest_start": 380, "latest_start": 510, "duration": [4, 6, 8], "soc_consumption": [32, 41, 51]},
    # {"id": 50, "from": 3, "to": 1, "earliest_start": 200, "latest_start": 330, "duration": [4, 6, 8], "soc_consumption": [19, 28, 38]},
    # # {"id": 51, "from": 1, "to": 3, "earliest_start": 70, "latest_start": 200, "duration": [4, 6, 8], "soc_consumption": [19, 28, 38]},
    # {"id": 52, "from": 4, "to": 1, "earliest_start": 260, "latest_start": 390, "duration": [5, 7, 9], "soc_consumption": [23, 32, 42]},
    
    # {"id": 53, "from": 2, "to": 3, "earliest_start": 420, "latest_start": 550, "duration": [7, 9, 12], "soc_consumption": [39, 48, 58]},
    # {"id": 54, "from": 3, "to": 2, "earliest_start": 120, "latest_start": 250, "duration": [7, 9, 12], "soc_consumption": [39, 48, 58]},
    # {"id": 55, "from": 1, "to": 2, "earliest_start": 340, "latest_start": 470, "duration": [5, 7, 9], "soc_consumption": [26, 35, 45]},
    # {"id": 56, "from": 4, "to": 3, "earliest_start": 180, "latest_start": 310, "duration": [4, 6, 8], "soc_consumption": [23, 32, 42]},
    # {"id": 57, "from": 2, "to": 1, "earliest_start": 30, "latest_start": 160, "duration": [7, 9, 12], "soc_consumption": [39, 48, 58]},
    
    # {"id": 58, "from": 3, "to": 4, "earliest_start": 450, "latest_start": 580, "duration": [4, 6, 8], "soc_consumption": [23, 32, 42]},
    # {"id": 59, "from": 1, "to": 4, "earliest_start": 110, "latest_start": 240, "duration": [5, 7, 9], "soc_consumption": [23, 32, 42]},
    # {"id": 60, "from": 4, "to": 2, "earliest_start": 290, "latest_start": 420, "duration": [4, 6, 8], "soc_consumption": [32, 41, 51]},
    # # {"id": 61, "from": 2, "to": 4, "earliest_start": 500, "latest_start": 630, "duration": [4, 6, 8], "soc_consumption": [32, 41, 51]},
    # {"id": 62, "from": 3, "to": 1, "earliest_start": 60, "latest_start": 190, "duration": [4, 6, 8], "soc_consumption": [19, 28, 38]},
    
    # {"id": 63, "from": 1, "to": 3, "earliest_start": 360, "latest_start": 490, "duration": [4, 6, 8], "soc_consumption": [19, 28, 38]},
    # {"id": 64, "from": 4, "to": 1, "earliest_start": 140, "latest_start": 270, "duration": [5, 7, 9], "soc_consumption": [23, 32, 42]},
    # {"id": 65, "from": 2, "to": 3, "earliest_start": 220, "latest_start": 350, "duration": [7, 9, 12], "soc_consumption": [39, 48, 58]},
    # {"id": 66, "from": 3, "to": 2, "earliest_start": 480, "latest_start": 610, "duration": [7, 9, 12], "soc_consumption": [39, 48, 58]},
    # {"id": 67, "from": 1, "to": 2, "earliest_start": 20, "latest_start": 150, "duration": [5, 7, 9], "soc_consumption": [26, 35, 45]},
    
    # {"id": 68, "from": 4, "to": 3, "earliest_start": 300, "latest_start": 430, "duration": [4, 6, 8], "soc_consumption": [23, 32, 42]},
    # {"id": 69, "from": 2, "to": 1, "earliest_start": 160, "latest_start": 290, "duration": [7, 9, 12], "soc_consumption": [39, 48, 58]},
    # {"id": 70, "from": 3, "to": 4, "earliest_start": 240, "latest_start": 370, "duration": [4, 6, 8], "soc_consumption": [23, 32, 42]},
    # {"id": 71, "from": 1, "to": 4, "earliest_start": 520, "latest_start": 650, "duration": [5, 7, 9], "soc_consumption": [23, 32, 42]},
    # {"id": 72, "from": 4, "to": 2, "earliest_start": 100, "latest_start": 230, "duration": [4, 6, 8], "soc_consumption": [32, 41, 51]},
    
    # {"id": 73, "from": 2, "to": 4, "earliest_start": 400, "latest_start": 530, "duration": [4, 6, 8], "soc_consumption": [32, 41, 51]},
    # {"id": 74, "from": 3, "to": 1, "earliest_start": 270, "latest_start": 400, "duration": [4, 6, 8], "soc_consumption": [19, 28, 38]},
    # {"id": 75, "from": 1, "to": 3, "earliest_start": 130, "latest_start": 260, "duration": [4, 6, 8], "soc_consumption": [19, 28, 38]},
    # {"id": 76, "from": 4, "to": 1, "earliest_start": 460, "latest_start": 590, "duration": [5, 7, 9], "soc_consumption": [23, 32, 42]},
    # {"id": 77, "from": 2, "to": 3, "earliest_start": 50, "latest_start": 180, "duration": [7, 9, 12], "soc_consumption": [39, 48, 58]},
    
    # {"id": 78, "from": 3, "to": 2, "earliest_start": 310, "latest_start": 440, "duration": [7, 9, 12], "soc_consumption": [39, 48, 58]},
    # {"id": 79, "from": 1, "to": 2, "earliest_start": 190, "latest_start": 320, "duration": [5, 7, 9], "soc_consumption": [26, 35, 45]},
    # {"id": 80, "from": 4, "to": 3, "earliest_start": 40, "latest_start": 170, "duration": [4, 6, 8], "soc_consumption": [23, 32, 42]},
    # {"id": 81, "from": 2, "to": 1, "earliest_start": 440, "latest_start": 570, "duration": [7, 9, 12], "soc_consumption": [39, 48, 58]},
    # {"id": 82, "from": 3, "to": 4, "earliest_start": 85, "latest_start": 215, "duration": [4, 6, 8], "soc_consumption": [23, 32, 42]},
    
    # {"id": 83, "from": 1, "to": 4, "earliest_start": 350, "latest_start": 480, "duration": [5, 7, 9], "soc_consumption": [23, 32, 42]},
    # {"id": 84, "from": 4, "to": 2, "earliest_start": 210, "latest_start": 340, "duration": [4, 6, 8], "soc_consumption": [32, 41, 51]},
    # {"id": 85, "from": 2, "to": 4, "earliest_start": 25, "latest_start": 155, "duration": [4, 6, 8], "soc_consumption": [32, 41, 51]},
    # {"id": 86, "from": 3, "to": 1, "earliest_start": 390, "latest_start": 520, "duration": [4, 6, 8], "soc_consumption": [19, 28, 38]},
    # {"id": 87, "from": 1, "to": 3, "earliest_start": 250, "latest_start": 380, "duration": [4, 6, 8], "soc_consumption": [19, 28, 38]},
    
    # {"id": 88, "from": 4, "to": 1, "earliest_start": 75, "latest_start": 205, "duration": [5, 7, 9], "soc_consumption": [23, 32, 42]},
    # {"id": 89, "from": 2, "to": 3, "earliest_start": 470, "latest_start": 600, "duration": [7, 9, 12], "soc_consumption": [39, 48, 58]},
    # {"id": 90, "from": 3, "to": 2, "earliest_start": 165, "latest_start": 295, "duration": [7, 9, 12], "soc_consumption": [39, 48, 58]},
    # {"id": 91, "from": 1, "to": 2, "earliest_start": 410, "latest_start": 540, "duration": [5, 7, 9], "soc_consumption": [26, 35, 45]},
    # {"id": 92, "from": 4, "to": 3, "earliest_start": 330, "latest_start": 460, "duration": [4, 6, 8], "soc_consumption": [23, 32, 42]},
    

]

def get_tasks():
    """获取所有任务列表"""
    return TASKS

def get_task_by_id(task_id):
    """根据任务ID获取任务信息"""
    for task in TASKS:
        if task["id"] == task_id:
            return task
    return None

def get_tasks_by_location(from_location=None, to_location=None):
    """根据起点或终点筛选任务"""
    filtered_tasks = []
    for task in TASKS:
        if from_location is not None and task["from"] != from_location:
            continue
        if to_location is not None and task["to"] != to_location:
            continue
        filtered_tasks.append(task)
    return filtered_tasks 
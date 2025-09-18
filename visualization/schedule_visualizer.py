#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eVTOL调度可视化器
提供调度表和甘特图的统一可视化功能
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from typing import Dict, List

# 设置中文字体
matplotlib.rcParams['font.family'] = 'FangSong'

def visualize_schedule_table(schedule: List[Dict], algorithm_name: str = "", save_path: str = None) -> None:
    """
    生成任务调度表的可视化图表
    
    参数:
        schedule: 调度结果列表，每个元素包含task_id, evtol_id, start_time, end_time, route, from, to, delay等
        algorithm_name: 算法名称，用于标题
        save_path: 保存路径，如果为None则使用默认路径
    """
    if not schedule:
        print("无调度数据可视化")
        return

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
        delay = task.get("delay", 0)
        table_data.append([
            task["task_id"],
            task["evtol_id"],
            task["from"],
            task["to"],
            task["start_time"],
            task["end_time"],
            task["route"],
            duration,
            f"{delay:.1f}"
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
    table.scale(1.2, 1.2)

    # 设置标题
    title = f'eVTOL 任务调度表'
    if algorithm_name:
        title += f' ({algorithm_name})'
    plt.title(title, pad=20)

    # 保存图形
    if save_path is None:
        save_path = f'picture_result/evtol_schedule_table_{algorithm_name.lower().replace(" ", "_")}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def visualize_schedule_gantt(schedule: List[Dict], algorithm_name: str = "", save_path: str = None, time_horizon: int = 720) -> None:
    """
    生成调度结果的可视化甘特图
    
    参数:
        schedule: 调度结果列表
        algorithm_name: 算法名称，用于标题
        save_path: 保存路径，如果为None则使用默认路径
        time_horizon: 调度时间范围（分钟）
    """
    if not schedule:
        print("无调度数据可视化")
        return

    # 确定eVTOL数量
    evtol_ids = set([task["evtol_id"] for task in schedule])
    if evtol_ids:
        num_evtols = max(evtol_ids) + 1
    else:
        num_evtols = 1

    # 创建图形
    fig, ax = plt.subplots(figsize=(20, max(6, num_evtols * 1.4)))

    # 定义颜色
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # 绘制任务
    for task in schedule:
        evtol_id = task["evtol_id"]
        start = task["start_time"]
        duration = task["end_time"] - task["start_time"]

        # 绘制任务块
        task_bar = ax.barh(evtol_id, duration, left=start, height=0.35,
                          color=colors[task["route"] % len(colors)],
                          edgecolor='white', linewidth=1.5, alpha=0.8)

        # 添加任务标签
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
    title = f'eVTOL 调度甘特图'
    if algorithm_name:
        title += f' ({algorithm_name})'
    ax.set_title(title)
    ax.set_yticks(range(num_evtols))
    ax.set_yticklabels([f'eVTOL {i}' for i in range(num_evtols)])
    
    # 设置时间刻度
    if schedule:
        min_time = min(task["start_time"] for task in schedule)
        max_time = max(task["end_time"] for task in schedule)
        time_range = max_time - min_time
        
        # 根据时间范围动态调整刻度间隔
        if time_range <= 120:
            tick_interval = 6
        elif time_range <= 300:
            tick_interval = 12
        elif time_range <= 600:
            tick_interval = 18
        else:
            tick_interval = 18
        
        start_tick = (min_time // tick_interval) * tick_interval
        end_tick = ((max_time // tick_interval) + 1) * tick_interval
        time_ticks = list(range(int(start_tick), int(end_tick) + 1, tick_interval))
        
        ax.set_xticks(time_ticks)
        ax.set_xlim(start_tick - tick_interval, end_tick + tick_interval)
    else:
        ax.set_xticks(range(0, time_horizon + 1, 30))
    
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)

    # 添加航线颜色图例
    num_routes = 3
    legend_elements = []
    for h in range(num_routes):
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, color=colors[h % len(colors)],
                              label=f'航线 {h} (高度层 {h})'))

    plt.subplots_adjust(right=0.85)
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.01, 1),
             title='高度层/航线说明')

    # 保存图形
    if save_path is None:
        save_path = f'picture_result/evtol_schedule_{algorithm_name.lower().replace(" ", "_")}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def extract_schedule_from_result(result: Dict, algorithm_name: str = "") -> List[Dict]:
    """
    从不同算法的结果中提取标准化的调度数据
    
    参数:
        result: 算法结果字典
        algorithm_name: 算法名称，用于调试
    
    返回:
        标准化的调度列表
    """
    if not result:
        return []
    
    # 如果已经是标准格式的schedule
    if "schedule" in result and isinstance(result["schedule"], list):
        return result["schedule"]
    
    # 对于其他格式，需要转换
    # 这里可以根据需要扩展转换逻辑
    print(f"警告: {algorithm_name} 的结果格式需要转换")
    return [] 
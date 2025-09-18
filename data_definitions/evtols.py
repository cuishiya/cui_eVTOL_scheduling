#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eVTOL调度系统eVTOL机队定义
定义系统中所有的eVTOL信息
"""

# eVTOL机队数据定义 - 基于example_task_chains.py
EVTOLS = [
    {"id": 0, "initial_position": 3, "initial_soc": 100, "initial_state": 0},
    {"id": 1, "initial_position": 1, "initial_soc": 100, "initial_state": 0},
    {"id": 2, "initial_position": 2, "initial_soc": 100, "initial_state": 0},
    # {"id": 3, "initial_position": 4, "initial_soc": 100, "initial_state": 0},
    # {"id": 4, "initial_position": 4, "initial_soc": 100, "initial_state": 0},
    # {"id": 5, "initial_position": 1, "initial_soc": 100, "initial_state": 0},
    # {"id": 6, "initial_position": 2, "initial_soc": 100, "initial_state": 0},
    # {"id": 7, "initial_position": 3, "initial_soc": 100, "initial_state": 0},
]

def get_evtols():
    """获取所有eVTOL列表"""
    return EVTOLS

def get_evtol_by_id(evtol_id):
    """根据eVTOL ID获取eVTOL信息"""
    for evtol in EVTOLS:
        if evtol["id"] == evtol_id:
            return evtol
    return None

def get_evtols_by_location(location_id):
    """根据初始位置获取eVTOL列表"""
    return [evtol for evtol in EVTOLS if evtol["initial_position"] == location_id]

def get_evtol_count():
    """获取eVTOL总数"""
    return len(EVTOLS) 
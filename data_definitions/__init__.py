#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eVTOL调度系统数据定义包
包含所有的系统数据定义：地点、任务、eVTOL机队等
"""

from .locations import get_locations, get_location_name, LOCATIONS
from .tasks import get_tasks, get_task_by_id, get_tasks_by_location, TASKS
from .evtols import get_evtols, get_evtol_by_id, get_evtols_by_location, get_evtol_count, EVTOLS

__all__ = [
    # 地点相关
    "get_locations", "get_location_name", "LOCATIONS",
    # 任务相关
    "get_tasks", "get_task_by_id", "get_tasks_by_location", "TASKS",
    # eVTOL相关
    "get_evtols", "get_evtol_by_id", "get_evtols_by_location", "get_evtol_count", "EVTOLS"
] 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eVTOL调度系统地点定义
定义系统中所有的地点信息
"""

# 定义地点
LOCATIONS = {
    1: "高铁站",
    2: "旅游区", 
    3: "居民区",
    4: "商业区"
}

def get_locations():
    """获取地点字典"""
    return LOCATIONS

def get_location_name(location_id):
    """根据地点ID获取地点名称"""
    return LOCATIONS.get(location_id, f"未知地点{location_id}") 
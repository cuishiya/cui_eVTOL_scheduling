#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
改进遗传算法多目标优化模块

集成改进:
1. 变邻域搜索(VNS)变异算子
2. 基于Q-learning的交叉与变异概率自适应调整

基于PyGMO框架的改进NSGA-II算法，求解eVTOL调度多目标优化问题
"""

from .evtol_scheduling_improved_nsga2 import (
    # 核心类
    eVTOLSchedulingProblem,
    QLearningGAController,
    VariableNeighborhoodSearch, 
    ImprovedNSGA2,
    
    # 主要求解函数
    solve_improved_nsga2,
    
    # 可视化函数
    visualize_improved_evolution_curves,
    visualize_improved_pareto_front,
    
    # 分析函数  
    analyze_algorithm_improvements,
    
    # 原有兼容函数
    solve_pygmo_nsga2,
    visualize_evolution_curves,
    visualize_pareto_front_evolution
)

__all__ = [
    # 核心类
    'eVTOLSchedulingProblem',
    'QLearningGAController', 
    'VariableNeighborhoodSearch',
    'ImprovedNSGA2',
    
    # 主要求解函数
    'solve_improved_nsga2',
    
    # 可视化函数
    'visualize_improved_evolution_curves',
    'visualize_improved_pareto_front',
    
    # 分析函数
    'analyze_algorithm_improvements',
    
    # 原有兼容函数
    'solve_pygmo_nsga2',
    'visualize_evolution_curves', 
    'visualize_pareto_front_evolution'
]

# 模块信息
__version__ = "1.0.0"
__author__ = "eVTOL Scheduling Team"
__description__ = "改进遗传算法多目标优化 - VNS变异 + Q-learning参数自适应"

def get_algorithm_info():
    """
    获取改进算法信息
    """
    return {
        "name": "改进NSGA-II",
        "version": __version__,
        "improvements": [
            "变邻域搜索(VNS)变异算子",
            "Q-learning参数自适应调整"
        ],
        "features": [
            "保持gurobi_multi相同数学模型",
            "真正多目标优化无权重组合", 
            "适用中大规模问题求解",
            "运算时间可接受解质量高"
        ],
        "components": {
            "QLearningGAController": "Q-learning参数自适应控制器",
            "VariableNeighborhoodSearch": "变邻域搜索变异算子",
            "ImprovedNSGA2": "改进NSGA-II主算法类"
        }
    }

def print_algorithm_info():
    """
    打印改进算法信息
    """
    info = get_algorithm_info()
    print("="*60)
    print(f"   {info['name']} v{info['version']}")
    print("="*60)
    print(f"📈 算法改进:")
    for improvement in info['improvements']:
        print(f"   • {improvement}")
    print(f"\n🎯 核心特性:")
    for feature in info['features']:
        print(f"   • {feature}")
    print(f"\n🔧 主要组件:")
    for component, description in info['components'].items():
        print(f"   • {component}: {description}")
    print("="*60)

if __name__ == "__main__":
    print_algorithm_info() 
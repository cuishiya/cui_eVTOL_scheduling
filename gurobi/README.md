# Gurobi (精确优化) 文件夹

本文件夹包含eVTOL调度问题的Gurobi求解器精确优化算法实现。

## 📁 文件结构

```
gurobi/
├── __init__.py                    # Python包初始化文件
├── evtol_scheduling_gurobi.py     # Gurobi核心算法实现
├── example_task_chains.py         # 使用示例和任务链演示
└── README.md                      # 本说明文件
```

## 🚀 快速开始

### 从项目根目录运行
```bash
# 推荐方式：使用根目录的主入口文件
python run_gurobi.py
```

### 直接在gurobi文件夹中运行
```bash
cd gurobi
python example_task_chains.py
```

## 🔧 核心组件

### 1. `evtol_scheduling_gurobi.py`
- `generate_task_chains()` 函数：基于位置连续性生成任务链
- `solve_evtol_scheduling_with_chains()` 函数：基于任务链的MILP求解
- `solve_evtol_scheduling_with_task_chains()` 函数：主求解函数
- 可视化函数：`visualize_schedule_gurobi()`, `visualize_schedule_table_gurobi()`

### 2. `example_task_chains.py`
- 完整的使用示例
- 任务链生成演示
- 调度结果分析

## 🎯 算法特性

- **精确求解**：基于混合整数线性规划(MILP)
- **任务链优化**：位置连续性保证
- **多约束处理**：时间窗、防撞、资源分配
- **加权目标**：能耗和延误的权衡优化

## 📊 输出结果

运行后将生成：
- `evtol_schedule_gurobi.png` - 调度甘特图
- `evtol_schedule_table_gurobi.png` - 详细调度表格

## 🔧 依赖要求

### 必需软件
- **Gurobi Optimizer 9.0+** (需要有效许可证)
- **Python 3.7+**

### Python包
```bash
pip install gurobipy numpy matplotlib pandas
```

## ⚙️ 参数配置

### 求解器参数
- `MIPGap`: 接受的最优性间隙 (默认0.2)
- `TimeLimit`: 求解时间限制 (默认1800秒)
- `MIPFocus`: 求解重点 (默认1 - 寻找可行解)

### 问题参数
- `time_horizon`: 调度时间范围 (默认720分钟)
- `max_chain_length`: 最大任务链长度 (默认10)
- `chain_interval_time`: 任务链间隔时间 (默认30分钟)

## 🎛️ 目标函数

使用加权求和法：
```
minimize: α × 标准化能耗 + β × 标准化延误
```

### 权重设置
- `α = 0.3` (能耗权重)
- `β = 0.7` (延误权重)

### 基准化处理
- 能耗基准：所有任务选择最低能耗航线的总和
- 延误基准：任务数量 × 40分钟

## 🔍 约束条件

1. **任务链分配唯一性**：每个任务链分配给唯一eVTOL
2. **航线选择唯一性**：每个任务选择唯一航线
3. **时间序列约束**：任务链内任务按顺序执行
4. **eVTOL冲突避免**：同一eVTOL不能同时执行多个任务
5. **高度层防撞**：同时使用相同航线的任务不能时间重叠
6. **时间窗约束**：任务不早于最早开始时间执行

## 📈 性能基准

| 规模 | 任务数 | eVTOL数 | 求解时间 | 内存使用 |
|------|--------|---------|----------|----------|
| 小型 | 10     | 3       | < 1分钟  | < 100MB  |
| 中型 | 25     | 5       | < 5分钟  | < 500MB  |
| 大型 | 50     | 10      | < 30分钟 | < 2GB    | 
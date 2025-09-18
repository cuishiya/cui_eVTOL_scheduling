# eVTOL 低空交通调度优化系统

[![Gurobi](https://img.shields.io/badge/Gurobi-9.0+-green.svg)](https://www.gurobi.com/)
3.0版本解决了位置连续性问题

本项目实现了一个基于混合整数线性规划(MILP)的eVTOL（电动垂直起降飞行器）低空交通调度优化系统。该系统为多架eVTOL分配多个运输任务，同时考虑防撞约束、低延误、能源节约等约束条件，生成最优的调度方案。

## 📁 项目结构

```
govy_eVTOL调度2.0/
├── evtol_scheduling_gurobi.py    # 核心优化模型实现
├── example.py                    # 示例使用脚本
├── README.md                     # 项目文档

```

### 核心文件说明

- **`evtol_scheduling_gurobi.py`**: 
  - `solve_evtol_scheduling_gurobi()`: 构建和求解MILP优化模型
  - `visualize_schedule_gurobi()`: 生成调度甘特图
  - `visualize_schedule_table_gurobi()`: 生成任务调度表

- **`example.py`**: 
  - 定义城市交通场景（高铁站、旅游区、居民区、商业区）
  - 配置任务数据和eVTOL机队信息
  - 执行优化求解并展示结果

## 🛠️ 安装与配置

### 环境要求

- Python 3.7+
- Gurobi 9.0+ (需要有效许可证)



## 🚀 快速开始

### 基本使用

```bash
# 运行示例
python example.py
```

### 自定义场景

```python
from evtol_scheduling_gurobi import solve_evtol_scheduling_gurobi

# 定义任务
tasks = [
    {
        "id": 0, 
        "from": 3,  # 起点：居民区
        "to": 1,    # 终点：高铁站
        "earliest_start": 0,  # 最早开始时间（分钟）
        "duration": [4, 7, 10],  # 3条航线的飞行时间
        "soc_consumption": [19, 32, 50]  # 3条航线的电量消耗
    },
    # 更多任务...
]

# 定义eVTOL机队
evtols = [
    {
        "id": 0, 
        "initial_position": 3,  # 初始位置
        "initial_soc": 100,     # 初始电量(%)
        "initial_state": 0      # 初始状态
    },
    # 更多eVTOL...
]


### 可视化输出

1. **甘特图** (`evtol_schedule_gurobi.png`): 显示每架eVTOL的任务执行时间线
2. **调度表** (`evtol_schedule_table_gurobi.png`): 表格形式展示详细调度信息

## ⚙️ 高级配置

### 求解器参数调优
```python
# 在evtol_scheduling_gurobi.py中调整
model.setParam('MIPGap', 0.1)      # 接受10%的次优解
model.setParam('TimeLimit', 1800)  # 30分钟时间限制
model.setParam('MIPFocus', 1)      # 重点寻找可行解
```

### 场景参数

- **时间窗口**: 调整 `time_horizon` 参数（默认720分钟）
- **航线数量**: 修改 `num_routes` 参数（默认3条）
- **安全间隔**: 调整相邻任务间隔时间（默认30分钟）



## 📈 性能基准（待测试）
如：

| 规模 | 任务数 | eVTOL数 | 求解时间 | 内存使用 |
|------|--------|---------|----------|----------|
| 小型 | 10     | 3       | < 1分钟  | < 100MB  |
| 中型 | 25     | 5       | < 5分钟  | < 500MB  |
| 大型 | 50     | 10      | < 30分钟 | < 2GB    |



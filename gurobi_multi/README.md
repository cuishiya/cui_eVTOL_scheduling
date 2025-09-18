# eVTOL调度epsilon约束优化 - Gurobi求解器

基于Gurobi商业求解器的eVTOL调度epsilon约束优化解决方案。

## 🎯 功能特点

- **精确优化**: 使用Gurobi商业求解器，保证数学最优性
- **epsilon约束方法**: 系统性地生成帕累托前沿
- **相同模型结构**: 与单目标版本使用相同的变量和约束
- **直接目标函数**: 优化使用原始目标值，无需基准化处理
- **帕累托前沿**: 自动生成和过滤真正的帕累托最优解
- **可视化分析**: 提供丰富的图表分析工具

## 📊 优化方法

### ε-约束方法 (Epsilon-Constraint)
- **原理**: 优化一个目标，将其他目标作为约束
- **优点**: 能找到真正的帕累托最优解，理论保证强
- **特点**: 系统性地变化约束值生成帕累托前沿
- **适用**: 对解质量要求高的场景

## 🚁 数学模型

### 决策变量
- `y[c,k,t]`: eVTOL k在时刻t开始执行任务串c
- `z[i,h]`: 任务i是否选择航线h
- `task_start[i]`: 任务i的开始时间
- `task_end[i]`: 任务i的结束时间
- `chain_start[c]`: 任务串c的开始时间

### 目标函数
1. **最小化总能耗**: `min Σ(soc_consumption[i,h] * z[i,h])`
2. **最小化总延误**: `min Σ(task_start[i] - earliest_start[i])`

### 约束条件
- 任务串分配唯一性约束
- 航线选择约束
- 时间窗约束
- 任务串内任务时间约束
- eVTOL资源约束
- 高度层防撞约束
- 任务串间隔约束

## 📁 文件结构

```
gurobi_multi/
├── __init__.py                        # 包初始化
├── evtol_scheduling_gurobi_multi.py   # 核心epsilon约束优化模块
├── example_gurobi_multi.py            # 使用示例
└── README.md                          # 说明文档
```

## 🚀 快速开始

### 基本使用

```python
from gurobi_multi import solve_evtol_scheduling_epsilon_constraint
from data_definitions import get_tasks, get_evtols

# 获取数据
tasks = get_tasks()
evtols = get_evtols()

# 执行epsilon约束优化
result = solve_evtol_scheduling_epsilon_constraint(
    tasks=tasks,
    evtols=evtols,
    num_points=20,
    verbose=True
)

# 分析结果
pareto_front = result["pareto_front"]
print(f"找到 {len(pareto_front)} 个帕累托最优解")
```

### 运行示例

```bash
cd gurobi_multi
python example_gurobi_multi.py
```

## 📈 结果分析

### 帕累托前沿
- 每个解都是非支配的
- 提供能耗与延误的权衡选择
- 自动过滤冗余解

### 代表性解决方案
- **最低能耗解**: 最小化总体能源消耗
- **最低延误解**: 最小化任务延误时间
- **中间解**: 在两个目标间的各种权衡

### 性能指标
- 解的数量和质量
- 求解时间分析
- 目标函数范围
- 约束值敏感性分析

## 🎨 可视化功能

### 帕累托前沿图
- 二维目标空间中的解分布
- 特殊解的标记和标注
- 前沿连线和趋势分析

### 收敛分析图
- 求解时间分布
- 目标函数分布
- 解的质量分析

## ⚙️ 参数配置

### 核心参数
- `num_points`: 帕累托前沿点数
- `time_horizon`: 调度时间窗
- `max_chain_length`: 最大任务链长度
- `verbose`: 详细输出控制

### 性能调优
- Gurobi求解器参数
- 时间限制设置
- 精度控制选项

## 🔍 方法特点

| 特性 | epsilon约束方法 |
|------|-----------------|
| 解质量 | 高 |
| 计算效率 | 中等 |
| 理论保证 | 强 |
| 实现复杂度 | 中等 |
| 帕累托前沿完整性 | 高 |
| 主观性 | 低 |

## 🛠️ 依赖要求

- Python 3.7+
- Gurobi Optimizer (需要许可证)
- NumPy
- Matplotlib
- Pandas

## 📝 使用注意

1. **Gurobi许可证**: 需要有效的Gurobi许可证
2. **计算资源**: epsilon约束方法可能需要较多计算时间
3. **参数调优**: 根据问题规模调整帕累托点数量
4. **结果验证**: 检查帕累托前沿的合理性
5. **约束范围**: epsilon约束方法需要合理的目标值范围

## 🔗 相关模块

- `gurobi/`: 单目标Gurobi优化
- `ga/`: NSGA-II多目标遗传算法
- `data_definitions/`: 统一数据定义 
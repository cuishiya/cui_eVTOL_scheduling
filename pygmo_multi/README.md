# PyGMO多目标优化 (改进版) 文件夹

本文件夹包含eVTOL调度问题的PyGMO多目标优化算法实现，**完全对应gurobi_multi数学模型**的所有决策变量和约束条件。

## 🚀 核心特性

- **数学模型一致性**: 与gurobi_multi实现完全相同的决策变量、约束条件、多目标函数
- **真正多目标优化**: 无权重组合，直接优化两个独立目标函数
- **实数编码方案**: 高效的遗传编码处理复杂组合优化问题
- **NSGA-II算法**: 经典多目标进化算法，寻找帕累托最优解集
- **约束处理**: 使用惩罚函数处理所有约束条件
- **多样性优化**: 优化的算法参数确保解的多样性

## 📁 文件结构

```
pygmo_multi/
├── __init__.py                           # 模块初始化
├── evtol_scheduling_pygmo_multi.py       # 核心优化算法实现
├── example_pygmo_multi.py                # 使用示例和详细说明
└── README.md                             # 本说明文件
```

## 🧬 遗传编码方案

### 决策变量对应关系

| gurobi_multi变量 | 含义 | PyGMO编码 | 说明 |
|------------------|------|-----------|------|
| `y[c,k,t]` | eVTOL k在时刻t开始执行任务串c | `y_evtol[c] + y_time[c]` | 每个任务串2个实数 |
| `z[i,h]` | 任务i使用航线h | `z_route[i]` | 每个任务1个实数 |
| `task_start[i]` | 任务i开始时间 | 从任务串时间推导+微调 | 计算得出 |
| `task_end[i]` | 任务i结束时间 | 从开始时间+持续时间计算 | 计算得出 |
| `chain_start[c]` | 任务串c开始时间 | 从y变量推导+微调 | 计算得出 |
| `chain_end[c]` | 任务串c结束时间 | 从最后任务结束时间计算 | 计算得出 |
| 其他辅助变量 | 约束处理变量 | 从主要变量推导计算 | 计算得出 |

### 编码结构

```
决策变量 = [y_variables | z_variables | task_offset_variables | chain_offset_variables]

总维度 = num_chains × 3 + num_tasks × 2
```

**详细说明:**
1. **y变量 (任务串分配)**: `num_chains × 2`维
   - `y_evtol[c]`: [0,1] → eVTOL ID [0, num_evtols-1]
   - `y_time[c]`: [0,1] → 开始时间 [0, time_horizon-1]

2. **z变量 (航线选择)**: `num_tasks`维
   - `z_route[i]`: [0,1] → 航线ID [0, num_routes-1]

3. **任务时间微调**: `num_tasks`维
   - `task_offset[i]`: [0,1] → 时间偏移 [0, 60分钟]

4. **任务串时间微调**: `num_chains`维
   - `chain_offset[c]`: [0,1] → 时间偏移 [0, 120分钟]

## 📋 约束条件对应

与gurobi_multi实现完全相同的约束条件，通过惩罚函数处理：

| 约束编号 | gurobi_multi约束 | PyGMO处理方式 | 惩罚权重 |
|----------|------------|---------------|----------|
| 2.1 | 任务串分配唯一性 | 检查每个串分配给唯一eVTOL和时刻 | 1000 |
| 2.2 | 航线选择唯一性 | 检查每个任务选择唯一航线 | 1000 |
| 2.3 | 任务串开始时间约束 | 检查串开始时间与y变量一致性 | 500 |
| 2.4 | 任务串内时间约束 | 检查串内任务时间顺序和间隔 | 800 |
| 2.5 | eVTOL冲突约束 | 检查同一eVTOL不同时执行多串 | 1200 |
| 2.6 | 高度层防撞约束 | 检查同航线任务时间不重叠 | 1500 |
| 2.7 | 任务串间隔约束 | 检查同eVTOL不同串间30分钟间隔 | 1000 |
| 2.8 | 时间窗约束 | 检查任务不早于最早开始时间 | 800 |

## 🎯 多目标函数

与gurobi_multi完全相同的两个独立目标函数（真正的多目标优化）：

```python
# 目标1: 总能耗 (与gurobi_multi epsilon约束方法相同)
objective1 = Σ(soc_consumption[i][h] * z[i,h])

# 目标2: 总延误 (与gurobi_multi epsilon约束方法相同)
objective2 = Σ(max(0, task_start[i] - earliest_start[i]))

# 重要: 无权重组合！这是真正的多目标优化
# NSGA-II直接优化这两个独立的目标函数
```

## 🔧 NSGA-II算法参数

优化的算法参数确保解的多样性：

```python
nsga2 = pg.nsga2(
    gen=200,           # 进化代数
    cr=0.9,            # 交叉概率
    eta_c=20,          # 交叉分布指数
    m=1.0/dimensions,  # 变异概率
    eta_m=20           # 变异分布指数
)
```

## 📊 使用示例

```python
from pygmo_multi.evtol_scheduling_pygmo_multi import solve_pygmo_nsga2
from data_definitions import get_tasks, get_evtols
from gurobi.evtol_scheduling_gurobi import generate_task_chains

# 加载数据
tasks = get_tasks()
evtols = get_evtols()
task_chains = generate_task_chains(tasks, max_chain_length=3)

# 求解
result = solve_pygmo_nsga2(
    tasks=tasks,
    evtols=evtols,
    task_chains=task_chains,
    time_horizon=720,
    population_size=100,
    generations=200,
    verbose=True
)

# 分析结果
pareto_front = result['pareto_front']
for i, sol in enumerate(pareto_front):
    print(f"解{i+1}: 能耗={sol['energy']:.1f}, 延误={sol['delay']:.1f}分钟")
```

## 🎛️ 算法优势

1. **模型一致性**: 与gurobi_multi数学模型100%对应，确保公平比较
2. **真正多目标**: NSGA-II自然处理多目标，无需人工设定权重
3. **处理复杂性**: 实数编码有效处理高维复杂约束优化问题
4. **解集丰富**: 一次运行获得多个帕累托最优解
5. **参数优化**: 精心调优的算法参数确保收敛性和多样性
6. **方法对应**: 与gurobi_multi的epsilon约束方法在概念上等价

## 📈 性能比较

| 算法 | 解的性质 | 计算时间 | 解的数量 | 适用场景 |
|------|----------|----------|----------|----------|
| Gurobi_multi (ε-约束) | 精确帕累托解 | 长 | 有限个点 | 小规模精确帕累托前沿 |
| PyGMO NSGA-II | 近似帕累托解 | 中等 | 多个帕累托解 | 大规模多目标优化 |

## 🔍 验证方法

1. **目标函数一致性**: 验证PyGMO和gurobi_multi的目标函数计算结果相同
2. **约束检查**: 验证PyGMO解满足gurobi_multi的所有约束条件
3. **帕累托前沿对比**: 比较NSGA-II与gurobi_multi epsilon-constraint方法的帕累托前沿
4. **收敛性分析**: 监控算法收敛过程和解的分布
5. **多目标性验证**: 确认两个目标函数确实存在冲突关系

## ⚠️ 注意事项

1. **种群大小**: 必须≥8且为4的倍数（NSGA-II要求）
2. **编码精度**: 实数映射到整数时可能有精度损失
3. **约束处理**: 惩罚函数可能影响解的质量，需要合理设置权重
4. **参数调优**: 不同问题规模可能需要调整算法参数

## 🚀 快速开始

```bash
# 运行示例
cd pygmo_multi
python example_pygmo_multi.py

# 查看详细的遗传编码说明和建模对应关系
# 程序会输出完整的编码方案和约束处理方式
```

## 📝 技术细节

### 解码过程
1. **y变量解码**: 实数 → eVTOL ID + 开始时间
2. **z变量解码**: 实数 → 航线选择
3. **时间计算**: 从任务串时间推导任务时间
4. **约束检查**: 逐一验证所有gurobi约束条件
5. **目标计算**: 使用相同的基准化公式

### 优化策略
1. **多样化初始化**: 生成具有不同特征的初始种群
2. **自适应惩罚**: 根据约束违反程度分层惩罚
3. **精英保留**: NSGA-II自动保留非支配解
4. **收敛监控**: 跟踪帕累托前沿的演化过程

这个实现确保了与gurobi_multi数学模型的完全一致性，为eVTOL调度问题提供了高效的多目标优化解决方案。真正的多目标优化避免了人工权重设定的主观性，能够发现更全面的帕累托最优解集。 
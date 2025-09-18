# 改进遗传算法多目标优化 文件夹

本文件夹包含改进的eVTOL调度问题多目标优化算法实现，基于标准NSGA-II算法集成两项重要改进。

## 🚀 算法改进亮点

### 🔧 改进1: 变邻域搜索(VNS)变异算子
- **传统变异**: 随机扰动个体基因
- **VNS变异**: 系统性地在多种邻域结构中搜索更优解
- **5种邻域结构**:
  1. 交换任务串的eVTOL分配
  2. 改变任务的航线选择  
  3. 微调任务串开始时间
  4. 交换同一eVTOL上任务串的执行顺序
  5. 单个任务串的局部优化

### 🧠 改进2: 基于Q-learning的参数自适应
- **传统方法**: 固定的交叉率和变异率参数
- **Q-learning控制**: 根据算法性能动态调整参数
- **状态特征**: 进化阶段、超体积趋势、改善率、多样性
- **动作空间**: 9种交叉率×变异率组合
- **自适应奖励**: 基于超体积、改善率、多样性的综合奖励

## 📁 文件结构

```
improved_ga_multi/
├── __init__.py                           # 模块初始化及信息函数
├── evtol_scheduling_improved_nsga2.py    # 改进NSGA-II算法核心实现
├── example_improved_nsga2.py             # 改进算法使用示例
└── README.md                             # 本说明文件
```

## 🏗️ 核心组件架构

### 1. QLearningGAController 类
**Q-learning参数自适应控制器**

```python
class QLearningGAController:
    - 状态编码: (进化阶段, 超体积趋势, 改善率, 多样性)
    - 动作空间: 3×3交叉率变异率组合
    - 探索策略: epsilon-greedy
    - 奖励函数: 超体积+改善率+多样性的加权和
```

**关键参数:**
- `learning_rate=0.1`: Q值学习率
- `epsilon=0.3`: 探索率 (逐步衰减)
- `discount_factor=0.95`: 折扣因子

### 2. VariableNeighborhoodSearch 类
**变邻域搜索变异算子**

```python
class VariableNeighborhoodSearch:
    - 邻域1: 交换eVTOL分配 (neighborhood_1_swap_evtol)
    - 邻域2: 改变航线选择 (neighborhood_2_swap_route)  
    - 邻域3: 微调开始时间 (neighborhood_3_adjust_time)
    - 邻域4: 交换执行顺序 (neighborhood_4_swap_chain_order)
    - 邻域5: 局部优化 (neighborhood_5_local_optimization)
```

**VNS算法流程:**
1. 从第一个邻域结构开始
2. 在当前邻域中生成邻居解
3. 如果邻居解更好，接受并回到第一个邻域
4. 否则，转到下一个邻域结构
5. 重复直到所有邻域都尝试过

### 3. ImprovedNSGA2 类
**改进的NSGA-II主算法**

```python
class ImprovedNSGA2:
    - 集成Q-learning控制器
    - 集成VNS变异算子
    - 动态参数调整
    - 完整的进化过程管理
```

## 🎯 算法特性对比

| 特性 | 标准NSGA-II | 改进NSGA-II |
|------|-------------|-------------|
| 变异策略 | 随机变异 | VNS多邻域搜索 |
| 参数控制 | 固定参数 | Q-learning自适应 |
| 局部搜索 | 无 | 5种邻域结构 |
| 收敛性 | 标准 | 改进的收敛速度 |
| 解质量 | 标准 | 更高质量解 |
| 适用规模 | 中等 | 中大规模 |

## 🚀 快速开始

### 基本使用

```python
from improved_ga_multi import solve_improved_nsga2
from data_definitions import get_tasks, get_evtols, get_locations
from gurobi.evtol_scheduling_gurobi import generate_task_chains

# 加载数据
tasks = get_tasks()
evtols = get_evtols()
locations = get_locations()

# 生成任务串
task_chains = generate_task_chains(tasks, locations, max_chain_length=5)

# 使用改进NSGA-II求解
result = solve_improved_nsga2(
    tasks=tasks,
    evtols=evtols,
    task_chains=task_chains,
    population_size=100,
    generations=200,
    verbose=True
)

# 分析结果
pareto_front = result['pareto_front']
print(f"找到 {len(pareto_front)} 个帕累托最优解")
```

### 运行示例

```bash
cd improved_ga_multi
python example_improved_nsga2.py
```

## 📊 可视化功能

### 1. 改进算法进化曲线
```python
from improved_ga_multi import visualize_improved_evolution_curves

visualize_improved_evolution_curves(
    result['evolution_data'],
    "picture_result/evolution_curves_improved_nsga2.png"
)
```

**包含6个子图:**
- 帕累托前沿解数量变化
- 能耗目标进化曲线
- 延误目标进化曲线  
- 超体积指标变化
- **Q-learning交叉率自适应过程**
- **Q-learning变异率自适应过程**

### 2. 改进帕累托前沿图
```python
from improved_ga_multi import visualize_improved_pareto_front

visualize_improved_pareto_front(
    result['pareto_front'],
    "picture_result/pareto_front_improved_nsga2.png"
)
```

### 3. 算法对比分析
```python
from improved_ga_multi import analyze_algorithm_improvements

# 对比标准NSGA-II和改进NSGA-II
analyze_algorithm_improvements(standard_result, improved_result)
```

## ⚙️ 算法参数配置

### 核心参数
- `population_size`: 种群大小 (推荐100)
- `generations`: 进化代数 (推荐200)
- `verbose`: 详细输出控制

### Q-learning参数
- `learning_rate`: 学习率 (默认0.1)
- `epsilon`: 探索率 (默认0.3，自动衰减)
- `discount_factor`: 折扣因子 (默认0.95)

### VNS参数
- `mutation_rate`: VNS应用比例 (默认0.3)
- `max_iterations`: VNS最大迭代次数 (默认10)

## 🔬 算法原理详解

### VNS变异原理
传统遗传变异算子通过随机扰动基因来产生新解，但这种方法缺乏针对性。VNS变异算子通过系统地搜索多种邻域结构，能够：

1. **提高局部搜索能力**: 在解的邻域内系统性搜索
2. **避免局部最优**: 通过多种邻域结构跳出局部最优
3. **保持解的可行性**: 邻域操作保证解的结构完整性

### Q-learning参数控制原理
传统遗传算法使用固定的交叉率和变异率，无法适应不同的搜索阶段。Q-learning控制器通过：

1. **状态感知**: 识别当前算法的进化状态
2. **动作选择**: 根据Q表选择最优的参数组合
3. **奖励学习**: 根据性能改善更新Q值
4. **探索平衡**: epsilon-greedy策略平衡探索与利用

## 📈 性能优势分析

### 1. 收敛性能
- **更快收敛**: VNS变异提供更好的局部搜索
- **稳定性**: Q-learning避免参数设置不当

### 2. 解质量
- **更高多样性**: 自适应参数保持种群多样性
- **更优前沿**: VNS帮助发现更优的帕累托解

### 3. 适用性
- **大规模问题**: 改进算法对大规模问题表现更好
- **鲁棒性**: 减少对初始参数设置的依赖

## 🔧 数学模型一致性

**重要说明**: 改进算法完全保持与gurobi_multi相同的数学模型:
- **相同决策变量**: y, z, task_start, task_end等
- **相同约束条件**: 所有2.1-2.8约束条件
- **相同目标函数**: 能耗最小化 + 延误最小化
- **相同编码方案**: 保持原有编码结构

改进**仅在算法层面**，不改变问题建模！

## 🛠️ 依赖要求

### 必需软件
- **Python 3.7+**
- **PyGMO 2.15+**

### Python包
```bash
pip install pygmo numpy matplotlib pandas
```

### 可选依赖
```bash
pip install seaborn  # 增强可视化效果
```

## 📚 使用建议

### 1. 参数调优建议
- **小规模问题**: population_size=50, generations=100
- **中等规模**: population_size=100, generations=200  
- **大规模问题**: population_size=150, generations=300

### 2. 性能监控
- 观察Q-learning参数变化趋势
- 监控超体积指标改善情况
- 检查帕累托前沿解的分布

### 3. 对比分析
- 建议与标准NSGA-II对比验证改进效果
- 使用多次独立运行评估算法稳定性
- 分析不同问题规模下的性能表现

## 🔗 相关模块

- `gurobi/`: 基准数学模型定义
- `gurobi_multi/`: Epsilon约束精确优化
- `pygmo_multi/`: 标准NSGA-II实现
- `data_definitions/`: 统一数据定义
- `visualization/`: 结果可视化工具

## 📄 算法引用

改进算法基于以下理论:
1. **VNS**: Variable Neighborhood Search (Mladenović & Hansen, 1997)
2. **Q-learning**: Q-learning for parameter control (Watkins & Dayan, 1992)  
3. **NSGA-II**: Non-dominated Sorting GA (Deb et al., 2002)

---

**🎯 总结**: 本模块通过集成VNS变异和Q-learning参数自适应，显著提升了标准NSGA-II在eVTOL调度问题上的求解性能，特别适用于中大规模的多目标优化问题。 
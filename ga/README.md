# GA (遗传算法) 文件夹

本文件夹包含eVTOL调度问题的NSGA-II多目标优化算法实现。

## 📁 文件结构

```
ga/
├── __init__.py                 # Python包初始化文件
├── evtol_nsga2.py             # NSGA-II核心算法实现
├── example_nsga2.py           # 使用示例和算法比较
├── run_nsga2_demo.py          # 快速演示脚本
├── requirements_nsga2.txt     # Python依赖包列表
└── README.md                  # 本说明文件
```

## 🚀 快速开始

### 从项目根目录运行
```bash
# 推荐方式：使用根目录的主入口文件
python run_nsga2.py
```

### 直接在ga文件夹中运行
```bash
cd ga
python -m example_nsga2
```

## 🔧 核心组件

### 1. `evtol_nsga2.py`
- `Individual` 类：遗传算法个体表示
- `eVTOL_NSGA2` 类：NSGA-II算法主体
- `solve_evtol_nsga2()` 函数：主求解函数
- 可视化函数：`visualize_pareto_front()`, `visualize_evolution_history()`

### 2. `example_nsga2.py`
- 完整的使用示例
- 算法性能分析
- 与传统方法的比较

### 3. `run_nsga2_demo.py`
- 快速演示脚本
- 简化的运行流程

## 🎯 算法特性

- **多目标优化**：同时优化能耗和延误
- **帕累托前沿**：获得多个非支配解
- **约束处理**：时间冲突、航线冲突等
- **可视化输出**：帕累托前沿图、进化历史图

## 📊 输出结果

运行后将生成：
- `evtol_pareto_front_nsga2.png` - 帕累托前沿散点图
- `evtol_evolution_history_nsga2.png` - 进化历史曲线图

## 🔧 依赖安装

```bash
pip install -r requirements_nsga2.txt
```

## 📈 参数调优

### 种群参数
- `population_size`: 种群大小 (默认100)
- `generations`: 进化代数 (默认200)

### 遗传参数  
- `crossover_prob`: 交叉概率 (默认0.9)
- `mutation_prob`: 变异概率 (默认0.1)

### 建议配置
- **快速测试**: 种群30, 代数50
- **标准配置**: 种群50, 代数100  
- **高精度**: 种群100, 代数200 
---
name: model-solver
description: 执行数学模型求解。集成多种求解器（SciPy、PuLP、CVXPY、Gurobi等），支持自动求解器选择和失败时的备选方案切换。
---

# 模型求解器 (Model Solver)

## 功能概述

执行数学模型的数值求解，支持多种求解器和自动故障切换。

## 支持的求解器

### 优化类

| 求解器 | 问题类型 | 许可证 | 特点 |
|--------|---------|--------|------|
| SciPy | LP, NLP | MIT | 通用、内置 |
| PuLP | LP, MILP | BSD | 易用、开源 |
| CVXPY | 凸优化 | Apache | 声明式建模 |
| Gurobi | LP, MILP, QP | 商业 | 高性能 |
| OR-Tools | CP, VRP | Apache | Google出品 |

### 机器学习类

| 求解器 | 问题类型 | 特点 |
|--------|---------|------|
| sklearn | 分类、回归、聚类 | 全面易用 |
| statsmodels | 统计模型 | 丰富的统计检验 |
| Prophet | 时间序列 | 自动化预测 |
| darts | 时间序列 | 多模型支持 |

### 微分方程类

| 求解器 | 问题类型 | 特点 |
|--------|---------|------|
| scipy.integrate | ODE | 多种方法 |
| FEniCS | PDE | 有限元 |
| PyDSTool | 动力系统 | 分岔分析 |

## 求解策略

### 自动求解器选择

```python
SOLVER_SELECTION = {
    'LP': ['scipy.linprog', 'pulp.CBC'],
    'MILP': ['pulp.CBC', 'gurobi'],
    'NLP': ['scipy.minimize', 'ipopt'],
    'Convex': ['cvxpy', 'scipy'],
    'TimeSeries': ['prophet', 'statsmodels'],
}
```

### 故障切换

```python
def solve_with_fallback(problem, primary_solver, fallback_solvers):
    try:
        return primary_solver.solve(problem)
    except SolverError:
        for solver in fallback_solvers:
            try:
                return solver.solve(problem)
            except:
                continue
        raise AllSolversFailedError()
```

## 输出格式

```json
{
  "solver_used": "PuLP-CBC",
  "solve_time": 2.34,
  "status": "optimal",
  "objective_value": 1234.56,
  "solution": {
    "x_1": 10.5,
    "x_2": 20.3,
    "y_1": 1,
    "y_2": 0
  },
  "solver_stats": {
    "iterations": 150,
    "gap": 0.0001,
    "nodes_explored": 45
  },
  "verification": {
    "constraints_satisfied": true,
    "bounds_satisfied": true
  }
}
```

## 使用示例

### 线性规划

```python
from scripts.solver_dispatcher import solve_optimization

result = solve_optimization(
    problem_type='LP',
    objective='minimize',
    c=[1, 2, 3],
    A_ub=[[1, 1, 1]],
    b_ub=[10]
)
```

### 时间序列预测

```python
from scripts.ml_solver import forecast_time_series

result = forecast_time_series(
    data=df,
    target_column='value',
    forecast_periods=30,
    model='prophet'
)
```

## 错误处理

| 错误类型 | 处理方式 |
|---------|---------|
| 问题无解 | 返回诊断信息 + 尝试松弛约束 |
| 数值问题 | 调整精度 + 缩放变量 |
| 超时 | 返回当前最优解 + 增加时间限制 |
| 内存不足 | 分解问题 + 降低精度 |

## 相关技能

- `model-builder` - 模型构建
- `code-verifier` - 代码验证
- `sensitivity-analyzer` - 敏感性分析

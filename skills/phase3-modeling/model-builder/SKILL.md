---
name: model-builder
description: 构建完整的数学模型。将目标函数、约束条件、变量定义整合为完整的数学规划模型，并自动生成可执行的求解代码。基于OR-LLM-Agent技术实现自然语言到代码的转换。
---

# 模型构建器 (Model Builder)

## 功能概述

将分析结果整合为完整的数学模型，并生成可执行的求解代码。

## 模型结构

### 完整数学模型格式

```latex
\textbf{Objective Function:}
\begin{equation}
    \min_{x} f(x) = \sum_{i \in I} c_i x_i
\end{equation}

\textbf{Subject to:}
\begin{align}
    \sum_{j \in J} a_{ij} x_j &\leq b_i & \forall i \in I & \text{(Capacity)} \\
    x_j &\geq 0 & \forall j \in J & \text{(Non-negativity)}
\end{align}

\textbf{Where:}
\begin{itemize}
    \item $x_j$: decision variable
    \item $c_i$: cost coefficient
    \item $a_{ij}$: constraint coefficient
\end{itemize}
```

## 代码生成

### 支持的框架

| 框架 | 适用问题 | 生成语言 |
|-----|---------|---------|
| SciPy | 一般优化 | Python |
| PuLP | LP/MILP | Python |
| CVXPY | 凸优化 | Python |
| Gurobi | 大规模优化 | Python |
| OR-Tools | 组合优化 | Python |

### 代码模板

#### 线性规划 (PuLP)

```python
from pulp import *

# Create problem
prob = LpProblem("MCM_Problem", LpMinimize)

# Decision variables
x = LpVariable.dicts("x", range(n), lowBound=0)

# Objective function
prob += lpSum([c[i] * x[i] for i in range(n)])

# Constraints
for i in range(m):
    prob += lpSum([a[i][j] * x[j] for j in range(n)]) <= b[i]

# Solve
prob.solve()

# Results
for v in prob.variables():
    print(f"{v.name} = {v.varValue}")
print(f"Optimal value = {value(prob.objective)}")
```

#### 非线性优化 (SciPy)

```python
from scipy.optimize import minimize

def objective(x):
    return x[0]**2 + x[1]**2

def constraint1(x):
    return x[0] + x[1] - 1  # >= 0

constraints = [{'type': 'ineq', 'fun': constraint1}]
bounds = [(0, None), (0, None)]

result = minimize(objective, x0=[0.5, 0.5], 
                  method='SLSQP', 
                  bounds=bounds, 
                  constraints=constraints)

print(f"Optimal x: {result.x}")
print(f"Optimal value: {result.fun}")
```

#### 时间序列 (Prophet)

```python
from prophet import Prophet
import pandas as pd

# Prepare data
df = pd.DataFrame({
    'ds': dates,
    'y': values
})

# Create and fit model
model = Prophet(
    changepoint_prior_scale=0.05,
    seasonality_mode='multiplicative'
)
model.fit(df)

# Make predictions
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Visualize
fig = model.plot(forecast)
```

## 输出格式

```json
{
  "model": {
    "name": "Resource Allocation Model",
    "type": "MILP",
    "objective": {
      "direction": "minimize",
      "expression": "\\sum_{i} c_i x_i",
      "description": "Total cost"
    },
    "variables": [
      {
        "name": "x_i",
        "type": "continuous",
        "domain": "[0, inf)",
        "count": "n"
      }
    ],
    "constraints": [
      {
        "name": "capacity",
        "expression": "\\sum_j a_{ij} x_j \\leq b_i",
        "count": "m"
      }
    ],
    "parameters": {
      "c": "cost coefficients",
      "a": "constraint matrix",
      "b": "right-hand side"
    }
  },
  "code": {
    "framework": "PuLP",
    "language": "Python",
    "file": "model_solver.py",
    "dependencies": ["pulp", "numpy", "pandas"]
  },
  "innovation": [
    "引入时间窗口约束",
    "考虑不确定性的鲁棒优化"
  ]
}
```

## 构建流程

1. **输入整合**: 收集变量、约束、目标
2. **模型形式化**: 转化为标准数学形式
3. **代码生成**: 生成可执行代码
4. **验证测试**: 用简单案例验证
5. **文档生成**: 生成LaTeX描述

## 相关技能

- `variable-definer` - 变量定义
- `constraint-identifier` - 约束识别
- `model-solver` - 模型求解
- `code-verifier` - 代码验证

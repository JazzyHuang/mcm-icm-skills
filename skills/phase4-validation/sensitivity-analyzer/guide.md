---
name: sensitivity-analyzer
description: 执行多层次敏感性分析。实现局部敏感性(OAT)、全局敏感性(Sobol指数)、Morris筛选方法和蒙特卡洛不确定性传播，全面评估参数对模型输出的影响。
---

# 敏感性分析器 (Sensitivity Analyzer)

## 功能概述

对数学模型进行全面的敏感性分析，评估参数变化对模型输出的影响程度。

## 分析方法

### 1. 局部敏感性分析 (OAT)
One-At-a-Time方法，逐个改变参数观察影响

```python
def oat_analysis(model, params, base_values, perturbation=0.1):
    sensitivities = {}
    base_output = model(**base_values)
    
    for param in params:
        perturbed = base_values.copy()
        perturbed[param] *= (1 + perturbation)
        new_output = model(**perturbed)
        sensitivities[param] = (new_output - base_output) / (perturbation * base_values[param])
    
    return sensitivities
```

### 2. 全局敏感性分析 (Sobol)
基于方差分解的全局敏感性指数

```python
from SALib.sample import saltelli
from SALib.analyze import sobol

problem = {
    'num_vars': 3,
    'names': ['x1', 'x2', 'x3'],
    'bounds': [[0, 1], [0, 1], [0, 1]]
}

# 生成样本
samples = saltelli.sample(problem, 2048, calc_second_order=True)

# 计算模型输出
Y = np.array([model(s) for s in samples])

# Sobol分析
Si = sobol.analyze(problem, Y, calc_second_order=True)

# Si['S1'] - 一阶敏感性指数
# Si['ST'] - 总效应敏感性指数
# Si['S2'] - 二阶交互效应
```

### 3. Morris方法
筛选重要参数的高效方法

```python
from SALib.sample import morris
from SALib.analyze import morris as morris_analyze

samples = morris.sample(problem, N=100, num_levels=4)
Y = np.array([model(s) for s in samples])
Si = morris_analyze.analyze(problem, samples, Y)

# Si['mu'] - 均值效应
# Si['mu_star'] - 绝对均值效应
# Si['sigma'] - 标准差(衡量交互效应)
```

### 4. 蒙特卡洛模拟
不确定性传播分析

```python
def monte_carlo_analysis(model, param_distributions, n_samples=10000):
    samples = []
    for _ in range(n_samples):
        params = {name: dist.rvs() for name, dist in param_distributions.items()}
        samples.append(model(**params))
    
    return {
        'mean': np.mean(samples),
        'std': np.std(samples),
        'percentiles': np.percentile(samples, [5, 25, 50, 75, 95])
    }
```

## 输出格式

```json
{
  "sensitivity_analysis": {
    "method": "sobol",
    "parameters": [
      {
        "name": "alpha",
        "S1": 0.45,
        "ST": 0.52,
        "rank": 1,
        "interpretation": "最重要参数，单独贡献45%方差"
      },
      {
        "name": "beta",
        "S1": 0.30,
        "ST": 0.35,
        "rank": 2,
        "interpretation": "第二重要参数"
      },
      {
        "name": "gamma",
        "S1": 0.15,
        "ST": 0.22,
        "rank": 3,
        "interpretation": "与其他参数有交互效应"
      }
    ],
    "interactions": [
      {
        "parameters": ["alpha", "beta"],
        "S2": 0.08,
        "interpretation": "存在中等程度交互"
      }
    ],
    "conclusions": [
      "alpha是最敏感参数，需要精确估计",
      "gamma与其他参数存在交互，不能孤立分析",
      "模型对参数变化总体稳健"
    ]
  },
  "visualization": {
    "tornado_chart": "figures/tornado.pdf",
    "sobol_bar": "figures/sobol_indices.pdf",
    "interaction_heatmap": "figures/interaction.pdf"
  }
}
```

## O奖标准

- 必须包含全局敏感性分析（不仅是OAT）
- 需要解释敏感性结果的物理/经济含义
- 讨论参数不确定性如何影响结论
- 提供参数排序和重要性判断

## 相关技能

- `model-validator` - 模型验证
- `uncertainty-quantifier` - 不确定性量化

---
name: causal-inference
description: 因果推断模块，支持Double Machine Learning、工具变量、双重差分等方法。适用于E题可持续性政策评估和F题政策分析。创新性评分0.90。
---

# 因果推断模块 (Causal Inference)

## 功能概述

提供因果推断方法进行政策效果评估：
1. Double Machine Learning (DML) - 去偏估计
2. Instrumental Variables (IV) - 处理内生性
3. Difference-in-Differences (DiD) - 政策评估
4. Causal Forest - 异质性效应

## 创新性评分：0.90/1.0

## 适用场景

| 方法 | 应用场景 | 假设要求 | MCM题型 |
|-----|---------|---------|---------|
| DML | 处理效应估计 | 无混杂 | E, F |
| IV | 内生性问题 | 有效工具 | E, F |
| DiD | 政策评估 | 平行趋势 | E, F |
| Causal Forest | 异质效应 | 无混杂 | E, F |

## 使用方法

### Double Machine Learning

```python
from causal_inference import DoubleMachineLearning

dml = DoubleMachineLearning(
    Y=outcome,           # 结果变量
    T=treatment,         # 处理变量  
    X=confounders,       # 混杂变量
    model_y='lightgbm',  # 结果模型
    model_t='lightgbm'   # 倾向得分模型
)

# 估计处理效应
ate = dml.estimate_ate()
print(f"Average Treatment Effect: {ate['effect']:.4f} ± {ate['std_error']:.4f}")

# 置信区间
ci = dml.confidence_interval(alpha=0.05)
```

### Difference-in-Differences

```python
from causal_inference import DifferenceInDifferences

did = DifferenceInDifferences(
    data=df,
    outcome='Y',
    treatment='treated',
    time='post',
    entity='id'
)

# 估计效应
effect = did.estimate()
did.plot_parallel_trends(save_path='figures/parallel_trends.pdf')
```

### Causal Forest

```python
from causal_inference import CausalForestEstimator

cf = CausalForestEstimator(
    n_estimators=100,
    min_samples_leaf=5
)

# 拟合
cf.fit(X, T, Y)

# 异质性处理效应
cate = cf.estimate_cate(X_test)
cf.plot_heterogeneity(save_path='figures/cate.pdf')
```

## 输出格式

```json
{
  "method": "DoubleMachineLearning",
  "estimate": {
    "ate": 0.156,
    "std_error": 0.023,
    "ci_lower": 0.111,
    "ci_upper": 0.201,
    "p_value": 0.001
  },
  "diagnostics": {
    "cross_fit_folds": 5,
    "outcome_r2": 0.85,
    "propensity_auc": 0.78
  },
  "figures": ["treatment_effect.pdf", "heterogeneity.pdf"]
}
```

## O奖加分建议

- 检验因果假设（如平行趋势）
- 进行敏感性分析（如Rosenbaum bounds）
- 展示异质性处理效应
- 与传统回归方法对比

## 相关技能

- `model-explainer` - 模型解释
- `sensitivity-analyzer` - 敏感性分析
- `chart-generator` - 结果可视化

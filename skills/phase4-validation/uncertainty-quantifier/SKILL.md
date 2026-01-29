---
name: uncertainty-quantifier
description: 量化模型预测的不确定性。实现置信区间估计、概率分布预测、多项式混沌展开等方法。
---

# 不确定性量化器 (Uncertainty Quantifier)

## 功能概述

量化模型预测的不确定性，提供置信区间和概率分布。

## 方法体系

### 1. 参数不确定性
- Bootstrap方法
- 贝叶斯推断
- 蒙特卡洛采样

### 2. 模型不确定性
- 模型集成
- 贝叶斯模型平均
- 预测区间

### 3. 输入不确定性
- 误差传播
- 多项式混沌展开(PCE)
- 随机配点法

## 代码示例

```python
import numpy as np
from scipy import stats

def bootstrap_confidence_interval(
    model, X, y, n_bootstrap=1000, confidence=0.95
):
    """Bootstrap置信区间"""
    predictions = []
    n = len(X)
    
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        X_boot, y_boot = X[idx], y[idx]
        model.fit(X_boot, y_boot)
        predictions.append(model.predict(X))
        
    predictions = np.array(predictions)
    alpha = 1 - confidence
    
    return {
        'lower': np.percentile(predictions, alpha/2 * 100, axis=0),
        'upper': np.percentile(predictions, (1-alpha/2) * 100, axis=0),
        'mean': np.mean(predictions, axis=0)
    }
```

## 输出格式

```json
{
  "uncertainty_analysis": {
    "method": "bootstrap",
    "confidence_level": 0.95,
    "results": {
      "point_estimate": 1234.56,
      "confidence_interval": {
        "lower": 1180.23,
        "upper": 1288.89
      },
      "coefficient_of_variation": 0.045
    },
    "interpretation": "95%置信度下，预测值在1180.23到1288.89之间"
  }
}
```

## 相关技能

- `sensitivity-analyzer` - 敏感性分析
- `model-validator` - 模型验证

---
name: model-validator
description: 验证和评估数学模型的性能。实现交叉验证、留出法验证、时间序列滚动验证，计算各类评估指标，确保模型可靠性。
---

# 模型验证器 (Model Validator)

## 功能概述

对数学模型进行全面的验证和评估，确保模型的可靠性和泛化能力。

## 验证方法

### 1. K折交叉验证
```python
from sklearn.model_selection import cross_val_score, KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
```

### 2. 留出法验证
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### 3. 时间序列滚动验证
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    # 训练和测试
```

### 4. 历史数据回测
对已知历史数据进行模型预测，比较预测与实际

## 评估指标

### 回归问题
| 指标 | 公式 | 解释 |
|-----|------|-----|
| RMSE | $\sqrt{\frac{1}{n}\sum(y-\hat{y})^2}$ | 均方根误差 |
| MAE | $\frac{1}{n}\sum|y-\hat{y}|$ | 平均绝对误差 |
| R² | $1-\frac{SS_{res}}{SS_{tot}}$ | 决定系数 |
| MAPE | $\frac{100}{n}\sum|\frac{y-\hat{y}}{y}|$ | 平均绝对百分比误差 |

### 分类问题
| 指标 | 说明 |
|-----|------|
| Accuracy | 准确率 |
| Precision | 精确率 |
| Recall | 召回率 |
| F1-Score | F1分数 |
| AUC-ROC | ROC曲线下面积 |

### 优化问题
| 指标 | 说明 |
|-----|------|
| Gap | 最优解与当前解的差距 |
| Feasibility | 解的可行性 |
| Runtime | 求解时间 |

## 输出格式

```json
{
  "validation_results": {
    "method": "5-fold cross-validation",
    "metrics": {
      "rmse": {
        "mean": 0.152,
        "std": 0.023,
        "values": [0.148, 0.165, 0.142, 0.155, 0.150]
      },
      "r2": {
        "mean": 0.923,
        "std": 0.015,
        "values": [0.928, 0.910, 0.935, 0.918, 0.924]
      }
    },
    "overfitting_check": {
      "train_score": 0.945,
      "test_score": 0.923,
      "gap": 0.022,
      "status": "acceptable"
    },
    "conclusions": [
      "模型R²达到0.923，解释力强",
      "训练-测试差距为2.2%，无明显过拟合",
      "模型稳定性好(std=0.015)"
    ]
  }
}
```

## O奖标准

- 必须有模型验证环节
- 使用适当的交叉验证方法
- 报告多个评估指标
- 讨论模型的局限性

## 相关技能

- `sensitivity-analyzer` - 敏感性分析
- `strengths-weaknesses` - 优缺点分析

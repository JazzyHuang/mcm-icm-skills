---
name: uncertainty-quantifier
description: 不确定性量化模块，支持保形预测(Conformal Prediction)和MC Dropout。提供分布无关的置信区间，增强模型预测可信度。O奖加分项。创新性评分0.92。
---

# 不确定性量化 (Uncertainty Quantification)

## 功能概述

提供多种不确定性量化方法：
1. **保形预测 (Conformal Prediction)** - 分布无关的置信区间
2. **MC Dropout** - 简单的贝叶斯近似
3. **集成方法 (Ensemble)** - 多模型不确定性

## 创新性评分：0.92/1.0

## 方法对比

| 方法 | 理论保证 | 计算成本 | 适用模型 |
|-----|---------|---------|---------|
| Conformal | 有(覆盖率保证) | 低 | 任意 |
| MC Dropout | 近似 | 中 | 神经网络 |
| Ensemble | 无 | 高 | 任意 |
| BNN | 有 | 很高 | 专用架构 |

## 使用方法

### 保形预测 (推荐)

```python
from uncertainty_quantifier import ConformalPredictor

# 创建保形预测器
cp = ConformalPredictor(
    model=trained_model,
    alpha=0.1  # 90%覆盖率
)

# 校准
cp.calibrate(X_cal, y_cal)

# 预测区间
y_pred, (lower, upper) = cp.predict_interval(X_test)

# 可视化
cp.plot_prediction_intervals(X_test, y_test, save_path='figures/conformal.pdf')
```

### MC Dropout

```python
from uncertainty_quantifier import MCDropout

# 包装现有模型
mc = MCDropout(model, n_samples=100, dropout_rate=0.1)

# 预测
mean, std = mc.predict_with_uncertainty(X_test)

# 置信区间
lower, upper = mc.confidence_interval(X_test, confidence=0.95)
```

### 集成不确定性

```python
from uncertainty_quantifier import EnsembleUncertainty

# 创建集成
ensemble = EnsembleUncertainty(
    base_model_class=RandomForest,
    n_models=10
)

# 训练
ensemble.fit(X_train, y_train)

# 预测
mean, std = ensemble.predict_with_uncertainty(X_test)
```

## 输出格式

```json
{
  "method": "ConformalPrediction",
  "coverage_target": 0.90,
  "actual_coverage": 0.91,
  "mean_interval_width": 0.45,
  "calibration_samples": 500,
  "figures": ["prediction_intervals.pdf", "coverage_analysis.pdf"]
}
```

## O奖加分建议

- 展示预测区间而非仅点预测
- 验证覆盖率（实际覆盖率 ≈ 目标覆盖率）
- 对比不同置信水平的区间宽度
- 强调"分布无关保证"的理论优势

## 相关技能

- `model-explainer` - 模型解释
- `sensitivity-analyzer` - 敏感性分析
- `chart-generator` - 可视化

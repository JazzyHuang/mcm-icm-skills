---
name: transformer-forecasting
description: Transformer时间序列预测模块，基于Temporal Fusion Transformer (TFT)实现多变量多步预测。支持可解释的注意力机制，适用于C题数据分析问题。创新性评分0.9。
---

# Transformer 时间序列预测

## 功能概述

基于Temporal Fusion Transformer实现高精度时间序列预测，支持：
1. 多变量输入（已知未来、已知历史、未知历史）
2. 多步预测（Multi-horizon forecasting）
3. 可解释的注意力权重
4. 预测区间估计

## 创新性评分：0.9/1.0

## 适用场景

| 场景 | 特点 | 示例 |
|-----|------|-----|
| 需求预测 | 多因素影响 | 销售、能源需求 |
| 金融预测 | 复杂模式 | 股价、汇率 |
| 气象预测 | 时空关联 | 温度、降水 |
| 流量预测 | 周期性 | 网络流量、交通 |

## 核心架构

```
                    ┌─────────────────┐
                    │  Variable       │
                    │  Selection      │
                    │  Network        │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
   ┌─────────┐        ┌─────────┐         ┌─────────┐
   │ Encoder │        │ Decoder │         │ Static  │
   │ (LSTM)  │───────▶│ (LSTM)  │◀────────│ Enrichm │
   └─────────┘        └────┬────┘         └─────────┘
                           │
                    ┌──────┴──────┐
                    │  Temporal   │
                    │  Self-Attn  │
                    └──────┬──────┘
                           │
                    ┌──────┴──────┐
                    │  Quantile   │
                    │  Outputs    │
                    └─────────────┘
```

## 使用方法

### 基础使用

```python
from transformer_forecasting import TFTForecaster

# 创建预测器
forecaster = TFTForecaster(
    max_encoder_length=60,      # 历史窗口
    max_prediction_length=20,   # 预测步数
    static_categoricals=['id'],
    time_varying_known=['dayofweek', 'month'],
    time_varying_unknown=['target']
)

# 训练
forecaster.fit(train_data, val_data, epochs=100)

# 预测
predictions = forecaster.predict(test_data)

# 获取注意力权重（可解释性）
attention = forecaster.get_attention_weights(test_data)
```

### 可解释性分析

```python
# 变量重要性
importance = forecaster.variable_importance()
forecaster.plot_variable_importance(save_path='figures/var_importance.pdf')

# 注意力可视化
forecaster.plot_attention_heatmap(sample_idx=0, save_path='figures/attention.pdf')

# 预测分解
decomposition = forecaster.decompose_prediction(sample_idx=0)
```

## 输出格式

```json
{
  "model": {
    "type": "TemporalFusionTransformer",
    "encoder_length": 60,
    "prediction_length": 20
  },
  "metrics": {
    "MAE": 0.123,
    "RMSE": 0.156,
    "MAPE": 0.045
  },
  "interpretability": {
    "variable_importance": {"var1": 0.35, "var2": 0.28},
    "attention_patterns": "saved to figures/"
  },
  "figures": [
    "predictions.pdf",
    "var_importance.pdf", 
    "attention.pdf"
  ]
}
```

## O奖加分建议

- 展示注意力权重的可解释性
- 与ARIMA/LSTM进行性能对比
- 使用SHAP分析特征贡献
- 提供预测区间（分位数预测）

## 相关技能

- `model-explainer` - SHAP/LIME解释
- `uncertainty-quantifier` - 不确定性量化
- `chart-generator` - 结果可视化

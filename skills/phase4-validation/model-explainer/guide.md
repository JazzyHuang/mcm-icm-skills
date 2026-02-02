---
name: model-explainer
description: 可解释AI模块，集成SHAP和LIME实现模型解释。提供全局和局部特征重要性分析，自动生成解释性图表，增强论文的可解释性和评委理解度。
---

# 模型解释器 (Model Explainer)

## 功能概述

集成多种可解释AI方法，提供：
1. SHAP值计算（全局和局部）
2. LIME局部解释
3. 部分依赖图(PDP)
4. 特征交互分析
5. 自动生成解释文本

## 创新性评分：0.85/1.0

## 支持的解释方法

| 方法 | 类型 | 适用模型 | 输出 |
|-----|------|---------|------|
| SHAP | 全局+局部 | 所有模型 | 特征贡献值 |
| LIME | 局部 | 黑盒模型 | 局部线性近似 |
| PDP | 全局 | 所有模型 | 特征边际效应 |
| ICE | 局部 | 所有模型 | 个体条件期望 |

## 使用方法

### SHAP分析

```python
from model_explainer import SHAPExplainer

explainer = SHAPExplainer(model)

# 全局分析
shap_values = explainer.explain_global(X_test)
explainer.plot_summary(save_path='figures/shap_summary.pdf')
explainer.plot_importance(save_path='figures/shap_importance.pdf')

# 局部分析
local_shap = explainer.explain_local(X_test[0])
explainer.plot_waterfall(local_shap, save_path='figures/shap_waterfall.pdf')
explainer.plot_force(local_shap, save_path='figures/shap_force.pdf')

# 特征依赖
explainer.plot_dependence('feature_name', save_path='figures/shap_dep.pdf')
```

### LIME分析

```python
from model_explainer import LIMEExplainer

explainer = LIMEExplainer(model, X_train, feature_names)

# 解释单个预测
explanation = explainer.explain_instance(X_test[0])
explainer.plot_explanation(explanation, save_path='figures/lime.pdf')
```

### 部分依赖图

```python
from model_explainer import PDPExplainer

explainer = PDPExplainer(model)

# 单特征PDP
explainer.plot_partial_dependence(
    X_test, 
    features=['feature1', 'feature2'],
    save_path='figures/pdp.pdf'
)

# 双特征交互
explainer.plot_interaction(
    X_test,
    features=('feature1', 'feature2'),
    save_path='figures/pdp_interaction.pdf'
)
```

## 输出格式

```json
{
  "method": "SHAP",
  "global_importance": {
    "feature1": 0.35,
    "feature2": 0.28,
    "feature3": 0.20
  },
  "local_explanation": {
    "sample_id": 0,
    "prediction": 0.78,
    "contributions": {
      "feature1": 0.15,
      "feature2": -0.08
    }
  },
  "figures": [
    "shap_summary.pdf",
    "shap_importance.pdf",
    "shap_waterfall.pdf"
  ],
  "narrative": "The model prediction is primarily driven by feature1 (35% importance)..."
}
```

## 图表类型

1. **Summary Plot**: 所有样本的SHAP值分布
2. **Importance Plot**: 特征重要性排序
3. **Waterfall Plot**: 单个预测的贡献分解
4. **Force Plot**: 预测的力图可视化
5. **Dependence Plot**: 特征与SHAP值的关系
6. **Interaction Plot**: 特征交互效应

## O奖加分建议

- 结合SHAP和模型预测讲故事
- 展示关键特征如何影响预测
- 提供反事实解释（如果X增加，结果会...）
- 与领域知识对比验证解释的合理性

## 相关技能

- `transformer-forecasting` - 可解释注意力
- `sensitivity-analyzer` - 敏感性分析
- `chart-generator` - 图表生成

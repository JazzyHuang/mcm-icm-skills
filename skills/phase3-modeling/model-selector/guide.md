---
name: model-selector
description: 智能推荐最佳的数学模型组合。基于问题特征自动匹配合适的模型类型，提供决策矩阵和备选方案，支持MCM/ICM所有题型。
---

# 模型选择器 (Model Selector)

## 功能概述

根据问题特征智能推荐最佳的数学模型组合，提供主模型和备选方案。

## 决策矩阵

### 按问题特征选择

| 问题特征 | 推荐模型 | 备选方案 |
|---------|---------|---------|
| 连续时间演化 | ODE/PDE | 差分方程、状态空间 |
| 离散优化 | MILP、DP | 启发式、近似算法 |
| 多目标权衡 | NSGA-II、Pareto | 加权和、约束法 |
| 网络结构 | 图论、网络流 | 社区检测、中心性 |
| 时间序列预测 | LSTM+Prophet混合 | Transformer、ARIMA |
| 不确定决策 | 随机规划、鲁棒优化 | 贝叶斯、Monte Carlo |
| 分类识别 | 集成学习(Stacking) | SVM、神经网络 |

### 按题型选择

| 题型 | 主推模型 | 创新方向 |
|------|---------|---------|
| A-连续 | ODE/PDE | 多尺度方法、自适应网格 |
| B-离散 | 图论+DP | 元启发式混合 |
| C-数据 | 集成学习 | 深度学习+可解释性 |
| D-运筹 | MILP+网络流 | 分解算法、列生成 |
| E-可持续 | 多目标+系统动力学 | 因果推断、情景分析 |
| F-政策 | 博弈论+ABM | 行为建模、网络效应 |

## 输出格式

```json
{
  "problem_type": "A",
  "problem_features": [
    "continuous_time",
    "spatial_distribution",
    "optimization"
  ],
  "recommendations": [
    {
      "rank": 1,
      "model_type": "PDE-constrained Optimization",
      "description": "偏微分方程约束的优化问题",
      "methods": ["Finite Element Method", "Adjoint Method"],
      "tools": ["FEniCS", "scipy.optimize"],
      "applicability_score": 0.92,
      "innovation_potential": "high",
      "reasons": [
        "问题涉及空间分布的连续变化",
        "存在明确的优化目标"
      ]
    },
    {
      "rank": 2,
      "model_type": "System Dynamics",
      "description": "系统动力学模型",
      "methods": ["Stock-Flow Diagram", "Causal Loop"],
      "tools": ["PySD", "scipy.integrate"],
      "applicability_score": 0.78,
      "innovation_potential": "medium",
      "reasons": [
        "可以捕捉系统反馈机制",
        "便于进行情景分析"
      ]
    }
  ],
  "hybrid_suggestions": [
    {
      "combination": "PDE + Machine Learning",
      "description": "使用ML学习PDE的残差或参数",
      "innovation_score": 0.95
    }
  ],
  "warnings": [
    "PDE求解可能计算量较大，建议使用自适应网格"
  ]
}
```

## 选择流程

1. **特征提取**: 从问题描述中提取关键特征
2. **规则匹配**: 使用决策树匹配候选模型
3. **适用性评分**: 计算每个模型的适用性分数
4. **创新性评估**: 评估模型的创新潜力
5. **混合建议**: 提供混合模型创新方向
6. **排序输出**: 按推荐程度排序输出

## 模型分类

### 预测类模型
- 回归: 线性/非线性/多元
- 时间序列: ARIMA, Prophet, LSTM
- 机器学习: RF, XGBoost, NN

### 优化类模型
- 线性规划: LP, ILP, MILP
- 非线性规划: NLP, SQP
- 元启发式: GA, SA, PSO

### 评价类模型
- 层次分析: AHP
- 多准则: TOPSIS, ELECTRE
- 数据包络: DEA

### 动态类模型
- 微分方程: ODE, PDE
- 差分方程
- 系统动力学

## 相关技能

- `problem-type-classifier` - 题型分类
- `hybrid-model-designer` - 混合模型设计
- `model-builder` - 模型构建

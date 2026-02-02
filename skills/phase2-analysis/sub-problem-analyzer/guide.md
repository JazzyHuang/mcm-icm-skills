---
name: sub-problem-analyzer
description: 深入分析每个子问题，评估数据需求、模型复杂度、求解可行性和时间成本。为每个子问题提供详细的实施建议。
---

# 子问题分析器 (Sub-Problem Analyzer)

## 功能概述

对每个分解后的子问题进行深入分析，提供实施建议。

## 分析维度

### 1. 数据需求评估
- 需要哪些数据
- 数据来源(题目提供/需要收集)
- 数据质量要求
- 缺失数据的处理策略

### 2. 模型复杂度估计
- 变量数量和类型
- 约束条件复杂度
- 是否涉及非线性
- 计算规模预估

### 3. 求解可行性判断
- 是否有解析解
- 数值方法选择
- 收敛性问题
- 计算资源需求

### 4. 时间成本预估
- 建模时间
- 编码时间
- 调试时间
- 写作时间

## 输出格式

```json
{
  "sub_problem_id": "sub1",
  "description": "子问题描述",
  "analysis": {
    "data_requirements": {
      "provided": ["data1.csv"],
      "needed": ["population data", "economic indicators"],
      "sources": ["World Bank", "UN Data"],
      "quality_notes": "需要进行缺失值插补"
    },
    "complexity": {
      "level": "medium",
      "variables_count": 15,
      "constraints_count": 8,
      "is_linear": false,
      "scale": "medium (< 10000 variables)"
    },
    "feasibility": {
      "analytical_solution": false,
      "numerical_methods": ["gradient descent", "genetic algorithm"],
      "convergence_risk": "low",
      "computational_time": "< 1 hour"
    },
    "time_estimate": {
      "modeling": "4 hours",
      "coding": "3 hours",
      "debugging": "2 hours",
      "writing": "3 hours",
      "total": "12 hours"
    }
  },
  "recommendations": [
    "建议使用遗传算法处理非线性优化",
    "数据预处理应优先完成",
    "预留调试时间应对可能的收敛问题"
  ],
  "risks": [
    {
      "risk": "数据缺失过多",
      "probability": "medium",
      "mitigation": "使用插值或简化假设"
    }
  ]
}
```

## 分析流程

1. **输入接收**: 接收问题分解树中的子问题
2. **维度分析**: 按四个维度进行深入分析
3. **风险识别**: 识别潜在风险和挑战
4. **建议生成**: 生成实施建议
5. **优先级排序**: 根据重要性和依赖关系排序

## 相关技能

- `problem-decomposer` - 问题分解
- `model-selector` - 模型选择

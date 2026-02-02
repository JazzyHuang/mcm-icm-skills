---
name: problem-type-classifier
description: 识别美赛题目类型(MCM A/B/C或ICM D/E/F)并推荐相应的建模策略。基于题目特征自动分类，提供针对性的方法建议。
---

# 问题类型分类器 (Problem Type Classifier)

## 功能概述

根据题目内容自动识别问题类型，并提供针对该类型的建模策略建议。

## 题型特征

| 题型 | 竞赛 | 特点 | 核心方法 |
|------|------|------|---------|
| A | MCM | 连续数学、物理建模 | ODE/PDE、有限元、变分 |
| B | MCM | 离散数学、算法设计 | 图论、动态规划、组合优化 |
| C | MCM | 数据分析、预测建模 | 机器学习、时间序列、统计 |
| D | ICM | 运筹优化、网络科学 | 整数规划、网络流、调度 |
| E | ICM | 可持续发展、环境科学 | 多目标优化、系统动力学 |
| F | ICM | 政策分析、社会科学 | 博弈论、仿真、决策分析 |

## 分类依据

### A题(连续)特征关键词
- 物理过程: heat, flow, diffusion, wave, dynamics
- 连续变化: continuous, rate, derivative, gradient
- 空间分布: spatial, distribution, field

### B题(离散)特征关键词
- 组合结构: network, graph, path, schedule
- 离散选择: discrete, integer, combinatorial
- 算法问题: algorithm, search, optimization

### C题(数据)特征关键词
- 数据处理: data, dataset, analyze, pattern
- 预测建模: predict, forecast, trend, time series
- 机器学习: classify, cluster, regression

### D题(运筹)特征关键词
- 运筹优化: optimize, schedule, allocate, route
- 网络问题: network, flow, capacity, logistics
- 资源分配: resource, assignment, distribution

### E题(可持续)特征关键词
- 环境问题: environment, sustainability, climate, ecosystem
- 资源管理: renewable, conservation, biodiversity
- 长期影响: long-term, impact, future generations

### F题(政策)特征关键词
- 政策分析: policy, regulation, government, law
- 社会问题: social, public, community, stakeholder
- 决策支持: decision, strategy, intervention

## 输出格式

```json
{
  "classified_type": "A",
  "confidence": 0.85,
  "contest": "MCM",
  "type_name": "连续问题",
  "feature_matches": {
    "A": 0.85,
    "B": 0.10,
    "C": 0.05
  },
  "recommended_methods": [
    {
      "method": "微分方程建模",
      "applicability": "high",
      "reason": "问题涉及连续变化过程"
    }
  ],
  "warnings": []
}
```

## 策略建议

根据分类结果提供：
1. 推荐的数学模型类型
2. 常用求解方法
3. 数据需求评估
4. 可能的陷阱和注意事项

## 相关技能

- `problem-parser` - 问题解析
- `model-selector` - 模型选择

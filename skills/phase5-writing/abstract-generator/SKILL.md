---
name: abstract-generator
description: 生成O奖级别的论文摘要。遵循300-500词标准，包含背景、方法、结果、结论，并通过12轮以上迭代优化达到最佳质量。
---

# 摘要生成器 (Abstract Generator)

## 功能概述

生成符合O奖标准的论文摘要，通过多轮迭代优化。

## O奖摘要标准

### 结构要求 (300-500词)
```
[背景 1-2句] → [问题陈述 1句] → [方法概述 2-3句]
→ [关键创新 1-2句] → [主要结果 2-3句] → [结论价值 1-2句]

Keywords: keyword1; keyword2; keyword3; keyword4; keyword5
```

### 内容要求
1. **信息密度高**: 每句话都有信息量
2. **逻辑连贯**: 各部分自然过渡
3. **突出创新**: 强调方法的独特性
4. **量化结果**: 用数字说明成效
5. **语言精炼**: 无冗余表达

## 迭代优化流程

### 第1-3轮: 内容完整性
- 检查是否覆盖所有必要元素
- 添加遗漏的关键信息
- 确保逻辑连贯

### 第4-6轮: 语言精炼
- 删除冗余词汇
- 缩短句子长度
- 提高信息密度

### 第7-9轮: 创新性突出
- 强调方法创新点
- 突出与现有方法的区别
- 强化贡献描述

### 第10-12轮: 最终润色
- 语法检查
- 词汇选择优化
- 节奏和可读性

## 示例摘要

```
The global transition to renewable energy demands optimal placement 
of solar infrastructure. This paper develops a comprehensive 
mathematical framework for solar panel deployment optimization 
that integrates physical modeling with data-driven approaches.

We propose a novel hybrid model combining partial differential 
equations for solar radiation simulation with genetic algorithms 
for multi-objective optimization. Our Physics-Informed Neural 
Network (PINN) approach ensures physical consistency while 
achieving computational efficiency.

The model was validated using 10-year solar radiation data across 
50 US cities, achieving an R² of 0.94 and reducing computational 
time by 65% compared to traditional methods. Sensitivity analysis 
reveals that latitude and panel angle are the most influential 
factors, with Sobol indices of 0.42 and 0.31 respectively.

Our framework provides actionable insights for solar energy 
companies, enabling a projected 23% increase in energy harvest 
efficiency. The model's flexibility allows adaptation to various 
geographic and climatic conditions.

Keywords: solar energy; optimization; PINN; renewable energy; 
mathematical modeling
```

## 输出格式

```json
{
  "abstract": {
    "version": 12,
    "text": "...",
    "word_count": 287,
    "keywords": ["solar energy", "optimization", ...],
    "structure_check": {
      "background": true,
      "problem": true,
      "method": true,
      "innovation": true,
      "results": true,
      "conclusion": true
    },
    "quality_score": 0.92
  },
  "iteration_log": [
    {"version": 1, "focus": "content", "changes": 15},
    {"version": 2, "focus": "content", "changes": 8},
    ...
  ]
}
```

## 相关技能

- `abstract-iterative-optimizer` - 摘要迭代优化
- `section-writer` - 章节写作

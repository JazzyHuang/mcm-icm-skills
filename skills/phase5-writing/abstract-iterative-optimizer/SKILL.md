---
name: abstract-iterative-optimizer
description: 对摘要进行多轮迭代优化。基于O奖摘要特征进行自动评分，优化信息密度、语言精炼度、逻辑连贯性，确保达到最高质量。
---

# 摘要迭代优化器 (Abstract Iterative Optimizer)

## 功能概述

对生成的摘要进行12轮以上的迭代优化，确保达到O奖水平。

## 优化维度

### 1. 信息密度
- 每句话的信息量
- 冗余表达删除
- 关键信息突出

### 2. 语言精炼度
- 词汇精确性
- 句子简洁性
- 表达专业性

### 3. 逻辑连贯性
- 段落过渡
- 因果关系清晰
- 论证完整

### 4. 创新点突出
- 方法创新描述
- 与现有方法对比
- 贡献明确表达

## 评分标准

| 维度 | 权重 | 评分标准 |
|------|------|---------|
| 完整性 | 20% | 包含所有必要元素 |
| 信息密度 | 25% | 无冗余，高密度 |
| 语言质量 | 25% | 专业、精炼 |
| 创新突出 | 20% | 创新点明确 |
| 可读性 | 10% | 流畅、易理解 |

## 迭代流程

```python
def optimize_abstract(initial_abstract, max_iterations=15):
    abstract = initial_abstract
    history = []
    
    for i in range(max_iterations):
        # 评估当前版本
        score = evaluate(abstract)
        history.append({'version': i, 'score': score, 'text': abstract})
        
        # 达到目标分数则停止
        if score >= 0.90:
            break
            
        # 确定优化重点
        focus = identify_weakness(abstract, score)
        
        # 执行优化
        abstract = apply_optimization(abstract, focus)
        
    return abstract, history
```

## 输出格式

```json
{
  "optimized_abstract": "...",
  "final_score": 0.92,
  "iterations": 12,
  "improvement": {
    "initial_score": 0.65,
    "final_score": 0.92,
    "improvement_rate": 0.42
  },
  "optimization_history": [
    {
      "iteration": 1,
      "focus": "completeness",
      "score_before": 0.65,
      "score_after": 0.72,
      "changes": ["Added background context", "Included quantitative results"]
    }
  ]
}
```

## 相关技能

- `abstract-generator` - 摘要生成
- `grammar-checker` - 语法检查

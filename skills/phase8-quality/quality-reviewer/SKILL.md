---
name: quality-reviewer
description: 全面审查论文质量。基于美赛评分标准评估模型合理性、算法创新性、可视化强度、敏感性分析、伦理审查等维度。
---

# 质量审查器 (Quality Reviewer)

## 功能概述

对论文进行全面质量审查，评估是否达到O奖标准。

## 评分维度

| 维度 | 权重 | 评估要点 |
|------|------|---------|
| 模型合理性 | 30% | 假设合理、模型匹配问题 |
| 算法创新性 | 25% | 方法创新、跨领域迁移 |
| 可视化强度 | 20% | 图表专业、数据可视化 |
| 敏感性分析 | 15% | 参数鲁棒性验证 |
| 伦理审查 | 10% | 公平性、社会影响 |

## 评分标准

### 模型合理性 (30分)
- 假设充分论证 (10分)
- 模型与问题匹配 (10分)
- 数学推导正确 (10分)

### 算法创新性 (25分)
- 方法新颖性 (10分)
- 改进/组合创新 (8分)
- 跨领域应用 (7分)

### 可视化强度 (20分)
- 图表数量和质量 (10分)
- 信息表达清晰 (5分)
- 专业出版级别 (5分)

### 敏感性分析 (15分)
- 全局敏感性 (8分)
- 结果解释 (4分)
- 鲁棒性结论 (3分)

### 伦理审查 (10分)
- 公平性考虑 (5分)
- 社会影响 (5分)

## 输出格式

```json
{
  "quality_review": {
    "total_score": 85,
    "grade": "O-Award Potential",
    "dimensions": {
      "model_rationality": {"score": 26, "max": 30},
      "innovation": {"score": 20, "max": 25},
      "visualization": {"score": 18, "max": 20},
      "sensitivity": {"score": 13, "max": 15},
      "ethics": {"score": 8, "max": 10}
    },
    "strengths": ["Strong mathematical foundation", "Innovative hybrid approach"],
    "improvements": ["Add more sensitivity analysis", "Expand ethical discussion"],
    "recommendation": "论文质量较高，建议完善敏感性分析部分"
  }
}
```

## 相关技能

- `hallucination-detector` - 幻觉检测
- `grammar-checker` - 语法检查

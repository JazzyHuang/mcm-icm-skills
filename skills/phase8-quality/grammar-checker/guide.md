---
name: grammar-checker
description: 检查学术英语语法。使用AJE Grammar Check标准，检查时态一致性、主谓一致、冠词使用、学术词汇选择，输出LAT评分。
---

# 语法检查器 (Grammar Checker)

## 功能概述

检查论文的学术英语语法质量，确保语言专业规范。

## 检查项

### 1. 时态一致性
- 方法描述用现在时
- 实验结果用过去时
- 普遍真理用现在时

### 2. 主谓一致
- 单复数一致
- 集合名词处理

### 3. 冠词使用
- 定冠词/不定冠词
- 专有名词冠词

### 4. 学术词汇
- 避免口语化表达
- 使用正式词汇
- 术语一致性

### 5. 语态使用
- 适当使用主动/被动
- 避免过度被动

## LAT评分标准

| 分数 | 等级 | 描述 |
|------|------|------|
| 9-10 | 优秀 | 接近母语水平 |
| 7-8 | 良好 | 少量小错误 |
| 5-6 | 及格 | 有明显错误但可理解 |
| <5 | 不及格 | 需要大幅修改 |

## 输出格式

```json
{
  "grammar_check": {
    "lat_score": 8.2,
    "grade": "良好",
    "issues": [
      {
        "type": "tense",
        "location": "Line 45",
        "original": "The model shows that...",
        "suggestion": "The model showed that...",
        "reason": "使用过去时描述实验结果"
      }
    ],
    "statistics": {
      "total_sentences": 350,
      "issues_found": 12,
      "issue_rate": 0.034
    }
  }
}
```

## 最低要求

- O奖目标: LAT ≥ 7.5

## 相关技能

- `academic-english-optimizer` - 学术英语优化
- `consistency-checker` - 一致性检查

---
name: fact-checker
description: 核查论文内容的事实准确性。检查数值计算、引用准确性、逻辑一致性，确保论文内容可靠。
---

# 事实核查器 (Fact Checker)

## 功能概述

验证论文中的事实准确性，防止错误和幻觉。

## 核查维度

### 1. 数值计算
- 公式推导正确性
- 计算结果准确性
- 数量级合理性

### 2. 引用准确性
- 引用内容与原文一致
- 数据来源可追溯
- 统计数字准确

### 3. 逻辑一致性
- 前后论述不矛盾
- 假设与结论一致
- 数据与图表对应

### 4. 数据匹配
- 文中数字与表格一致
- 图表与描述一致
- 结果与方法匹配

## 输出格式

```json
{
  "fact_check": {
    "status": "passed",
    "issues_found": 2,
    "issues": [
      {
        "type": "numerical",
        "location": "Section 3.2",
        "description": "计算结果与公式不一致",
        "severity": "high",
        "suggestion": "重新计算或检查公式"
      }
    ],
    "verified_items": 45,
    "confidence": 0.95
  }
}
```

## 相关技能

- `section-writer` - 章节写作
- `hallucination-detector` - 幻觉检测

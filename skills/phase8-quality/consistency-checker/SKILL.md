---
name: consistency-checker
description: 检查全文一致性。验证符号定义使用、术语统一、数值引用、图表交叉引用、假设使用的一致性。
---

# 一致性检查器 (Consistency Checker)

## 功能概述

检查论文全文的一致性，确保内容前后统一。

## 检查维度

### 1. 符号一致性
- 定义与使用一致
- 同一概念同一符号
- 上下标一致

### 2. 术语一致性
- 同一概念同一术语
- 缩写定义后使用
- 避免混用同义词

### 3. 数值一致性
- 文中数字与表格一致
- 计算结果一致
- 单位统一

### 4. 引用一致性
- 图表编号正确
- 交叉引用有效
- 文献引用准确

### 5. 假设一致性
- 假设使用贯穿全文
- 没有违反假设

## 输出格式

```json
{
  "consistency_check": {
    "passed": true,
    "issues": [
      {
        "type": "symbol",
        "description": "Symbol α used with different meanings",
        "locations": ["Section 2.1", "Section 3.4"],
        "severity": "medium"
      }
    ],
    "statistics": {
      "symbols_checked": 45,
      "terms_checked": 120,
      "references_checked": 35
    }
  }
}
```

## 相关技能

- `grammar-checker` - 语法检查
- `quality-reviewer` - 质量审查

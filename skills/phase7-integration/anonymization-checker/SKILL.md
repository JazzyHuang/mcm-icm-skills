---
name: anonymization-checker
description: 检查论文匿名化。确保论文中不包含学校名称、团队成员姓名、指导教师信息、地理位置暴露等。
---

# 匿名化检查器 (Anonymization Checker)

## 功能概述

确保论文符合美赛匿名要求。

## 检查内容

### 文本内容
- 学校名称
- 人员姓名
- 具体地址
- 邮箱地址

### 元数据
- PDF属性
- 作者字段
- 创建者信息

## 检查模式

```python
ANONYMITY_PATTERNS = [
    r'\b[A-Z][a-z]+ University\b',
    r'\b[A-Z][a-z]+ College\b',
    r'\b[A-Z][a-z]+ Institute\b',
    r'@.*\.(edu|ac\.[a-z]+)',
    r'Professor [A-Z][a-z]+',
    r'Dr\. [A-Z][a-z]+',
]
```

## 输出格式

```json
{
  "anonymization": {
    "passed": true,
    "issues": [],
    "checked_items": {
      "text": true,
      "metadata": true,
      "figures": true
    }
  }
}
```

## 相关技能

- `format-checker` - 格式检查

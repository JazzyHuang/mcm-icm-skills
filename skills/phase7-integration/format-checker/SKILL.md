---
name: format-checker
description: 检查论文格式合规性。验证页数限制、字体大小、匿名性、控制号显示、图表编号等美赛规范。
---

# 格式检查器 (Format Checker)

## 功能概述

检查论文是否符合美赛格式要求。

## 检查项

### 1. 页数检查
- 总页数 ≤ 25页
- 包含摘要页

### 2. 字体检查
- 正文 ≥ 12pt
- 图表标签 ≥ 8pt

### 3. 匿名性检查
- 无学校名称
- 无团队成员姓名
- 无指导教师姓名

### 4. 控制号检查
- 正确显示在指定位置
- 格式正确（5位数字）

### 5. 图表检查
- 编号连续
- 有标题和标签
- 引用正确

### 6. 引用格式
- 格式统一
- 编号连续或按作者排序

## 检查脚本

```python
def check_format(pdf_path):
    checks = {
        'page_count': check_pages(pdf_path) <= 25,
        'font_size': check_font_size(pdf_path) >= 12,
        'anonymity': check_anonymity(pdf_path),
        'control_number': check_control_number(pdf_path),
        'figures': check_figure_numbering(pdf_path),
        'citations': check_citation_format(pdf_path)
    }
    return checks
```

## 输出格式

```json
{
  "format_check": {
    "passed": true,
    "checks": {
      "pages": {"passed": true, "value": 24, "limit": 25},
      "font": {"passed": true, "min_size": 12},
      "anonymity": {"passed": true, "issues": []},
      "control_number": {"passed": true, "format": "XXXXX"},
      "figures": {"passed": true, "count": 8, "sequential": true},
      "citations": {"passed": true, "format": "consistent"}
    }
  }
}
```

## 相关技能

- `latex-compiler` - LaTeX编译
- `anonymization-checker` - 匿名化检查

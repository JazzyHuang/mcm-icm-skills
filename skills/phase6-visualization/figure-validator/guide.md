---
name: figure-validator
description: 验证图表质量和规范性。检查分辨率、标签完整性、单位标注、图例准确性、编号连续性。
---

# 图表验证器 (Figure Validator)

## 功能概述

验证生成图表的质量和规范性。

## 检查项

### 1. 分辨率检查
- 最小300 DPI
- 尺寸适合页面

### 2. 标签完整性
- X轴标签
- Y轴标签
- 图例标签
- 单位标注

### 3. 编号检查
- 编号连续
- 与文中引用对应

### 4. 格式检查
- 字体大小合适
- 颜色对比度
- 线条清晰

## 输出格式

```json
{
  "validation": {
    "passed": true,
    "checks": {
      "resolution": {"passed": true, "value": 300},
      "labels": {"passed": true, "missing": []},
      "numbering": {"passed": true, "sequence": [1,2,3,4]},
      "format": {"passed": true, "issues": []}
    }
  }
}
```

## 相关技能

- `chart-generator` - 图表生成
- `table-formatter` - 表格格式化

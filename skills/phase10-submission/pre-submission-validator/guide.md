---
name: pre-submission-validator
description: 提交前全面验证。验证PDF可打开、页数符合要求、内容完整无缺失、格式正确无乱码、图表正常显示。
---

# 提交前验证器 (Pre-Submission Validator)

## 功能概述

在提交前进行最终全面验证。

## 验证项

### 1. 文件验证
- PDF可正常打开
- 文件未损坏
- 大小合理

### 2. 内容验证
- 所有章节存在
- 无空白页
- 无缺失内容

### 3. 格式验证
- 无乱码
- 字体正确嵌入
- 排版正确

### 4. 视觉验证
- 图表正常显示
- 公式正确渲染
- 表格完整

## 输出格式

```json
{
  "validation": {
    "passed": true,
    "checks": {
      "file_integrity": true,
      "content_complete": true,
      "format_correct": true,
      "visual_check": true
    },
    "ready_for_submission": true
  }
}
```

## 相关技能

- `submission-preparer` - 提交准备
- `submission-checklist` - 提交检查清单

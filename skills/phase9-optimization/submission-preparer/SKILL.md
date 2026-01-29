---
name: submission-preparer
description: 准备提交材料。生成最终PDF、规范化文件名、清理元数据、检查文件大小。
---

# 提交准备器 (Submission Preparer)

## 功能概述

准备最终提交材料，确保符合所有提交要求。

## 准备任务

### 1. 生成最终PDF
- 完整编译
- 检查页数
- 验证内容完整

### 2. 文件名规范
- 格式: 控制号.pdf
- 示例: 2412345.pdf

### 3. 元数据清理
- 移除作者信息
- 清除创建者信息
- 删除修改历史

### 4. 文件大小检查
- 推荐 < 20MB
- 压缩大图片

## 提交清单

```markdown
□ PDF文件名为控制号
□ 总页数 ≤ 25页
□ 摘要页为第一页
□ 无学校/姓名信息
□ 控制号正确显示
□ 选题标识正确
□ 所有图表清晰
□ 参考文献完整
```

## 输出格式

```json
{
  "submission": {
    "file": "2412345.pdf",
    "size_mb": 5.2,
    "pages": 24,
    "ready": true,
    "checklist_complete": true
  }
}
```

## 相关技能

- `pre-submission-validator` - 提交前验证
- `submission-checklist` - 提交检查清单

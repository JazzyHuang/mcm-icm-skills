---
name: compilation-error-handler
description: 处理LaTeX编译错误。自动定位错误位置、分析错误原因、尝试自动修复。
---

# 编译错误处理器 (Compilation Error Handler)

## 功能概述

处理LaTeX编译过程中的错误，尝试自动修复。

## 常见错误类型

### 1. 语法错误
- 缺失括号
- 环境未闭合
- 命令拼写错误

### 2. 包冲突
- 包加载顺序
- 不兼容的包

### 3. 引用错误
- 未定义的引用
- 缺失的图片文件

### 4. 编码问题
- 非UTF-8字符
- 特殊字符未转义

## 自动修复

```python
ERROR_FIXES = {
    'Undefined control sequence': 'check_package_import',
    'Missing $ inserted': 'wrap_in_math_mode',
    'File not found': 'check_file_path',
    'Missing \\begin{document}': 'add_document_begin',
}
```

## 输出格式

```json
{
  "error_handling": {
    "errors_found": 2,
    "errors_fixed": 2,
    "errors": [
      {
        "type": "syntax",
        "line": 125,
        "message": "Missing $ inserted",
        "fix": "Added math mode delimiters",
        "fixed": true
      }
    ]
  }
}
```

## 相关技能

- `latex-compiler` - LaTeX编译
- `format-checker` - 格式检查

---
name: problem-parser
description: 解析美赛题目PDF或文本，提取结构化问题描述。支持MCM A/B/C和ICM D/E/F所有题型。能够识别子问题、数据要求、约束条件等关键信息。当用户提供美赛题目时自动触发。
---

# 问题解析器 (Problem Parser)

## 功能概述

解析美赛(MCM/ICM)题目文档，提取结构化的问题描述，为后续建模提供基础。

## 输入格式

接受以下格式的题目输入：
- PDF文件路径
- 纯文本内容
- Markdown格式内容

## 输出格式

输出结构化的JSON格式问题描述：

```json
{
  "problem_type": "A",
  "problem_title": "问题标题",
  "background": "问题背景描述",
  "main_questions": [
    {
      "id": 1,
      "description": "第一个子问题描述",
      "requirements": ["要求1", "要求2"],
      "deliverables": ["交付物1"]
    }
  ],
  "provided_data": {
    "files": ["data1.csv"],
    "description": "数据描述"
  },
  "constraints": ["约束条件1", "约束条件2"],
  "keywords": ["关键词1", "关键词2"]
}
```

## 处理流程

1. **文档预处理**
   - PDF转文本 (使用pdfplumber)
   - 清理格式噪声
   - 识别文档结构

2. **内容提取**
   - 识别题型标识 (Problem A/B/C/D/E/F)
   - 提取题目标题
   - 分割主要段落

3. **结构化分析**
   - 识别子问题 (通常以数字编号)
   - 提取约束条件
   - 识别数据要求
   - 提取关键术语

4. **输出生成**
   - 生成JSON格式输出
   - 验证必要字段完整性

## 边界情况处理

- **图片/表格**: 使用OCR识别，提取到`images`和`tables`字段
- **多语言**: 自动检测语言，非英文内容标记处理
- **格式异常**: 尽可能提取，缺失字段标记为null并添加警告

## 使用脚本

```python
from scripts.pdf_parser import parse_problem_pdf
from scripts.structure_analyzer import analyze_structure

# 解析PDF
raw_text = parse_problem_pdf("problem.pdf")

# 分析结构
structured = analyze_structure(raw_text)
```

## 示例

输入题目文本：
```
2026 MCM Problem A: Optimizing Solar Panel Placement

Background: Solar energy is becoming increasingly important...

Your team should:
1. Develop a model to determine optimal panel angles...
2. Analyze the impact of geographic location...
3. Write a memo to a solar company...
```

输出：
```json
{
  "problem_type": "A",
  "problem_title": "Optimizing Solar Panel Placement",
  "background": "Solar energy is becoming increasingly important...",
  "main_questions": [
    {
      "id": 1,
      "description": "Develop a model to determine optimal panel angles",
      "type": "modeling"
    },
    {
      "id": 2,
      "description": "Analyze the impact of geographic location",
      "type": "analysis"
    },
    {
      "id": 3,
      "description": "Write a memo to a solar company",
      "type": "communication"
    }
  ],
  "keywords": ["solar panel", "optimization", "geographic location"]
}
```

## 相关技能

- `problem-type-classifier` - 题型分类
- `variable-definer` - 变量定义

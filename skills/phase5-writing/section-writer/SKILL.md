---
name: section-writer
description: 生成美赛论文各章节内容。遵循学术写作规范，使用XtraGPT上下文感知修订技术，确保内容准确、逻辑清晰、语言专业。
---

# 章节写作器 (Section Writer)

## 功能概述

生成美赛论文的各个章节内容，遵循学术写作规范。

## 章节结构

### 1. Problem Restatement (问题重述)
- 用自己的语言重新阐述问题
- 识别关键约束和要求
- 明确研究目标

### 2. Problem Analysis (问题分析)
- 问题背景分析
- 关键因素识别
- 研究方法概述

### 3. Assumptions and Justifications (假设与论证)
- 列出所有假设
- 每个假设的论证
- 假设的影响分析

### 4. Notations and Definitions (符号与定义)
- 变量定义表
- 符号说明
- 术语解释

### 5. Model Design (模型设计)
- 建模思路
- 数学推导
- 模型创新点

### 6. Model Implementation (模型实现)
- 算法描述
- 参数设置
- 求解过程

### 7. Results and Analysis (结果与分析)
- 主要结果展示
- 结果解释
- 与预期比较

### 8. Sensitivity Analysis (敏感性分析)
- 参数敏感性
- 鲁棒性分析
- 结论稳定性

### 9. Model Evaluation (模型评估)
- 优点总结
- 缺点承认
- 改进方向

### 10. Conclusions (结论)
- 主要发现
- 实际意义
- 未来工作

## 写作规范

### 时态使用
- **方法描述**: 一般现在时
- **实验过程**: 一般过去时
- **结果讨论**: 一般现在时

### 语态使用
- 主动语态优先
- 适度使用被动语态
- 避免"we think"等主观表达

### 避免事项
- 套话和空洞表述
- 感叹号
- Firstly/Secondly等词
- 过度自夸

## 输出格式

每个章节输出LaTeX格式：

```latex
\section{Model Design}

\subsection{Overview}
Our modeling approach consists of three main components...

\subsection{Mathematical Formulation}
The objective function is defined as:
\begin{equation}
    \min_{x} \sum_{i=1}^{n} c_i x_i
\end{equation}
subject to the following constraints...
```

## 相关技能

- `fact-checker` - 事实核查
- `abstract-generator` - 摘要生成

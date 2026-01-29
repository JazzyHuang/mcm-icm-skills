---
name: variable-definer
description: 定义美赛建模中的所有变量、参数和符号。生成清晰无歧义的变量定义表，使用标准数学符号，遵循学术论文规范。
---

# 变量定义器 (Variable Definer)

## 功能概述

定义数学模型中的所有变量、参数和符号，生成规范的变量定义表。

## 变量分类

### 1. 决策变量 (Decision Variables)
模型需要确定的未知量
- 符号: 通常用 $x$, $y$, $z$ 或带下标的变量
- 示例: $x_i$ 表示第 $i$ 个城市的投资额

### 2. 参数 (Parameters)
模型中的已知常量
- 符号: 通常用希腊字母或大写字母
- 示例: $\alpha$ 表示增长率, $N$ 表示总数量

### 3. 状态变量 (State Variables)
描述系统状态的量
- 符号: 通常用函数形式
- 示例: $S(t)$ 表示时刻 $t$ 的库存

### 4. 中间变量 (Intermediate Variables)
计算过程中使用的辅助变量
- 符号: 通常用小写字母
- 示例: $h$ 表示步长

### 5. 集合和索引 (Sets and Indices)
定义问题结构
- 符号: 集合用大写字母, 索引用小写
- 示例: $i \in I$, $t \in T$

## 输出格式

### LaTeX 变量定义表

```latex
\section{Notations and Definitions}

\begin{table}[h]
\centering
\caption{Notation Summary}
\begin{tabular}{cll}
\toprule
\textbf{Symbol} & \textbf{Description} & \textbf{Unit} \\
\midrule
\multicolumn{3}{l}{\textit{Sets and Indices}} \\
$I$ & Set of locations & - \\
$T$ & Set of time periods & - \\
$i, j$ & Location indices & - \\
$t$ & Time index & - \\
\midrule
\multicolumn{3}{l}{\textit{Parameters}} \\
$N$ & Total population & persons \\
$\alpha$ & Growth rate & $\%$/year \\
$c_{ij}$ & Transportation cost from $i$ to $j$ & \$/unit \\
\midrule
\multicolumn{3}{l}{\textit{Decision Variables}} \\
$x_i$ & Investment at location $i$ & \$ \\
$y_{ij}$ & Flow from $i$ to $j$ & units \\
\midrule
\multicolumn{3}{l}{\textit{State Variables}} \\
$S_i(t)$ & Inventory at location $i$ at time $t$ & units \\
\bottomrule
\end{tabular}
\end{table}
```

### JSON 格式

```json
{
  "sets": [
    {
      "symbol": "I",
      "latex": "I",
      "description": "Set of locations",
      "unit": null
    }
  ],
  "indices": [
    {
      "symbol": "i",
      "latex": "i",
      "description": "Location index",
      "domain": "I"
    }
  ],
  "parameters": [
    {
      "symbol": "N",
      "latex": "N",
      "description": "Total population",
      "unit": "persons",
      "value_range": "positive integer"
    }
  ],
  "decision_variables": [
    {
      "symbol": "x_i",
      "latex": "x_i",
      "description": "Investment at location i",
      "unit": "dollars",
      "domain": "non-negative real",
      "index": "i ∈ I"
    }
  ],
  "state_variables": [
    {
      "symbol": "S_i(t)",
      "latex": "S_i(t)",
      "description": "Inventory at location i at time t",
      "unit": "units",
      "index": "i ∈ I, t ∈ T"
    }
  ]
}
```

## 命名规范

### 符号选择原则
1. **一致性**: 全文使用相同符号表示相同概念
2. **直观性**: 符号应与概念有一定联系
3. **标准性**: 遵循领域内的通用约定
4. **简洁性**: 避免过于复杂的符号

### 常用符号约定

| 概念 | 推荐符号 | 避免使用 |
|------|---------|---------|
| 时间 | $t$, $T$ | $time$ |
| 空间 | $x$, $y$, $z$ | $space$ |
| 数量 | $n$, $N$ | $num$ |
| 概率 | $p$, $P$ | $prob$ |
| 成本 | $c$, $C$ | $cost$ |
| 比率 | $r$, $\alpha$, $\beta$ | $rate$ |

## 相关技能

- `assumption-generator` - 假设生成
- `constraint-identifier` - 约束识别
- `model-builder` - 模型构建

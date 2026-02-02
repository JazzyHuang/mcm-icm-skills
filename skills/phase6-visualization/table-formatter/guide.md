---
name: table-formatter
description: 生成专业的LaTeX表格。支持三线表格式、自动对齐、数值格式化，遵循booktabs风格。
---

# 表格格式化器 (Table Formatter)

## 功能概述

生成符合学术规范的LaTeX表格。

## 表格类型

### 1. 变量定义表
列出所有符号和变量的定义

### 2. 参数设置表
展示模型参数及其取值

### 3. 结果对比表
比较不同方法的结果

### 4. 敏感性分析表
展示参数敏感性结果

## LaTeX格式

### 三线表 (booktabs)

```latex
\begin{table}[htbp]
\centering
\caption{Model Parameters}
\label{tab:parameters}
\begin{tabular}{@{}lll@{}}
\toprule
Symbol & Description & Value \\
\midrule
$\alpha$ & Growth rate & 0.05 \\
$\beta$ & Decay factor & 0.02 \\
$N$ & Population size & 10000 \\
\bottomrule
\end{tabular}
\end{table}
```

### 多列表格

```latex
\begin{table}[htbp]
\centering
\caption{Comparison of Methods}
\label{tab:comparison}
\begin{tabular}{@{}lcccc@{}}
\toprule
Method & RMSE & MAE & R² & Time (s) \\
\midrule
Our Model & \textbf{0.152} & \textbf{0.098} & \textbf{0.923} & 2.34 \\
Baseline 1 & 0.198 & 0.134 & 0.876 & 1.23 \\
Baseline 2 & 0.175 & 0.112 & 0.901 & 5.67 \\
\bottomrule
\end{tabular}
\end{table}
```

## 输出格式

```json
{
  "table": {
    "type": "comparison",
    "latex": "...",
    "caption": "Comparison of different methods",
    "label": "tab:comparison",
    "columns": ["Method", "RMSE", "MAE", "R²"],
    "rows": 4
  }
}
```

## 相关技能

- `chart-generator` - 图表生成
- `figure-validator` - 图表验证

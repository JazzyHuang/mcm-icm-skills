---
name: citation-manager
description: 智能管理论文引用。使用BibLaTeX管理参考文献，智能推荐引用位置，自动格式化BibTeX。
---

# 引用管理器 (Citation Manager)

## 功能概述

管理论文中的所有引用，确保格式统一和内容准确。

## 功能列表

### 1. 引用位置推荐
基于上下文智能推荐适合引用的位置

### 2. BibTeX格式化
自动格式化和验证BibTeX条目

### 3. 引用完整性检查
确保所有引用都在参考文献中

### 4. 未使用引用清理
移除未在文中引用的参考文献

## BibTeX格式

```bibtex
@article{smith2024optimization,
    author = {Smith, John and Doe, Jane},
    title = {Optimization Methods for Solar Panel Placement},
    journal = {Energy Systems},
    year = {2024},
    volume = {15},
    number = {3},
    pages = {123--145},
    doi = {10.1234/es.2024.001}
}
```

## 引用命令

```latex
% 单个引用
\cite{smith2024optimization}

% 多个引用
\cite{smith2024optimization,doe2023analysis}

% 页码引用
\cite[p.~125]{smith2024optimization}
```

## 输出格式

```json
{
  "citations": {
    "total": 25,
    "used_in_text": 23,
    "unused": ["author2020paper"],
    "missing": [],
    "bibtex_file": "references.bib"
  }
}
```

## 相关技能

- `literature-searcher` - 文献检索
- `citation-validator` - 引用验证

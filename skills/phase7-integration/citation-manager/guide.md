---
name: citation-manager
description: 智能管理论文引用。使用BibLaTeX管理参考文献，智能推荐引用位置，自动格式化BibTeX。支持引用分类管理、多样性统计和自动补充建议。
---

# 引用管理器 (Citation Manager)

## 功能概述

管理论文中的所有引用，确保格式统一、内容准确，并满足引用多样性要求。

## 功能列表

### 1. 引用位置推荐
基于上下文智能推荐适合引用的位置

### 2. BibTeX格式化
自动格式化和验证BibTeX条目

### 3. 引用完整性检查
确保所有引用都在参考文献中

### 4. 未使用引用清理
移除未在文中引用的参考文献

### 5. 引用分类管理（新增）
将引用按来源类别分类：
- 学术论文 (academic_papers)
- 政府报告 (government_reports)
- 官方数据 (official_data)
- 题目引用 (problem_references)
- 其他来源 (other_sources)

### 6. 多样性统计（新增）
- 统计各类别引用数量
- 计算多样性评分
- 检查是否满足最低要求

### 7. 自动补充建议（新增）
当某类别引用不足时，提供搜索建议

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

### 基础统计

```json
{
  "citations": {
    "total": 12,
    "used_in_text": 11,
    "unused": ["author2020paper"],
    "missing": [],
    "bibtex_file": "references.bib"
  }
}
```

### 分类统计（新增）

```json
{
  "citation_categories": {
    "academic_papers": {
      "count": 5,
      "citations": ["smith2024opt", "jones2023model", "chen2024analysis", "lee2023study", "wang2024method"]
    },
    "government_reports": {
      "count": 2,
      "citations": ["worldbank2024report", "un2024sustainable"]
    },
    "official_data": {
      "count": 2,
      "citations": ["worldbank_gdp_2024", "oecd_education_2024"]
    },
    "problem_references": {
      "count": 2,
      "citations": ["mcm2024problema", "mcm2024dataa"]
    },
    "other_sources": {
      "count": 1,
      "citations": ["github2024repo"]
    }
  },
  "diversity_metrics": {
    "total_citations": 12,
    "categories_covered": 5,
    "diversity_score": 0.85,
    "meets_requirements": true
  },
  "category_requirements": {
    "academic_papers": {"required": 3, "actual": 5, "status": "pass"},
    "government_reports": {"required": 1, "actual": 2, "status": "pass"},
    "official_data": {"required": 1, "actual": 2, "status": "pass"},
    "problem_references": {"required": 1, "actual": 2, "status": "pass"},
    "other_sources": {"required": 0, "actual": 1, "status": "pass"}
  }
}
```

## 引用分类规则

### 自动分类

根据BibTeX条目类型和来源自动分类：

| BibTeX类型 | 来源标记 | 分类 |
|-----------|---------|------|
| @article, @inproceedings | - | academic_papers |
| @techreport | government, official | government_reports |
| @online | data, worldbank, un, oecd | official_data |
| @misc | mcm, icm, comap, problem | problem_references |
| 其他 | - | other_sources |

### 手动标记

在BibTeX条目中添加`category`字段：

```bibtex
@techreport{worldbank2024energy,
  author = {{World Bank}},
  title = {Global Energy Report 2024},
  ...
  category = {government}
}
```

## 多样性检查

### 最低要求

| 类别 | 最低数量 | 权重 |
|------|----------|------|
| 学术论文 | 3 | 40% |
| 政府报告 | 1 | 15% |
| 官方数据 | 1 | 15% |
| 题目引用 | 1 | 15% |
| 其他来源 | 0 | 15% |

### 多样性评分

- 达到最低要求：0.75+
- 超额引用奖励：最高1.0

### 补充建议

当某类别不足时，自动生成搜索建议：

```json
{
  "recommendations": [
    {
      "category": "government_reports",
      "message": "需要至少1篇政府报告",
      "search_suggestions": [
        "使用WebSearch搜索: [主题] government report official",
        "检查World Bank, UN, OECD发布的相关报告"
      ]
    }
  ]
}
```

## 使用流程

### 在论文整合阶段

```python
# 1. 收集所有引用
citations = collect_all_citations([
    'deep-reference-searcher',
    'problem-reference-extractor',
    'data-collector'
])

# 2. 分类和统计
categorized = categorize_citations(citations)
diversity = calculate_diversity(categorized)

# 3. 检查多样性
if not diversity['meets_requirements']:
    recommendations = generate_recommendations(categorized)
    # 执行补充搜索
    
# 4. 生成最终BibTeX文件
export_bibtex(citations, 'references.bib')
```

## 相关技能

- `literature-searcher` - 文献检索
- `citation-validator` - 引用验证
- `deep-reference-searcher` - 多源深度搜索
- `citation-diversity-validator` - 多样性验证
- `problem-reference-extractor` - 题目引用提取
- `data-collector` - 数据源引用

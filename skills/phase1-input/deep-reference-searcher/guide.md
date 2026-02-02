---
name: deep-reference-searcher
description: 多源深度文献搜索器。整合Semantic Scholar、arXiv、CrossRef、OpenAlex、Google Scholar等多个学术数据源，同时支持政府报告、官方数据源、新闻媒体等非学术源的搜索。确保引用来源的广泛性和多样性，满足美赛O奖级别要求。
---

# 多源深度文献搜索器 (Deep Reference Searcher)

## 功能概述

自动从多个数据源深度搜索与美赛题目相关的参考文献，确保引用来源的广泛性和多样性。

## 核心理念

美赛O奖论文的引用需要体现参赛者广泛深刻的信息搜集能力，引用源应包含：
- 学术论文（期刊、会议、预印本）
- 政府/官方报告
- 官方数据源
- 题目附带信息
- 新闻/媒体源（可选）
- 技术文档（可选）

## 支持的数据源

### 1. 学术数据源

| 数据源 | API Key | 优先级 | 特点 |
|--------|---------|--------|------|
| Semantic Scholar | 可选 | 1 | AI驱动，引用关系丰富 |
| OpenAlex | 否 | 2 | 免费，240M+文献 |
| arXiv | 否 | 3 | 预印本，最新研究 |
| CrossRef | 否 | 4 | DOI权威元数据 |
| Google Scholar | 是(SerpAPI) | 5 | 覆盖面最广 |
| PubMed | 否 | 6 | 医学/生物领域 |

### 2. 政府/官方数据源

| 数据源 | 类型 | 覆盖领域 |
|--------|------|----------|
| World Bank Publications | 报告 | 经济、发展 |
| UN Publications | 报告 | 全球发展、可持续性 |
| OECD iLibrary | 报告 | 经济、政策 |
| CDC/WHO Reports | 报告 | 健康、流行病 |
| DOE/EPA Reports | 报告 | 能源、环境 |

### 3. 官方数据源（生成数据引用）

| 数据源 | URL | 数据类型 |
|--------|-----|----------|
| World Bank Data | data.worldbank.org | 经济/社会指标 |
| UN Data | data.un.org | 全球发展指标 |
| OECD Data | stats.oecd.org | 经济统计 |
| Kaggle Datasets | kaggle.com | 机器学习数据集 |

### 4. 新闻/媒体源

| 来源 | 可靠性评分 | 适用场景 |
|------|-----------|----------|
| Reuters | 高 | 经济、国际 |
| NY Times | 高 | 社会、政策 |
| Nature News | 高 | 科学新闻 |
| BBC | 高 | 国际新闻 |

## 搜索策略

### 多轮迭代搜索

```
第一轮: 广度搜索
├── 学术论文搜索 (Semantic Scholar + OpenAlex)
├── 政府报告搜索 (WebSearch)
└── 数据源搜索 (WebSearch)

第二轮: 深度搜索
├── 高引用论文的参考文献追踪
├── 相关作者的其他工作
└── 特定机构的专题报告

第三轮: 补充搜索
├── 检查引用多样性
├── 补充缺失类别
└── 验证引用质量
```

### 关键词扩展

```python
def expand_keywords(problem_keywords):
    """扩展搜索关键词"""
    expanded = []
    for kw in problem_keywords:
        expanded.extend([
            f"{kw} mathematical model",
            f"{kw} optimization algorithm",
            f"{kw} simulation study",
            f"{kw} case study analysis"
        ])
    return expanded
```

## 输出格式

### 引用分类结构

```json
{
  "citations": {
    "academic_papers": [
      {
        "bibtex_key": "smith2024optimization",
        "title": "...",
        "source": "semantic_scholar",
        "category": "academic",
        "doi": "10.1234/xxx",
        "citation_count": 150,
        "relevance_score": 0.92
      }
    ],
    "government_reports": [...],
    "official_data": [...],
    "problem_references": [...],
    "other_sources": [...]
  },
  "diversity_metrics": {
    "total_citations": 12,
    "categories_covered": 5,
    "diversity_score": 0.85
  },
  "bibtex_file": "references.bib"
}
```

### BibTeX输出

```bibtex
% Academic Papers
@article{smith2024optimization,
  author = {Smith, John},
  title = {Optimization Methods...},
  journal = {Energy Systems},
  year = {2024},
  doi = {10.1234/xxx}
}

% Government Reports
@techreport{worldbank2024energy,
  author = {{World Bank}},
  title = {Global Energy Outlook 2024},
  institution = {World Bank Group},
  year = {2024},
  url = {https://...}
}

% Official Data Sources
@online{worldbank_data_gdp,
  author = {{World Bank}},
  title = {World Bank Open Data: GDP per capita},
  year = {2024},
  url = {https://data.worldbank.org/indicator/NY.GDP.PCAP.CD},
  urldate = {2024-01-15}
}

% Problem Statement References
@misc{mcm2024problem,
  author = {{COMAP}},
  title = {2024 MCM Problem A},
  year = {2024},
  howpublished = {MCM/ICM Contest},
  note = {Official problem statement}
}
```

## 使用方式

```python
from scripts.multi_source_searcher import MultiSourceSearcher

searcher = MultiSourceSearcher(config_path="config/citation_sources.yaml")

# 执行多源搜索
results = searcher.search_all_sources(
    query="solar panel optimization energy efficiency",
    problem_context={
        "type": "A",
        "domain": "renewable energy",
        "keywords": ["solar", "optimization", "efficiency"]
    },
    min_citations=10,
    ensure_diversity=True
)

# 检查多样性
if results['diversity_metrics']['diversity_score'] < 0.75:
    # 自动补充缺失类别
    results = searcher.supplement_missing_categories(results)
```

## 多样性要求

### 最低要求

| 类别 | 最少数量 | 权重 |
|------|----------|------|
| 学术论文 | 3 | 40% |
| 政府报告 | 1 | 15% |
| 官方数据 | 1 | 15% |
| 题目引用 | 1 | 15% |
| 其他来源 | 0 | 15% |

### 多样性评分计算

```python
def calculate_diversity_score(citations):
    """计算引用多样性评分"""
    category_weights = {
        'academic_papers': 0.40,
        'government_reports': 0.15,
        'official_data': 0.15,
        'problem_references': 0.15,
        'other_sources': 0.15
    }
    
    score = 0
    for category, weight in category_weights.items():
        if category in citations and len(citations[category]) > 0:
            # 有该类别引用则获得该类别权重
            score += weight
            # 超过最低要求则额外加分
            if len(citations[category]) > MIN_REQUIRED[category]:
                score += weight * 0.2
    
    return min(score, 1.0)
```

## 错误处理

### API失败降级策略

```
Semantic Scholar 失败 → 切换到 OpenAlex
OpenAlex 失败 → 切换到 CrossRef
Google Scholar 失败 → 使用 WebSearch 补充
```

### 搜索结果不足处理

```
学术论文不足 → 扩展关键词，增加搜索范围
政府报告不足 → 使用 WebSearch 深度搜索
数据源不足 → 检查题目是否提供数据
```

## 相关技能

- `literature-searcher` - 基础文献检索
- `citation-validator` - 引用验证
- `citation-diversity-validator` - 多样性验证
- `problem-reference-extractor` - 题目引用提取
- `ai-deep-search-guide` - AI深度搜索引导

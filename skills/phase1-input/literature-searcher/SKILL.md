---
name: literature-searcher
description: 自动检索与美赛题目相关的学术文献。使用Semantic Scholar、CrossRef、arXiv等API获取高质量论文，生成BibTeX格式引用。支持基于关键词的智能搜索和相关度排序。
---

# 文献检索器 (Literature Searcher)

## 功能概述

自动检索与建模问题相关的学术文献，为论文提供可靠的参考文献支持。

## 支持的数据源

### 1. Semantic Scholar
- **特点**: AI驱动的学术搜索，提供引用关系
- **API**: https://api.semanticscholar.org/
- **需要API Key**: 推荐但非必需

### 2. CrossRef
- **特点**: DOI注册机构，权威元数据
- **API**: https://api.crossref.org/
- **需要API Key**: 否

### 3. arXiv
- **特点**: 预印本库，最新研究
- **API**: https://export.arxiv.org/api/
- **需要API Key**: 否

## 搜索策略

### 1. 关键词提取
从题目中提取关键词:
- 问题领域关键词
- 方法类关键词
- 应用场景关键词

### 2. 查询构建
```python
# 多视角查询生成
queries = [
    f"{domain} mathematical model",
    f"{method} algorithm optimization",
    f"{application} case study"
]
```

### 3. 结果排序
- 引用量权重
- 发表时间权重 (近3年优先)
- 相关度分数

## 输出格式

### BibTeX格式
```bibtex
@article{author2024title,
  author = {Author, First and Author, Second},
  title = {Paper Title},
  journal = {Journal Name},
  year = {2024},
  volume = {10},
  pages = {1-15},
  doi = {10.1234/example}
}
```

### 结构化JSON
```json
{
  "papers": [
    {
      "title": "Paper Title",
      "authors": ["Author1", "Author2"],
      "year": 2024,
      "venue": "Journal Name",
      "doi": "10.1234/example",
      "citation_count": 150,
      "abstract": "...",
      "relevance_score": 0.85,
      "bibtex": "..."
    }
  ],
  "total_found": 100,
  "search_queries": ["query1", "query2"],
  "bibtex_file": "references.bib"
}
```

## 使用脚本

```python
from scripts.semantic_scholar import search_papers
from scripts.citation_formatter import format_bibtex

# 搜索论文
papers = search_papers(
    query="solar panel optimization mathematical model",
    limit=20,
    year_range=(2020, 2024)
)

# 生成BibTeX
bibtex = format_bibtex(papers)
```

## 引用验证

- 验证DOI有效性
- 检查作者信息完整性
- 确认出版信息准确

## 相关技能

- `citation-validator` - 引用验证
- `citation-manager` - 引用管理

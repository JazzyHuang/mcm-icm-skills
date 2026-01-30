---
name: literature-searcher
description: 自动检索与美赛题目相关的学术文献。使用Semantic Scholar、OpenAlex、CrossRef、arXiv、Google Scholar等多个API获取高质量论文，生成BibTeX格式引用。支持基于关键词的智能搜索和相关度排序。
---

# 文献检索器 (Literature Searcher)

## 功能概述

自动检索与建模问题相关的学术文献，为论文提供可靠的参考文献支持。

## 支持的数据源

### 1. Semantic Scholar
- **特点**: AI驱动的学术搜索，提供引用关系
- **API**: https://api.semanticscholar.org/
- **需要API Key**: 推荐但非必需
- **脚本**: `scripts/semantic_scholar.py`

### 2. OpenAlex (新增)
- **特点**: 免费开放API，240M+学术文献，无需API Key
- **API**: https://api.openalex.org/
- **需要API Key**: 否
- **脚本**: `scripts/openalex_api.py`
- **优势**: 覆盖面广，支持DOI搜索，提供引用关系和作者信息

### 3. CrossRef
- **特点**: DOI注册机构，权威元数据
- **API**: https://api.crossref.org/
- **需要API Key**: 否
- **脚本**: `scripts/crossref_api.py` (如需单独使用)

### 4. arXiv
- **特点**: 预印本库，最新研究
- **API**: https://export.arxiv.org/api/
- **需要API Key**: 否
- **脚本**: `scripts/arxiv_api.py` (如需单独使用)

### 5. Google Scholar (新增)
- **特点**: 覆盖面最广的学术搜索引擎
- **API**: 通过SerpAPI代理访问
- **需要API Key**: 是 (SerpAPI Key)
- **脚本**: `scripts/google_scholar_api.py`
- **配置**: 在 `config/api_keys.yaml` 中设置 `serpapi_key`

### 6. PubMed (医学/生物领域)
- **特点**: 医学和生命科学文献数据库
- **API**: https://eutils.ncbi.nlm.nih.gov/entrez/
- **需要API Key**: 推荐但非必需
- **脚本**: `scripts/pubmed_api.py` (如需单独使用)

## 数据源优先级

| 优先级 | 数据源 | 适用场景 |
|--------|--------|----------|
| 1 | Semantic Scholar | 通用学术搜索，引用分析 |
| 2 | OpenAlex | 大规模搜索，免费无限制 |
| 3 | arXiv | 最新预印本，CS/数学/物理 |
| 4 | CrossRef | DOI验证，元数据补全 |
| 5 | Google Scholar | 最广覆盖，需API Key |
| 6 | PubMed | 医学/生物专题 |

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

## 使用示例

### 示例1：使用Semantic Scholar搜索

```python
from scripts.semantic_scholar import search_papers, generate_bibtex

# 搜索与优化相关的论文
papers = search_papers(
    query="solar panel optimization mathematical model",
    limit=20,
    year_range=(2020, 2024)
)

# 打印搜索结果
for paper in papers:
    print(f"标题: {paper['title']}")
    print(f"作者: {', '.join(paper['authors'][:3])}")
    print(f"年份: {paper['year']}, 引用数: {paper['citation_count']}")
    print(f"DOI: {paper['doi']}")
    print("---")

# 导出为BibTeX文件
for paper in papers:
    bibtex = generate_bibtex(paper)
    print(bibtex)
```

### 示例2：使用CrossRef API搜索

```python
from scripts.crossref_api import search_works, search_and_export

# 搜索最近5年的相关文献
works = search_works(
    query="machine learning renewable energy",
    limit=15,
    filter_params={'from-pub-date': '2020'}
)

# 导出为BibTeX文件
result = search_and_export(
    query="machine learning renewable energy",
    output_file="references.bib",
    limit=15,
    year_from=2020
)
print(f"找到 {result['works_found']} 篇论文，已导出到 {result['output_file']}")
```

### 示例3：使用arXiv API搜索预印本

```python
from scripts.arxiv_api import search_papers, search_and_export

# 搜索机器学习相关预印本
papers = search_papers(
    query="neural network optimization",
    limit=10,
    categories=['cs.LG', 'stat.ML']  # 限制到机器学习分类
)

# 获取特定论文的详情
from scripts.arxiv_api import get_paper_by_id
paper = get_paper_by_id("2301.12345")
if paper:
    print(f"标题: {paper['title']}")
    print(f"摘要: {paper['abstract'][:200]}...")
```

### 示例4：使用PubMed搜索医学文献

```python
from scripts.pubmed_api import search_papers, search_and_export

# 搜索流行病学建模相关文献
papers = search_papers(
    query="epidemiological modeling COVID-19",
    limit=10,
    date_range=(2020, 2024)
)

# 导出结果
for paper in papers:
    print(f"PMID: {paper['pmid']}")
    print(f"标题: {paper['title']}")
    print(f"期刊: {paper['journal']}")
```

### 示例5：多源综合搜索

```python
from scripts.semantic_scholar import search_papers as ss_search
from scripts.crossref_api import search_works as cr_search
from scripts.arxiv_api import search_papers as arxiv_search

# 从多个源搜索
query = "optimization algorithm energy systems"

# Semantic Scholar
ss_papers = ss_search(query, limit=5)
print(f"Semantic Scholar: {len(ss_papers)} 篇")

# CrossRef
cr_works = cr_search(query, limit=5)
print(f"CrossRef: {len(cr_works)} 篇")

# arXiv
arxiv_papers = arxiv_search(query, limit=5)
print(f"arXiv: {len(arxiv_papers)} 篇")

# 合并去重（基于DOI或标题）
all_papers = ss_papers + cr_works + arxiv_papers
# ... 去重逻辑 ...
```

## 引用验证

- 验证DOI有效性
- 检查作者信息完整性
- 确认出版信息准确

## 与其他技能集成

- **前置技能**: `problem-parser` 提供关键词
- **后续技能**: 
  - `citation-validator` - 验证引用真实性
  - `citation-diversity-validator` - 验证引用多样性
  - `citation-manager` - 管理最终引用列表

## 注意事项

- API调用有速率限制，请参考 `config/shared_constants.yaml`
- 推荐使用Semantic Scholar作为主要搜索源
- 对于特定领域（如医学），建议使用专业数据库（如PubMed）

## 更新日志

- 2026-01-30: 添加crossref_api.py、arxiv_api.py、pubmed_api.py
- 2026-01-30: 添加使用示例

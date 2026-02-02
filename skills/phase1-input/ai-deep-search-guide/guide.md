---
name: ai-deep-search-guide
description: AI深度搜索引导器。显式引导AI使用WebSearch工具和MCP服务器进行多轮、多角度的深度网络搜索，确保搜集到广泛多样的参考文献来源。这是确保美赛引用多样性的核心技能。
---

# AI深度搜索引导器 (AI Deep Search Guide)

## 功能概述

此技能的核心目的是**显式引导AI使用其内置的深度搜索能力**，包括：
- Cursor/Claude 的 **WebSearch** 工具
- **MCP服务器**（如 context7）进行文档和API检索
- **多轮迭代搜索**策略

## 重要说明

**当执行此技能时，你（AI）必须主动执行以下搜索操作，而不是仅仅描述它们。**

## 执行流程

### 第一轮：广度搜索（并行执行）

使用 **WebSearch** 工具同时执行以下搜索：

```
搜索1: "[主题关键词] academic research papers peer-reviewed studies"
目的: 获取学术论文线索

搜索2: "[主题关键词] government report official statistics data"
目的: 获取政府报告和官方数据

搜索3: "[主题关键词] mathematical model methodology algorithm"
目的: 获取方法论文献

搜索4: "[主题关键词] case study real-world application"
目的: 获取实际应用案例
```

### 第二轮：深度搜索（基于第一轮结果）

根据第一轮搜索结果，进行针对性深度搜索：

```
搜索5: "[高引用论文作者] other publications related work"
目的: 追踪领域权威作者的其他工作

搜索6: "[相关政府机构] [主题] report publication"
目的: 获取特定机构的专题报告
例如: "World Bank renewable energy report 2024"
      "EPA climate change policy document"
      "UN sustainable development goals report"

搜索7: "[主题] dataset statistics official source"
目的: 确认官方数据来源可引用性
```

### 第三轮：补充搜索（确保多样性）

检查当前引用覆盖情况，针对缺失类别进行补充搜索：

```
检查清单:
- [ ] 学术论文 (至少3篇) → 若不足，搜索: "[主题] journal article systematic review"
- [ ] 政府报告 (至少1篇) → 若不足，搜索: "[主题] government white paper policy report"
- [ ] 官方数据 (至少1个) → 若不足，搜索: "[主题] World Bank UN OECD data indicator"
- [ ] 题目引用 (必须) → 确保引用COMAP题目说明
- [ ] 其他来源 (可选) → 可搜索: "[主题] industry report technical documentation"
```

## MCP工具使用指南

### 使用 context7 MCP服务器

当需要获取库/框架的官方文档时：

```
调用 context7 MCP 工具:
1. 搜索相关库的最新文档
2. 获取API使用说明
3. 获取方法论的官方描述
```

**适用场景：**
- 使用Python科学计算库（NumPy, SciPy, pandas）
- 使用优化求解器（Gurobi, CPLEX, PuLP）
- 使用机器学习框架（scikit-learn, PyTorch）

### MCP调用示例

```
当论文使用了特定算法库时：
1. 调用 context7 搜索 "[库名] documentation"
2. 获取官方引用方式（如BibTeX）
3. 添加到参考文献
```

## 搜索结果处理

### 提取可引用信息

从每个搜索结果中提取：
1. **论文/报告标题**
2. **作者/机构**
3. **发布年份**
4. **DOI或URL**
5. **数据源（如适用）**

### 生成BibTeX条目

```bibtex
% 学术论文
@article{author2024topic,
  author = {Author, First and Author, Second},
  title = {Paper Title from Search},
  journal = {Journal Name},
  year = {2024},
  doi = {10.xxxx/xxx}
}

% 政府报告
@techreport{worldbank2024report,
  author = {{World Bank}},
  title = {Report Title},
  institution = {World Bank Group},
  year = {2024},
  url = {https://...}
}

% 官方数据
@online{undata2024indicator,
  author = {{United Nations}},
  title = {UN Data: Indicator Name},
  year = {2024},
  url = {https://data.un.org/...},
  urldate = {2024-01-15}
}
```

## 搜索策略最佳实践

### 关键词扩展策略

```python
def expand_search_keywords(base_keywords, problem_domain):
    """扩展搜索关键词以提高覆盖率"""
    expanded = []
    
    # 学术搜索关键词
    academic_suffixes = [
        "mathematical model",
        "optimization algorithm", 
        "simulation study",
        "quantitative analysis",
        "systematic review"
    ]
    
    # 政府报告关键词
    government_keywords = [
        "government report",
        "policy document",
        "official statistics",
        "white paper",
        "regulatory analysis"
    ]
    
    # 数据源关键词
    data_keywords = [
        "World Bank data",
        "UN statistics",
        "OECD indicators",
        "official dataset",
        "public data"
    ]
    
    for kw in base_keywords:
        for suffix in academic_suffixes:
            expanded.append(f"{kw} {suffix}")
        for gov in government_keywords:
            expanded.append(f"{kw} {gov}")
        for data in data_keywords:
            expanded.append(f"{kw} {data}")
    
    return expanded
```

### 领域特定搜索指南

| 题目领域 | 推荐搜索机构 | 关键数据源 |
|----------|-------------|-----------|
| 能源/环境 | DOE, EPA, IEA | EIA, World Bank |
| 经济/金融 | IMF, World Bank, Fed | FRED, BLS |
| 健康/医疗 | CDC, WHO, NIH | PubMed, UN |
| 社会/政策 | Census, OECD | UN Data, Eurostat |
| 交通/物流 | DOT, FHWA | BTS, TRB |

## 执行检查清单

在完成搜索后，确认以下事项：

```
✓ 引用多样性检查
  - [ ] 学术论文数量 >= 3
  - [ ] 政府/官方报告数量 >= 1
  - [ ] 官方数据源引用 >= 1
  - [ ] 题目引用已包含
  - [ ] 总引用数量在 8-15 之间

✓ 引用质量检查
  - [ ] 所有DOI已验证
  - [ ] URL可访问
  - [ ] 年份合理（近5年优先）
  - [ ] 作者信息完整

✓ 覆盖面检查
  - [ ] 覆盖至少4个不同类别
  - [ ] 多样性评分 >= 0.75
```

## 错误处理

### 搜索无结果时

```
1. 尝试更宽泛的关键词
2. 使用同义词替换
3. 移除限定词（如年份范围）
4. 切换到相关领域搜索
```

### API限制时

```
1. 等待后重试
2. 切换到备选数据源
3. 使用WebSearch作为后备
```

## 输出格式

执行完所有搜索后，输出以下结构化结果：

```json
{
  "search_summary": {
    "total_searches_performed": 8,
    "successful_searches": 7,
    "failed_searches": 1
  },
  "citations_found": {
    "academic_papers": [
      {"title": "...", "doi": "...", "source": "websearch"}
    ],
    "government_reports": [...],
    "official_data": [...],
    "problem_references": [...],
    "other_sources": [...]
  },
  "diversity_check": {
    "categories_covered": 5,
    "total_citations": 12,
    "diversity_score": 0.85,
    "status": "pass"
  },
  "recommendations": []
}
```

## 相关技能

- `deep-reference-searcher` - 多源文献搜索
- `literature-searcher` - 基础文献检索
- `citation-diversity-validator` - 多样性验证
- `problem-reference-extractor` - 题目引用提取

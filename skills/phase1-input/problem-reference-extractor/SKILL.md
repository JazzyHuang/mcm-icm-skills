---
name: problem-reference-extractor
description: 题目信息引用提取器。从美赛题目PDF/文本中自动提取所有提到的数据源、背景资料、网站链接等信息，并生成规范的BibTeX引用条目。确保论文引用包含题目附带的所有相关信息源。
---

# 题目信息引用提取器 (Problem Reference Extractor)

## 功能概述

自动从MCM/ICM题目中提取所有可引用的信息源，包括：
- 题目本身（Problem Statement）
- 附带的数据文件（CSV、Excel等）
- 题目中提到的网站和数据源
- 背景资料和参考文献
- 附加的背景说明文档

## 重要性

美赛评委期望看到参赛者充分利用题目提供的所有信息。引用题目附带的数据源和背景资料：
1. 展示对题目的深入理解
2. 体现信息搜集的完整性
3. 确保数据来源的可追溯性
4. 满足引用多样性要求

## 提取范围

### 1. 题目本身

```bibtex
@misc{mcm2026problemX,
  author = {{COMAP}},
  title = {{2026 MCM Problem X: Problem Title}},
  year = {2026},
  howpublished = {\url{https://www.contest.comap.com/undergraduate/contests/mcm/}},
  note = {Mathematical Contest in Modeling - Official Problem Statement}
}
```

### 2. 附带数据文件

```bibtex
@misc{mcm2026dataX,
  author = {{COMAP}},
  title = {{2026 MCM Problem X Dataset}},
  year = {2026},
  howpublished = {Provided dataset file: 2026\_MCM\_Problem\_X\_Data.csv},
  note = {Official contest dataset provided with problem statement}
}
```

### 3. 题目中提到的外部数据源

常见的外部数据源包括：
- 政府数据网站（如 data.gov, census.gov）
- 国际组织数据（如 World Bank, UN Data）
- 专业数据库（如 Kaggle, GitHub）
- 新闻来源（如提供的新闻链接）

### 4. 背景资料引用

题目中可能提到的背景资料：
- 科学文献或报告
- 新闻报道
- 技术标准
- 历史数据来源

## 提取流程

```
1. 解析题目文本
   ├── 提取题目标题和类型
   ├── 识别数据文件引用
   └── 标记URL和数据源提及

2. 识别外部引用
   ├── 检测URL模式
   ├── 识别数据源名称（World Bank, UN等）
   └── 提取引用的报告/文献名称

3. 分类处理
   ├── 题目引用（必需）
   ├── 数据文件引用（如有）
   ├── 外部数据源引用
   └── 背景文献引用

4. 生成BibTeX
   ├── 选择合适的条目类型
   ├── 填充必要字段
   └── 生成唯一的bibtex_key
```

## 输出格式

### JSON结构

```json
{
  "problem_info": {
    "year": 2026,
    "type": "A",
    "title": "Problem Title",
    "url": "https://www.contest.comap.com/..."
  },
  "extracted_references": {
    "problem_statement": {
      "bibtex_key": "mcm2026problema",
      "bibtex": "@misc{...}"
    },
    "data_files": [
      {
        "filename": "2026_MCM_Problem_A_Data.csv",
        "bibtex_key": "mcm2026dataa",
        "bibtex": "@misc{...}"
      }
    ],
    "external_sources": [
      {
        "name": "World Bank Open Data",
        "url": "https://data.worldbank.org/...",
        "bibtex_key": "worldbank2026data",
        "bibtex": "@online{...}"
      }
    ],
    "background_references": [
      {
        "title": "Referenced Report Title",
        "source": "Institution Name",
        "bibtex_key": "institution2024report",
        "bibtex": "@techreport{...}"
      }
    ]
  },
  "total_citations": 4,
  "bibtex_file_content": "..."
}
```

### BibTeX输出示例

```bibtex
% ===========================================
% Problem Statement References
% Extracted from MCM/ICM 2026 Problem A
% ===========================================

% Official Problem Statement
@misc{mcm2026problema,
  author = {{COMAP}},
  title = {{2026 MCM Problem A: Optimizing Solar Panel Placement}},
  year = {2026},
  howpublished = {\url{https://www.contest.comap.com/undergraduate/contests/mcm/}},
  note = {Mathematical Contest in Modeling - Official Problem Statement}
}

% Provided Dataset
@misc{mcm2026dataa,
  author = {{COMAP}},
  title = {{2026 MCM Problem A Dataset: Solar Radiation Data}},
  year = {2026},
  howpublished = {Provided dataset file: 2026\_MCM\_Problem\_A\_Data.csv},
  note = {Official contest dataset containing solar radiation measurements}
}

% External Data Source Mentioned in Problem
@online{nrel2026solar,
  author = {{National Renewable Energy Laboratory}},
  title = {{NREL Solar Resource Data}},
  year = {2026},
  url = {https://www.nrel.gov/gis/solar.html},
  urldate = {2026-01-30},
  note = {Referenced in problem statement as supplementary data source}
}

% Background Report Mentioned
@techreport{iea2025solar,
  author = {{International Energy Agency}},
  title = {{Solar PV Global Supply Chains Report}},
  institution = {IEA},
  year = {2025},
  url = {https://www.iea.org/reports/solar-pv-global-supply-chains},
  note = {Background report referenced in problem statement}
}
```

## 使用脚本

```python
from scripts.reference_extractor import ProblemReferenceExtractor

extractor = ProblemReferenceExtractor()

# 从题目文本提取引用
results = extractor.extract_from_text(
    problem_text="2026 MCM Problem A: Solar Panel Optimization...",
    problem_type="A",
    year=2026,
    data_files=["2026_MCM_Problem_A_Data.csv"]
)

# 从解析后的题目结构提取
results = extractor.extract_from_parsed(
    parsed_problem={
        "problem_type": "A",
        "problem_title": "Solar Panel Optimization",
        "background": "...",
        "provided_data": {"files": ["data.csv"]},
        "keywords": ["solar", "optimization"]
    },
    year=2026
)

# 导出BibTeX
extractor.export_bibtex("problem_references.bib")
```

## 识别模式

### URL识别

```python
URL_PATTERNS = [
    r'https?://[^\s<>"]+',
    r'www\.[^\s<>"]+',
    r'data\.worldbank\.org[^\s]*',
    r'data\.un\.org[^\s]*'
]
```

### 数据源关键词

```python
DATA_SOURCE_KEYWORDS = {
    'World Bank': ['World Bank', 'worldbank', 'data.worldbank.org'],
    'United Nations': ['UN Data', 'United Nations', 'data.un.org'],
    'OECD': ['OECD', 'oecd.org'],
    'US Census': ['Census Bureau', 'census.gov'],
    'EPA': ['EPA', 'Environmental Protection Agency'],
    'CDC': ['CDC', 'Centers for Disease Control'],
    'NASA': ['NASA', 'nasa.gov'],
    'NOAA': ['NOAA', 'National Oceanic'],
    'Kaggle': ['Kaggle', 'kaggle.com'],
    'GitHub': ['GitHub', 'github.com']
}
```

### 报告/文献识别

```python
REPORT_PATTERNS = [
    r'according to (?:the )?(.+?) report',
    r'(?:the )?(.+?) published by',
    r'data from (?:the )?(.+)',
    r'(?:as reported|as stated) (?:by|in) (.+)'
]
```

## 特殊处理

### 多个数据文件

当题目提供多个数据文件时：

```bibtex
@misc{mcm2026dataa1,
  title = {{2026 MCM Problem A Dataset 1: Historical Data}},
  ...
}

@misc{mcm2026dataa2,
  title = {{2026 MCM Problem A Dataset 2: Geographic Data}},
  ...
}
```

### 附加背景文档

某些题目附带PDF背景说明：

```bibtex
@misc{mcm2026backgrounda,
  author = {{COMAP}},
  title = {{2026 MCM Problem A: Background Information}},
  year = {2026},
  howpublished = {Supplementary PDF document provided with problem},
  note = {Additional background information for Problem A}
}
```

## 验证检查

### 必需引用检查

```
✓ 题目声明引用（Problem Statement） - 必需
✓ 数据文件引用（如提供） - 必需
✓ 明确提到的外部数据源 - 推荐
✓ 背景文献（如提到） - 推荐
```

### 输出质量检查

```
✓ BibTeX语法正确
✓ 特殊字符已转义
✓ URL格式正确
✓ 年份信息准确
✓ bibtex_key唯一且有意义
```

## 相关技能

- `problem-parser` - 题目解析
- `deep-reference-searcher` - 深度文献搜索
- `citation-validator` - 引用验证
- `citation-manager` - 引用管理

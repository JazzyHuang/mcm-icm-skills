---
name: citation-diversity-validator
description: 引用多样性验证器。验证论文引用是否覆盖足够多的信息源类别（学术论文、政府报告、官方数据、题目引用等），计算多样性评分，确保满足美赛O奖级别的引用要求。
---

# 引用多样性验证器 (Citation Diversity Validator)

## 功能概述

验证论文引用的多样性，确保：
1. 覆盖至少4个不同的信息源类别
2. 多样性评分达到0.75以上
3. 各类别满足最低引用数量要求
4. 提供补充建议以改善多样性

## 验证标准

### 引用类别

| 类别 | 最低要求 | 权重 | 说明 |
|------|----------|------|------|
| 学术论文 | 3篇 | 40% | 期刊、会议、预印本 |
| 政府报告 | 1篇 | 15% | 官方机构发布的报告 |
| 官方数据 | 1个 | 15% | World Bank、UN等数据源 |
| 题目引用 | 1个 | 15% | Problem Statement及数据 |
| 其他来源 | 0个 | 15% | 新闻、技术文档等 |

### 多样性评分计算

```
基础分 = Σ(类别权重 × 是否有该类别引用)
奖励分 = Σ(类别权重 × 0.2 × 是否超过最低要求)
总分 = min(基础分 + 奖励分, 1.0)
```

### 通过标准

```json
{
  "min_total_citations": 8,
  "max_total_citations": 15,
  "min_categories": 4,
  "min_diversity_score": 0.75,
  "required_categories": ["academic_papers", "problem_references"]
}
```

## 输入格式

### 引用列表格式

```json
{
  "citations": [
    {
      "bibtex_key": "smith2024optimization",
      "title": "Optimization Methods...",
      "category": "academic",
      "source": "semantic_scholar",
      "year": 2024,
      "doi": "10.1234/xxx"
    },
    {
      "bibtex_key": "worldbank2024report",
      "title": "Global Energy Report",
      "category": "government",
      "source": "websearch",
      "year": 2024,
      "url": "https://..."
    }
  ]
}
```

### 支持的类别标签

- `academic` → academic_papers
- `government` → government_reports
- `data` → official_data
- `problem` → problem_references
- `media`, `technical`, `other` → other_sources

## 输出格式

### 验证结果

```json
{
  "validation_result": {
    "overall_status": "pass",
    "diversity_score": 0.82,
    "categories_covered": 5,
    "total_citations": 12
  },
  "category_details": {
    "academic_papers": {
      "count": 5,
      "required": 3,
      "status": "pass",
      "citations": ["smith2024...", "jones2023..."]
    },
    "government_reports": {
      "count": 2,
      "required": 1,
      "status": "pass",
      "citations": ["worldbank2024...", "un2024..."]
    },
    "official_data": {
      "count": 2,
      "required": 1,
      "status": "pass",
      "citations": ["worldbank_data...", "oecd_data..."]
    },
    "problem_references": {
      "count": 1,
      "required": 1,
      "status": "pass",
      "citations": ["mcm2024problema"]
    },
    "other_sources": {
      "count": 2,
      "required": 0,
      "status": "pass",
      "citations": ["github2024...", "news2024..."]
    }
  },
  "recommendations": [],
  "warnings": []
}
```

### 失败时的输出

```json
{
  "validation_result": {
    "overall_status": "fail",
    "diversity_score": 0.55,
    "categories_covered": 2,
    "total_citations": 6
  },
  "category_details": {
    "academic_papers": {
      "count": 5,
      "required": 3,
      "status": "pass"
    },
    "government_reports": {
      "count": 0,
      "required": 1,
      "status": "fail"
    },
    "official_data": {
      "count": 0,
      "required": 1,
      "status": "fail"
    },
    "problem_references": {
      "count": 1,
      "required": 1,
      "status": "pass"
    },
    "other_sources": {
      "count": 0,
      "required": 0,
      "status": "pass"
    }
  },
  "recommendations": [
    {
      "priority": "high",
      "category": "government_reports",
      "message": "需要至少1篇政府/官方报告。建议搜索：'[主题] government report official'",
      "search_suggestions": [
        "[主题] World Bank report",
        "[主题] UN publication",
        "[主题] government white paper"
      ]
    },
    {
      "priority": "high",
      "category": "official_data",
      "message": "需要至少1个官方数据源引用。检查是否使用了World Bank或UN Data的数据。",
      "search_suggestions": [
        "确认data-collector获取的数据已生成引用",
        "检查题目是否提供了数据文件"
      ]
    },
    {
      "priority": "medium",
      "message": "当前总引用数为6，建议增加到8-15篇以展示广泛的信息搜集能力。"
    }
  ],
  "warnings": [
    "多样性评分(0.55)低于最低要求(0.75)",
    "仅覆盖2个类别，需要至少4个类别"
  ]
}
```

## 使用方式

```python
from scripts.diversity_validator import CitationDiversityValidator

validator = CitationDiversityValidator()

# 验证引用列表
result = validator.validate(citations=[
    {"bibtex_key": "smith2024", "category": "academic", ...},
    {"bibtex_key": "worldbank2024", "category": "government", ...},
    ...
])

# 检查是否通过
if result['validation_result']['overall_status'] == 'pass':
    print("引用多样性验证通过")
else:
    print("需要补充引用:")
    for rec in result['recommendations']:
        print(f"  - {rec['message']}")
```

## 自动补充建议

### 按类别的搜索建议

```python
SEARCH_SUGGESTIONS = {
    'government_reports': [
        "[主题] government report official statistics",
        "[主题] World Bank publication",
        "[主题] UN report policy",
        "[主题] OECD analysis"
    ],
    'official_data': [
        "确保data-collector已为获取的数据生成引用",
        "[主题] World Bank data indicator",
        "[主题] UN statistics database",
        "[主题] official government data"
    ],
    'problem_references': [
        "必须引用MCM/ICM官方题目声明",
        "如有数据文件，需引用官方提供的数据集"
    ],
    'academic_papers': [
        "[主题] peer-reviewed journal article",
        "[主题] systematic review meta-analysis",
        "[主题] mathematical modeling study"
    ]
}
```

## 验证流程

```
1. 分类统计
   ├── 解析所有引用的category字段
   ├── 将引用分配到5个类别
   └── 统计每个类别的数量

2. 基础检查
   ├── 检查总引用数量 (8-15)
   ├── 检查必需类别是否存在
   └── 检查各类别最低要求

3. 多样性评分
   ├── 计算基础分（类别覆盖）
   ├── 计算奖励分（超额引用）
   └── 生成总评分

4. 生成报告
   ├── 汇总验证结果
   ├── 生成改进建议
   └── 标记警告信息
```

## 集成到工作流

### 在阶段1结束时调用

```markdown
阶段1完成检查：
1. 收集所有生成的引用
   - deep-reference-searcher 输出的学术引用
   - problem-reference-extractor 输出的题目引用
   - data-collector 输出的数据源引用
   - ai-deep-search-guide 输出的补充引用

2. 调用 citation-diversity-validator 验证

3. 如果验证失败：
   - 根据建议执行补充搜索
   - 重新验证直到通过
```

### 在阶段7引用管理时再次调用

```markdown
最终验证：
1. 收集论文中实际使用的所有引用
2. 验证多样性
3. 检查未使用的引用
4. 生成最终引用报告
```

## 配置选项

```yaml
# citation_diversity_config.yaml
validation_rules:
  min_total_citations: 8
  max_total_citations: 15
  min_categories: 4
  min_diversity_score: 0.75

category_requirements:
  academic_papers:
    min: 3
    weight: 0.40
  government_reports:
    min: 1
    weight: 0.15
  official_data:
    min: 1
    weight: 0.15
  problem_references:
    min: 1
    weight: 0.15
    required: true  # 必须有
  other_sources:
    min: 0
    weight: 0.15

# 严格模式：所有最低要求必须满足
strict_mode: true
```

## 相关技能

- `deep-reference-searcher` - 多源文献搜索
- `ai-deep-search-guide` - AI深度搜索引导
- `problem-reference-extractor` - 题目引用提取
- `citation-manager` - 引用管理
- `citation-validator` - 引用真实性验证

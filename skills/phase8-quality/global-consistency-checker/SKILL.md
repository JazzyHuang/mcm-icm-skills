---
name: global-consistency-checker
description: 全局一致性验证器，检查跨章节的术语一致性、数据引用一致性、符号使用一致性、图表编号一致性，确保论文整体的逻辑和表述统一。
dependencies: [section-writer, latex-compiler, citation-manager]
outputs: [consistency_report, inconsistencies, corrections, consistency_score]
---

# 全局一致性验证器 (Global Consistency Checker)

## 概述

论文的一致性是专业性的重要体现。本技能检查全文的术语、数据、符号、图表引用等的一致性，确保论文整体逻辑统一、表述规范。

## 核心原则

### 一致性的重要性

```
不一致问题会导致:
1. 降低可信度: 评委质疑数据和分析的可靠性
2. 阅读困难: 读者难以跟踪概念和变量
3. 扣分风险: 明显的不一致可能导致降级
4. 专业性质疑: 暗示论文准备不充分
```

### 需要检查的一致性类型

| 类型 | 检查项 | 严重程度 |
|------|--------|---------|
| 术语 | 同一概念使用相同术语 | 高 |
| 数据 | 同一数值在全文一致 | 高 |
| 符号 | 变量符号定义和使用一致 | 高 |
| 图表 | 编号连续、引用正确 | 中 |
| 缩写 | 首次出现时定义，后续一致使用 | 中 |
| 格式 | 数字格式、单位表示一致 | 低 |

## 一致性检查维度

### 1. 术语一致性 (Terminology Consistency)

```python
def check_terminology_consistency(document):
    """检查术语一致性"""
    
    issues = []
    
    # 检测同义词/近义词的混用
    TERM_GROUPS = {
        'model': ['model', 'framework', 'approach', 'method', 'algorithm'],
        'accuracy': ['accuracy', 'precision', 'correctness'],
        'data': ['data', 'dataset', 'samples', 'observations'],
        'result': ['result', 'outcome', 'finding', 'output'],
        'parameter': ['parameter', 'hyperparameter', 'coefficient', 'variable']
    }
    
    for concept, terms in TERM_GROUPS.items():
        used_terms = find_used_terms(document, terms)
        
        if len(used_terms) > 1:
            # 检查是否有明确区分
            if not has_explicit_distinction(document, used_terms):
                issues.append({
                    'type': 'terminology',
                    'concept': concept,
                    'terms_used': used_terms,
                    'suggestion': f"Consider using '{used_terms[0]}' consistently throughout",
                    'severity': 'medium'
                })
    
    # 检测专业术语拼写变体
    SPELLING_VARIANTS = {
        'optimization': ['optimization', 'optimisation'],
        'modeling': ['modeling', 'modelling'],
        'behavior': ['behavior', 'behaviour'],
        'analyze': ['analyze', 'analyse']
    }
    
    for term, variants in SPELLING_VARIANTS.items():
        used_variants = find_used_terms(document, variants)
        if len(used_variants) > 1:
            issues.append({
                'type': 'spelling_variant',
                'term': term,
                'variants_used': used_variants,
                'suggestion': f"Use consistent spelling: '{used_variants[0]}'",
                'severity': 'low'
            })
    
    return issues
```

### 2. 数据一致性 (Data Consistency)

```python
def check_data_consistency(document):
    """检查数据一致性"""
    
    issues = []
    
    # 提取所有数值及其上下文
    numbers = extract_numbers_with_context(document)
    
    # 按语义分组
    grouped_numbers = group_by_semantic_context(numbers)
    
    for context, values in grouped_numbers.items():
        if len(set(values)) > 1:
            issues.append({
                'type': 'data_inconsistency',
                'context': context,
                'values_found': list(set(values)),
                'locations': find_locations(document, values),
                'suggestion': f"Same metric '{context}' has different values",
                'severity': 'high'
            })
    
    # 检查表格数据与正文数据一致性
    table_data = extract_table_data(document)
    text_data = extract_text_data(document)
    
    for key in set(table_data.keys()) & set(text_data.keys()):
        if table_data[key] != text_data[key]:
            issues.append({
                'type': 'table_text_mismatch',
                'metric': key,
                'table_value': table_data[key],
                'text_value': text_data[key],
                'suggestion': 'Ensure table and text values match',
                'severity': 'high'
            })
    
    # 检查摘要数据与正文数据一致性
    abstract_data = extract_abstract_data(document)
    body_data = extract_body_data(document)
    
    for key in set(abstract_data.keys()) & set(body_data.keys()):
        if abstract_data[key] != body_data[key]:
            issues.append({
                'type': 'abstract_body_mismatch',
                'metric': key,
                'abstract_value': abstract_data[key],
                'body_value': body_data[key],
                'suggestion': 'Abstract and body should report same values',
                'severity': 'high'
            })
    
    return issues
```

### 3. 符号一致性 (Symbol Consistency)

```python
def check_symbol_consistency(document):
    """检查数学符号一致性"""
    
    issues = []
    
    # 提取符号定义
    symbol_definitions = extract_symbol_definitions(document)
    
    # 提取符号使用
    symbol_usages = extract_symbol_usages(document)
    
    # 检查未定义的符号
    for symbol in symbol_usages:
        if symbol not in symbol_definitions:
            issues.append({
                'type': 'undefined_symbol',
                'symbol': symbol,
                'locations': symbol_usages[symbol],
                'suggestion': f"Symbol '{symbol}' used but not defined",
                'severity': 'high'
            })
    
    # 检查符号重定义
    for symbol, definitions in symbol_definitions.items():
        if len(definitions) > 1:
            meanings = [d['meaning'] for d in definitions]
            if len(set(meanings)) > 1:
                issues.append({
                    'type': 'symbol_redefinition',
                    'symbol': symbol,
                    'definitions': definitions,
                    'suggestion': f"Symbol '{symbol}' has multiple definitions",
                    'severity': 'high'
                })
    
    # 检查上下标一致性
    subscript_patterns = extract_subscript_patterns(document)
    for pattern, instances in subscript_patterns.items():
        if not is_consistent(instances):
            issues.append({
                'type': 'subscript_inconsistency',
                'pattern': pattern,
                'instances': instances,
                'suggestion': 'Use consistent subscript notation',
                'severity': 'medium'
            })
    
    # 检查向量/矩阵表示法一致性
    notation_styles = check_vector_matrix_notation(document)
    if len(notation_styles) > 1:
        issues.append({
            'type': 'notation_inconsistency',
            'styles_found': notation_styles,
            'suggestion': 'Use consistent notation for vectors/matrices',
            'severity': 'medium'
        })
    
    return issues
```

### 4. 图表引用一致性 (Figure/Table Reference Consistency)

```python
def check_figure_table_consistency(document):
    """检查图表引用一致性"""
    
    issues = []
    
    # 提取所有图表
    figures = extract_figures(document)
    tables = extract_tables(document)
    
    # 提取所有引用
    figure_refs = extract_figure_references(document)
    table_refs = extract_table_references(document)
    
    # 检查编号连续性
    figure_numbers = [f['number'] for f in figures]
    if not is_consecutive(figure_numbers):
        issues.append({
            'type': 'non_consecutive_figures',
            'numbers': figure_numbers,
            'suggestion': 'Figure numbers should be consecutive',
            'severity': 'medium'
        })
    
    table_numbers = [t['number'] for t in tables]
    if not is_consecutive(table_numbers):
        issues.append({
            'type': 'non_consecutive_tables',
            'numbers': table_numbers,
            'suggestion': 'Table numbers should be consecutive',
            'severity': 'medium'
        })
    
    # 检查引用的图表是否存在
    for ref in figure_refs:
        if ref['number'] not in figure_numbers:
            issues.append({
                'type': 'invalid_figure_reference',
                'reference': ref,
                'suggestion': f"Figure {ref['number']} referenced but not found",
                'severity': 'high'
            })
    
    for ref in table_refs:
        if ref['number'] not in table_numbers:
            issues.append({
                'type': 'invalid_table_reference',
                'reference': ref,
                'suggestion': f"Table {ref['number']} referenced but not found",
                'severity': 'high'
            })
    
    # 检查未引用的图表
    referenced_figures = set(ref['number'] for ref in figure_refs)
    for fig in figures:
        if fig['number'] not in referenced_figures:
            issues.append({
                'type': 'unreferenced_figure',
                'figure': fig['number'],
                'suggestion': f"Figure {fig['number']} not referenced in text",
                'severity': 'medium'
            })
    
    return issues
```

### 5. 缩写一致性 (Abbreviation Consistency)

```python
def check_abbreviation_consistency(document):
    """检查缩写一致性"""
    
    issues = []
    
    # 提取缩写定义（首次出现时的完整形式）
    abbreviations = extract_abbreviation_definitions(document)
    
    # 检查是否首次使用时定义
    for abbr, info in abbreviations.items():
        if info['first_use_position'] < info['definition_position']:
            issues.append({
                'type': 'abbreviation_before_definition',
                'abbreviation': abbr,
                'suggestion': f"'{abbr}' used before definition",
                'severity': 'medium'
            })
    
    # 检查是否在摘要中重新定义
    abstract_abbrs = extract_abbreviations_from_section(document, 'abstract')
    body_abbrs = extract_abbreviations_from_section(document, 'body')
    
    for abbr in abstract_abbrs:
        if abbr in body_abbrs:
            # 摘要和正文都需要定义
            pass  # 这是允许的
    
    # 检查缩写使用一致性
    for abbr, info in abbreviations.items():
        # 检查是否有时用全称有时用缩写
        full_form_after_def = count_full_form_after_definition(
            document, abbr, info['full_form'], info['definition_position']
        )
        if full_form_after_def > 0:
            issues.append({
                'type': 'inconsistent_abbreviation_use',
                'abbreviation': abbr,
                'full_form_count': full_form_after_def,
                'suggestion': f"After defining '{abbr}', use it consistently instead of full form",
                'severity': 'low'
            })
    
    return issues
```

### 6. 格式一致性 (Format Consistency)

```python
def check_format_consistency(document):
    """检查格式一致性"""
    
    issues = []
    
    # 数字格式
    number_formats = analyze_number_formats(document)
    if len(number_formats) > 1:
        issues.append({
            'type': 'number_format_inconsistency',
            'formats_found': number_formats,
            'suggestion': 'Use consistent number formatting (e.g., 1,000 vs 1000)',
            'severity': 'low'
        })
    
    # 百分比格式
    percentage_formats = analyze_percentage_formats(document)
    if len(percentage_formats) > 1:
        issues.append({
            'type': 'percentage_format_inconsistency',
            'formats_found': percentage_formats,
            'suggestion': 'Use consistent percentage format (e.g., 50% vs 0.5)',
            'severity': 'low'
        })
    
    # 单位格式
    unit_formats = analyze_unit_formats(document)
    if has_inconsistent_units(unit_formats):
        issues.append({
            'type': 'unit_format_inconsistency',
            'details': unit_formats,
            'suggestion': 'Use consistent unit formatting',
            'severity': 'low'
        })
    
    # 引用格式
    citation_formats = analyze_citation_formats(document)
    if len(citation_formats) > 1:
        issues.append({
            'type': 'citation_format_inconsistency',
            'formats_found': citation_formats,
            'suggestion': 'Use consistent citation format',
            'severity': 'medium'
        })
    
    return issues
```

## 检查流程

```python
def run_global_consistency_check(document):
    """
    执行全局一致性检查
    
    Args:
        document: 论文文档对象
    
    Returns:
        dict: 一致性检查报告
    """
    report = {
        'issues': [],
        'issue_counts': {},
        'consistency_score': 0.0,
        'sections_checked': [],
        'recommendations': []
    }
    
    # 执行各类检查
    checkers = [
        ('terminology', check_terminology_consistency),
        ('data', check_data_consistency),
        ('symbol', check_symbol_consistency),
        ('figure_table', check_figure_table_consistency),
        ('abbreviation', check_abbreviation_consistency),
        ('format', check_format_consistency)
    ]
    
    for check_type, checker in checkers:
        issues = checker(document)
        report['issues'].extend(issues)
        report['issue_counts'][check_type] = len(issues)
        report['sections_checked'].append(check_type)
    
    # 计算一致性评分
    report['consistency_score'] = calculate_consistency_score(report['issues'])
    
    # 生成建议
    report['recommendations'] = generate_recommendations(report['issues'])
    
    # 按严重程度排序
    report['issues'] = sorted(
        report['issues'], 
        key=lambda x: {'high': 0, 'medium': 1, 'low': 2}[x['severity']]
    )
    
    return report
```

## 输出格式

```json
{
  "consistency_score": 0.87,
  "quality_level": "good",
  "issue_counts": {
    "terminology": 2,
    "data": 1,
    "symbol": 3,
    "figure_table": 1,
    "abbreviation": 2,
    "format": 1
  },
  "issues": [
    {
      "id": "C001",
      "type": "data_inconsistency",
      "severity": "high",
      "description": "RMSE value differs between abstract (0.032) and results section (0.0312)",
      "locations": [
        {"section": "abstract", "line": 15, "value": "0.032"},
        {"section": "results", "line": 234, "value": "0.0312"}
      ],
      "suggestion": "Use consistent value: 0.0312",
      "auto_fix_available": true
    },
    {
      "id": "C002",
      "type": "undefined_symbol",
      "severity": "high",
      "description": "Symbol 'α' used in equation (5) but not defined",
      "locations": [
        {"section": "model", "equation": 5}
      ],
      "suggestion": "Add definition of α in the Notation section",
      "auto_fix_available": false
    },
    {
      "id": "C003",
      "type": "terminology",
      "severity": "medium",
      "description": "Mixed use of 'model' and 'framework' for the same concept",
      "locations": [
        {"section": "introduction", "uses": ["framework", "model"]},
        {"section": "methodology", "uses": ["model"]}
      ],
      "suggestion": "Use 'model' consistently throughout",
      "auto_fix_available": true
    },
    {
      "id": "C004",
      "type": "unreferenced_figure",
      "severity": "medium",
      "description": "Figure 7 not referenced in text",
      "locations": [
        {"section": "results", "figure": 7}
      ],
      "suggestion": "Add reference to Figure 7 in the relevant section",
      "auto_fix_available": false
    }
  ],
  "recommendations": [
    "Fix high-severity data inconsistency in abstract immediately",
    "Define all mathematical symbols before use",
    "Consider creating a consistent terminology guide",
    "Review all figures to ensure they are referenced"
  ],
  "auto_corrections": {
    "available": 3,
    "applied": 0,
    "corrections": [
      {
        "issue_id": "C001",
        "original": "0.032",
        "corrected": "0.0312",
        "location": "abstract, line 15"
      }
    ]
  }
}
```

## 质量评分标准

| 评分 | 等级 | 描述 |
|------|------|------|
| 0.95-1.00 | Excellent | 几乎无一致性问题 |
| 0.85-0.94 | Good | 少量低严重度问题 |
| 0.70-0.84 | Fair | 存在中等问题 |
| 0.50-0.69 | Poor | 多个高严重度问题 |
| <0.50 | Critical | 严重一致性问题 |

**O奖标准**: consistency_score ≥ 0.90

## 与其他技能集成

### 上游依赖
- `section-writer`: 获取各章节内容
- `latex-compiler`: 获取编译后的文档
- `citation-manager`: 获取引用信息

### 下游输出
- `quality-reviewer`: 提供一致性评分
- `final-polisher`: 提供需修正的问题列表
- `submission-preparer`: 确认提交前一致性

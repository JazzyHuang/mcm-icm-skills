---
name: limitation-analyzer
description: 自动识别模型的局限性和假设限制，生成诚实、学术化的自我批评段落和改进方向建议。
dependencies: [model-builder, assumption-generator, sensitivity-analyzer, error-analyzer]
outputs: [limitations_list, self_critique, improvement_suggestions, limitations_section]
---

# 模型局限性分析器 (Limitation Analyzer)

## 概述

O奖论文的显著特点之一是"诚实的自我批评"。本技能自动识别模型的局限性、假设的限制条件，并生成学术化的自我批评段落，展示学术成熟度。

## 核心原则

### 为什么需要讨论局限性

```
1. 学术诚信: 展示对方法局限的清醒认识
2. 可信度: 诚实承认局限比掩盖更有说服力
3. 学术成熟: 表明作者具备批判性思维能力
4. 改进方向: 为后续研究指明方向
```

### O奖论文局限性讨论特点

- ✅ 诚实承认局限，但有分寸
- ✅ 解释局限的原因和影响
- ✅ 讨论在什么条件下局限性更显著
- ✅ 提出具体的改进方向
- ❌ 避免: 过度自贬、避重就轻、泛泛而谈

## 局限性类型分类

### 1. 数据相关局限

```python
DATA_LIMITATIONS = {
    'data_scarcity': {
        'description': '数据量不足',
        'template': 'Due to limited availability of {data_type} data, our model was trained on {n} samples, which may not fully capture {phenomenon}.',
        'impact': '可能影响模型泛化能力',
        'mitigation': '未来可收集更多数据或使用数据增强技术'
    },
    'data_quality': {
        'description': '数据质量问题',
        'template': 'The {data_source} data contains {quality_issue}, which may introduce {effect} into our predictions.',
        'impact': '可能引入系统性偏差',
        'mitigation': '未来可使用更高质量的数据源'
    },
    'temporal_limitation': {
        'description': '时间范围限制',
        'template': 'Our analysis is based on data from {start_year} to {end_year}, which may not reflect {long_term_pattern}.',
        'impact': '可能无法捕捉长期趋势',
        'mitigation': '可扩展历史数据或使用迁移学习'
    },
    'spatial_limitation': {
        'description': '空间范围限制',
        'template': 'The model was developed for {region}, and its applicability to {other_regions} requires further validation.',
        'impact': '地理泛化能力未验证',
        'mitigation': '需要在其他地区进行验证'
    },
    'missing_variables': {
        'description': '缺失变量',
        'template': 'Important factors such as {missing_factors} were not included due to data unavailability.',
        'impact': '可能遗漏重要影响因素',
        'mitigation': '未来可整合更多数据源'
    }
}
```

### 2. 模型相关局限

```python
MODEL_LIMITATIONS = {
    'simplifying_assumptions': {
        'description': '简化假设',
        'template': 'We assume {assumption}, which simplifies {real_world_complexity}. This may limit accuracy when {condition}.',
        'impact': '在特定条件下精度下降',
        'mitigation': '可通过放松假设或使用更复杂模型'
    },
    'linearity_assumption': {
        'description': '线性假设',
        'template': 'Our model assumes linear relationships between {variables}, which may not capture nonlinear dynamics in {scenario}.',
        'impact': '无法捕捉非线性效应',
        'mitigation': '可使用非线性模型或核方法'
    },
    'stationarity_assumption': {
        'description': '平稳性假设',
        'template': 'The model assumes stationarity, but {phenomenon} may exhibit non-stationary behavior over {time_scale}.',
        'impact': '长期预测可能不可靠',
        'mitigation': '可使用时变参数或自适应方法'
    },
    'independence_assumption': {
        'description': '独立性假设',
        'template': 'We assume independence between {entities}, while in reality {dependency} may exist.',
        'impact': '可能低估协同效应',
        'mitigation': '可使用相关性建模或网络模型'
    },
    'computational_constraints': {
        'description': '计算约束',
        'template': 'Due to computational constraints, we limited {parameter} to {value}, which may affect {aspect}.',
        'impact': '可能未达到全局最优',
        'mitigation': '使用更多计算资源或更高效算法'
    }
}
```

### 3. 验证相关局限

```python
VALIDATION_LIMITATIONS = {
    'limited_validation': {
        'description': '验证范围有限',
        'template': 'Our model was validated on {validation_data}, but its performance on {other_scenarios} remains untested.',
        'impact': '泛化能力未充分验证',
        'mitigation': '需要更广泛的验证'
    },
    'no_external_validation': {
        'description': '缺乏外部验证',
        'template': 'The model has not been validated against independent external datasets.',
        'impact': '存在过拟合风险',
        'mitigation': '使用独立数据集验证'
    },
    'synthetic_validation': {
        'description': '合成数据验证',
        'template': 'Part of the validation was performed on synthetic data, which may not fully represent real-world complexity.',
        'impact': '真实场景表现可能不同',
        'mitigation': '需要真实数据验证'
    }
}
```

### 4. 方法论局限

```python
METHODOLOGY_LIMITATIONS = {
    'parameter_sensitivity': {
        'description': '参数敏感性',
        'template': 'Model performance is sensitive to {parameter}, and optimal values may vary across {contexts}.',
        'impact': '需要针对性调参',
        'mitigation': '开发自适应参数选择方法'
    },
    'interpretability': {
        'description': '可解释性',
        'template': 'The {model_type} nature of our approach limits interpretability, making it difficult to understand {mechanism}.',
        'impact': '难以提供决策解释',
        'mitigation': '结合可解释性技术如SHAP'
    },
    'scalability': {
        'description': '可扩展性',
        'template': 'The computational complexity of O({complexity}) may limit scalability to {large_scale_scenario}.',
        'impact': '大规模应用受限',
        'mitigation': '开发近似算法或分布式版本'
    }
}
```

## 局限性识别流程

```python
def identify_limitations(model_info, assumptions, validation_results, error_analysis):
    """
    自动识别模型局限性
    
    Args:
        model_info: 模型信息
        assumptions: 假设列表
        validation_results: 验证结果
        error_analysis: 误差分析结果
    
    Returns:
        dict: 局限性分析结果
    """
    limitations = []
    
    # 1. 分析假设限制
    for assumption in assumptions:
        limitation = analyze_assumption_limitation(assumption)
        if limitation:
            limitations.append(limitation)
    
    # 2. 分析数据限制
    data_limitations = analyze_data_limitations(model_info['data'])
    limitations.extend(data_limitations)
    
    # 3. 分析模型限制
    model_limitations = analyze_model_limitations(model_info['model_type'])
    limitations.extend(model_limitations)
    
    # 4. 从误差分析中识别限制
    error_based = identify_from_errors(error_analysis)
    limitations.extend(error_based)
    
    # 5. 分析验证限制
    validation_limitations = analyze_validation_gaps(validation_results)
    limitations.extend(validation_limitations)
    
    # 6. 评估严重程度并排序
    limitations = rank_by_severity(limitations)
    
    # 7. 生成改进建议
    for lim in limitations:
        lim['improvement'] = generate_improvement_suggestion(lim)
    
    return limitations
```

### 从假设分析局限

```python
def analyze_assumption_limitation(assumption):
    """分析假设的局限性"""
    
    limitation = {
        'type': 'assumption',
        'assumption': assumption['statement'],
        'justification': assumption.get('justification', ''),
        'when_violated': identify_violation_scenarios(assumption),
        'impact': estimate_violation_impact(assumption),
        'severity': 'medium'  # 根据影响程度调整
    }
    
    # 如果假设在敏感性分析中显示高影响，提高严重程度
    if assumption.get('sensitivity_rank', 10) <= 3:
        limitation['severity'] = 'high'
    
    return limitation
```

### 从误差模式识别局限

```python
def identify_from_errors(error_analysis):
    """从误差模式识别局限性"""
    
    limitations = []
    
    # 检查边界误差
    if error_analysis['error_sources'].get('boundary_effects'):
        limitations.append({
            'type': 'model',
            'category': 'boundary_effects',
            'description': 'Higher prediction errors observed at boundary conditions',
            'evidence': error_analysis['error_sources']['boundary_effects'],
            'severity': 'medium'
        })
    
    # 检查极端值误差
    if error_analysis['error_sources'].get('extreme_values'):
        limitations.append({
            'type': 'model',
            'category': 'extreme_value_handling',
            'description': 'Model performance degrades for extreme input values',
            'evidence': error_analysis['error_sources']['extreme_values'],
            'severity': 'high'
        })
    
    # 检查系统性偏差
    if error_analysis['distribution'].get('has_bias'):
        limitations.append({
            'type': 'model',
            'category': 'systematic_bias',
            'description': f"Systematic {error_analysis['distribution']['bias_direction']} bias detected",
            'evidence': error_analysis['distribution'],
            'severity': 'high'
        })
    
    return limitations
```

## 自我批评段落生成

### 段落结构

```
1. 开头: 承认存在局限性
2. 列举: 具体的局限性（3-5个）
3. 解释: 局限性的原因和影响
4. 条件: 在什么条件下局限性更显著
5. 展望: 改进方向和未来工作
```

### 生成模板

```python
def generate_limitations_section(limitations):
    """生成局限性讨论章节"""
    
    section = """
\\subsection{Limitations and Future Work}

While our model demonstrates promising results, several limitations should be acknowledged:

"""
    
    # 按类型分组
    grouped = group_by_type(limitations)
    
    # 生成各类型局限性讨论
    for lim_type, lims in grouped.items():
        if lims:
            section += generate_limitation_paragraph(lim_type, lims)
    
    # 生成改进方向
    section += generate_future_work(limitations)
    
    return section

def generate_limitation_paragraph(lim_type, limitations):
    """生成单类型局限性段落"""
    
    TYPE_HEADERS = {
        'data': 'Data-related limitations',
        'model': 'Model-related limitations',
        'validation': 'Validation limitations',
        'assumption': 'Assumption-related constraints'
    }
    
    paragraph = f"\\textbf{{{TYPE_HEADERS[lim_type]}}}: "
    
    for i, lim in enumerate(limitations[:3]):  # 最多3个
        if i > 0:
            paragraph += "Additionally, "
        
        paragraph += f"{lim['description']} "
        
        if lim.get('when_violated'):
            paragraph += f"This limitation is particularly relevant when {lim['when_violated']}. "
        
        if lim.get('impact'):
            paragraph += f"This may result in {lim['impact']}. "
    
    paragraph += "\n\n"
    return paragraph
```

## 改进建议生成

```python
def generate_improvement_suggestions(limitations):
    """生成具体的改进建议"""
    
    suggestions = []
    
    for lim in limitations:
        suggestion = {
            'limitation': lim['description'],
            'short_term': generate_short_term_fix(lim),
            'long_term': generate_long_term_improvement(lim),
            'research_direction': suggest_research_direction(lim)
        }
        suggestions.append(suggestion)
    
    return suggestions

def generate_future_work(limitations):
    """生成未来工作段落"""
    
    future_work = """
\\textbf{Future Directions}: Based on the identified limitations, several directions 
for future research emerge:

\\begin{enumerate}
"""
    
    # 提取独特的改进方向
    directions = extract_unique_directions(limitations)
    
    for direction in directions[:4]:  # 最多4个方向
        future_work += f"\\item {direction}\n"
    
    future_work += "\\end{enumerate}\n"
    
    return future_work
```

## 输出格式

```json
{
  "limitations": [
    {
      "id": "L1",
      "type": "data",
      "category": "temporal_limitation",
      "description": "The model was trained on data from 2010-2022, which may not capture longer-term climate patterns.",
      "severity": "medium",
      "when_violated": "predicting beyond 2030",
      "impact": "reduced accuracy for long-term projections",
      "evidence": "Sensitivity analysis shows 15% accuracy drop for 10+ year predictions",
      "improvement": {
        "short_term": "Include confidence intervals that widen for longer projections",
        "long_term": "Integrate paleoclimate data or ensemble projections"
      }
    },
    {
      "id": "L2",
      "type": "assumption",
      "category": "independence_assumption",
      "description": "We assume independence between regional migration flows, while in reality correlated patterns may exist.",
      "severity": "high",
      "when_violated": "analyzing interconnected regions",
      "impact": "may underestimate cascade effects",
      "evidence": "Residual analysis shows spatial correlation (Moran's I = 0.23)",
      "improvement": {
        "short_term": "Add spatial correlation terms",
        "long_term": "Develop network-based migration model"
      }
    },
    {
      "id": "L3",
      "type": "model",
      "category": "interpretability",
      "description": "The neural network components limit interpretability of the decision process.",
      "severity": "medium",
      "when_violated": "explaining predictions to policymakers",
      "impact": "difficult to provide actionable insights",
      "improvement": {
        "short_term": "Apply SHAP analysis for feature importance",
        "long_term": "Develop inherently interpretable hybrid model"
      }
    }
  ],
  "self_critique_paragraph": "While our model demonstrates...",
  "future_work_paragraph": "Based on the identified limitations...",
  "limitations_section_latex": "\\subsection{Limitations and Future Work}...",
  "severity_summary": {
    "high": 1,
    "medium": 2,
    "low": 0
  }
}
```

## LaTeX输出模板

```latex
\subsection{Limitations and Future Work}

While our model demonstrates promising results with 23\% accuracy improvement 
over baselines, several limitations should be acknowledged.

\textbf{Data-related limitations}: The model was trained on data from 2010-2022, 
which may not fully capture longer-term climate patterns. This limitation is 
particularly relevant for projections beyond 2030, where prediction confidence 
decreases. Additionally, the spatial resolution of available climate data 
(0.5° × 0.5°) may miss localized effects important for community-level predictions.

\textbf{Model assumptions}: We assume independence between regional migration flows, 
while in reality correlated patterns may exist, particularly in adjacent regions. 
Residual analysis reveals moderate spatial correlation (Moran's I = 0.23, p < 0.01), 
suggesting this assumption may lead to underestimation of cascade effects.

\textbf{Interpretability}: The neural network components, while improving predictive 
accuracy, limit the interpretability of the decision process. This poses challenges 
when explaining model predictions to policymakers who require actionable insights.

\textbf{Future Directions}: Based on these limitations, several directions for 
future research emerge:
\begin{enumerate}
\item Extending the training data to include paleoclimate records for capturing 
longer-term patterns
\item Developing a network-based migration model that explicitly accounts for 
inter-regional dependencies
\item Integrating SHAP or similar interpretability techniques to provide 
feature-level explanations
\item Validating the model in additional geographic contexts to assess 
transferability
\end{enumerate}
```

## 写作原则

### 应该
- ✅ 诚实承认，但客观描述
- ✅ 提供证据支持（敏感性分析、误差模式）
- ✅ 说明在什么条件下局限更显著
- ✅ 提出具体可行的改进方向
- ✅ 用学术语言表达

### 避免
- ❌ 过度自贬（"our model is fundamentally flawed"）
- ❌ 避重就轻（只说无关紧要的局限）
- ❌ 泛泛而谈（"more data would be helpful"）
- ❌ 责怪外部因素（"the data was bad"）

## 与其他技能集成

### 上游依赖
- `assumption-generator`: 获取假设列表
- `sensitivity-analyzer`: 获取参数敏感性
- `error-analyzer`: 获取误差模式
- `model-builder`: 获取模型信息

### 下游输出
- `section-writer`: 提供局限性章节内容
- `abstract-generator`: 提供自我批评素材
- `quality-reviewer`: 提供完整性检查

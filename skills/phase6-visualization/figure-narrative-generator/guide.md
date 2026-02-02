---
name: figure-narrative-generator
description: 为论文中的每个图表生成专业的叙事文字，确保图表与正文形成有机整体，提升信息传达效率。
dependencies: [chart-generator, section-writer, model-solver]
outputs: [figure_narratives, caption_text, in_text_references, figure_story]
---

# 图表叙事生成器 (Figure Narrative Generator)

## 概述

图表不仅要美观，更要"会讲故事"。本技能为每个图表生成配套的叙事文字，包括图题(Caption)、正文引用句、结果解释，确保图表与正文形成有机整体。

## 核心原则

### 图表叙事三要素

```
1. Caption (图题): 独立说明，不看正文也能理解
2. In-text Reference (正文引用): 自然引入图表
3. Interpretation (解释): 分析图表含义和意义
```

### O奖图表叙事特点

- ✅ Caption完整独立：包含What, How, Key Finding
- ✅ 正文引用自然：不是简单"见图X"
- ✅ 结果解释深入：不仅描述还要分析
- ❌ 避免：图表与正文脱节、解释缺失、重复描述

## 图表类型与叙事策略

### 类型1: 结果展示图 (Result Figures)

**适用**: 模型输出、预测结果、优化解

**Caption模板**:
```
Figure X. [What the figure shows]. [How it was generated/key parameters]. 
[Key finding or pattern]. [Optional: comparison or significance].
```

**示例**:
```
Figure 3. Predicted migration patterns for 2050 under the RCP 8.5 scenario. 
The model was trained on historical data from 2000-2020 and validated against 
2020-2023 observations. Red arrows indicate primary migration corridors, with 
arrow thickness proportional to predicted flow magnitude. The model predicts 
a 340% increase in climate-induced migration compared to the 2020 baseline.
```

**正文引用模板**:
```
[Context sentence]. As shown in Figure X, [key observation]. 
This [supports/suggests/indicates] that [interpretation]. 
Notably, [additional insight from the figure].
```

### 类型2: 对比图 (Comparison Figures)

**适用**: 方法对比、场景对比、时间序列对比

**Caption模板**:
```
Figure X. Comparison of [what is being compared] across [comparison dimensions]. 
[Method used for comparison]. [Key difference or finding].
```

**示例**:
```
Figure 5. Comparison of prediction accuracy between our PINN model and three 
baseline methods across different noise levels. Error bars represent 95% 
confidence intervals from 100 Monte Carlo runs. Our method maintains >90% 
accuracy even at 30% noise, while baseline methods degrade significantly 
beyond 10% noise levels.
```

**正文引用模板**:
```
Figure X compares [subjects] in terms of [metric]. 
[Subject A] demonstrates [behavior], whereas [Subject B] shows [different behavior]. 
The most significant difference occurs when [condition], where [specific comparison].
```

### 类型3: 流程/架构图 (Process/Architecture Figures)

**适用**: 模型架构、算法流程、系统框架

**Caption模板**:
```
Figure X. [Type of diagram] of [what it represents]. 
[Key components or stages]. [Data/information flow direction]. 
[Optional: what makes this architecture unique].
```

**示例**:
```
Figure 2. Architecture of the proposed hybrid PINN-Transformer model. 
The framework consists of three main components: (1) physics encoder 
incorporating conservation laws, (2) transformer-based temporal processor, 
and (3) uncertainty quantification module. Arrows indicate data flow 
direction. The dashed box highlights our novel physics-attention mechanism 
that enforces physical constraints during attention computation.
```

### 类型4: 敏感性分析图 (Sensitivity Analysis Figures)

**适用**: Sobol指数、参数影响、不确定性

**Caption模板**:
```
Figure X. [Type of sensitivity analysis] results for [model name]. 
[Parameters analyzed]. [Method used]. [Key finding about parameter importance].
```

**示例**:
```
Figure 7. Sobol sensitivity indices for the climate migration model. 
First-order indices (blue) and total-order indices (orange) are shown 
for the eight key parameters. Error bars indicate 95% bootstrap confidence 
intervals from 10,000 samples. Temperature change (ΔT) exhibits the highest 
total sensitivity (ST = 0.42), indicating substantial interaction effects 
with other parameters.
```

**正文引用模板**:
```
Sensitivity analysis (Figure X) reveals that [parameter] is the most 
influential factor, accounting for [percentage]% of output variance. 
Interestingly, [parameter] shows high interaction effects (ST - S1 = [value]), 
suggesting [interpretation of interaction].
```

### 类型5: 地理/空间图 (Geographic/Spatial Figures)

**适用**: 地图、空间分布、区域分析

**Caption模板**:
```
Figure X. [Spatial representation] of [what is mapped] for [region/time]. 
[Color/symbol meaning]. [Data source]. [Key spatial pattern].
```

**示例**:
```
Figure 4. Spatial distribution of predicted wildfire risk across California 
for Summer 2025. Risk levels are categorized into five tiers (Very Low to 
Extreme) based on our integrated risk index. Data sources include MODIS 
satellite imagery and historical fire records (2010-2023). The Central 
Valley shows significantly elevated risk compared to coastal regions.
```

## 叙事生成流程

```python
def generate_figure_narrative(figure_info, context):
    """
    为图表生成完整叙事
    
    Args:
        figure_info: 图表信息字典
            - type: 图表类型
            - data: 数据摘要
            - key_findings: 关键发现
            - parameters: 相关参数
        context: 上下文信息
            - section: 所在章节
            - related_content: 相关正文内容
            - previous_figures: 前文图表
    
    Returns:
        dict: 叙事文字
    """
    narrative = {}
    
    # 1. 生成Caption
    narrative['caption'] = generate_caption(
        figure_info['type'],
        figure_info['data'],
        figure_info['key_findings']
    )
    
    # 2. 生成正文引用句
    narrative['in_text_reference'] = generate_reference(
        figure_info,
        context['section']
    )
    
    # 3. 生成结果解释
    narrative['interpretation'] = generate_interpretation(
        figure_info['key_findings'],
        context['related_content']
    )
    
    # 4. 检查叙事质量
    narrative['quality_score'] = evaluate_narrative_quality(narrative)
    
    return narrative
```

## Caption生成规则

### 必须包含的元素

```python
CAPTION_ELEMENTS = {
    'what': "图表展示的内容",
    'how': "数据来源或方法",
    'finding': "关键发现或模式",
    'significance': "意义或对比（可选）"
}

def generate_caption(figure_type, data_summary, key_findings):
    """生成图题"""
    
    # 根据图表类型选择模板
    template = CAPTION_TEMPLATES[figure_type]
    
    # 填充模板
    caption = template.format(
        what=describe_what(data_summary),
        how=describe_how(data_summary),
        finding=summarize_finding(key_findings),
        significance=explain_significance(key_findings)
    )
    
    # 确保Caption完整独立
    caption = ensure_self_contained(caption)
    
    return caption
```

### Caption长度指南

| 图表复杂度 | 建议长度 | 句子数 |
|-----------|---------|--------|
| 简单 | 30-50词 | 2-3句 |
| 中等 | 50-80词 | 3-4句 |
| 复杂 | 80-120词 | 4-5句 |

## 正文引用生成

### 引用句模板库

```python
REFERENCE_TEMPLATES = {
    # 引入型
    'introduction': [
        "As illustrated in Figure {n}, {observation}.",
        "Figure {n} shows {content}, revealing {insight}.",
        "The results, presented in Figure {n}, indicate that {finding}.",
    ],
    
    # 对比型
    'comparison': [
        "Comparing the results in Figure {n}, we observe that {comparison}.",
        "Figure {n} highlights the difference between {A} and {B}, where {detail}.",
        "As Figure {n} demonstrates, {subject} outperforms {baseline} by {metric}.",
    ],
    
    # 支持论证型
    'support': [
        "This observation is further supported by Figure {n}, which shows {evidence}.",
        "Figure {n} provides additional evidence that {claim}.",
        "The pattern observed in Figure {n} confirms {hypothesis}.",
    ],
    
    # 深入分析型
    'analysis': [
        "A closer examination of Figure {n} reveals {detail}.",
        "Notably, Figure {n} shows an unexpected {phenomenon}, suggesting {interpretation}.",
        "The trend visible in Figure {n} can be attributed to {explanation}.",
    ]
}
```

### 引用句生成规则

```python
def generate_reference(figure_info, section_context):
    """生成正文引用句"""
    
    # 根据上下文选择引用类型
    ref_type = determine_reference_type(section_context)
    
    # 选择合适的模板
    template = select_template(REFERENCE_TEMPLATES[ref_type])
    
    # 提取关键信息
    observation = extract_main_observation(figure_info)
    insight = extract_insight(figure_info)
    
    # 生成引用句
    reference = template.format(
        n=figure_info['number'],
        observation=observation,
        insight=insight,
        # ... 其他参数
    )
    
    return reference
```

## 结果解释生成

### 解释深度层次

```
Level 1: 描述 (What)
└─ "图X显示了Y的分布"

Level 2: 观察 (Pattern)
└─ "可以观察到，随着X增加，Y呈现上升趋势"

Level 3: 分析 (Why)
└─ "这种趋势可能是由于Z因素的影响"

Level 4: 意义 (So What)
└─ "这一发现表明，在实际应用中应当考虑..."
```

**O奖要求**: 至少达到Level 3，最好Level 4

### 解释生成模板

```python
def generate_interpretation(key_findings, related_content):
    """生成结果解释"""
    
    interpretation = {
        'description': describe_finding(key_findings),
        'pattern': identify_pattern(key_findings),
        'analysis': analyze_cause(key_findings, related_content),
        'significance': explain_significance(key_findings)
    }
    
    # 组合成段落
    paragraph = combine_interpretation_elements(interpretation)
    
    return paragraph
```

## 输出格式

```json
{
  "figure_narratives": [
    {
      "figure_number": 1,
      "figure_type": "architecture",
      "caption": {
        "text": "Figure 1. Architecture of the proposed hybrid PINN-Transformer model...",
        "word_count": 65,
        "elements_check": {
          "what": true,
          "how": true,
          "finding": true
        }
      },
      "in_text_reference": {
        "text": "As illustrated in Figure 1, our hybrid architecture integrates physical constraints directly into the attention mechanism...",
        "reference_type": "introduction",
        "position_suggestion": "After model description paragraph"
      },
      "interpretation": {
        "text": "The architecture leverages the complementary strengths of physics-based modeling and deep learning. The physics encoder ensures that predictions satisfy fundamental conservation laws, while the transformer component captures complex temporal dependencies that traditional methods struggle to model. This hybrid approach achieves 23% higher accuracy than pure neural network baselines while maintaining physical consistency.",
        "depth_level": 4,
        "word_count": 58
      }
    },
    {
      "figure_number": 2,
      "figure_type": "sensitivity",
      "caption": {
        "text": "Figure 2. Sobol sensitivity indices for the climate migration model...",
        "word_count": 72
      },
      "in_text_reference": {
        "text": "Sensitivity analysis (Figure 2) reveals that temperature change (ΔT) is the most influential parameter...",
        "reference_type": "analysis"
      },
      "interpretation": {
        "text": "The high total-order index of temperature change (ST = 0.42) indicates substantial interaction effects with other parameters, particularly precipitation. This suggests that the combined effect of temperature and precipitation changes may be more severe than their individual impacts would suggest, highlighting the importance of integrated climate projections in migration modeling.",
        "depth_level": 4
      }
    }
  ],
  "narrative_statistics": {
    "total_figures": 8,
    "average_caption_length": 68,
    "average_interpretation_depth": 3.5,
    "figures_with_level_4_interpretation": 6
  },
  "quality_assessment": {
    "overall_score": 0.88,
    "caption_quality": 0.90,
    "reference_naturalness": 0.85,
    "interpretation_depth": 0.88
  }
}
```

## 常见错误避免

### 错误1: Caption过于简单
```
❌ "Figure 3. Results."
❌ "Figure 3. Model output."

✅ "Figure 3. Predicted migration flows under RCP 8.5 scenario (2050). 
   Arrow thickness indicates relative flow magnitude. The model predicts 
   a 340% increase in climate-induced migration from baseline."
```

### 错误2: 引用句机械
```
❌ "See Figure 3 for the results."
❌ "Figure 3 shows the output."

✅ "As Figure 3 illustrates, migration patterns exhibit strong 
   clustering around coastal corridors, with flows intensifying 
   toward urban centers."
```

### 错误3: 解释缺乏深度
```
❌ "The figure shows that A is greater than B."

✅ "The figure reveals that A exceeds B by 47%, a difference 
   attributable to the compounding effects of temperature rise 
   and land degradation. This finding has significant implications 
   for regional planning, suggesting that..."
```

### 错误4: 图表与正文脱节
```
❌ 图表在文中未被引用
❌ 引用了但没有解释
❌ 解释与图表内容不符

✅ 每个图表都有引用和解释
✅ 解释与图表内容紧密相关
✅ 图表支持正文论点
```

## 质量检查清单

- [ ] 每个图表都有完整Caption
- [ ] Caption包含What, How, Finding三要素
- [ ] 每个图表在正文中被引用
- [ ] 引用句自然流畅，不机械
- [ ] 有深度的结果解释（Level 3+）
- [ ] 图表编号连续且正确引用
- [ ] 无孤立图表（无引用无解释）

## 与其他技能集成

### 上游依赖
- `chart-generator`: 获取图表信息和数据
- `section-writer`: 获取正文上下文
- `model-solver`: 获取结果数据

### 下游输出
- `latex-compiler`: 提供Caption和引用文本
- `consistency-checker`: 检查图表-正文一致性
- `quality-reviewer`: 提供叙事质量评估

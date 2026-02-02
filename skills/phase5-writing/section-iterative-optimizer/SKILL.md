---
name: section-iterative-optimizer
description: 对论文各章节内容进行多轮迭代优化。基于Self-Refine模式（生成→自我评估→改进→重复），确保每个章节达到O奖水平的深度和质量。
dependencies: [section-writer, sensitivity-analyzer, model-builder]
outputs: [optimized_sections, section_scores, optimization_history, final_quality_metrics]
---

# 章节迭代优化器 (Section Iterative Optimizer)

## 功能概述

对section-writer生成的各章节内容进行多轮迭代优化，确保每个章节达到O奖论文的深度和质量标准。采用Self-Refine模式：生成 → 自我评估 → 针对性改进 → 重复，直到达标或达到最大迭代次数。

## 核心原则

### Self-Refine模式

基于2025年LLM最佳实践研究，Self-Refine迭代优化可提升约**20%**的输出质量。

```
循环流程:
1. 生成初始内容
2. 自我评估（识别弱点）
3. 针对性改进
4. 重新评估
5. 重复直到达标
```

## 章节评估维度

### 通用维度（适用于所有章节）

| 维度 | 权重 | 评分标准 |
|------|------|----------|
| 字数达标 | 20% | 是否达到最小字数要求 |
| 深度分析 | 25% | 是否有深层次的分析和解释 |
| 量化表达 | 20% | 具体数字和量化表述的数量 |
| 学术语言 | 15% | 专业词汇、避免口语化 |
| 逻辑连贯 | 10% | 段落间过渡、论证逻辑 |
| 无Chinglish | 10% | 避免中式英语表达 |

### 章节特定要求

#### Model Design章节 (1500-2500词)

**必需元素**:
- 建模动机 (300+词): 为什么选择这个模型
- 数学推导 (500+词): 完整的公式推导过程
- 物理/数学含义 (300+词): 公式中各项的解释
- 与备选方法对比 (300+词): 为什么不选其他方法
- 创新点声明 (200+词): 本文的创新之处

**评估规则**:
```python
MODEL_DESIGN_CRITERIA = {
    'min_words': 1500,
    'required_elements': ['motivation', 'derivation', 'meaning', 'comparison', 'innovation'],
    'min_equations': 5,
    'min_quantifications': 5,
    'depth_markers': ['because', 'therefore', 'indicates', 'demonstrates', 'reveals']
}
```

#### Results Analysis章节 (1200-1800词)

**必需元素**:
- 主要结果展示
- 结果深度解释（不只是描述）
- 结果与预期对比
- 结果的实际意义

**评估规则**:
```python
RESULTS_ANALYSIS_CRITERIA = {
    'min_words': 1200,
    'required_elements': ['results', 'interpretation', 'comparison', 'significance'],
    'min_quantifications': 8,
    'depth_score_threshold': 0.85
}
```

#### Sensitivity Analysis章节 (800-1200词)

**必需元素**:
- Sobol指数计算和解释
- 参数重要性排序
- 交互效应分析
- 鲁棒性结论

**评估规则**:
```python
SENSITIVITY_CRITERIA = {
    'min_words': 800,
    'required_elements': ['sobol_indices', 'ranking', 'interactions', 'robustness'],
    'global_method_required': True
}
```

## 评估函数

```python
def evaluate_section(section_name: str, content: str) -> Dict:
    """
    评估章节内容质量
    
    Returns:
        {
            'total_score': float,
            'dimension_scores': Dict[str, float],
            'weakest_dimension': str,
            'specific_issues': List[str],
            'suggestions': List[str]
        }
    """
    criteria = SECTION_CRITERIA.get(section_name, DEFAULT_CRITERIA)
    
    scores = {
        'word_count': evaluate_word_count(content, criteria['min_words']),
        'depth': evaluate_depth(content),
        'quantification': evaluate_quantification(content, criteria['min_quantifications']),
        'language': evaluate_language_quality(content),
        'logic': evaluate_logical_flow(content),
        'chinglish': evaluate_chinglish(content)
    }
    
    # 检查必需元素
    element_scores = {}
    for element in criteria['required_elements']:
        element_scores[element] = check_element_presence(content, element)
    
    scores['required_elements'] = sum(element_scores.values()) / len(element_scores)
    
    # 计算加权总分
    weights = DIMENSION_WEIGHTS[section_name]
    total_score = sum(scores[dim] * weights.get(dim, 0.1) for dim in scores)
    
    return {
        'total_score': total_score,
        'dimension_scores': scores,
        'element_scores': element_scores,
        'weakest_dimension': min(scores, key=scores.get),
        'specific_issues': identify_issues(content, scores),
        'suggestions': generate_suggestions(section_name, scores)
    }
```

## 迭代优化流程

```python
async def optimize_section(
    section_name: str, 
    initial_content: str, 
    max_iterations: int = 8
) -> Dict:
    """
    迭代优化章节内容
    
    Args:
        section_name: 章节名称
        initial_content: 初始内容
        max_iterations: 最大迭代次数
    
    Returns:
        优化结果
    """
    content = initial_content
    history = []
    criteria = SECTION_CRITERIA[section_name]
    
    for iteration in range(max_iterations):
        # 1. 评估当前内容
        evaluation = evaluate_section(section_name, content)
        
        history.append({
            'iteration': iteration + 1,
            'content_length': len(content.split()),
            'total_score': evaluation['total_score'],
            'dimension_scores': evaluation['dimension_scores'],
            'weakest_dimension': evaluation['weakest_dimension']
        })
        
        # 2. 检查是否达标
        if (evaluation['total_score'] >= 0.85 and 
            evaluation['dimension_scores']['word_count'] >= 0.95):
            break
        
        # 3. 确定优化重点
        focus = evaluation['weakest_dimension']
        
        # 4. 针对性改进
        content = await apply_targeted_improvement(
            section_name,
            content,
            focus,
            evaluation['suggestions']
        )
    
    return {
        'optimized_content': content,
        'final_score': evaluation['total_score'],
        'iterations': len(history),
        'history': history,
        'final_word_count': len(content.split())
    }
```

## 针对性改进策略

### 字数不足时的扩展策略

```python
async def expand_content(section_name: str, content: str, target_words: int) -> str:
    """
    扩展内容到目标字数
    """
    current_words = len(content.split())
    deficit = target_words - current_words
    
    expansion_prompt = f"""
    以下是论文的{section_name}章节，当前{current_words}词，需要扩展到{target_words}词。
    
    当前内容:
    {content}
    
    扩展策略（按优先级）:
    1. 增加更多细节和具体例子
    2. 添加与其他部分的关联分析
    3. 深化现有论点的解释
    4. 增加量化表述
    5. 添加方法论的对比分析
    
    要求:
    - 保持原有内容的核心观点不变
    - 新增内容必须有实质意义，不是简单的词汇重复
    - 使用学术英语
    - 增加至少{deficit // 50}个具体数字
    
    输出完整的扩展后章节。
    """
    
    return await llm.complete(expansion_prompt, max_tokens=8192)
```

### 深度不足时的增强策略

```python
async def deepen_analysis(section_name: str, content: str) -> str:
    """
    增强分析深度
    """
    deepen_prompt = f"""
    以下是论文的{section_name}章节，分析深度不足，需要增强。
    
    当前内容:
    {content}
    
    增强策略:
    1. 每个结论后增加"这是因为..."的解释
    2. 每个发现后增加"这表明..."的意义分析
    3. 增加因果关系的论证
    4. 增加与文献的对比分析
    5. 增加对异常情况的讨论
    
    深度标记词（必须增加使用）:
    - because, therefore, indicates, demonstrates
    - reveals, suggests, contributes, impacts
    - consequently, thus, hence, implies
    
    输出增强后的章节内容。
    """
    
    return await llm.complete(deepen_prompt, max_tokens=8192)
```

### Chinglish修正策略

```python
CHINGLISH_BLACKLIST = [
    ('with the development of', 'as X advances'),
    ('in recent years', 'recently'),
    ('more and more', 'increasingly'),
    ('plays an important role', 'is crucial for'),
    ('has great influence on', 'significantly affects'),
    ('put forward', 'propose'),
    ('make a contribution', 'contribute'),
    ('it is well known that', '[删除，直接陈述]'),
    ('as we all know', '[删除]'),
    ('nowadays', 'currently'),
]

async def fix_chinglish(content: str) -> str:
    """修正中式英语表达"""
    fix_prompt = f"""
    以下内容可能包含中式英语表达，请修正为地道的学术英语。
    
    内容:
    {content}
    
    中式英语黑名单及替换:
    {CHINGLISH_BLACKLIST}
    
    其他常见问题:
    - 避免使用"with the xxx of"结构
    - 避免过多使用被动语态
    - 避免使用空洞的形容词（very, really, quite）
    - 确保主语明确，避免"it is...that"结构
    
    输出修正后的内容。
    """
    
    return await llm.complete(fix_prompt, max_tokens=8192)
```

## 输出格式

```json
{
  "optimized_sections": {
    "model_design": {
      "content": "...",
      "word_count": 1823,
      "final_score": 0.91,
      "iterations": 5
    },
    "results_analysis": {
      "content": "...",
      "word_count": 1456,
      "final_score": 0.88,
      "iterations": 4
    }
  },
  "section_scores": {
    "model_design": {
      "total": 0.91,
      "word_count": 1.0,
      "depth": 0.89,
      "quantification": 0.85,
      "language": 0.93,
      "logic": 0.90,
      "chinglish": 0.95
    }
  },
  "optimization_history": [
    {
      "section": "model_design",
      "iteration": 1,
      "focus": "word_count",
      "score_before": 0.62,
      "score_after": 0.75,
      "changes": ["Expanded derivation section by 400 words"]
    }
  ],
  "final_quality_metrics": {
    "total_word_count": 8543,
    "average_section_score": 0.89,
    "all_sections_passed": true,
    "improvement_rate": 0.35
  }
}
```

## 质量门禁

### 必须通过
- ✅ 所有章节达到最小字数要求
- ✅ 每个章节包含必需元素
- ✅ 无严重Chinglish问题
- ✅ 章节间逻辑连贯

### 推荐标准
- 每章节得分 ≥ 0.85
- 平均章节得分 ≥ 0.88
- 总字数符合O奖标准（8000-10000词）

## 相关技能

### 上游依赖
- `section-writer` - 提供初始章节内容
- `model-builder` - 提供模型细节用于扩展
- `sensitivity-analyzer` - 提供敏感性分析结果

### 下游输出
- `consistency-checker` - 章节间一致性检查
- `final-polisher` - 最终润色

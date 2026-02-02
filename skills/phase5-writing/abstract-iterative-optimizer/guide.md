---
name: abstract-iterative-optimizer
description: 对摘要进行多轮迭代优化。基于O奖摘要特征进行自动评分，优化Hook质量、信息密度、量化表达、语言精炼度、逻辑连贯性，确保达到O奖水平。
dependencies: [abstract-generator, abstract-first-impression, grammar-checker]
outputs: [optimized_abstract, final_score, optimization_history, dimension_scores]
---

# 摘要迭代优化器 (Abstract Iterative Optimizer)

## 功能概述

对生成的摘要进行12轮以上的迭代优化，确保达到O奖水平。每轮优化针对评分最低的维度进行改进，直到所有维度均达标或达到最大迭代次数。

## 核心原则

### O奖摘要黄金标准

```
评委评估流程:
1. 第一印象 (5秒) - Hook句质量
2. 结构扫描 (30秒) - 完整性检查
3. 内容评估 (2-3分钟) - 方法+结果
4. 最终判断 (30秒) - 通过/淘汰
```

**关键洞察**: Hook决定生死，量化决定档次，创新决定高度。

## 评分维度（增强版）

### 维度1: Hook质量 (新增)
**权重: 15%**

| 子维度 | 评分标准 |
|--------|---------|
| 注意力吸引 | 是否能在5秒内抓住读者 (0-1) |
| 数据冲击力 | 开头是否有震撼性数据 (0-1) |
| 避免陈词 | 是否避免"With the development of..."等 (0-1) |
| 长度控制 | Hook句是否在25-35词范围 (0-1) |

**Hook质量检测规则**:
```python
HOOK_BLACKLIST = [
    "With the development of",
    "With the rapid development",
    "In recent years",
    "It is well known that",
    "As we all know",
    "Nowadays",
    "Currently",
    "At present",
    "plays an important role",
    "has attracted widespread attention"
]

def evaluate_hook_quality(abstract):
    first_sentence = extract_first_sentence(abstract)
    
    scores = {
        'attention': 0.0,
        'data_impact': 0.0,
        'avoid_cliche': 1.0,
        'length_control': 0.0
    }
    
    # 检查陈词滥调
    for cliche in HOOK_BLACKLIST:
        if cliche.lower() in first_sentence.lower():
            scores['avoid_cliche'] = 0.0
            break
    
    # 检查数据冲击力
    if contains_specific_numbers(first_sentence):
        scores['data_impact'] = 0.8
        if contains_comparison_or_contrast(first_sentence):
            scores['data_impact'] = 1.0
    
    # 检查长度
    word_count = len(first_sentence.split())
    if 25 <= word_count <= 35:
        scores['length_control'] = 1.0
    elif 20 <= word_count < 25 or 35 < word_count <= 40:
        scores['length_control'] = 0.7
    else:
        scores['length_control'] = 0.4
    
    # 注意力吸引（综合判断）
    scores['attention'] = (scores['data_impact'] + scores['avoid_cliche']) / 2
    
    return sum(scores.values()) / 4
```

### 维度2: 量化密度 (新增)
**权重: 15%**

| 子维度 | 评分标准 |
|--------|---------|
| 数字数量 | 摘要中具体数字的数量 (目标: 4-8个) |
| 数字分布 | 数字是否均匀分布在各部分 |
| 精确度 | 数字是否足够精确 (避免约数) |
| 意义性 | 数字是否有实际意义 |

**量化密度检测规则**:
```python
def evaluate_quantification_density(abstract):
    # 提取所有数字
    numbers = extract_numbers(abstract)
    word_count = len(abstract.split())
    
    scores = {
        'number_count': 0.0,
        'distribution': 0.0,
        'precision': 0.0,
        'meaningfulness': 0.0
    }
    
    # 数字数量评分 (4-8个为最佳)
    num_count = len(numbers)
    if 4 <= num_count <= 8:
        scores['number_count'] = 1.0
    elif 2 <= num_count < 4 or 8 < num_count <= 10:
        scores['number_count'] = 0.7
    elif num_count < 2:
        scores['number_count'] = 0.3
    else:
        scores['number_count'] = 0.5  # 过多
    
    # 分布均匀性
    scores['distribution'] = evaluate_number_distribution(abstract, numbers)
    
    # 精确度（检查是否使用约数）
    vague_patterns = ['about', 'approximately', 'around', 'roughly', 'nearly']
    precise_count = sum(1 for n in numbers if not any(v in n['context'] for v in vague_patterns))
    scores['precision'] = precise_count / max(len(numbers), 1)
    
    # 意义性（是否是关键结果数字）
    meaningful_keywords = ['%', 'accuracy', 'error', 'improvement', 'reduction', 'increase']
    meaningful_count = sum(1 for n in numbers if any(k in n['context'] for k in meaningful_keywords))
    scores['meaningfulness'] = meaningful_count / max(len(numbers), 1)
    
    return sum(scores.values()) / 4
```

### 维度3: 信息密度
**权重: 20%**

| 子维度 | 评分标准 |
|--------|---------|
| 冗余度 | 无重复表达、无空洞词汇 |
| 信息量 | 每句话传达的有效信息量 |
| 关键信息 | 核心内容是否突出 |

**检测规则**:
```python
FILLER_WORDS = [
    'very', 'really', 'quite', 'rather', 'somewhat',
    'basically', 'actually', 'generally', 'usually',
    'it should be noted that', 'it is worth mentioning',
    'it can be seen that', 'it is obvious that'
]

def evaluate_information_density(abstract):
    word_count = len(abstract.split())
    sentences = split_sentences(abstract)
    
    # 检测填充词
    filler_count = sum(1 for f in FILLER_WORDS if f.lower() in abstract.lower())
    filler_score = max(0, 1 - filler_count * 0.1)
    
    # 计算平均句子信息量
    avg_info_per_sentence = estimate_information_per_sentence(sentences)
    
    # 检测重复表达
    repetition_score = 1 - calculate_repetition_ratio(abstract)
    
    return (filler_score + avg_info_per_sentence + repetition_score) / 3
```

### 维度4: 语言质量
**权重: 20%**

| 子维度 | 评分标准 |
|--------|---------|
| 专业性 | 使用学术词汇，避免口语化 |
| 语法正确 | 无语法错误 |
| 时态一致 | 时态使用正确且一致 |
| 术语准确 | 专业术语使用准确 |

### 维度5: 创新突出
**权重: 15%**

| 子维度 | 评分标准 |
|--------|---------|
| 创新描述 | 是否明确描述创新点 |
| 对比清晰 | 是否与现有方法对比 |
| 贡献量化 | 创新贡献是否量化表达 |

**创新表达关键词**:
```python
INNOVATION_KEYWORDS = [
    'novel', 'new', 'innovative', 'first', 'unique',
    'propose', 'develop', 'introduce', 'present',
    'outperform', 'improve', 'enhance', 'exceed',
    'unlike', 'compared to', 'in contrast to', 'whereas'
]
```

### 维度6: 结构完整
**权重: 10%**

| 必需元素 | 检查项 |
|---------|--------|
| Hook | 开篇吸引句 |
| Background | 问题背景 |
| Problem | 问题陈述 |
| Method | 方法概述 |
| Results | 量化结果 |
| Value | 实际价值 |
| Keywords | 5-7个关键词 |

### 维度7: 可读性
**权重: 5%**

| 子维度 | 评分标准 |
|--------|---------|
| 句长控制 | 平均句长15-25词 |
| 过渡自然 | 段落间过渡流畅 |
| 逻辑清晰 | 论证逻辑清晰 |

## 综合评分公式

```python
def calculate_total_score(abstract):
    """计算摘要综合评分"""
    
    dimensions = {
        'hook_quality': (evaluate_hook_quality(abstract), 0.15),
        'quantification': (evaluate_quantification_density(abstract), 0.15),
        'information_density': (evaluate_information_density(abstract), 0.20),
        'language_quality': (evaluate_language_quality(abstract), 0.20),
        'innovation_highlight': (evaluate_innovation(abstract), 0.15),
        'structure_completeness': (evaluate_structure(abstract), 0.10),
        'readability': (evaluate_readability(abstract), 0.05)
    }
    
    total_score = sum(score * weight for score, weight in dimensions.values())
    
    return {
        'total_score': total_score,
        'dimension_scores': {k: v[0] for k, v in dimensions.items()},
        'weakest_dimension': min(dimensions, key=lambda k: dimensions[k][0])
    }
```

## 迭代优化策略

### 阶段1: 结构修复 (迭代1-3)
- 目标: 确保所有必需元素存在
- 重点: 补充缺失部分
- 通过标准: 结构完整度 ≥ 0.95

### 阶段2: Hook优化 (迭代4-6)
- 目标: 打造震撼开头
- 重点: 替换陈词滥调、增加数据冲击
- 通过标准: Hook质量 ≥ 0.85

### 阶段3: 量化增强 (迭代7-9)
- 目标: 增加量化表达
- 重点: 补充具体数字、避免模糊表述
- 通过标准: 量化密度 ≥ 0.80

### 阶段4: 语言精炼 (迭代10-12)
- 目标: 提升语言质量
- 重点: 删除冗余、提升专业性
- 通过标准: 语言质量 ≥ 0.85

### 阶段5: 最终润色 (迭代13-15)
- 目标: 整体打磨
- 重点: 流畅度、一致性
- 通过标准: 总分 ≥ 0.90

## 优化迭代流程

```python
def optimize_abstract(initial_abstract, max_iterations=15):
    """
    迭代优化摘要
    
    Args:
        initial_abstract: 初始摘要
        max_iterations: 最大迭代次数
    
    Returns:
        优化后的摘要和优化历史
    """
    abstract = initial_abstract
    history = []
    
    # 加载O奖基准库
    o_award_benchmarks = load_o_award_benchmarks()
    
    for iteration in range(max_iterations):
        # 评估当前版本
        evaluation = calculate_total_score(abstract)
        
        history.append({
            'iteration': iteration + 1,
            'abstract': abstract,
            'total_score': evaluation['total_score'],
            'dimension_scores': evaluation['dimension_scores'],
            'weakest_dimension': evaluation['weakest_dimension']
        })
        
        # 达到目标分数则停止
        if evaluation['total_score'] >= 0.90:
            break
        
        # 确定优化重点（基于最弱维度）
        focus = evaluation['weakest_dimension']
        
        # 获取O奖基准参考
        benchmark_examples = get_relevant_benchmarks(
            o_award_benchmarks, 
            focus, 
            abstract
        )
        
        # 执行针对性优化
        abstract = apply_targeted_optimization(
            abstract, 
            focus, 
            benchmark_examples
        )
    
    return {
        'optimized_abstract': abstract,
        'final_score': evaluation['total_score'],
        'iterations': len(history),
        'dimension_scores': evaluation['dimension_scores'],
        'optimization_history': history
    }

def apply_targeted_optimization(abstract, focus, benchmarks):
    """根据薄弱维度执行针对性优化"""
    
    optimization_strategies = {
        'hook_quality': optimize_hook,
        'quantification': optimize_quantification,
        'information_density': optimize_density,
        'language_quality': optimize_language,
        'innovation_highlight': optimize_innovation,
        'structure_completeness': optimize_structure,
        'readability': optimize_readability
    }
    
    optimizer = optimization_strategies.get(focus)
    if optimizer:
        return optimizer(abstract, benchmarks)
    
    return abstract
```

## 针对性优化函数

### Hook优化
```python
def optimize_hook(abstract, benchmarks):
    """优化Hook句"""
    first_sentence = extract_first_sentence(abstract)
    
    # 检查是否包含陈词滥调
    if contains_cliche(first_sentence):
        # 使用数据冲击模式重写
        new_hook = rewrite_with_data_impact(first_sentence, benchmarks)
        abstract = replace_first_sentence(abstract, new_hook)
    
    # 检查是否有具体数字
    if not contains_specific_numbers(first_sentence):
        # 从正文提取关键数据插入Hook
        key_number = extract_most_impactful_number(abstract)
        new_hook = inject_number_into_hook(first_sentence, key_number)
        abstract = replace_first_sentence(abstract, new_hook)
    
    return abstract
```

### 量化增强
```python
def optimize_quantification(abstract, benchmarks):
    """增强量化表达"""
    
    # 识别可量化但未量化的陈述
    vague_statements = find_vague_statements(abstract)
    
    for statement in vague_statements:
        # 尝试从模型结果中获取具体数字
        specific_value = get_specific_value_for_statement(statement)
        if specific_value:
            abstract = replace_vague_with_specific(abstract, statement, specific_value)
    
    # 替换约数表达
    abstract = replace_approximations(abstract)
    # "about 30%" -> "28.7%"
    # "significantly improved" -> "improved by 34%"
    
    return abstract
```

## O奖基准对比

### 基准库结构
```json
{
  "problem_type_A": {
    "hooks": [
      {
        "text": "...",
        "type": "data_impact",
        "score": 0.95
      }
    ],
    "quantification_examples": [...],
    "innovation_expressions": [...]
  }
}
```

### 对标评估
```python
def compare_with_benchmark(abstract, problem_type):
    """与O奖基准对比"""
    benchmarks = load_benchmarks(problem_type)
    
    comparison = {
        'hook_similarity': compare_hooks(abstract, benchmarks['hooks']),
        'quantification_level': compare_quantification(abstract, benchmarks),
        'language_level': compare_language_quality(abstract, benchmarks),
        'overall_gap': calculate_gap_to_benchmark(abstract, benchmarks)
    }
    
    return comparison
```

## 输出格式

```json
{
  "optimized_abstract": "Climate change threatens to displace over 200 million people by 2050...",
  "final_score": 0.92,
  "iterations": 12,
  "dimension_scores": {
    "hook_quality": 0.91,
    "quantification": 0.88,
    "information_density": 0.93,
    "language_quality": 0.94,
    "innovation_highlight": 0.89,
    "structure_completeness": 0.98,
    "readability": 0.90
  },
  "improvement_summary": {
    "initial_score": 0.65,
    "final_score": 0.92,
    "improvement_rate": 0.415,
    "key_improvements": [
      "Replaced cliche opening with data-impact hook",
      "Added 4 specific quantitative results",
      "Removed 6 filler words",
      "Strengthened innovation description"
    ]
  },
  "optimization_history": [
    {
      "iteration": 1,
      "focus": "structure_completeness",
      "score_before": 0.65,
      "score_after": 0.72,
      "changes": ["Added missing background context", "Included keywords section"]
    },
    {
      "iteration": 4,
      "focus": "hook_quality", 
      "score_before": 0.55,
      "score_after": 0.78,
      "changes": ["Replaced 'In recent years...' with data-driven opening"]
    }
  ],
  "benchmark_comparison": {
    "target_problem_type": "MCM_C",
    "similarity_to_o_award": 0.87,
    "gap_analysis": "Hook quality slightly below benchmark, consider more impactful opening data"
  }
}
```

## 质量门禁

### 必须通过的检查
1. ✅ Hook不包含黑名单短语
2. ✅ 至少包含4个具体数字
3. ✅ 结构完整（7个元素齐全）
4. ✅ 总词数在300-500之间
5. ✅ 无严重语法错误

### 推荐达到的标准
- Hook质量 ≥ 0.85
- 量化密度 ≥ 0.80
- 总分 ≥ 0.90

## 相关技能

### 上游依赖
- `abstract-generator` - 提供初始摘要
- `abstract-first-impression` - 提供Hook句

### 下游输出
- `grammar-checker` - 语法检查
- `quality-reviewer` - 最终质量审查

## 常见优化案例

### 案例1: Hook优化
```
❌ Before: "In recent years, climate change has become a serious problem."
✅ After: "Climate change threatens to displace 200 million people by 2050, yet current models fail to capture migration dynamics."
```

### 案例2: 量化增强
```
❌ Before: "Our model significantly outperforms existing methods."
✅ After: "Our model achieves 94.3% prediction accuracy, outperforming the state-of-the-art by 12.7%."
```

### 案例3: 信息密度提升
```
❌ Before: "It should be noted that the results clearly demonstrate that our approach is very effective."
✅ After: "Our approach reduces prediction error by 23%."
```

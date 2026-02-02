---
name: chinglish-detector
description: 检测并修正中式英语(Chinglish)问题，确保论文达到母语级别的英语表达质量。包含50+常见中式英语模式库和自动修正建议。
dependencies: [section-writer, grammar-checker]
outputs: [detection_results, corrections, chinglish_score, improved_text]
---

# 中式英语检测器 (Chinglish Detector)

## 概述

中式英语是影响美赛论文语言质量的主要因素之一。本技能通过模式匹配和语义分析，检测常见的中式英语表达，并提供地道的英语替换建议。

## 设计原理

### 中式英语的三大来源

1. **直译思维**: 将中文表达直接翻译为英文
2. **语法迁移**: 将中文语法结构套用到英文
3. **搭配错误**: 使用不符合英语习惯的词语搭配

### 检测策略

```
文本输入
    │
    ├─ 模式匹配检测 (基于规则)
    │   └─ 匹配预定义的中式英语模式
    │
    ├─ 搭配检测 (基于语料)
    │   └─ 检查词语搭配是否地道
    │
    └─ 语义检测 (基于上下文)
        └─ 分析表达是否自然
```

## 中式英语模式库 (50+ 模式)

### 类别1: 陈词滥调开头 (Cliché Openings)

| ID | 中式英语 | 正确表达 | 说明 |
|----|---------|---------|------|
| C01 | With the development of... | As X advances/evolves | 过度使用的开头 |
| C02 | With the rapid development of... | Recent advances in X | 同上 |
| C03 | In recent years... | Recently / Over the past decade | 更精确 |
| C04 | Nowadays... | Currently / At present | 口语化 |
| C05 | As we all know... | [直接陈述] | 假设读者知识 |
| C06 | It is well known that... | [直接陈述] | 无意义填充 |
| C07 | As is known to all... | [直接陈述] | 同上 |
| C08 | With the coming of... | The emergence of / The advent of | 直译 |
| C09 | Under the background of... | In the context of / Given | 直译 |
| C10 | Along with the progress of... | As X progresses | 直译 |

### 类别2: 程度表达 (Degree Expressions)

| ID | 中式英语 | 正确表达 | 说明 |
|----|---------|---------|------|
| D01 | more and more | increasingly | 口语化 |
| D02 | more and more important | increasingly important | 同上 |
| D03 | very very | extremely / highly | 重复强调 |
| D04 | a lot of | numerous / substantial / many | 口语化 |
| D05 | lots of | numerous / many | 口语化 |
| D06 | big/small | large/small, significant/minor | 口语化 |
| D07 | good/bad | effective/ineffective, favorable/unfavorable | 口语化 |
| D08 | get better/worse | improve/deteriorate | 口语化 |
| D09 | so...that | sufficiently...to / ...enough to | 有时可用 |
| D10 | very unique | unique (无需修饰) | 逻辑错误 |

### 类别3: 动词搭配 (Verb Collocations)

| ID | 中式英语 | 正确表达 | 说明 |
|----|---------|---------|------|
| V01 | play an important role | serve as / function as | 过度使用 |
| V02 | plays a vital role | is essential/crucial for | 过度使用 |
| V03 | has great influence on | significantly affects/influences | 更直接 |
| V04 | make a contribution to | contribute to | 更简洁 |
| V05 | give a description of | describe | 名词化 |
| V06 | make an analysis of | analyze | 名词化 |
| V07 | conduct research on | research / investigate | 更简洁 |
| V08 | carry out experiments | perform/conduct experiments | 搭配问题 |
| V09 | put forward a method | propose/present a method | 直译 |
| V10 | solve the problem | address/tackle the problem | 更学术 |

### 类别4: 冗余表达 (Redundancy)

| ID | 中式英语 | 正确表达 | 说明 |
|----|---------|---------|------|
| R01 | basic fundamentals | fundamentals | 同义重复 |
| R02 | future prospects | prospects | 同义重复 |
| R03 | past history | history | 同义重复 |
| R04 | true fact | fact | 同义重复 |
| R05 | final conclusion | conclusion | 同义重复 |
| R06 | completely eliminate | eliminate | 同义重复 |
| R07 | absolutely essential | essential | 同义重复 |
| R08 | advance planning | planning | 同义重复 |
| R09 | end result | result | 同义重复 |
| R10 | free gift | gift | 同义重复 |

### 类别5: 介词使用 (Preposition Usage)

| ID | 中式英语 | 正确表达 | 说明 |
|----|---------|---------|------|
| P01 | in the aspect of | in terms of / regarding | 直译 |
| P02 | in the field of X | in X | 冗余 |
| P03 | in the process of | while/when -ing | 冗余 |
| P04 | through the method of | by/using | 冗余 |
| P05 | according to X's opinion | according to X | 冗余 |
| P06 | for the purpose of | to/for | 冗余 |
| P07 | in spite of the fact that | although/despite | 冗余 |
| P08 | due to the fact that | because | 冗余 |
| P09 | on the basis of | based on | 更简洁 |
| P10 | by means of | by/through | 更简洁 |

### 类别6: 句式结构 (Sentence Structure)

| ID | 中式英语 | 正确表达 | 说明 |
|----|---------|---------|------|
| S01 | The reason is because... | The reason is that... / Because... | 重复 |
| S02 | The purpose is to... | This aims to... / We aim to... | 更自然 |
| S03 | It can be seen that... | [直接陈述] | 冗余开头 |
| S04 | It should be noted that... | Notably, / Note that | 更简洁 |
| S05 | There is no doubt that... | Clearly, / Undoubtedly, | 更简洁 |
| S06 | It goes without saying that... | Obviously, / Clearly, | 过于口语 |
| S07 | As far as X is concerned... | Regarding X, / For X, | 更简洁 |
| S08 | In order to... | To... | 更简洁 |
| S09 | So as to... | To... | 更简洁 |
| S10 | For the sake of... | For... | 更简洁 |

### 类别7: 学术写作特有 (Academic Writing Specific)

| ID | 中式英语 | 正确表达 | 说明 |
|----|---------|---------|------|
| A01 | This paper will discuss... | This paper discusses... | 时态问题 |
| A02 | In this paper, we will... | In this paper, we... | 时态问题 |
| A03 | Through this study... | This study... | 冗余 |
| A04 | By using the method... | Using the method... | 冗余 |
| A05 | The experimental results show that... | Results indicate/show that... | 冗余 |
| A06 | After careful analysis... | Analysis reveals... | 主观 |
| A07 | It is proved that... | This demonstrates/shows that... | 过强声明 |
| A08 | Obviously/Clearly (无数据支撑) | Results suggest/indicate | 主观判断 |
| A09 | We think/believe that... | Evidence suggests that... | 主观 |
| A10 | It is certain that... | This indicates/suggests that... | 过强声明 |

## 检测实现

### 主检测函数

```python
def detect_chinglish(text: str) -> dict:
    """
    检测文本中的中式英语问题
    
    Args:
        text: 待检测的文本
        
    Returns:
        检测结果字典
    """
    results = {
        'total_issues': 0,
        'issues_by_category': {},
        'issues': [],
        'chinglish_score': 0.0,  # 0=无问题, 1=严重
        'suggested_corrections': []
    }
    
    # 加载模式库
    patterns = load_chinglish_patterns()
    
    # 按类别检测
    for category, category_patterns in patterns.items():
        category_issues = []
        
        for pattern in category_patterns:
            matches = find_pattern_matches(text, pattern)
            
            for match in matches:
                issue = {
                    'id': pattern['id'],
                    'category': category,
                    'matched_text': match['text'],
                    'position': match['position'],
                    'chinglish_pattern': pattern['chinglish'],
                    'correction': pattern['correction'],
                    'explanation': pattern['explanation'],
                    'severity': pattern.get('severity', 'medium')
                }
                category_issues.append(issue)
                results['issues'].append(issue)
        
        results['issues_by_category'][category] = len(category_issues)
    
    results['total_issues'] = len(results['issues'])
    results['chinglish_score'] = calculate_chinglish_score(text, results['issues'])
    results['suggested_corrections'] = generate_corrections(text, results['issues'])
    
    return results
```

### 评分计算

```python
def calculate_chinglish_score(text: str, issues: list) -> float:
    """
    计算中式英语严重程度评分
    
    评分标准:
    - 0.00-0.20: 优秀 (接近母语水平)
    - 0.21-0.40: 良好 (少量问题)
    - 0.41-0.60: 中等 (明显问题)
    - 0.61-0.80: 较差 (大量问题)
    - 0.81-1.00: 严重 (需要大幅修改)
    """
    word_count = len(text.split())
    
    # 基础分数 = 问题数 / 词数
    base_score = len(issues) / max(word_count, 1) * 10
    
    # 严重程度加权
    severity_weights = {'high': 1.5, 'medium': 1.0, 'low': 0.5}
    weighted_issues = sum(
        severity_weights.get(issue['severity'], 1.0) 
        for issue in issues
    )
    weighted_score = weighted_issues / max(word_count, 1) * 10
    
    # 综合评分
    final_score = min(1.0, (base_score + weighted_score) / 2)
    
    return round(final_score, 3)
```

## 自动修正

### 修正策略

```python
def generate_corrections(text: str, issues: list) -> list:
    """生成修正建议"""
    corrections = []
    
    # 按位置排序（从后向前修正避免位置偏移）
    sorted_issues = sorted(issues, key=lambda x: x['position'][0], reverse=True)
    
    corrected_text = text
    
    for issue in sorted_issues:
        original = issue['matched_text']
        suggestion = issue['correction']
        
        # 保持原文大小写
        if original[0].isupper():
            suggestion = suggestion[0].upper() + suggestion[1:]
        
        correction = {
            'original': original,
            'suggested': suggestion,
            'position': issue['position'],
            'category': issue['category'],
            'confidence': calculate_correction_confidence(issue)
        }
        
        corrections.append(correction)
        
        # 应用修正
        start, end = issue['position']
        corrected_text = corrected_text[:start] + suggestion + corrected_text[end:]
    
    return {
        'corrections': corrections,
        'corrected_text': corrected_text
    }
```

### 上下文感知修正

```python
def context_aware_correction(text: str, issue: dict) -> str:
    """
    基于上下文选择最佳修正
    
    某些中式英语有多个可能的修正，需要根据上下文选择
    """
    context_window = extract_context(text, issue['position'], window_size=50)
    
    possible_corrections = issue.get('corrections', [issue['correction']])
    
    if len(possible_corrections) == 1:
        return possible_corrections[0]
    
    # 基于上下文选择最佳修正
    best_correction = select_best_correction(
        context_window, 
        possible_corrections
    )
    
    return best_correction
```

## 输出格式

```json
{
  "detection_summary": {
    "total_issues": 12,
    "chinglish_score": 0.35,
    "quality_level": "good",
    "issues_by_category": {
      "cliche_openings": 2,
      "degree_expressions": 3,
      "verb_collocations": 2,
      "redundancy": 3,
      "preposition_usage": 1,
      "sentence_structure": 1
    }
  },
  "issues": [
    {
      "id": "C01",
      "category": "cliche_openings",
      "matched_text": "With the development of technology",
      "position": [0, 32],
      "chinglish_pattern": "With the development of...",
      "correction": "As technology advances",
      "explanation": "过度使用的陈词滥调开头，建议直接陈述或使用更精确的表达",
      "severity": "high"
    },
    {
      "id": "D01",
      "category": "degree_expressions",
      "matched_text": "more and more important",
      "position": [156, 179],
      "chinglish_pattern": "more and more",
      "correction": "increasingly important",
      "explanation": "口语化表达，学术写作应使用更正式的副词",
      "severity": "medium"
    }
  ],
  "corrections": {
    "applied_count": 12,
    "high_confidence": 10,
    "medium_confidence": 2,
    "corrected_text": "As technology advances, this issue has become increasingly important..."
  },
  "recommendations": [
    "开头使用了陈词滥调，建议重写为更有冲击力的表达",
    "发现3处口语化程度表达，建议使用学术词汇替换",
    "发现2处冗余表达，建议精简"
  ]
}
```

## 质量标准

### O奖论文中式英语标准

| 等级 | 分数范围 | 说明 | 是否达标 |
|------|---------|------|---------|
| A | 0.00-0.15 | 接近母语水平 | ✅ O奖水平 |
| B | 0.16-0.25 | 少量小问题 | ✅ 可接受 |
| C | 0.26-0.40 | 有明显问题 | ⚠️ 需改进 |
| D | 0.41-0.60 | 问题较多 | ❌ 不达标 |
| F | >0.60 | 严重问题 | ❌ 需重写 |

**O奖目标**: Chinglish Score ≤ 0.20

## 与其他技能集成

### 上游依赖
- `section-writer`: 章节写作完成后检测
- `abstract-generator`: 摘要生成后检测

### 下游输出
- `grammar-checker`: 配合语法检查
- `academic-english-optimizer`: 提供优化目标
- `final-polisher`: 最终润色参考

## 使用示例

```python
# 检测并修正中式英语
from chinglish_detector import detect_and_correct

text = """
With the development of artificial intelligence, machine learning 
plays an important role in many fields. More and more researchers 
conduct research on this topic. In order to solve this problem, 
we put forward a new method. Through the method of deep learning, 
we can get better results.
"""

result = detect_and_correct(text)

print(f"检测到 {result['total_issues']} 处中式英语问题")
print(f"中式英语评分: {result['chinglish_score']}")
print(f"\n修正后的文本:\n{result['corrected_text']}")

# 输出:
# 检测到 6 处中式英语问题
# 中式英语评分: 0.45
# 
# 修正后的文本:
# Recent advances in artificial intelligence have made machine learning 
# essential in many fields. An increasing number of researchers 
# investigate this topic. To address this problem, 
# we propose a new method. Using deep learning, 
# we achieve improved results.
```

## 模式库扩展

模式库存储在 `knowledge_base/chinglish_patterns.json`，支持持续扩展。

添加新模式的格式:
```json
{
  "id": "NEW01",
  "category": "category_name",
  "chinglish": "中式英语模式",
  "correction": "正确表达",
  "explanation": "说明",
  "severity": "high/medium/low",
  "examples": [
    {"wrong": "错误示例", "correct": "正确示例"}
  ]
}
```

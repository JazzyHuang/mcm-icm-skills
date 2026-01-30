---
name: abstract-first-impression
description: 生成摘要"第一印象"——专注于开篇Hook句设计，确保在评委10分钟初筛中脱颖而出
dependencies: [problem-parser, problem-type-classifier, model-selector]
outputs: [hook_sentence, abstract_opening, hook_type, hook_score]
---

# 摘要第一印象生成器

## 概述

美赛评委在初筛阶段仅用约10分钟阅读摘要，摘要的第一句话（Hook Sentence）决定了评委的第一印象。本技能专注于生成能够立即抓住评委注意力的开篇句。

## 核心原则

### 评委心理模型

```
评委初筛流程:
1. 扫描摘要开头 (5秒) → 第一印象形成
2. 快速浏览全文结构 (30秒) → 完整性判断
3. 阅读摘要详情 (2-3分钟) → 方法和结果评估
4. 决定分类 (30秒) → 通过/淘汰/待定
```

**关键洞察**: 开头5秒决定评委是否愿意认真阅读。

## Hook句三大模式

### 模式1: 问题重要性钩子 (Problem Significance Hook)

**适用场景**: 问题本身具有重大现实意义

**结构模板**:
```
[令人震惊的事实/数据] + [问题的紧迫性] + [本研究的回应]
```

**示例**:
```
Climate change threatens to displace over 200 million people by 2050, yet 
current migration prediction models fail to capture the complex interplay 
of environmental, economic, and social factors—our integrated multi-agent 
framework addresses this critical gap.
```

**评分标准**:
- 数据冲击力 (0-1): 是否使用震撼性统计数据
- 紧迫性表达 (0-1): 是否传达问题的急迫性
- 研究关联度 (0-1): 是否自然引出本研究

### 模式2: 数据冲击钩子 (Data Impact Hook)

**适用场景**: 研究结果具有令人惊讶的发现

**结构模板**:
```
[出人意料的发现/数字] + [与常规认知的对比] + [方法简述]
```

**示例**:
```
Our analysis reveals that a mere 3.7% adjustment in traffic signal timing 
can reduce urban congestion by up to 28%—a finding that challenges the 
prevailing assumption that major infrastructure investments are required 
for meaningful improvement.
```

**评分标准**:
- 数字精确性 (0-1): 使用具体而非模糊数字
- 反直觉程度 (0-1): 与常规预期的偏离度
- 可信度 (0-1): 结果是否在合理范围内

### 模式3: 方法创新钩子 (Method Innovation Hook)

**适用场景**: 提出了新颖的建模方法或独特的问题视角

**结构模板**:
```
[传统方法的局限] + [创新方法的核心思想] + [预期突破]
```

**示例**:
```
While traditional optimization approaches treat wildfire evacuation as a 
static routing problem, we reconceptualize it as a dynamic multi-objective 
game between evacuees and spreading fire fronts, enabling real-time 
adaptive route recommendations that reduce evacuation time by 34%.
```

**评分标准**:
- 对比清晰度 (0-1): 新旧方法对比是否明确
- 创新表述 (0-1): 创新点是否清晰传达
- 价值量化 (0-1): 是否有量化的改进预期

## 生成流程

```python
def generate_first_impression(problem_analysis, model_info, results_preview):
    """
    生成摘要第一印象
    
    Args:
        problem_analysis: 问题分析结果
        model_info: 模型信息
        results_preview: 初步结果预览
    
    Returns:
        dict: {
            'hook_sentence': str,      # 生成的Hook句
            'hook_type': str,          # Hook类型
            'hook_score': float,       # Hook质量评分
            'abstract_opening': str,   # 完整开篇段落(前2-3句)
            'alternatives': list       # 备选Hook句列表
        }
    """
    
    # 步骤1: 分析问题特征，确定最佳Hook类型
    hook_type = select_hook_type(problem_analysis, model_info, results_preview)
    
    # 步骤2: 提取Hook素材
    hook_materials = extract_hook_materials(problem_analysis, results_preview)
    
    # 步骤3: 生成候选Hook句
    candidates = generate_hook_candidates(hook_type, hook_materials)
    
    # 步骤4: 评估并选择最佳Hook
    best_hook, score = evaluate_and_select(candidates)
    
    # 步骤5: 扩展为完整开篇
    abstract_opening = expand_to_opening(best_hook, problem_analysis)
    
    return {
        'hook_sentence': best_hook,
        'hook_type': hook_type,
        'hook_score': score,
        'abstract_opening': abstract_opening,
        'alternatives': candidates[:3]  # 返回前3个备选
    }
```

## Hook素材提取

### 从问题分析中提取

```python
def extract_from_problem(problem_analysis):
    """提取问题相关的Hook素材"""
    return {
        'affected_scale': extract_affected_population_or_scale(),  # 影响规模
        'economic_impact': extract_economic_figures(),              # 经济影响
        'urgency_indicators': extract_urgency_signals(),            # 紧迫性指标
        'knowledge_gaps': extract_current_limitations(),            # 现有知识缺口
        'real_world_context': extract_practical_relevance()         # 现实关联
    }
```

### 从结果预览中提取

```python
def extract_from_results(results_preview):
    """提取结果相关的Hook素材"""
    return {
        'key_findings': extract_main_findings(),           # 主要发现
        'improvement_metrics': extract_improvements(),      # 改进指标
        'surprising_insights': extract_unexpected_results(), # 意外发现
        'comparison_baselines': extract_baselines()         # 对比基准
    }
```

## Hook类型选择决策树

```
问题特征分析
    │
    ├─ 问题涉及重大社会/环境/经济问题？
    │   └─ YES → 有震撼性统计数据？
    │              ├─ YES → 模式1: 问题重要性钩子
    │              └─ NO  → 继续评估
    │
    ├─ 研究结果有反直觉发现？
    │   └─ YES → 可以量化表达？
    │              ├─ YES → 模式2: 数据冲击钩子
    │              └─ NO  → 继续评估
    │
    └─ 方法具有明显创新性？
        └─ YES → 模式3: 方法创新钩子
        └─ NO  → 回退到模式1（默认最安全）
```

## 题型特定Hook策略

### MCM A题 (连续问题)
- 优先使用: 方法创新钩子
- 强调: PDE/ODE的物理意义、数值方法的精度改进
- 示例关键词: "continuous dynamics", "physical constraints", "numerical precision"

### MCM B题 (离散问题)
- 优先使用: 数据冲击钩子
- 强调: 组合优化的效率提升、图论方法的新应用
- 示例关键词: "combinatorial complexity", "polynomial-time", "network optimization"

### MCM C题 (数据问题)
- 优先使用: 数据冲击钩子
- 强调: 预测精度、模式发现、大数据洞察
- 示例关键词: "predictive accuracy", "hidden patterns", "data-driven insights"

### ICM D题 (运筹学)
- 优先使用: 问题重要性钩子
- 强调: 资源优化效果、决策支持价值
- 示例关键词: "resource allocation", "decision optimization", "operational efficiency"

### ICM E题 (可持续性)
- 优先使用: 问题重要性钩子
- 强调: 环境影响、可持续发展、长期效益
- 示例关键词: "sustainability", "environmental impact", "long-term viability"

### ICM F题 (政策)
- 优先使用: 问题重要性钩子
- 强调: 政策影响、社会效益、实施可行性
- 示例关键词: "policy implications", "societal impact", "implementation feasibility"

## 质量评估维度

### Hook质量评分表 (总分1.0)

| 维度 | 权重 | 评分标准 |
|------|------|---------|
| 注意力吸引 | 0.25 | 是否能在5秒内抓住读者注意力 |
| 信息密度 | 0.20 | 单位词数传达的有效信息量 |
| 专业性 | 0.20 | 术语使用是否准确、语言是否学术化 |
| 具体性 | 0.20 | 是否有具体数字、避免模糊表达 |
| 流畅性 | 0.15 | 语法正确、节奏感好、易于阅读 |

### 评分公式

```python
def calculate_hook_score(hook_sentence):
    """计算Hook质量评分"""
    scores = {
        'attention': evaluate_attention_grabbing(hook_sentence),
        'density': evaluate_information_density(hook_sentence),
        'professionalism': evaluate_academic_tone(hook_sentence),
        'specificity': evaluate_specificity(hook_sentence),
        'fluency': evaluate_fluency(hook_sentence)
    }
    
    weights = {
        'attention': 0.25,
        'density': 0.20,
        'professionalism': 0.20,
        'specificity': 0.20,
        'fluency': 0.15
    }
    
    return sum(scores[k] * weights[k] for k in scores)
```

## 常见错误避免

### 错误1: 陈词滥调开头
```
❌ "With the rapid development of technology..."
❌ "In recent years, ... has attracted widespread attention..."
❌ "It is well known that..."

✅ 直接陈述震撼性事实或发现
```

### 错误2: 过于模糊
```
❌ "This paper proposes a novel method to solve the problem."
❌ "Our model achieves good results."

✅ "Our hybrid PINN-transformer model reduces prediction error by 47%."
```

### 错误3: 缺乏数据支撑
```
❌ "Climate change is a serious problem."
❌ "Traffic congestion costs a lot."

✅ "Urban traffic congestion costs the US economy $87 billion annually."
```

### 错误4: 过长或过复杂
```
❌ 超过40词的开头句
❌ 包含3个以上从句

✅ 控制在25-35词，最多2个从句
```

## 输出格式

```json
{
  "hook_sentence": "Climate change threatens to displace over 200 million people by 2050...",
  "hook_type": "problem_significance",
  "hook_score": 0.87,
  "abstract_opening": "Climate change threatens to displace over 200 million people by 2050, yet current migration prediction models fail to capture the complex interplay of environmental, economic, and social factors. We develop a multi-agent simulation framework that integrates climate projections, economic indicators, and social network dynamics to predict migration patterns with 23% higher accuracy than existing approaches.",
  "alternatives": [
    {
      "sentence": "Our integrated framework predicts climate-induced migration with 23% higher accuracy...",
      "type": "data_impact",
      "score": 0.82
    },
    {
      "sentence": "By reconceptualizing migration as a multi-agent game rather than a statistical extrapolation...",
      "type": "method_innovation", 
      "score": 0.79
    }
  ],
  "evaluation": {
    "attention": 0.90,
    "density": 0.85,
    "professionalism": 0.88,
    "specificity": 0.85,
    "fluency": 0.87
  }
}
```

## O奖基准参考

### O奖摘要开头特征统计 (近5年数据)

| 特征 | 占比 |
|------|------|
| 包含具体数字 | 92% |
| 使用对比结构 | 78% |
| 提及现实影响 | 85% |
| 开头句≤35词 | 88% |
| 避免陈词滥调 | 100% |

### 高分Hook句库 (按题型分类)

参见: `knowledge_base/o_award_corpus/hook_examples.json`

## 与其他技能集成

### 上游依赖
- `problem-parser`: 获取问题分析结果
- `problem-type-classifier`: 确定题型以选择Hook策略
- `model-selector`: 了解所选模型的创新点

### 下游输出
- `abstract-generator`: 提供Hook句和开篇段落
- `abstract-iterative-optimizer`: 提供Hook质量评分作为优化目标

## 迭代优化

如果初始Hook评分 < 0.80，执行以下优化:

1. **换类型**: 尝试其他两种Hook模式
2. **增数据**: 补充更多具体数字
3. **精简化**: 删除冗余词汇，提高信息密度
4. **对标**: 参考O奖库中相似题型的Hook句

目标: Hook评分 ≥ 0.85

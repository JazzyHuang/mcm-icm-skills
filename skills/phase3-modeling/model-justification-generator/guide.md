---
name: model-justification-generator
description: 自动生成模型选择论证章节，包含模型对比分析、选择理由阐述、创新贡献声明，确保评委理解"为什么选这个模型"。
dependencies: [model-selector, hybrid-model-designer, problem-analyzer]
outputs: [justification_section, comparison_table, contribution_statement, model_diagram]
---

# 模型选择论证生成器 (Model Justification Generator)

## 概述

美赛评委重点关注"为什么选择这个模型"，而非"模型是什么"。本技能自动生成完整的模型选择论证章节，包括与备选方法的对比、选择理由的阐述、以及创新贡献的声明。

## 核心原则

### 评委关注点

```
评委评估模型选择时关注:
1. 问题-模型匹配度: 模型是否真正适合问题特点？
2. 选择合理性: 为什么不选其他方法？
3. 创新价值: 相比现有方法有何改进？
4. 理论支撑: 选择是否有理论依据？
```

### O奖论文模型选择特点

- ✅ 明确列出2-3种备选方法
- ✅ 量化比较各方法优缺点
- ✅ 清晰阐述选择理由
- ✅ 突出创新贡献
- ❌ 避免: 堆砌模型、缺乏论证、选择随意

## 论证结构

### 完整论证框架

```
1. 问题特征分析
   └─ 提取问题的关键数学特征
   
2. 候选方法识别
   └─ 列出3-5种可能的建模方法
   
3. 方法对比分析
   └─ 多维度比较各方法
   
4. 选择理由阐述
   └─ 解释为什么选择该模型
   
5. 创新贡献声明
   └─ 说明相比现有方法的改进
   
6. 理论支撑
   └─ 引用支持该选择的文献
```

## 生成流程

### 步骤1: 问题特征提取

```python
def extract_problem_features(problem_analysis):
    """
    提取问题的关键数学特征
    
    Returns:
        dict: 问题特征字典
    """
    features = {
        # 数据特征
        'data_type': identify_data_type(),      # 连续/离散/混合
        'data_scale': estimate_data_scale(),     # 小/中/大规模
        'dimensionality': get_dimensionality(),  # 低/中/高维
        
        # 问题特征
        'problem_type': classify_problem_type(), # 优化/预测/分类等
        'constraint_type': identify_constraints(), # 线性/非线性/无约束
        'objective_type': identify_objective(),  # 单目标/多目标
        
        # 物理特征
        'physical_laws': identify_physical_laws(), # 是否涉及物理定律
        'time_dependency': has_time_dependency(),  # 是否时间相关
        'spatial_structure': has_spatial_structure(), # 是否空间结构
        
        # 不确定性
        'uncertainty_level': estimate_uncertainty(), # 不确定性程度
        'noise_level': estimate_noise()  # 噪声水平
    }
    
    return features
```

### 步骤2: 候选方法识别

```python
def identify_candidate_methods(problem_features, problem_type):
    """
    根据问题特征识别候选方法
    
    Args:
        problem_features: 问题特征
        problem_type: 题型 (A/B/C/D/E/F)
    
    Returns:
        list: 候选方法列表
    """
    candidates = []
    
    # 基于题型的方法库
    METHOD_LIBRARY = {
        'A': {  # 连续问题
            'traditional': ['ODE Solver', 'FEM', 'Finite Difference'],
            'modern': ['PINN', 'Neural Operators', 'Physics-Informed ML'],
            'hybrid': ['PINN + Traditional', 'ML-Enhanced FEM']
        },
        'B': {  # 离散问题
            'traditional': ['Graph Theory', 'Dynamic Programming', 'Greedy'],
            'modern': ['GNN', 'Reinforcement Learning'],
            'hybrid': ['GNN + Optimization', 'RL + Heuristics']
        },
        'C': {  # 数据问题
            'traditional': ['Time Series', 'Regression', 'ARIMA'],
            'modern': ['Transformer', 'LSTM', 'XGBoost'],
            'hybrid': ['Transformer + Statistical', 'Ensemble Methods']
        },
        'D': {  # 运筹学
            'traditional': ['Linear Programming', 'Integer Programming', 'Network Flow'],
            'modern': ['Meta-heuristics', 'RL for OR'],
            'hybrid': ['Exact + Heuristic', 'Decomposition Methods']
        },
        'E': {  # 可持续性
            'traditional': ['System Dynamics', 'Multi-Objective Optimization'],
            'modern': ['Agent-Based Modeling', 'Causal Inference'],
            'hybrid': ['SD + ABM', 'Causal + Optimization']
        },
        'F': {  # 政策
            'traditional': ['Game Theory', 'Simulation', 'Decision Trees'],
            'modern': ['Multi-Agent RL', 'Causal ML'],
            'hybrid': ['Game + Simulation', 'Policy Learning']
        }
    }
    
    # 获取该题型的方法库
    type_methods = METHOD_LIBRARY.get(problem_type, METHOD_LIBRARY['C'])
    
    # 根据问题特征筛选
    for category, methods in type_methods.items():
        for method in methods:
            score = calculate_method_fitness(method, problem_features)
            candidates.append({
                'method': method,
                'category': category,
                'fitness_score': score
            })
    
    # 返回前5个最匹配的方法
    return sorted(candidates, key=lambda x: x['fitness_score'], reverse=True)[:5]
```

### 步骤3: 方法对比分析

```python
def generate_comparison_analysis(selected_method, candidate_methods, problem_features):
    """
    生成方法对比分析
    
    Returns:
        dict: 对比分析结果
    """
    comparison_dimensions = [
        'accuracy',          # 精度
        'computational_cost', # 计算成本
        'interpretability',  # 可解释性
        'scalability',       # 可扩展性
        'data_requirement',  # 数据需求
        'theoretical_basis', # 理论基础
        'implementation',    # 实现难度
        'innovation'         # 创新性
    ]
    
    comparison_table = []
    
    for method in candidate_methods[:3]:  # 对比前3个方法
        row = {'method': method['method']}
        
        for dim in comparison_dimensions:
            score, explanation = evaluate_method_on_dimension(
                method['method'], 
                dim, 
                problem_features
            )
            row[dim] = {
                'score': score,        # 1-5分
                'explanation': explanation
            }
        
        comparison_table.append(row)
    
    return {
        'dimensions': comparison_dimensions,
        'comparison_table': comparison_table,
        'summary': generate_comparison_summary(comparison_table)
    }
```

### 步骤4: 选择理由生成

```python
def generate_selection_rationale(selected_method, comparison, problem_features):
    """
    生成模型选择理由
    
    Returns:
        str: 选择理由段落
    """
    rationale_template = """
    ## Model Selection Rationale
    
    We select {method_name} for the following reasons:
    
    ### 1. Problem-Model Fit
    {problem_fit_explanation}
    
    ### 2. Comparative Advantages
    Compared to alternative approaches:
    {comparative_advantages}
    
    ### 3. Theoretical Support
    {theoretical_support}
    
    ### 4. Practical Considerations
    {practical_considerations}
    """
    
    # 生成问题-模型匹配说明
    problem_fit = explain_problem_model_fit(selected_method, problem_features)
    
    # 生成比较优势
    advantages = extract_comparative_advantages(selected_method, comparison)
    
    # 生成理论支持
    theory = find_theoretical_support(selected_method)
    
    # 生成实际考虑
    practical = explain_practical_considerations(selected_method, problem_features)
    
    return rationale_template.format(
        method_name=selected_method,
        problem_fit_explanation=problem_fit,
        comparative_advantages=advantages,
        theoretical_support=theory,
        practical_considerations=practical
    )
```

### 步骤5: 创新贡献声明生成

```python
def generate_contribution_statement(selected_method, innovations, problem_type):
    """
    生成创新贡献声明
    
    Returns:
        str: Our Contribution 段落
    """
    contribution_template = """
    ## Our Contributions
    
    This paper makes the following contributions:
    
    {contributions_list}
    
    Specifically, our approach differs from existing methods in that:
    {differentiation}
    """
    
    # 识别创新点
    contributions = identify_contributions(selected_method, innovations)
    
    # 生成贡献列表
    contributions_list = format_contributions_list(contributions)
    
    # 生成差异化说明
    differentiation = explain_differentiation(selected_method, problem_type)
    
    return contribution_template.format(
        contributions_list=contributions_list,
        differentiation=differentiation
    )
```

## 对比表格模板

### 标准对比表格

```latex
\begin{table}[h]
\centering
\caption{Comparison of Candidate Methods}
\label{tab:method_comparison}
\begin{tabular}{lccccc}
\toprule
\textbf{Method} & \textbf{Accuracy} & \textbf{Efficiency} & \textbf{Interpretability} & \textbf{Scalability} & \textbf{Innovation} \\
\midrule
Traditional ODE & ★★★★☆ & ★★★☆☆ & ★★★★★ & ★★☆☆☆ & ★★☆☆☆ \\
Pure Neural Net & ★★★★★ & ★★★★☆ & ★☆☆☆☆ & ★★★★★ & ★★★☆☆ \\
\textbf{PINN (Ours)} & ★★★★★ & ★★★★☆ & ★★★★☆ & ★★★★☆ & ★★★★★ \\
\bottomrule
\end{tabular}
\end{table}
```

### 量化对比表格

```latex
\begin{table}[h]
\centering
\caption{Quantitative Comparison of Methods}
\label{tab:quantitative_comparison}
\begin{tabular}{lcccc}
\toprule
\textbf{Method} & \textbf{RMSE} & \textbf{Time (s)} & \textbf{Parameters} & \textbf{Data Required} \\
\midrule
Linear Regression & 0.142 & 0.01 & 15 & 100 \\
Random Forest & 0.098 & 0.52 & 1,234 & 500 \\
Neural Network & 0.067 & 12.3 & 15,678 & 2,000 \\
\textbf{Our Hybrid Model} & \textbf{0.043} & 8.7 & 8,456 & 1,000 \\
\bottomrule
\end{tabular}
\end{table}
```

## 选择理由模板

### 模板1: 问题驱动型

```markdown
The nature of [problem description] presents several unique challenges:
1. [Challenge 1]
2. [Challenge 2]
3. [Challenge 3]

These characteristics make [selected method] particularly suitable because:
- [Reason 1]: [Explanation with evidence]
- [Reason 2]: [Explanation with evidence]
- [Reason 3]: [Explanation with evidence]

In contrast, traditional approaches such as [alternative method] fail to address 
[specific limitation], as demonstrated by [evidence/citation].
```

### 模板2: 对比论证型

```markdown
We considered three main approaches for this problem:

**Approach A: [Method Name]**
- Strengths: [List strengths]
- Limitations: [List limitations relevant to this problem]

**Approach B: [Method Name]**
- Strengths: [List strengths]
- Limitations: [List limitations relevant to this problem]

**Our Approach: [Selected Method]**
- Addresses limitation X of Approach A through [mechanism]
- Overcomes challenge Y of Approach B via [mechanism]
- Achieves [specific improvement] compared to both alternatives
```

### 模板3: 创新突出型

```markdown
Existing methods for [problem type] typically rely on [common approach], 
which suffers from [limitation]. Our key insight is that [novel perspective].

This leads us to propose [method name], which:
1. **Innovates** by [key innovation]
2. **Improves** upon existing work by [specific improvement]
3. **Enables** [new capability] that was previously impossible

Preliminary experiments show that our approach achieves [quantitative result], 
representing a [X%] improvement over the state-of-the-art.
```

## 创新贡献声明模板

### 标准格式

```markdown
## Our Contributions

The main contributions of this paper are as follows:

1. **Novel Modeling Framework**: We propose [model name], which is the first 
   to [novel aspect]. This addresses the key challenge of [problem] that 
   existing methods cannot handle effectively.

2. **Theoretical Analysis**: We provide [theoretical contribution], which 
   establishes [theoretical guarantee/insight]. This extends the work of 
   [previous work] to [new domain/setting].

3. **Algorithmic Innovation**: We develop [algorithm name] that achieves 
   [performance improvement] while reducing [cost/complexity] by [amount].

4. **Comprehensive Validation**: We validate our approach on [dataset/scenario],
   demonstrating [key result] compared to [baseline methods].
```

### 简洁格式

```markdown
**Key Innovation**: Unlike traditional [approach] that [limitation], 
our method [key difference], enabling [capability/improvement].
```

## 输出格式

```json
{
  "justification_section": {
    "latex": "\\section{Model Selection and Justification}...",
    "markdown": "## Model Selection and Justification..."
  },
  "comparison_analysis": {
    "dimensions": ["accuracy", "efficiency", "interpretability", "scalability"],
    "comparison_table": [
      {
        "method": "Traditional ODE Solver",
        "accuracy": {"score": 4, "explanation": "High accuracy for smooth solutions"},
        "efficiency": {"score": 3, "explanation": "Moderate computational cost"},
        "interpretability": {"score": 5, "explanation": "Full physical interpretation"},
        "scalability": {"score": 2, "explanation": "Limited to moderate problem sizes"}
      },
      {
        "method": "PINN (Ours)",
        "accuracy": {"score": 5, "explanation": "Achieves 0.043 RMSE"},
        "efficiency": {"score": 4, "explanation": "8.7s training time"},
        "interpretability": {"score": 4, "explanation": "Physics-informed architecture"},
        "scalability": {"score": 4, "explanation": "Handles large-scale problems"}
      }
    ],
    "latex_table": "\\begin{table}...",
    "summary": "Our PINN approach outperforms alternatives in 3 out of 4 dimensions..."
  },
  "selection_rationale": {
    "problem_fit": "The problem exhibits nonlinear dynamics with sparse data...",
    "advantages": ["Higher accuracy (12% improvement)", "Better scalability", "Physical consistency"],
    "theoretical_support": ["Universal approximation [1]", "Physics-informed learning [2]"],
    "full_text": "We select PINN for the following reasons..."
  },
  "contribution_statement": {
    "contributions": [
      "Novel hybrid PINN-transformer architecture",
      "Theoretical convergence analysis",
      "23% accuracy improvement over baselines"
    ],
    "full_text": "The main contributions of this paper are..."
  },
  "model_diagram": {
    "tikz_code": "\\begin{tikzpicture}...",
    "description": "Architecture diagram showing the hybrid model structure"
  }
}
```

## 质量检查清单

### 必须包含的元素

- [ ] 至少对比2种备选方法
- [ ] 每种方法有具体优缺点说明
- [ ] 选择理由有量化依据
- [ ] 创新点有明确陈述
- [ ] 包含对比表格

### 应避免的问题

- ❌ 仅列出方法名称而无分析
- ❌ 选择理由过于主观（"我们认为..."）
- ❌ 缺乏与备选方法的对比
- ❌ 创新点描述模糊
- ❌ 过度吹嘘（使用过多"novel"、"innovative"）

## 与其他技能集成

### 上游依赖
- `problem-analyzer`: 获取问题特征分析
- `model-selector`: 获取候选模型列表
- `hybrid-model-designer`: 获取混合模型设计

### 下游输出
- `section-writer`: 提供模型选择章节内容
- `abstract-generator`: 提供方法概述素材
- `chart-generator`: 提供对比图表数据

## 使用示例

```python
# 生成模型选择论证
from model_justification_generator import generate_justification

problem_features = {
    'problem_type': 'continuous',
    'data_scale': 'medium',
    'physical_laws': True,
    'uncertainty_level': 'moderate'
}

selected_model = {
    'name': 'Physics-Informed Neural Network',
    'innovations': ['Hybrid architecture', 'Physical constraints integration']
}

alternatives = [
    {'name': 'Traditional FEM', 'type': 'traditional'},
    {'name': 'Pure Neural Network', 'type': 'modern'}
]

result = generate_justification(
    selected_model=selected_model,
    alternatives=alternatives,
    problem_features=problem_features,
    problem_type='A'
)

print(result['justification_section']['latex'])
```

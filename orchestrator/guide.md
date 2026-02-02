---
name: mcm-orchestrator
description: MCM/ICM论文生成的主编排器。负责协调所有子技能的执行、管理状态、处理错误恢复。基于AgentOrchestra架构实现层级式多智能体协调。
---

# MCM/ICM 主编排器

## 功能概述

主编排器是整个系统的核心，负责：
1. 解析用户输入并初始化流程
2. 按阶段协调各子技能执行
3. 管理全局状态和检查点
4. 处理错误和执行恢复策略

## 执行流程

```
用户输入 → 初始化 → 阶段1-10顺序执行 → 输出论文
              ↓
         错误处理 ← 检查点恢复
```

## 阶段技能映射

```python
PHASE_SKILLS = {
    1: ['problem-parser', 'problem-type-classifier', 'data-collector', 
        'literature-searcher', 'citation-validator'],
    2: ['problem-decomposer', 'sub-problem-analyzer', 'assumption-generator',
        'variable-definer', 'constraint-identifier'],
    3: ['model-selector', 'hybrid-model-designer', 'model-builder',
        'model-solver', 'code-verifier'],
    4: ['sensitivity-analyzer', 'uncertainty-quantifier', 'model-validator',
        'strengths-weaknesses', 'ethical-analyzer'],
    5: ['section-writer', 'fact-checker', 'abstract-generator',
        'abstract-iterative-optimizer', 'memo-letter-writer'],
    6: ['chart-generator', 'publication-scaler', 'table-formatter',
        'figure-validator'],
    7: ['latex-compiler', 'compilation-error-handler', 'citation-manager',
        'format-checker', 'anonymization-checker'],
    8: ['quality-reviewer', 'hallucination-detector', 'grammar-checker',
        'consistency-checker'],
    9: ['final-polisher', 'academic-english-optimizer', 'submission-preparer'],
    10: ['pre-submission-validator', 'submission-checklist']
}
```

## 并行执行策略

阶段1中以下技能可以并行执行：
- problem-parser + data-collector + literature-searcher

其他阶段按顺序执行以确保依赖关系。

## 错误恢复

参见 `error_recovery.py` 了解详细的错误分类和恢复策略。

## 状态管理

参见 `state_manager.py` 了解状态持久化和检查点机制。

## 使用示例

```python
from orchestrator.mcm_orchestrator import MCMOrchestrator

orchestrator = MCMOrchestrator(config)
result = await orchestrator.execute_pipeline({
    'problem_text': '...',
    'problem_type': 'A',
    'data_files': [...],
    'team_control_number': 'XXXXX'
})
```

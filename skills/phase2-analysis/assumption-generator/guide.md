---
name: assumption-generator
description: 为美赛建模生成合理的假设并提供充分论证。遵循O奖标准，确保每个假设都有现实依据或数学依据，假设间保持一致性。
---

# 假设生成器 (Assumption Generator)

## 功能概述

为数学建模生成合理的假设，并提供充分的论证。

## O奖标准

> "假设的合理性比假设本身更重要，即使是奇怪的假设也能通过充分论证圆回来"

### 核心原则
1. **每个假设必须有论证**
2. **假设间保持一致性**
3. **考虑假设对结果的影响**
4. **提供敏感性说明**

## 假设类型

### 1. 简化假设 (Simplifying Assumptions)
减少模型复杂度的假设
- 忽略某些次要因素
- 线性化非线性关系
- 离散化连续过程

### 2. 数据假设 (Data Assumptions)
关于数据质量和完整性的假设
- 数据准确性
- 缺失数据处理
- 数据代表性

### 3. 系统假设 (System Assumptions)
关于系统行为的假设
- 稳态假设
- 封闭系统假设
- 独立性假设

### 4. 边界假设 (Boundary Assumptions)
关于问题范围的假设
- 时间范围
- 空间范围
- 考虑因素范围

## 输出格式

```markdown
## Assumptions

### Assumption 1: Steady-State Condition
**Statement**: We assume the system operates under steady-state conditions, where all variables remain constant over time.

**Justification**: 
- The problem description indicates long-term behavior analysis
- Historical data shows minimal fluctuation over the relevant time period
- This simplification allows the use of algebraic equations instead of differential equations

**Impact Analysis**: 
This assumption reduces model complexity from dynamic to static, potentially missing transient behaviors but capturing equilibrium outcomes.

**Sensitivity**: 
If this assumption is violated, model predictions may deviate by up to 15% based on preliminary sensitivity analysis.

---

### Assumption 2: Data Completeness
**Statement**: We assume the provided dataset is representative and complete for the geographic regions considered.

**Justification**:
- The data source (World Bank) is authoritative and regularly validated
- We performed data quality checks showing <5% missing values
- Missing values were interpolated using established statistical methods

**Impact Analysis**:
Data quality directly affects model accuracy. Our preprocessing ensures data integrity while preserving statistical properties.

**Sensitivity**:
Alternative imputation methods were tested, showing results consistent within ±3%.
```

## 生成流程

1. **问题分析**: 识别需要假设的领域
2. **假设生成**: 为每个领域生成合理假设
3. **论证构建**: 为每个假设提供论证
4. **一致性检查**: 确保假设间不矛盾
5. **影响评估**: 评估假设对结果的影响
6. **格式化输出**: 按标准格式输出

## 常见假设领域

| 领域 | 典型假设 | 论证方向 |
|------|---------|---------|
| 时间 | 稳态/动态 | 问题时间尺度 |
| 空间 | 均匀/非均匀 | 空间相关性 |
| 数据 | 准确/有噪声 | 数据源可靠性 |
| 关系 | 线性/非线性 | 经验/理论支持 |
| 独立性 | 独立/相关 | 物理机制 |

## 相关技能

- `problem-decomposer` - 问题分解
- `variable-definer` - 变量定义
- `sensitivity-analyzer` - 敏感性分析

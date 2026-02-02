---
name: constraint-identifier
description: 识别和形式化美赛建模中的约束条件。将文字描述的约束转化为数学表达式，分类整理各类约束，确保约束的完整性和一致性。
---

# 约束识别器 (Constraint Identifier)

## 功能概述

识别问题中的各类约束条件，将其形式化为数学表达式。

## 约束分类

### 1. 物理约束 (Physical Constraints)
由自然规律决定的约束
- 能量守恒
- 质量守恒
- 物理定律限制

### 2. 资源约束 (Resource Constraints)
资源有限性带来的约束
- 预算限制
- 容量限制
- 人力限制

### 3. 逻辑约束 (Logical Constraints)
逻辑关系带来的约束
- 互斥约束
- 蕴含约束
- 优先级约束

### 4. 边界约束 (Boundary Constraints)
变量取值范围的约束
- 非负约束
- 上下界约束
- 整数约束

### 5. 时序约束 (Temporal Constraints)
时间关系带来的约束
- 先后顺序
- 时间窗口
- 周期性要求

## 输出格式

### 数学表达式格式

```latex
\subsection{Constraints}

\textbf{Resource Constraints:}
\begin{align}
    \sum_{i \in I} x_i &\leq B & \text{(Budget constraint)} \\
    x_i &\leq C_i & \forall i \in I & \text{(Capacity constraint)}
\end{align}

\textbf{Physical Constraints:}
\begin{align}
    \sum_{j} f_{ij} - \sum_{j} f_{ji} &= d_i & \forall i \in I & \text{(Flow conservation)}
\end{align}

\textbf{Logical Constraints:}
\begin{align}
    y_i &\leq M \cdot z_i & \forall i \in I & \text{(Big-M linking constraint)} \\
    z_i + z_j &\leq 1 & \forall (i,j) \in E & \text{(Mutual exclusion)}
\end{align}

\textbf{Boundary Constraints:}
\begin{align}
    x_i &\geq 0 & \forall i \in I & \text{(Non-negativity)} \\
    z_i &\in \{0, 1\} & \forall i \in I & \text{(Binary)}
\end{align}
```

### JSON 格式

```json
{
  "constraints": [
    {
      "id": "C1",
      "type": "resource",
      "name": "Budget constraint",
      "description": "Total investment cannot exceed budget",
      "mathematical_form": "\\sum_{i \\in I} x_i \\leq B",
      "variables_involved": ["x_i", "B"],
      "is_linear": true,
      "is_equality": false
    },
    {
      "id": "C2",
      "type": "physical",
      "name": "Flow conservation",
      "description": "Inflow equals outflow plus demand at each node",
      "mathematical_form": "\\sum_{j} f_{ij} - \\sum_{j} f_{ji} = d_i",
      "variables_involved": ["f_ij", "d_i"],
      "is_linear": true,
      "is_equality": true
    },
    {
      "id": "C3",
      "type": "boundary",
      "name": "Non-negativity",
      "description": "Variables must be non-negative",
      "mathematical_form": "x_i \\geq 0",
      "variables_involved": ["x_i"],
      "is_linear": true,
      "is_equality": false
    }
  ],
  "summary": {
    "total_constraints": 10,
    "linear_constraints": 8,
    "nonlinear_constraints": 2,
    "equality_constraints": 3,
    "inequality_constraints": 7
  }
}
```

## 识别流程

1. **文本分析**: 从题目中提取约束描述
2. **分类归档**: 按类型对约束进行分类
3. **形式化**: 将文字描述转化为数学表达式
4. **变量关联**: 识别约束涉及的变量
5. **一致性检查**: 确保约束间不矛盾
6. **完整性检查**: 确保没有遗漏重要约束

## 常见约束模式

| 约束类型 | 数学形式 | 适用场景 |
|---------|---------|---------|
| 容量 | $x \leq C$ | 资源限制 |
| 守恒 | $\sum_{in} = \sum_{out}$ | 流量问题 |
| 分配 | $\sum x_i = 1$ | 比例分配 |
| Big-M | $y \leq Mz$ | 逻辑连接 |
| 互斥 | $z_1 + z_2 \leq 1$ | 选择问题 |

## 相关技能

- `variable-definer` - 变量定义
- `model-builder` - 模型构建

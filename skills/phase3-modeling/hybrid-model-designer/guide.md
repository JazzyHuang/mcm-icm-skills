---
name: hybrid-model-designer
description: 设计创新的混合数学模型。通过组合不同类型的模型，创造具有创新性的建模方法，提升论文的创新分数。基于2025年Hybrid Modeling Design Patterns。
---

# 混合模型设计器 (Hybrid Model Designer)

## 功能概述

设计具有创新性的混合数学模型，通过组合不同方法来获得更好的效果。

## 混合模式 (Design Patterns)

### 1. 串联混合 (Serial Hybrid)
```
物理模型 → 数据模型修正

输入 → [物理模型] → 初步结果 → [ML模型] → 修正结果 → 输出
```

**适用场景**: 有物理先验知识但需要数据修正
**示例**: ODE模型输出 + 神经网络残差学习

### 2. 并联混合 (Parallel Hybrid)
```
        ┌→ [物理模型] → 结果1 ─┐
输入 ──┤                      ├→ [融合] → 输出
        └→ [数据模型] → 结果2 ─┘
```

**适用场景**: 两种方法各有优势
**融合方法**: 加权平均、投票、堆叠

### 3. 嵌入式混合 (Embedded Hybrid)
```
[物理模型] 中的某些组件由 [ML模型] 替代

y = f(x) + ε  →  y = f(x) + NN(x)
```

**适用场景**: 物理模型有系统偏差
**示例**: ML学习物理模型的残差

### 4. 约束式混合 (Constrained Hybrid)
```
min Loss(NN(x), y)
s.t. Physics_Law(NN(x)) = 0
```

**适用场景**: 需要保证物理一致性
**示例**: 物理信息神经网络(PINN)

## 创新组合建议

### A题(连续)创新
| 基础模型 | 混合方向 | 创新点 |
|---------|---------|--------|
| ODE | + 机器学习 | 自动参数估计 |
| PDE | + 神经网络 | Physics-Informed NN |
| 有限元 | + 优化 | 形状优化 |

### B题(离散)创新
| 基础模型 | 混合方向 | 创新点 |
|---------|---------|--------|
| 图论 | + 强化学习 | 自适应路径规划 |
| DP | + 近似算法 | 大规模求解 |
| 整数规划 | + 启发式 | 列生成/分支定界 |

### C题(数据)创新
| 基础模型 | 混合方向 | 创新点 |
|---------|---------|--------|
| 时间序列 | + 深度学习 | 多尺度特征融合 |
| 集成学习 | + 可解释性 | SHAP值分析 |
| 聚类 | + 网络 | 社区发现 |

### D题(运筹)创新
| 基础模型 | 混合方向 | 创新点 |
|---------|---------|--------|
| MILP | + 元启发式 | 大规模问题 |
| 网络流 | + 随机优化 | 不确定性建模 |
| 调度 | + 仿真 | 在线调整 |

### E题(可持续)创新
| 基础模型 | 混合方向 | 创新点 |
|---------|---------|--------|
| 多目标 | + 不确定性 | 鲁棒Pareto |
| 系统动力学 | + Agent | 行为异质性 |
| 生态模型 | + 优化 | 最优控制 |

### F题(政策)创新
| 基础模型 | 混合方向 | 创新点 |
|---------|---------|--------|
| 博弈论 | + 网络 | 演化博弈 |
| ABM | + ML | 行为学习 |
| 仿真 | + 因果 | 政策效果分析 |

## 输出格式

```json
{
  "hybrid_model": {
    "name": "Physics-Informed Neural Network for Heat Transfer",
    "pattern": "constrained",
    "components": [
      {
        "type": "physics",
        "model": "Heat Equation PDE",
        "role": "constraint"
      },
      {
        "type": "data",
        "model": "Neural Network",
        "role": "function_approximator"
      }
    ],
    "innovation_points": [
      "将物理定律作为损失函数的一部分",
      "无需标注数据，只需边界条件",
      "可处理逆问题"
    ],
    "implementation": {
      "framework": "PyTorch + DeepXDE",
      "key_equations": [
        "Loss = MSE_data + λ * MSE_physics"
      ]
    },
    "expected_improvement": "精度提升20%，数据需求减少50%"
  }
}
```

## 相关技能

- `model-selector` - 模型选择
- `model-builder` - 模型构建

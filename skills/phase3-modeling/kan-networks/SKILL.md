---
name: kan-networks
description: Kolmogorov-Arnold Networks (KAN) 模块，2025年ICLR重大创新。与传统MLP不同，KAN将可学习激活函数放在边上而非节点上，参数效率提升100倍，可解释性强。适用于PDE求解、符号回归、物理定律发现。创新性评分0.98。
---

# Kolmogorov-Arnold Networks (KAN)

## 功能概述

KAN是2025年最具影响力的神经网络架构创新（ICLR 2025接收），核心特点：
1. **边上的可学习激活函数**：使用B-spline代替固定激活函数
2. **参数效率**：2层宽度10的KAN ≈ 4层宽度100的MLP精度
3. **可解释性**：学习的是1D函数，可直接可视化
4. **神经缩放定律更快**：收敛速度更快

## 创新性评分：0.98/1.0

## 与MLP对比

| 特性 | MLP | KAN |
|-----|-----|-----|
| 激活函数位置 | 节点 | 边 |
| 激活函数类型 | 固定(ReLU等) | 可学习(B-spline) |
| 参数效率 | 基准 | 100倍提升 |
| 可解释性 | 黑盒 | 可视化函数 |
| 神经缩放 | α=4 | α=3(更快) |

## 适用场景

| 场景 | 特点 | MCM题型 |
|-----|------|---------|
| PDE求解 | 物理约束 + 高精度 | A |
| 符号回归 | 发现数学公式 | A, C |
| 物理定律发现 | 从数据中提取定律 | A |
| 函数逼近 | 少参数高精度 | 通用 |

## 使用方法

### 基础使用

```python
from kan_networks import KAN

# 创建KAN网络
model = KAN(
    layers=[2, 5, 5, 1],  # 输入2维，两个隐藏层各5节点，输出1维
    grid_size=5,          # B-spline网格点数
    spline_order=3        # 样条阶数
)

# 训练
model.fit(X_train, y_train, epochs=1000)

# 预测
y_pred = model.predict(X_test)

# 可视化学到的函数
model.plot_activations(save_path='figures/kan_functions.pdf')
```

### 符号回归（公式发现）

```python
from kan_networks import KANSymbolicRegression

# 创建符号回归器
sr = KANSymbolicRegression(
    candidate_functions=['sin', 'cos', 'exp', 'log', 'sqrt', 'x^2']
)

# 从数据中发现公式
formula = sr.fit_and_extract(X, y)
print(f"发现的公式: {formula}")
# 输出示例: y = 0.5 * sin(x1) + x2^2
```

### 与PINN结合（物理信息KAN）

```python
from kan_networks import PhysicsInformedKAN

# 物理信息KAN
pikan = PhysicsInformedKAN(
    layers=[2, 10, 10, 1],
    physics_loss_fn=heat_equation_residual
)

pikan.train(
    collocation_points=interior_points,
    boundary_conditions=bc_data,
    epochs=10000
)
```

## 输出格式

```json
{
  "model": {
    "type": "KAN",
    "architecture": [2, 5, 5, 1],
    "spline_order": 3,
    "total_params": 150
  },
  "training": {
    "epochs": 1000,
    "final_loss": 1.2e-6
  },
  "interpretability": {
    "extracted_formula": "0.5*sin(x1) + x2^2",
    "function_plots": "kan_functions.pdf"
  },
  "comparison": {
    "mlp_params_for_same_accuracy": 15000,
    "efficiency_gain": "100x"
  }
}
```

## O奖加分建议

- 展示KAN学到的激活函数可视化
- 与同精度MLP对比参数量
- 如果可能，提取符号公式
- 强调"2025年ICLR最新方法"

## 相关技能

- `physics-informed-nn` - PINN模块
- `neural-operators` - FNO神经算子
- `model-explainer` - 模型解释

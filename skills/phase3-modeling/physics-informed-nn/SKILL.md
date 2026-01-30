---
name: physics-informed-nn
description: 物理信息神经网络(PINN)模块，将物理定律嵌入神经网络训练过程。适用于A题连续问题（流体力学、热传导、波动方程等），支持多种PDE/ODE约束，提供与传统求解器的对比验证。2025年SOTA方法，创新性极高。
---

# 物理信息神经网络 (Physics-Informed Neural Network)

## 功能概述

将物理定律（PDE/ODE）作为软约束嵌入神经网络损失函数，实现：
1. 用少量数据解决正问题（求解PDE）
2. 数据驱动的参数反演（逆问题）
3. 物理一致性保证

## 创新性评分：0.95/1.0

## 适用场景

| 问题类型 | 典型方程 | 应用场景 |
|---------|---------|---------|
| 热传导 | ∂u/∂t = α∇²u | 温度分布预测 |
| 波动 | ∂²u/∂t² = c²∇²u | 声波、地震波 |
| 流体 | Navier-Stokes | 流场模拟 |
| 扩散 | ∂u/∂t = D∇²u | 污染物扩散 |
| 对流扩散 | ∂u/∂t + v·∇u = D∇²u | 物质输运 |

## 核心架构

```
输入层 (x, t) → 隐藏层 (DNN) → 输出层 u(x,t)
                    ↓
           自动微分计算导数
                    ↓
    Loss = Loss_data + λ·Loss_physics + Loss_BC + Loss_IC
```

## 使用方法

### 1. 基础使用

```python
from pinn import PINNSolver, HeatEquation

# 定义物理方程
equation = HeatEquation(alpha=0.01)

# 定义边界条件
boundary_conditions = {
    'dirichlet': {'x=0': 0, 'x=1': 0},
    'initial': lambda x: np.sin(np.pi * x)
}

# 创建并训练求解器
solver = PINNSolver(
    equation=equation,
    domain={'x': (0, 1), 't': (0, 1)},
    boundary_conditions=boundary_conditions,
    network_config={'layers': [2, 64, 64, 64, 1]}
)
solver.train(epochs=20000)

# 预测
u_pred = solver.predict(x_test, t_test)
```

### 2. 高级功能

```python
# 自适应采样（残差大的区域增加采样点）
solver.enable_adaptive_sampling(method='RAR', threshold=1e-3)

# 不确定性量化（集成方法）
ensemble = solver.create_ensemble(n_models=5)
mean, std = ensemble.predict_with_uncertainty(x_test, t_test)

# 与传统方法对比
comparison = solver.compare_with_fdm(dx=0.01, dt=0.001)
```

## 支持的方程

- `HeatEquation` - 热传导方程
- `WaveEquation` - 波动方程  
- `BurgersEquation` - Burgers方程
- `DiffusionEquation` - 扩散方程
- `AdvectionDiffusionEquation` - 对流扩散方程
- `PoissonEquation` - Poisson方程
- `NavierStokes2D` - 2D Navier-Stokes

## 输出

```json
{
  "model": {"type": "PINN", "equation": "HeatEquation"},
  "training": {"epochs": 20000, "final_loss": 1.2e-5},
  "validation": {"l2_error": 0.0023, "max_error": 0.0089},
  "figures": ["pinn_solution.pdf", "pinn_loss.pdf", "pinn_error.pdf"]
}
```

## O奖加分建议

- 与传统有限差分/有限元结果对比验证
- 展示PINN在稀疏数据下的优势
- 进行参数敏感性分析
- 可视化物理残差分布

## 相关技能

- `neural-operators` - FNO/DeepONet（更快的PDE求解）
- `kan-networks` - KAN网络（更高精度）
- `sensitivity-analyzer` - 敏感性分析

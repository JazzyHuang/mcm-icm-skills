---
name: neural-operators
description: 神经算子模块，支持FNO(傅里叶神经算子)和DeepONet。PDE求解速度比传统方法快1000倍，分辨率无关，适用于A题连续问题。创新性评分0.96。
---

# 神经算子 (Neural Operators)

## 功能概述

神经算子学习无穷维函数空间之间的映射（算子），核心优势：
1. **分辨率无关**：训练后可在任意网格上推理
2. **极速推理**：比传统FEM/FDM快1000倍
3. **泛化能力强**：可处理参数化PDE家族

## 创新性评分：0.96/1.0

## 支持的算子

| 算子 | 特点 | 适用场景 |
|-----|------|---------|
| FNO | 傅里叶空间核参数化 | 规则网格PDE |
| DeepONet | Branch-Trunk架构 | 非均匀观测 |
| PINO | 物理信息神经算子 | 小数据PDE |

## 与传统方法对比

| 方法 | 单次求解时间 | 分辨率变化 | 参数泛化 |
|-----|------------|-----------|---------|
| FDM | 秒级 | 需重新计算 | 需重新计算 |
| FEM | 分钟级 | 需重新网格 | 需重新计算 |
| FNO | 毫秒级 | 直接推理 | 直接推理 |

## 使用方法

### FNO (傅里叶神经算子)

```python
from neural_operators import FNO2d

# 创建FNO模型
model = FNO2d(
    modes1=12,      # 傅里叶模式数（x方向）
    modes2=12,      # 傅里叶模式数（y方向）
    width=32,       # 通道宽度
    in_channels=1,  # 输入通道（如初始条件）
    out_channels=1  # 输出通道（如解场）
)

# 训练
model.fit(initial_conditions, solutions, epochs=500)

# 预测（任意分辨率！）
u_pred = model.predict(new_initial_condition)
```

### DeepONet

```python
from neural_operators import DeepONet

# 创建DeepONet
model = DeepONet(
    branch_layers=[100, 100, 100],  # 处理输入函数
    trunk_layers=[2, 100, 100],     # 处理查询位置
    output_dim=1
)

# 训练
model.fit(input_functions, query_points, output_values)

# 预测
u_pred = model.predict(new_function, new_query_points)
```

### PINO (物理信息神经算子)

```python
from neural_operators import PINO

# 物理信息FNO
model = PINO(
    modes=12,
    width=32,
    pde_loss_fn=navier_stokes_residual
)

# 训练（结合数据和物理）
model.fit(
    data=(X_data, y_data),
    collocation_points=interior_points,
    lambda_physics=0.1
)
```

## 输出格式

```json
{
  "model": {
    "type": "FNO2d",
    "modes": [12, 12],
    "width": 32
  },
  "training": {
    "epochs": 500,
    "final_loss": 1.2e-4
  },
  "performance": {
    "inference_time_ms": 2.3,
    "traditional_fdm_time_ms": 2300,
    "speedup": "1000x"
  },
  "error": {
    "relative_l2": 0.012
  }
}
```

## O奖加分建议

- 展示FNO vs FDM/FEM的速度对比
- 展示分辨率泛化能力（低分辨率训练→高分辨率推理）
- 强调"1000倍加速"的实际价值
- 对于不确定参数，展示参数空间的快速扫描

## 相关技能

- `physics-informed-nn` - PINN模块
- `kan-networks` - KAN网络
- `sensitivity-analyzer` - 敏感性分析

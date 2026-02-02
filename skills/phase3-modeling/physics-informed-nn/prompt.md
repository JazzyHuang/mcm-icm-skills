# PINN实现任务 (Physics-Informed Neural Network)

## 角色

你是物理信息神经网络(PINN)专家，负责为当前问题设计和实现PINN模型。PINN是2025年前沿方法，创新性评分0.95，是O奖论文的有力武器。

## 输入

- `problem_type`: 题目类型 (通常为A类)
- `governing_equations`: 控制方程（PDE/ODE）
- `boundary_conditions`: 边界条件
- `initial_conditions`: 初始条件
- `domain`: 计算域
- `data_points`: 观测数据（如有）

---

## PINN核心原理

PINN通过在损失函数中嵌入物理约束，使神经网络学习满足物理定律的解：

$$\mathcal{L}_{total} = \lambda_d \mathcal{L}_{data} + \lambda_p \mathcal{L}_{physics} + \lambda_b \mathcal{L}_{boundary}$$

其中：
- $\mathcal{L}_{data}$: 数据损失（与观测数据的误差）
- $\mathcal{L}_{physics}$: 物理损失（PDE残差）
- $\mathcal{L}_{boundary}$: 边界损失（边界条件满足程度）

---

## 输出要求

### 1. 网络架构设计

```python
import torch
import torch.nn as nn
import numpy as np

class PINN(nn.Module):
    """
    Physics-Informed Neural Network
    
    用于求解偏微分方程的神经网络，通过物理约束正则化。
    
    Args:
        input_dim: 输入维度 (空间+时间)
        hidden_dims: 隐藏层维度列表
        output_dim: 输出维度
        activation: 激活函数 ('tanh', 'sin', 'gelu')
    """
    
    def __init__(
        self, 
        input_dim: int = 2,
        hidden_dims: list = [64, 64, 64, 64],
        output_dim: int = 1,
        activation: str = 'tanh'
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 选择激活函数
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sin':
            self.activation = torch.sin
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # 构建网络层
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.layers = nn.ModuleList(layers)
        
        # Xavier初始化
        self._init_weights()
    
    def _init_weights(self):
        """Xavier初始化权重"""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation(layer(x))
        return self.layers[-1](x)
```

### 2. 物理约束嵌入

```python
class PhysicsLoss:
    """
    物理约束损失函数
    
    实现各种PDE的残差计算
    """
    
    @staticmethod
    def heat_equation_residual(model, x, t, alpha=0.1):
        """
        热传导方程残差: u_t = α * u_xx
        
        Args:
            model: PINN模型
            x: 空间坐标
            t: 时间坐标
            alpha: 热扩散系数
        """
        x.requires_grad_(True)
        t.requires_grad_(True)
        
        inputs = torch.cat([x, t], dim=1)
        u = model(inputs)
        
        # 计算导数
        u_t = torch.autograd.grad(
            u, t, grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]
        
        u_x = torch.autograd.grad(
            u, x, grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]
        
        u_xx = torch.autograd.grad(
            u_x, x, grad_outputs=torch.ones_like(u_x),
            create_graph=True, retain_graph=True
        )[0]
        
        # PDE残差
        residual = u_t - alpha * u_xx
        return residual
    
    @staticmethod
    def burgers_equation_residual(model, x, t, nu=0.01):
        """
        Burgers方程残差: u_t + u * u_x = ν * u_xx
        """
        x.requires_grad_(True)
        t.requires_grad_(True)
        
        inputs = torch.cat([x, t], dim=1)
        u = model(inputs)
        
        u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
        
        residual = u_t + u * u_x - nu * u_xx
        return residual
    
    @staticmethod
    def navier_stokes_residual(model, x, y, t, Re=100):
        """
        Navier-Stokes方程残差 (2D不可压缩流)
        """
        # 实现NS方程残差...
        pass
```

### 3. 训练策略

```python
class PINNTrainer:
    """
    PINN训练器
    
    实现自适应采样、学习率调度、早停等功能
    """
    
    def __init__(
        self,
        model: PINN,
        physics_loss_fn: callable,
        lambda_data: float = 1.0,
        lambda_physics: float = 1.0,
        lambda_boundary: float = 1.0
    ):
        self.model = model
        self.physics_loss_fn = physics_loss_fn
        self.lambda_data = lambda_data
        self.lambda_physics = lambda_physics
        self.lambda_boundary = lambda_boundary
        
        # 优化器
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=1000, factor=0.5
        )
        
        # 训练历史
        self.history = {
            'total_loss': [],
            'data_loss': [],
            'physics_loss': [],
            'boundary_loss': []
        }
    
    def compute_loss(self, x_data, u_data, x_physics, x_boundary, u_boundary):
        """计算总损失"""
        
        # 数据损失
        u_pred = self.model(x_data)
        loss_data = torch.mean((u_pred - u_data) ** 2)
        
        # 物理损失
        residual = self.physics_loss_fn(self.model, x_physics)
        loss_physics = torch.mean(residual ** 2)
        
        # 边界损失
        u_boundary_pred = self.model(x_boundary)
        loss_boundary = torch.mean((u_boundary_pred - u_boundary) ** 2)
        
        # 总损失
        total_loss = (
            self.lambda_data * loss_data +
            self.lambda_physics * loss_physics +
            self.lambda_boundary * loss_boundary
        )
        
        return total_loss, loss_data, loss_physics, loss_boundary
    
    def train(
        self,
        x_data, u_data,
        x_physics,
        x_boundary, u_boundary,
        epochs: int = 10000,
        print_every: int = 1000
    ):
        """训练PINN"""
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            total_loss, loss_data, loss_physics, loss_boundary = self.compute_loss(
                x_data, u_data, x_physics, x_boundary, u_boundary
            )
            
            total_loss.backward()
            self.optimizer.step()
            self.scheduler.step(total_loss)
            
            # 记录历史
            self.history['total_loss'].append(total_loss.item())
            self.history['data_loss'].append(loss_data.item())
            self.history['physics_loss'].append(loss_physics.item())
            self.history['boundary_loss'].append(loss_boundary.item())
            
            if (epoch + 1) % print_every == 0:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  Total Loss: {total_loss.item():.6f}")
                print(f"  Data Loss: {loss_data.item():.6f}")
                print(f"  Physics Loss: {loss_physics.item():.6f}")
                print(f"  Boundary Loss: {loss_boundary.item():.6f}")
        
        return self.history
    
    def adaptive_sampling(self, x_domain, n_points=1000):
        """
        自适应采样：在残差大的区域增加采样点
        """
        with torch.no_grad():
            residual = self.physics_loss_fn(self.model, x_domain)
            residual_abs = torch.abs(residual).squeeze()
            
            # 根据残差大小进行加权采样
            weights = residual_abs / residual_abs.sum()
            indices = torch.multinomial(weights, n_points, replacement=True)
            
            return x_domain[indices]
```

### 4. 完整实现示例

```python
# 完整的PINN求解热传导方程示例

import torch
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 问题设置
L = 1.0  # 域长度
T = 1.0  # 终止时间
alpha = 0.1  # 热扩散系数

# 创建模型
model = PINN(
    input_dim=2,  # (x, t)
    hidden_dims=[50, 50, 50, 50],
    output_dim=1,
    activation='tanh'
)

# 物理损失函数
def physics_loss(model, inputs):
    x, t = inputs[:, 0:1], inputs[:, 1:2]
    return PhysicsLoss.heat_equation_residual(model, x, t, alpha)

# 生成训练数据
# 内部配点
n_interior = 10000
x_interior = torch.rand(n_interior, 1) * L
t_interior = torch.rand(n_interior, 1) * T
x_physics = torch.cat([x_interior, t_interior], dim=1)

# 边界条件 u(0,t) = u(L,t) = 0
n_boundary = 200
t_bc = torch.rand(n_boundary, 1) * T
x_bc_left = torch.zeros(n_boundary, 1)
x_bc_right = torch.ones(n_boundary, 1) * L
x_boundary = torch.cat([
    torch.cat([x_bc_left, t_bc], dim=1),
    torch.cat([x_bc_right, t_bc], dim=1)
], dim=0)
u_boundary = torch.zeros(2 * n_boundary, 1)

# 初始条件 u(x,0) = sin(πx)
n_initial = 200
x_ic = torch.rand(n_initial, 1) * L
t_ic = torch.zeros(n_initial, 1)
x_data = torch.cat([x_ic, t_ic], dim=1)
u_data = torch.sin(np.pi * x_ic)

# 创建训练器并训练
trainer = PINNTrainer(
    model=model,
    physics_loss_fn=lambda m, x: physics_loss(m, x),
    lambda_data=1.0,
    lambda_physics=1.0,
    lambda_boundary=10.0  # 边界条件权重更高
)

history = trainer.train(
    x_data, u_data,
    x_physics,
    x_boundary, u_boundary,
    epochs=20000,
    print_every=2000
)

# 验证结果
print(f"Final Physics Loss: {history['physics_loss'][-1]:.6f}")
print(f"Final Boundary Loss: {history['boundary_loss'][-1]:.6f}")
```

---

## 与传统方法对比

| 指标 | PINN | 有限元(FEM) | 有限差分(FDM) |
|------|------|-------------|---------------|
| 网格依赖 | 无网格 | 需要网格 | 需要网格 |
| 复杂几何 | 容易 | 困难 | 困难 |
| 计算效率 | 训练慢，推理快 | 一次性计算 | 一次性计算 |
| 数据融合 | 原生支持 | 需要同化 | 需要同化 |
| 创新性 | 0.95 | 0.40 | 0.30 |

---

## 输出格式

```json
{
  "pinn_model": {
    "architecture": {
      "input_dim": 2,
      "hidden_dims": [64, 64, 64, 64],
      "output_dim": 1,
      "activation": "tanh",
      "total_params": 12993
    },
    "physics_constraints": {
      "pde_type": "heat_equation",
      "equation": "u_t = α * u_xx",
      "boundary_conditions": ["Dirichlet: u(0,t)=0, u(L,t)=0"],
      "initial_condition": "u(x,0) = sin(πx)"
    }
  },
  "training_config": {
    "epochs": 20000,
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "lambda_data": 1.0,
    "lambda_physics": 1.0,
    "lambda_boundary": 10.0
  },
  "pinn_results": {
    "final_total_loss": 1.23e-5,
    "final_physics_loss": 8.45e-6,
    "final_boundary_loss": 2.13e-7,
    "training_time_seconds": 245.3,
    "convergence_epoch": 15000
  },
  "comparison_with_fem": {
    "relative_error": 0.023,
    "speedup_factor": 12.5,
    "conclusion": "PINN achieves comparable accuracy with 12.5x faster inference"
  },
  "implementation_code": "完整Python代码"
}
```

---

## 执行说明

1. 确定问题的控制方程（PDE/ODE）
2. 设计网络架构（输入输出维度、隐藏层）
3. 实现物理约束损失函数
4. 生成训练数据（配点、边界点、初始点）
5. 训练模型并监控收敛
6. 与传统方法对比验证
7. 输出完整代码和结果

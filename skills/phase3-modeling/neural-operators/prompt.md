# 神经算子实现任务 (Neural Operators - FNO/DeepONet)

## 角色

你是神经算子专家，负责实现Fourier Neural Operator (FNO)和Deep Operator Network (DeepONet)。神经算子是2025年前沿方法，创新性评分0.96，能够实现分辨率无关的PDE求解。

## 背景

神经算子学习的是**函数到函数的映射**（算子），而非函数本身：
- **FNO**: 在频域进行卷积，实现分辨率无关
- **DeepONet**: 分支-主干架构，学习参数化算子

---

## 输入

- `problem_type`: 题目类型
- `pde_type`: PDE类型
- `input_function`: 输入函数（初始条件/边界条件/强迫项）
- `output_function`: 输出函数（解）
- `resolution`: 网格分辨率

---

## FNO (Fourier Neural Operator)

### 核心原理

FNO在频域进行学习，通过傅里叶变换实现全局感受野：

$$\mathcal{F}[v_{t+1}](k) = \sigma\left(W \cdot \mathcal{F}[v_t](k) + R_\phi(k) \cdot \mathcal{F}[v_t](k)\right)$$

### 完整实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

class SpectralConv2d(nn.Module):
    """
    2D傅里叶层
    
    在频域进行卷积，保留前modes_x和modes_y个频率
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes_x: int,
        modes_y: int
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x  # 保留的x方向频率数
        self.modes_y = modes_y  # 保留的y方向频率数
        
        self.scale = 1 / (in_channels * out_channels)
        
        # 复数权重矩阵
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes_x, modes_y, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes_x, modes_y, dtype=torch.cfloat)
        )
    
    def compl_mul2d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """复数矩阵乘法"""
        # input: (batch, in_channels, x, y)
        # weights: (in_channels, out_channels, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入, shape (batch, channels, size_x, size_y)
            
        Returns:
            输出, shape (batch, out_channels, size_x, size_y)
        """
        batch_size = x.shape[0]
        size_x, size_y = x.shape[-2], x.shape[-1]
        
        # 傅里叶变换
        x_ft = torch.fft.rfft2(x)
        
        # 初始化输出
        out_ft = torch.zeros(
            batch_size, self.out_channels, size_x, size_y // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        
        # 频域乘法（只处理低频部分）
        out_ft[:, :, :self.modes_x, :self.modes_y] = \
            self.compl_mul2d(x_ft[:, :, :self.modes_x, :self.modes_y], self.weights1)
        out_ft[:, :, -self.modes_x:, :self.modes_y] = \
            self.compl_mul2d(x_ft[:, :, -self.modes_x:, :self.modes_y], self.weights2)
        
        # 逆傅里叶变换
        x = torch.fft.irfft2(out_ft, s=(size_x, size_y))
        
        return x


class FNO2d(nn.Module):
    """
    2D Fourier Neural Operator
    
    用于求解2D PDE问题
    """
    
    def __init__(
        self,
        modes_x: int = 12,
        modes_y: int = 12,
        width: int = 32,
        input_dim: int = 3,  # (x, y, a(x,y))
        output_dim: int = 1
    ):
        super().__init__()
        
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.width = width
        
        # 输入提升层
        self.fc0 = nn.Linear(input_dim, width)
        
        # 傅里叶层
        self.conv0 = SpectralConv2d(width, width, modes_x, modes_y)
        self.conv1 = SpectralConv2d(width, width, modes_x, modes_y)
        self.conv2 = SpectralConv2d(width, width, modes_x, modes_y)
        self.conv3 = SpectralConv2d(width, width, modes_x, modes_y)
        
        # 局部线性变换
        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)
        
        # 输出投影层
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入, shape (batch, size_x, size_y, input_dim)
            
        Returns:
            输出, shape (batch, size_x, size_y, output_dim)
        """
        # 提升维度
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)  # (batch, width, size_x, size_y)
        
        # 傅里叶层 + 残差连接
        x1 = self.conv0(x) + self.w0(x)
        x1 = F.gelu(x1)
        
        x2 = self.conv1(x1) + self.w1(x1)
        x2 = F.gelu(x2)
        
        x3 = self.conv2(x2) + self.w2(x2)
        x3 = F.gelu(x3)
        
        x4 = self.conv3(x3) + self.w3(x3)
        
        # 投影回输出空间
        x = x4.permute(0, 2, 3, 1)  # (batch, size_x, size_y, width)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        
        return x


class FNOTrainer:
    """FNO训练器"""
    
    def __init__(self, model: FNO2d, lr: float = 1e-3):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=100, gamma=0.5
        )
        self.history = {'train_loss': [], 'val_loss': []}
    
    def train(
        self,
        train_loader,
        val_loader,
        epochs: int = 500,
        print_every: int = 50
    ):
        """训练FNO"""
        for epoch in range(epochs):
            # 训练
            self.model.train()
            train_loss = 0
            for x, y in train_loader:
                self.optimizer.zero_grad()
                y_pred = self.model(x)
                loss = F.mse_loss(y_pred, y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            self.history['train_loss'].append(train_loss)
            
            # 验证
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for x, y in val_loader:
                    y_pred = self.model(x)
                    val_loss += F.mse_loss(y_pred, y).item()
            
            val_loss /= len(val_loader)
            self.history['val_loss'].append(val_loss)
            
            self.scheduler.step()
            
            if (epoch + 1) % print_every == 0:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  Train Loss: {train_loss:.6f}")
                print(f"  Val Loss: {val_loss:.6f}")
        
        return self.history
```

---

## DeepONet (Deep Operator Network)

### 核心原理

DeepONet使用分支-主干架构学习算子：

$$G(u)(y) = \sum_{k=1}^{p} b_k(u) \cdot t_k(y)$$

- **Branch网络**: 编码输入函数 u
- **Trunk网络**: 编码评估位置 y

### 完整实现

```python
class BranchNet(nn.Module):
    """
    分支网络：编码输入函数
    """
    
    def __init__(
        self,
        input_dim: int,  # 输入函数的离散点数
        hidden_dim: int = 128,
        output_dim: int = 100  # 基函数数量
    ):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: 输入函数值, shape (batch, input_dim)
            
        Returns:
            基函数系数, shape (batch, output_dim)
        """
        return self.net(u)


class TrunkNet(nn.Module):
    """
    主干网络：编码评估位置
    """
    
    def __init__(
        self,
        coord_dim: int = 2,  # 坐标维度 (x, y) 或 (x, t)
        hidden_dim: int = 128,
        output_dim: int = 100  # 基函数数量
    ):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(coord_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y: 评估位置, shape (batch, n_points, coord_dim)
            
        Returns:
            基函数值, shape (batch, n_points, output_dim)
        """
        return self.net(y)


class DeepONet(nn.Module):
    """
    Deep Operator Network
    
    学习从输入函数到输出函数的映射（算子）
    """
    
    def __init__(
        self,
        branch_input_dim: int,
        trunk_coord_dim: int = 2,
        hidden_dim: int = 128,
        basis_dim: int = 100
    ):
        super().__init__()
        
        self.branch = BranchNet(branch_input_dim, hidden_dim, basis_dim)
        self.trunk = TrunkNet(trunk_coord_dim, hidden_dim, basis_dim)
        
        # 偏置项
        self.bias = nn.Parameter(torch.zeros(1))
    
    def forward(
        self,
        u: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            u: 输入函数, shape (batch, branch_input_dim)
            y: 评估位置, shape (batch, n_points, coord_dim)
            
        Returns:
            输出函数值, shape (batch, n_points)
        """
        # 分支网络输出
        b = self.branch(u)  # (batch, basis_dim)
        
        # 主干网络输出
        t = self.trunk(y)  # (batch, n_points, basis_dim)
        
        # 内积 + 偏置
        # b: (batch, basis_dim) -> (batch, 1, basis_dim)
        # t: (batch, n_points, basis_dim)
        output = torch.sum(b.unsqueeze(1) * t, dim=-1) + self.bias
        
        return output
    
    def predict_full_field(
        self,
        u: torch.Tensor,
        grid_x: torch.Tensor,
        grid_y: torch.Tensor
    ) -> torch.Tensor:
        """
        预测完整场
        
        Args:
            u: 输入函数
            grid_x, grid_y: 网格坐标
            
        Returns:
            完整场预测
        """
        batch = u.shape[0]
        nx, ny = grid_x.shape[0], grid_y.shape[0]
        
        # 创建网格点
        xx, yy = torch.meshgrid(grid_x, grid_y, indexing='ij')
        y = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
        y = y.unsqueeze(0).expand(batch, -1, -1)
        
        # 预测
        output = self.forward(u, y)
        output = output.reshape(batch, nx, ny)
        
        return output


# ============ 使用示例 ============

def example_fno_darcy_flow():
    """
    示例：用FNO求解Darcy流动问题
    
    -∇·(a(x)∇u(x)) = f(x)
    """
    # 生成模拟数据
    batch_size = 100
    size = 64
    
    # 输入：渗透率场 a(x,y) + 坐标网格
    x = torch.linspace(0, 1, size)
    y = torch.linspace(0, 1, size)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    
    # 模拟渗透率场
    a = torch.rand(batch_size, size, size) * 10 + 1
    
    # 组合输入 (x, y, a)
    grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(batch_size, -1, -1, -1)
    inputs = torch.cat([grid, a.unsqueeze(-1)], dim=-1)  # (batch, size, size, 3)
    
    # 模拟输出（真实情况需要数值求解器生成）
    outputs = torch.rand(batch_size, size, size, 1)
    
    # 创建FNO
    model = FNO2d(
        modes_x=12,
        modes_y=12,
        width=32,
        input_dim=3,
        output_dim=1
    )
    
    print(f"FNO Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # 前向传播测试
    y_pred = model(inputs)
    print(f"Input shape: {inputs.shape}")
    print(f"Output shape: {y_pred.shape}")
    
    return model


def example_deeponet_heat():
    """
    示例：用DeepONet学习热传导算子
    
    学习从初始条件 u_0(x) 到 u(x, t) 的映射
    """
    # 输入函数在100个点采样
    branch_input_dim = 100
    
    # 创建DeepONet
    model = DeepONet(
        branch_input_dim=branch_input_dim,
        trunk_coord_dim=2,  # (x, t)
        hidden_dim=128,
        basis_dim=100
    )
    
    print(f"DeepONet Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # 模拟输入
    batch_size = 32
    n_eval_points = 50
    
    u0 = torch.rand(batch_size, branch_input_dim)  # 初始条件
    y = torch.rand(batch_size, n_eval_points, 2)   # 评估位置 (x, t)
    
    # 前向传播
    output = model(u0, y)
    print(f"Input u0 shape: {u0.shape}")
    print(f"Eval points shape: {y.shape}")
    print(f"Output shape: {output.shape}")
    
    return model


if __name__ == "__main__":
    fno = example_fno_darcy_flow()
    deeponet = example_deeponet_heat()
```

---

## FNO vs DeepONet vs PINN

| 特性 | FNO | DeepONet | PINN |
|------|-----|----------|------|
| 学习对象 | 参数化PDE族 | 算子（函数到函数） | 单个PDE实例 |
| 分辨率无关 | 是 | 是 | 否 |
| 训练数据需求 | 高（需要多组解） | 高 | 低（可无数据） |
| 推理速度 | 极快（1000x加速） | 快 | 中等 |
| 创新分数 | 0.96 | 0.92 | 0.95 |
| 适用场景 | 重复求解同类PDE | 参数化问题 | 单次求解 |

---

## 输出格式

```json
{
  "neural_operator_type": "FNO",
  "architecture": {
    "modes_x": 12,
    "modes_y": 12,
    "width": 32,
    "layers": 4,
    "total_params": 2373921
  },
  "training": {
    "epochs": 500,
    "train_loss": 0.00123,
    "val_loss": 0.00156,
    "training_time_hours": 2.5
  },
  "performance": {
    "inference_time_ms": 0.8,
    "speedup_vs_fem": 1250,
    "relative_error": 0.023
  },
  "resolution_independence": {
    "trained_resolution": 64,
    "tested_resolutions": [32, 64, 128, 256],
    "errors": [0.028, 0.023, 0.021, 0.020],
    "conclusion": "Error decreases with higher resolution, demonstrating resolution independence"
  },
  "innovation_highlights": [
    "实现分辨率无关的PDE求解",
    "推理速度比FEM快1250倍",
    "支持参数化问题的实时预测"
  ]
}
```

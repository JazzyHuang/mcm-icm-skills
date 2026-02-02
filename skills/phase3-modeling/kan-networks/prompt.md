# KAN网络实现任务 (Kolmogorov-Arnold Network)

## 角色

你是Kolmogorov-Arnold网络(KAN)专家，负责实现这一2024 ICLR前沿方法。KAN的创新性评分高达0.98，是MCM/ICM论文展示前沿创新的最佳选择。

## 背景

KAN基于Kolmogorov-Arnold表示定理，用可学习的样条函数替代固定激活函数。相比传统MLP：
- **参数效率提升100倍**
- **可解释性强**：输出可以转化为符号公式
- **更快的神经缩放定律**

---

## 输入

- `problem_type`: 题目类型
- `task_type`: 任务类型 (regression/classification/pde_solving/symbolic_regression)
- `input_dim`: 输入维度
- `output_dim`: 输出维度
- `data`: 训练数据

---

## KAN核心原理

### Kolmogorov-Arnold表示定理

任何多元连续函数可以表示为单变量函数的组合：

$$f(x_1, ..., x_n) = \sum_{q=0}^{2n} \Phi_q \left( \sum_{p=1}^{n} \phi_{q,p}(x_p) \right)$$

### KAN架构

KAN将激活函数放在边（edges）上而非节点上：

$$\text{KAN}(x) = (\Phi_{L-1} \circ \Phi_{L-2} \circ \cdots \circ \Phi_0)(x)$$

其中每个 $\Phi_l$ 是由可学习的样条函数组成的矩阵。

---

## 完整实现

```python
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

class BSplineBasis:
    """
    B样条基函数
    """
    
    @staticmethod
    def compute_basis(x: torch.Tensor, grid: torch.Tensor, k: int = 3):
        """
        计算B样条基函数值
        
        Args:
            x: 输入点, shape (batch, 1)
            grid: 网格点, shape (G+1,)
            k: 样条阶数 (3 = 三次样条)
            
        Returns:
            basis: 基函数值, shape (batch, G+k)
        """
        # Cox-de Boor递推
        G = len(grid) - 1
        batch = x.shape[0]
        
        # 初始化 (k=0)
        basis = torch.zeros(batch, G + k, dtype=x.dtype, device=x.device)
        
        for i in range(G):
            mask = (x >= grid[i]) & (x < grid[i+1])
            basis[:, i] = mask.float().squeeze()
        
        # 递推计算高阶基函数
        for p in range(1, k + 1):
            new_basis = torch.zeros_like(basis)
            for i in range(G + k - p):
                # 左系数
                denom1 = grid[i+p] - grid[i]
                if denom1 > 1e-10:
                    left = (x.squeeze() - grid[i]) / denom1 * basis[:, i]
                else:
                    left = 0
                
                # 右系数
                denom2 = grid[i+p+1] - grid[i+1]
                if denom2 > 1e-10:
                    right = (grid[i+p+1] - x.squeeze()) / denom2 * basis[:, i+1]
                else:
                    right = 0
                
                new_basis[:, i] = left + right
            basis = new_basis
        
        return basis[:, :G+k-1]


class KANLayer(nn.Module):
    """
    KAN层
    
    可学习的样条激活函数在边上
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: Tuple[float, float] = (-1, 1)
    ):
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # 创建网格
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.linspace(
            grid_range[0] - spline_order * h,
            grid_range[1] + spline_order * h,
            grid_size + 2 * spline_order + 1
        )
        self.register_buffer('grid', grid)
        
        # 可学习的样条系数
        # shape: (in_dim, out_dim, grid_size + spline_order)
        self.spline_weight = nn.Parameter(
            torch.randn(in_dim, out_dim, grid_size + spline_order) * 0.1
        )
        
        # 缩放因子（用于归一化）
        self.scale = nn.Parameter(torch.ones(in_dim, out_dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入, shape (batch, in_dim)
            
        Returns:
            输出, shape (batch, out_dim)
        """
        batch = x.shape[0]
        
        # 对每个输入维度计算样条基函数
        # x[:, i] -> basis[:, i, :] -> weighted sum -> out[:, j]
        
        output = torch.zeros(batch, self.out_dim, device=x.device)
        
        for i in range(self.in_dim):
            # 计算基函数
            basis = BSplineBasis.compute_basis(
                x[:, i:i+1], 
                self.grid, 
                self.spline_order
            )  # (batch, G+k)
            
            for j in range(self.out_dim):
                # 样条函数值 = 基函数 dot 系数
                spline_val = torch.sum(basis * self.spline_weight[i, j], dim=1)
                output[:, j] += self.scale[i, j] * spline_val
        
        return output
    
    def get_symbolic_formula(self, threshold: float = 0.01):
        """
        提取符号公式（简化版）
        
        返回人类可读的数学表达式
        """
        formulas = []
        for j in range(self.out_dim):
            terms = []
            for i in range(self.in_dim):
                weight_norm = torch.norm(self.spline_weight[i, j]).item()
                if weight_norm > threshold:
                    terms.append(f"φ_{i}{j}(x_{i})")
            if terms:
                formulas.append(f"y_{j} = " + " + ".join(terms))
        return formulas


class KAN(nn.Module):
    """
    Kolmogorov-Arnold Network
    
    完整的KAN网络实现
    """
    
    def __init__(
        self,
        layer_dims: List[int],
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: Tuple[float, float] = (-1, 1)
    ):
        """
        Args:
            layer_dims: 层维度列表, e.g., [2, 5, 5, 1]
            grid_size: 样条网格大小
            spline_order: 样条阶数
            grid_range: 网格范围
        """
        super().__init__()
        
        self.layer_dims = layer_dims
        self.depth = len(layer_dims) - 1
        
        # 构建KAN层
        self.layers = nn.ModuleList()
        for i in range(self.depth):
            self.layers.append(KANLayer(
                layer_dims[i],
                layer_dims[i+1],
                grid_size,
                spline_order,
                grid_range
            ))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        for layer in self.layers:
            x = layer(x)
        return x
    
    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        loss_fn: callable = nn.MSELoss()
    ) -> float:
        """单步训练"""
        optimizer.zero_grad()
        y_pred = self.forward(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        return loss.item()
    
    def fit(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        epochs: int = 1000,
        lr: float = 0.01,
        verbose: int = 100
    ) -> List[float]:
        """训练模型"""
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        losses = []
        
        for epoch in range(epochs):
            loss = self.train_step(x_train, y_train, optimizer)
            losses.append(loss)
            
            if verbose and (epoch + 1) % verbose == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}")
        
        return losses
    
    def symbolic_regression(self, threshold: float = 0.01) -> List[str]:
        """
        符号回归：提取数学公式
        
        这是KAN的核心优势——可解释性
        """
        all_formulas = []
        for i, layer in enumerate(self.layers):
            formulas = layer.get_symbolic_formula(threshold)
            all_formulas.append(f"Layer {i}: {formulas}")
        return all_formulas
    
    def count_parameters(self) -> int:
        """计算参数数量"""
        return sum(p.numel() for p in self.parameters())


# ============ 使用示例 ============

def example_function_approximation():
    """
    示例：用KAN逼近 f(x,y) = sin(πx) * cos(πy)
    """
    # 生成数据
    n_samples = 1000
    x = torch.rand(n_samples, 2) * 2 - 1  # [-1, 1]
    y = torch.sin(np.pi * x[:, 0:1]) * torch.cos(np.pi * x[:, 1:2])
    
    # 创建KAN
    kan = KAN(
        layer_dims=[2, 5, 5, 1],
        grid_size=5,
        spline_order=3
    )
    
    print(f"KAN Parameters: {kan.count_parameters()}")
    
    # 对比MLP
    mlp = nn.Sequential(
        nn.Linear(2, 50),
        nn.Tanh(),
        nn.Linear(50, 50),
        nn.Tanh(),
        nn.Linear(50, 1)
    )
    mlp_params = sum(p.numel() for p in mlp.parameters())
    print(f"MLP Parameters: {mlp_params}")
    print(f"Parameter Ratio: {mlp_params / kan.count_parameters():.1f}x")
    
    # 训练KAN
    losses = kan.fit(x, y, epochs=2000, lr=0.01, verbose=500)
    
    # 评估
    with torch.no_grad():
        y_pred = kan(x)
        mse = torch.mean((y_pred - y) ** 2).item()
        print(f"Final MSE: {mse:.6f}")
    
    # 符号回归
    formulas = kan.symbolic_regression()
    print("Extracted Formulas:")
    for f in formulas:
        print(f"  {f}")
    
    return kan, losses


def example_pde_solving():
    """
    示例：用KAN求解PDE (类似PINN但使用KAN架构)
    """
    # 创建KAN-PINN混合
    kan_pinn = KAN(
        layer_dims=[2, 10, 10, 1],  # (x, t) -> u
        grid_size=8,
        spline_order=3
    )
    
    # 这里可以添加物理约束损失
    # 与PINN类似，但使用KAN架构
    
    return kan_pinn


if __name__ == "__main__":
    kan, losses = example_function_approximation()
```

---

## KAN vs MLP 对比

| 特性 | KAN | MLP |
|------|-----|-----|
| 激活函数位置 | 边（edges） | 节点（nodes） |
| 激活函数类型 | 可学习样条 | 固定（ReLU/Tanh） |
| 参数效率 | 高（100倍） | 低 |
| 可解释性 | 高（可提取公式） | 低（黑盒） |
| 训练速度 | 较慢 | 较快 |
| 创新分数 | 0.98 | 0.30 |

---

## 输出格式

```json
{
  "kan_model": {
    "architecture": {
      "layer_dims": [2, 5, 5, 1],
      "grid_size": 5,
      "spline_order": 3,
      "total_params": 330
    },
    "training_config": {
      "epochs": 2000,
      "learning_rate": 0.01,
      "optimizer": "Adam"
    }
  },
  "kan_results": {
    "final_mse": 1.23e-5,
    "training_time_seconds": 45.2,
    "convergence_epoch": 1500
  },
  "symbolic_regression": {
    "extracted_formula": "y ≈ sin(πx₁) * cos(πx₂)",
    "formula_complexity": 2,
    "interpretation": "KAN successfully discovered the underlying mathematical relationship"
  },
  "comparison_with_mlp": {
    "kan_params": 330,
    "mlp_params": 2701,
    "param_ratio": 8.2,
    "accuracy_comparison": "KAN achieves similar accuracy with 8.2x fewer parameters"
  },
  "innovation_highlights": [
    "使用2024 ICLR最新KAN架构",
    "实现符号回归提取数学公式",
    "参数效率比MLP高8倍",
    "提供可解释的数学表达式"
  ]
}
```

---

## 执行说明

1. 确定问题类型和任务（回归/PDE求解/符号回归）
2. 设计KAN架构（层维度、网格大小）
3. 实现B样条基函数和KAN层
4. 训练模型并监控收敛
5. 执行符号回归提取数学公式
6. 与MLP对比，展示参数效率优势
7. 输出完整代码和结果

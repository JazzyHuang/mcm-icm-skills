"""
Kolmogorov-Arnold Networks (KAN) Implementation
KAN网络实现

2025年ICLR重大创新，参数效率比MLP高100倍，可解释性强。
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import json


class BSplineBasis(nn.Module):
    """B-Spline基函数"""
    
    def __init__(self, grid_size: int = 5, spline_order: int = 3, 
                 grid_range: Tuple[float, float] = (-1, 1)):
        super().__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_range = grid_range
        
        # 创建网格点
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.linspace(
            grid_range[0] - spline_order * h,
            grid_range[1] + spline_order * h,
            grid_size + 2 * spline_order + 1
        )
        self.register_buffer('grid', grid)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算B-spline基函数值
        
        Args:
            x: 输入张量 [..., 1]
            
        Returns:
            B-spline基函数值 [..., num_bases]
        """
        x = x.unsqueeze(-1)  # [..., 1, 1]
        grid = self.grid  # [num_knots]
        
        # 递归计算B-spline
        bases = self._compute_bspline(x, self.spline_order)
        return bases
        
    def _compute_bspline(self, x, k):
        """递归计算k阶B-spline"""
        grid = self.grid
        
        if k == 0:
            # 0阶: 指示函数
            bases = ((x >= grid[:-1]) & (x < grid[1:])).float()
        else:
            # 递归
            bases_prev = self._compute_bspline(x, k - 1)
            
            # 左侧系数
            left_num = x - grid[:-(k+1)]
            left_den = grid[k:-1] - grid[:-(k+1)]
            left = left_num / (left_den + 1e-8) * bases_prev[..., :-1]
            
            # 右侧系数
            right_num = grid[k+1:] - x
            right_den = grid[k+1:] - grid[1:-k]
            right = right_num / (right_den + 1e-8) * bases_prev[..., 1:]
            
            bases = left + right
            
        return bases


class KANLayer(nn.Module):
    """KAN单层：边上的可学习激活函数"""
    
    def __init__(self, in_features: int, out_features: int,
                 grid_size: int = 5, spline_order: int = 3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 每条边一个B-spline
        self.num_edges = in_features * out_features
        self.spline_basis = BSplineBasis(grid_size, spline_order)
        
        # 可学习的spline系数
        num_bases = grid_size + spline_order
        self.coef = nn.Parameter(torch.randn(out_features, in_features, num_bases) * 0.1)
        
        # 可选的残差连接（SiLU基础函数）
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, in_features]
            
        Returns:
            [batch, out_features]
        """
        batch_size = x.shape[0]
        
        # 计算B-spline基函数值
        # x: [batch, in_features] -> [batch, in_features, num_bases]
        bases = self.spline_basis(x)
        
        # Spline输出: sum over bases
        # [batch, in_features, num_bases] @ [out, in, num_bases]^T
        spline_out = torch.einsum('bin,oin->bo', bases, self.coef)
        
        # 基础函数（残差）
        base_out = torch.nn.functional.silu(x) @ self.base_weight.T
        
        return spline_out + base_out


class KAN(nn.Module):
    """Kolmogorov-Arnold Network"""
    
    def __init__(self, layers: List[int], grid_size: int = 5,
                 spline_order: int = 3, device: str = 'auto'):
        """
        Args:
            layers: 层尺寸列表，如[2, 5, 5, 1]
            grid_size: B-spline网格点数
            spline_order: 样条阶数
        """
        super().__init__()
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.layers_config = layers
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # 构建KAN层
        self.kan_layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.kan_layers.append(
                KANLayer(layers[i], layers[i+1], grid_size, spline_order)
            )
            
        self.to(self.device)
        self.training_history = {'loss': []}
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.kan_layers:
            x = layer(x)
        return x
        
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000,
            learning_rate: float = 1e-3, batch_size: int = 64,
            verbose: int = 100) -> Dict:
        """训练KAN"""
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        y = torch.tensor(y, dtype=torch.float32, device=self.device)
        
        if len(y.shape) == 1:
            y = y.unsqueeze(-1)
            
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            # Mini-batch训练
            perm = torch.randperm(n_samples)
            total_loss = 0
            n_batches = 0
            
            for i in range(0, n_samples, batch_size):
                idx = perm[i:i+batch_size]
                X_batch, y_batch = X[idx], y[idx]
                
                optimizer.zero_grad()
                y_pred = self.forward(X_batch)
                loss = torch.mean((y_pred - y_batch) ** 2)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
                
            scheduler.step()
            avg_loss = total_loss / n_batches
            self.training_history['loss'].append(avg_loss)
            
            if verbose and (epoch + 1) % verbose == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6e}")
                
        return self.training_history
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        self.eval()
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
            y_pred = self.forward(X)
            return y_pred.cpu().numpy()
            
    def count_parameters(self) -> int:
        """计算参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    def plot_activations(self, layer_idx: int = 0, save_path: str = None):
        """可视化学到的激活函数"""
        import matplotlib.pyplot as plt
        
        layer = self.kan_layers[layer_idx]
        in_features = layer.in_features
        out_features = layer.out_features
        
        # 创建输入范围
        x_range = torch.linspace(-2, 2, 100, device=self.device)
        
        fig, axes = plt.subplots(out_features, in_features, 
                                 figsize=(3*in_features, 2.5*out_features))
        if out_features == 1:
            axes = [axes]
        if in_features == 1:
            axes = [[ax] for ax in axes]
            
        for i in range(out_features):
            for j in range(in_features):
                ax = axes[i][j] if out_features > 1 else axes[j]
                
                # 计算该边的激活函数
                x_input = torch.zeros(100, in_features, device=self.device)
                x_input[:, j] = x_range
                
                bases = layer.spline_basis(x_input[:, j:j+1])
                y_spline = torch.einsum('bn,n->b', bases, layer.coef[i, j]).detach().cpu()
                
                ax.plot(x_range.cpu().numpy(), y_spline.numpy(), 'b-', linewidth=2)
                ax.set_xlabel(f'x{j}')
                ax.set_ylabel(f'φ_{i},{j}')
                ax.grid(True, alpha=0.3)
                ax.axhline(0, color='black', linewidth=0.5)
                ax.axvline(0, color='black', linewidth=0.5)
                
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
        
    def save_results(self, output_path: str):
        """保存结果"""
        results = {
            'model': {
                'type': 'KAN',
                'architecture': self.layers_config,
                'grid_size': self.grid_size,
                'spline_order': self.spline_order,
                'total_params': self.count_parameters()
            },
            'training': {
                'epochs': len(self.training_history['loss']),
                'final_loss': self.training_history['loss'][-1] if self.training_history['loss'] else None
            }
        }
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        return results


class KANSymbolicRegression:
    """KAN符号回归 - 从数据中发现数学公式"""
    
    def __init__(self, candidate_functions: List[str] = None):
        """
        Args:
            candidate_functions: 候选函数列表
        """
        self.candidate_functions = candidate_functions or [
            'sin', 'cos', 'exp', 'log', 'sqrt', 'x^2', 'x^3', 'tanh'
        ]
        self.model = None
        self.extracted_formula = None
        
    def fit_and_extract(self, X: np.ndarray, y: np.ndarray,
                        epochs: int = 2000, threshold: float = 0.1) -> str:
        """
        训练KAN并提取符号公式
        
        Args:
            X: 输入数据
            y: 目标值
            epochs: 训练轮数
            threshold: 函数匹配阈值
            
        Returns:
            提取的数学公式字符串
        """
        in_dim = X.shape[1] if len(X.shape) > 1 else 1
        
        # 训练KAN
        self.model = KAN(layers=[in_dim, 5, 1], grid_size=10)
        self.model.fit(X, y, epochs=epochs, verbose=epochs//5)
        
        # 尝试匹配候选函数（简化实现）
        formula_parts = []
        
        for i in range(in_dim):
            # 生成测试输入
            x_test = np.zeros((100, in_dim))
            x_test[:, i] = np.linspace(-2, 2, 100)
            
            y_pred = self.model.predict(x_test).flatten()
            
            # 匹配候选函数
            best_match = self._match_function(x_test[:, i], y_pred)
            if best_match:
                formula_parts.append(f"{best_match}(x{i})")
                
        self.extracted_formula = " + ".join(formula_parts) if formula_parts else "complex_function"
        return self.extracted_formula
        
    def _match_function(self, x: np.ndarray, y: np.ndarray) -> Optional[str]:
        """匹配候选函数"""
        best_match = None
        best_score = float('inf')
        
        candidates = {
            'sin': np.sin,
            'cos': np.cos,
            'x^2': lambda x: x**2,
            'x^3': lambda x: x**3,
            'exp': lambda x: np.exp(np.clip(x, -5, 5)),
            'tanh': np.tanh,
            'linear': lambda x: x,
        }
        
        for name, func in candidates.items():
            try:
                y_candidate = func(x)
                # 尝试线性拟合
                coef = np.polyfit(y_candidate, y, 1)
                y_fit = coef[0] * y_candidate + coef[1]
                mse = np.mean((y - y_fit) ** 2)
                
                if mse < best_score:
                    best_score = mse
                    if abs(coef[0]) > 0.1:  # 系数足够大
                        best_match = f"{coef[0]:.2f}*{name}"
            except:
                continue
                
        return best_match if best_score < 0.5 else None


class PhysicsInformedKAN(KAN):
    """物理信息KAN - 结合物理约束"""
    
    def __init__(self, layers: List[int], physics_loss_fn: Callable,
                 grid_size: int = 5, spline_order: int = 3):
        """
        Args:
            layers: 网络结构
            physics_loss_fn: 物理损失函数 f(model, x) -> residual
        """
        super().__init__(layers, grid_size, spline_order)
        self.physics_loss_fn = physics_loss_fn
        
    def train(self, collocation_points: np.ndarray,
              boundary_conditions: Tuple[np.ndarray, np.ndarray],
              epochs: int = 10000, lambda_physics: float = 1.0,
              lambda_bc: float = 10.0, verbose: int = 1000):
        """训练物理信息KAN"""
        
        # 转换数据
        x_col = torch.tensor(collocation_points, dtype=torch.float32, 
                            device=self.device, requires_grad=True)
        x_bc, y_bc = boundary_conditions
        x_bc = torch.tensor(x_bc, dtype=torch.float32, device=self.device)
        y_bc = torch.tensor(y_bc, dtype=torch.float32, device=self.device)
        
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # 物理损失
            loss_physics = self.physics_loss_fn(self, x_col)
            
            # 边界条件损失
            u_bc_pred = self.forward(x_bc)
            loss_bc = torch.mean((u_bc_pred - y_bc) ** 2)
            
            # 总损失
            total_loss = lambda_physics * loss_physics + lambda_bc * loss_bc
            total_loss.backward()
            optimizer.step()
            
            self.training_history['loss'].append(total_loss.item())
            
            if verbose and (epoch + 1) % verbose == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss.item():.6e}")


# 示例物理损失函数
def heat_equation_residual(model, x):
    """热传导方程残差: du/dt = alpha * d²u/dx²"""
    x.requires_grad_(True)
    u = model(x)
    
    # 计算导数
    grad_outputs = torch.ones_like(u)
    grads = torch.autograd.grad(u, x, grad_outputs=grad_outputs, create_graph=True)[0]
    
    u_x = grads[:, 0:1]
    u_t = grads[:, 1:2] if x.shape[1] > 1 else torch.zeros_like(u_x)
    
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), 
                               create_graph=True)[0][:, 0:1]
    
    alpha = 0.01
    residual = u_t - alpha * u_xx
    return torch.mean(residual ** 2)


if __name__ == '__main__':
    print("Testing KAN Networks...")
    
    # 测试基础KAN
    np.random.seed(42)
    X = np.random.randn(500, 2)
    y = np.sin(X[:, 0]) + X[:, 1] ** 2  # 真实函数
    
    model = KAN(layers=[2, 5, 1], grid_size=5)
    model.fit(X, y, epochs=500, verbose=100)
    
    y_pred = model.predict(X)
    mse = np.mean((y_pred.flatten() - y) ** 2)
    print(f"KAN MSE: {mse:.6f}")
    print(f"KAN Parameters: {model.count_parameters()}")
    
    # 对比MLP参数量（大约需要100倍参数达到同样精度）
    print(f"Estimated MLP params for same accuracy: ~{model.count_parameters() * 100}")
    
    # 测试符号回归
    sr = KANSymbolicRegression()
    formula = sr.fit_and_extract(X, y, epochs=500)
    print(f"Extracted formula: {formula}")
    
    print("\nKAN Networks test completed!")

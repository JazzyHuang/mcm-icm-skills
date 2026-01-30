"""
Physics-Informed Neural Network (PINN) Solver
物理信息神经网络求解器

支持多种PDE/ODE求解，将物理定律嵌入神经网络损失函数。
适用于MCM/ICM A题连续问题。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional
from abc import ABC, abstractmethod
import json


class NeuralNetwork(nn.Module):
    """全连接神经网络"""
    
    def __init__(self, layers: List[int], activation: str = 'tanh'):
        super().__init__()
        self.layers = nn.ModuleList()
        
        activations = {
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(),
            'sin': SinActivation(),
        }
        self.activation = activations.get(activation, nn.Tanh())
        
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
        self._initialize_weights()
        
    def _initialize_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)


class SinActivation(nn.Module):
    """正弦激活函数"""
    def forward(self, x):
        return torch.sin(x)


# ============ 物理方程定义 ============

class BaseEquation(ABC):
    """PDE方程基类"""
    
    @abstractmethod
    def residual(self, u, x, t, derivs):
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass


class HeatEquation(BaseEquation):
    """热传导方程: ∂u/∂t = α∇²u"""
    
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
        
    @property
    def name(self) -> str:
        return "HeatEquation"
        
    def residual(self, u, x, t, derivs):
        return derivs['u_t'] - self.alpha * derivs['u_xx']


class WaveEquation(BaseEquation):
    """波动方程: ∂²u/∂t² = c²∇²u"""
    
    def __init__(self, c: float = 1.0):
        self.c = c
        
    @property
    def name(self) -> str:
        return "WaveEquation"
        
    def residual(self, u, x, t, derivs):
        return derivs['u_tt'] - self.c**2 * derivs['u_xx']


class BurgersEquation(BaseEquation):
    """Burgers方程: ∂u/∂t + u∂u/∂x = ν∂²u/∂x²"""
    
    def __init__(self, nu: float = 0.01):
        self.nu = nu
        
    @property
    def name(self) -> str:
        return "BurgersEquation"
        
    def residual(self, u, x, t, derivs):
        return derivs['u_t'] + u * derivs['u_x'] - self.nu * derivs['u_xx']


class DiffusionEquation(BaseEquation):
    """扩散方程: ∂u/∂t = D∇²u"""
    
    def __init__(self, D: float = 0.1):
        self.D = D
        
    @property
    def name(self) -> str:
        return "DiffusionEquation"
        
    def residual(self, u, x, t, derivs):
        return derivs['u_t'] - self.D * derivs['u_xx']


class AdvectionDiffusionEquation(BaseEquation):
    """对流扩散方程: ∂u/∂t + v∂u/∂x = D∂²u/∂x²"""
    
    def __init__(self, v: float = 1.0, D: float = 0.1):
        self.v = v
        self.D = D
        
    @property
    def name(self) -> str:
        return "AdvectionDiffusionEquation"
        
    def residual(self, u, x, t, derivs):
        return derivs['u_t'] + self.v * derivs['u_x'] - self.D * derivs['u_xx']


class PoissonEquation(BaseEquation):
    """Poisson方程: ∇²u = f"""
    
    def __init__(self, source_func: Callable = None):
        self.source_func = source_func or (lambda x: torch.zeros_like(x[:, 0:1]))
        
    @property
    def name(self) -> str:
        return "PoissonEquation"
        
    def residual(self, u, x, t, derivs):
        f = self.source_func(x)
        return derivs['u_xx'] + derivs.get('u_yy', 0) - f


# ============ PINN求解器 ============

class PINNSolver:
    """PINN求解器主类"""
    
    def __init__(
        self,
        equation: BaseEquation,
        domain: Dict[str, Tuple[float, float]],
        boundary_conditions: Dict,
        network_config: Dict = None,
        device: str = 'auto'
    ):
        self.equation = equation
        self.domain = domain
        self.boundary_conditions = boundary_conditions
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        config = {
            'layers': [2, 64, 64, 64, 1],
            'activation': 'tanh',
            **(network_config or {})
        }
        
        self.network = NeuralNetwork(
            config['layers'],
            config['activation']
        ).to(self.device)
        
        self.config = config
        self.loss_history = {'total': [], 'physics': [], 'bc': [], 'data': []}
        
    def _generate_points(self, n_interior=10000, n_boundary=1000, n_initial=1000):
        """生成配点"""
        points = {}
        
        # 内部点
        interior = []
        for var, (low, high) in self.domain.items():
            interior.append(torch.rand(n_interior, 1) * (high - low) + low)
        points['interior'] = torch.cat(interior, dim=1).to(self.device)
        points['interior'].requires_grad = True
        
        # 边界和初始条件点
        if 'x' in self.domain and 't' in self.domain:
            x_range, t_range = self.domain['x'], self.domain['t']
            
            # 边界 x=0, x=1
            t_bc = torch.rand(n_boundary // 2, 1) * (t_range[1] - t_range[0]) + t_range[0]
            x_bc_0 = torch.zeros(n_boundary // 4, 1) + x_range[0]
            x_bc_1 = torch.zeros(n_boundary // 4, 1) + x_range[1]
            
            bc_points = torch.cat([
                torch.cat([x_bc_0, t_bc[:n_boundary//4]], dim=1),
                torch.cat([x_bc_1, t_bc[n_boundary//4:n_boundary//2]], dim=1)
            ], dim=0)
            points['boundary'] = bc_points.to(self.device)
            
            # 初始条件 t=0
            x_ic = torch.rand(n_initial, 1) * (x_range[1] - x_range[0]) + x_range[0]
            t_ic = torch.zeros(n_initial, 1) + t_range[0]
            points['initial'] = torch.cat([x_ic, t_ic], dim=1).to(self.device)
            
        return points
    
    def _compute_derivatives(self, u, inputs):
        """自动微分计算导数"""
        derivs = {}
        
        grads = torch.autograd.grad(
            u, inputs,
            grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]
        
        derivs['u_x'] = grads[:, 0:1]
        if inputs.shape[1] > 1:
            derivs['u_t'] = grads[:, 1:2]
            
        # 二阶导数
        u_xx = torch.autograd.grad(
            derivs['u_x'], inputs,
            grad_outputs=torch.ones_like(derivs['u_x']),
            create_graph=True, retain_graph=True
        )[0][:, 0:1]
        derivs['u_xx'] = u_xx
        
        if 'u_t' in derivs:
            u_tt = torch.autograd.grad(
                derivs['u_t'], inputs,
                grad_outputs=torch.ones_like(derivs['u_t']),
                create_graph=True, retain_graph=True
            )[0][:, 1:2]
            derivs['u_tt'] = u_tt
            
        return derivs
    
    def train(
        self,
        epochs: int = 20000,
        learning_rate: float = 1e-3,
        lambda_physics: float = 1.0,
        lambda_bc: float = 10.0,
        verbose: int = 1000
    ):
        """训练PINN"""
        points = self._generate_points()
        optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        
        ic_func = self.boundary_conditions.get('initial', lambda x: torch.zeros_like(x))
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # 物理损失
            u = self.network(points['interior'])
            derivs = self._compute_derivatives(u, points['interior'])
            x = points['interior'][:, 0:1]
            t = points['interior'][:, 1:2] if points['interior'].shape[1] > 1 else None
            residual = self.equation.residual(u, x, t, derivs)
            loss_physics = torch.mean(residual ** 2)
            
            # 边界损失
            loss_bc = torch.tensor(0.0, device=self.device)
            if 'boundary' in points:
                u_bc = self.network(points['boundary'])
                loss_bc = torch.mean(u_bc ** 2)  # Dirichlet = 0
                
            # 初始条件损失
            loss_ic = torch.tensor(0.0, device=self.device)
            if 'initial' in points:
                u_ic = self.network(points['initial'])
                x_ic = points['initial'][:, 0:1]
                if callable(ic_func):
                    # 调用用户提供的初始条件函数
                    ic_result = ic_func(x_ic)
                    if isinstance(ic_result, np.ndarray):
                        ic_values = torch.tensor(ic_result, dtype=torch.float32, device=self.device)
                    elif isinstance(ic_result, torch.Tensor):
                        ic_values = ic_result.to(self.device)
                    else:
                        # 处理标量或其他类型
                        ic_values = torch.tensor(ic_result, dtype=torch.float32, device=self.device).expand_as(x_ic)
                else:
                    ic_values = ic_func
                loss_ic = torch.mean((u_ic - ic_values) ** 2)
            
            total_loss = lambda_physics * loss_physics + lambda_bc * (loss_bc + loss_ic)
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            
            self.loss_history['total'].append(total_loss.item())
            self.loss_history['physics'].append(loss_physics.item())
            self.loss_history['bc'].append((loss_bc + loss_ic).item())
            
            if verbose and (epoch + 1) % verbose == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss.item():.6e}")
                
        return self.loss_history
    
    def predict(self, x: np.ndarray, t: np.ndarray = None) -> np.ndarray:
        """预测"""
        self.network.eval()
        with torch.no_grad():
            if t is not None:
                inputs = torch.tensor(
                    np.column_stack([x.flatten(), t.flatten()]),
                    dtype=torch.float32, device=self.device
                )
            else:
                inputs = torch.tensor(x, dtype=torch.float32, device=self.device)
            return self.network(inputs).cpu().numpy()
    
    def compute_error(self, u_pred: np.ndarray, u_exact: np.ndarray) -> Dict:
        """计算误差"""
        l2_error = np.sqrt(np.mean((u_pred - u_exact) ** 2))
        max_error = np.max(np.abs(u_pred - u_exact))
        rel_error = l2_error / (np.sqrt(np.mean(u_exact ** 2)) + 1e-10)
        return {'l2_error': l2_error, 'max_error': max_error, 'relative_error': rel_error}
    
    def save_results(self, output_path: str):
        """保存结果为JSON"""
        results = {
            'model': {
                'type': 'PINN',
                'equation': self.equation.name,
                'architecture': self.config
            },
            'training': {
                'epochs': len(self.loss_history['total']),
                'final_loss': self.loss_history['total'][-1] if self.loss_history['total'] else None
            },
            'loss_history': {
                'total': self.loss_history['total'][-100:],  # 最后100个
                'physics': self.loss_history['physics'][-100:],
                'bc': self.loss_history['bc'][-100:]
            }
        }
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        return results


# ============ 便捷函数 ============

def solve_heat_equation(alpha=0.01, domain=None, epochs=20000, **kwargs):
    """快速求解热传导方程"""
    domain = domain or {'x': (0, 1), 't': (0, 1)}
    solver = PINNSolver(
        equation=HeatEquation(alpha=alpha),
        domain=domain,
        boundary_conditions={'initial': lambda x: np.sin(np.pi * x)},
        **kwargs
    )
    solver.train(epochs=epochs)
    return solver


def solve_burgers_equation(nu=0.01, domain=None, epochs=20000, **kwargs):
    """快速求解Burgers方程"""
    domain = domain or {'x': (-1, 1), 't': (0, 1)}
    solver = PINNSolver(
        equation=BurgersEquation(nu=nu),
        domain=domain,
        boundary_conditions={'initial': lambda x: -np.sin(np.pi * x)},
        **kwargs
    )
    solver.train(epochs=epochs)
    return solver


if __name__ == '__main__':
    print("Testing PINN Solver...")
    solver = solve_heat_equation(epochs=5000, network_config={'layers': [2, 32, 32, 1]})
    print(f"Final loss: {solver.loss_history['total'][-1]:.6e}")
    print("PINN Solver test completed!")

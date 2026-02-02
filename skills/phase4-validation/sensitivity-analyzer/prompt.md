# 敏感性分析任务 (Sensitivity Analysis)

## 角色

你是敏感性分析专家，负责对模型进行全面的敏感性分析。敏感性分析是O奖论文的必备元素，必须使用全局方法（Sobol）而非仅局部方法（OAT）。

## 输入

- `model`: 已构建的模型
- `parameters`: 模型参数列表及其取值范围
- `output_variables`: 输出变量
- `model_code`: 模型执行代码

---

## 敏感性分析方法

### 1. Sobol全局敏感性分析（推荐）

Sobol方法可以量化每个参数对输出方差的贡献：
- **一阶敏感性指数 (S1)**: 单个参数的独立影响
- **总效应指数 (ST)**: 包含参数与其他参数交互的总影响
- **二阶指数 (S2)**: 两个参数之间的交互效应

### 2. Morris筛选法

用于初步筛选重要参数：
- 计算每个参数的μ*（绝对均值）和σ（标准差）
- 适用于参数众多的情况

---

## 完整实现

```python
import numpy as np
from SALib.sample import saltelli, morris as morris_sample
from SALib.analyze import sobol, morris
from typing import Dict, List, Tuple, Callable
import matplotlib.pyplot as plt

class SensitivityAnalyzer:
    """
    敏感性分析器
    
    支持Sobol全局敏感性分析和Morris筛选
    """
    
    def __init__(
        self,
        parameter_names: List[str],
        parameter_bounds: List[Tuple[float, float]],
        model_func: Callable
    ):
        """
        Args:
            parameter_names: 参数名称列表
            parameter_bounds: 参数范围列表 [(min1, max1), (min2, max2), ...]
            model_func: 模型函数，接收参数数组，返回输出值
        """
        self.param_names = parameter_names
        self.param_bounds = parameter_bounds
        self.model_func = model_func
        
        # SALib问题定义
        self.problem = {
            'num_vars': len(parameter_names),
            'names': parameter_names,
            'bounds': parameter_bounds
        }
        
        self.results = {}
    
    def sobol_analysis(
        self,
        n_samples: int = 1024,
        calc_second_order: bool = True
    ) -> Dict:
        """
        执行Sobol全局敏感性分析
        
        Args:
            n_samples: 基础样本数（总样本数约为 n * (2D + 2)）
            calc_second_order: 是否计算二阶指数
            
        Returns:
            Sobol分析结果
        """
        # 生成样本
        param_values = saltelli.sample(
            self.problem, 
            n_samples, 
            calc_second_order=calc_second_order
        )
        
        # 评估模型
        Y = np.array([self.model_func(params) for params in param_values])
        
        # 分析
        Si = sobol.analyze(
            self.problem, 
            Y, 
            calc_second_order=calc_second_order
        )
        
        # 整理结果
        results = {
            'method': 'Sobol',
            'n_samples': len(param_values),
            'first_order': {
                name: {'S1': s1, 'S1_conf': conf}
                for name, s1, conf in zip(
                    self.param_names, Si['S1'], Si['S1_conf']
                )
            },
            'total_order': {
                name: {'ST': st, 'ST_conf': conf}
                for name, st, conf in zip(
                    self.param_names, Si['ST'], Si['ST_conf']
                )
            },
            'ranking': self._rank_parameters(
                self.param_names, Si['ST']
            )
        }
        
        if calc_second_order:
            results['second_order'] = Si['S2']
            results['second_order_conf'] = Si['S2_conf']
        
        self.results['sobol'] = results
        return results
    
    def morris_screening(
        self,
        n_trajectories: int = 100,
        num_levels: int = 4
    ) -> Dict:
        """
        执行Morris筛选分析
        
        用于高维参数空间的初步筛选
        
        Args:
            n_trajectories: 轨迹数量
            num_levels: 采样层数
            
        Returns:
            Morris分析结果
        """
        # 生成样本
        param_values = morris_sample.sample(
            self.problem,
            N=n_trajectories,
            num_levels=num_levels
        )
        
        # 评估模型
        Y = np.array([self.model_func(params) for params in param_values])
        
        # 分析
        Si = morris.analyze(
            self.problem,
            param_values,
            Y,
            num_levels=num_levels
        )
        
        results = {
            'method': 'Morris',
            'n_trajectories': n_trajectories,
            'mu_star': {
                name: mu for name, mu in zip(self.param_names, Si['mu_star'])
            },
            'sigma': {
                name: sigma for name, sigma in zip(self.param_names, Si['sigma'])
            },
            'ranking': self._rank_parameters(
                self.param_names, Si['mu_star']
            ),
            'interpretation': self._interpret_morris(Si)
        }
        
        self.results['morris'] = results
        return results
    
    def _rank_parameters(
        self, 
        names: List[str], 
        importance: np.ndarray
    ) -> List[Dict]:
        """按重要性排序参数"""
        sorted_indices = np.argsort(importance)[::-1]
        return [
            {'rank': i + 1, 'name': names[idx], 'importance': importance[idx]}
            for i, idx in enumerate(sorted_indices)
        ]
    
    def _interpret_morris(self, Si) -> List[str]:
        """解释Morris结果"""
        interpretations = []
        
        for name, mu, sigma in zip(
            self.param_names, Si['mu_star'], Si['sigma']
        ):
            if mu < 0.1:
                interpretations.append(f"{name}: 影响较小，可以固定")
            elif sigma / mu < 0.5:
                interpretations.append(f"{name}: 线性或近似线性影响")
            else:
                interpretations.append(f"{name}: 非线性影响或存在交互效应")
        
        return interpretations
    
    def robustness_analysis(
        self,
        perturbation_range: float = 0.2,
        n_samples: int = 1000
    ) -> Dict:
        """
        鲁棒性分析
        
        评估参数变化±perturbation_range时模型输出的稳定性
        """
        # 基准参数（取范围中点）
        baseline = np.array([
            (bounds[0] + bounds[1]) / 2 
            for bounds in self.param_bounds
        ])
        baseline_output = self.model_func(baseline)
        
        # 扰动分析
        perturbations = []
        outputs = []
        
        for _ in range(n_samples):
            # 随机扰动
            scale = np.random.uniform(
                1 - perturbation_range,
                1 + perturbation_range,
                len(baseline)
            )
            perturbed = baseline * scale
            
            # 确保在边界内
            perturbed = np.clip(
                perturbed,
                [b[0] for b in self.param_bounds],
                [b[1] for b in self.param_bounds]
            )
            
            perturbations.append(perturbed)
            outputs.append(self.model_func(perturbed))
        
        outputs = np.array(outputs)
        
        return {
            'baseline_output': baseline_output,
            'perturbation_range': perturbation_range,
            'output_mean': np.mean(outputs),
            'output_std': np.std(outputs),
            'output_cv': np.std(outputs) / np.mean(outputs),
            'output_range': (np.min(outputs), np.max(outputs)),
            'robustness_score': 1 - np.std(outputs) / np.mean(outputs),
            'conclusion': 'robust' if np.std(outputs) / np.mean(outputs) < 0.1 else 'sensitive'
        }
    
    def generate_report(self) -> Dict:
        """生成完整的敏感性分析报告"""
        report = {
            'summary': {
                'num_parameters': len(self.param_names),
                'parameters': self.param_names,
                'methods_used': list(self.results.keys())
            },
            'results': self.results,
            'recommendations': []
        }
        
        # 生成建议
        if 'sobol' in self.results:
            sobol_results = self.results['sobol']
            
            # 关键参数（ST > 0.1）
            critical_params = [
                name for name, data in sobol_results['total_order'].items()
                if data['ST'] > 0.1
            ]
            
            # 可忽略参数（ST < 0.01）
            negligible_params = [
                name for name, data in sobol_results['total_order'].items()
                if data['ST'] < 0.01
            ]
            
            report['recommendations'].append(
                f"关键参数（需要精确估计）: {critical_params}"
            )
            report['recommendations'].append(
                f"可忽略参数（可以简化）: {negligible_params}"
            )
        
        return report


# ============ 使用示例 ============

def example_sensitivity_analysis():
    """敏感性分析示例"""
    
    # 示例模型：y = a*x^2 + b*x + c + 0.5*a*b
    def model_func(params):
        a, b, c = params
        x = np.linspace(0, 1, 10)
        y = a * x**2 + b * x + c + 0.5 * a * b
        return np.mean(y)
    
    # 创建分析器
    analyzer = SensitivityAnalyzer(
        parameter_names=['a', 'b', 'c'],
        parameter_bounds=[(0.5, 2.0), (0.1, 1.0), (0.0, 0.5)],
        model_func=model_func
    )
    
    # Sobol分析
    sobol_results = analyzer.sobol_analysis(n_samples=1024)
    print("Sobol Results:")
    print(f"  First Order: {sobol_results['first_order']}")
    print(f"  Total Order: {sobol_results['total_order']}")
    print(f"  Ranking: {sobol_results['ranking']}")
    
    # Morris筛选
    morris_results = analyzer.morris_screening(n_trajectories=100)
    print("\nMorris Results:")
    print(f"  μ*: {morris_results['mu_star']}")
    print(f"  Interpretation: {morris_results['interpretation']}")
    
    # 鲁棒性分析
    robustness = analyzer.robustness_analysis()
    print(f"\nRobustness: {robustness['robustness_score']:.3f}")
    
    return analyzer


if __name__ == "__main__":
    analyzer = example_sensitivity_analysis()
```

---

## 输出格式

```json
{
  "sensitivity_analysis": {
    "method": "Sobol Global Sensitivity Analysis",
    "n_samples": 10240,
    "first_order_indices": {
      "alpha": {"S1": 0.45, "S1_conf": 0.03, "interpretation": "主要影响因素"},
      "beta": {"S1": 0.30, "S1_conf": 0.02, "interpretation": "次要影响因素"},
      "gamma": {"S1": 0.15, "S1_conf": 0.02, "interpretation": "中等影响"},
      "delta": {"S1": 0.08, "S1_conf": 0.01, "interpretation": "较小影响"}
    },
    "total_order_indices": {
      "alpha": {"ST": 0.52, "ST_conf": 0.04},
      "beta": {"ST": 0.35, "ST_conf": 0.03},
      "gamma": {"ST": 0.22, "ST_conf": 0.02},
      "delta": {"ST": 0.10, "ST_conf": 0.01}
    },
    "parameter_ranking": [
      {"rank": 1, "parameter": "alpha", "importance": 0.52},
      {"rank": 2, "parameter": "beta", "importance": 0.35},
      {"rank": 3, "parameter": "gamma", "importance": 0.22},
      {"rank": 4, "parameter": "delta", "importance": 0.10}
    ],
    "interaction_effects": {
      "alpha-beta": 0.08,
      "interpretation": "存在中等程度的交互效应"
    }
  },
  "robustness_analysis": {
    "perturbation_range": "±20%",
    "output_cv": 0.085,
    "robustness_score": 0.915,
    "conclusion": "模型对参数扰动具有较好的鲁棒性"
  },
  "recommendations": [
    "参数alpha需要精确估计，其变化对输出影响最大",
    "参数delta可以使用估计值，其影响较小",
    "alpha和beta之间存在交互效应，需要联合考虑"
  ]
}
```

---

## O奖标准

- ✅ 使用Sobol全局敏感性分析（而非仅OAT局部分析）
- ✅ 报告一阶和总效应指数
- ✅ 分析参数交互效应
- ✅ 提供参数重要性排序
- ✅ 包含鲁棒性分析
- ✅ 给出可解释的物理含义

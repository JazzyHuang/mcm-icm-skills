"""
Uncertainty Quantification Module
不确定性量化模块

支持保形预测、MC Dropout、集成方法等。
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union, Callable
from sklearn.model_selection import train_test_split
import json


class ConformalPredictor:
    """
    保形预测 - 分布无关的不确定性量化
    
    提供有理论保证的预测区间：P(Y ∈ C(X)) ≥ 1 - α
    """
    
    def __init__(self, model, alpha: float = 0.1):
        """
        Args:
            model: 任意预测模型（需有predict方法）
            alpha: 误覆盖率（1-alpha为目标覆盖率）
        """
        self.model = model
        self.alpha = alpha
        self.quantile = None
        self.calibration_scores = None
        
    def _compute_nonconformity(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """计算非一致性分数（残差绝对值）"""
        if hasattr(self.model, 'predict'):
            y_pred = self.model.predict(X)
        else:
            y_pred = self.model(X)
            
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
            
        y_pred = np.array(y_pred).flatten()
        y = np.array(y).flatten()
        
        return np.abs(y - y_pred)
        
    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray):
        """
        校准预测器
        
        Args:
            X_cal: 校准集特征
            y_cal: 校准集标签
        """
        self.calibration_scores = self._compute_nonconformity(X_cal, y_cal)
        
        # 计算分位数（加1确保有限样本覆盖保证）
        n = len(self.calibration_scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.quantile = np.quantile(self.calibration_scores, min(q_level, 1.0))
        
        return self
        
    def predict_interval(self, X: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        预测并返回置信区间
        
        Args:
            X: 测试数据
            
        Returns:
            (点预测, (下界, 上界))
        """
        if self.quantile is None:
            raise ValueError("Must call calibrate() first")
            
        if hasattr(self.model, 'predict'):
            y_pred = self.model.predict(X)
        else:
            y_pred = self.model(X)
            
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
            
        y_pred = np.array(y_pred).flatten()
        
        lower = y_pred - self.quantile
        upper = y_pred + self.quantile
        
        return y_pred, (lower, upper)
        
    def evaluate_coverage(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """评估覆盖率"""
        y_pred, (lower, upper) = self.predict_interval(X_test)
        y_test = np.array(y_test).flatten()
        
        covered = (y_test >= lower) & (y_test <= upper)
        coverage = np.mean(covered)
        
        interval_width = upper - lower
        
        return {
            'coverage': coverage,
            'target_coverage': 1 - self.alpha,
            'mean_interval_width': np.mean(interval_width),
            'median_interval_width': np.median(interval_width),
            'quantile_threshold': self.quantile
        }
        
    def plot_prediction_intervals(self, X: np.ndarray, y_true: np.ndarray = None,
                                  feature_idx: int = 0, save_path: str = None):
        """绘制预测区间"""
        import matplotlib.pyplot as plt
        
        y_pred, (lower, upper) = self.predict_interval(X)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 排序用于绘图
        if len(X.shape) > 1:
            x_plot = X[:, feature_idx]
        else:
            x_plot = X
            
        sort_idx = np.argsort(x_plot)
        x_sorted = x_plot[sort_idx]
        
        # 置信区间
        ax.fill_between(x_sorted, lower[sort_idx], upper[sort_idx],
                       alpha=0.3, color='blue', label=f'{(1-self.alpha)*100:.0f}% CI')
        
        # 预测值
        ax.plot(x_sorted, y_pred[sort_idx], 'b-', linewidth=2, label='Prediction')
        
        # 真实值
        if y_true is not None:
            y_true = np.array(y_true).flatten()
            ax.scatter(x_sorted, y_true[sort_idx], c='red', s=20, alpha=0.5, label='True')
            
        ax.set_xlabel(f'Feature {feature_idx}')
        ax.set_ylabel('Value')
        ax.set_title('Conformal Prediction Intervals')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)  # 防止内存泄漏
        return fig


class MCDropout:
    """
    MC Dropout - 贝叶斯近似不确定性
    
    通过在推理时保持Dropout来估计不确定性
    """
    
    def __init__(self, model: nn.Module, n_samples: int = 100,
                 dropout_rate: float = 0.1):
        """
        Args:
            model: PyTorch模型
            n_samples: MC采样次数
            dropout_rate: Dropout比率
        """
        self.model = model
        self.n_samples = n_samples
        self.dropout_rate = dropout_rate
        
        # 确保模型有Dropout层
        self._ensure_dropout()
        
    def _ensure_dropout(self):
        """确保模型有Dropout层"""
        has_dropout = False
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                has_dropout = True
                break
                
        if not has_dropout:
            print("Warning: Model has no Dropout layers. MC Dropout may not work properly.")
            
    def _enable_dropout(self):
        """启用Dropout（即使在eval模式）"""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
                
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        带不确定性的预测
        
        Args:
            X: 输入数据
            
        Returns:
            (均值, 标准差)
        """
        self.model.eval()
        self._enable_dropout()
        
        X_tensor = torch.tensor(X, dtype=torch.float32)
        if next(self.model.parameters()).is_cuda:
            X_tensor = X_tensor.cuda()
            
        predictions = []
        
        with torch.no_grad():
            for _ in range(self.n_samples):
                pred = self.model(X_tensor)
                predictions.append(pred.cpu().numpy())
                
        predictions = np.array(predictions)
        mean = np.mean(predictions, axis=0)
        std = np.std(predictions, axis=0)
        
        return mean.flatten(), std.flatten()
        
    def confidence_interval(self, X: np.ndarray, confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """计算置信区间"""
        mean, std = self.predict_with_uncertainty(X)
        
        # 正态近似
        from scipy import stats
        z = stats.norm.ppf((1 + confidence) / 2)
        
        lower = mean - z * std
        upper = mean + z * std
        
        return lower, upper


class EnsembleUncertainty:
    """
    集成方法不确定性
    
    通过多个模型的预测差异估计不确定性
    """
    
    def __init__(self, base_model_class, n_models: int = 10, **model_kwargs):
        """
        Args:
            base_model_class: 基模型类
            n_models: 模型数量
            model_kwargs: 传递给基模型的参数
        """
        self.base_model_class = base_model_class
        self.n_models = n_models
        self.model_kwargs = model_kwargs
        self.models = []
        
    def fit(self, X: np.ndarray, y: np.ndarray, bootstrap: bool = True):
        """
        训练集成模型
        
        Args:
            X: 训练数据
            y: 标签
            bootstrap: 是否使用Bootstrap采样
        """
        n_samples = len(X)
        
        for i in range(self.n_models):
            model = self.base_model_class(**self.model_kwargs)
            
            if bootstrap:
                # Bootstrap采样
                idx = np.random.choice(n_samples, n_samples, replace=True)
                X_boot, y_boot = X[idx], y[idx]
            else:
                X_boot, y_boot = X, y
                
            model.fit(X_boot, y_boot)
            self.models.append(model)
            
        return self
        
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """预测并返回不确定性"""
        predictions = []
        
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
            
        predictions = np.array(predictions)
        mean = np.mean(predictions, axis=0)
        std = np.std(predictions, axis=0)
        
        return mean, std


class QuantileRegressor:
    """
    分位数回归 - 直接预测分位数
    """
    
    def __init__(self, quantiles: List[float] = None):
        """
        Args:
            quantiles: 要预测的分位数列表，如[0.05, 0.5, 0.95]
        """
        self.quantiles = quantiles or [0.05, 0.5, 0.95]
        self.models = {}
        
    def fit(self, X: np.ndarray, y: np.ndarray, model_class=None, **kwargs):
        """训练分位数回归模型"""
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            
            for q in self.quantiles:
                model = GradientBoostingRegressor(
                    loss='quantile',
                    alpha=q,
                    **kwargs
                )
                model.fit(X, y)
                self.models[q] = model
        except ImportError:
            print("sklearn required for QuantileRegressor")
            
        return self
        
    def predict(self, X: np.ndarray) -> Dict[float, np.ndarray]:
        """预测各分位数"""
        results = {}
        for q, model in self.models.items():
            results[q] = model.predict(X)
        return results
        
    def predict_interval(self, X: np.ndarray, 
                         lower_q: float = 0.05, 
                         upper_q: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """预测区间"""
        preds = self.predict(X)
        
        lower = preds.get(lower_q, preds[min(preds.keys())])
        upper = preds.get(upper_q, preds[max(preds.keys())])
        median = preds.get(0.5, (lower + upper) / 2)
        
        return median, lower, upper


def compare_uncertainty_methods(X_train, y_train, X_test, y_test, model) -> Dict:
    """比较不同不确定性量化方法"""
    results = {}
    
    # 划分校准集
    X_tr, X_cal, y_tr, y_cal = train_test_split(X_train, y_train, test_size=0.2)
    
    # 保形预测
    cp = ConformalPredictor(model, alpha=0.1)
    cp.calibrate(X_cal, y_cal)
    results['conformal'] = cp.evaluate_coverage(X_test, y_test)
    
    return results


if __name__ == '__main__':
    print("Testing Uncertainty Quantification...")
    
    # 创建测试数据
    np.random.seed(42)
    X = np.random.randn(500, 1)
    y = np.sin(X.flatten()) + np.random.randn(500) * 0.2
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_cal, y_train, y_cal = train_test_split(X_train, y_train, test_size=0.25)
    
    # 简单模型
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    
    # 测试保形预测
    print("\n1. Conformal Prediction:")
    cp = ConformalPredictor(model, alpha=0.1)
    cp.calibrate(X_cal, y_cal)
    
    y_pred, (lower, upper) = cp.predict_interval(X_test)
    eval_results = cp.evaluate_coverage(X_test, y_test)
    
    print(f"Target coverage: {eval_results['target_coverage']:.2%}")
    print(f"Actual coverage: {eval_results['coverage']:.2%}")
    print(f"Mean interval width: {eval_results['mean_interval_width']:.4f}")
    
    # 测试集成不确定性
    print("\n2. Ensemble Uncertainty:")
    ensemble = EnsembleUncertainty(RandomForestRegressor, n_models=5, n_estimators=50)
    ensemble.fit(X_train, y_train)
    mean, std = ensemble.predict_with_uncertainty(X_test)
    print(f"Mean std: {np.mean(std):.4f}")
    
    print("\nUncertainty Quantification test completed!")

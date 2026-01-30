"""
Causal Inference Estimators
因果推断估计器

支持DML、IV、DiD等方法，适用于MCM/ICM E/F题政策分析。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
import json

try:
    from econml.dml import DML, LinearDML
    from econml.grf import CausalForest
    ECONML_AVAILABLE = True
except ImportError:
    ECONML_AVAILABLE = False


class DoubleMachineLearning:
    """Double Machine Learning估计器"""
    
    def __init__(
        self,
        Y: np.ndarray,
        T: np.ndarray,
        X: np.ndarray,
        model_y: str = 'rf',
        model_t: str = 'rf',
        n_folds: int = 5
    ):
        """
        Args:
            Y: 结果变量 (n,)
            T: 处理变量 (n,)
            X: 混杂变量 (n, d)
            model_y: 结果模型类型
            model_t: 倾向得分模型类型
            n_folds: 交叉拟合折数
        """
        self.Y = np.array(Y)
        self.T = np.array(T)
        self.X = np.array(X)
        self.n_folds = n_folds
        
        self.model_y = self._get_model(model_y)
        self.model_t = self._get_model(model_t)
        
        self.ate = None
        self.std_error = None
        self.residuals_y = None
        self.residuals_t = None
        
    def _get_model(self, model_type: str):
        """获取模型"""
        models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'gb': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'lr': LinearRegression(),
            'lightgbm': GradientBoostingRegressor(n_estimators=100)  # 替代
        }
        return models.get(model_type, RandomForestRegressor())
        
    def estimate_ate(self) -> Dict:
        """估计平均处理效应"""
        n = len(self.Y)
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        residuals_y = np.zeros(n)
        residuals_t = np.zeros(n)
        
        # 交叉拟合
        for train_idx, test_idx in kf.split(self.X):
            # 拟合结果模型 E[Y|X]
            self.model_y.fit(self.X[train_idx], self.Y[train_idx])
            y_pred = self.model_y.predict(self.X[test_idx])
            residuals_y[test_idx] = self.Y[test_idx] - y_pred
            
            # 拟合处理模型 E[T|X]
            self.model_t.fit(self.X[train_idx], self.T[train_idx])
            t_pred = self.model_t.predict(self.X[test_idx])
            residuals_t[test_idx] = self.T[test_idx] - t_pred
            
        self.residuals_y = residuals_y
        self.residuals_t = residuals_t
        
        # 第二阶段回归
        # ATE = Cov(residuals_y, residuals_t) / Var(residuals_t)
        ate = np.sum(residuals_y * residuals_t) / np.sum(residuals_t ** 2)
        
        # 标准误
        psi = residuals_y - ate * residuals_t
        var_ate = np.mean(psi ** 2 * residuals_t ** 2) / (np.mean(residuals_t ** 2) ** 2)
        std_error = np.sqrt(var_ate / n)
        
        self.ate = ate
        self.std_error = std_error
        
        return {
            'effect': ate,
            'std_error': std_error,
            'ci_lower': ate - 1.96 * std_error,
            'ci_upper': ate + 1.96 * std_error,
            'p_value': 2 * (1 - self._norm_cdf(abs(ate / std_error)))
        }
        
    def _norm_cdf(self, x):
        """标准正态CDF"""
        return 0.5 * (1 + np.tanh(x * 0.7978845608))  # 近似
        
    def confidence_interval(self, alpha: float = 0.05) -> Tuple[float, float]:
        """计算置信区间"""
        if self.ate is None:
            self.estimate_ate()
        z = 1.96 if alpha == 0.05 else 2.576 if alpha == 0.01 else 1.645
        return (self.ate - z * self.std_error, self.ate + z * self.std_error)
        
    def plot_effect(self, save_path: str = None):
        """绘制处理效应图"""
        import matplotlib.pyplot as plt
        
        if self.ate is None:
            self.estimate_ate()
            
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # 效应点估计和置信区间
        ax.errorbar([0], [self.ate], yerr=[[self.ate - self.confidence_interval()[0]], 
                    [self.confidence_interval()[1] - self.ate]], 
                    fmt='o', capsize=5, capthick=2, color='blue', markersize=10)
        ax.axhline(0, color='red', linestyle='--', linewidth=1, label='No Effect')
        
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([])
        ax.set_ylabel('Average Treatment Effect')
        ax.set_title(f'DML Estimate: {self.ate:.4f} ± {1.96*self.std_error:.4f}')
        ax.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig


class DifferenceInDifferences:
    """双重差分估计器"""
    
    def __init__(
        self,
        data: pd.DataFrame,
        outcome: str,
        treatment: str,
        time: str,
        entity: str = None
    ):
        """
        Args:
            data: 面板数据
            outcome: 结果变量名
            treatment: 处理组指示变量名
            time: 时间（前后）指示变量名
            entity: 实体ID变量名
        """
        self.data = data.copy()
        self.outcome = outcome
        self.treatment = treatment
        self.time = time
        self.entity = entity
        
        self.effect = None
        self.std_error = None
        
    def estimate(self) -> Dict:
        """估计DiD效应"""
        # 计算四个均值
        y_treat_post = self.data[(self.data[self.treatment] == 1) & 
                                  (self.data[self.time] == 1)][self.outcome].mean()
        y_treat_pre = self.data[(self.data[self.treatment] == 1) & 
                                 (self.data[self.time] == 0)][self.outcome].mean()
        y_ctrl_post = self.data[(self.data[self.treatment] == 0) & 
                                 (self.data[self.time] == 1)][self.outcome].mean()
        y_ctrl_pre = self.data[(self.data[self.treatment] == 0) & 
                                (self.data[self.time] == 0)][self.outcome].mean()
        
        # DiD估计
        effect = (y_treat_post - y_treat_pre) - (y_ctrl_post - y_ctrl_pre)
        
        # 回归方法计算标准误
        self.data['interaction'] = self.data[self.treatment] * self.data[self.time]
        X = self.data[[self.treatment, self.time, 'interaction']].values
        X = np.column_stack([np.ones(len(X)), X])
        y = self.data[self.outcome].values
        
        # OLS
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ beta
        sigma2 = np.sum(residuals ** 2) / (len(y) - 4)
        var_beta = sigma2 * np.linalg.inv(X.T @ X)
        std_error = np.sqrt(var_beta[3, 3])
        
        self.effect = effect
        self.std_error = std_error
        
        return {
            'effect': effect,
            'std_error': std_error,
            'ci_lower': effect - 1.96 * std_error,
            'ci_upper': effect + 1.96 * std_error,
            'components': {
                'treat_post': y_treat_post,
                'treat_pre': y_treat_pre,
                'ctrl_post': y_ctrl_post,
                'ctrl_pre': y_ctrl_pre
            }
        }
        
    def plot_parallel_trends(self, save_path: str = None):
        """绘制平行趋势图"""
        import matplotlib.pyplot as plt
        
        result = self.estimate() if self.effect is None else None
        components = self.estimate()['components']
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # 处理组
        ax.plot([0, 1], [components['treat_pre'], components['treat_post']], 
               'o-', color='blue', linewidth=2, markersize=8, label='Treatment')
        
        # 对照组
        ax.plot([0, 1], [components['ctrl_pre'], components['ctrl_post']], 
               'o-', color='red', linewidth=2, markersize=8, label='Control')
        
        # 反事实
        counterfactual = components['treat_pre'] + (components['ctrl_post'] - components['ctrl_pre'])
        ax.plot([0, 1], [components['treat_pre'], counterfactual], 
               '--', color='blue', linewidth=1, alpha=0.5, label='Counterfactual')
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Pre-treatment', 'Post-treatment'])
        ax.set_ylabel(self.outcome)
        ax.set_title(f'Difference-in-Differences (Effect = {self.effect:.4f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig


class CausalForestEstimator:
    """因果森林估计器"""
    
    def __init__(
        self,
        n_estimators: int = 100,
        min_samples_leaf: int = 5,
        random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        
        self.model = None
        self.cate = None
        
    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        """拟合因果森林"""
        if ECONML_AVAILABLE:
            self.model = CausalForest(
                n_estimators=self.n_estimators,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state
            )
            self.model.fit(X, T, Y)
        else:
            # 简化实现：使用差异模型
            # 对处理组和对照组分别拟合回归
            self.model_t1 = RandomForestRegressor(n_estimators=self.n_estimators)
            self.model_t0 = RandomForestRegressor(n_estimators=self.n_estimators)
            
            mask_t1 = T == 1
            mask_t0 = T == 0
            
            self.model_t1.fit(X[mask_t1], Y[mask_t1])
            self.model_t0.fit(X[mask_t0], Y[mask_t0])
            
    def estimate_cate(self, X: np.ndarray) -> np.ndarray:
        """估计条件平均处理效应"""
        if ECONML_AVAILABLE and self.model is not None:
            self.cate = self.model.predict(X)
        else:
            # 简化实现
            y1_pred = self.model_t1.predict(X)
            y0_pred = self.model_t0.predict(X)
            self.cate = y1_pred - y0_pred
            
        return self.cate
        
    def plot_heterogeneity(self, X: np.ndarray = None, feature_idx: int = 0, 
                          feature_name: str = None, save_path: str = None):
        """绘制异质性效应图"""
        import matplotlib.pyplot as plt
        
        if self.cate is None:
            if X is not None:
                self.estimate_cate(X)
            else:
                raise ValueError("Need to estimate CATE first or provide X")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # CATE分布
        axes[0].hist(self.cate, bins=30, edgecolor='black', alpha=0.7)
        axes[0].axvline(np.mean(self.cate), color='red', linestyle='--', 
                        label=f'Mean = {np.mean(self.cate):.4f}')
        axes[0].set_xlabel('CATE')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Conditional Average Treatment Effects')
        axes[0].legend()
        
        # CATE vs 特征
        if X is not None:
            name = feature_name or f'Feature {feature_idx}'
            axes[1].scatter(X[:, feature_idx], self.cate, alpha=0.5)
            # 添加趋势线
            z = np.polyfit(X[:, feature_idx], self.cate, 1)
            p = np.poly1d(z)
            x_line = np.linspace(X[:, feature_idx].min(), X[:, feature_idx].max(), 100)
            axes[1].plot(x_line, p(x_line), 'r--', linewidth=2)
            axes[1].set_xlabel(name)
            axes[1].set_ylabel('CATE')
            axes[1].set_title(f'Heterogeneous Effect by {name}')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig


def generate_causal_report(Y, T, X, method='dml') -> Dict:
    """生成因果分析报告"""
    report = {}
    
    if method == 'dml':
        dml = DoubleMachineLearning(Y, T, X)
        report['dml'] = dml.estimate_ate()
        
    return report


if __name__ == '__main__':
    print("Testing Causal Inference Estimators...")
    
    # 模拟数据
    np.random.seed(42)
    n = 1000
    X = np.random.randn(n, 5)
    T = (np.random.randn(n) + X[:, 0] > 0).astype(int)
    Y = 2 * T + X[:, 0] + X[:, 1] + np.random.randn(n)
    
    # DML
    dml = DoubleMachineLearning(Y, T, X)
    result = dml.estimate_ate()
    print(f"DML ATE: {result['effect']:.4f} ± {result['std_error']:.4f}")
    print(f"True effect: 2.0")
    
    # DiD
    df = pd.DataFrame({
        'Y': np.random.randn(200),
        'treated': np.repeat([0, 1], 100),
        'post': np.tile([0, 1], 100)
    })
    df.loc[(df['treated'] == 1) & (df['post'] == 1), 'Y'] += 0.5
    
    did = DifferenceInDifferences(df, 'Y', 'treated', 'post')
    did_result = did.estimate()
    print(f"DiD Effect: {did_result['effect']:.4f}")
    
    print("Causal Inference test completed!")

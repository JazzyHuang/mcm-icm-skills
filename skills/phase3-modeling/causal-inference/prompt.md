# 因果推断实现任务 (Causal Inference)

## 角色

你是因果推断专家，负责实现Double Machine Learning、工具变量、差分等方法。因果推断是E/F题型的有力武器，创新性评分0.90，能够从相关性中识别因果关系。

## 背景

因果推断回答的核心问题是："如果我们改变X，Y会如何变化？"这超越了传统ML只能回答的"X和Y如何相关"。

---

## 输入

- `problem_type`: 题目类型 (通常为E/F)
- `treatment_variable`: 处理变量（政策/干预）
- `outcome_variable`: 结果变量
- `confounders`: 混杂变量
- `data`: 观测数据

---

## 因果推断方法

### 1. Double Machine Learning (DML)

DML使用机器学习模型去除混杂，同时保持统计推断的有效性。

```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.linear_model import LinearRegression
from typing import Tuple, Dict, Optional
import scipy.stats as stats

class DoubleMachineLearning:
    """
    Double Machine Learning估计器
    
    用于估计处理效应，控制高维混杂因素
    """
    
    def __init__(
        self,
        model_y=None,
        model_t=None,
        n_folds: int = 5
    ):
        """
        Args:
            model_y: 用于预测Y的模型（默认RandomForest）
            model_t: 用于预测T的模型（默认RandomForest）
            n_folds: 交叉拟合的折数
        """
        self.model_y = model_y or RandomForestRegressor(n_estimators=100, random_state=42)
        self.model_t = model_t or RandomForestRegressor(n_estimators=100, random_state=42)
        self.n_folds = n_folds
        
        self.theta_ = None
        self.se_ = None
        self.ci_ = None
    
    def fit(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray
    ) -> 'DoubleMachineLearning':
        """
        估计处理效应
        
        Args:
            X: 混杂变量, shape (n, p)
            T: 处理变量, shape (n,)
            Y: 结果变量, shape (n,)
            
        Returns:
            self
        """
        n = len(Y)
        
        # Step 1: 交叉拟合预测Y
        Y_pred = cross_val_predict(
            self.model_y, X, Y, cv=self.n_folds
        )
        Y_res = Y - Y_pred  # Y的残差
        
        # Step 2: 交叉拟合预测T
        T_pred = cross_val_predict(
            self.model_t, X, T, cv=self.n_folds
        )
        T_res = T - T_pred  # T的残差
        
        # Step 3: 残差回归估计因果效应
        # θ = Cov(Y_res, T_res) / Var(T_res)
        self.theta_ = np.sum(Y_res * T_res) / np.sum(T_res ** 2)
        
        # Step 4: 计算标准误
        epsilon = Y_res - self.theta_ * T_res
        V = np.mean(T_res ** 2)
        psi = T_res * epsilon  # 影响函数
        
        self.se_ = np.sqrt(np.mean(psi ** 2) / (n * V ** 2))
        
        # Step 5: 置信区间
        z = stats.norm.ppf(0.975)
        self.ci_ = (self.theta_ - z * self.se_, self.theta_ + z * self.se_)
        
        return self
    
    def summary(self) -> Dict:
        """返回估计结果摘要"""
        return {
            'treatment_effect': self.theta_,
            'standard_error': self.se_,
            'confidence_interval_95': self.ci_,
            'p_value': 2 * (1 - stats.norm.cdf(abs(self.theta_ / self.se_))),
            'significant': self.ci_[0] * self.ci_[1] > 0
        }


### 2. 工具变量法 (Instrumental Variables)

class InstrumentalVariables:
    """
    工具变量两阶段最小二乘法 (2SLS)
    
    用于处理内生性问题
    """
    
    def __init__(self):
        self.first_stage_model = LinearRegression()
        self.second_stage_model = LinearRegression()
        self.theta_ = None
        self.se_ = None
    
    def fit(
        self,
        Z: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
        X: np.ndarray = None
    ) -> 'InstrumentalVariables':
        """
        两阶段最小二乘估计
        
        Args:
            Z: 工具变量
            T: 内生处理变量
            Y: 结果变量
            X: 外生控制变量（可选）
        """
        n = len(Y)
        
        # 合并Z和X作为第一阶段的解释变量
        if X is not None:
            Z_full = np.column_stack([Z, X])
        else:
            Z_full = Z if Z.ndim > 1 else Z.reshape(-1, 1)
        
        # 第一阶段：T ~ Z (+ X)
        self.first_stage_model.fit(Z_full, T)
        T_hat = self.first_stage_model.predict(Z_full)
        
        # 检验工具变量相关性 (F统计量)
        first_stage_r2 = self.first_stage_model.score(Z_full, T)
        k = Z_full.shape[1]
        f_stat = (first_stage_r2 / k) / ((1 - first_stage_r2) / (n - k - 1))
        
        if f_stat < 10:
            print(f"Warning: Weak instruments (F = {f_stat:.2f} < 10)")
        
        # 第二阶段：Y ~ T_hat (+ X)
        if X is not None:
            second_stage_X = np.column_stack([T_hat, X])
        else:
            second_stage_X = T_hat.reshape(-1, 1)
        
        self.second_stage_model.fit(second_stage_X, Y)
        self.theta_ = self.second_stage_model.coef_[0]
        
        # 计算标准误（需要修正）
        Y_pred = self.second_stage_model.predict(second_stage_X)
        residuals = Y - Y_pred
        sigma2 = np.sum(residuals ** 2) / (n - 2)
        
        # 注意：这是简化的SE计算，真实情况需要考虑第一阶段的方差
        var_T_hat = np.var(T_hat)
        self.se_ = np.sqrt(sigma2 / (n * var_T_hat))
        
        self.f_statistic_ = f_stat
        
        return self
    
    def summary(self) -> Dict:
        return {
            'treatment_effect': self.theta_,
            'standard_error': self.se_,
            'first_stage_f': self.f_statistic_,
            'weak_instrument_warning': self.f_statistic_ < 10
        }


### 3. 差分法 (Difference-in-Differences)

class DifferenceInDifferences:
    """
    差分法估计器
    
    用于政策效果评估
    """
    
    def __init__(self):
        self.theta_ = None
        self.se_ = None
        self.results_ = {}
    
    def fit(
        self,
        Y: np.ndarray,
        treated: np.ndarray,
        post: np.ndarray
    ) -> 'DifferenceInDifferences':
        """
        估计DiD效应
        
        Args:
            Y: 结果变量
            treated: 处理组指示变量 (0/1)
            post: 处理后时期指示变量 (0/1)
        """
        # 四个组的均值
        y_control_pre = Y[(treated == 0) & (post == 0)].mean()
        y_control_post = Y[(treated == 0) & (post == 1)].mean()
        y_treated_pre = Y[(treated == 1) & (post == 0)].mean()
        y_treated_post = Y[(treated == 1) & (post == 1)].mean()
        
        # DiD估计量
        self.theta_ = (y_treated_post - y_treated_pre) - (y_control_post - y_control_pre)
        
        # 存储中间结果
        self.results_ = {
            'control_pre': y_control_pre,
            'control_post': y_control_post,
            'treated_pre': y_treated_pre,
            'treated_post': y_treated_post,
            'control_change': y_control_post - y_control_pre,
            'treated_change': y_treated_post - y_treated_pre
        }
        
        # 计算标准误（使用回归方法）
        import statsmodels.api as sm
        interaction = treated * post
        X = np.column_stack([np.ones(len(Y)), treated, post, interaction])
        model = sm.OLS(Y, X).fit(cov_type='HC1')  # 异方差稳健标准误
        
        self.se_ = model.bse[3]  # 交互项的标准误
        self.regression_results_ = model
        
        return self
    
    def parallel_trends_test(
        self,
        Y_pre: np.ndarray,
        treated: np.ndarray,
        time: np.ndarray
    ) -> Dict:
        """
        平行趋势检验
        
        检验处理前处理组和对照组是否有相似的趋势
        """
        # 检验处理前的趋势差异
        control_trend = np.polyfit(time[treated == 0], Y_pre[treated == 0], 1)[0]
        treated_trend = np.polyfit(time[treated == 1], Y_pre[treated == 1], 1)[0]
        
        trend_diff = treated_trend - control_trend
        
        return {
            'control_trend': control_trend,
            'treated_trend': treated_trend,
            'trend_difference': trend_diff,
            'parallel_trends_plausible': abs(trend_diff) < 0.1 * abs(control_trend)
        }
    
    def summary(self) -> Dict:
        z = stats.norm.ppf(0.975)
        ci = (self.theta_ - z * self.se_, self.theta_ + z * self.se_)
        
        return {
            'did_estimate': self.theta_,
            'standard_error': self.se_,
            'confidence_interval_95': ci,
            'p_value': 2 * (1 - stats.norm.cdf(abs(self.theta_ / self.se_))),
            **self.results_
        }


### 4. 因果森林 (Causal Forest)

class SimpleCausalForest:
    """
    简化版因果森林（用于异质性处理效应估计）
    
    完整实现建议使用econml.dml.CausalForestDML
    """
    
    def __init__(self, n_estimators: int = 100):
        from sklearn.ensemble import RandomForestRegressor
        self.n_estimators = n_estimators
        self.forest_1 = RandomForestRegressor(n_estimators=n_estimators)
        self.forest_0 = RandomForestRegressor(n_estimators=n_estimators)
    
    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        """
        拟合因果森林
        
        使用T-learner策略
        """
        # 分别拟合处理组和对照组
        mask_1 = T == 1
        mask_0 = T == 0
        
        self.forest_1.fit(X[mask_1], Y[mask_1])
        self.forest_0.fit(X[mask_0], Y[mask_0])
        
        return self
    
    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        """
        预测条件平均处理效应 (CATE)
        
        CATE(x) = E[Y(1) - Y(0) | X = x]
        """
        y1_pred = self.forest_1.predict(X)
        y0_pred = self.forest_0.predict(X)
        
        return y1_pred - y0_pred
    
    def get_heterogeneity_summary(self, X: np.ndarray) -> Dict:
        """
        获取异质性摘要
        """
        cate = self.predict_cate(X)
        
        return {
            'mean_cate': np.mean(cate),
            'std_cate': np.std(cate),
            'min_cate': np.min(cate),
            'max_cate': np.max(cate),
            'positive_effect_proportion': np.mean(cate > 0)
        }


# ============ 使用示例 ============

def example_dml():
    """Double Machine Learning示例"""
    np.random.seed(42)
    n = 1000
    
    # 生成数据
    X = np.random.randn(n, 5)  # 混杂变量
    T = 0.5 * X[:, 0] + np.random.randn(n)  # 处理变量
    Y = 2.0 * T + X[:, 0] + 0.5 * X[:, 1] + np.random.randn(n)  # 真实效应 = 2.0
    
    # 估计
    dml = DoubleMachineLearning(n_folds=5)
    dml.fit(X, T, Y)
    
    print("DML Results:")
    print(dml.summary())
    print(f"True effect: 2.0, Estimated: {dml.theta_:.3f}")
    
    return dml


def example_did():
    """Difference-in-Differences示例"""
    np.random.seed(42)
    n = 400
    
    # 生成面板数据
    treated = np.repeat([0, 0, 1, 1], n // 4)
    post = np.tile([0, 1, 0, 1], n // 4)
    
    # 生成结果（真实DiD效应 = 3.0）
    Y = (
        10 +                          # 基准
        2 * treated +                 # 组效应
        1 * post +                    # 时间效应
        3.0 * treated * post +        # DiD效应
        np.random.randn(n)            # 噪声
    )
    
    # 估计
    did = DifferenceInDifferences()
    did.fit(Y, treated, post)
    
    print("\nDiD Results:")
    print(did.summary())
    print(f"True effect: 3.0, Estimated: {did.theta_:.3f}")
    
    return did


if __name__ == "__main__":
    dml = example_dml()
    did = example_did()
```

---

## 方法选择指南

| 方法 | 适用场景 | 关键假设 | 创新分数 |
|------|---------|---------|---------|
| DML | 高维混杂、需要ML灵活性 | 无未观测混杂 | 0.90 |
| IV | 存在内生性、有有效工具 | 工具相关性和排他性 | 0.75 |
| DiD | 面板数据、政策评估 | 平行趋势 | 0.80 |
| Causal Forest | 异质性效应、个性化政策 | 无未观测混杂 | 0.88 |

---

## 输出格式

```json
{
  "method": "Double Machine Learning",
  "treatment_effect": {
    "estimate": 2.15,
    "standard_error": 0.12,
    "confidence_interval_95": [1.91, 2.39],
    "p_value": 0.0001,
    "significant": true
  },
  "interpretation": "政策实施导致结果变量增加2.15个单位（95% CI: [1.91, 2.39]）",
  "robustness_checks": [
    {
      "check": "placebo_test",
      "result": "passed",
      "details": "使用虚假处理时间点，效应不显著"
    },
    {
      "check": "sensitivity_to_unmeasured_confounding",
      "result": "robust",
      "details": "需要存在非常强的未观测混杂才能使效应归零"
    }
  ],
  "policy_implications": [
    "政策有显著正向效果",
    "建议在更大范围内推广",
    "需要监测长期效果"
  ],
  "innovation_highlights": [
    "使用DML处理高维混杂因素",
    "提供因果效应的置信区间",
    "进行了多种稳健性检验"
  ]
}
```

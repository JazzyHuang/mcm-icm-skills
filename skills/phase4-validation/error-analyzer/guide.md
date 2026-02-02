---
name: error-analyzer
description: 自动计算和分析模型误差，生成误差分布图、与基准方法对比、误差来源分析，为论文提供完整的误差分析章节。
dependencies: [model-solver, model-validator, sensitivity-analyzer]
outputs: [error_metrics, error_distribution, baseline_comparison, error_sources, visualization]
---

# 误差分析生成器 (Error Analyzer)

## 概述

误差分析是模型验证的核心环节，评委期望看到全面、深入、可视化的误差分析。本技能自动计算各类误差指标，生成误差分布图，与基准方法对比，并分析误差来源。

## 核心原则

### 评委期望

```
评委在误差分析章节关注:
1. 误差量化: 使用多种指标全面衡量误差
2. 误差分布: 误差是否均匀，是否有系统性偏差
3. 基准对比: 与现有方法相比如何
4. 误差来源: 为什么会有这些误差
5. 可接受性: 误差是否在可接受范围内
```

### O奖论文误差分析特点

- ✅ 使用3-5种误差指标
- ✅ 包含误差分布可视化
- ✅ 与至少2种基准方法对比
- ✅ 分析误差来源和模式
- ✅ 讨论误差的实际影响
- ❌ 避免: 仅报告单一指标、无可视化、无对比

## 误差指标体系

### 1. 回归任务误差指标

```python
REGRESSION_METRICS = {
    # 绝对误差类
    'MAE': {
        'name': 'Mean Absolute Error',
        'formula': 'MAE = (1/n) * Σ|y_i - ŷ_i|',
        'interpretation': '平均绝对偏差，对异常值不敏感',
        'unit': '与目标变量相同'
    },
    'RMSE': {
        'name': 'Root Mean Square Error',
        'formula': 'RMSE = √[(1/n) * Σ(y_i - ŷ_i)²]',
        'interpretation': '均方根误差，对大误差更敏感',
        'unit': '与目标变量相同'
    },
    'MaxAE': {
        'name': 'Maximum Absolute Error',
        'formula': 'MaxAE = max|y_i - ŷ_i|',
        'interpretation': '最大偏差，反映极端情况',
        'unit': '与目标变量相同'
    },
    
    # 相对误差类
    'MAPE': {
        'name': 'Mean Absolute Percentage Error',
        'formula': 'MAPE = (100/n) * Σ|y_i - ŷ_i|/|y_i|',
        'interpretation': '平均绝对百分比误差，适合比较不同量级',
        'unit': '%'
    },
    'RMSPE': {
        'name': 'Root Mean Square Percentage Error',
        'formula': 'RMSPE = √[(1/n) * Σ((y_i - ŷ_i)/y_i)²] * 100',
        'interpretation': '均方根百分比误差',
        'unit': '%'
    },
    
    # 拟合优度类
    'R2': {
        'name': 'Coefficient of Determination',
        'formula': 'R² = 1 - Σ(y_i - ŷ_i)² / Σ(y_i - ȳ)²',
        'interpretation': '决定系数，解释方差比例',
        'unit': '无量纲'
    },
    'Adjusted_R2': {
        'name': 'Adjusted R-squared',
        'formula': 'Adj R² = 1 - (1-R²)(n-1)/(n-p-1)',
        'interpretation': '调整后决定系数，考虑自由度',
        'unit': '无量纲'
    }
}
```

### 2. 分类任务误差指标

```python
CLASSIFICATION_METRICS = {
    'Accuracy': {
        'formula': 'Accuracy = (TP + TN) / (TP + TN + FP + FN)',
        'interpretation': '整体正确率'
    },
    'Precision': {
        'formula': 'Precision = TP / (TP + FP)',
        'interpretation': '预测为正中实际为正的比例'
    },
    'Recall': {
        'formula': 'Recall = TP / (TP + FN)',
        'interpretation': '实际为正中被正确预测的比例'
    },
    'F1_Score': {
        'formula': 'F1 = 2 * (Precision * Recall) / (Precision + Recall)',
        'interpretation': '精确率和召回率的调和平均'
    },
    'AUC_ROC': {
        'formula': 'Area Under ROC Curve',
        'interpretation': '分类器区分能力的综合指标'
    }
}
```

### 3. 优化任务误差指标

```python
OPTIMIZATION_METRICS = {
    'Optimality_Gap': {
        'formula': 'Gap = (f_found - f_optimal) / f_optimal * 100%',
        'interpretation': '与最优解的差距'
    },
    'Feasibility_Violation': {
        'formula': 'Violation = Σmax(0, g_i(x))',
        'interpretation': '约束违反程度'
    },
    'Convergence_Rate': {
        'formula': '迭代次数或收敛曲线',
        'interpretation': '达到一定精度所需的计算量'
    }
}
```

## 误差分析流程

```python
def analyze_errors(predictions, ground_truth, baselines, task_type):
    """
    全面分析模型误差
    
    Args:
        predictions: 模型预测值
        ground_truth: 真实值
        baselines: 基准方法结果字典
        task_type: 任务类型 ('regression', 'classification', 'optimization')
    
    Returns:
        dict: 完整的误差分析结果
    """
    analysis = {}
    
    # 1. 计算误差指标
    analysis['metrics'] = calculate_error_metrics(
        predictions, ground_truth, task_type
    )
    
    # 2. 分析误差分布
    analysis['distribution'] = analyze_error_distribution(
        predictions, ground_truth
    )
    
    # 3. 与基准方法对比
    analysis['baseline_comparison'] = compare_with_baselines(
        predictions, ground_truth, baselines
    )
    
    # 4. 误差来源分析
    analysis['error_sources'] = identify_error_sources(
        predictions, ground_truth
    )
    
    # 5. 生成可视化
    analysis['visualizations'] = generate_error_visualizations(analysis)
    
    # 6. 生成文本描述
    analysis['narrative'] = generate_error_narrative(analysis)
    
    return analysis
```

### 误差分布分析

```python
def analyze_error_distribution(predictions, ground_truth):
    """分析误差分布特征"""
    
    errors = predictions - ground_truth
    
    distribution = {
        # 基本统计量
        'mean': np.mean(errors),
        'std': np.std(errors),
        'median': np.median(errors),
        'skewness': scipy.stats.skew(errors),
        'kurtosis': scipy.stats.kurtosis(errors),
        
        # 分位数
        'percentiles': {
            '5%': np.percentile(errors, 5),
            '25%': np.percentile(errors, 25),
            '75%': np.percentile(errors, 75),
            '95%': np.percentile(errors, 95)
        },
        
        # 正态性检验
        'normality_test': scipy.stats.shapiro(errors),
        
        # 系统性偏差检验
        'bias_test': scipy.stats.ttest_1samp(errors, 0),
        
        # 异常值检测
        'outliers': detect_outliers(errors),
        
        # 误差模式
        'patterns': identify_error_patterns(errors, ground_truth)
    }
    
    return distribution
```

### 基准方法对比

```python
def compare_with_baselines(predictions, ground_truth, baselines):
    """与基准方法对比"""
    
    comparison = {
        'our_method': calculate_metrics(predictions, ground_truth),
        'baselines': {},
        'improvements': {},
        'statistical_tests': {}
    }
    
    for name, baseline_pred in baselines.items():
        # 计算基准方法的误差
        baseline_metrics = calculate_metrics(baseline_pred, ground_truth)
        comparison['baselines'][name] = baseline_metrics
        
        # 计算改进幅度
        improvement = {}
        for metric in baseline_metrics:
            if metric in ['R2', 'Accuracy', 'F1_Score']:  # 越大越好
                imp = (comparison['our_method'][metric] - baseline_metrics[metric]) / baseline_metrics[metric] * 100
            else:  # 越小越好
                imp = (baseline_metrics[metric] - comparison['our_method'][metric]) / baseline_metrics[metric] * 100
            improvement[metric] = imp
        comparison['improvements'][name] = improvement
        
        # 统计显著性检验
        comparison['statistical_tests'][name] = perform_significance_test(
            predictions, baseline_pred, ground_truth
        )
    
    return comparison
```

### 误差来源识别

```python
def identify_error_sources(predictions, ground_truth, features=None):
    """识别误差来源"""
    
    errors = np.abs(predictions - ground_truth)
    
    sources = {
        # 按数据区域分析
        'regional_analysis': analyze_by_region(errors, features),
        
        # 按时间段分析（如果适用）
        'temporal_analysis': analyze_by_time(errors),
        
        # 按目标值大小分析
        'magnitude_analysis': analyze_by_magnitude(errors, ground_truth),
        
        # 按特征值分析
        'feature_importance': analyze_by_features(errors, features),
        
        # 边界效应分析
        'boundary_effects': analyze_boundary_effects(errors),
        
        # 可能的系统性误差来源
        'systematic_sources': [
            identify_data_quality_issues(),
            identify_model_limitations(),
            identify_assumption_violations()
        ]
    }
    
    return sources
```

## 可视化生成

### 误差分布图

```python
def generate_error_distribution_plot(errors):
    """生成误差分布直方图"""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 直方图
    axes[0].hist(errors, bins=50, density=True, alpha=0.7, color='steelblue')
    axes[0].axvline(x=0, color='red', linestyle='--', label='Zero Error')
    axes[0].axvline(x=np.mean(errors), color='green', linestyle='-', label=f'Mean: {np.mean(errors):.3f}')
    axes[0].set_xlabel('Error')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Error Distribution')
    axes[0].legend()
    
    # Q-Q图
    scipy.stats.probplot(errors, plot=axes[1])
    axes[1].set_title('Q-Q Plot (Normal Distribution)')
    
    return fig
```

### 预测-实际对比图

```python
def generate_prediction_vs_actual_plot(predictions, ground_truth):
    """生成预测值vs实际值散点图"""
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 散点图
    ax.scatter(ground_truth, predictions, alpha=0.5, s=20)
    
    # 理想线
    min_val = min(ground_truth.min(), predictions.min())
    max_val = max(ground_truth.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    # 拟合线
    z = np.polyfit(ground_truth, predictions, 1)
    p = np.poly1d(z)
    ax.plot(ground_truth, p(ground_truth), 'g-', alpha=0.8, label=f'Fit: y={z[0]:.3f}x+{z[1]:.3f}')
    
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title('Prediction vs Actual')
    ax.legend()
    
    # 添加R²标注
    r2 = calculate_r2(predictions, ground_truth)
    ax.annotate(f'R² = {r2:.4f}', xy=(0.05, 0.95), xycoords='axes fraction')
    
    return fig
```

### 基准对比条形图

```python
def generate_baseline_comparison_plot(comparison):
    """生成基准方法对比图"""
    
    metrics = ['RMSE', 'MAE', 'R2']
    methods = ['Our Method'] + list(comparison['baselines'].keys())
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(4*len(metrics), 5))
    
    for i, metric in enumerate(metrics):
        values = [comparison['our_method'][metric]]
        values.extend([comparison['baselines'][m][metric] for m in comparison['baselines']])
        
        colors = ['#2ecc71' if j == 0 else '#3498db' for j in range(len(methods))]
        
        axes[i].bar(methods, values, color=colors)
        axes[i].set_title(metric)
        axes[i].set_ylabel('Value')
        
        # 添加数值标签
        for j, v in enumerate(values):
            axes[i].annotate(f'{v:.4f}', xy=(j, v), ha='center', va='bottom')
    
    plt.tight_layout()
    return fig
```

## 输出格式

```json
{
  "error_metrics": {
    "regression": {
      "MAE": 0.0234,
      "RMSE": 0.0312,
      "MAPE": 3.45,
      "R2": 0.9567,
      "Adjusted_R2": 0.9523
    }
  },
  "error_distribution": {
    "mean": -0.0012,
    "std": 0.0311,
    "median": -0.0008,
    "skewness": 0.23,
    "kurtosis": 2.87,
    "normality_test": {
      "statistic": 0.9876,
      "p_value": 0.234,
      "is_normal": true
    },
    "bias_test": {
      "statistic": -0.89,
      "p_value": 0.374,
      "has_bias": false
    },
    "outliers": {
      "count": 12,
      "percentage": 1.2,
      "indices": [45, 123, ...]
    }
  },
  "baseline_comparison": {
    "our_method": {"RMSE": 0.0312, "MAE": 0.0234, "R2": 0.9567},
    "baselines": {
      "Linear Regression": {"RMSE": 0.0567, "MAE": 0.0423, "R2": 0.8934},
      "Random Forest": {"RMSE": 0.0423, "MAE": 0.0312, "R2": 0.9234}
    },
    "improvements": {
      "Linear Regression": {"RMSE": 44.97, "MAE": 44.68, "R2": 7.09},
      "Random Forest": {"RMSE": 26.24, "MAE": 25.00, "R2": 3.61}
    },
    "statistical_tests": {
      "Linear Regression": {"test": "paired_t", "p_value": 0.0001, "significant": true},
      "Random Forest": {"test": "paired_t", "p_value": 0.0023, "significant": true}
    }
  },
  "error_sources": {
    "magnitude_analysis": {
      "low_values": {"mean_error": 0.045, "percentage": 15},
      "medium_values": {"mean_error": 0.028, "percentage": 70},
      "high_values": {"mean_error": 0.052, "percentage": 15}
    },
    "systematic_sources": [
      "Higher errors observed at boundary conditions",
      "Slight underestimation bias in extreme events"
    ]
  },
  "visualizations": {
    "error_distribution": "figures/error_distribution.pdf",
    "prediction_vs_actual": "figures/pred_vs_actual.pdf",
    "baseline_comparison": "figures/baseline_comparison.pdf",
    "error_by_magnitude": "figures/error_by_magnitude.pdf"
  },
  "narrative": {
    "metrics_description": "Our model achieves an RMSE of 0.0312 and R² of 0.9567...",
    "distribution_analysis": "The error distribution is approximately normal with no significant bias...",
    "comparison_summary": "Our method outperforms all baselines, achieving 45% lower RMSE than linear regression...",
    "sources_discussion": "Error analysis reveals higher prediction errors at boundary conditions..."
  }
}
```

## LaTeX输出模板

```latex
\subsection{Error Analysis}

Table \ref{tab:error_metrics} summarizes the error metrics of our model.

\begin{table}[h]
\centering
\caption{Error Metrics of the Proposed Model}
\label{tab:error_metrics}
\begin{tabular}{lc}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
RMSE & 0.0312 \\
MAE & 0.0234 \\
MAPE & 3.45\% \\
$R^2$ & 0.9567 \\
Adjusted $R^2$ & 0.9523 \\
\bottomrule
\end{tabular}
\end{table}

Figure \ref{fig:error_dist} shows the distribution of prediction errors. 
The distribution is approximately normal (Shapiro-Wilk test: $W = 0.988$, 
$p = 0.234$) with mean close to zero ($\mu = -0.0012$), indicating no 
systematic bias. The standard deviation of 0.0311 suggests consistent 
prediction quality across samples.

\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{figures/error_distribution.pdf}
\caption{Distribution of prediction errors (left) and Q-Q plot (right).}
\label{fig:error_dist}
\end{figure}

Table \ref{tab:baseline_comp} compares our method with baseline approaches.

\begin{table}[h]
\centering
\caption{Comparison with Baseline Methods}
\label{tab:baseline_comp}
\begin{tabular}{lccc}
\toprule
\textbf{Method} & \textbf{RMSE} & \textbf{MAE} & \textbf{$R^2$} \\
\midrule
Linear Regression & 0.0567 & 0.0423 & 0.8934 \\
Random Forest & 0.0423 & 0.0312 & 0.9234 \\
\textbf{Our Method} & \textbf{0.0312} & \textbf{0.0234} & \textbf{0.9567} \\
\midrule
Improvement vs. LR & 44.97\% & 44.68\% & 7.09\% \\
Improvement vs. RF & 26.24\% & 25.00\% & 3.61\% \\
\bottomrule
\end{tabular}
\end{table}

Our method significantly outperforms both baselines (paired t-test, $p < 0.01$).
```

## 与其他技能集成

### 上游依赖
- `model-solver`: 获取模型预测结果
- `model-validator`: 获取验证数据
- `sensitivity-analyzer`: 了解参数敏感性

### 下游输出
- `section-writer`: 提供误差分析章节内容
- `chart-generator`: 提供可视化数据
- `quality-reviewer`: 提供验证完整性评估

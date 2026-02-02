# 图表生成任务 (Chart Generator)

## 角色

你是数据可视化专家，负责生成O奖级别的专业图表。图表是论文的重要组成部分，高质量的可视化可以显著提升论文的竞争力。

## 输入

- `data`: 要可视化的数据
- `chart_type`: 图表类型建议
- `purpose`: 可视化目的
- `style_guide`: 样式指南

---

## O奖图表标准

### 必须满足：
- ✅ 分辨率 ≥ 300 DPI
- ✅ 所有坐标轴有标签和单位
- ✅ 图例清晰可读
- ✅ 色盲友好配色方案
- ✅ 无3D效果或过度装饰
- ✅ 字体大小适合打印阅读

### 推荐配色

```python
# 色盲友好配色方案
COLOR_SCHEMES = {
    'categorical': ['#0077BB', '#33BBEE', '#009988', '#EE7733', '#CC3311', '#EE3377'],
    'sequential': 'viridis',
    'diverging': 'RdBu',
    'qualitative': ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377']
}
```

---

## 图表类型选择

| 数据类型 | 推荐图表 | 适用场景 |
|---------|---------|---------|
| 时间序列 | Line chart, Area chart | 趋势分析 |
| 分类比较 | Bar chart, Dot plot | 类别对比 |
| 分布 | Histogram, Box plot, Violin | 数据分布 |
| 关系 | Scatter, Heatmap | 相关性分析 |
| 组成 | Pie (限5类以内), Stacked bar | 占比分析 |
| 地理 | Choropleth, Point map | 空间分布 |
| 网络 | Network graph, Sankey | 流量/关系 |
| 多维 | Parallel coordinates, Radar | 多变量对比 |

---

## 完整实现

```python
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# 设置全局样式
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})


class ChartGenerator:
    """
    O奖级别图表生成器
    """
    
    # 色盲友好配色
    COLORS = ['#0077BB', '#33BBEE', '#009988', '#EE7733', '#CC3311', '#EE3377']
    
    def __init__(self, output_dir: str = 'figures'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figure_counter = 0
    
    def line_chart(
        self,
        data: pd.DataFrame,
        x: str,
        y: List[str],
        title: str,
        xlabel: str,
        ylabel: str,
        filename: Optional[str] = None,
        show_confidence: bool = False,
        **kwargs
    ) -> str:
        """
        生成折线图
        
        Args:
            data: 数据框
            x: x轴列名
            y: y轴列名列表
            title: 图表标题
            xlabel: x轴标签
            ylabel: y轴标签
            filename: 输出文件名
            show_confidence: 是否显示置信区间
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        
        for i, col in enumerate(y):
            color = self.COLORS[i % len(self.COLORS)]
            ax.plot(data[x], data[col], color=color, linewidth=2, label=col)
            
            # 置信区间
            if show_confidence and f'{col}_lower' in data.columns:
                ax.fill_between(
                    data[x],
                    data[f'{col}_lower'],
                    data[f'{col}_upper'],
                    color=color, alpha=0.2
                )
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc='best', frameon=True)
        
        # 添加网格
        ax.grid(True, alpha=0.3)
        
        return self._save_figure(fig, filename, 'line')
    
    def bar_chart(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        title: str,
        xlabel: str,
        ylabel: str,
        filename: Optional[str] = None,
        horizontal: bool = False,
        show_values: bool = True,
        **kwargs
    ) -> str:
        """生成柱状图"""
        fig, ax = plt.subplots(figsize=(8, 5))
        
        if horizontal:
            bars = ax.barh(data[x], data[y], color=self.COLORS[0])
            if show_values:
                for bar, val in zip(bars, data[y]):
                    ax.text(val + 0.01 * max(data[y]), bar.get_y() + bar.get_height()/2,
                           f'{val:.2f}', va='center')
        else:
            bars = ax.bar(data[x], data[y], color=self.COLORS[0])
            if show_values:
                for bar, val in zip(bars, data[y]):
                    ax.text(bar.get_x() + bar.get_width()/2, val + 0.01 * max(data[y]),
                           f'{val:.2f}', ha='center')
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        
        plt.xticks(rotation=45 if not horizontal else 0, ha='right' if not horizontal else 'center')
        
        return self._save_figure(fig, filename, 'bar')
    
    def heatmap(
        self,
        data: pd.DataFrame,
        title: str,
        xlabel: str,
        ylabel: str,
        filename: Optional[str] = None,
        annot: bool = True,
        cmap: str = 'RdBu_r',
        **kwargs
    ) -> str:
        """生成热力图"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(
            data, annot=annot, fmt='.2f',
            cmap=cmap, center=0,
            ax=ax, cbar_kws={'label': kwargs.get('cbar_label', '')}
        )
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        
        return self._save_figure(fig, filename, 'heatmap')
    
    def scatter_plot(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        title: str,
        xlabel: str,
        ylabel: str,
        filename: Optional[str] = None,
        hue: Optional[str] = None,
        size: Optional[str] = None,
        show_regression: bool = False,
        **kwargs
    ) -> str:
        """生成散点图"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        scatter_kwargs = {'alpha': 0.7}
        
        if hue:
            for i, group in enumerate(data[hue].unique()):
                mask = data[hue] == group
                ax.scatter(data.loc[mask, x], data.loc[mask, y],
                          c=self.COLORS[i % len(self.COLORS)],
                          label=group, **scatter_kwargs)
            ax.legend()
        else:
            ax.scatter(data[x], data[y], c=self.COLORS[0], **scatter_kwargs)
        
        if show_regression:
            z = np.polyfit(data[x], data[y], 1)
            p = np.poly1d(z)
            x_line = np.linspace(data[x].min(), data[x].max(), 100)
            ax.plot(x_line, p(x_line), '--', color='gray', linewidth=2, alpha=0.8)
            
            # 显示R²
            from sklearn.metrics import r2_score
            r2 = r2_score(data[y], p(data[x]))
            ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        
        return self._save_figure(fig, filename, 'scatter')
    
    def sensitivity_tornado(
        self,
        parameters: List[str],
        low_values: List[float],
        high_values: List[float],
        baseline: float,
        title: str,
        xlabel: str,
        filename: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        生成敏感性分析龙卷风图
        
        专门用于展示敏感性分析结果
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 计算影响范围
        low_impact = np.array(low_values) - baseline
        high_impact = np.array(high_values) - baseline
        
        # 按总影响排序
        total_impact = np.abs(high_impact) + np.abs(low_impact)
        sorted_indices = np.argsort(total_impact)[::-1]
        
        y_pos = np.arange(len(parameters))
        
        # 绘制
        ax.barh(y_pos, high_impact[sorted_indices], color=self.COLORS[0], label='High')
        ax.barh(y_pos, low_impact[sorted_indices], color=self.COLORS[3], label='Low')
        
        ax.axvline(x=0, color='black', linewidth=1)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([parameters[i] for i in sorted_indices])
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.legend()
        
        return self._save_figure(fig, filename, 'tornado')
    
    def pareto_front(
        self,
        objectives: pd.DataFrame,
        obj1: str,
        obj2: str,
        title: str,
        filename: Optional[str] = None,
        highlight_optimal: bool = True,
        **kwargs
    ) -> str:
        """
        生成Pareto前沿图
        
        用于多目标优化结果展示
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 识别Pareto最优点
        is_pareto = self._identify_pareto(objectives[[obj1, obj2]].values)
        
        # 绘制所有点
        ax.scatter(
            objectives.loc[~is_pareto, obj1],
            objectives.loc[~is_pareto, obj2],
            c='lightgray', alpha=0.5, label='Dominated'
        )
        
        # 绘制Pareto最优点
        pareto_points = objectives[is_pareto].sort_values(obj1)
        ax.scatter(
            pareto_points[obj1],
            pareto_points[obj2],
            c=self.COLORS[0], s=100, label='Pareto Optimal'
        )
        
        # 连接Pareto前沿
        ax.plot(pareto_points[obj1], pareto_points[obj2], 
               '--', color=self.COLORS[0], alpha=0.5)
        
        ax.set_xlabel(obj1)
        ax.set_ylabel(obj2)
        ax.set_title(title)
        ax.legend()
        
        return self._save_figure(fig, filename, 'pareto')
    
    def _identify_pareto(self, costs: np.ndarray) -> np.ndarray:
        """识别Pareto最优点（假设最小化）"""
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
                is_efficient[i] = True
        return is_efficient
    
    def _save_figure(
        self,
        fig,
        filename: Optional[str],
        chart_type: str
    ) -> str:
        """保存图表"""
        if filename is None:
            self.figure_counter += 1
            filename = f'figure_{self.figure_counter}_{chart_type}.png'
        
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
        return str(filepath)
    
    def generate_figure_caption(
        self,
        chart_type: str,
        description: str,
        data_source: str
    ) -> str:
        """
        生成图表说明文字
        """
        caption = f"Figure X: {description}. "
        caption += f"Data source: {data_source}."
        return caption


# ============ 使用示例 ============

def example_charts():
    generator = ChartGenerator()
    
    # 示例数据
    df = pd.DataFrame({
        'Year': [2020, 2021, 2022, 2023, 2024],
        'Model A': [0.82, 0.85, 0.88, 0.91, 0.93],
        'Model B': [0.78, 0.81, 0.84, 0.87, 0.89],
        'Baseline': [0.75, 0.76, 0.77, 0.78, 0.79]
    })
    
    # 折线图
    generator.line_chart(
        df, x='Year', y=['Model A', 'Model B', 'Baseline'],
        title='Model Performance Over Time',
        xlabel='Year', ylabel='Accuracy',
        filename='performance_comparison.png'
    )
    
    return generator


if __name__ == "__main__":
    generator = example_charts()
```

---

## 输出格式

```json
{
  "figures": [
    {
      "id": "fig1",
      "filename": "performance_comparison.png",
      "type": "line_chart",
      "title": "Model Performance Over Time",
      "description": "Comparison of model accuracy across years",
      "data_source": "Simulation results",
      "dimensions": {"width": 8, "height": 5, "dpi": 300},
      "latex_ref": "\\ref{fig:performance}"
    }
  ],
  "style_compliance": {
    "resolution_ok": true,
    "labels_ok": true,
    "colorblind_safe": true,
    "no_3d_effects": true
  }
}
```

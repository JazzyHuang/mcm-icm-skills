"""
Sensitivity Analysis Charts
敏感性分析专用图表

包含Sobol指数图、龙卷风图、Morris筛选图等
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple

try:
    from .nature_style import set_nature_style, NATURE_COLORS
except ImportError:
    NATURE_COLORS = ['#0077BB', '#33BBEE', '#009988', '#EE7733', '#CC3311']
    def set_nature_style():
        pass


class SensitivityCharts:
    """敏感性分析专用图表"""
    
    def __init__(self, style: str = 'nature'):
        self.style = style
        self.colors = NATURE_COLORS
        set_nature_style()
        
    def sobol_bar(self, S1: Dict[str, float], ST: Dict[str, float],
                  title: str = 'Sobol Sensitivity Indices',
                  save_path: str = None):
        """
        Sobol指数柱状图（一阶 vs 总阶）
        
        Args:
            S1: 一阶Sobol指数 {'param1': 0.3, 'param2': 0.5, ...}
            ST: 总阶Sobol指数
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        params = list(S1.keys())
        n_params = len(params)
        
        # 排序
        sorted_idx = np.argsort([ST[p] for p in params])[::-1]
        params_sorted = [params[i] for i in sorted_idx]
        
        s1_values = [S1[p] for p in params_sorted]
        st_values = [ST[p] for p in params_sorted]
        
        # 柱状图
        bar_width = 0.35
        x = np.arange(n_params)
        
        ax.barh(x - bar_width/2, s1_values, bar_width, 
               label='First-order (S1)', color=self.colors[0], alpha=0.8)
        ax.barh(x + bar_width/2, st_values, bar_width,
               label='Total-order (ST)', color=self.colors[1], alpha=0.8)
        
        ax.set_yticks(x)
        ax.set_yticklabels(params_sorted)
        ax.set_xlabel('Sensitivity Index')
        ax.set_title(title)
        ax.legend()
        ax.axvline(0, color='black', linewidth=0.5)
        
        # 添加数值标签
        for i, (s1, st) in enumerate(zip(s1_values, st_values)):
            ax.text(s1 + 0.01, i - bar_width/2, f'{s1:.3f}', va='center', fontsize=8)
            ax.text(st + 0.01, i + bar_width/2, f'{st:.3f}', va='center', fontsize=8)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
        
    def sobol_pie(self, S1: Dict[str, float], title: str = 'Variance Decomposition',
                  save_path: str = None):
        """Sobol指数饼图（方差分解）"""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 排序并取前N个
        sorted_items = sorted(S1.items(), key=lambda x: x[1], reverse=True)
        top_n = sorted_items[:7]
        other = sum(v for _, v in sorted_items[7:])
        
        labels = [item[0] for item in top_n]
        sizes = [item[1] for item in top_n]
        
        if other > 0:
            labels.append('Other')
            sizes.append(other)
            
        # 归一化
        total = sum(sizes)
        sizes = [s/total for s in sizes]
        
        wedges, texts, autotexts = ax.pie(
            sizes, 
            labels=labels,
            autopct='%1.1f%%',
            colors=self.colors[:len(sizes)],
            explode=[0.02] * len(sizes),
            startangle=90
        )
        
        ax.set_title(title)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
        
    def tornado_diagram(self, params: List[str], low_values: List[float],
                        high_values: List[float], baseline: float,
                        title: str = 'Tornado Diagram',
                        xlabel: str = 'Output Value', save_path: str = None):
        """
        龙卷风图（单因素敏感性）
        
        Args:
            params: 参数名列表
            low_values: 参数取低值时的输出
            high_values: 参数取高值时的输出
            baseline: 基准输出值
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 计算影响范围并排序
        ranges = [abs(h - l) for l, h in zip(low_values, high_values)]
        sorted_idx = np.argsort(ranges)[::-1]
        
        params_sorted = [params[i] for i in sorted_idx]
        low_sorted = [low_values[i] for i in sorted_idx]
        high_sorted = [high_values[i] for i in sorted_idx]
        
        y = np.arange(len(params))
        
        # 绘制柱状
        for i, (low, high, param) in enumerate(zip(low_sorted, high_sorted, params_sorted)):
            # 低值影响
            if low < baseline:
                ax.barh(i, baseline - low, left=low, height=0.6,
                       color=self.colors[0], alpha=0.8)
            else:
                ax.barh(i, low - baseline, left=baseline, height=0.6,
                       color=self.colors[0], alpha=0.8)
                       
            # 高值影响
            if high > baseline:
                ax.barh(i, high - baseline, left=baseline, height=0.6,
                       color=self.colors[1], alpha=0.8)
            else:
                ax.barh(i, baseline - high, left=high, height=0.6,
                       color=self.colors[1], alpha=0.8)
                       
        ax.axvline(baseline, color='black', linestyle='-', linewidth=1.5)
        ax.set_yticks(y)
        ax.set_yticklabels(params_sorted)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        
        # 图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=self.colors[0], label='Low Value'),
            Patch(facecolor=self.colors[1], label='High Value')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
        
    def morris_scatter(self, mu_star: Dict[str, float], sigma: Dict[str, float],
                       title: str = 'Morris Screening',
                       save_path: str = None):
        """
        Morris筛选散点图
        
        Args:
            mu_star: 绝对平均效应 {'param1': 0.5, ...}
            sigma: 标准差
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        params = list(mu_star.keys())
        mu_values = [mu_star[p] for p in params]
        sigma_values = [sigma[p] for p in params]
        
        # 散点图
        scatter = ax.scatter(mu_values, sigma_values, 
                            s=100, c=self.colors[0], alpha=0.7, edgecolors='white')
        
        # 添加标签
        for i, param in enumerate(params):
            ax.annotate(param, (mu_values[i], sigma_values[i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
            
        # 添加参考线
        max_val = max(max(mu_values), max(sigma_values))
        ax.plot([0, max_val], [0, max_val], '--', color='gray', alpha=0.5, 
               label='σ = μ*')
        ax.plot([0, max_val], [0, 0.5*max_val], ':', color='gray', alpha=0.5,
               label='σ = 0.5μ*')
        
        ax.set_xlabel('μ* (Mean of Absolute Elementary Effects)')
        ax.set_ylabel('σ (Standard Deviation)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 分区标注
        ax.text(max_val*0.7, max_val*0.1, 'Linear/Additive', fontsize=10, alpha=0.7)
        ax.text(max_val*0.2, max_val*0.8, 'Nonlinear/\nInteractive', fontsize=10, alpha=0.7)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
        
    def spider_sensitivity(self, params: List[str], 
                           sensitivity_data: Dict[str, List[Tuple[float, float]]],
                           title: str = 'Spider Sensitivity Plot',
                           save_path: str = None):
        """
        蜘蛛图（参数变化对输出的影响）
        
        Args:
            params: 参数名列表
            sensitivity_data: {param: [(变化百分比, 输出值), ...]}
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, param in enumerate(params):
            data = sensitivity_data[param]
            x = [d[0] for d in data]  # 变化百分比
            y = [d[1] for d in data]  # 输出值
            ax.plot(x, y, 'o-', color=self.colors[i % len(self.colors)],
                   linewidth=2, markersize=6, label=param)
            
        ax.axvline(0, color='black', linestyle='--', linewidth=0.5)
        ax.set_xlabel('Parameter Change (%)')
        ax.set_ylabel('Output Value')
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
        
    def heatmap_interaction(self, params: List[str], interaction_matrix: np.ndarray,
                           title: str = 'Parameter Interaction Heatmap',
                           save_path: str = None):
        """
        参数交互作用热力图
        
        Args:
            params: 参数名列表
            interaction_matrix: 交互作用矩阵 (n_params x n_params)
        """
        fig, ax = plt.subplots(figsize=(8, 7))
        
        im = ax.imshow(interaction_matrix, cmap='RdYlBu_r', aspect='auto')
        
        # 标签
        ax.set_xticks(range(len(params)))
        ax.set_yticks(range(len(params)))
        ax.set_xticklabels(params, rotation=45, ha='right')
        ax.set_yticklabels(params)
        
        # 数值标注
        for i in range(len(params)):
            for j in range(len(params)):
                val = interaction_matrix[i, j]
                color = 'white' if abs(val) > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                       color=color, fontsize=8)
                       
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label='Interaction Index')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
        
    def confidence_band(self, x: np.ndarray, y_mean: np.ndarray,
                        y_std: np.ndarray, n_std: float = 2,
                        xlabel: str = '', ylabel: str = '',
                        title: str = 'Uncertainty Band',
                        save_path: str = None):
        """
        不确定性带图
        
        Args:
            x: x轴数据
            y_mean: y均值
            y_std: y标准差
            n_std: 置信区间标准差倍数
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 均值线
        ax.plot(x, y_mean, color=self.colors[0], linewidth=2, label='Mean')
        
        # 置信带
        y_lower = y_mean - n_std * y_std
        y_upper = y_mean + n_std * y_std
        ax.fill_between(x, y_lower, y_upper, color=self.colors[0], alpha=0.2,
                       label=f'±{n_std}σ')
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig


if __name__ == '__main__':
    print("Testing Sensitivity Charts...")
    
    charts = SensitivityCharts()
    
    # Sobol指数
    S1 = {'param_a': 0.35, 'param_b': 0.25, 'param_c': 0.15, 'param_d': 0.10, 'param_e': 0.08}
    ST = {'param_a': 0.45, 'param_b': 0.32, 'param_c': 0.20, 'param_d': 0.15, 'param_e': 0.12}
    charts.sobol_bar(S1, ST, title='Sobol Indices Example')
    
    # 龙卷风图
    params = ['Price', 'Demand', 'Cost', 'Efficiency']
    low_values = [85, 92, 78, 95]
    high_values = [115, 108, 122, 105]
    baseline = 100
    charts.tornado_diagram(params, low_values, high_values, baseline)
    
    # Morris散点
    mu_star = {'A': 0.8, 'B': 0.6, 'C': 0.4, 'D': 0.3, 'E': 0.2}
    sigma = {'A': 0.3, 'B': 0.5, 'C': 0.2, 'D': 0.4, 'E': 0.1}
    charts.morris_scatter(mu_star, sigma)
    
    print("Sensitivity Charts test completed!")
    plt.show()

"""
Sensitivity Analysis Charts (Enhanced Version)
敏感性分析专用图表（增强版）

包含Sobol指数图、龙卷风图、Morris筛选图、交互效应热力图、不确定性区间图等
新增：交互效应热力图、参数重要性排序图、二阶Sobol可视化
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

try:
    from .nature_style import set_nature_style, NATURE_COLORS
except ImportError:
    NATURE_COLORS = ['#0077BB', '#33BBEE', '#009988', '#EE7733', '#CC3311', '#EE3377', '#BBBBBB']
    def set_nature_style():
        pass


class SensitivityCharts:
    """敏感性分析专用图表（增强版）"""
    
    def __init__(self, style: str = 'nature'):
        self.style = style
        self.colors = NATURE_COLORS
        set_nature_style()
        
    def sobol_bar(self, S1: Dict[str, float], ST: Dict[str, float],
                  S1_conf: Dict[str, Tuple[float, float]] = None,
                  ST_conf: Dict[str, Tuple[float, float]] = None,
                  title: str = 'Sobol Sensitivity Indices',
                  save_path: str = None):
        """
        Sobol指数柱状图（一阶 vs 总阶）带置信区间
        
        Args:
            S1: 一阶Sobol指数 {'param1': 0.3, 'param2': 0.5, ...}
            ST: 总阶Sobol指数
            S1_conf: 一阶指数置信区间 {'param1': (0.25, 0.35), ...}
            ST_conf: 总阶指数置信区间
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        params = list(S1.keys())
        n_params = len(params)
        
        # 按总阶指数排序
        sorted_idx = np.argsort([ST[p] for p in params])[::-1]
        params_sorted = [params[i] for i in sorted_idx]
        
        s1_values = [S1[p] for p in params_sorted]
        st_values = [ST[p] for p in params_sorted]
        
        # 置信区间
        s1_errors = None
        st_errors = None
        if S1_conf:
            s1_errors = [[S1[p] - S1_conf[p][0] for p in params_sorted],
                         [S1_conf[p][1] - S1[p] for p in params_sorted]]
        if ST_conf:
            st_errors = [[ST[p] - ST_conf[p][0] for p in params_sorted],
                         [ST_conf[p][1] - ST[p] for p in params_sorted]]
        
        # 柱状图
        bar_width = 0.35
        x = np.arange(n_params)
        
        bars1 = ax.barh(x - bar_width/2, s1_values, bar_width, 
                        label='First-order ($S_1$)', color=self.colors[0], alpha=0.8,
                        xerr=s1_errors, capsize=3, error_kw={'elinewidth': 1})
        bars2 = ax.barh(x + bar_width/2, st_values, bar_width,
                        label='Total-order ($S_T$)', color=self.colors[1], alpha=0.8,
                        xerr=st_errors, capsize=3, error_kw={'elinewidth': 1})
        
        ax.set_yticks(x)
        ax.set_yticklabels(params_sorted)
        ax.set_xlabel('Sensitivity Index')
        ax.set_title(title)
        ax.legend(loc='lower right')
        ax.axvline(0, color='black', linewidth=0.5)
        ax.set_xlim(0, max(max(st_values), 1.0) * 1.15)
        
        # 添加数值标签
        for i, (s1, st) in enumerate(zip(s1_values, st_values)):
            ax.text(s1 + 0.02, i - bar_width/2, f'{s1:.3f}', va='center', fontsize=8)
            ax.text(st + 0.02, i + bar_width/2, f'{st:.3f}', va='center', fontsize=8)
        
        # 添加交互效应标注
        ax.text(0.98, 0.02, '$S_T - S_1$ = Interaction Effect', 
                transform=ax.transAxes, fontsize=9, ha='right', style='italic', alpha=0.7)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    
    def sobol_interaction_heatmap(self, S2: Dict[Tuple[str, str], float],
                                   params: List[str],
                                   title: str = 'Second-Order Sobol Indices (Interaction Effects)',
                                   save_path: str = None):
        """
        二阶Sobol指数热力图（交互效应）- 新增功能
        
        Args:
            S2: 二阶Sobol指数 {('param1', 'param2'): 0.05, ...}
            params: 参数名列表
        """
        fig, ax = plt.subplots(figsize=(9, 8))
        
        n_params = len(params)
        interaction_matrix = np.zeros((n_params, n_params))
        
        # 填充矩阵
        for (p1, p2), value in S2.items():
            if p1 in params and p2 in params:
                i, j = params.index(p1), params.index(p2)
                interaction_matrix[i, j] = value
                interaction_matrix[j, i] = value  # 对称
        
        # 创建自定义颜色映射
        cmap = LinearSegmentedColormap.from_list('interaction', 
                                                  ['#FFFFFF', '#FFF7BC', '#FEC44F', '#D95F0E', '#993404'])
        
        # 绘制热力图
        im = ax.imshow(interaction_matrix, cmap=cmap, aspect='auto', vmin=0)
        
        # 标签
        ax.set_xticks(range(n_params))
        ax.set_yticks(range(n_params))
        ax.set_xticklabels(params, rotation=45, ha='right')
        ax.set_yticklabels(params)
        
        # 数值标注
        for i in range(n_params):
            for j in range(n_params):
                val = interaction_matrix[i, j]
                if val > 0.001:  # 只标注非零值
                    color = 'white' if val > 0.05 else 'black'
                    ax.text(j, i, f'{val:.3f}', ha='center', va='center', 
                           color=color, fontsize=8)
        
        ax.set_title(title)
        
        # 颜色条
        cbar = plt.colorbar(im, ax=ax, label='$S_{ij}$ (Second-order Index)')
        
        # 对角线标记
        for i in range(n_params):
            ax.add_patch(plt.Rectangle((i-0.5, i-0.5), 1, 1, fill=False, 
                                       edgecolor='gray', linestyle='--', linewidth=1))
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    
    def parameter_importance_ranking(self, importance_scores: Dict[str, float],
                                      uncertainty: Dict[str, float] = None,
                                      threshold: float = 0.1,
                                      title: str = 'Parameter Importance Ranking',
                                      save_path: str = None):
        """
        参数重要性排序图（带阈值线）- 新增功能
        
        Args:
            importance_scores: 重要性分数 {'param': score}
            uncertainty: 不确定性/误差 {'param': std}
            threshold: 重要性阈值线
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 排序
        sorted_params = sorted(importance_scores.keys(), 
                               key=lambda x: importance_scores[x], reverse=True)
        scores = [importance_scores[p] for p in sorted_params]
        
        # 确定颜色（超过阈值为高亮色）
        colors = [self.colors[0] if s >= threshold else self.colors[6] for s in scores]
        
        y_pos = np.arange(len(sorted_params))
        
        # 绘制柱状图
        if uncertainty:
            errors = [uncertainty.get(p, 0) for p in sorted_params]
            ax.barh(y_pos, scores, color=colors, alpha=0.8, xerr=errors, 
                   capsize=3, error_kw={'elinewidth': 1})
        else:
            ax.barh(y_pos, scores, color=colors, alpha=0.8)
        
        # 阈值线
        ax.axvline(threshold, color='red', linestyle='--', linewidth=1.5, 
                  label=f'Threshold = {threshold}')
        
        # 标签
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_params)
        ax.set_xlabel('Importance Score')
        ax.set_title(title)
        ax.legend(loc='lower right')
        
        # 添加分类标注
        important_count = sum(1 for s in scores if s >= threshold)
        ax.text(0.98, 0.98, f'Important: {important_count}/{len(scores)}', 
                transform=ax.transAxes, fontsize=10, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    
    def uncertainty_bands(self, x: np.ndarray, y_mean: np.ndarray,
                          y_percentiles: Dict[str, np.ndarray],
                          xlabel: str = 'Input',
                          ylabel: str = 'Output',
                          title: str = 'Prediction with Uncertainty Bands',
                          save_path: str = None):
        """
        多层不确定性带图（新增功能）
        
        Args:
            x: x轴数据
            y_mean: y均值
            y_percentiles: 不同置信水平的百分位数
                          {'50': (lower, upper), '90': (lower, upper), '99': (lower, upper)}
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 按置信水平绘制（从宽到窄）
        confidence_levels = sorted(y_percentiles.keys(), key=lambda x: int(x), reverse=True)
        alphas = np.linspace(0.1, 0.3, len(confidence_levels))
        
        for i, level in enumerate(confidence_levels):
            lower, upper = y_percentiles[level]
            ax.fill_between(x, lower, upper, color=self.colors[0], alpha=alphas[i],
                           label=f'{level}% CI')
        
        # 均值线
        ax.plot(x, y_mean, color=self.colors[0], linewidth=2, label='Mean')
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
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
        if total > 0:
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
                        param_ranges: Dict[str, Tuple[float, float]] = None,
                        title: str = 'Tornado Diagram',
                        xlabel: str = 'Output Value', save_path: str = None):
        """
        龙卷风图（单因素敏感性）增强版
        
        Args:
            params: 参数名列表
            low_values: 参数取低值时的输出
            high_values: 参数取高值时的输出
            baseline: 基准输出值
            param_ranges: 参数变化范围 {'param': (low, high)}
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
        
        # 添加参数变化范围标注
        if param_ranges:
            for i, param in enumerate(params_sorted):
                if param in param_ranges:
                    low_r, high_r = param_ranges[param]
                    ax.text(baseline, i + 0.35, f'[{low_r}, {high_r}]', 
                           fontsize=7, ha='center', alpha=0.7)
                       
        ax.axvline(baseline, color='black', linestyle='-', linewidth=1.5)
        ax.set_yticks(y)
        ax.set_yticklabels(params_sorted)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        
        # 图例
        legend_elements = [
            mpatches.Patch(facecolor=self.colors[0], label='Low Value'),
            mpatches.Patch(facecolor=self.colors[1], label='High Value')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        # 基准值标注
        ax.annotate(f'Baseline = {baseline:.2f}', xy=(baseline, len(params) - 0.5),
                   xytext=(baseline + (ax.get_xlim()[1] - baseline) * 0.1, len(params) - 0.5),
                   fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'))
        
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
        max_val = max(max(mu_values) if mu_values else 1, max(sigma_values) if sigma_values else 1)
        ax.plot([0, max_val], [0, max_val], '--', color='gray', alpha=0.5, 
               label='$\\sigma = \\mu^*$')
        ax.plot([0, max_val], [0, 0.5*max_val], ':', color='gray', alpha=0.5,
               label='$\\sigma = 0.5\\mu^*$')
        
        ax.set_xlabel('$\\mu^*$ (Mean of Absolute Elementary Effects)')
        ax.set_ylabel('$\\sigma$ (Standard Deviation)')
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
    
    def sensitivity_summary_dashboard(self, S1: Dict[str, float], ST: Dict[str, float],
                                       S2: Dict[Tuple[str, str], float] = None,
                                       title: str = 'Sensitivity Analysis Summary',
                                       save_path: str = None):
        """
        敏感性分析综合仪表板（新增功能）
        
        一图包含：一阶指数、总阶指数、交互效应热力图、重要性排序
        """
        fig = plt.figure(figsize=(14, 10))
        
        params = list(S1.keys())
        n_params = len(params)
        
        # 子图1: Sobol指数柱状图
        ax1 = fig.add_subplot(2, 2, 1)
        sorted_idx = np.argsort([ST[p] for p in params])[::-1]
        params_sorted = [params[i] for i in sorted_idx]
        s1_values = [S1[p] for p in params_sorted]
        st_values = [ST[p] for p in params_sorted]
        
        bar_width = 0.35
        x = np.arange(n_params)
        ax1.barh(x - bar_width/2, s1_values, bar_width, label='$S_1$', color=self.colors[0])
        ax1.barh(x + bar_width/2, st_values, bar_width, label='$S_T$', color=self.colors[1])
        ax1.set_yticks(x)
        ax1.set_yticklabels(params_sorted)
        ax1.set_xlabel('Sensitivity Index')
        ax1.set_title('First-order vs Total-order Indices')
        ax1.legend()
        
        # 子图2: 方差分解饼图
        ax2 = fig.add_subplot(2, 2, 2)
        sorted_items = sorted(S1.items(), key=lambda x: x[1], reverse=True)[:5]
        other = sum(v for _, v in list(S1.items())[5:])
        labels = [item[0] for item in sorted_items]
        sizes = [item[1] for item in sorted_items]
        if other > 0:
            labels.append('Other')
            sizes.append(other)
        total = sum(sizes)
        if total > 0:
            sizes = [s/total for s in sizes]
        ax2.pie(sizes, labels=labels, autopct='%1.1f%%', colors=self.colors[:len(sizes)])
        ax2.set_title('Variance Decomposition ($S_1$)')
        
        # 子图3: 交互效应（S_T - S_1）
        ax3 = fig.add_subplot(2, 2, 3)
        interaction = [ST[p] - S1[p] for p in params_sorted]
        colors_int = [self.colors[3] if i > 0.05 else self.colors[6] for i in interaction]
        ax3.barh(x, interaction, color=colors_int, alpha=0.8)
        ax3.set_yticks(x)
        ax3.set_yticklabels(params_sorted)
        ax3.set_xlabel('$S_T - S_1$ (Interaction Effect)')
        ax3.set_title('Interaction Effects')
        ax3.axvline(0.05, color='red', linestyle='--', alpha=0.5, label='Threshold=0.05')
        ax3.legend()
        
        # 子图4: 二阶交互热力图（如果有）
        ax4 = fig.add_subplot(2, 2, 4)
        if S2:
            interaction_matrix = np.zeros((n_params, n_params))
            for (p1, p2), value in S2.items():
                if p1 in params and p2 in params:
                    i, j = params.index(p1), params.index(p2)
                    interaction_matrix[i, j] = value
                    interaction_matrix[j, i] = value
            im = ax4.imshow(interaction_matrix, cmap='YlOrRd', aspect='auto')
            ax4.set_xticks(range(n_params))
            ax4.set_yticks(range(n_params))
            ax4.set_xticklabels(params, rotation=45, ha='right', fontsize=8)
            ax4.set_yticklabels(params, fontsize=8)
            ax4.set_title('Second-order Indices ($S_{ij}$)')
            plt.colorbar(im, ax=ax4)
        else:
            # 如果没有二阶指数，显示累积贡献图
            cumulative = np.cumsum(sorted([ST[p] for p in params], reverse=True))
            ax4.plot(range(1, n_params + 1), cumulative, 'o-', color=self.colors[0])
            ax4.axhline(0.9, color='red', linestyle='--', alpha=0.5, label='90% threshold')
            ax4.set_xlabel('Number of Parameters')
            ax4.set_ylabel('Cumulative $S_T$')
            ax4.set_title('Cumulative Variance Explained')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig


if __name__ == '__main__':
    print("Testing Enhanced Sensitivity Charts...")
    
    charts = SensitivityCharts()
    
    # Sobol指数
    S1 = {'param_a': 0.35, 'param_b': 0.25, 'param_c': 0.15, 'param_d': 0.10, 'param_e': 0.08}
    ST = {'param_a': 0.45, 'param_b': 0.32, 'param_c': 0.20, 'param_d': 0.15, 'param_e': 0.12}
    
    # 带置信区间的Sobol图
    S1_conf = {'param_a': (0.30, 0.40), 'param_b': (0.20, 0.30), 'param_c': (0.10, 0.20),
               'param_d': (0.05, 0.15), 'param_e': (0.03, 0.13)}
    ST_conf = {'param_a': (0.40, 0.50), 'param_b': (0.27, 0.37), 'param_c': (0.15, 0.25),
               'param_d': (0.10, 0.20), 'param_e': (0.07, 0.17)}
    charts.sobol_bar(S1, ST, S1_conf, ST_conf, title='Sobol Indices with Confidence Intervals')
    
    # 二阶Sobol热力图
    params = list(S1.keys())
    S2 = {('param_a', 'param_b'): 0.08, ('param_a', 'param_c'): 0.03,
          ('param_b', 'param_c'): 0.05, ('param_c', 'param_d'): 0.02}
    charts.sobol_interaction_heatmap(S2, params)
    
    # 参数重要性排序
    importance = {'param_a': 0.45, 'param_b': 0.32, 'param_c': 0.20, 
                  'param_d': 0.15, 'param_e': 0.08}
    charts.parameter_importance_ranking(importance, threshold=0.15)
    
    # 综合仪表板
    charts.sensitivity_summary_dashboard(S1, ST, S2)
    
    # 龙卷风图
    params_t = ['Price', 'Demand', 'Cost', 'Efficiency']
    low_values = [85, 92, 78, 95]
    high_values = [115, 108, 122, 105]
    baseline = 100
    charts.tornado_diagram(params_t, low_values, high_values, baseline)
    
    print("Enhanced Sensitivity Charts test completed!")
    plt.show()

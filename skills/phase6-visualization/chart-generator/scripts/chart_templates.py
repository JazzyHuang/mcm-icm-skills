"""
Chart Templates - 完整图表模板库
适用于MCM/ICM论文的专业图表

包含: 箱线图、小提琴图、散点图、平行坐标、热力图、气泡图等15+种图表
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import seaborn as sns

# 导入样式配置
try:
    from .nature_style import set_nature_style, NATURE_COLORS
except ImportError:
    NATURE_COLORS = ['#0077BB', '#33BBEE', '#009988', '#EE7733', '#CC3311', '#EE3377', '#BBBBBB']


class ChartTemplates:
    """专业图表模板集合"""
    
    def __init__(self, style: str = 'academic'):
        """
        Args:
            style: 'academic', 'nature', 'minimal'
        """
        self.style = style
        self.colors = NATURE_COLORS
        self._setup_style()
        
    def _setup_style(self):
        """设置全局样式"""
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
            'font.size': 10,
            'axes.linewidth': 0.8,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
        })
        
    # ============ 分布类图表 ============
    
    def boxplot(self, data: Dict[str, np.ndarray], xlabel: str = '', 
                ylabel: str = '', title: str = '', save_path: str = None):
        """箱线图"""
        fig, ax = plt.subplots(figsize=(8, 5))
        
        positions = range(len(data))
        bp = ax.boxplot(
            list(data.values()),
            positions=positions,
            widths=0.6,
            patch_artist=True,
            medianprops={'color': 'black', 'linewidth': 1.5}
        )
        
        # 设置颜色
        for patch, color in zip(bp['boxes'], self.colors[:len(data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            
        ax.set_xticks(positions)
        ax.set_xticklabels(list(data.keys()))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
        
    def violinplot(self, data: Dict[str, np.ndarray], xlabel: str = '',
                   ylabel: str = '', title: str = '', save_path: str = None):
        """小提琴图"""
        fig, ax = plt.subplots(figsize=(8, 5))
        
        positions = range(len(data))
        parts = ax.violinplot(
            list(data.values()),
            positions=positions,
            showmeans=True,
            showmedians=True
        )
        
        # 设置颜色
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(self.colors[i % len(self.colors)])
            pc.set_alpha(0.7)
            
        ax.set_xticks(positions)
        ax.set_xticklabels(list(data.keys()))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
        
    def histogram(self, data: np.ndarray, bins: int = 30, xlabel: str = '',
                  ylabel: str = 'Frequency', title: str = '', 
                  kde: bool = True, save_path: str = None):
        """直方图（带KDE）"""
        fig, ax = plt.subplots(figsize=(8, 5))
        
        ax.hist(data, bins=bins, density=True, alpha=0.7, 
                color=self.colors[0], edgecolor='white')
        
        if kde:
            from scipy import stats
            kde_x = np.linspace(data.min(), data.max(), 200)
            kde_y = stats.gaussian_kde(data)(kde_x)
            ax.plot(kde_x, kde_y, color=self.colors[1], linewidth=2, label='KDE')
            ax.legend()
            
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
        
    def beeswarm(self, data: Dict[str, np.ndarray], xlabel: str = '',
                 ylabel: str = '', title: str = '', save_path: str = None):
        """蜂群图（Beeswarm）"""
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # 转换为长格式
        all_data = []
        all_labels = []
        for label, values in data.items():
            all_data.extend(values)
            all_labels.extend([label] * len(values))
            
        df = pd.DataFrame({'value': all_data, 'group': all_labels})
        
        # 使用stripplot近似beeswarm
        sns.stripplot(data=df, x='group', y='value', ax=ax, 
                     palette=self.colors[:len(data)], alpha=0.7, jitter=0.3)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
        
    # ============ 关系类图表 ============
    
    def scatter(self, x: np.ndarray, y: np.ndarray, color: np.ndarray = None,
                size: np.ndarray = None, xlabel: str = '', ylabel: str = '',
                title: str = '', colorbar_label: str = '', save_path: str = None):
        """散点图"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        scatter = ax.scatter(
            x, y,
            c=color if color is not None else self.colors[0],
            s=size if size is not None else 50,
            alpha=0.7,
            cmap='viridis' if color is not None else None,
            edgecolors='white',
            linewidths=0.5
        )
        
        if color is not None:
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(colorbar_label)
            
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
        
    def bubble(self, x: np.ndarray, y: np.ndarray, size: np.ndarray,
               color: np.ndarray = None, labels: List[str] = None,
               xlabel: str = '', ylabel: str = '', title: str = '',
               save_path: str = None):
        """气泡图"""
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # 归一化大小
        size_normalized = (size - size.min()) / (size.max() - size.min()) * 1000 + 100
        
        scatter = ax.scatter(
            x, y,
            s=size_normalized,
            c=color if color is not None else self.colors[0],
            alpha=0.6,
            cmap='viridis' if color is not None else None,
            edgecolors='white',
            linewidths=1
        )
        
        # 添加标签
        if labels:
            for i, label in enumerate(labels):
                ax.annotate(label, (x[i], y[i]), fontsize=8, ha='center')
                
        if color is not None:
            plt.colorbar(scatter, ax=ax)
            
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
        
    def heatmap(self, data: np.ndarray, x_labels: List[str] = None,
                y_labels: List[str] = None, title: str = '',
                annotate: bool = True, cmap: str = 'viridis',
                save_path: str = None):
        """热力图"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(data, cmap=cmap, aspect='auto')
        
        # 添加数值标注
        if annotate and data.size < 200:
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    text = ax.text(j, i, f'{data[i, j]:.2f}',
                                  ha='center', va='center', fontsize=8)
                                  
        if x_labels:
            ax.set_xticks(range(len(x_labels)))
            ax.set_xticklabels(x_labels, rotation=45, ha='right')
        if y_labels:
            ax.set_yticks(range(len(y_labels)))
            ax.set_yticklabels(y_labels)
            
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
        
    def parallel_coordinates(self, data: pd.DataFrame, class_column: str,
                            title: str = '', save_path: str = None):
        """平行坐标图"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 标准化数据
        cols = [c for c in data.columns if c != class_column]
        data_norm = data.copy()
        for col in cols:
            data_norm[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())
            
        # 绘制
        classes = data[class_column].unique()
        for i, cls in enumerate(classes):
            class_data = data_norm[data_norm[class_column] == cls][cols]
            for _, row in class_data.iterrows():
                ax.plot(range(len(cols)), row.values, 
                       color=self.colors[i % len(self.colors)], alpha=0.3)
                       
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=45, ha='right')
        ax.set_ylabel('Normalized Value')
        ax.set_title(title)
        
        # 图例
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color=self.colors[i], label=cls)
                         for i, cls in enumerate(classes)]
        ax.legend(handles=legend_elements)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
        
    # ============ 趋势类图表 ============
    
    def line_with_confidence(self, x: np.ndarray, y: np.ndarray,
                             y_lower: np.ndarray, y_upper: np.ndarray,
                             xlabel: str = '', ylabel: str = '',
                             title: str = '', label: str = 'Model',
                             save_path: str = None):
        """带置信区间的折线图"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(x, y, color=self.colors[0], linewidth=2, label=label)
        ax.fill_between(x, y_lower, y_upper, color=self.colors[0], alpha=0.2)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
        
    def multi_line(self, x: np.ndarray, y_dict: Dict[str, np.ndarray],
                   xlabel: str = '', ylabel: str = '', title: str = '',
                   save_path: str = None):
        """多条折线图"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, (label, y) in enumerate(y_dict.items()):
            ax.plot(x, y, color=self.colors[i % len(self.colors)], 
                   linewidth=2, label=label, marker='o', markersize=4)
            
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
        
    # ============ 比较类图表 ============
    
    def grouped_bar(self, categories: List[str], values_dict: Dict[str, List],
                    xlabel: str = '', ylabel: str = '', title: str = '',
                    save_path: str = None):
        """分组柱状图"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        n_groups = len(categories)
        n_bars = len(values_dict)
        bar_width = 0.8 / n_bars
        
        for i, (label, values) in enumerate(values_dict.items()):
            x = np.arange(n_groups) + (i - n_bars/2 + 0.5) * bar_width
            ax.bar(x, values, bar_width, label=label, 
                  color=self.colors[i % len(self.colors)], alpha=0.8)
            
        ax.set_xticks(range(n_groups))
        ax.set_xticklabels(categories)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
        
    def stacked_bar(self, categories: List[str], values_dict: Dict[str, List],
                    xlabel: str = '', ylabel: str = '', title: str = '',
                    save_path: str = None):
        """堆叠柱状图"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bottom = np.zeros(len(categories))
        
        for i, (label, values) in enumerate(values_dict.items()):
            ax.bar(categories, values, bottom=bottom, label=label,
                  color=self.colors[i % len(self.colors)], alpha=0.8)
            bottom += np.array(values)
            
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
        
    def radar(self, categories: List[str], values_dict: Dict[str, List],
              title: str = '', save_path: str = None):
        """雷达图"""
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        for i, (label, values) in enumerate(values_dict.items()):
            values = list(values) + [values[0]]  # 闭合
            ax.plot(angles, values, 'o-', linewidth=2, 
                   color=self.colors[i % len(self.colors)], label=label)
            ax.fill(angles, values, alpha=0.25, 
                   color=self.colors[i % len(self.colors)])
            
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_title(title)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
        
    # ============ 高级图表 ============
    
    def sankey(self, sources: List[int], targets: List[int], 
               values: List[float], labels: List[str],
               title: str = '', save_path: str = None):
        """桑基图"""
        try:
            import plotly.graph_objects as go
            
            fig = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=labels,
                    color=self.colors[:len(labels)]
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values
                )
            )])
            
            fig.update_layout(title_text=title, font_size=10)
            
            if save_path:
                fig.write_image(save_path)
            return fig
            
        except ImportError:
            print("Plotly not available for Sankey diagram")
            return None
            
    def bump_chart(self, data: pd.DataFrame, time_col: str, 
                   rank_col: str, entity_col: str,
                   title: str = '', save_path: str = None):
        """凸点图（排名变化）"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        entities = data[entity_col].unique()
        times = sorted(data[time_col].unique())
        
        for i, entity in enumerate(entities):
            entity_data = data[data[entity_col] == entity].sort_values(time_col)
            ax.plot(entity_data[time_col], entity_data[rank_col], 
                   'o-', color=self.colors[i % len(self.colors)],
                   linewidth=2, markersize=8, label=entity)
            
        ax.invert_yaxis()  # 排名1在顶部
        ax.set_xlabel('Time')
        ax.set_ylabel('Rank')
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig


if __name__ == '__main__':
    print("Testing Chart Templates...")
    
    templates = ChartTemplates()
    
    # 测试箱线图
    data = {
        'Group A': np.random.randn(100),
        'Group B': np.random.randn(100) + 0.5,
        'Group C': np.random.randn(100) - 0.5
    }
    templates.boxplot(data, xlabel='Groups', ylabel='Value', title='Box Plot Example')
    
    # 测试散点图
    x = np.random.randn(100)
    y = x + np.random.randn(100) * 0.5
    templates.scatter(x, y, xlabel='X', ylabel='Y', title='Scatter Plot Example')
    
    print("Chart Templates test completed!")
    plt.show()

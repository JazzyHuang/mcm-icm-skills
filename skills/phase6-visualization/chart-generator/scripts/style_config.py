"""
图表样式配置
定义学术出版级别的图表样式
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# 学术论文默认配置
ACADEMIC_STYLE = {
    # 字体设置
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    
    # 图形设置
    'figure.figsize': (6, 4),
    'figure.dpi': 300,
    'figure.facecolor': 'white',
    'figure.edgecolor': 'white',
    
    # 轴设置
    'axes.linewidth': 0.8,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': False,
    
    # 刻度设置
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    
    # 图例设置
    'legend.fontsize': 9,
    'legend.frameon': False,
    'legend.borderpad': 0.4,
    
    # 线条设置
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    
    # 保存设置
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
}

# 色盲友好调色板
COLORBLIND_PALETTE = {
    'blue': '#0077BB',
    'cyan': '#33BBEE',
    'teal': '#009988',
    'orange': '#EE7733',
    'red': '#CC3311',
    'magenta': '#EE3377',
    'grey': '#BBBBBB',
}

# Viridis风格调色板
VIRIDIS_COLORS = [
    '#440154', '#482878', '#3E4A89', '#31688E',
    '#26828E', '#1F9E89', '#35B779', '#6DCD59',
    '#B4DE2C', '#FDE725'
]

# 美赛推荐配色
MCM_PALETTE = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'tertiary': '#2ca02c',
    'quaternary': '#d62728',
    'quinary': '#9467bd',
}


def set_academic_style():
    """应用学术论文样式"""
    plt.rcParams.update(ACADEMIC_STYLE)
    sns.set_palette("colorblind")


def get_color_palette(n_colors: int = 5, style: str = 'mcm'):
    """
    获取配色方案
    
    Args:
        n_colors: 需要的颜色数量
        style: 风格 ('mcm', 'viridis', 'colorblind')
    """
    if style == 'mcm':
        colors = list(MCM_PALETTE.values())[:n_colors]
    elif style == 'viridis':
        step = len(VIRIDIS_COLORS) // n_colors
        colors = VIRIDIS_COLORS[::step][:n_colors]
    else:
        colors = list(COLORBLIND_PALETTE.values())[:n_colors]
    return colors


def create_figure(
    nrows: int = 1, 
    ncols: int = 1, 
    figsize: tuple = None,
    style: str = 'academic'
) -> tuple:
    """
    创建配置好的图形
    
    Args:
        nrows: 子图行数
        ncols: 子图列数
        figsize: 图形大小 (width, height)
        style: 样式
        
    Returns:
        (fig, ax) 或 (fig, axes)
    """
    set_academic_style()
    
    if figsize is None:
        # 根据子图数量自动计算大小
        width = 3.5 * ncols
        height = 3 * nrows
        figsize = (width, height)
        
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    return fig, axes


def save_figure(fig, filename: str, formats: list = None):
    """
    保存图形为多种格式
    
    Args:
        fig: matplotlib Figure对象
        filename: 文件名（不含扩展名）
        formats: 格式列表，默认 ['pdf', 'png']
    """
    if formats is None:
        formats = ['pdf', 'png']
        
    for fmt in formats:
        fig.savefig(
            f'{filename}.{fmt}',
            format=fmt,
            dpi=300 if fmt == 'png' else None,
            bbox_inches='tight',
            pad_inches=0.1
        )


# 常用图表模板
class ChartTemplates:
    """常用图表模板"""
    
    @staticmethod
    def line_with_confidence(ax, x, y, y_lower, y_upper, 
                            label=None, color=None):
        """带置信区间的折线图"""
        if color is None:
            color = MCM_PALETTE['primary']
            
        ax.plot(x, y, color=color, linewidth=1.5, label=label)
        ax.fill_between(x, y_lower, y_upper, color=color, alpha=0.2)
        
    @staticmethod
    def bar_comparison(ax, categories, values_list, labels, 
                      colors=None):
        """分组柱状图比较"""
        if colors is None:
            colors = get_color_palette(len(values_list))
            
        n_groups = len(categories)
        n_bars = len(values_list)
        bar_width = 0.8 / n_bars
        
        for i, (values, label, color) in enumerate(zip(values_list, labels, colors)):
            x = range(n_groups)
            x_offset = [(xi + (i - n_bars/2 + 0.5) * bar_width) for xi in x]
            ax.bar(x_offset, values, bar_width, label=label, color=color)
            
        ax.set_xticks(range(n_groups))
        ax.set_xticklabels(categories)
        ax.legend()
        
    @staticmethod
    def heatmap(ax, data, x_labels=None, y_labels=None, 
               cmap='viridis', annotate=True):
        """热力图"""
        im = ax.imshow(data, cmap=cmap, aspect='auto')
        
        if annotate:
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    ax.text(j, i, f'{data[i,j]:.2f}',
                           ha='center', va='center', fontsize=8)
                           
        if x_labels:
            ax.set_xticks(range(len(x_labels)))
            ax.set_xticklabels(x_labels)
        if y_labels:
            ax.set_yticks(range(len(y_labels)))
            ax.set_yticklabels(y_labels)
            
        return im


if __name__ == '__main__':
    # 测试代码
    import numpy as np
    
    set_academic_style()
    
    # 测试折线图
    fig, ax = create_figure()
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    y_lower = y - 0.2
    y_upper = y + 0.2
    
    ChartTemplates.line_with_confidence(ax, x, y, y_lower, y_upper, 
                                       label='Model Prediction')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.legend()
    
    save_figure(fig, 'test_figure')
    print("Test figure saved")

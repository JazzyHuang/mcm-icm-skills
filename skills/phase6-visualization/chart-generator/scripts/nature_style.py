"""
Nature/Science Journal Style Configuration
Nature/Science期刊级别图表样式配置

遵循Nature Research Figure Guide 2025规范
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from typing import Dict, List, Optional


# ============ Nature 2025 样式配置 ============

NATURE_STYLE = {
    # 字体设置
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7,  # Nature: 5-7pt
    
    # 图形设置
    'figure.figsize': (3.5, 2.5),  # 单栏宽度
    'figure.dpi': 300,
    'figure.facecolor': 'white',
    'figure.edgecolor': 'white',
    
    # 轴设置
    'axes.linewidth': 0.5,
    'axes.labelsize': 7,
    'axes.titlesize': 8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': False,  # Nature不推荐背景网格
    'axes.axisbelow': True,
    
    # 刻度设置
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    
    # 图例设置
    'legend.fontsize': 6,
    'legend.frameon': False,
    'legend.borderpad': 0.4,
    
    # 线条设置
    'lines.linewidth': 0.75,
    'lines.markersize': 4,
    
    # 保存设置
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'savefig.transparent': False,
    
    # PDF字体嵌入
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
}


# ============ 色盲友好调色板 ============

# Wong (2011) Color Palette - Nature Methods推荐
WONG_COLORBLIND = [
    '#000000',  # Black
    '#E69F00',  # Orange
    '#56B4E9',  # Sky Blue
    '#009E73',  # Bluish Green
    '#F0E442',  # Yellow
    '#0072B2',  # Blue
    '#D55E00',  # Vermillion
    '#CC79A7',  # Reddish Purple
]

# Paul Tol's Bright Palette
TOL_BRIGHT = [
    '#4477AA',  # Blue
    '#EE6677',  # Red
    '#228833',  # Green
    '#CCBB44',  # Yellow
    '#66CCEE',  # Cyan
    '#AA3377',  # Purple
    '#BBBBBB',  # Grey
]

# Viridis-derived discrete colors
VIRIDIS_DISCRETE = [
    '#440154',  # Dark purple
    '#3B528B',  # Blue-purple
    '#21918C',  # Teal
    '#5DC863',  # Green
    '#FDE725',  # Yellow
]

# Nature推荐的MCM配色
NATURE_COLORS = WONG_COLORBLIND[1:]  # 不包含黑色作为数据颜色


# ============ 图形尺寸规范 ============

FIGURE_SIZES = {
    'single_column': (3.5, 2.5),    # 单栏 ~89mm
    'double_column': (7.0, 4.0),    # 双栏 ~183mm
    'half_page': (7.0, 5.0),        # 半页
    'full_page': (7.0, 9.0),        # 整页
    'square_small': (3.5, 3.5),     # 小正方形
    'square_large': (5.0, 5.0),     # 大正方形
}


# ============ 函数定义 ============

def set_nature_style():
    """应用Nature样式"""
    plt.rcParams.update(NATURE_STYLE)
    

def get_figure_size(size_type: str = 'single_column') -> tuple:
    """获取标准图形尺寸"""
    return FIGURE_SIZES.get(size_type, FIGURE_SIZES['single_column'])


def get_color_palette(n_colors: int = 5, palette: str = 'wong') -> List[str]:
    """
    获取色盲友好调色板
    
    Args:
        n_colors: 需要的颜色数量
        palette: 'wong', 'tol', 'viridis'
    """
    palettes = {
        'wong': WONG_COLORBLIND,
        'tol': TOL_BRIGHT,
        'viridis': VIRIDIS_DISCRETE,
        'nature': NATURE_COLORS,
    }
    
    colors = palettes.get(palette, WONG_COLORBLIND)
    return colors[:n_colors]


def create_figure(nrows: int = 1, ncols: int = 1, 
                  size_type: str = 'single_column',
                  figsize: tuple = None) -> tuple:
    """
    创建Nature规范的图形
    
    Args:
        nrows: 子图行数
        ncols: 子图列数
        size_type: 预定义尺寸类型
        figsize: 自定义尺寸 (width, height)
    """
    set_nature_style()
    
    if figsize is None:
        base_size = get_figure_size(size_type)
        figsize = (base_size[0] * ncols, base_size[1] * nrows)
        
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    return fig, axes


def save_figure(fig, filename: str, formats: List[str] = None):
    """
    保存图形（PDF优先）
    
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
            dpi=300,
            bbox_inches='tight',
            pad_inches=0.05,
            facecolor='white',
            edgecolor='none'
        )


def add_panel_label(ax, label: str, x: float = -0.15, y: float = 1.05,
                    fontsize: int = 10, fontweight: str = 'bold'):
    """
    添加子图标签 (a), (b), (c) 等
    
    Args:
        ax: matplotlib Axes对象
        label: 标签文本 ('a', 'b', 等)
        x, y: 标签位置（相对于轴）
    """
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight=fontweight,
            va='top', ha='left')


def style_axis(ax, xlabel: str = None, ylabel: str = None,
               title: str = None, legend: bool = False):
    """
    统一设置轴样式
    
    Args:
        ax: matplotlib Axes对象
        xlabel, ylabel: 轴标签
        title: 标题
        legend: 是否显示图例
    """
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
        
    # 移除上方和右侧边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if legend:
        ax.legend(frameon=False)


# ============ WCAG 无障碍检查 ============

def check_color_contrast(color1: str, color2: str) -> float:
    """
    检查两个颜色的对比度（WCAG 2.1 AA标准需要≥4.5）
    
    Args:
        color1, color2: 十六进制颜色值
        
    Returns:
        对比度比值
    """
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
    def relative_luminance(rgb):
        r, g, b = [x/255 for x in rgb]
        r = r/12.92 if r <= 0.03928 else ((r+0.055)/1.055)**2.4
        g = g/12.92 if g <= 0.03928 else ((g+0.055)/1.055)**2.4
        b = b/12.92 if b <= 0.03928 else ((b+0.055)/1.055)**2.4
        return 0.2126*r + 0.7152*g + 0.0722*b
        
    l1 = relative_luminance(hex_to_rgb(color1))
    l2 = relative_luminance(hex_to_rgb(color2))
    
    lighter = max(l1, l2)
    darker = min(l1, l2)
    
    return (lighter + 0.05) / (darker + 0.05)


def validate_accessibility(colors: List[str], background: str = '#FFFFFF') -> Dict:
    """
    验证颜色组合的无障碍性
    
    Args:
        colors: 要验证的颜色列表
        background: 背景色
        
    Returns:
        验证结果字典
    """
    results = {
        'colors': colors,
        'background': background,
        'contrasts': {},
        'wcag_aa_pass': [],
        'wcag_aaa_pass': []
    }
    
    for color in colors:
        contrast = check_color_contrast(color, background)
        results['contrasts'][color] = contrast
        
        if contrast >= 4.5:
            results['wcag_aa_pass'].append(color)
        if contrast >= 7.0:
            results['wcag_aaa_pass'].append(color)
            
    results['all_accessible'] = len(results['wcag_aa_pass']) == len(colors)
    
    return results


# ============ 色盲模拟 ============

def simulate_colorblind(color: str, type: str = 'deuteranopia') -> str:
    """
    模拟色盲看到的颜色
    
    Args:
        color: 原始颜色（十六进制）
        type: 'deuteranopia', 'protanopia', 'tritanopia'
        
    Returns:
        模拟后的颜色
    """
    # 简化实现 - 实际应用建议使用colorspacious库
    hex_color = color.lstrip('#')
    r, g, b = [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]
    
    if type == 'deuteranopia':  # 红绿色盲
        # 简化转换矩阵
        r_new = int(0.625 * r + 0.375 * g)
        g_new = int(0.7 * r + 0.3 * g)
        b_new = b
    elif type == 'protanopia':  # 红色盲
        r_new = int(0.567 * r + 0.433 * g)
        g_new = int(0.558 * r + 0.442 * g)
        b_new = b
    else:  # tritanopia - 蓝色盲
        r_new = r
        g_new = int(0.95 * g + 0.05 * b)
        b_new = int(0.433 * g + 0.567 * b)
        
    r_new = min(255, max(0, r_new))
    g_new = min(255, max(0, g_new))
    b_new = min(255, max(0, b_new))
    
    return f'#{r_new:02x}{g_new:02x}{b_new:02x}'


if __name__ == '__main__':
    print("Testing Nature Style Configuration...")
    
    # 应用样式
    set_nature_style()
    
    # 创建示例图
    fig, ax = create_figure(size_type='single_column')
    
    colors = get_color_palette(5, 'wong')
    x = np.linspace(0, 10, 100)
    
    for i, color in enumerate(colors):
        ax.plot(x, np.sin(x + i), color=color, label=f'Series {i+1}')
        
    style_axis(ax, xlabel='Time (s)', ylabel='Amplitude', 
               title='Nature Style Example', legend=True)
    add_panel_label(ax, 'a')
    
    # 检查无障碍性
    accessibility = validate_accessibility(colors)
    print(f"All colors accessible: {accessibility['all_accessible']}")
    print(f"Contrasts: {accessibility['contrasts']}")
    
    plt.tight_layout()
    print("Nature Style test completed!")
    plt.show()

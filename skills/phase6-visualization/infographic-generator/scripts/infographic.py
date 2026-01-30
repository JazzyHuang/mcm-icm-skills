"""
Scientific Infographic Generator
科学信息图生成器

遵循Nature 2025设计原则，自动生成专业科学信息图。
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import json


# ============ 样式配置 ============

INFOGRAPHIC_STYLES = {
    'nature': {
        'font_family': 'Arial',
        'title_size': 18,
        'subtitle_size': 14,
        'body_size': 10,
        'caption_size': 8,
        'colors': {
            'primary': '#0077BB',
            'secondary': '#33BBEE',
            'accent': '#EE7733',
            'success': '#009988',
            'warning': '#CC3311',
            'neutral': '#BBBBBB',
            'background': '#FFFFFF',
            'text': '#333333'
        },
        'margin': 0.05,
        'spacing': 0.02
    },
    'minimal': {
        'font_family': 'Helvetica',
        'title_size': 16,
        'subtitle_size': 12,
        'body_size': 9,
        'caption_size': 7,
        'colors': {
            'primary': '#2C3E50',
            'secondary': '#7F8C8D',
            'accent': '#E74C3C',
            'success': '#27AE60',
            'warning': '#F39C12',
            'neutral': '#BDC3C7',
            'background': '#FFFFFF',
            'text': '#2C3E50'
        },
        'margin': 0.04,
        'spacing': 0.015
    }
}

ICONS = {
    'up': '↑',
    'down': '↓',
    'speed': '⚡',
    'money': '$',
    'check': '✓',
    'warning': '⚠',
    'star': '★',
    'arrow': '→',
    'bullet': '•'
}


class InfographicGenerator:
    """科学信息图生成器"""
    
    def __init__(self, style: str = 'nature'):
        """
        Args:
            style: 'nature' or 'minimal'
        """
        self.style = INFOGRAPHIC_STYLES.get(style, INFOGRAPHIC_STYLES['nature'])
        self.colors = self.style['colors']
        
    def create_summary_infographic(
        self,
        title: str,
        key_findings: List[Dict],
        main_figure: plt.Figure = None,
        methods: List[str] = None,
        conclusion: str = None,
        figsize: Tuple[float, float] = (12, 8),
        save_path: str = None
    ) -> plt.Figure:
        """
        创建论文摘要信息图
        
        Args:
            title: 标题
            key_findings: [{"text": "...", "value": "85%", "icon": "up"}, ...]
            main_figure: 主图（可选）
            methods: 方法步骤列表
            conclusion: 结论
        """
        fig = plt.figure(figsize=figsize, facecolor=self.colors['background'])
        
        # 使用GridSpec布局
        gs = GridSpec(3, 3, figure=fig, hspace=0.15, wspace=0.1,
                     left=0.05, right=0.95, top=0.92, bottom=0.05)
        
        # 1. 标题区域（顶部）
        ax_title = fig.add_subplot(gs[0, :])
        self._draw_title(ax_title, title)
        
        # 2. 关键发现（左侧）
        ax_findings = fig.add_subplot(gs[1, 0])
        self._draw_key_findings(ax_findings, key_findings)
        
        # 3. 主图区域（中间）
        ax_main = fig.add_subplot(gs[1, 1:])
        if main_figure:
            self._embed_figure(ax_main, main_figure)
        else:
            self._draw_placeholder(ax_main, "Main Visualization")
            
        # 4. 方法和结论（底部）
        ax_bottom = fig.add_subplot(gs[2, :])
        self._draw_footer(ax_bottom, methods, conclusion)
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor=self.colors['background'])
        return fig
        
    def _draw_title(self, ax, title: str):
        """绘制标题区域"""
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # 标题背景
        rect = patches.FancyBboxPatch(
            (0, 0), 1, 1, boxstyle="round,pad=0.02",
            facecolor=self.colors['primary'], alpha=0.1
        )
        ax.add_patch(rect)
        
        # 标题文本
        ax.text(0.5, 0.5, title,
               fontsize=self.style['title_size'],
               fontweight='bold',
               color=self.colors['text'],
               ha='center', va='center',
               fontfamily=self.style['font_family'])
               
    def _draw_key_findings(self, ax, findings: List[Dict]):
        """绘制关键发现"""
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        n = len(findings)
        if n == 0:
            return
            
        height = 0.8 / n
        
        for i, finding in enumerate(findings):
            y = 0.9 - i * height - height/2
            
            # 图标/数值
            icon = ICONS.get(finding.get('icon', 'bullet'), '•')
            value = finding.get('value', '')
            text = finding.get('text', '')
            
            # 背景框
            rect = patches.FancyBboxPatch(
                (0.05, y - height/2 + 0.02), 0.9, height - 0.04,
                boxstyle="round,pad=0.02",
                facecolor=self.colors['secondary'],
                alpha=0.2
            )
            ax.add_patch(rect)
            
            # 数值（大字）
            ax.text(0.15, y + 0.05, value,
                   fontsize=self.style['subtitle_size'],
                   fontweight='bold',
                   color=self.colors['primary'],
                   ha='left', va='center')
                   
            # 说明文字
            ax.text(0.15, y - 0.08, text,
                   fontsize=self.style['caption_size'],
                   color=self.colors['text'],
                   ha='left', va='center')
                   
            # 图标
            ax.text(0.08, y, icon,
                   fontsize=self.style['body_size'],
                   color=self.colors['accent'],
                   ha='center', va='center')
                   
    def _draw_placeholder(self, ax, text: str):
        """绘制占位符"""
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        rect = patches.FancyBboxPatch(
            (0.05, 0.05), 0.9, 0.9,
            boxstyle="round,pad=0.02",
            facecolor=self.colors['neutral'],
            alpha=0.3,
            linestyle='--',
            edgecolor=self.colors['text']
        )
        ax.add_patch(rect)
        
        ax.text(0.5, 0.5, f"[{text}]",
               fontsize=self.style['body_size'],
               color=self.colors['text'],
               ha='center', va='center',
               style='italic')
               
    def _embed_figure(self, ax, fig_to_embed):
        """嵌入外部图形（简化实现）"""
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        ax.text(0.5, 0.5, "[Figure Embedded Here]",
               fontsize=self.style['body_size'],
               color=self.colors['text'],
               ha='center', va='center')
               
    def _draw_footer(self, ax, methods: List[str], conclusion: str):
        """绘制底部（方法和结论）"""
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # 方法流程
        if methods:
            n = len(methods)
            step_width = 0.8 / n
            
            for i, method in enumerate(methods):
                x = 0.1 + i * step_width + step_width/2
                
                # 圆圈
                circle = plt.Circle((x, 0.7), 0.06, 
                                   color=self.colors['primary'], alpha=0.8)
                ax.add_patch(circle)
                
                # 步骤编号
                ax.text(x, 0.7, str(i+1),
                       fontsize=self.style['body_size'],
                       color='white',
                       ha='center', va='center',
                       fontweight='bold')
                       
                # 方法名
                ax.text(x, 0.5, method,
                       fontsize=self.style['caption_size'],
                       color=self.colors['text'],
                       ha='center', va='center')
                       
                # 连接箭头
                if i < n - 1:
                    ax.annotate('', xy=(x + step_width*0.6, 0.7),
                               xytext=(x + 0.08, 0.7),
                               arrowprops=dict(arrowstyle='->',
                                             color=self.colors['neutral']))
                                             
        # 结论
        if conclusion:
            ax.text(0.5, 0.15, f"Conclusion: {conclusion}",
                   fontsize=self.style['body_size'],
                   color=self.colors['text'],
                   ha='center', va='center',
                   style='italic',
                   bbox=dict(boxstyle='round', facecolor=self.colors['success'],
                            alpha=0.1))
                            
    def create_methods_flow(
        self,
        steps: List[Dict],
        connections: List[Tuple[int, int]] = None,
        figsize: Tuple[float, float] = (14, 4),
        save_path: str = None
    ) -> plt.Figure:
        """
        创建方法流程图
        
        Args:
            steps: [{"name": "...", "description": "..."}, ...]
            connections: [(from_idx, to_idx), ...]
        """
        fig, ax = plt.subplots(figsize=figsize, facecolor=self.colors['background'])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        n = len(steps)
        if n == 0:
            return fig
            
        box_width = 0.8 / n
        box_height = 0.4
        
        positions = {}
        
        for i, step in enumerate(steps):
            x = 0.1 + i * box_width + box_width/2 - 0.08
            y = 0.5 - box_height/2
            
            positions[i] = (x + 0.08, y + box_height/2)
            
            # 步骤框
            rect = patches.FancyBboxPatch(
                (x, y), 0.16, box_height,
                boxstyle="round,pad=0.02",
                facecolor=self.colors['primary'],
                alpha=0.8
            )
            ax.add_patch(rect)
            
            # 步骤名
            ax.text(x + 0.08, y + box_height*0.7,
                   step.get('name', f'Step {i+1}'),
                   fontsize=self.style['body_size'],
                   fontweight='bold',
                   color='white',
                   ha='center', va='center')
                   
            # 描述
            ax.text(x + 0.08, y + box_height*0.3,
                   step.get('description', ''),
                   fontsize=self.style['caption_size'],
                   color='white',
                   ha='center', va='center',
                   alpha=0.9)
                   
        # 绘制连接
        if connections is None:
            connections = [(i, i+1) for i in range(n-1)]
            
        for src, tgt in connections:
            if src in positions and tgt in positions:
                x1, y1 = positions[src]
                x2, y2 = positions[tgt]
                
                ax.annotate('', xy=(x2 - 0.1, y2),
                           xytext=(x1 + 0.1, y1),
                           arrowprops=dict(arrowstyle='->',
                                         color=self.colors['accent'],
                                         lw=2))
                                         
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
        
    def create_comparison(
        self,
        scenarios: List[str],
        metrics: Dict[str, List[float]],
        highlight: str = None,
        figsize: Tuple[float, float] = (10, 6),
        save_path: str = None
    ) -> plt.Figure:
        """
        创建方案对比图
        
        Args:
            scenarios: 方案名列表
            metrics: {"指标名": [方案1值, 方案2值, ...], ...}
            highlight: 要高亮的方案名
        """
        fig, ax = plt.subplots(figsize=figsize, facecolor=self.colors['background'])
        
        n_scenarios = len(scenarios)
        n_metrics = len(metrics)
        
        x = np.arange(n_metrics)
        bar_width = 0.8 / n_scenarios
        
        for i, scenario in enumerate(scenarios):
            values = [metrics[m][i] for m in metrics.keys()]
            offset = (i - n_scenarios/2 + 0.5) * bar_width
            
            color = self.colors['accent'] if scenario == highlight else self.colors['primary']
            alpha = 1.0 if scenario == highlight else 0.7
            
            bars = ax.bar(x + offset, values, bar_width,
                         label=scenario, color=color, alpha=alpha)
                         
            # 添加数值标签
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{val:.0f}', ha='center', va='bottom',
                       fontsize=self.style['caption_size'])
                       
        ax.set_xticks(x)
        ax.set_xticklabels(metrics.keys(), fontsize=self.style['body_size'])
        ax.set_ylabel('Value', fontsize=self.style['body_size'])
        ax.legend(loc='upper right', frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
        
    def create_highlight_stats(
        self,
        stats: List[Dict],
        figsize: Tuple[float, float] = (12, 3),
        save_path: str = None
    ) -> plt.Figure:
        """
        创建关键统计数据展示
        
        Args:
            stats: [{"label": "...", "value": "...", "description": "..."}, ...]
        """
        fig, ax = plt.subplots(figsize=figsize, facecolor=self.colors['background'])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        n = len(stats)
        if n == 0:
            return fig
            
        width = 0.9 / n
        
        for i, stat in enumerate(stats):
            x = 0.05 + i * width + width/2
            
            # 数值（大字）
            ax.text(x, 0.65, stat.get('value', ''),
                   fontsize=self.style['title_size'] * 1.5,
                   fontweight='bold',
                   color=self.colors['primary'],
                   ha='center', va='center')
                   
            # 标签
            ax.text(x, 0.35, stat.get('label', ''),
                   fontsize=self.style['body_size'],
                   fontweight='bold',
                   color=self.colors['text'],
                   ha='center', va='center')
                   
            # 描述
            ax.text(x, 0.15, stat.get('description', ''),
                   fontsize=self.style['caption_size'],
                   color=self.colors['neutral'],
                   ha='center', va='center')
                   
            # 分隔线
            if i < n - 1:
                ax.axvline(x + width/2, color=self.colors['neutral'], 
                          alpha=0.3, linestyle='--')
                          
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig


if __name__ == '__main__':
    print("Testing Infographic Generator...")
    
    generator = InfographicGenerator(style='nature')
    
    # 测试摘要信息图
    key_findings = [
        {"text": "模型精度提升", "value": "95%", "icon": "up"},
        {"text": "计算速度提升", "value": "1000x", "icon": "speed"},
        {"text": "成本节约", "value": "$2.5M", "icon": "money"}
    ]
    
    methods = ["数据收集", "特征工程", "模型训练", "验证分析"]
    conclusion = "创新方法显著优于传统方案"
    
    fig = generator.create_summary_infographic(
        title="基于深度学习的优化方法研究",
        key_findings=key_findings,
        methods=methods,
        conclusion=conclusion
    )
    
    # 测试方法流程图
    steps = [
        {"name": "数据预处理", "description": "清洗、标准化"},
        {"name": "特征工程", "description": "PCA、选择"},
        {"name": "PINN建模", "description": "物理约束"},
        {"name": "验证分析", "description": "交叉验证"}
    ]
    
    fig2 = generator.create_methods_flow(steps)
    
    # 测试对比图
    scenarios = ["基准方案", "方案A", "方案B"]
    metrics = {
        "成本": [100, 85, 78],
        "效率": [70, 88, 92],
        "风险": [50, 35, 40]
    }
    
    fig3 = generator.create_comparison(scenarios, metrics, highlight="方案B")
    
    # 测试统计展示
    stats = [
        {"label": "数据样本", "value": "10,000+", "description": "多源数据融合"},
        {"label": "预测精度", "value": "99.2%", "description": "行业领先水平"},
        {"label": "处理速度", "value": "0.1s", "description": "实时响应"}
    ]
    
    fig4 = generator.create_highlight_stats(stats)
    
    print("Infographic Generator test completed!")
    plt.show()

"""
Advanced Charts - 高级图表模块
Sankey图、Circle Packing、Bump Chart、Beeswarm等

2025年前沿可视化技术，适用于MCM/ICM高级论文。
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

NATURE_COLORS = ['#0077BB', '#33BBEE', '#009988', '#EE7733', '#CC3311', '#EE3377', '#BBBBBB']


class AdvancedCharts:
    """高级图表集合"""
    
    def __init__(self, color_palette: List[str] = None):
        self.colors = color_palette or NATURE_COLORS
        
    # ============ Sankey图 ============
    
    def sankey_diagram(self, sources: List[int], targets: List[int],
                       values: List[float], labels: List[str],
                       title: str = 'Sankey Diagram',
                       node_colors: List[str] = None,
                       save_path: str = None):
        """
        桑基图 - 流量可视化
        
        Args:
            sources: 源节点索引列表
            targets: 目标节点索引列表
            values: 流量值列表
            labels: 节点标签列表
        """
        if not PLOTLY_AVAILABLE:
            return self._sankey_matplotlib(sources, targets, values, labels, title, save_path)
            
        node_colors = node_colors or self.colors[:len(labels)]
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                color=node_colors
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=[f'rgba{tuple(list(int(node_colors[s].lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + [0.4])}'
                       for s in sources]
            )
        )])
        
        fig.update_layout(
            title_text=title,
            font_size=12,
            width=900,
            height=600
        )
        
        if save_path:
            fig.write_image(save_path)
        return fig
        
    def _sankey_matplotlib(self, sources, targets, values, labels, title, save_path):
        """Matplotlib版本的Sankey（简化）"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 简化实现：使用箭头表示流量
        n_nodes = len(labels)
        
        # 计算节点位置
        levels = {}
        for i, label in enumerate(labels):
            if i in sources and i not in targets:
                levels[i] = 0
            elif i in targets and i not in sources:
                levels[i] = 2
            else:
                levels[i] = 1
                
        # 按层分组
        layer_nodes = {0: [], 1: [], 2: []}
        for node, level in levels.items():
            layer_nodes[level].append(node)
            
        # 计算位置
        positions = {}
        for level, nodes in layer_nodes.items():
            for i, node in enumerate(nodes):
                y = (i + 0.5) / max(len(nodes), 1)
                positions[node] = (level * 0.4 + 0.1, y)
                
        # 绘制节点
        for node, (x, y) in positions.items():
            ax.add_patch(plt.Rectangle((x-0.02, y-0.03), 0.04, 0.06,
                        color=self.colors[node % len(self.colors)], alpha=0.8))
            ax.text(x, y, labels[node], ha='center', va='center', fontsize=9)
            
        # 绘制流量
        max_val = max(values)
        for s, t, v in zip(sources, targets, values):
            sx, sy = positions[s]
            tx, ty = positions[t]
            width = (v / max_val) * 0.03 + 0.005
            ax.annotate('', xy=(tx-0.02, ty), xytext=(sx+0.02, sy),
                       arrowprops=dict(arrowstyle='->', color='gray',
                                      lw=width*50, alpha=0.5))
                                      
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
        
    # ============ Circle Packing ============
    
    def circle_packing(self, data: Dict, title: str = 'Circle Packing',
                       save_path: str = None):
        """
        圆形堆叠图 - 层级数据可视化
        
        Args:
            data: 层级数据 {'name': 'root', 'children': [{'name': 'A', 'value': 10}, ...]}
        """
        if PLOTLY_AVAILABLE:
            return self._circle_packing_plotly(data, title, save_path)
        return self._circle_packing_matplotlib(data, title, save_path)
        
    def _circle_packing_plotly(self, data, title, save_path):
        """Plotly版本Circle Packing（使用Treemap近似）"""
        # 展平层级数据
        ids, labels, parents, values = [], [], [], []
        
        def flatten(node, parent=''):
            node_id = node.get('name', 'root')
            ids.append(node_id)
            labels.append(node_id)
            parents.append(parent)
            
            if 'children' in node:
                values.append(sum(c.get('value', 0) for c in node['children']))
                for child in node['children']:
                    flatten(child, node_id)
            else:
                values.append(node.get('value', 1))
                
        flatten(data)
        
        fig = go.Figure(go.Treemap(
            ids=ids,
            labels=labels,
            parents=parents,
            values=values,
            branchvalues='total',
            marker=dict(
                colors=self.colors[:len(ids)],
                line=dict(width=2)
            )
        ))
        
        fig.update_layout(title=title, width=700, height=700)
        
        if save_path:
            fig.write_image(save_path)
        return fig
        
    def _circle_packing_matplotlib(self, data, title, save_path):
        """Matplotlib版本Circle Packing"""
        import matplotlib.patches as patches
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 简化：只处理一层
        if 'children' in data:
            children = data['children']
            n = len(children)
            
            # 计算圆的位置（简单的圆形布局）
            total_value = sum(c.get('value', 1) for c in children)
            
            angles = np.linspace(0, 2*np.pi, n, endpoint=False)
            
            for i, (child, angle) in enumerate(zip(children, angles)):
                value = child.get('value', 1)
                radius = np.sqrt(value / total_value) * 0.3
                
                x = 0.5 + 0.3 * np.cos(angle)
                y = 0.5 + 0.3 * np.sin(angle)
                
                circle = plt.Circle((x, y), radius, 
                                    color=self.colors[i % len(self.colors)],
                                    alpha=0.7)
                ax.add_patch(circle)
                ax.text(x, y, child.get('name', ''), ha='center', va='center', fontsize=9)
                
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_title(title)
        ax.axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
        
    # ============ Bump Chart ============
    
    def bump_chart(self, data: pd.DataFrame, time_col: str,
                   rank_col: str, entity_col: str,
                   title: str = 'Bump Chart',
                   save_path: str = None):
        """
        凸点图 - 排名变化追踪
        
        Args:
            data: DataFrame with time, rank, and entity columns
            time_col: 时间列名
            rank_col: 排名列名
            entity_col: 实体列名
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        entities = data[entity_col].unique()
        times = sorted(data[time_col].unique())
        
        for i, entity in enumerate(entities):
            entity_data = data[data[entity_col] == entity].sort_values(time_col)
            
            x = entity_data[time_col].values
            y = entity_data[rank_col].values
            
            ax.plot(x, y, 'o-', color=self.colors[i % len(self.colors)],
                   linewidth=2.5, markersize=10, label=entity)
            
            # 在起点和终点标注实体名
            ax.annotate(entity, (x[0], y[0]), xytext=(-10, 0),
                       textcoords='offset points', ha='right', va='center', fontsize=9)
            ax.annotate(entity, (x[-1], y[-1]), xytext=(10, 0),
                       textcoords='offset points', ha='left', va='center', fontsize=9)
                       
        ax.invert_yaxis()  # 排名1在顶部
        ax.set_xticks(times)
        ax.set_xlabel('Time Period')
        ax.set_ylabel('Rank')
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 隐藏图例（因为已经在图上标注）
        # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
        
    # ============ Beeswarm Plot ============
    
    def beeswarm_plot(self, data: pd.DataFrame, category_col: str,
                      value_col: str, title: str = 'Beeswarm Plot',
                      save_path: str = None):
        """
        蜂群图 - 展示分布（比箱线图信息更丰富）
        
        Args:
            data: DataFrame
            category_col: 分类列名
            value_col: 数值列名
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        categories = data[category_col].unique()
        
        for i, cat in enumerate(categories):
            values = data[data[category_col] == cat][value_col].values
            
            # 计算水平偏移以避免重叠
            n = len(values)
            jitter = np.zeros(n)
            
            # 简单的避免重叠算法
            sorted_idx = np.argsort(values)
            values_sorted = values[sorted_idx]
            
            for j in range(1, n):
                if abs(values_sorted[j] - values_sorted[j-1]) < (values.max() - values.min()) * 0.02:
                    jitter[sorted_idx[j]] = jitter[sorted_idx[j-1]] + 0.1
                    if jitter[sorted_idx[j]] > 0.3:
                        jitter[sorted_idx[j]] = -0.3
                        
            x = i + jitter
            
            ax.scatter(x, values, c=self.colors[i % len(self.colors)],
                      alpha=0.6, s=50, edgecolors='white', linewidths=0.5)
                      
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories)
        ax.set_ylabel(value_col)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
        
    # ============ Streamgraph ============
    
    def streamgraph(self, data: pd.DataFrame, time_col: str,
                    value_cols: List[str], title: str = 'Streamgraph',
                    save_path: str = None):
        """
        流图 - 时间堆叠面积图的变体
        
        Args:
            data: DataFrame
            time_col: 时间列名
            value_cols: 数值列名列表
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        times = data[time_col].values
        values = data[value_cols].values.T  # [n_series, n_times]
        
        # 计算基线（居中）
        total = values.sum(axis=0)
        baseline = -total / 2
        
        # 堆叠
        y_stack = np.vstack([baseline, baseline + np.cumsum(values, axis=0)])
        
        for i in range(len(value_cols)):
            ax.fill_between(times, y_stack[i], y_stack[i+1],
                           color=self.colors[i % len(self.colors)],
                           alpha=0.8, label=value_cols[i])
                           
        ax.set_xlabel(time_col)
        ax.set_ylabel('Value')
        ax.set_title(title)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
        
    # ============ Arc Diagram ============
    
    def arc_diagram(self, nodes: List[str], edges: List[Tuple[int, int, float]],
                    title: str = 'Arc Diagram', save_path: str = None):
        """
        弧形图 - 线性网络可视化
        
        Args:
            nodes: 节点名列表
            edges: 边列表 [(source_idx, target_idx, weight), ...]
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        
        n = len(nodes)
        x_positions = np.linspace(0.1, 0.9, n)
        
        # 绘制节点
        ax.scatter(x_positions, [0.2] * n, s=100, c=self.colors[0], zorder=5)
        
        for i, (x, node) in enumerate(zip(x_positions, nodes)):
            ax.text(x, 0.15, node, ha='center', va='top', fontsize=9, rotation=45)
            
        # 绘制弧形边
        max_weight = max(e[2] for e in edges) if edges else 1
        
        for src, tgt, weight in edges:
            x1, x2 = x_positions[src], x_positions[tgt]
            
            # 计算弧的中点和高度
            mid_x = (x1 + x2) / 2
            height = abs(x2 - x1) * 0.5
            width = abs(x2 - x1)
            
            # 绘制弧
            arc = mpatches.Arc((mid_x, 0.2), width, height * 2,
                              angle=0, theta1=0, theta2=180,
                              color=self.colors[1], 
                              linewidth=1 + (weight / max_weight) * 3,
                              alpha=0.6)
            ax.add_patch(arc)
            
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.8)
        ax.set_title(title)
        ax.axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
        
    # ============ Dumbbell Chart ============
    
    def dumbbell_chart(self, categories: List[str], 
                       values1: List[float], values2: List[float],
                       label1: str = 'Before', label2: str = 'After',
                       title: str = 'Dumbbell Chart', save_path: str = None):
        """
        哑铃图 - 比较两个时间点/条件
        
        Args:
            categories: 类别列表
            values1, values2: 两组数值
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        y_positions = range(len(categories))
        
        # 绘制连接线
        for i, (v1, v2) in enumerate(zip(values1, values2)):
            ax.plot([v1, v2], [i, i], 'gray', linewidth=2, alpha=0.5)
            
        # 绘制点
        ax.scatter(values1, y_positions, s=100, c=self.colors[0], 
                  label=label1, zorder=5)
        ax.scatter(values2, y_positions, s=100, c=self.colors[1],
                  label=label2, zorder=5)
        
        ax.set_yticks(y_positions)
        ax.set_yticklabels(categories)
        ax.set_xlabel('Value')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig


if __name__ == '__main__':
    print("Testing Advanced Charts...")
    
    charts = AdvancedCharts()
    
    # Sankey图
    sources = [0, 0, 1, 1, 2, 2]
    targets = [3, 4, 3, 4, 3, 4]
    values = [10, 5, 8, 12, 6, 9]
    labels = ['Source A', 'Source B', 'Source C', 'Target 1', 'Target 2']
    charts.sankey_diagram(sources, targets, values, labels, title='Energy Flow')
    
    # Bump Chart
    data = pd.DataFrame({
        'time': [1, 1, 1, 2, 2, 2, 3, 3, 3],
        'rank': [1, 2, 3, 2, 1, 3, 1, 3, 2],
        'entity': ['A', 'B', 'C'] * 3
    })
    charts.bump_chart(data, 'time', 'rank', 'entity', title='Ranking Changes')
    
    # Beeswarm
    data = pd.DataFrame({
        'category': ['X'] * 50 + ['Y'] * 50 + ['Z'] * 50,
        'value': np.concatenate([
            np.random.randn(50),
            np.random.randn(50) + 1,
            np.random.randn(50) - 0.5
        ])
    })
    charts.beeswarm_plot(data, 'category', 'value', title='Distribution Comparison')
    
    # Dumbbell
    categories = ['Category A', 'Category B', 'Category C', 'Category D']
    values1 = [10, 15, 8, 20]
    values2 = [15, 12, 18, 22]
    charts.dumbbell_chart(categories, values1, values2, title='Before vs After')
    
    print("Advanced Charts test completed!")
    plt.show()

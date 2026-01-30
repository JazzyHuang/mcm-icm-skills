"""
Interactive Charts with Plotly
Plotly交互式图表

支持3D可视化、网络图、Pareto前沿等高级图表
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class InteractiveCharts:
    """Plotly交互式图表集合"""
    
    def __init__(self, color_palette: List[str] = None):
        if not PLOTLY_AVAILABLE:
            print("Warning: Plotly not available. Install with: pip install plotly kaleido")
            
        self.colors = color_palette or [
            '#0077BB', '#33BBEE', '#009988', '#EE7733', 
            '#CC3311', '#EE3377', '#BBBBBB'
        ]
        
    # ============ 3D图表 ============
    
    def surface_3d(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                   xlabel: str = 'X', ylabel: str = 'Y', zlabel: str = 'Z',
                   title: str = '3D Surface', colorscale: str = 'Viridis',
                   save_path: str = None):
        """3D曲面图"""
        if not PLOTLY_AVAILABLE:
            return None
            
        fig = go.Figure(data=[go.Surface(
            x=X, y=Y, z=Z,
            colorscale=colorscale,
            opacity=0.9
        )])
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=xlabel,
                yaxis_title=ylabel,
                zaxis_title=zlabel,
            ),
            width=800,
            height=600,
            font=dict(family='Arial', size=12)
        )
        
        if save_path:
            fig.write_image(save_path)
        return fig
        
    def scatter_3d(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                   color: np.ndarray = None, size: np.ndarray = None,
                   labels: List[str] = None, xlabel: str = 'X',
                   ylabel: str = 'Y', zlabel: str = 'Z',
                   title: str = '3D Scatter', save_path: str = None):
        """3D散点图"""
        if not PLOTLY_AVAILABLE:
            return None
            
        fig = go.Figure(data=[go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=size if size is not None else 5,
                color=color if color is not None else self.colors[0],
                colorscale='Viridis' if color is not None else None,
                opacity=0.8
            ),
            text=labels
        )])
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=xlabel,
                yaxis_title=ylabel,
                zaxis_title=zlabel,
            ),
            width=800,
            height=600
        )
        
        if save_path:
            fig.write_image(save_path)
        return fig
        
    def contour_3d(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                   xlabel: str = 'X', ylabel: str = 'Y',
                   title: str = 'Contour Plot', n_levels: int = 20,
                   save_path: str = None):
        """等高线图（带3D效果）"""
        if not PLOTLY_AVAILABLE:
            return None
            
        fig = go.Figure(data=[go.Contour(
            x=X[0] if len(X.shape) > 1 else X,
            y=Y[:, 0] if len(Y.shape) > 1 else Y,
            z=Z,
            colorscale='Viridis',
            ncontours=n_levels,
            contours=dict(
                showlabels=True,
                labelfont=dict(size=10, color='white')
            )
        )])
        
        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            width=800,
            height=600
        )
        
        if save_path:
            fig.write_image(save_path)
        return fig
        
    # ============ 网络图 ============
    
    def network_graph(self, nodes: List[Dict], edges: List[Dict],
                      title: str = 'Network Graph',
                      layout: str = 'spring', save_path: str = None):
        """
        交互式网络图
        
        Args:
            nodes: [{'id': 'A', 'label': 'Node A', 'size': 10, 'color': '#FF0000'}, ...]
            edges: [{'source': 'A', 'target': 'B', 'weight': 1}, ...]
            layout: 'spring', 'circular', 'random'
        """
        if not PLOTLY_AVAILABLE:
            return None
            
        # 计算节点位置
        try:
            import networkx as nx
            G = nx.Graph()
            for node in nodes:
                G.add_node(node['id'])
            for edge in edges:
                G.add_edge(edge['source'], edge['target'])
                
            if layout == 'spring':
                pos = nx.spring_layout(G)
            elif layout == 'circular':
                pos = nx.circular_layout(G)
            else:
                pos = nx.random_layout(G)
        except ImportError:
            # 简单的随机布局
            pos = {node['id']: (np.random.rand(), np.random.rand()) for node in nodes}
            
        # 边轨迹
        edge_x, edge_y = [], []
        for edge in edges:
            x0, y0 = pos[edge['source']]
            x1, y1 = pos[edge['target']]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # 节点轨迹
        node_x = [pos[node['id']][0] for node in nodes]
        node_y = [pos[node['id']][1] for node in nodes]
        node_colors = [node.get('color', self.colors[0]) for node in nodes]
        node_sizes = [node.get('size', 10) for node in nodes]
        node_labels = [node.get('label', node['id']) for node in nodes]
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_labels,
            textposition='top center',
            marker=dict(
                showscale=False,
                color=node_colors,
                size=node_sizes,
                line_width=1
            )
        )
        
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=title,
                           showlegend=False,
                           hovermode='closest',
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           width=800,
                           height=600
                       ))
        
        if save_path:
            fig.write_image(save_path)
        return fig
        
    def network_3d(self, nodes: List[Dict], edges: List[Dict],
                   title: str = '3D Network', save_path: str = None):
        """3D网络图"""
        if not PLOTLY_AVAILABLE:
            return None
            
        # 3D随机布局
        pos = {node['id']: (np.random.rand(), np.random.rand(), np.random.rand()) 
               for node in nodes}
        
        # 边
        edge_x, edge_y, edge_z = [], [], []
        for edge in edges:
            x0, y0, z0 = pos[edge['source']]
            x1, y1, z1 = pos[edge['target']]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])
            
        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # 节点
        node_x = [pos[node['id']][0] for node in nodes]
        node_y = [pos[node['id']][1] for node in nodes]
        node_z = [pos[node['id']][2] for node in nodes]
        
        node_trace = go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers+text',
            text=[node.get('label', node['id']) for node in nodes],
            marker=dict(
                size=[node.get('size', 8) for node in nodes],
                color=[node.get('color', self.colors[0]) for node in nodes],
                line_width=1
            )
        )
        
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(title=title, width=800, height=600)
        
        if save_path:
            fig.write_image(save_path)
        return fig
        
    # ============ 多目标优化图表 ============
    
    def pareto_front_2d(self, objectives: np.ndarray, labels: List[str] = None,
                        highlight_idx: List[int] = None,
                        xlabel: str = 'Objective 1',
                        ylabel: str = 'Objective 2',
                        title: str = 'Pareto Front', save_path: str = None):
        """2D Pareto前沿图"""
        if not PLOTLY_AVAILABLE:
            return None
            
        fig = go.Figure()
        
        # 所有点
        fig.add_trace(go.Scatter(
            x=objectives[:, 0],
            y=objectives[:, 1],
            mode='markers',
            marker=dict(size=8, color=self.colors[0], opacity=0.6),
            name='Solutions',
            text=labels
        ))
        
        # 高亮点
        if highlight_idx:
            fig.add_trace(go.Scatter(
                x=objectives[highlight_idx, 0],
                y=objectives[highlight_idx, 1],
                mode='markers',
                marker=dict(size=12, color=self.colors[1], symbol='star'),
                name='Selected'
            ))
            
        # Pareto前沿线
        sorted_idx = np.argsort(objectives[:, 0])
        pareto_x, pareto_y = [objectives[sorted_idx[0], 0]], [objectives[sorted_idx[0], 1]]
        min_y = objectives[sorted_idx[0], 1]
        
        for idx in sorted_idx[1:]:
            if objectives[idx, 1] <= min_y:
                pareto_x.append(objectives[idx, 0])
                pareto_y.append(objectives[idx, 1])
                min_y = objectives[idx, 1]
                
        fig.add_trace(go.Scatter(
            x=pareto_x, y=pareto_y,
            mode='lines',
            line=dict(color=self.colors[2], width=2, dash='dash'),
            name='Pareto Front'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            width=800,
            height=600
        )
        
        if save_path:
            fig.write_image(save_path)
        return fig
        
    def pareto_front_3d(self, objectives: np.ndarray,
                        labels: List[str] = None,
                        xlabel: str = 'Objective 1',
                        ylabel: str = 'Objective 2',
                        zlabel: str = 'Objective 3',
                        title: str = '3D Pareto Front', save_path: str = None):
        """3D Pareto前沿图"""
        if not PLOTLY_AVAILABLE:
            return None
            
        fig = go.Figure(data=[go.Scatter3d(
            x=objectives[:, 0],
            y=objectives[:, 1],
            z=objectives[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=objectives[:, 0],  # 按第一个目标着色
                colorscale='Viridis',
                opacity=0.8
            ),
            text=labels
        )])
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=xlabel,
                yaxis_title=ylabel,
                zaxis_title=zlabel,
            ),
            width=800,
            height=600
        )
        
        if save_path:
            fig.write_image(save_path)
        return fig
        
    # ============ 其他高级图表 ============
    
    def sankey_diagram(self, sources: List[int], targets: List[int],
                       values: List[float], labels: List[str],
                       title: str = 'Sankey Diagram', save_path: str = None):
        """桑基图"""
        if not PLOTLY_AVAILABLE:
            return None
            
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
        
        fig.update_layout(title=title, font=dict(size=10), width=800, height=600)
        
        if save_path:
            fig.write_image(save_path)
        return fig
        
    def sunburst(self, ids: List[str], labels: List[str],
                 parents: List[str], values: List[float],
                 title: str = 'Sunburst Chart', save_path: str = None):
        """旭日图"""
        if not PLOTLY_AVAILABLE:
            return None
            
        fig = go.Figure(go.Sunburst(
            ids=ids,
            labels=labels,
            parents=parents,
            values=values,
            branchvalues='total'
        ))
        
        fig.update_layout(title=title, width=700, height=700)
        
        if save_path:
            fig.write_image(save_path)
        return fig
        
    def parallel_categories(self, df: pd.DataFrame, dimensions: List[str],
                           color_col: str = None, title: str = 'Parallel Categories',
                           save_path: str = None):
        """平行类别图"""
        if not PLOTLY_AVAILABLE:
            return None
            
        fig = px.parallel_categories(
            df, 
            dimensions=dimensions,
            color=color_col,
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(title=title, width=900, height=600)
        
        if save_path:
            fig.write_image(save_path)
        return fig


def export_to_static(fig, path: str, format: str = 'pdf', scale: int = 2):
    """导出Plotly图表为静态图像"""
    if fig is None:
        return
    try:
        fig.write_image(path, format=format, scale=scale)
    except Exception as e:
        print(f"Export failed: {e}. Install kaleido: pip install kaleido")


if __name__ == '__main__':
    print("Testing Interactive Charts...")
    
    if PLOTLY_AVAILABLE:
        charts = InteractiveCharts()
        
        # 3D曲面
        x = np.linspace(-5, 5, 50)
        y = np.linspace(-5, 5, 50)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(np.sqrt(X**2 + Y**2))
        
        fig = charts.surface_3d(X, Y, Z, title='3D Surface Example')
        if fig:
            fig.show()
            
        # Pareto前沿
        objectives = np.random.rand(50, 2)
        objectives[:, 1] = 1 - objectives[:, 0] + np.random.rand(50) * 0.2
        
        fig = charts.pareto_front_2d(objectives, title='Pareto Front Example')
        if fig:
            fig.show()
            
        print("Interactive Charts test completed!")
    else:
        print("Plotly not available, skipping tests")

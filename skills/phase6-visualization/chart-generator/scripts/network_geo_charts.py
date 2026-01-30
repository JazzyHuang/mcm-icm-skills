"""
Network and Geographic Visualization
网络图和地理可视化模块

支持NetworkX网络图、地理热力图、流动图等
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    
try:
    import geopandas as gpd
    import folium
    GEO_AVAILABLE = True
except ImportError:
    GEO_AVAILABLE = False


NATURE_COLORS = ['#0077BB', '#33BBEE', '#009988', '#EE7733', '#CC3311', '#EE3377']


class NetworkVisualizer:
    """网络图可视化"""
    
    def __init__(self, color_palette: List[str] = None):
        self.colors = color_palette or NATURE_COLORS
        
    def create_graph(self, nodes: List[Dict], edges: List[Dict]) -> 'nx.Graph':
        """
        创建NetworkX图
        
        Args:
            nodes: [{'id': 'A', 'label': 'Node A', 'weight': 1}, ...]
            edges: [{'source': 'A', 'target': 'B', 'weight': 1}, ...]
        """
        if not NETWORKX_AVAILABLE:
            print("NetworkX not available. Install with: pip install networkx")
            return None
            
        G = nx.Graph()
        
        for node in nodes:
            G.add_node(node['id'], **{k: v for k, v in node.items() if k != 'id'})
            
        for edge in edges:
            G.add_edge(edge['source'], edge['target'], 
                      weight=edge.get('weight', 1))
            
        return G
        
    def draw_network(self, G, layout: str = 'spring', node_size_attr: str = None,
                    edge_width_attr: str = 'weight', node_color_attr: str = None,
                    title: str = 'Network Graph', figsize: Tuple = (10, 8),
                    save_path: str = None):
        """
        绘制网络图
        
        Args:
            G: NetworkX图
            layout: 'spring', 'circular', 'kamada_kawai', 'spectral', 'shell'
            node_size_attr: 节点大小属性
            edge_width_attr: 边宽度属性
            node_color_attr: 节点颜色属性
        """
        if G is None:
            return None
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # 布局
        layouts = {
            'spring': nx.spring_layout,
            'circular': nx.circular_layout,
            'kamada_kawai': nx.kamada_kawai_layout,
            'spectral': nx.spectral_layout,
            'shell': nx.shell_layout,
            'random': nx.random_layout
        }
        pos = layouts.get(layout, nx.spring_layout)(G)
        
        # 节点大小
        if node_size_attr:
            node_sizes = [G.nodes[n].get(node_size_attr, 300) * 100 for n in G.nodes()]
        else:
            node_sizes = 500
            
        # 节点颜色
        if node_color_attr:
            node_colors = [G.nodes[n].get(node_color_attr, 0) for n in G.nodes()]
        else:
            node_colors = self.colors[0]
            
        # 边宽度
        if edge_width_attr:
            edge_widths = [G[u][v].get(edge_width_attr, 1) for u, v in G.edges()]
        else:
            edge_widths = 1
            
        # 绘制
        nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths, alpha=0.5, 
                              edge_color='gray')
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                              node_color=node_colors, alpha=0.8,
                              cmap=plt.cm.viridis if node_color_attr else None)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=10)
        
        ax.set_title(title)
        ax.axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
        
    def draw_community(self, G, communities: List[List[str]] = None,
                       title: str = 'Community Structure', save_path: str = None):
        """
        绘制社区结构
        
        Args:
            G: NetworkX图
            communities: 社区列表 [['A', 'B'], ['C', 'D'], ...]
        """
        if G is None:
            return None
            
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 自动检测社区
        if communities is None and NETWORKX_AVAILABLE:
            try:
                from networkx.algorithms import community
                communities = list(community.louvain_communities(G))
            except:
                communities = [list(G.nodes())]
                
        pos = nx.spring_layout(G)
        
        # 按社区着色
        node_colors = []
        for node in G.nodes():
            for i, comm in enumerate(communities):
                if node in comm:
                    node_colors.append(self.colors[i % len(self.colors)])
                    break
            else:
                node_colors.append('gray')
                
        nx.draw(G, pos, ax=ax, node_color=node_colors, node_size=500,
               with_labels=True, font_size=9, alpha=0.8)
        
        ax.set_title(title)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
        
    def draw_flow_network(self, G, flow_values: Dict = None,
                         title: str = 'Network Flow', save_path: str = None):
        """
        绘制网络流图
        
        Args:
            G: NetworkX有向图
            flow_values: 边流量 {('A', 'B'): 10, ...}
        """
        if G is None:
            return None
            
        fig, ax = plt.subplots(figsize=(12, 8))
        pos = nx.spring_layout(G)
        
        # 边宽度按流量
        if flow_values:
            max_flow = max(flow_values.values())
            edge_widths = [flow_values.get((u, v), 1) / max_flow * 5 + 0.5 
                          for u, v in G.edges()]
        else:
            edge_widths = 1
            
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=500, 
                              node_color=self.colors[0], alpha=0.8)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=10)
        nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths,
                              alpha=0.6, edge_color='gray',
                              connectionstyle="arc3,rad=0.1",
                              arrows=True, arrowsize=15)
        
        # 边标签
        if flow_values:
            edge_labels = {(u, v): f'{flow_values.get((u, v), 0):.0f}' 
                          for u, v in G.edges()}
            nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
            
        ax.set_title(title)
        ax.axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
        
    def centrality_analysis(self, G, save_path: str = None) -> Dict:
        """
        中心性分析及可视化
        
        Returns:
            各种中心性指标
        """
        if G is None:
            return {}
            
        # 计算中心性
        degree_cent = nx.degree_centrality(G)
        betweenness_cent = nx.betweenness_centrality(G)
        closeness_cent = nx.closeness_centrality(G)
        
        try:
            eigenvector_cent = nx.eigenvector_centrality(G, max_iter=500)
        except:
            eigenvector_cent = degree_cent
            
        # 可视化
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        metrics = [
            ('Degree Centrality', degree_cent),
            ('Betweenness Centrality', betweenness_cent),
            ('Closeness Centrality', closeness_cent),
            ('Eigenvector Centrality', eigenvector_cent)
        ]
        
        pos = nx.spring_layout(G)
        
        for ax, (name, cent) in zip(axes.flat, metrics):
            node_sizes = [cent[n] * 3000 + 100 for n in G.nodes()]
            node_colors = [cent[n] for n in G.nodes()]
            
            nx.draw(G, pos, ax=ax, node_size=node_sizes, node_color=node_colors,
                   cmap=plt.cm.viridis, with_labels=True, font_size=8, alpha=0.8)
            ax.set_title(name)
            
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return {
            'degree': degree_cent,
            'betweenness': betweenness_cent,
            'closeness': closeness_cent,
            'eigenvector': eigenvector_cent
        }


class GeoVisualizer:
    """地理可视化"""
    
    def __init__(self, color_palette: List[str] = None):
        self.colors = color_palette or NATURE_COLORS
        
    def choropleth_map(self, data: 'gpd.GeoDataFrame', column: str,
                       title: str = 'Choropleth Map', cmap: str = 'viridis',
                       save_path: str = None):
        """
        地理热力图（分级统计图）
        
        Args:
            data: GeoPandas GeoDataFrame
            column: 用于着色的列名
        """
        if not GEO_AVAILABLE:
            print("GeoPandas not available. Install with: pip install geopandas")
            return None
            
        fig, ax = plt.subplots(figsize=(12, 8))
        
        data.plot(column=column, ax=ax, legend=True, cmap=cmap,
                 legend_kwds={'label': column, 'orientation': 'horizontal'})
        
        ax.set_title(title)
        ax.axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
        
    def point_map(self, lons: np.ndarray, lats: np.ndarray, 
                  values: np.ndarray = None, labels: List[str] = None,
                  title: str = 'Point Map', save_path: str = None):
        """
        点分布图（使用matplotlib）
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        if values is not None:
            scatter = ax.scatter(lons, lats, c=values, s=50, cmap='viridis', 
                                alpha=0.7, edgecolors='white')
            plt.colorbar(scatter, ax=ax)
        else:
            ax.scatter(lons, lats, s=50, c=self.colors[0], alpha=0.7, 
                      edgecolors='white')
            
        if labels:
            for i, label in enumerate(labels):
                ax.annotate(label, (lons[i], lats[i]), fontsize=8,
                           xytext=(3, 3), textcoords='offset points')
                           
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
        
    def flow_map(self, origins: List[Tuple[float, float]],
                 destinations: List[Tuple[float, float]],
                 weights: List[float] = None,
                 title: str = 'Flow Map', save_path: str = None):
        """
        流向图
        
        Args:
            origins: [(lon, lat), ...]
            destinations: [(lon, lat), ...]
            weights: 流量权重
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        max_weight = max(weights) if weights else 1
        
        for i, (orig, dest) in enumerate(zip(origins, destinations)):
            weight = weights[i] if weights else 1
            width = (weight / max_weight) * 3 + 0.5
            alpha = 0.3 + (weight / max_weight) * 0.5
            
            ax.annotate('', xy=dest, xytext=orig,
                       arrowprops=dict(arrowstyle='->', color=self.colors[0],
                                      lw=width, alpha=alpha))
                                      
        # 绘制点
        all_lons = [o[0] for o in origins] + [d[0] for d in destinations]
        all_lats = [o[1] for o in origins] + [d[1] for d in destinations]
        ax.scatter(all_lons[:len(origins)], all_lats[:len(origins)], 
                  s=50, c=self.colors[1], label='Origin', zorder=5)
        ax.scatter(all_lons[len(origins):], all_lats[len(origins):],
                  s=50, c=self.colors[2], label='Destination', zorder=5)
        
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
        
    def create_folium_map(self, center: Tuple[float, float] = (0, 0),
                          zoom: int = 2) -> 'folium.Map':
        """创建交互式Folium地图"""
        if not GEO_AVAILABLE:
            return None
        return folium.Map(location=center, zoom_start=zoom)
        
    def add_markers(self, m: 'folium.Map', lats: List[float], lons: List[float],
                    labels: List[str] = None, save_path: str = None):
        """添加标记点"""
        if m is None:
            return None
            
        for i, (lat, lon) in enumerate(zip(lats, lons)):
            label = labels[i] if labels else f'Point {i}'
            folium.Marker(
                location=[lat, lon],
                popup=label,
                icon=folium.Icon(color='blue')
            ).add_to(m)
            
        if save_path:
            m.save(save_path)
        return m


if __name__ == '__main__':
    print("Testing Network and Geo Charts...")
    
    # 网络图测试
    net_viz = NetworkVisualizer()
    
    nodes = [
        {'id': 'A', 'label': 'Node A', 'weight': 3},
        {'id': 'B', 'label': 'Node B', 'weight': 2},
        {'id': 'C', 'label': 'Node C', 'weight': 4},
        {'id': 'D', 'label': 'Node D', 'weight': 1},
        {'id': 'E', 'label': 'Node E', 'weight': 2}
    ]
    
    edges = [
        {'source': 'A', 'target': 'B', 'weight': 2},
        {'source': 'A', 'target': 'C', 'weight': 3},
        {'source': 'B', 'target': 'D', 'weight': 1},
        {'source': 'C', 'target': 'D', 'weight': 2},
        {'source': 'D', 'target': 'E', 'weight': 4},
        {'source': 'B', 'target': 'E', 'weight': 1}
    ]
    
    G = net_viz.create_graph(nodes, edges)
    if G:
        net_viz.draw_network(G, layout='spring', title='Network Example')
        
    # 地理图测试
    geo_viz = GeoVisualizer()
    lons = np.random.uniform(-180, 180, 20)
    lats = np.random.uniform(-90, 90, 20)
    values = np.random.rand(20)
    geo_viz.point_map(lons, lats, values, title='Random Points')
    
    print("Network and Geo Charts test completed!")
    plt.show()

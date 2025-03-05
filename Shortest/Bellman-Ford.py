"""
Bellman-Ford算法实现
用于解决单源最短路径问题，适用于存在负权边的图
可以检测负权环
时间复杂度: O(V*E)，其中V是顶点数，E是边数
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def bellman_ford(graph, source):
    """
    实现Bellman-Ford算法
    
    参数:
        graph: 图的表示，使用边列表 [(u, v, w)]，其中u和v是顶点，w是权重
        source: 源顶点
        
    返回:
        distances: 源顶点到所有其他顶点的最短距离
        predecessors: 最短路径中每个顶点的前驱节点
        negative_cycle: 是否存在负权环
    """
    # 获取所有顶点
    vertices = set()
    for u, v, _ in graph:
        vertices.add(u)
        vertices.add(v)
    
    # 初始化距离和前驱
    distances = {vertex: float('inf') for vertex in vertices}
    distances[source] = 0
    predecessors = {vertex: None for vertex in vertices}
    
    # 松弛操作 V-1 次
    for _ in range(len(vertices) - 1):
        for u, v, w in graph:
            if distances[u] != float('inf') and distances[u] + w < distances[v]:
                distances[v] = distances[u] + w
                predecessors[v] = u
    
    # 检测负权环
    negative_cycle = False
    for u, v, w in graph:
        if distances[u] != float('inf') and distances[u] + w < distances[v]:
            negative_cycle = True
            break
            
    return distances, predecessors, negative_cycle

def reconstruct_path(predecessors, destination):
    """
    根据前驱数组重建最短路径
    
    参数:
        predecessors: 前驱节点字典
        destination: 目标顶点
        
    返回:
        path: 从源顶点到目标顶点的最短路径
    """
    path = []
    current = destination
    
    while current is not None:
        path.append(current)
        current = predecessors[current]
        
    return path[::-1]  # 反转路径

def visualize_graph(graph, path=None):
    """
    可视化图和最短路径
    
    参数:
        graph: 图的表示，使用边列表 [(u, v, w)]
        path: 要高亮显示的路径
    """
    G = nx.DiGraph()
    
    # 添加边和权重
    for u, v, w in graph:
        G.add_edge(u, v, weight=w)
    
    # 创建图布局
    pos = nx.spring_layout(G, seed=42)
    
    plt.figure(figsize=(10, 7))
    
    # 绘制边
    nx.draw_networkx_edges(G, pos, width=1, alpha=0.7)
    
    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_size=500, alpha=0.8)
    
    # 绘制节点标签
    nx.draw_networkx_labels(G, pos, font_size=12)
    
    # 绘制边权重
    edge_labels = {(u, v): f"{w}" for u, v, w in graph}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
    
    # 如果有路径，高亮显示
    if path:
        path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=3, alpha=0.8, edge_color='r')
    
    plt.title("图可视化")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# 示例应用
if __name__ == "__main__":
    # 定义图（使用边列表表示）
    # 格式：(起点, 终点, 权重)
    graph = [
        ('A', 'B', 4),
        ('A', 'C', 2),
        ('B', 'C', 3),
        ('B', 'D', 2),
        ('B', 'E', 3),
        ('C', 'B', 1),
        ('C', 'D', 4),
        ('C', 'E', 5),
        ('E', 'D', -5)
    ]
    
    source = 'A'
    destination = 'D'
    
    # 运行Bellman-Ford算法
    distances, predecessors, negative_cycle = bellman_ford(graph, source)
    
    # 输出结果
    print(f"从顶点 {source} 到各顶点的最短距离:")
    for vertex, distance in distances.items():
        print(f"{vertex}: {distance}")
    
    if negative_cycle:
        print("\n警告: 图中存在负权环!")
    else:
        print("\n图中不存在负权环.")
    
    # 重建最短路径
    path = reconstruct_path(predecessors, destination)
    print(f"\n从 {source} 到 {destination} 的最短路径: {' -> '.join(path)}")
    print(f"路径长度: {distances[destination]}")
    
    # 可视化图和最短路径
    visualize_graph(graph, path)
    
    # 创建一个带有负权环的图例子
    graph_with_cycle = [
        ('A', 'B', 1),
        ('B', 'C', 2),
        ('C', 'D', 3),
        ('D', 'E', -2),
        ('E', 'B', -5)
    ]
    
    # 运行Bellman-Ford算法检测负权环
    _, _, has_negative_cycle = bellman_ford(graph_with_cycle, 'A')
    print("\n带有负权环的图例子:")
    if has_negative_cycle:
        print("检测到负权环!")
    
    # 可视化带有负权环的图
    visualize_graph(graph_with_cycle)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dijkstra算法实现及实例应用
"""

import heapq
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def dijkstra(graph, start):
    """
    使用Dijkstra算法计算从起点到所有其他节点的最短路径
    
    参数:
        graph: 图的邻接表表示，格式为 {节点: {邻接节点: 距离}}
        start: 起始节点
        
    返回:
        distances: 从起点到每个节点的最短距离
        predecessors: 最短路径中每个节点的前驱节点
    """
    # 初始化距离字典和前驱节点字典
    distances = {node: float('infinity') for node in graph}
    distances[start] = 0
    predecessors = {node: None for node in graph}
    
    # 优先队列，用于选择当前距离最小的节点
    priority_queue = [(0, start)]
    
    while priority_queue:
        # 获取当前距离最小的节点
        current_distance, current_node = heapq.heappop(priority_queue)
        
        # 如果找到的距离大于已知距离，则跳过
        if current_distance > distances[current_node]:
            continue
            
        # 检查当前节点的所有邻居
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            
            # 如果找到更短的路径，则更新距离和前驱节点
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                predecessors[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))
    
    return distances, predecessors

def get_path(predecessors, start, end):
    """
    根据前驱节点字典重建从起点到终点的路径
    
    参数:
        predecessors: 前驱节点字典
        start: 起始节点
        end: 终点节点
        
    返回:
        path: 从起点到终点的路径列表
    """
    path = []
    current = end
    
    # 如果终点无法到达
    if predecessors[end] is None and end != start:
        return []
        
    # 从终点回溯到起点
    while current is not None:
        path.append(current)
        current = predecessors[current]
        
    # 反转路径，使其从起点开始
    return path[::-1]

def visualize_graph(graph, shortest_path=None):
    """
    可视化图和最短路径
    
    参数:
        graph: 图的邻接表表示
        shortest_path: 最短路径节点列表
    """
    G = nx.DiGraph()
    
    # 添加节点和边
    for node, neighbors in graph.items():
        for neighbor, weight in neighbors.items():
            G.add_edge(node, neighbor, weight=weight)
    
    # 创建布局
    pos = nx.spring_layout(G, seed=42)
    
    plt.figure(figsize=(10, 8))
    
    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue')
    
    # 绘制边
    edges = G.edges()
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=1.5, alpha=0.5, arrows=True)
    
    # 绘制边权重
    edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    # 绘制节点标签
    nx.draw_networkx_labels(G, pos, font_size=15, font_family='sans-serif')
    
    # 高亮显示最短路径
    if shortest_path and len(shortest_path) > 1:
        path_edges = [(shortest_path[i], shortest_path[i+1]) for i in range(len(shortest_path)-1)]
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=3, alpha=0.8, 
                              edge_color='r', arrows=True)
    
    plt.title("图的可视化 (红色为最短路径)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    """
    Dijkstra算法的示例应用
    """
    # 创建示例图 (邻接表表示)
    graph = {
        'A': {'B': 5, 'C': 1},
        'B': {'A': 5, 'C': 2, 'D': 1},
        'C': {'A': 1, 'B': 2, 'D': 4, 'E': 8},
        'D': {'B': 1, 'C': 4, 'E': 3, 'F': 6},
        'E': {'C': 8, 'D': 3},
        'F': {'D': 6}
    }
    
    start_node = 'A'
    end_node = 'F'
    
    # 计算最短路径
    distances, predecessors = dijkstra(graph, start_node)
    
    # 获取从起点到终点的路径
    shortest_path = get_path(predecessors, start_node, end_node)
    
    # 打印结果
    print(f"从节点 {start_node} 到所有其他节点的最短距离:")
    for node, distance in distances.items():
        print(f"到节点 {node}: {distance}")
    
    print(f"\n从节点 {start_node} 到节点 {end_node} 的最短路径:")
    if shortest_path:
        path_str = ' -> '.join(shortest_path)
        print(f"路径: {path_str}")
        print(f"总距离: {distances[end_node]}")
    else:
        print(f"无法从节点 {start_node} 到达节点 {end_node}")
    
    # 可视化图和最短路径
    visualize_graph(graph, shortest_path)

if __name__ == "__main__":
    main()

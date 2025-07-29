#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Floyd-Warshall算法实现
用于计算图中所有点对之间的最短路径
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def floyd_warshall(graph):
    """
    Floyd-Warshall算法实现
    
    参数:
        graph: 邻接矩阵，表示加权有向图，graph[i][j]表示从i到j的权重
               如果i和j之间没有边，则graph[i][j]为float('inf')
               
    返回:
        dist: 最短路径矩阵，dist[i][j]表示从i到j的最短路径长度
        pred: 前驱矩阵，用于重建最短路径
    """
    n = len(graph)
    
    # 初始化距离矩阵和前驱矩阵
    dist = np.copy(graph)
    pred = np.zeros((n, n), dtype=int)
    
    # 初始化前驱矩阵
    for i in range(n):
        for j in range(n):
            if i != j and dist[i][j] < float('inf'):
                pred[i][j] = i
            else:
                pred[i][j] = -1
    
    # Floyd-Warshall算法核心部分
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    pred[i][j] = pred[k][j]
    
    return dist, pred

def reconstruct_path(pred, start, end):
    """
    根据前驱矩阵重建从start到end的最短路径
    
    参数:
        pred: 前驱矩阵
        start: 起始节点
        end: 终止节点
        
    返回:
        path: 最短路径的节点列表
    """
    if pred[start][end] == -1:
        return []
    
    path = [end]
    while path[0] != start:
        path.insert(0, pred[start][path[0]])
    
    return path

def visualize_graph(graph, shortest_paths=None):
    """
    可视化图和最短路径
    
    参数:
        graph: 邻接矩阵
        shortest_paths: 包含要高亮显示的路径的字典，格式为 {(start, end): path}
    """
    n = len(graph)
    G = nx.DiGraph()
    
    # 添加节点
    for i in range(n):
        G.add_node(i)
    
    # 添加边
    for i in range(n):
        for j in range(n):
            if i != j and graph[i][j] < float('inf'):
                G.add_edge(i, j, weight=graph[i][j])
    
    # 设置节点位置
    pos = nx.spring_layout(G, seed=42)
    
    plt.figure(figsize=(10, 8))
    
    # 绘制图
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
    nx.draw_networkx_labels(G, pos, font_size=12)
    
    # 绘制边和权重
    edge_labels = {(i, j): graph[i][j] for i, j in G.edges()}
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, arrowsize=15)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
    
    # 高亮显示最短路径
    if shortest_paths:
        for (start, end), path in shortest_paths.items():
            if len(path) > 1:
                path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
                nx.draw_networkx_edges(G, pos, edgelist=path_edges, 
                                      width=2.0, alpha=1.0, edge_color='r', 
                                      arrowsize=20)
    
    plt.title("有向图及最短路径可视化")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    # 创建一个示例图（邻接矩阵表示）
    # 使用float('inf')表示不存在的边
    inf = float('inf')
    example_graph = np.array([
        [0, 3, inf, 5, inf],
        [2, 0, inf, 4, inf],
        [inf, 1, 0, inf, 4],
        [inf, inf, 2, 0, 5],
        [inf, inf, inf, 1, 0]
    ])
    
    n = len(example_graph)
    print("原始图的邻接矩阵:")
    for i in range(n):
        print([example_graph[i][j] if example_graph[i][j] != inf else "inf" for j in range(n)])
    
    # 运行Floyd-Warshall算法
    dist, pred = floyd_warshall(example_graph)
    
    print("\n最短路径矩阵:")
    for i in range(n):
        print([dist[i][j] if dist[i][j] != inf else "inf" for j in range(n)])
    
    print("\n前驱矩阵:")
    for i in range(n):
        print(pred[i])
    
    # 输出特定点对之间的最短路径
    start, end = 0, 4  # 从节点0到节点4的最短路径
    path = reconstruct_path(pred, start, end)
    
    if path:
        print(f"\n从节点{start}到节点{end}的最短路径: {path}")
        print(f"路径长度: {dist[start][end]}")
        
        # 展示该路径经过的边
        path_str = ' -> '.join(map(str, path))
        print(f"路径详细信息: {path_str}")
    else:
        print(f"\n从节点{start}到节点{end}不存在路径")
    
    # 可视化图和最短路径
    shortest_paths = {(start, end): path}
    visualize_graph(example_graph, shortest_paths)
    
    # 展示所有节点对的最短路径
    print("\n所有节点对的最短路径:")
    for i in range(n):
        for j in range(n):
            if i != j:
                path = reconstruct_path(pred, i, j)
                if path:
                    path_str = ' -> '.join(map(str, path))
                    print(f"从节点{i}到节点{j}: {path_str} (距离: {dist[i][j]})")
                else:
                    print(f"从节点{i}到节点{j}不存在路径")

if __name__ == "__main__":
    main()

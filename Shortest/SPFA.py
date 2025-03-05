"""
SPFA (Shortest Path Faster Algorithm) 算法实现
是对Bellman-Ford算法的队列优化版本，用于计算带权图中单源最短路径问题
可以处理负权边，但不能处理负权环
"""
import heapq
from collections import defaultdict, deque
import time
import random
import matplotlib.pyplot as plt
import numpy as np

class Graph:
    """图的表示"""
    def __init__(self):
        # 使用邻接表存储图，key为顶点，value为(相邻顶点, 权重)的列表
        self.adj_list = defaultdict(list)
    
    def add_edge(self, u, v, w):
        """添加一条从顶点u到顶点v，权重为w的边"""
        self.adj_list[u].append((v, w))
    
    def spfa(self, source, n):
        """
        SPFA算法实现
        
        参数:
            source: 源顶点
            n: 顶点总数
            
        返回:
            dist: 源顶点到各顶点的最短距离
            prev: 记录最短路径的前驱节点
        """
        # 初始化距离数组和前驱数组
        dist = {i: float('inf') for i in range(n)}
        dist[source] = 0
        prev = {i: -1 for i in range(n)}
        
        # 初始化队列和访问状态
        queue = deque([source])
        in_queue = [False] * n
        in_queue[source] = True
        
        # 记录每个顶点入队次数，用于检测负权环
        count = [0] * n
        
        while queue:
            u = queue.popleft()
            in_queue[u] = False
            
            # 遍历u的所有邻居
            for v, weight in self.adj_list[u]:
                # 松弛操作
                if dist[u] + weight < dist[v]:
                    dist[v] = dist[u] + weight
                    prev[v] = u
                    
                    # 如果v不在队列中，将其加入队列
                    if not in_queue[v]:
                        queue.append(v)
                        in_queue[v] = True
                        count[v] += 1
                        
                        # 如果一个顶点入队次数超过n，说明存在负权环
                        if count[v] >= n:
                            raise ValueError("图中存在负权环，无法求解最短路径")
        
        return dist, prev

    def print_shortest_path(self, source, target, prev):
        """打印从源顶点到目标顶点的最短路径"""
        if prev[target] == -1 and source != target:
            print(f"从顶点 {source} 到顶点 {target} 不存在路径")
            return
        
        path = []
        current = target
        while current != -1:
            path.append(current)
            current = prev[current]
        
        path.reverse()
        path_str = " -> ".join(map(str, path))
        print(f"从顶点 {source} 到顶点 {target} 的最短路径是: {path_str}")

def compare_with_dijkstra(n_vertices, n_edges, has_negative_weights=False):
    """比较SPFA和Dijkstra算法的性能"""
    # 创建随机图
    g_spfa = Graph()
    g_dijkstra = Graph()
    
    # 生成随机边
    edges = set()
    for _ in range(n_edges):
        u = random.randint(0, n_vertices-1)
        v = random.randint(0, n_vertices-1)
        if u != v and (u, v) not in edges:
            weight_range = (-10, 100) if has_negative_weights else (1, 100)
            w = random.randint(*weight_range)
            g_spfa.add_edge(u, v, w)
            g_dijkstra.add_edge(u, v, w)
            edges.add((u, v))
    
    # 计时SPFA
    start_time = time.time()
    try:
        dist_spfa, _ = g_spfa.spfa(0, n_vertices)
        spfa_time = time.time() - start_time
    except ValueError:
        print("SPFA检测到负权环，无法计算最短路径")
        return None, None
    
    # 如果有负权边，Dijkstra算法不适用
    if has_negative_weights:
        return spfa_time, None
    
    # 计时Dijkstra
    start_time = time.time()
    dist_dijkstra = dijkstra(g_dijkstra, 0, n_vertices)
    dijkstra_time = time.time() - start_time
    
    return spfa_time, dijkstra_time

def dijkstra(graph, source, n):
    """Dijkstra算法实现，用于与SPFA比较"""
    dist = {i: float('inf') for i in range(n)}
    dist[source] = 0
    visited = [False] * n
    
    pq = [(0, source)]  # (距离, 顶点)
    
    while pq:
        d, u = heapq.heappop(pq)
        
        if visited[u]:
            continue
        
        visited[u] = True
        
        for v, weight in graph.adj_list[u]:
            if not visited[v] and d + weight < dist[v]:
                dist[v] = d + weight
                heapq.heappush(pq, (dist[v], v))
    
    return dist

def performance_comparison():
    """性能比较及可视化"""
    vertices_range = [100, 500, 1000, 2000, 5000]
    spfa_times = []
    dijkstra_times = []
    
    for n in vertices_range:
        # 稀疏图：边数约为顶点数的2倍
        n_edges = n * 2
        spfa_time, dijkstra_time = compare_with_dijkstra(n, n_edges)
        
        if spfa_time is not None:
            spfa_times.append(spfa_time)
            dijkstra_times.append(dijkstra_time)
    
    # 绘制性能比较图
    plt.figure(figsize=(10, 6))
    plt.plot(vertices_range[:len(spfa_times)], spfa_times, 'o-', label='SPFA')
    plt.plot(vertices_range[:len(dijkstra_times)], dijkstra_times, 's-', label='Dijkstra')
    plt.xlabel('顶点数量')
    plt.ylabel('运行时间 (秒)')
    plt.title('SPFA vs Dijkstra 算法性能比较')
    plt.legend()
    plt.grid(True)
    plt.savefig('spfa_dijkstra_comparison.png')
    plt.show()

def main():
    """主函数，演示SPFA算法的使用"""
    # 创建一个示例图
    g = Graph()
    
    # 添加边 (u, v, weight)
    g.add_edge(0, 1, 4)
    g.add_edge(0, 2, 2)
    g.add_edge(1, 2, 5)
    g.add_edge(1, 3, 10)
    g.add_edge(2, 1, 3)
    g.add_edge(2, 3, 2)
    g.add_edge(2, 4, 8)
    g.add_edge(3, 4, 7)
    g.add_edge(4, 3, 9)
    
    # 源顶点
    source = 0
    n_vertices = 5
    
    # 运行SPFA算法
    print(f"使用SPFA算法计算从顶点 {source} 到所有其他顶点的最短路径:")
    try:
        dist, prev = g.spfa(source, n_vertices)
        
        # 打印最短距离
        for i in range(n_vertices):
            print(f"从顶点 {source} 到顶点 {i} 的最短距离为: {dist[i]}")
        
        # 打印最短路径
        for i in range(n_vertices):
            g.print_shortest_path(source, i, prev)
        
        print("\n带有负权边的图示例:")
        g_neg = Graph()
        g_neg.add_edge(0, 1, 4)
        g_neg.add_edge(0, 2, 2)
        g_neg.add_edge(1, 2, -5)  # 负权边
        g_neg.add_edge(1, 3, 10)
        g_neg.add_edge(2, 3, 2)
        
        dist_neg, prev_neg = g_neg.spfa(0, 4)
        for i in range(4):
            print(f"从顶点 0 到顶点 {i} 的最短距离为: {dist_neg[i]}")
            g_neg.print_shortest_path(0, i, prev_neg)
        
    except ValueError as e:
        print(f"错误: {e}")

    # 调用性能比较函数
    # print("\n性能比较...")
    # performance_comparison()

if __name__ == "__main__":
    main()

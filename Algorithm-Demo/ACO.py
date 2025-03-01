"""
蚁群优化算法(Ant Colony Optimization, ACO)解决旅行商问题(TSP)
"""

import numpy as np
import matplotlib.pyplot as plt
import time

class AntColonyOptimization:
    def __init__(self, city_positions, n_ants=50, n_iterations=200, alpha=1.0, beta=2.0, 
                 rho=0.5, q=100, initial_pheromone=1.0):
        """
        初始化蚁群优化算法
        
        参数:
            city_positions: 城市坐标数组，形状为 [n_cities, 2]
            n_ants: 蚂蚁数量
            n_iterations: 迭代次数
            alpha: 信息素重要性参数
            beta: 启发式信息重要性参数
            rho: 信息素蒸发率
            q: 信息素强度参数
            initial_pheromone: 初始信息素浓度
        """
        self.city_positions = city_positions
        self.n_cities = len(city_positions)
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha  # 信息素重要性参数
        self.beta = beta    # 启发式信息重要性参数
        self.rho = rho      # 信息素蒸发率
        self.q = q          # 信息素强度参数
        
        # 计算城市间距离矩阵
        self.distance_matrix = self._calculate_distances()
        
        # 初始化信息素矩阵
        self.pheromone_matrix = np.ones((self.n_cities, self.n_cities)) * initial_pheromone
        
        # 启发式信息矩阵 (距离的倒数)
        self.heuristic_matrix = 1.0 / (self.distance_matrix + np.eye(self.n_cities))
        np.fill_diagonal(self.heuristic_matrix, 0)  # 对角线设为0，避免自循环
        
        # 存储最佳路径和长度
        self.best_path = None
        self.best_distance = float('inf')
        
        # 存储每次迭代的最佳距离，用于可视化
        self.iteration_best_distances = []
        
    def _calculate_distances(self):
        """计算城市间的欧氏距离矩阵"""
        distances = np.zeros((self.n_cities, self.n_cities))
        for i in range(self.n_cities):
            for j in range(self.n_cities):
                if i != j:
                    # 计算两个城市之间的欧氏距离
                    distances[i, j] = np.sqrt(np.sum((self.city_positions[i] - self.city_positions[j]) ** 2))
        return distances
    
    def _select_next_city(self, ant, current_city, visited_cities):
        """
        根据概率选择蚂蚁下一个要访问的城市
        
        参数:
            ant: 蚂蚁编号
            current_city: 当前所在城市
            visited_cities: 已访问城市列表
            
        返回:
            下一个城市的索引
        """
        # 创建未访问城市的掩码
        unvisited_mask = np.ones(self.n_cities, dtype=bool)
        unvisited_mask[visited_cities] = False
        
        # 如果所有城市都已访问，返回起始城市
        if not np.any(unvisited_mask):
            return visited_cities[0]
        
        # 计算转移概率
        pheromone = self.pheromone_matrix[current_city, :]
        heuristic = self.heuristic_matrix[current_city, :]
        
        # 计算分子
        numerator = (pheromone ** self.alpha) * (heuristic ** self.beta)
        # 将已访问城市的概率设为0
        numerator = numerator * unvisited_mask
        
        # 如果所有值都为0，随机选择一个未访问的城市
        if np.sum(numerator) == 0:
            unvisited_cities = np.where(unvisited_mask)[0]
            return np.random.choice(unvisited_cities)
        
        # 计算概率
        probabilities = numerator / np.sum(numerator)
        
        # 根据概率选择下一个城市
        next_city = np.random.choice(range(self.n_cities), p=probabilities)
        return next_city
    
    def _construct_solutions(self):
        """
        构建所有蚂蚁的解决方案
        
        返回:
            all_paths: 所有蚂蚁的路径
            all_distances: 所有蚂蚁的路径长度
        """
        all_paths = []
        all_distances = []
        
        for ant in range(self.n_ants):
            # 随机选择起始城市
            start_city = np.random.randint(0, self.n_cities)
            path = [start_city]
            visited_cities = [start_city]
            current_city = start_city
            
            # 构建完整路径
            while len(visited_cities) < self.n_cities:
                next_city = self._select_next_city(ant, current_city, visited_cities)
                path.append(next_city)
                visited_cities.append(next_city)
                current_city = next_city
            
            # 添加返回起始城市，形成闭环
            path.append(path[0])
            
            # 计算路径总长度
            distance = self._calculate_path_distance(path)
            
            all_paths.append(path)
            all_distances.append(distance)
            
            # 更新最佳路径
            if distance < self.best_distance:
                self.best_distance = distance
                self.best_path = path.copy()
        
        return all_paths, all_distances
    
    def _calculate_path_distance(self, path):
        """计算给定路径的总长度"""
        distance = 0
        for i in range(len(path) - 1):
            distance += self.distance_matrix[path[i], path[i+1]]
        return distance
    
    def _update_pheromones(self, all_paths, all_distances):
        """
        更新信息素矩阵
        
        参数:
            all_paths: 所有蚂蚁的路径
            all_distances: 所有蚂蚁的路径长度
        """
        # 信息素蒸发
        self.pheromone_matrix *= (1 - self.rho)
        
        # 信息素更新
        for ant in range(self.n_ants):
            path = all_paths[ant]
            distance = all_distances[ant]
            
            # 更新该蚂蚁经过的边的信息素
            for i in range(len(path) - 1):
                city_from = path[i]
                city_to = path[i+1]
                # 信息素增量与路径长度成反比
                delta_pheromone = self.q / distance
                self.pheromone_matrix[city_from, city_to] += delta_pheromone
                self.pheromone_matrix[city_to, city_from] += delta_pheromone  # 对称更新
    
    def run(self):
        """运行蚁群优化算法"""
        start_time = time.time()
        
        for iteration in range(self.n_iterations):
            # 构建所有蚂蚁的解决方案
            all_paths, all_distances = self._construct_solutions()
            
            # 更新信息素
            self._update_pheromones(all_paths, all_distances)
            
            # 记录当前迭代的最佳距离
            iteration_best = min(all_distances)
            self.iteration_best_distances.append(iteration_best)
            
            # 打印进度
            if (iteration + 1) % 10 == 0 or iteration == 0:
                print(f"迭代 {iteration + 1}/{self.n_iterations}, 当前最佳距离: {self.best_distance:.2f}")
        
        end_time = time.time()
        print(f"总运行时间: {end_time - start_time:.2f} 秒")
        print(f"最佳路径长度: {self.best_distance:.2f}")
        
        return self.best_path, self.best_distance
    
    def plot_progress(self):
        """绘制算法收敛过程"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.iteration_best_distances)
        plt.title('蚁群算法收敛过程')
        plt.xlabel('迭代次数')
        plt.ylabel('最佳路径长度')
        plt.grid(True)
        plt.show()
    
    def plot_solution(self):
        """绘制最佳路径"""
        plt.figure(figsize=(10, 8))
        
        # 绘制城市点
        plt.scatter(self.city_positions[:, 0], self.city_positions[:, 1], 
                   c='red', marker='o', s=100)
        
        # 标记城市编号
        for i, (x, y) in enumerate(self.city_positions):
            plt.text(x + 0.1, y + 0.1, str(i), fontsize=12)
        
        # 绘制最佳路径
        if self.best_path is not None:
            for i in range(len(self.best_path) - 1):
                city_from = self.best_path[i]
                city_to = self.best_path[i+1]
                plt.plot([self.city_positions[city_from, 0], self.city_positions[city_to, 0]],
                         [self.city_positions[city_from, 1], self.city_positions[city_to, 1]],
                         'b-', alpha=0.7)
        
        plt.title(f'旅行商问题的蚁群优化算法解决方案 (距离: {self.best_distance:.2f})')
        plt.grid(True)
        plt.show()

# 演示代码
if __name__ == "__main__":
    # 设置随机种子，确保结果可重现
    np.random.seed(42)
    
    # 创建随机城市坐标 (为了演示，我们使用20个城市)
    n_cities = 20
    city_positions = np.random.rand(n_cities, 2) * 100
    
    # 初始化ACO求解器
    aco = AntColonyOptimization(
        city_positions=city_positions,
        n_ants=50,             # 蚂蚁数量
        n_iterations=100,      # 迭代次数
        alpha=1.0,             # 信息素重要性参数
        beta=2.0,              # 启发式信息重要性参数
        rho=0.5,               # 信息素蒸发率
        q=100,                 # 信息素强度参数
        initial_pheromone=1.0  # 初始信息素浓度
    )
    
    # 运行算法
    best_path, best_distance = aco.run()
    
    # 可视化结果
    aco.plot_progress()
    aco.plot_solution()
    
    # 打印最佳路径
    print("最佳路径:", best_path)
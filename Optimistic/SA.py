#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模拟退火算法(Simulated Annealing)求解旅行商问题(TSP)的Python实现
旅行商问题：给定一系列城市和它们之间的距离，寻找一条最短的路径使旅行商恰好访问每个城市一次并回到起点。
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import random
import time

class SimulatedAnnealing:
    """模拟退火算法实现类"""
    
    def __init__(self, coords, temperature=100, alpha=0.99, stopping_temperature=1e-8, stopping_iter=10000):
        """
        初始化模拟退火算法的参数
        
        参数:
            coords: 城市坐标列表，每个元素是(x, y)坐标
            temperature: 初始温度
            alpha: 温度衰减系数
            stopping_temperature: 停止温度
            stopping_iter: 停止迭代次数
        """
        self.coords = coords  # 城市坐标
        self.n_cities = len(coords)  # 城市数量
        self.temperature = temperature  # 初始温度
        self.alpha = alpha  # 温度衰减系数
        self.stopping_temperature = stopping_temperature  # 停止温度
        self.stopping_iter = stopping_iter  # 停止迭代次数
        
        # 计算各城市间距离矩阵
        self.dist_matrix = self._compute_distance_matrix()
        
        # 初始解（随机路径）
        self.curr_solution = self._initial_solution()
        self.best_solution = self.curr_solution.copy()
        
        # 计算当前解和最优解的路径长度
        self.curr_distance = self._calculate_total_distance(self.curr_solution)
        self.best_distance = self.curr_distance
        
        # 记录迭代过程中的距离变化，用于绘图
        self.distances = [self.curr_distance]
        
        # 无改进次数计数器
        self.iterations_without_improvement = 0
    
    def _compute_distance_matrix(self):
        """计算所有城市间的距离矩阵"""
        dist_matrix = np.zeros((self.n_cities, self.n_cities))
        for i in range(self.n_cities):
            for j in range(self.n_cities):
                if i != j:
                    # 计算欧几里得距离
                    dist_matrix[i][j] = math.sqrt(
                        (self.coords[i][0] - self.coords[j][0]) ** 2 + 
                        (self.coords[i][1] - self.coords[j][1]) ** 2
                    )
        return dist_matrix
    
    def _initial_solution(self):
        """生成初始解（随机排列城市）"""
        solution = list(range(self.n_cities))
        random.shuffle(solution)
        return solution
    
    def _calculate_total_distance(self, solution):
        """计算路径总长度"""
        total_dist = 0
        for i in range(self.n_cities - 1):
            total_dist += self.dist_matrix[solution[i]][solution[i+1]]
        # 回到起点，形成闭环路径
        total_dist += self.dist_matrix[solution[-1]][solution[0]]
        return total_dist
    
    def _get_neighbor(self):
        """
        获取邻域解
        使用2-opt方法：随机选择两个位置，交换它们之间的子路径
        """
        # 复制当前解
        neighbor = self.curr_solution.copy()
        
        # 随机选择两个不同的位置
        l = random.randint(0, self.n_cities - 1)
        r = random.randint(0, self.n_cities - 1)
        
        # 保证l < r
        if l > r:
            l, r = r, l
            
        # 反转l和r之间的子路径
        neighbor[l:r+1] = reversed(neighbor[l:r+1])
        
        return neighbor
    
    def run(self):
        """运行模拟退火算法"""
        start_time = time.time()  # 记录开始时间
        
        print("开始模拟退火算法...")
        print(f"初始路径长度: {self.curr_distance:.2f}")
        
        # 迭代直到满足停止条件
        iter_num = 0
        while self.temperature > self.stopping_temperature and iter_num < self.stopping_iter:
            # 获取邻域解
            neighbor = self._get_neighbor()
            # 计算邻域解的路径长度
            neighbor_distance = self._calculate_total_distance(neighbor)
            
            # 计算接受概率
            if neighbor_distance < self.curr_distance:  # 更好的解，直接接受
                accept = True
            else:  # 较差的解，以一定概率接受
                # 计算能量差
                delta = neighbor_distance - self.curr_distance
                # 计算接受概率
                accept_probability = math.exp(-delta / self.temperature)
                # 生成随机数(0-1)，决定是否接受较差解
                accept = random.random() < accept_probability
            
            # 根据接受概率更新当前解
            if accept:
                self.curr_solution = neighbor
                self.curr_distance = neighbor_distance
                
                # 更新最优解
                if self.curr_distance < self.best_distance:
                    self.best_solution = self.curr_solution.copy()
                    self.best_distance = self.curr_distance
                    self.iterations_without_improvement = 0
                else:
                    self.iterations_without_improvement += 1
            else:
                self.iterations_without_improvement += 1
            
            # 记录当前最优距离
            self.distances.append(self.best_distance)
            
            # 降温
            self.temperature *= self.alpha
            iter_num += 1
            
            # 每100次迭代输出一次当前状态
            if iter_num % 100 == 0:
                print(f"迭代次数: {iter_num}, 温度: {self.temperature:.4f}, 当前最优距离: {self.best_distance:.2f}")
        
        execution_time = time.time() - start_time
        print(f"模拟退火完成! 总迭代次数: {iter_num}")
        print(f"最优路径长度: {self.best_distance:.2f}")
        print(f"执行时间: {execution_time:.2f} 秒")
        
        return self.best_solution, self.best_distance, self.distances
    
    def visualize_route(self):
        """可视化最优路径"""
        plt.figure(figsize=(10, 6))
        
        # 绘制城市点
        x = [self.coords[i][0] for i in range(self.n_cities)]
        y = [self.coords[i][1] for i in range(self.n_cities)]
        plt.scatter(x, y, c='red', s=50)
        
        # 给城市标号
        for i in range(self.n_cities):
            plt.annotate(f"{i}", (x[i], y[i]), fontsize=12)
        
        # 绘制最优路径
        route_x = [self.coords[self.best_solution[i % self.n_cities]][0] for i in range(self.n_cities + 1)]
        route_y = [self.coords[self.best_solution[i % self.n_cities]][1] for i in range(self.n_cities + 1)]
        plt.plot(route_x, route_y, 'b-', linewidth=0.8)
        
        plt.title(f"模拟退火算法求解TSP - 最优路径长度: {self.best_distance:.2f}")
        plt.xlabel("X 坐标")
        plt.ylabel("Y 坐标")
        plt.grid(True)
        plt.tight_layout()
        
        # 保存图像
        plt.savefig("tsp_route.png")
        plt.show()
    
    def plot_learning(self):
        """绘制学习曲线（距离随迭代次数的变化）"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.distances)
        plt.title("模拟退火算法 - 收敛曲线")
        plt.xlabel("迭代次数")
        plt.ylabel("路径长度")
        plt.grid(True)
        plt.tight_layout()
        
        # 保存图像
        plt.savefig("tsp_learning_curve.png")
        plt.show()


def main():
    """主函数"""
    # 生成随机城市坐标
    seed = 42  # 随机种子，保证可重复性
    random.seed(seed)
    np.random.seed(seed)
    
    # 城市数量
    n_cities = 20
    
    # 在[0, 100]范围内生成城市坐标
    coords = [[random.uniform(0, 100), random.uniform(0, 100)] for _ in range(n_cities)]
    
    # 初始化和运行模拟退火算法
    sa = SimulatedAnnealing(
        coords=coords,
        temperature=10000,
        alpha=0.99,
        stopping_temperature=1e-6,
        stopping_iter=100000
    )
    
    # 运行算法
    best_solution, best_distance, distances = sa.run()
    
    # 输出结果
    print("最优路径:", best_solution)
    print(f"最优路径长度: {best_distance:.2f}")
    
    # 可视化
    sa.visualize_route()
    sa.plot_learning()


if __name__ == "__main__":
    main()

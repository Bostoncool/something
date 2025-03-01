#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
蜣螂优化算法(Dung Beetle Optimizer, DBO)的Python实现
参考文献: Abdollahzadeh B, Gharehchopogh F S, Mirjalili S. Dung beetle optimizer: A novel nature-inspired metaheuristic algorithm[J]. 
         Advances in Engineering Software, 2022, 168: 103087.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class DungBeetleOptimizer:
    """
    蜣螂优化算法实现类
    """
    def __init__(self, objective_func, dim, lb, ub, population_size=30, max_iter=100):
        """
        初始化蜣螂优化算法参数
        
        参数:
        objective_func: 目标函数 (适应度函数)
        dim: 问题维度
        lb: 搜索空间下界 (list或numpy数组)
        ub: 搜索空间上界 (list或numpy数组)
        population_size: 种群大小
        max_iter: 最大迭代次数
        """
        self.objective_func = objective_func  # 目标函数
        self.dim = dim                        # 问题维度
        self.lb = np.array(lb)                # 下界
        self.ub = np.array(ub)                # 上界
        self.population_size = population_size  # 种群大小
        self.max_iter = max_iter              # 最大迭代次数
        
        # 参数初始化
        self.position = None                  # 蜣螂位置
        self.fitness = None                   # 适应度值
        self.best_position = None             # 全局最优位置
        self.best_fitness = float('inf')      # 全局最优适应度
        self.convergence_curve = np.zeros(max_iter)  # 收敛曲线
    
    def initialize_population(self):
        """
        初始化蜣螂种群位置
        """
        # 在搜索空间中随机生成种群
        self.position = np.random.uniform(
            low=self.lb, 
            high=self.ub, 
            size=(self.population_size, self.dim)
        )
        
        # 计算初始种群的适应度
        self.fitness = np.array([self.objective_func(p) for p in self.position])
        
        # 更新全局最优
        best_idx = np.argmin(self.fitness)
        if self.fitness[best_idx] < self.best_fitness:
            self.best_fitness = self.fitness[best_idx]
            self.best_position = self.position[best_idx].copy()
    
    def optimize(self):
        """
        执行蜣螂优化算法主循环
        
        返回:
        best_position: 找到的最优位置
        best_fitness: 最优位置对应的适应度值
        convergence_curve: 收敛曲线
        """
        # 初始化种群
        self.initialize_population()
        
        # 算法主循环
        for t in range(self.max_iter):
            # 对每只蜣螂进行操作
            for i in range(self.population_size):
                # 步骤1: 计算因子c1，影响滚球行为
                c1 = 2 * np.exp(-(4 * t / self.max_iter) ** 2)
                
                # 步骤2: 滚球行为 - 向最优位置移动
                r1 = np.random.random(self.dim)
                new_position = self.position[i] + c1 * r1 * (self.best_position - self.position[i])
                
                # 步骤3: 埋藏行为 - 随机探索
                r2 = np.random.random()
                r3 = np.random.random(self.dim)
                
                if r2 < 0.5:
                    # 引入一些随机性，模拟蜣螂埋藏粪球的行为
                    random_vector = np.random.random(self.dim)
                    new_position = new_position + r3 * (random_vector * (self.ub - self.lb) + self.lb)
                else:
                    # 局部搜索
                    random_index = np.random.randint(0, self.population_size)
                    random_beetle = self.position[random_index]
                    new_position = new_position + r3 * (random_beetle - self.position[i])
                
                # 边界处理
                new_position = np.clip(new_position, self.lb, self.ub)
                
                # 计算新位置的适应度
                new_fitness = self.objective_func(new_position)
                
                # 如果新位置更好，则更新
                if new_fitness < self.fitness[i]:
                    self.position[i] = new_position
                    self.fitness[i] = new_fitness
                    
                    # 更新全局最优
                    if new_fitness < self.best_fitness:
                        self.best_fitness = new_fitness
                        self.best_position = new_position.copy()
            
            # 更新收敛曲线
            self.convergence_curve[t] = self.best_fitness
            
            # 可选: 打印每次迭代的信息
            if (t+1) % 10 == 0 or t == 0:
                print(f"迭代 {t+1}: 最优适应度 = {self.best_fitness:.6f}")
        
        return self.best_position, self.best_fitness, self.convergence_curve


def sphere_function(x):
    """
    球函数 (Sphere Function) - 一个常用的测试函数
    全局最优解: f(0,0,...,0) = 0
    """
    return np.sum(x**2)


def rosenbrock_function(x):
    """
    罗森布鲁克函数 (Rosenbrock Function) - 另一个常用的测试函数
    全局最优解: f(1,1,...,1) = 0
    """
    sum_val = 0
    for i in range(len(x) - 1):
        sum_val += 100 * (x[i+1] - x[i]**2)**2 + (x[i] - 1)**2
    return sum_val


def rastrigin_function(x):
    """
    拉斯特里金函数 (Rastrigin Function) - 带有多个局部最优解的测试函数
    全局最优解: f(0,0,...,0) = 0
    """
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


def test_dbo_algorithm():
    """
    测试蜣螂优化算法
    """
    print("=" * 50)
    print("蜣螂优化算法(DBO)测试")
    print("=" * 50)
    
    # 测试案例1: 优化球函数
    print("\n测试案例1: 优化球函数(Sphere Function)")
    dim = 10
    lb = [-100] * dim
    ub = [100] * dim
    
    dbo = DungBeetleOptimizer(
        objective_func=sphere_function,
        dim=dim,
        lb=lb,
        ub=ub,
        population_size=30,
        max_iter=100
    )
    
    best_position, best_fitness, convergence_curve = dbo.optimize()
    
    print("\n最终结果:")
    print(f"最优解: {best_position}")
    print(f"最优适应度: {best_fitness:.6f}")
    
    # 绘制收敛曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, dbo.max_iter+1), convergence_curve, 'b-', linewidth=2)
    plt.yscale('log')
    plt.xlabel('迭代次数')
    plt.ylabel('适应度值 (对数刻度)')
    plt.title('球函数的DBO算法收敛曲线')
    plt.grid(True)
    plt.savefig('DBO_sphere_convergence.png', dpi=300)
    plt.show()
    
    # 测试案例2: 优化Rastrigin函数
    print("\n测试案例2: 优化Rastrigin函数")
    dim = 2  # 使用2维以便可视化
    lb = [-5.12] * dim
    ub = [5.12] * dim
    
    dbo = DungBeetleOptimizer(
        objective_func=rastrigin_function,
        dim=dim,
        lb=lb,
        ub=ub,
        population_size=50,
        max_iter=100
    )
    
    best_position, best_fitness, convergence_curve = dbo.optimize()
    
    print("\n最终结果:")
    print(f"最优解: {best_position}")
    print(f"最优适应度: {best_fitness:.6f}")
    
    # 绘制收敛曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, dbo.max_iter+1), convergence_curve, 'r-', linewidth=2)
    plt.yscale('log')
    plt.xlabel('迭代次数')
    plt.ylabel('适应度值 (对数刻度)')
    plt.title('Rastrigin函数的DBO算法收敛曲线')
    plt.grid(True)
    plt.savefig('DBO_rastrigin_convergence.png', dpi=300)
    plt.show()
    
    # 为Rastrigin函数绘制搜索空间和最优解
    if dim == 2:
        x = np.linspace(lb[0], ub[0], 100)
        y = np.linspace(lb[1], ub[1], 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = rastrigin_function(np.array([X[i, j], Y[i, j]]))
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, linewidth=0)
        ax.scatter(best_position[0], best_position[1], best_fitness, c='r', marker='*', s=100, label='最优解')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('f(x, y)')
        ax.set_title('Rastrigin函数的搜索空间和DBO找到的最优解')
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.legend()
        plt.savefig('DBO_rastrigin_surface.png', dpi=300)
        plt.show()


if __name__ == "__main__":
    test_dbo_algorithm()

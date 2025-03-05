"""
灰狼优化算法(Grey Wolf Optimizer, GWO)实现
该算法模拟灰狼的社会等级和狩猎行为，是一种群体智能优化算法。
灰狼社会等级分为：α(领导者)、β(次领导者)、δ(中等级)和ω(最低等级)
猎物的搜索过程由α、β、δ引导，ω狼根据它们的位置更新自己。
"""

import numpy as np
import matplotlib.pyplot as plt


class GreyWolfOptimizer:
    def __init__(self, objective_function, dim, lb, ub, population_size=30, max_iter=200):
        """
        初始化灰狼优化器
        
        参数:
            objective_function: 目标函数（适应度函数），用于评估解的质量
            dim: 问题维度（决策变量数量）
            lb: 下界（可以是标量或向量）
            ub: 上界（可以是标量或向量）
            population_size: 种群大小（灰狼数量）
            max_iter: 最大迭代次数
        """
        self.objective_function = objective_function
        self.dim = dim  # 维度
        
        # 确保上下界是向量形式
        if np.isscalar(lb):
            self.lb = np.ones(dim) * lb
        else:
            self.lb = np.array(lb)
        
        if np.isscalar(ub):
            self.ub = np.ones(dim) * ub
        else:
            self.ub = np.array(ub)
            
        self.population_size = population_size  # 种群大小
        self.max_iter = max_iter  # 最大迭代次数
        
        # 初始化种群
        self.positions = np.zeros((population_size, dim))  # 灰狼位置
        self.fitness = np.zeros(population_size)  # 适应度值
        
        # 初始化Alpha, Beta和Delta狼的位置和适应度
        self.alpha_pos = np.zeros(dim)
        self.alpha_score = float('inf')
        
        self.beta_pos = np.zeros(dim)
        self.beta_score = float('inf')
        
        self.delta_pos = np.zeros(dim)
        self.delta_score = float('inf')
        
        # 收敛历史记录
        self.convergence_curve = np.zeros(max_iter)
    
    def initialize_population(self):
        """
        初始化灰狼种群
        """
        # 在搜索空间中随机生成灰狼位置
        self.positions = np.random.uniform(0, 1, (self.population_size, self.dim)) * (self.ub - self.lb) + self.lb
        
        # 计算每个灰狼的适应度
        for i in range(self.population_size):
            self.fitness[i] = self.objective_function(self.positions[i, :])
            
            # 更新Alpha, Beta和Delta狼
            if self.fitness[i] < self.alpha_score:
                self.alpha_score = self.fitness[i]
                self.alpha_pos = self.positions[i, :].copy()
                
            if (self.fitness[i] > self.alpha_score) and (self.fitness[i] < self.beta_score):
                self.beta_score = self.fitness[i]
                self.beta_pos = self.positions[i, :].copy()
                
            if (self.fitness[i] > self.alpha_score) and (self.fitness[i] > self.beta_score) and (self.fitness[i] < self.delta_score):
                self.delta_score = self.fitness[i]
                self.delta_pos = self.positions[i, :].copy()
    
    def optimize(self):
        """
        执行优化过程
        
        返回:
            最优解（Alpha狼位置）和对应的适应度值
        """
        # 初始化种群
        self.initialize_population()
        
        # 迭代优化
        for iter in range(self.max_iter):
            # 更新a参数（从2线性减少到0）
            a = 2 - iter * (2 / self.max_iter)
            
            # 更新每个灰狼的位置
            for i in range(self.population_size):
                # 对每个维度更新位置
                for j in range(self.dim):
                    # Alpha狼引导的位置更新
                    r1 = np.random.rand()
                    r2 = np.random.rand()
                    
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    
                    D_alpha = abs(C1 * self.alpha_pos[j] - self.positions[i, j])
                    X1 = self.alpha_pos[j] - A1 * D_alpha
                    
                    # Beta狼引导的位置更新
                    r1 = np.random.rand()
                    r2 = np.random.rand()
                    
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    
                    D_beta = abs(C2 * self.beta_pos[j] - self.positions[i, j])
                    X2 = self.beta_pos[j] - A2 * D_beta
                    
                    # Delta狼引导的位置更新
                    r1 = np.random.rand()
                    r2 = np.random.rand()
                    
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    
                    D_delta = abs(C3 * self.delta_pos[j] - self.positions[i, j])
                    X3 = self.delta_pos[j] - A3 * D_delta
                    
                    # 根据Alpha, Beta和Delta的位置更新当前狼的位置
                    self.positions[i, j] = (X1 + X2 + X3) / 3
                
                # 确保位置在搜索空间内
                self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)
                
                # 更新适应度
                self.fitness[i] = self.objective_function(self.positions[i])
                
                # 更新Alpha, Beta, Delta
                if self.fitness[i] < self.alpha_score:
                    self.alpha_score = self.fitness[i]
                    self.alpha_pos = self.positions[i].copy()
                
                if (self.fitness[i] > self.alpha_score) and (self.fitness[i] < self.beta_score):
                    self.beta_score = self.fitness[i]
                    self.beta_pos = self.positions[i].copy()
                
                if (self.fitness[i] > self.alpha_score) and (self.fitness[i] > self.beta_score) and (self.fitness[i] < self.delta_score):
                    self.delta_score = self.fitness[i]
                    self.delta_pos = self.positions[i].copy()
            
            # 记录当前迭代的最优解（Alpha）
            self.convergence_curve[iter] = self.alpha_score
            
            # 打印迭代信息
            if (iter + 1) % 10 == 0:
                print(f"迭代 {iter+1}: 最优值 = {self.alpha_score}")
        
        return self.alpha_pos, self.alpha_score
    
    def plot_convergence(self):
        """
        绘制收敛曲线
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, self.max_iter + 1), self.convergence_curve, 'b', linewidth=2)
        plt.title('灰狼优化算法收敛曲线')
        plt.xlabel('迭代次数')
        plt.ylabel('目标函数值')
        plt.grid(True)
        plt.show()


# 测试函数示例：Sphere函数（简单的凸函数，最小值为0，位于原点）
def sphere_function(x):
    """
    Sphere测试函数
    f(x) = sum(x_i^2)
    全局最小值: f(0,...,0) = 0
    """
    return np.sum(np.square(x))


# 演示如何使用灰狼优化算法
if __name__ == "__main__":
    # 设置问题参数
    dim = 10  # 问题维度
    lb = -100  # 下界
    ub = 100   # 上界
    
    # 创建优化器
    gwo = GreyWolfOptimizer(
        objective_function=sphere_function,
        dim=dim,
        lb=lb,
        ub=ub,
        population_size=30,
        max_iter=200
    )
    
    # 执行优化
    best_position, best_score = gwo.optimize()
    
    # 打印结果
    print("\n优化结果:")
    print(f"最优值: {best_score}")
    print(f"最优解: {best_position}")
    
    # 绘制收敛曲线
    gwo.plot_convergence()


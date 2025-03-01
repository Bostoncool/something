import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

class DifferentialEvolution:
    """
    差分进化算法(Differential Evolution, DE)实现
    
    差分进化是一种基于种群的优化算法，特别适合于连续优化问题。
    它通过变异、交叉和选择操作来不断改进解决方案。
    """
    
    def __init__(self, objective_func, bounds, pop_size=50, F=0.5, CR=0.7, max_iter=100):
        """
        初始化差分进化算法
        
        参数:
            objective_func: 目标函数 (适应度函数)，用于评估个体的质量
            bounds: 各维度的取值范围，形式为[(min1, max1), (min2, max2), ...]
            pop_size: 种群大小
            F: 缩放因子，用于控制变异的程度
            CR: 交叉概率
            max_iter: 最大迭代次数
        """
        self.objective_func = objective_func  # 目标函数
        self.bounds = bounds  # 各维度的取值范围
        self.dimensions = len(bounds)  # 问题维度
        self.pop_size = pop_size  # 种群大小
        self.F = F  # 缩放因子
        self.CR = CR  # 交叉概率
        self.max_iter = max_iter  # 最大迭代次数
        
        # 记录算法运行历史
        self.history = {
            'best_fitness': [],
            'avg_fitness': [],
            'best_solution': None
        }
    
    def initialize_population(self):
        """
        初始化种群
        
        创建一个随机种群，每个个体的各维度值都在指定范围内
        
        返回:
            population: 初始化的种群，形状为 (pop_size, dimensions)
        """
        population = np.zeros((self.pop_size, self.dimensions))
        
        # 对每个维度，生成在其范围内的随机值
        for i in range(self.dimensions):
            lower_bound, upper_bound = self.bounds[i]
            population[:, i] = np.random.uniform(lower_bound, upper_bound, size=self.pop_size)
            
        return population
    
    def evaluate_population(self, population):
        """
        评估种群中每个个体的适应度
        
        参数:
            population: 当前种群
            
        返回:
            fitness_values: 每个个体的适应度值
        """
        fitness_values = np.zeros(self.pop_size)
        
        for i in range(self.pop_size):
            fitness_values[i] = self.objective_func(population[i])
            
        return fitness_values
    
    def mutation(self, population, fitness_values):
        """
        差分变异操作
        
        对每个个体，随机选择3个不同的其他个体，然后用它们创建一个变异向量
        
        参数:
            population: 当前种群
            fitness_values: 每个个体的适应度值
            
        返回:
            mutants: 变异后的种群
        """
        mutants = np.zeros_like(population)
        
        for i in range(self.pop_size):
            # 随机选择3个不同的个体索引，且都不等于当前索引 i
            candidates = list(range(self.pop_size))
            candidates.remove(i)
            random_indices = np.random.choice(candidates, 3, replace=False)
            a, b, c = random_indices
            
            # 创建变异向量: x_a + F * (x_b - x_c)
            mutant_vector = population[a] + self.F * (population[b] - population[c])
            
            # 确保变异向量在边界范围内
            for j in range(self.dimensions):
                lower_bound, upper_bound = self.bounds[j]
                mutant_vector[j] = np.clip(mutant_vector[j], lower_bound, upper_bound)
                
            mutants[i] = mutant_vector
            
        return mutants
    
    def crossover(self, population, mutants):
        """
        二项式交叉操作
        
        对每个个体的每个维度，以一定概率(CR)从变异向量获取值，否则保持原值
        
        参数:
            population: 当前种群
            mutants: 变异后的种群
            
        返回:
            trials: 交叉后的种群
        """
        trials = np.zeros_like(population)
        
        for i in range(self.pop_size):
            # 确保至少有一个维度发生交叉
            j_rand = np.random.randint(0, self.dimensions)
            
            for j in range(self.dimensions):
                # 以概率CR使用变异向量的值，或者在j_rand处强制使用
                if np.random.random() < self.CR or j == j_rand:
                    trials[i, j] = mutants[i, j]
                else:
                    trials[i, j] = population[i, j]
                    
        return trials
    
    def selection(self, population, trials, fitness_values):
        """
        选择操作
        
        如果试验向量优于当前个体，则替换当前个体
        
        参数:
            population: 当前种群
            trials: 试验种群
            fitness_values: 当前种群的适应度值
            
        返回:
            new_population: 选择后的新种群
            new_fitness_values: 新种群的适应度值
        """
        new_population = np.copy(population)
        new_fitness_values = np.copy(fitness_values)
        
        for i in range(self.pop_size):
            trial_fitness = self.objective_func(trials[i])
            
            # 如果试验向量更好（适应度值更小），则替换
            if trial_fitness < fitness_values[i]:
                new_population[i] = trials[i]
                new_fitness_values[i] = trial_fitness
                
        return new_population, new_fitness_values
    
    def optimize(self, verbose=True):
        """
        运行差分进化算法的优化过程
        
        参数:
            verbose: 是否打印优化过程信息
            
        返回:
            best_solution: 找到的最佳解
            best_fitness: 最佳解的适应度值
        """
        start_time = time.time()
        
        # 初始化种群
        population = self.initialize_population()
        fitness_values = self.evaluate_population(population)
        
        # 获取当前最佳个体
        best_idx = np.argmin(fitness_values)
        best_solution = population[best_idx]
        best_fitness = fitness_values[best_idx]
        
        # 记录初始状态
        self.history['best_fitness'].append(best_fitness)
        self.history['avg_fitness'].append(np.mean(fitness_values))
        
        # 主循环
        for iteration in range(self.max_iter):
            # 变异
            mutants = self.mutation(population, fitness_values)
            
            # 交叉
            trials = self.crossover(population, mutants)
            
            # 选择
            population, fitness_values = self.selection(population, trials, fitness_values)
            
            # 更新最佳个体
            current_best_idx = np.argmin(fitness_values)
            current_best_fitness = fitness_values[current_best_idx]
            
            if current_best_fitness < best_fitness:
                best_solution = population[current_best_idx]
                best_fitness = current_best_fitness
            
            # 记录历史
            self.history['best_fitness'].append(best_fitness)
            self.history['avg_fitness'].append(np.mean(fitness_values))
            
            # 打印当前状态
            if verbose and (iteration + 1) % 10 == 0:
                print(f"迭代 {iteration + 1}/{self.max_iter}: 最佳适应度 = {best_fitness:.6f}")
        
        elapsed_time = time.time() - start_time
        
        # 保存最佳解
        self.history['best_solution'] = best_solution
        
        if verbose:
            print(f"\n优化完成!")
            print(f"最佳解: {best_solution}")
            print(f"最佳适应度: {best_fitness:.6f}")
            print(f"运行时间: {elapsed_time:.2f} 秒")
        
        return best_solution, best_fitness
    
    def plot_convergence(self):
        """
        绘制收敛曲线
        显示算法的收敛性能
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.history['best_fitness'])), self.history['best_fitness'], 'b-', label='最佳适应度')
        plt.plot(range(len(self.history['avg_fitness'])), self.history['avg_fitness'], 'r-', label='平均适应度')
        plt.xlabel('迭代次数')
        plt.ylabel('适应度值')
        plt.title('差分进化算法收敛曲线')
        plt.legend()
        plt.grid(True)
        plt.show()


# 定义几个常用的测试函数

def sphere_function(x):
    """
    球函数 (Sphere Function)
    全局最小值: f(0,...,0) = 0
    
    适合作为简单的测试函数
    """
    return np.sum(x**2)

def rosenbrock_function(x):
    """
    Rosenbrock函数
    全局最小值: f(1,...,1) = 0
    
    这是一个经典的优化测试函数，很难优化因为它有一个狭长的山谷形状
    """
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)

def rastrigin_function(x):
    """
    Rastrigin函数
    全局最小值: f(0,...,0) = 0
    
    这是一个有多个局部最小值的函数，用于测试算法跳出局部最优的能力
    """
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

def plot_3d_function(func, bounds, points=50):
    """
    绘制3D函数图
    
    参数:
        func: 要绘制的函数
        bounds: 函数的边界，格式为[(x_min, x_max), (y_min, y_max)]
        points: 每个维度上的点数
    """
    x = np.linspace(bounds[0][0], bounds[0][1], points)
    y = np.linspace(bounds[1][0], bounds[1][1], points)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    for i in range(points):
        for j in range(points):
            Z[i, j] = func(np.array([X[i, j], Y[i, j]]))
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    if func.__name__ == 'sphere_function':
        ax.set_title('球函数 (Sphere Function)')
    elif func.__name__ == 'rosenbrock_function':
        ax.set_title('Rosenbrock函数')
    elif func.__name__ == 'rastrigin_function':
        ax.set_title('Rastrigin函数')
    else:
        ax.set_title('3D函数图')
    
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


# 示例：使用差分进化算法最小化Rastrigin函数
if __name__ == "__main__":
    print("差分进化算法(DE)示例 - 最小化Rastrigin函数")
    print("-" * 60)
    
    # 问题设置
    dimensions = 10  # 使用10维Rastrigin函数
    bounds = [(-5.12, 5.12)] * dimensions  # Rastrigin函数的标准范围
    
    # 创建DE优化器
    de = DifferentialEvolution(
        objective_func=rastrigin_function,
        bounds=bounds,
        pop_size=50,
        F=0.8,
        CR=0.7,
        max_iter=200
    )
    
    # 运行优化
    best_solution, best_fitness = de.optimize(verbose=True)
    
    # 绘制收敛曲线
    de.plot_convergence()
    
    # 如果是2D问题，绘制函数图
    if dimensions == 2:
        plot_3d_function(rastrigin_function, bounds)
        
        # 在图上标记找到的最优点
        plt.figure(figsize=(8, 6))
        x = np.linspace(bounds[0][0], bounds[0][1], 100)
        y = np.linspace(bounds[1][0], bounds[1][1], 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        for i in range(100):
            for j in range(100):
                Z[i, j] = rastrigin_function(np.array([X[i, j], Y[i, j]]))
        
        plt.contourf(X, Y, Z, 50, cmap='viridis')
        plt.colorbar()
        plt.plot(best_solution[0], best_solution[1], 'ro', markersize=10)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Rastrigin函数等高线图和找到的最优点')
        plt.show()
    
    # 多维问题的分量图
    if dimensions > 2:
        plt.figure(figsize=(12, 6))
        plt.bar(range(dimensions), best_solution)
        plt.xlabel('维度')
        plt.ylabel('数值')
        plt.title('最优解的各维度数值')
        plt.grid(True)
        plt.show()

    print("\n试试其他测试函数:")
    print("1. 修改main函数中的objective_func来测试不同的函数")
    print("2. 调整算法参数（F, CR, 种群大小）以观察对性能的影响")
    print("3. 尝试提高维度数量来增加问题难度")

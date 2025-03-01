import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class PSO:
    """
    粒子群优化(Particle Swarm Optimization)算法实现
    """
    def __init__(self, fitness_func, dim, pop_size=50, max_iter=200, lb=None, ub=None, w=0.8, c1=2, c2=2):
        """
        初始化PSO算法参数
        
        参数:
            fitness_func: 适应度函数(目标函数)，算法将寻找其最小值
            dim: 问题的维度
            pop_size: 种群大小(粒子数量)
            max_iter: 最大迭代次数
            lb: 搜索空间的下界，可以是标量或数组
            ub: 搜索空间的上界，可以是标量或数组
            w: 惯性权重，控制粒子保持原来速度的程度
            c1: 认知系数，控制粒子向自身历史最优位置移动的程度
            c2: 社会系数，控制粒子向群体历史最优位置移动的程度
        """
        self.fitness_func = fitness_func  # 适应度函数
        self.dim = dim                    # 维度
        self.pop_size = pop_size          # 种群大小
        self.max_iter = max_iter          # 最大迭代次数
        
        # 设置搜索空间边界
        if lb is None:
            self.lb = -100 * np.ones(dim)
        else:
            self.lb = np.array([lb] * dim) if np.isscalar(lb) else np.array(lb)
            
        if ub is None:
            self.ub = 100 * np.ones(dim)
        else:
            self.ub = np.array([ub] * dim) if np.isscalar(ub) else np.array(ub)
            
        # 算法参数
        self.w = w      # 惯性权重
        self.c1 = c1    # 认知系数
        self.c2 = c2    # 社会系数
        
        # 初始化粒子的位置和速度
        self.X = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))  # 粒子位置
        self.V = np.random.uniform(-1, 1, (self.pop_size, self.dim))             # 粒子速度
        
        # 初始化每个粒子的最佳位置和适应度
        self.pbest = self.X.copy()  # 个体最佳位置
        self.pbest_fitness = np.array([self.fitness_func(p) for p in self.X])  # 个体最佳适应度
        
        # 初始化全局最佳位置和适应度
        gbest_idx = np.argmin(self.pbest_fitness)
        self.gbest = self.pbest[gbest_idx].copy()  # 全局最佳位置
        self.gbest_fitness = self.pbest_fitness[gbest_idx]  # 全局最佳适应度
        
        # 用于记录每次迭代的最佳适应度
        self.history = []
        
        # 用于可视化
        self.all_positions = []
    
    def optimize(self, verbose=True, animate=False):
        """
        运行PSO优化算法
        
        参数:
            verbose: 是否打印每次迭代的信息
            animate: 是否生成动画 (仅适用于2D问题)
        
        返回:
            全局最佳位置和对应的适应度值
        """
        # 存储初始位置用于可视化
        if animate and self.dim == 2:
            self.all_positions.append(self.X.copy())
        
        # 迭代优化
        for i in range(self.max_iter):
            # 更新每个粒子
            for j in range(self.pop_size):
                # 生成随机数
                r1 = np.random.random(self.dim)
                r2 = np.random.random(self.dim)
                
                # 更新速度
                self.V[j] = (self.w * self.V[j] + 
                            self.c1 * r1 * (self.pbest[j] - self.X[j]) + 
                            self.c2 * r2 * (self.gbest - self.X[j]))
                
                # 更新位置
                self.X[j] = self.X[j] + self.V[j]
                
                # 边界处理：将粒子位置限制在搜索空间内
                self.X[j] = np.clip(self.X[j], self.lb, self.ub)
                
                # 计算新位置的适应度
                fitness = self.fitness_func(self.X[j])
                
                # 更新个体最佳
                if fitness < self.pbest_fitness[j]:
                    self.pbest[j] = self.X[j].copy()
                    self.pbest_fitness[j] = fitness
                    
                    # 更新全局最佳
                    if fitness < self.gbest_fitness:
                        self.gbest = self.X[j].copy()
                        self.gbest_fitness = fitness
            
            # 记录当前迭代的最佳适应度
            self.history.append(self.gbest_fitness)
            
            # 存储当前所有粒子位置用于可视化
            if animate and self.dim == 2:
                self.all_positions.append(self.X.copy())
            
            # 打印当前迭代信息
            if verbose and (i + 1) % 10 == 0:
                print(f"迭代 {i+1}/{self.max_iter}, 最佳适应度: {self.gbest_fitness:.6f}")
        
        if verbose:
            print(f"优化完成! 最佳适应度: {self.gbest_fitness:.6f}")
            print(f"最佳位置: {self.gbest}")
        
        return self.gbest, self.gbest_fitness
    
    def plot_convergence(self):
        """绘制收敛曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.history) + 1), self.history)
        plt.title('PSO算法收敛曲线')
        plt.xlabel('迭代次数')
        plt.ylabel('最佳适应度值')
        plt.grid(True)
        plt.show()
    
    def animate_optimization(self, contour_func=None, interval=20):
        """
        创建优化过程的动画 (仅适用于2D问题)
        
        参数:
            contour_func: 用于创建轮廓图的函数，如果为None，则使用适应度函数
            interval: 动画帧之间的时间间隔(毫秒)
        """
        if self.dim != 2:
            print("动画仅支持2D问题")
            return
        
        if not self.all_positions:
            print("请先运行优化算法并设置animate=True")
            return
        
        func = contour_func if contour_func is not None else self.fitness_func
        
        # 创建网格点
        x = np.linspace(self.lb[0], self.ub[0], 100)
        y = np.linspace(self.lb[1], self.ub[1], 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        # 计算每个网格点的函数值
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = func(np.array([X[i, j], Y[i, j]]))
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 8))
        contour = ax.contourf(X, Y, Z, 50, cmap='viridis', alpha=0.5)
        particles, = ax.plot([], [], 'ro', ms=4, alpha=0.7)
        title = ax.set_title('')
        fig.colorbar(contour)
        
        ax.set_xlim(self.lb[0], self.ub[0])
        ax.set_ylim(self.lb[1], self.ub[1])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        def init():
            particles.set_data([], [])
            title.set_text('')
            return particles, title
        
        def animate(i):
            positions = self.all_positions[i]
            particles.set_data(positions[:, 0], positions[:, 1])
            title.set_text(f'迭代 {i}/{len(self.all_positions)-1}')
            return particles, title
        
        anim = FuncAnimation(fig, animate, frames=len(self.all_positions),
                              init_func=init, blit=True, interval=interval)
        plt.close()  # 防止显示静态图形
        return anim


# 示例：使用PSO算法寻找Rastrigin函数的最小值
def rastrigin(x):
    """
    Rastrigin函数，一个常用的测试函数，有多个局部最小值
    全局最小值为f(0,0,...,0) = 0
    """
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

def rosenbrock(x):
    """
    Rosenbrock函数，另一个常用的测试函数，有一个狭长的山谷
    全局最小值为f(1,1,...,1) = 0
    """
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def ackley(x):
    """
    Ackley函数，具有多个局部最小值和一个全局最小值
    全局最小值为f(0,0,...,0) = 0
    """
    a, b, c = 20, 0.2, 2*np.pi
    d = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(c*x))
    term1 = -a * np.exp(-b * np.sqrt(sum1/d))
    term2 = -np.exp(sum2/d)
    return term1 + term2 + a + np.exp(1)

if __name__ == "__main__":
    # 设置随机种子以获得可重复的结果
    np.random.seed(42)
    
    print("=== PSO算法求解Rastrigin函数最小值 ===")
    # 创建PSO实例，求解2维Rastrigin函数
    dim = 2  # 问题维度
    pso = PSO(
        fitness_func=rastrigin,  # 适应度函数
        dim=dim,                 # 维度
        pop_size=50,             # 粒子数量
        max_iter=100,            # 最大迭代次数
        lb=-5.12,                # 搜索空间下界
        ub=5.12,                 # 搜索空间上界
        w=0.729,                 # 惯性权重
        c1=1.49445,              # 认知系数
        c2=1.49445               # 社会系数
    )
    
    # 运行优化算法
    best_position, best_fitness = pso.optimize(verbose=True, animate=True)
    
    # 绘制收敛曲线
    pso.plot_convergence()
    
    # 创建并保存优化过程的动画
    anim = pso.animate_optimization()
    
    # 显示动画
    from IPython.display import HTML
    HTML(anim.to_jshtml())
    
    # 或者保存动画为GIF
    # anim.save('pso_rastrigin.gif', writer='pillow', fps=10)
    
    print("\n=== PSO算法求解Rosenbrock函数最小值 ===")
    # 创建PSO实例，求解2维Rosenbrock函数
    pso_rb = PSO(
        fitness_func=rosenbrock,
        dim=2,
        pop_size=50,
        max_iter=150,
        lb=-2.048,
        ub=2.048,
        w=0.729,
        c1=1.49445,
        c2=1.49445
    )
    
    # 运行优化算法
    best_position_rb, best_fitness_rb = pso_rb.optimize(verbose=True)
    
    # 绘制收敛曲线
    pso_rb.plot_convergence() 
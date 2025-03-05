#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
遗传算法(Genetic Algorithm)实现
目标：最大化函数 f(x) = x sin(10πx) + 2.0，其中 x ∈ [0, 2]
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import math

# 设置随机种子，保证结果可重现
np.random.seed(42)
random.seed(42)

class GeneticAlgorithm:
    def __init__(self, 
                 pop_size=50,           # 种群大小
                 chromosome_length=20,   # 染色体长度（二进制编码位数）
                 crossover_rate=0.8,     # 交叉概率
                 mutation_rate=0.1,      # 变异概率
                 max_generations=100,    # 最大迭代次数
                 x_bounds=(0, 2)):       # x的取值范围
        """
        初始化遗传算法参数
        
        参数:
            pop_size: 种群大小
            chromosome_length: 染色体长度（二进制编码位数）
            crossover_rate: 交叉概率
            mutation_rate: 变异概率
            max_generations: 最大迭代次数
            x_bounds: x的取值范围，元组(min, max)
        """
        self.pop_size = pop_size
        self.chromosome_length = chromosome_length
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.x_bounds = x_bounds
        
        # 记录每代的最佳适应度和平均适应度
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_x_history = []
        
        # 初始化种群
        self.population = self.initialize_population()
    
    def initialize_population(self):
        """
        种群初始化：随机生成二进制编码的初始种群
        
        返回:
            初始种群，二维数组，每行是一个个体（染色体）
        """
        # 生成随机的二进制种群，每个个体是一个长度为chromosome_length的二进制串
        return np.random.randint(0, 2, size=(self.pop_size, self.chromosome_length))
    
    def binary_to_decimal(self, binary):
        """
        将二进制编码转换为十进制数值
        
        参数:
            binary: 二进制编码数组
            
        返回:
            对应的十进制数值，映射到x_bounds范围内
        """
        # 将二进制数组转换为字符串
        binary_str = ''.join(map(str, binary))
        # 将二进制字符串转换为十进制数值
        decimal = int(binary_str, 2)
        # 将十进制数值映射到x_bounds范围内
        x_min, x_max = self.x_bounds
        x = x_min + decimal * (x_max - x_min) / (2**self.chromosome_length - 1)
        return x
    
    def fitness_function(self, x):
        """
        适应度函数：计算目标函数 f(x) = x sin(10πx) + 2.0 的值
        
        参数:
            x: 自变量x的值
            
        返回:
            适应度值（目标函数值）
        """
        return x * math.sin(10 * math.pi * x) + 2.0
    
    def calculate_fitness(self, population):
        """
        计算种群中所有个体的适应度
        
        参数:
            population: 种群，二维数组
            
        返回:
            适应度数组，每个元素对应一个个体的适应度
        """
        fitness = np.zeros(len(population))
        for i, individual in enumerate(population):
            # 将二进制编码转换为十进制数值
            x = self.binary_to_decimal(individual)
            # 计算适应度
            fitness[i] = self.fitness_function(x)
        return fitness
    
    def select(self, population, fitness):
        """
        选择操作：使用轮盘赌（Roulette Wheel）选择法
        
        参数:
            population: 当前种群
            fitness: 适应度数组
            
        返回:
            选择后的新种群
        """
        # 计算选择概率
        total_fitness = np.sum(fitness)
        selection_prob = fitness / total_fitness if total_fitness > 0 else np.ones(len(fitness)) / len(fitness)
        
        # 计算累积概率
        cumulative_prob = np.cumsum(selection_prob)
        
        # 轮盘赌选择
        new_population = np.zeros((self.pop_size, self.chromosome_length), dtype=int)
        for i in range(self.pop_size):
            # 生成[0,1)之间的随机数
            r = random.random()
            # 找到第一个累积概率大于r的个体
            for j in range(len(cumulative_prob)):
                if r <= cumulative_prob[j]:
                    new_population[i] = population[j].copy()
                    break
        
        return new_population
    
    def crossover(self, population):
        """
        交叉操作：单点交叉
        
        参数:
            population: 当前种群
            
        返回:
            交叉后的新种群
        """
        new_population = population.copy()
        
        # 随机配对进行交叉
        for i in range(0, self.pop_size, 2):
            # 如果达到交叉概率，则进行交叉操作
            if random.random() < self.crossover_rate and i+1 < self.pop_size:
                # 随机选择交叉点
                crossover_point = random.randint(1, self.chromosome_length - 1)
                
                # 交换交叉点后的基因
                temp = new_population[i, crossover_point:].copy()
                new_population[i, crossover_point:] = new_population[i+1, crossover_point:].copy()
                new_population[i+1, crossover_point:] = temp
        
        return new_population
    
    def mutate(self, population):
        """
        变异操作：随机变异
        
        参数:
            population: 当前种群
            
        返回:
            变异后的新种群
        """
        new_population = population.copy()
        
        # 对每个个体的每个基因进行变异操作
        for i in range(self.pop_size):
            for j in range(self.chromosome_length):
                # 如果达到变异概率，则进行变异操作
                if random.random() < self.mutation_rate:
                    # 基因翻转（0变1，1变0）
                    new_population[i, j] = 1 - new_population[i, j]
        
        return new_population
    
    def evolve(self):
        """
        进化过程：包括选择、交叉和变异操作
        """
        # 计算当前种群的适应度
        fitness = self.calculate_fitness(self.population)
        
        # 记录当前代的最佳适应度和平均适应度
        best_idx = np.argmax(fitness)
        best_fitness = fitness[best_idx]
        avg_fitness = np.mean(fitness)
        best_x = self.binary_to_decimal(self.population[best_idx])
        
        self.best_fitness_history.append(best_fitness)
        self.avg_fitness_history.append(avg_fitness)
        self.best_x_history.append(best_x)
        
        # 选择操作
        selected_population = self.select(self.population, fitness)
        
        # 交叉操作
        crossover_population = self.crossover(selected_population)
        
        # 变异操作
        mutated_population = self.mutate(crossover_population)
        
        # 更新种群
        self.population = mutated_population
    
    def run(self):
        """
        运行遗传算法
        
        返回:
            最优解x和对应的适应度值
        """
        print("开始遗传算法优化...")
        print(f"参数设置: 种群大小={self.pop_size}, 染色体长度={self.chromosome_length}, "
              f"交叉概率={self.crossover_rate}, 变异概率={self.mutation_rate}, "
              f"最大迭代次数={self.max_generations}")
        
        # 进化循环
        for generation in range(self.max_generations):
            self.evolve()
            
            # 打印当前代的信息
            if (generation + 1) % 10 == 0 or generation == 0:
                print(f"第 {generation+1} 代: 最佳适应度 = {self.best_fitness_history[-1]:.6f}, "
                      f"平均适应度 = {self.avg_fitness_history[-1]:.6f}, "
                      f"最佳 x = {self.best_x_history[-1]:.6f}")
        
        # 找到最优解
        best_generation = np.argmax(self.best_fitness_history)
        best_fitness = self.best_fitness_history[best_generation]
        best_x = self.best_x_history[best_generation]
        
        print("\n优化完成!")
        print(f"最优解: x = {best_x:.6f}")
        print(f"最优适应度: f(x) = {best_fitness:.6f}")
        print(f"在第 {best_generation+1} 代获得最优解")
        
        return best_x, best_fitness
    
    def plot_results(self):
        """
        绘制优化结果图表
        """
        # 创建一个包含两个子图的图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # 绘制适应度变化曲线
        generations = range(1, len(self.best_fitness_history) + 1)
        ax1.plot(generations, self.best_fitness_history, 'r-', label='最佳适应度')
        ax1.plot(generations, self.avg_fitness_history, 'b-', label='平均适应度')
        ax1.set_xlabel('代数')
        ax1.set_ylabel('适应度')
        ax1.set_title('适应度随代数的变化')
        ax1.legend()
        ax1.grid(True)
        
        # 绘制目标函数曲线和最优解
        x = np.linspace(self.x_bounds[0], self.x_bounds[1], 1000)
        y = [self.fitness_function(xi) for xi in x]
        
        best_generation = np.argmax(self.best_fitness_history)
        best_x = self.best_x_history[best_generation]
        best_fitness = self.best_fitness_history[best_generation]
        
        ax2.plot(x, y, 'b-', label='目标函数 f(x) = x sin(10πx) + 2.0')
        ax2.scatter([best_x], [best_fitness], color='red', s=100, marker='*', label=f'最优解: x = {best_x:.6f}')
        ax2.set_xlabel('x')
        ax2.set_ylabel('f(x)')
        ax2.set_title('目标函数和最优解')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('genetic_algorithm_results.png', dpi=300)
        plt.show()


def main():
    """
    主函数：创建遗传算法实例并运行
    """
    # 创建遗传算法实例 此处参数可以调整
    ga = GeneticAlgorithm(
        pop_size=100,           # 种群大小
        chromosome_length=22,   # 染色体长度
        crossover_rate=0.8,     # 交叉概率
        mutation_rate=0.1,      # 变异概率
        max_generations=200,    # 最大迭代次数
        x_bounds=(0, 2)         # x的取值范围
    )
    
    # 运行遗传算法
    best_x, best_fitness = ga.run()
    
    # 绘制结果
    ga.plot_results()


if __name__ == "__main__":
    main()

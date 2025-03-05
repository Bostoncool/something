import numpy as np
import matplotlib.pyplot as plt
import heapq
from matplotlib.colors import ListedColormap

class Node:
    """表示网格中的一个节点"""
    def __init__(self, position, parent=None):
        self.position = position  # 节点位置 (x, y)
        self.parent = parent      # 父节点
        
        self.g = 0  # 从起点到当前节点的代价
        self.h = 0  # 从当前节点到终点的估计代价（启发函数）
        self.f = 0  # 总代价 f = g + h
    
    def __eq__(self, other):
        return self.position == other.position
    
    def __lt__(self, other):
        return self.f < other.f
    
    def __hash__(self):
        return hash(self.position)

def heuristic(node, goal):
    """计算启发函数值 - 使用曼哈顿距离"""
    return abs(node.position[0] - goal.position[0]) + abs(node.position[1] - goal.position[1])

def get_neighbors(node, grid):
    """获取节点的相邻节点"""
    neighbors = []
    grid_shape = grid.shape
    
    # 定义可能的移动方向（上、右、下、左）
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    
    for direction in directions:
        new_position = (node.position[0] + direction[0], node.position[1] + direction[1])
        
        # 检查新位置是否在网格范围内且不是障碍物
        if (0 <= new_position[0] < grid_shape[0] and 
            0 <= new_position[1] < grid_shape[1] and 
            grid[new_position] == 0):  # 0表示可通行
            
            neighbors.append(Node(new_position, node))
    
    return neighbors

def a_star_search(grid, start, end):
    """
    使用A*算法寻找从起点到终点的路径
    
    参数:
    grid -- 二维网格，0表示可通行，1表示障碍物
    start -- 起点坐标 (x, y)
    end -- 终点坐标 (x, y)
    
    返回:
    path -- 找到的路径，如果没有路径返回None
    visited -- 访问过的节点
    """
    # 创建起点和终点节点
    start_node = Node(start)
    end_node = Node(end)
    
    # 初始化开放列表和关闭列表
    open_list = []
    closed_set = set()
    
    # 将起点加入开放列表
    heapq.heappush(open_list, start_node)
    
    # 记录已访问的节点
    visited = []
    
    # 开始搜索
    while open_list:
        # 获取f值最小的节点
        current_node = heapq.heappop(open_list)
        closed_set.add(current_node.position)
        visited.append(current_node.position)
        
        # 如果找到目标，构建路径并返回
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1], visited  # 返回反转后的路径
        
        # 获取当前节点的相邻节点
        neighbors = get_neighbors(current_node, grid)
        
        for neighbor in neighbors:
            # 如果相邻节点已经在关闭列表中，跳过
            if neighbor.position in closed_set:
                continue
            
            # 计算从起点经过当前节点到相邻节点的代价
            neighbor.g = current_node.g + 1
            neighbor.h = heuristic(neighbor, end_node)
            neighbor.f = neighbor.g + neighbor.h
            
            # 检查是否已在开放列表中并且具有更低的f值
            add_to_open = True
            for i, open_node in enumerate(open_list):
                if neighbor == open_node and neighbor.g >= open_node.g:
                    add_to_open = False
                    break
            
            if add_to_open:
                heapq.heappush(open_list, neighbor)
    
    # 如果开放列表为空仍未找到路径，则返回None
    return None, visited

def visualize_path(grid, start, end, path=None, visited=None):
    """可视化网格和找到的路径"""
    # 创建绘图网格
    plot_grid = grid.copy()
    
    # 标记访问过的节点
    if visited:
        for position in visited:
            plot_grid[position] = 2  # 2表示访问过的节点
    
    # 标记路径
    if path:
        for position in path:
            plot_grid[position] = 3  # 3表示路径
    
    # 标记起点和终点
    plot_grid[start] = 4  # 4表示起点
    plot_grid[end] = 5    # 5表示终点
    
    # 设置颜色映射
    colors = ['white', 'black', 'lightblue', 'green', 'red', 'purple']
    cmap = ListedColormap(colors)
    
    # 绘制网格
    plt.figure(figsize=(10, 10))
    plt.imshow(plot_grid, cmap=cmap)
    plt.grid(True, color='gray', linestyle='-', linewidth=0.5)
    plt.xticks(np.arange(-0.5, grid.shape[1], 1), [])
    plt.yticks(np.arange(-0.5, grid.shape[0], 1), [])
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='white', edgecolor='gray', label='可通行'),
        Patch(facecolor='black', edgecolor='gray', label='障碍物'),
        Patch(facecolor='lightblue', edgecolor='gray', label='已访问'),
        Patch(facecolor='green', edgecolor='gray', label='路径'),
        Patch(facecolor='red', edgecolor='gray', label='起点'),
        Patch(facecolor='purple', edgecolor='gray', label='终点')
    ]
    plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
    
    plt.title('A* 算法路径规划')
    plt.tight_layout()
    plt.savefig('a_star_path.png')
    plt.show()

def create_random_grid(size, obstacle_prob=0.3):
    """创建一个随机网格，有一定概率生成障碍物"""
    grid = np.random.choice([0, 1], size=(size, size), p=[1-obstacle_prob, obstacle_prob])
    return grid

def main():
    # 创建一个20x20的随机网格
    grid_size = 20
    grid = create_random_grid(grid_size, obstacle_prob=0.3)
    
    # 设置起点和终点
    start = (0, 0)
    end = (grid_size-1, grid_size-1)
    
    # 确保起点和终点是可通行的
    grid[start] = 0
    grid[end] = 0
    
    print("使用A*算法寻找路径...")
    path, visited = a_star_search(grid, start, end)
    
    if path:
        print(f"找到路径! 长度: {len(path)}")
        print(f"访问节点数: {len(visited)}")
    else:
        print("无法找到路径!")
    
    # 可视化结果
    visualize_path(grid, start, end, path, visited)

if __name__ == "__main__":
    main() 
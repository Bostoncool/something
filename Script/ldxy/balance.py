import sympy as sp

# 定义变量
x = sp.symbols('x')

# 输入比例
r = float(input('输入币商给的比例：'))

# 定义方程
equation = sp.Eq(x / (0.85 * r), 30 + (x - 113) / (0.9 * r))

# 求解方程
solution = sp.solve(equation, x)

# 输出解
print("实际倒币需求超过:", solution,"可以考虑开周卡")
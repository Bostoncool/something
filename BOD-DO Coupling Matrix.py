#以下是“多河段 BOD-DO 耦合矩阵模型”的程序代码
# Q-在断面i处注入河流的流量,m3/s
# L，O-由断面i注入河流的污水的污染物（例如BOD）浓度与溶解氧（DO）浓度,mg/L
# kd-BOD的降解速度常数,/d
# ka-大气复氧速度,/d
# t-由断面i到断面i+1经过时间，d
# Q3-在断面i处引出的流水流量,m3/s
# L20,O20-河流背景的BOD和DO浓度，mg/L
# N-河流断面个数，常数
# 输入题目中变量值，记得用空格隔开
 
import numpy as np
#需要一个numpy库，否则无法使用矩阵运算
 
 
# 计算函数
def calculate(N, Q1, Q, L, O, kd, ka, t, Q3, T, L20, O20):
    Os = 468 / (31.6 + T)  # 饱和氧算法
    Q2 = np.zeros((N, 1))
    
    for i in range(N):
        Q2[i, 0] = Q1[i, 0] - Q3[i, 0] + Q[i, 0]  # 河流Q和BOD的平衡关系，连续性原理
        Q1[i + 1, 0] = Q2[i, 0]  # 由上一个河段流到断面i的河水流量
    
    a = np.zeros((N, 1))
    for i in range(N):
        a[i, 0] = np.exp(-kd[i, 0] * t[i, 0])  # S-P模型中BOD的变化规律，阿尔法α改为了a
    
    A = np.zeros((N, N))
    for j in range(N):
        for i in range(N):
            if j == i:
                A[i, j] = 1  # 对角线元素
            elif j == i - 1:
                A[i, j] = -a[i, 0] * (Q1[i, 0] - Q3[i, 0]) / Q2[i, 0]  # A矩阵计算有效值
            else:
                A[i, j] = 0
    
    B = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                B[i, j] = Q[i, 0] / Q2[i, 0]  # B矩阵计算有效值
            else:
                B[i, j] = 0
    
    g = np.zeros((N, 1))
    g[0, 0] = a[0, 0] * (Q1[0, 0] - Q3[0, 0]) / Q2[0, 0] * L20  # 零维矩阵
    
    y = np.zeros((N, 1))
    for i in range(N):
        y[i, 0] = np.exp(-ka[i, 0] * t[i, 0])
    
    C = np.zeros((N, N))
    for j in range(N):
        for i in range(N):
            if j == i:
                C[i, j] = 1
            elif j == i - 1:
                C[i, j] = (Q1[i, 0] - Q3[i, 0]) / Q2[i, 0] * - y[i, 0]
            else:
                C[i, j] = 0
    
    D = np.zeros((N, N))
    for j in range(N):
        for i in range(N):
            if j == i - 1:
                D[i, j] = (Q1[i, 0] - Q3[i, 0]) / Q2[i, 0] * kd[i, 0] / (ka[i, 0] - kd[i, 0]) * (a[i, 0] - y[i, 0])
            else:
                D[i, j] = 0
    
    f = np.zeros((N, 1))
    for i in range(N):
        f[i, 0] = (Q1[i, 0] - Q3[i, 0]) / Q2[i, 0] * Os * (1 - y[i, 0])
    
    h = np.zeros((N, 1))
    h[0, 0] = (Q1[0, 0] - Q3[0, 0]) / Q2[0, 0] * y[0, 0] * O20 - (Q1[0, 0] - Q3[0, 0]) / Q2[0, 0] * kd[0, 0] / (ka[0, 0] - kd[0, 0]) * (a[0, 0] - y[0, 0]) * L20
    
    U = np.linalg.inv(A).dot(B)  
    # U是BOD对BOD响应矩阵
 
    V = -np.linalg.inv(C).dot(D).dot(np.linalg.inv(A)).dot(B)  
    # V是溶解氧对BOD的响应矩阵
 
    m = np.linalg.inv(A).dot(g)
    # m是BOD对BOD响应矩阵的逆矩阵乘以g
 
    n = np.linalg.inv(C).dot(B).dot(O) + np.linalg.inv(C).dot(f + h) - np.linalg.inv(C).dot(D).dot(np.linalg.inv(A)).dot(g)
    # n是溶解氧对BOD响应矩阵的逆矩阵乘以BOD对BOD响应矩阵的逆矩阵乘以O，再加上溶解氧对BOD响应矩阵的逆矩阵乘以f+h，再减去溶解氧对BOD响应矩阵的逆矩阵乘以BOD对BOD响应矩阵的逆矩阵乘以g
    
    L2 = U.dot(L) + m
    # L2是BOD对BOD响应矩阵乘以L，再加上m，意义就是各个断面的BOD浓度
 
    O2 = V.dot(L) + n
    # O2是溶解氧对BOD响应矩阵乘以L，再加上n，意义就是各个断面的溶解氧浓度
    
    res = [U, V, m, n, L2, O2] # 返回结果
    return res
 
N = int(input('请输入河流断面个数n: '))
Q1 = np.zeros((N+1, 1))  # 修改为numpy数组，N+1是行数，1是列数
Q1[0] = float(input('请输入背景河水流量Q0(m3/s): '))
 
print(f'请输入{N}个断面的污水流量Q(m3/s)，用空格分隔：')
Q = np.array([float(x) for x in input().split()]).reshape(N, 1)
 
print(f'请输入{N}个断面的BOD浓度L(mg/L)，用空格分隔：')
L = np.array([float(x) for x in input().split()]).reshape(N, 1)
 
print(f'请输入{N}个断面的DO浓度O(mg/L)，用空格分隔：')
O = np.array([float(x) for x in input().split()]).reshape(N, 1)
 
print(f'请输入{N}个断面的BOD降解速度常数kd(/d)，用空格分隔：')
kd = np.array([float(x) for x in input().split()]).reshape(N, 1)
 
print(f'请输入{N}个断面的大气复氧速度ka(/d)，用空格分隔：')
ka = np.array([float(x) for x in input().split()]).reshape(N, 1)
 
print(f'请输入{N}个断面间的经过时间t(d)，用空格分隔：')
t = np.array([float(x) for x in input().split()]).reshape(N, 1)
 
print(f'请输入{N}个断面的引出流量Q3(m3/s)，用空格分隔：')
Q3 = np.array([float(x) for x in input().split()]).reshape(N, 1)
 
T = float(input('请输入河水水温/℃: '))
L20 = float(input('请输入河流背景的BOD浓度L20(mg/L): '))
O20 = float(input('请输入河流背景的DO浓度O20(mg/L): '))
 
res = calculate(N, Q1, Q, L, O, kd, ka, t, Q3, T, L20, O20)
 
# 输出结果
print('U响应矩阵：')
print(res[0])
print('V相应矩阵：')
print(res[1])
print('m向量：')
print(res[2])
print('n向量：')
print(res[3])
print('各断面BOD浓度：')
print(res[4])
print('各断面DO浓度：')
print(res[5])
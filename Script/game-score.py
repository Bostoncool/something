# 我们可以使用带有剪枝（Pruning）的嵌套循环来显著提高效率。
# 基本思想是：按积分从高到低依次确定 a,b,c,d 的值，
# 在每一步都检查当前的部分解是否仍然可能满足所有约束条件，
# 如果不可能，则提前终止该分支的搜索。

import math
import time # 可以移除，除非需要计时

def solve_game_scores_optimized(n, y):
    """
    优化算法：查找所有满足条件的积分组合。

    Args:
        n (int): 当前天数 (1 <= n <= 14).
        y (int): 目标总积分.

    Returns:
        list: 包含所有解的列表，每个解为 (K, a, b, c, d, e) 的元组。
              其中 K 是总局数, a,b,c,d,e 分别是获得 202, 162, 132, 99, 66 分的局数。
              如果无解则返回空列表。
    """
    solutions = []
    max_total_games = 6 * n
    scores = [202, 162, 132, 99, 66]

    # 基本有效性检查
    if y < 0:
        return []
    if y == 0:
        # 积分为0，唯一可能是玩了0局 (K=0)
        # 需要确保 n 允许 0 局 (总是允许)
        return [(0, 0, 0, 0, 0, 0)]
    # 如果目标分数y大于理论最大可能分数 (n天内全玩最高分)
    if y > scores[0] * max_total_games:
        return []
    # 如果目标分数y大于0，但小于最低单局分数66
    if y > 0 and y < scores[4]:
        return []

    # --- 计算 K 的理论边界 ---
    # 最小 K (基于最高分) - 注意处理除以0的情况 (虽然本例分数 > 0)
    min_k_score = math.ceil(y / scores[0]) if scores[0] > 0 else 0
    # 最大 K (基于最低分)
    max_k_score = math.floor(y / scores[4]) if scores[4] > 0 else float('inf')

    # 结合 n 天的限制，得到 K 的最终有效范围 [min_k_eff, max_k_eff]
    min_k_eff = min_k_score
    max_k_eff = min(max_total_games, max_k_score)

    # 如果最小有效 K 大于最大有效 K，则无解
    if min_k_eff > max_k_eff:
        return []

    # --- 嵌套循环搜索 ---
    # 循环 a (202分)
    # a 的最大值不能超过总局数上限，也不能超过仅用a得到y所需局数
    max_a = min(max_k_eff, y // scores[0])
    for a in range(max_a + 1):
        rem_y_1 = y - scores[0] * a         # 剩余分数
        # 剩余局数上限 (b+c+d+e 的上限)
        # 不能超过总局数上限减去a，也不能超过理论上用剩余分数能达到的最大局数
        rem_k_upper_1 = min(max_k_eff - a, math.floor(rem_y_1 / scores[4]) if scores[4] > 0 else float('inf'))

        # 剪枝: 如果剩余分数为负 或 剩余局数上限为负
        if rem_y_1 < 0 or rem_k_upper_1 < 0:
            continue

        # 计算剩余局数下限 (b+c+d+e 的下限)
        # K = a+b+c+d+e >= min_k_eff => b+c+d+e >= min_k_eff - a
        # 同时，剩余局数也需要至少 ceil(rem_y_1 / scores[1]) 才能得到剩余分数（用最高分算）
        rem_k_lower_1 = max(0, min_k_eff - a, math.ceil(rem_y_1 / scores[1]) if scores[1] > 0 and rem_y_1 > 0 else 0)


        # 如果剩余局数下限 > 剩余局数上限，矛盾
        if rem_k_lower_1 > rem_k_upper_1:
             continue


        # 循环 b (162分)
        max_b = min(rem_k_upper_1, rem_y_1 // scores[1])
        for b in range(max_b + 1):
            rem_y_2 = rem_y_1 - scores[1] * b
            rem_k_upper_2 = min(rem_k_upper_1 - b, math.floor(rem_y_2 / scores[4]) if scores[4] > 0 else float('inf'))

            if rem_y_2 < 0 or rem_k_upper_2 < 0:
                continue

            rem_k_lower_2 = max(0, rem_k_lower_1 - b, math.ceil(rem_y_2 / scores[2]) if scores[2] > 0 and rem_y_2 > 0 else 0)

            # 剪枝
            if rem_k_lower_2 > rem_k_upper_2:
                continue


            # 循环 c (132分)
            max_c = min(rem_k_upper_2, rem_y_2 // scores[2])
            for c in range(max_c + 1):
                rem_y_3 = rem_y_2 - scores[2] * c
                rem_k_upper_3 = min(rem_k_upper_2 - c, math.floor(rem_y_3 / scores[4]) if scores[4] > 0 else float('inf'))

                if rem_y_3 < 0 or rem_k_upper_3 < 0:
                    continue

                rem_k_lower_3 = max(0, rem_k_lower_2 - c, math.ceil(rem_y_3 / scores[3]) if scores[3] > 0 and rem_y_3 > 0 else 0)

                # 剪枝
                if rem_k_lower_3 > rem_k_upper_3:
                    continue

                # 循环 d (99分)
                max_d = min(rem_k_upper_3, rem_y_3 // scores[3])
                for d in range(max_d + 1):
                    rem_y_4 = rem_y_3 - scores[3] * d
                    # 此时剩余局数仅由 e 构成，所以上限就是 e 的上限
                    rem_k_upper_4 = min(rem_k_upper_3 - d, math.floor(rem_y_4 / scores[4]) if scores[4] > 0 else float('inf')) # 这是 e 的上限

                    if rem_y_4 < 0 or rem_k_upper_4 < 0:
                        continue

                    # 剩余局数下限，也是 e 的下限
                    rem_k_lower_4 = max(0, rem_k_lower_3 - d, math.ceil(rem_y_4 / scores[4]) if scores[4] > 0 and rem_y_4 > 0 else 0) # 这是 e 的下限

                    # 剪枝
                    if rem_k_lower_4 > rem_k_upper_4:
                         continue

                    # 求解 e (66分)
                    # 需要满足 rem_y_4 == scores[4] * e
                    # 且 rem_k_lower_4 <= e <= rem_k_upper_4
                    if scores[4] == 0: # 避免除以零，虽然本例不会
                        if rem_y_4 == 0 and rem_k_lower_4 <= 0 <= rem_k_upper_4:
                            e = 0 # 如果分数为0，需要0局
                        else:
                            continue # 分数非0但除数是0，无解；或局数范围不允许0
                    elif rem_y_4 % scores[4] == 0:
                        e = rem_y_4 // scores[4]
                        # 检查 e 是否在剩余局数范围内
                        if e >= rem_k_lower_4 and e <= rem_k_upper_4:
                            # 找到一个有效解
                            k = a + b + c + d + e
                            # 最终确认 K 在有效范围内 (通常由循环边界保证，但做个检查)
                            if k >= min_k_eff and k <= max_k_eff:
                                solutions.append((k, a, b, c, d, e))
                    # else: # rem_y_4 不能被 scores[4] 整除，跳过

    # 对结果排序（可选，为了展示一致性）
    solutions.sort()
    return solutions

# --- 如何使用 ---
if __name__ == '__main__':
    # 示例：第 n 天，总分 y
    n_test = 14
    # n 可以是 1 到 14 之间的任意整数

    y_test = 14307

    start_time = time.time() # 开始计时
    possible_solutions = solve_game_scores_optimized(n_test, y_test)
    end_time = time.time()   # 结束计时

    print(f"对于第 {n_test} 天，总分 {y_test} 的可能解 ({len(possible_solutions)} 种):")
    if not possible_solutions:
        print("  没有找到满足条件的解。")
    else:
        for sol in possible_solutions:
            k, a, b, c, d, e = sol
            print(f"  总局数 K={k}, 202分: {a}局, 162分: {b}局, 132分: {c}局, 99分: {d}局, 66分: {e}局")

    print(f"\n计算耗时: {end_time - start_time:.6f} 秒")
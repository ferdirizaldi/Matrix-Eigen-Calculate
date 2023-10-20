# Python の time モジュールと numpy モジュールをインポートします。
import random
import time
import numpy as np

# 連立1次方程式を解くプログラム
# 部分的なピボット付きガウスの消去法と numpy を利用します。

def print_matrix(mat):
    # 行列を表示します。
    for i in range(len(mat)):
        print(mat[i])

def swap_rows(mat, vec, row1, row2):
    # 2つの行を交換します。
    mat[row1], mat[row2] = mat[row2], mat[row1]
    vec[row1], vec[row2] = vec[row2], vec[row1]

def find_max_row(mat, col, n):
    # 指定した列で最大値を持つ行を見つけます。
    max_index = col
    for i in range(col+1, n):
        if abs(mat[i][col]) > abs(mat[max_index][col]):
            max_index = i
    return max_index

def gaussian_elimination(mat, vec):
    # 部分的なピボット付きガウスの消去法で連立1次方程式を解きます。

    n = len(mat)

    for i in range(n):
        # 最大値を持つ行を見つけて現在の行と交換します。
        max_index = find_max_row(mat, i, n)
        if i != max_index:
            swap_rows(mat, vec, i, max_index)

        # 各行から適切な倍数を引きます。
        for j in range(i+1, n):
            ratio = mat[j][i]/mat[i][i]

            for k in range(n):
                mat[j][k] = mat[j][k] - ratio * mat[i][k]
            vec[j] = vec[j] - ratio * vec[i]

    # 解を格納します。
    x = [0 for i in range(n)]

    # 後方代入を行います。
    for i in range(n-1, -1, -1):
        x[i] = vec[i]/mat[i][i]

        for j in range(i-1, -1, -1):
            vec[j] = vec[j] - mat[j][i] * x[i]

    return x

def solve_linear_equations_np(mat, vec):
    # numpy の linalg.solve 関数を用いて連立1次方程式を解きます。
    return np.linalg.solve(mat, vec)

mat = [[random.randint(1,10) for _ in range(100)] for _ in range(100)]
vec = [random.randint(1,10) for _ in range(100)]

mat_np = np.array(mat)
vec_np = np.array(vec)

print("行列:")
print_matrix(mat)

print("\nベクトル:")
print(vec)

# Gaussian Elimination の実行時間を測定します。
start_time = time.time()

solution_ge = gaussian_elimination(mat, vec)

end_time = time.time()

print("\nGaussian Elimination の解:", solution_ge)
print("Gaussian Elimination の実行時間: {:.10f}秒\n".format(end_time - start_time))

# numpy の実行時間を測定します。
start_time = time.time()

solution_np = solve_linear_equations_np(mat_np, vec_np)

end_time = time.time()

print("Numpy の解:", solution_np)
print("Numpy の実行時間: {:.10f}秒".format(end_time - start_time))

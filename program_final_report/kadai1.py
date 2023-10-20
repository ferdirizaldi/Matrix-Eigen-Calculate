import time
import numpy as np

def calculate_determinant(matrix):
    # 2x2行列の場合の基本ケース
    if len(matrix) == 2:
        return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]

    determinant = 0
    for c in range(len(matrix)):
        determinant += ((-1)**c)*matrix[0][c]*calculate_determinant(get_submatrix(matrix, 0, c))
    return determinant

def get_submatrix(matrix, row, col):
    return [[matrix[i][j] for j in range(len(matrix[i])) if j != col] for i in range(len(matrix)) if i != row]

# 100x100のランダム行列を定義
matrix = np.random.randint(0, 10, (100, 100)).tolist()

print("与えられた行列の先頭5x5は次の通りです：")  # 全部を印刷すると非常に長くなるので、先頭の5x5だけを印刷します
for row in matrix[:5]:
    print(row[:5])

# Recursive method (非常に時間がかかるためコメントアウトします)
# print("\n再帰メソッドによるこの行列の行列式は", calculate_determinant(matrix), "です")
# end_time_recursive = time.time()
# print("再帰メソッドの実行時間は{:.7f}".format(end_time_recursive - start_time_recursive), "秒です")

# NumPy method
start_time_numpy = time.time()
matrix_np = np.array(matrix)
print("\nNumPyによるこの行列の行列式は", np.linalg.det(matrix_np), "です")
end_time_numpy = time.time()
print("NumPyの実行時間は{:.7f}".format(end_time_numpy - start_time_numpy), "秒です")

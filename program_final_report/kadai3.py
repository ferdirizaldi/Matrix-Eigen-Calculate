import math
import time
import numpy as np

def print_matrix(mat):
    # 行列を表示します。
    for i in range(len(mat)):
        print(mat[i])

def max_off_diag(A):
    n = len(A)
    max_val = 0.0
    for i in range(n-1):
        for j in range(i+1, n):
            if abs(A[i][j]) >= max_val:
                max_val = abs(A[i][j])
                k = i
                l = j
    return k, l, max_val

def rotate(A, p, q):
    tau = (A[q][q] - A[p][p]) / (2.0 * A[p][q])
    if tau >= 0:
        t = 1.0 / (tau + math.sqrt(1.0 + tau**2))
    else:
        t = -1.0 / (-tau + math.sqrt(1.0 + tau**2))
    c = 1.0 / math.sqrt(1.0 + t**2)
    s = t * c
    return c, s, t

def jacobi(A, tol=1e-6):
    n = len(A)
    max_iter = 100
    V = [[0.0]*n for _ in range(n)]
    for i in range(n):
        V[i][i] = 1.0
    for _ in range(max_iter):
        p, q, max_val = max_off_diag(A)
        if max_val < tol:
            return [A[i][i] for i in range(n)], V
        c, s, t = rotate(A, p, q)
        for i in range(n):
            if i != p and i != q:
                A[p][i], A[q][i] = c*A[p][i] - s*A[q][i], s*A[p][i] + c*A[q][i]
                A[i][p], A[i][q] = A[p][i], A[q][i]
            V[i][p], V[i][q] = c*V[i][p] - s*V[i][q], s*V[i][p] + c*V[i][q]
        A[p][p], A[q][q] = A[p][p] - t*A[p][q], A[q][q] + t*A[p][q]
        A[p][q] = A[q][p] = 0.0
    return [A[i][i] for i in range(n)], V

# 10x10のランダムな行列を生成します。
A_random = np.random.randint(-10, 10, (100, 100))
# 行列を対称にします。
A = 0.5 * (A_random + A_random.T)
print_matrix(A)

# Measure time for Jacobi method
start_time = time.time()
eigvals_jacobi, eigvecs_jacobi = jacobi(A)
end_time = time.time()
print("Jacobi Method")
print('Execution Time:{:.10f}'.format(end_time - start_time))
print('Eigenvalues:', eigvals_jacobi)

np.set_printoptions(precision=8)
print('Eigenvectors:\n', np.array(eigvecs_jacobi))

# Measure time for numpy's method
A_np = np.array(A)
start_time = time.time()
eigvals_np, eigvecs_np = np.linalg.eig(A_np)
end_time = time.time()
print("\nNumpy's Method")
print('Execution Time:{:.10f}'.format(end_time - start_time))
print('Eigenvalues:', eigvals_np)
print('Eigenvectors:\n', eigvecs_np)
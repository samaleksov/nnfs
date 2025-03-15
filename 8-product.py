import numpy as np

def mul(A, B):
    rows_A = len(A)
    cols_A = len(A[0])
    rows_B = len(B)
    cols_B = len(B[0])

    result = [None] * rows_A
    if cols_A != rows_B:
        return result

    for i in range(rows_A):
        result[i] = [0] * cols_B
        for j in range(cols_B):
            for k in range(rows_B):
                result[i][j] += A[i][k] * B[k][j]
    return result

A = [
    [0.49, 0.97, 0.53, 0.05],
    [0.33, 0.65, 0.62, 0.51],
    [1.00, 0.38, 0.61, 0.45],
    [0.74, 0.27, 0.64, 0.17],
    [0.36, 0.17, 0.96, 0.12]
]

B = [
    [0.79, 0.32, 0.68, 0.90, 0.77],
    [0.18, 0.39, 0.12, 0.93, 0.09],
    [0.87, 0.42, 0.60, 0.71, 0.12],
    [0.45, 0.55, 0.40, 0.78, 0.81]
]

C = mul(A, B)

print(np.array(C))
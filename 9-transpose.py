import numpy as np

def dot(A, B):
    # numpy behaviour, first parameter specifies output shape
    # 2 vectors
    if isinstance(B, list) and len(B) > 0 and isinstance(B[0], list) and len(B[0]) == 1 and len(A) == 1 and isinstance(A[0], list):
        result = 0
        for i in range(len(A[0])):
            result += A[0][i] * B[i][0]
        return result
    # matrix and vector
    if isinstance(B, list) and len(B) > 0 and not isinstance(B[0], list) and len(A) > 0 and isinstance(A[0], list):
        result = []
        for a in A:
            result.append(dot(a, B))
        return result
    result = 0
    for a, b in zip(A, B):
        result += a*b
    return result

def transpose(A):
    rows_A = len(A)
    cols_A = len(A[0]) if isinstance(A[0], list) else 0

    # 1-D array
    if cols_A == 0:
        return [[a] for a in A]

    result = [None] * cols_A

    for i in range(cols_A):
        result[i] = [None] * rows_A
        for j in range(rows_A):
            result[i][j] = A[j][i]

    return result 

A = [
    [ 0,  1,  2,  3,  4],
    [ 5,  6,  7,  8,  9],
    [10, 11, 12, 13, 14],
    [15, 16, 17, 18, 19]
]

A_T = transpose(A)

print('A')
print(np.array(A))
print('A transposed')
print(np.array(A_T))

a = [[1, 2, 3]]
b = [[2, 3, 4]]
b_T = transpose(b)

c = dot(a, b_T)

print()
print('dot product a . b transposed')
print(np.array(c))
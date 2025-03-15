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

def diagflat(A):
    result = []
    for i, row in enumerate(A):
        row_items = []
        for j, value in enumerate(A):
            row_items.append(A[i][0] if i == j else 0)
        result.append(row_items)
    return result

def matrix_sub(A, B):
    result = []
    cols = len(A[0])
    rows = len(A)

    for i, row in enumerate(A):
        row_result = []
        for j, value in enumerate(row):
            row_result.append(A[i][j] - B[i][j])
        result.append(row_result)
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

softmax_output = [0.7, 0.1, 0.2]

softmax_output = transpose(softmax_output)
print(softmax_output)

print(diagflat(softmax_output))

print(mul(softmax_output, transpose(softmax_output)))

print(matrix_sub(diagflat(softmax_output), mul(softmax_output, transpose(softmax_output))))
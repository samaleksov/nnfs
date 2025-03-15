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

def sum(A, B):
    # matrix and vector
    if isinstance(B, list) and len(B) > 0 and not isinstance(B[0], list) and len(A) > 0 and isinstance(A[0], list):
        result = []
        for a in A:
            result.append(sum(a, B))
        return result
    result = []
    for a, b in zip(A, B):
        result.append(a+b)
    return result

inputs = [
    [1.0, 2.0, 3.0, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
]

weights = [
    [0.2, 0.8, -0.5, 1.0],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]

biases = [2.0 , 3.0, 0.5]

weights2 = [
    [0.1, -0.14, 0.5],
    [-0.5, 0.12, -0.33],
    [-0.44, 0.73, -0.13]
]

biases2 = [-1 , 2, -0.5]

layer1_outputs = sum(mul(inputs, transpose(weights)), biases)

# we feed the output of layer 1 as input for layer 2
layer2_outputs = sum(mul(layer1_outputs, transpose(weights2)), biases2)

print(layer2_outputs)
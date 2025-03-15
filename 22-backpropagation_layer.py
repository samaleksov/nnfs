import numpy as np

def dot(A, B):
    result = 0
    for a, b in zip(A, B):
        result += a*b
    return result

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
    # matrix and 1 row matrix
    if isinstance(B, list) and len(B) == 1 and len(A) > 0 and isinstance(A[0], list):
        result = []
        for a in A:
            result.append(sum(a, B[0]))
        return result

    # matrix and numpy vector
    if isinstance(B, np.ndarray) and len(B) == 1 and len(A) > 0 and isinstance(A[0], list):
        result = []
        for a in A:
            result.append(sum(a, B[0]))
        return result

    # matrix and vector
    if isinstance(B, list) and len(B) > 0 and not isinstance(B[0], list) and len(A) > 0 and isinstance(A[0], list):
        result = []
        for a in A:
            result.append(sum(a, B))
        return result
    # matrix and matrix
    if isinstance(B, list) and len(B) > 0 and isinstance(B[0], list) and len(A) > 0 and isinstance(A[0], list):
        result = []
        for a, b in zip(A, B):
            result.append(sum(a, b))
        return result
    result = []
    for a, b in zip(A, B):
        result.append(a+b)
    return result

def scale_matrix(n, A):
    result = []
    for row in A:
        tmp = []
        for value in row: 
            tmp.append(n * value)
        result.append(tmp)
    return result

def sum_cols(A):
    result = []
    cols = len(A[0])
    rows = len(A)

    for col in range(cols):
        colsum = 0
        for row in range(rows):
            colsum += A[row][col]
        result.append(colsum)
    return [result]

def max_number_matrix(n, A):
    result = []
    for row in A:
        tmp = []
        for value in row: 
            tmp.append(max(0, value))
        result.append(tmp)
    return result
    
# component-wise product
def hadamard(A, B):
    result = []
    cols = len(A[0])
    rows = len(A)

    for i, row in enumerate(A):
        row_result = []
        for j, value in enumerate(row):
            row_result.append(A[i][j] * B[i][j])
        result.append(row_result)
    return result
            

dvalues = [[1., 1., 1.]]
weights = transpose([[0.2, 0.8, -0.5, 1.],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]])

dx0 = weights[0][0]*dvalues[0][0] + weights[0][1]*dvalues[0][1] + weights[0][2]*dvalues[0][2]
dx1 = weights[1][0]*dvalues[0][0] + weights[1][1]*dvalues[0][1] + weights[1][2]*dvalues[0][2]
dx2 = weights[2][0]*dvalues[0][0] + weights[2][1]*dvalues[0][1] + weights[2][2]*dvalues[0][2]
dx3 = weights[3][0]*dvalues[0][0] + weights[3][1]*dvalues[0][1] + weights[3][2]*dvalues[0][2]

dinputs = [dx0, dx1, dx2, dx3]
print(dinputs)

dx0 = dot(weights[0], dvalues[0])
dx1 = dot(weights[1], dvalues[0])
dx2 = dot(weights[2], dvalues[0])
dx3 = dot(weights[3], dvalues[0])

dinputs = [dx0, dx1, dx2, dx3]
print(dinputs)

print(mul(dvalues, transpose(weights))[0])

dvalues = [[1., 1., 1.],
           [2., 2., 2.],
           [3., 3., 3.]]

print(mul(dvalues, transpose(weights)))

inputs = [[1., 2., 3., 2.5],
          [2., 5., -1., 2.],
          [-1.5, 2.7, 3.3, -0.8]]

dweights = mul(transpose(inputs), dvalues)
print(dweights)

biases = [[2, 3, 0.5]]

dbiases = sum_cols(dvalues)
print(dbiases)

z = [[1, 2, -3, -4],
     [2, -7, -1, 3],
     [-1, 2, 5, -1]]
    
dvalues = [[1, 2, 3, 4],
           [5, 6, 7, 8],
           [9, 10, 11, 12]]

drelu = [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]]

for i, row in enumerate(z):
    for j, value in enumerate(row):
        drelu[i][j] = 1 if value > 0 else 0 

print(drelu)

drelu = hadamard(drelu, dvalues)

print(drelu)


dvalues = [[1., 1., 1.],
           [2., 2., 2.],
           [3., 3., 3.]]

inputs = [[1., 2., 3., 2.5],
          [2., 5., -1., 2.],
          [-1.5, 2.7, 3.3, -0.8]]

weights = transpose([[0.2, 0.8, -0.5, 1.],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]])

biases = [[2, 3, 0.5]]

layer_outputs = sum(mul(inputs, weights), biases)
relu_outputs = max_number_matrix(0, layer_outputs)

drelu = relu_outputs.copy()
for i, row in enumerate(layer_outputs):
    for j, value in enumerate(row):
        drelu[i][j] = 0 if value <= 0 else drelu[i][j]

dinputs = mul(drelu, transpose(weights))
dweights = mul(transpose(inputs), drelu)
dbiases = sum_cols(drelu)

weights = sum(weights, scale_matrix(-0.001, dweights))
biases = sum(biases, scale_matrix(-0.001, dbiases))

print(weights)
print(biases)
import numpy as np

def dot(A, B):
    # numpy behaviour, first parameter specifies output shape
    if isinstance(B, list) and len(B) > 0 and not isinstance(B[0], list) and len(A) > 0 and isinstance(A[0], list):
        result = []
        for a in A:
            result.append(dot(a, B))
        return result
    result = 0
    for a, b in zip(A, B):
        result += a*b
    return result

def sum(A, B):
    result = []
    for a, b in zip(A, B):
        result.append(a+b)
    return result

inputs = [1, 2, 3, 2.5]
weights = [
    [0.2, 0.8, -0.5, 1.0],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87],
]

biases = [2, 3, 0.5]

print('Ours:')
layer_outputs = dot(weights, inputs)
print(layer_outputs)
layer_outputs = sum(layer_outputs, biases)
print(layer_outputs)

print("Numpy:")
layer_outputs = np.dot(weights, inputs)
print(layer_outputs)
layer_outputs += biases
print(layer_outputs)
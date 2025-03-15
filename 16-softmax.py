import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

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

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = [0.] * n_neurons
    
    def forward(self, inputs):
        self.output = sum(mul(inputs, self.weights), self.biases)

class Activation_ReLu:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        prob = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = prob

X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLu()

dense2 = Layer_Dense(3, 3)

activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])
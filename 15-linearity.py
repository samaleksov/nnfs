import numpy as np
import matplotlib.pyplot as plt
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

X, y = spiral_data(samples=100, classes=3)

# Manual example of 2 hidden layers
# 4 neurons per layer with ReLU activations
# 1 output neuron

dense1_1 = Layer_Dense(1, 1)
dense1_1.weights[0][0] = 6 

dense1_2 = Layer_Dense(1, 1)
dense1_2.weights[0][0] = -1
dense1_2.biases[0] = 0.7

dense2_1 = Layer_Dense(1, 1)
dense2_1.weights[0][0] = 3.5
dense2_1.biases[0] = -0.42

dense2_2 = Layer_Dense(1, 1)
dense2_2.weights[0][0] = -1
dense2_2.biases[0] = 0.27

dense3_1 = Layer_Dense(1, 1)
dense3_1.weights[0][0] = -3.5
dense3_1.biases[0] = 1.35

dense3_2 = Layer_Dense(1, 1)
dense3_2.weights[0][0] = -1
dense3_2.biases[0] = 0.30

dense8_1 = Layer_Dense(1, 1)
dense8_2 = Layer_Dense(1, 1)
dense8_2.biases[0] = 1

output_layer = Layer_Dense(4, 1)

output_layer.weights[0][0] = -1
output_layer.weights[1][0] = -1
output_layer.weights[2][0] = -1
output_layer.weights[3][0] = 0.97

activation1_1 = Activation_ReLu()
activation1_2 = Activation_ReLu()
activation2_1 = Activation_ReLu()
activation2_2 = Activation_ReLu()
activation3_1 = Activation_ReLu()
activation3_2 = Activation_ReLu()
activation8_1 = Activation_ReLu()
activation8_2 = Activation_ReLu()

x = np.linspace(0, 1, 200)

nn_in = x.reshape(-1,1)

dense1_1.forward(nn_in)
activation1_1.forward(dense1_1.output)
dense1_2.forward(activation1_1.output)
activation1_2.forward(dense1_2.output)

dense2_1.forward(nn_in)
activation2_1.forward(dense2_1.output)
dense2_2.forward(activation2_1.output)
activation2_2.forward(dense2_2.output)

dense3_1.forward(nn_in)
activation3_1.forward(dense3_1.output)
dense3_2.forward(activation3_1.output)
activation3_2.forward(dense3_2.output)

dense8_1.forward(nn_in)
activation8_1.forward(dense8_1.output)
dense8_2.forward(activation8_1.output)
activation8_2.forward(dense8_2.output)

output_layer.forward([[activation1_2.output, activation2_2.output, activation3_2.output, activation8_2.output]])

nn_out = np.array(output_layer.output).reshape(1,-1)[0]

y_sin = np.sin(2 * np.pi * x)

plt.figure(figsize=(10, 6))

plt.plot(x, y_sin, 'b-', label='sin(x)', linewidth=2)
plt.plot(x, nn_out, 'r--', label='nn', linewidth=2)

plt.grid(True, linestyle='--', alpha=0.7)

plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)

plt.legend(fontsize=10)

plt.show()
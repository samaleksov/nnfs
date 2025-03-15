import numpy as np
import matplotlib.pyplot as plt
import nnfs
import math
from nnfs.datasets import vertical_data, spiral_data

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
        self.output = []
        for row in inputs:
            activations = []
            for value in row: 
                activations.append(max(0, value))
            self.output.append(activations)

        # self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        # exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # prob = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        # self.output = prob
        self.output = []
        for row in inputs:
            activations = []
            largest_value = -99999
            total = 0
            for value in row:
                if value > largest_value:
                    largest_value = value
            for value in row:
                exp_value = math.exp(value - largest_value)
                total += exp_value
                activations.append(exp_value)
            for i, value in enumerate(activations):
                activations[i] /= total
            self.output.append(activations)

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        # data_loss = np.mean(sample_losses)
        data_loss = 0
        n = 0
        total = 0
        for value in sample_losses:
            total += value
            n += 1
        data_loss = total / n
            
        return data_loss

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        # y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        y_pred_clipped = []
        for row in y_pred:
            pred = []
            for value in row:
                pred.append(min(1 - 1e-7, max(value, 1e-7)))
            y_pred_clipped.append(pred)

        correct_confidences = []
        if len(y_true.shape) == 1:
            #correct_confidences = y_pred_clipped[
            #    range(samples),
            #    y_true
            #]
            for i, row in enumerate(y_pred_clipped):
                correct_confidences.append(row[y_true[i]])
        elif len(y_true.shape) == 2:
            for i, row in enumerate(y_pred_clipped):
                total = 0
                for j, value in enumerate(row):
                    total += value * y_true[i][j]
                correct_confidences.append(total)
            # correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)


        neg_log_likelihoods = []
        for value in correct_confidences:
            neg_log_likelihoods.append(-math.log(value))
            
        # neg_log_likelihoods = -np.log(correct_confidences)
        return neg_log_likelihoods

X, y = spiral_data(samples=100, classes=3)

#plt.scatter(X[:,0], X[:,1], c=y, s=40, cmap='brg')
#plt.show()


dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLu()
dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

loss_function = Loss_CategoricalCrossentropy()

lowest_loss = 99999999
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()

for iteration in range(10000):
    dense1.weights += 0.05 * np.random.randn(2, 3)
    dense1.biases += 0.05 * np.random.randn(1, 3)
    dense2.weights += 0.05 * np.random.randn(3, 3)
    dense2.biases += 0.05 * np.random.randn(1, 3)

    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    loss = loss_function.calculate(activation2.output, y)
    predictions = []
    for row in activation2.output:
        largest_index = 0
        largest = row[largest_index]
        for i, value in enumerate(row):
            if value > largest:
                largest = value
                largest_index = i
        predictions.append(largest_index)
    #predictions2 = np.argmax(activation2.output, axis=1)
    #for p1, p2 in zip(predictions, predictions2):
    #    assert(p1 == p2)
    total = 0
    for pred, target in zip(predictions, y):
        if (pred == target):
            total += 1
    accuracy = total / len(predictions)

    # accuracy2 = np.mean(predictions2 == y)

    if loss < lowest_loss:
        print('New weights found, iteration:', iteration, 'loss:', loss, 'acc:', accuracy)
        lowest_loss = loss
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
    else:
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()
import numpy as np
import matplotlib.pyplot as plt
import nnfs
import math
from nnfs.datasets import vertical_data, spiral_data

nnfs.init()

def dot(A, B):
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

def identity_matrix(len):
    result = []
    for i in range(len):
        row = []
        for j in range(len):
            row.append(1 if i == j else 0)
        result.append(row)
    return result
            
def mul(A, B):
    rows_A = len(A)
    cols_A = len(A[0])
    rows_B = len(B)

    if isinstance(B[0], float):
        return [dot(A[i], B) for i in range(rows_A)]

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

# fix matmul to work with more shapes
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = [0.] * n_neurons
    
    def forward(self, inputs):
        self.inputs = inputs
        self.output = sum(mul(inputs, self.weights), self.biases)
    
    def backward(self, dvalues):
        self.dweights = mul(transpose(self.inputs), dvalues)
        self.dbiases = sum_cols(dvalues)
        self.dinputs = mul(dvalues, transpose(self.values))

class Activation_ReLu:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = []
        # self.output = np.maximum(0, inputs)
        for row in inputs:
            activations = []
            for value in row: 
                activations.append(max(0, value))
            self.output.append(activations)
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        for i, row in enumerate(dvalues):
            for j, value in enumerate(row):
                self.dinputs[i][j] = 0 if value <= 0 else self.dinputs[i][j]

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
    def backward(self, dvalues):
        self.dinputs = [None] * len(dvalues)
        self.dinputs2 = self.dinputs.copy()
        for index, (single_output, single_values) in enumerate(zip(self.output, dvalues)):
            single_output = transpose(single_output)
            jacobian_matrix = matrix_sub(diagflat(single_output), mul(single_output, transpose(single_output)))
            self.dinputs[index] = mul(jacobian_matrix, single_values)

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
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            tmp = []
            id = identity_matrix(labels)
            for i in y_true:
                tmp.append(id[i])
            y_true = tmp
        #self.dinputs = -y_true / dvalues
        #self.dinputs = self.dinputs /samples
        self.dinputs = [None] * samples
        for i, row in enumerate(y_true):
            self.dinputs[i] = [None] * len(row)
            for j, value in enumerate(row):
                self.dinputs[i][j] = - y_true[i][j] / dvalues[i][j]
                self.dinputs[i][j] /= samples

class Activation_Softmax_Loss_CategoricalCrossentropy():
    
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()
        
    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            tmp = []
            for row in y_true:
                largest_index = 0
                largest = row[largest_index]
                for i, value in enumerate(row):
                    if value > largest:
                        largest = value
                        largest_index = i
                tmp.append(largest_index)
            y_true = tmp
        self.dinputs = dvalues.copy()

        for i, row in enumerate(self.dinputs):
            for j, value in enumerate(row):
                self.dinputs[i][j] = self.dinputs[i][j] - 1 if j == y_true[i] else self.dinputs[i][j]
                self.dinputs[i][j] /= samples
        #self.dinputs[range(samples), y_true] -= 1
        #self.dinputs = self.dinputs / samples
        
softmax_outputs = [[0.7, 0.1, 0.2],
                   [0.1, 0.5, 0.4],
                   [0.02, 0.9, 0.08]]

class_targets = np.array([0, 1, 1])


softmax_loss = Activation_Softmax_Loss_CategoricalCrossentropy()
softmax_loss.backward(softmax_outputs, class_targets)
dvalues1 = softmax_loss.dinputs

softmax_outputs = [[0.7, 0.1, 0.2],
                   [0.1, 0.5, 0.4],
                   [0.02, 0.9, 0.08]]

activation = Activation_Softmax()
activation.output = softmax_outputs
loss = Loss_CategoricalCrossentropy()
loss.backward(softmax_outputs, class_targets)
activation.backward(loss.dinputs)
dvalues2 = activation.dinputs

print('Gradients: combined loss and activation:')
print(dvalues1)

print('Gradients: separate loss and activation:')
print(dvalues2)
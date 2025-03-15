import numpy as np
import nnfs
import random
import math
from nnfs.datasets import spiral_data

nnfs.init(random_seed=0)

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

def standard_normal():
    # Generate two independent random numbers from a uniform distribution
    u1 = random.random()
    u2 = random.random()
    print(u1, u2)


    # Box-Muller transform to get two standard normal random variables
    z0 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
    z1 = math.sqrt(-2 * math.log(u1)) * math.sin(2 * math.pi * u2)

    print(z0, z1)
    #print(z1)
    return z0
# Constants for the Ziggurat algorithm
R = 3.442619855899  # Tail cut-off
V = 9.91256303526217e-3  # Area under the curve

# Precompute the tables
NUM_BLOCKS = 256
x = [0.0] * (NUM_BLOCKS + 1)
y = [0.0] * NUM_BLOCKS
f = [0.0] * NUM_BLOCKS

def ziggurat_setup():
    """Precompute the Ziggurat tables."""
    x[NUM_BLOCKS] = R
    f[NUM_BLOCKS] = math.exp(-0.5 * R * R)

    for i in range(NUM_BLOCKS - 1, 0, -1):
        x[i] = math.sqrt(-2.0 * math.log(f[i + 1] + V / x[i + 1]))
        f[i] = math.exp(-0.5 * x[i] * x[i])

    for i in range(NUM_BLOCKS):
        y[i] = f[i + 1]

ziggurat_setup()

def ziggurat_normal():
    """Generate a normally distributed random number using the Ziggurat algorithm."""
    while True:
        # Step 1: Generate a random index and uniform x
        i = random.randint(0, NUM_BLOCKS - 1)
        u = 2.0 * random.random() - 1.0
        x_val = u * x[i]

        # Step 2: Accept if within the core region
        if abs(x_val) < x[i + 1]:
            return x_val

        # Step 3: Check within the tail
        if i == 0:
            while True:
                x_tail = -math.log(random.random()) / R
                y_tail = -math.log(random.random())
                if 2.0 * y_tail >= x_tail * x_tail:
                    return R + x_tail if u > 0 else -R - x_tail

        # Step 4: Accept if under the curve
        if y[i] + random.random() * (y[i - 1] - y[i]) < math.exp(-0.5 * x_val * x_val):
            return x_val
#X, y = spiral_data(samples=100, classes=3)

#dense1 = Layer_Dense(2, 3)
#dense1.forward(X)

#print(dense1.output[:5])

np.random.seed(seed=0)
random.seed(0)

gauss_cache = 0.54

py_state = random.getstate()
state_tuple = py_state[1]
random.setstate((3, state_tuple, None))
np.random.set_state(('MT19937', state_tuple[:-1], state_tuple[-1], 0, gauss_cache))
mt = np.random.MT19937()
mt.state = ('MT19937', state_tuple[:-1], state_tuple[-1], 0, gauss_cache)
rs = np.random.RandomState(mt)
print(np.random.get_state(legacy=True))

print(np.random.random_sample())
print(random.random())
print(rs.random_sample())
print(np.random.random_sample())
print(random.random())
print(rs.random_sample())

print(np.random.rand())
print(random.random())
print(rs.rand())

#print(np.random.randn(1))
#print(random.gauss(mu=0, sigma=1))
#print(np.random.randn(1))
#print(random.gauss(mu=0, sigma=1))
#print(standard_normal())
#print(rs.standard_normal())
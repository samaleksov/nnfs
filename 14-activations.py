import math

def step(x):
    return 1 if x > 0 else 0

def linear(x):
    return x
    
def sigmoid(x):
    # e**-x == exp(-x)
    return 1/(1+math.e**-x)

def relu(x):
    return x if x > 0 else 0

print('Step(0.1)', step(0.1))
print('Step(0.0)', step(0.0))

print('Sigmoid(-5)', sigmoid(-5))
print('Sigmoid(5)', sigmoid(5))

print('ReLU(-1)', relu(-1))
print('ReLU(2)', relu(2))
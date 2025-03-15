import matplotlib.pyplot as plt
import numpy as np

# Line

def f(x):
    return 2*x

x = [0, 1, 2, 3, 4]
y = [f(x) for x in x]
"""
print(x)
print(y)
print('Rate of change:', (y[1]-y[0]) / (x[1]-x[0]))
plt.plot(x, y)
plt.show()
exit(1)
"""

# Quad

def f(x):
    return 2*x*x

y = [f(x) for x in x]

"""
print(x)
print(y)
plt.plot(x, y)
print(f'Rate of change [{x[0]}, {x[1]}]:', (y[1]-y[0]) / (x[1]-x[0]))
print(f'Rate of change [{x[2]}, {x[3]}]:', (y[3]-y[2]) / (x[3]-x[2]))
plt.show()
exit(1)
"""

# Instantaneous rate of change

p2_delta = 0.0001
x1 = 1
x2 = x1 + p2_delta

y1 = f(x1)
y2 = f(x2)

approximate_derivative = (y2-y1)/(x2-x1)
print(approximate_derivative)

x = [x / 1000 for x in range(0, 5000, 1)]
y = [f(x) for x in x]

plt.plot(x, y)

x1 = 2
x2 = x1 + p2_delta

y1 = f(x1)
y2 = f(x2)

approximate_derivative = (y2-y1)/(x2-x1)
print(approximate_derivative)
b = y2 - approximate_derivative*x2

def tangent_line(x):
    return approximate_derivative * x + b

to_plot = [x1-0.9, x1, x1+0.9]
plt.plot(to_plot, [tangent_line(i) for i in to_plot])

print('Approximate derivative for f(x)', f'where x = {x1} is {approximate_derivative}')

colors = ['k', 'g', 'r', 'b', 'c']

def approximate_tangent_line(x, approximate_derivative):
    return (approximate_derivative*x) + b

for i in range(5):
    x1 = i
    x2 = x1 + p2_delta
    y1 = f(x1)
    y2 = f(x2)
    approximate_derivative = (y2-y1)/(x2-x1)
    b = y2 - approximate_derivative*x2

    to_plot = [x1-0.9, x1, x1+0.9]
    plt.scatter(x1, y1, c=colors[i])
    plt.plot(to_plot, [approximate_tangent_line(point, approximate_derivative) for point in to_plot], c=colors[i])

    print('Approximate derivative for f(x)', f'where x = {x1} is {approximate_derivative}')

plt.show()
def dot(A, B):
    result = 0
    for a, b in zip(A, B):
        result += a*b
    return result

def sum(A, B):
    result = []
    for a, b in zip(A, B):
        result.append(a+b)
    return result

# scalar output - sum of component-wise multiplications
print(dot([1, 2, 3], [2, 3, 4]))

# vector output - component-wise sum
print(sum([1, 2, 3], [2, 3, 4]))
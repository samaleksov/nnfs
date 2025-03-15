scalar = 1
l = [1, 5, 6, 2]
lol = [[1, 5, 6, 2], [3, 2, 1, 3]]
lolol = [[[1, 5, 6, 2],
          [3, 2, 1, 3]],
         [[5, 2, 1, 2],
          [6, 4, 8, 4]],
         [[2, 8, 5, 3],
          [1, 1, 9, 4]]]

# 0 D tensor
print(scalar)

# 1 D tensor - 1 D vector or array
print(l)

# 2 D tensor - 2 D matrix - array of 1 D arrays
print(lol)

# 3 D tensor - array of 2 D matrices
print(lolol)

import numpy as np
print("Ai Initiated")
test_array = np.array([1,2,3])
print(test_array)
test_matrix = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9]
])
print(test_matrix)
print(test_matrix * 2)
print(test_matrix * np.array([1,0,0]))
print(test_matrix * np.array([1,0,2]))
print(np.array([1,0,2])*test_matrix)
print("jetzt kommt @")
print(test_matrix @ np.array([1,0,2]))
print("jetzt kommt dot")
print(test_matrix.dot(np.array([1,0,2])))

test_2_matrix = np.array([
    [2,0,1],
    [1,2,0],
    [0,1,2]
])

print("multiple matrix multiplication")
mul = test_matrix @ test_2_matrix
print(mul)
vec = np.array([1,2,3])
vec2 = mul @ vec
vec3 = test_matrix @ (test_2_matrix @ vec)
print("combined matrix:")
print(vec2)
print("after each othder:")
print(vec3)
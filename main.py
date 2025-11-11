import numpy as np
# print("Ai Initiated")
# test_array = np.array([1,2,3])
# print(test_array)
# test_matrix = np.array([
#     [1,2,3],
#     [4,5,6],
#     [7,8,9]
# ])
# print(test_matrix)
# print(test_matrix * 2)
# print(test_matrix * np.array([1,0,0]))
# print(test_matrix * np.array([1,0,2]))
# print(np.array([1,0,2])*test_matrix)
# print("jetzt kommt @")
# print(test_matrix @ np.array([1,0,2]))
# print("jetzt kommt dot")
# print(test_matrix.dot(np.array([1,0,2])))

# test_2_matrix = np.array([
#     [2,0,1],
#     [1,2,0],
#     [0,1,2]
# ])

# print("multiple matrix multiplication")
# mul = test_matrix @ test_2_matrix
# print(mul)
# vec = np.array([1,2,3])
# vec2 = mul @ vec
# vec3 = test_matrix @ (test_2_matrix @ vec)
# print("combined matrix:")
# print(vec2)
# print("after each othder:")
# print(vec3)

weights = {
    1: np.random.randn(4,3),
    2: np.random.randn(3,4),
}
biases = {
    1:np.zeros((4,1)),
    2:np.zeros((3,1))
}

def sigmoid(x):
    return 1/(1+np.exp(-x))

# a0 = np.array([[0.5],[0],[0.9]])
# z1 = weights[1] @ a0 + biases[1]
# a1 = sigmoid(z1)

# z2 = weights[2] @ a1 + biases[2]
# a2 = sigmoid(z2)
# print(a2)    


activations = {
    0: np.array([[0.5],[0],[0.1]])
}

for l in range(len(weights)):
    z = weights[l+1] @ activations[l] + biases[l+1]
    a = sigmoid(z)
    activations[l+1] = a

print(activations[len(weights)])
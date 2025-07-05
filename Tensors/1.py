import torch
import numpy as np

# Range tensor
# range_tensor = torch.range(1, 10, dtype=torch.float32) ## torch.range is deprecated and will be removed in a future release because its behavior is inconsistent with Python's range builtin.
range_tensor = torch.arange(1, 10, dtype=torch.float32)
print(f"Range tensor: {range_tensor} \n")

# Random range
random_tensor = torch.randn(20)
print(f"Random range: {random_tensor} \n")


# Convert numpy array to tensor
np_arr = np.random.randn(20)
np_to_tensor = torch.tensor(np_arr, dtype=torch.float32)
print(f"Numpy to tensor: {np_to_tensor}\n")

# Concat two tensor
tensor_1 = torch.arange(10, 20, dtype=torch.float32)
tensor_2 = torch.arange(20, 30, dtype=torch.float32)
combined = torch.cat([tensor_1, tensor_2])
print(f"Combined: {combined}\n")

# Zeros mat
zero_matrix = torch.zeros(3, 4)
print(f"Zeros: {zero_matrix}\n")

# Ones mat
one_matrix = torch.ones(2, 2) # 2 rows, 2 columns shape
print(f"Ones matrix: {one_matrix} \n")


# Create random mat in torch
normal_random_matrix = torch.randn(2, 10)
print(f"Random matrix: {normal_random_matrix} \n")


# identity  matrix
identity_matrix = torch.eye(3) # 3x3 identity matrix
print(f"Identity matrix: {identity_matrix} \n")


# Slice mat
print(f"Slice in torch its like python lists: {normal_random_matrix[0][:2]}")


# Operators
print(normal_random_matrix * 10)
print(normal_random_matrix // 10)
print(normal_random_matrix % 10)
print(normal_random_matrix + 10)
print(normal_random_matrix - 10)
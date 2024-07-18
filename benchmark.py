import time
import numpy as np
import hyperplexer
import matrix_multiplication_py

def generate_matrix(size):
    return [[i + j for j in range(size)] for i in range(size)]

def benchmark(func, mat1, mat2, label):
    start_time = time.time()
    result = func(mat1, mat2)
    end_time = time.time()
    for row in result[:2]:
        print(row[:2])  # Print only the first 10 columns
    return result, end_time - start_time

def benchmark_numpy(mat1, mat2, label):
    start_time = time.time()
    result = np.dot(mat1, mat2)
    end_time = time.time()
    for row in result[:2]:
        print(row[:2])  # Print only the first 10 columns
    return result, end_time - start_time

size = 2000  # Adjust size for higher computational demand
mat1 = generate_matrix(size)
mat2 = generate_matrix(size)

# Convert lists to NumPy arrays
mat1_np = np.array(mat1)
mat2_np = np.array(mat2)

# Benchmark C extension module
result_c, time_c = benchmark(hyperplexer.matrix_multiply, mat1, mat2, "C Extension")
print(f"C Extension Result Time: {time_c} seconds\n")

# Benchmark NumPy
result_np, time_np = benchmark_numpy(mat1_np, mat2_np, "NumPy")
print(f"NumPy Result Time: {time_np} seconds\n")

# Benchmark Python implementation
result_py, time_py = benchmark(matrix_multiplication_py.matrix_multiply, mat1, mat2, "Python")
print(f"Python Result Time: {time_py} seconds\n")


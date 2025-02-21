import numpy as np

# Function to generate random fp16 values
def generate_fp16_matrix(size):
    return np.random.rand(size, size).astype(np.float16)

# Function to save the matrix to a binary file
def save_matrix_to_bin(file_name, matrix):
    matrix.tofile(file_name)

# Function to perform matrix multiplication and truncate to specified size
def truncated_matrix_multiplication(matrix_a, matrix_b, size):
    truncated_a = matrix_a.flatten()[:size * size].reshape(size, size)
    truncated_b = matrix_b.flatten()[:size * size].reshape(size, size)
    result = np.matmul(truncated_a, truncated_b)
    return result.astype(np.float16)

# Generate and save the reference matrices for 128x128, 256x256, and 512x512 sizes
sizes = [128, 256, 512, 1024]
for s in sizes:
    np.random.seed(0)
    matrix_a = generate_fp16_matrix(s)
    matrix_b = generate_fp16_matrix(s)
    
    # Save the operand matrices to binary files
    save_matrix_to_bin("input.a.bin", matrix_a)
    save_matrix_to_bin(f"input.a.rand01.fp16.m{s}n{s}k{s}.row.bin", matrix_a)
    save_matrix_to_bin("input.b.bin", matrix_b)
    save_matrix_to_bin(f"input.b.rand01.fp16.m{s}n{s}k{s}.row.bin", matrix_b)

    ref_matrix = truncated_matrix_multiplication(matrix_a, matrix_b, s)
    save_matrix_to_bin(f"ref{s}.bin", ref_matrix)

print("All files generated successfully.")


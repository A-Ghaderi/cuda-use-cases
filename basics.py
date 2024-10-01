from numba import cuda
import numpy as np
import time
import math


# Initialize a large array on the CPU with ones
x_cpu = np.ones(shape=(100000000), dtype=np.float32)


# Function to increment each element of an array by one on the CPU
def host_increment_by_one(arr):
    for i in range(len(arr)):
        arr[i] += 1


# Uncomment these lines to test the CPU function
start = time.time()
result_cpu = host_increment_by_one(x_cpu)
print("CPU Time:", time.time() - start)


#################################################################################################################################################


# CUDA kernel to increment each element of a 1D array by one on the GPU
@cuda.jit
def device_increment_by_one(arr):
    pos = cuda.grid(1)
    if pos < arr.size:
        arr[pos] += 1


# Transfer the array data to the GPU
x_gpu = cuda.to_device(np.ones(shape=(100000000), dtype=np.float32))


# Define the number of threads per block and blocks per grid
threadsperblock = 256
blockspergrid = (x_cpu.size + (threadsperblock - 1)) // threadsperblock 


# Measure the execution time for incrementing the array on the GPU
print("GPU Time for 1D:")
cuda.synchronize()
start = time.time()
device_increment_by_one[blockspergrid, threadsperblock](x_gpu)
cuda.synchronize()
print(time.time() - start)


# Copy the result back to the CPU and print a small portion of it
result_1d = x_gpu.copy_to_host()
print("1D GPU Result (first 10 elements):", result_1d[:10])


#################################################################################################################################################


# CUDA kernel to increment each element of a 2D array by one on the GPU
@cuda.jit
def device_increment_2D_by_one(arr):
    x, y = cuda.grid(2)
    if x < arr.shape[0] and y < arr.shape[1]:
        arr[x, y] += 1


# Transfer the 2D array data to the GPU
x_gpu = cuda.to_device(np.ones(shape=(10000, 10000), dtype=np.float32))


# Define the number of threads per block and blocks per grid for 2D arrays
threadsperblock = (16, 16)
blockspergrid_x = math.ceil(x_gpu.shape[0] / threadsperblock[0])
blockspergrid_y = math.ceil(x_gpu.shape[1] / threadsperblock[1])
blockspergrid = (blockspergrid_x, blockspergrid_y)


# Measure the execution time for incrementing the 2D array on the GPU
print("GPU Time for 2D:")
cuda.synchronize()
start = time.time()
device_increment_2D_by_one[blockspergrid, threadsperblock](x_gpu)
cuda.synchronize()
print(time.time() - start)


# Copy the result back to the CPU and print a small portion of it
result_2d = x_gpu.copy_to_host()
print("2D GPU Result (first row, first 10 elements):", result_2d[0, :10])

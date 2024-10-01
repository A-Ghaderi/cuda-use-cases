import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
from pycuda.compiler import SourceModule


# Define CUDA kernel code
kernel_code = """
global void normalize(float *image, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;


    if (x < width && y < height) {
        image[idx] = image[idx] / 255.0f;  // Normalize to [0,1]
    }
}


global void convolve(float *image, float *output, float *kernel, int width, int height, int kernel_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;


    int k_half = kernel_size / 2;
    float value = 0.0f;


    if (x < width && y < height) {
        for (int ky = -k_half; ky <= k_half; ++ky) {
            for (int kx = -k_half; kx <= k_half; ++kx) {
                int ix = min(max(x + kx, 0), width - 1);
                int iy = min(max(y + ky, 0), height - 1);
                int image_idx = iy * width + ix;
                int kernel_idx = (ky + k_half) * kernel_size + (kx + k_half);
                value += image[image_idx] * kernel[kernel_idx];
            }
        }
        output[idx] = value;
    }
}


global void reduce_sum(float *image, float *result, int width, int height) {
    extern shared float shared_data[];


    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;


    shared_data[tid] = (idx < width * height) ? image[idx] : 0.0f;
    __syncthreads();


    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }


    if (tid == 0) {
        atomicAdd(result, shared_data[0]);
    }
}


global void custom_transform(float *image, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;


    if (x < width && y < height) {
        image[idx] = sqrt(image[idx]);  // Example transformation: square root
    }
}
"""

# Compile the kernel code
mod = SourceModule(kernel_code)


# Get kernel functions
normalize = mod.get_function("normalize")
convolve = mod.get_function("convolve")
reduce_sum = mod.get_function("reduce_sum")
custom_transform = mod.get_function("custom_transform")


# Example function to process a batch of images
def process_images_batch(image_batch, kernel):
    height, width = image_batch.shape[1], image_batch.shape[2]
    num_images = image_batch.shape[0]
    kernel_size = kernel.shape[0]


    # Allocate GPU memory
    image_gpu = cuda.mem_alloc(image_batch.nbytes)
    output_gpu = cuda.mem_alloc(image_batch.nbytes)
    kernel_gpu = cuda.mem_alloc(kernel.nbytes)
    result_gpu = cuda.mem_alloc(np.float32(0).nbytes)


    # Copy kernel to GPU
    cuda.memcpy_htod(kernel_gpu, kernel)


    # Block and grid dimensions
    block_size = (16, 16, 1)
    grid_size = ((width + block_size[0] - 1) // block_size[0],
                 (height + block_size[1] - 1) // block_size[1],
                 num_images)


    for i in range(num_images):
        # Copy image to GPU
        cuda.memcpy_htod(image_gpu, image_batch[i])


        # Normalize
        normalize(image_gpu, np.int32(width), np.int32(height), block=block_size, grid=grid_size)


        # Convolve
        convolve(image_gpu, output_gpu, kernel_gpu, np.int32(width), np.int32(height), np.int32(kernel_size), block=block_size, grid=grid_size)


        # Custom transform
        custom_transform(output_gpu, np.int32(width), np.int32(height), block=block_size, grid=grid_size)


        # Reduce sum
        result = np.array([0], dtype=np.float32)
        cuda.memcpy_htod(result_gpu, result)
        reduce_sum(output_gpu, result_gpu, np.int32(width), np.int32(height), block=(256, 1, 1), grid=((width * height + 255) // 256, 1, 1), shared=256 * 4)


        # Copy result back to CPU
        cuda.memcpy_dtoh(result, result_gpu)
        print(f"Sum of pixels for image {i}: {result[0]}")


        # Copy processed image back to CPU
        cuda.memcpy_dtoh(image_batch[i], output_gpu)


# Example usage
if name == "__main__":
    # Create a batch of random images (e.g., 10 images of 256x256 pixels)
    image_batch = np.random.rand(10, 256, 256).astype(np.float32)


    # Example kernel (e.g., 3x3 Gaussian blur)
    kernel = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]], dtype=np.float32).flatten()
    kernel /= kernel.sum()  # Normalize kernel


    # Process the images
    process_images_batch(image_batch, kernel)


    # Output the processed images
    print("Processed images:", image_batch)

#include <stdio.h>
# define BLOCKSIZE 256


// define the kernel function: __global__ indicates that this function will be executed on the GPU
// and can be called from the host (CPU) code.
__global__ void square_coalesced(const float* in_array, float* out_array, const int N) {
    // Get the thread ID, block ID, and block size. These are set by the CUDA runtime when launching the kernel.
    const int threadid = threadIdx.x;
    const int blockid = blockIdx.x;
    const int globalid = (blockid * BLOCKSIZE + threadid);

    if (globalid >= N) {
        return; // Out of bounds
    }

    // load one float32 from the input array
    // Note: this is a coalesced write, as the intput array is accessed in a contiguous manner: 
    // The loads of thread 0:8 will be coalesced into one transaction of 32 bytes
    // The loads of thread 8:16 will be coalesced into one transaction of 32 bytes
    // The loads of thread 16:24 will be coalesced into one transaction of 32 bytes
    // The loads of thread 24:32 will be coalesced into one transaction of 32 bytes
    // ...
    const float in = in_array[globalid];

    // perform the square operation
    const float out = in * in;

    // store the result in the output array
    // Note: this is a coalesced write, as the output array is accessed in a contiguous manner
    out_array[globalid] = out;
}


int main() {
    // Define the size of the input and output arrays
    const int N = 1024*1024*1024;

    // Define host and device pointers for the input and output arrays
    float *in_array, *out_array;
    float *d_in_array, *d_out_array;

    // Allocate host memory
    in_array = (float*) malloc(N * sizeof(float));
    out_array = (float*) malloc(N * sizeof(float));

    // Initialize input array with non-compressable data
    for (int i = 0; i < N; i++) {
        in_array[i] = static_cast<float>(i);
    }

    // Allocate device memory
    cudaMalloc((void**)&d_in_array, N * sizeof(float));
    cudaMalloc((void**)&d_out_array, N * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_in_array, in_array, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel with 256 threads per block and enough blocks to cover the array.
    int num_blocks = (N + (BLOCKSIZE)-1) / (BLOCKSIZE);

    // warm up runs
    square_coalesced<<<num_blocks, BLOCKSIZE>>>(d_in_array, d_out_array, N);
    square_coalesced<<<num_blocks, BLOCKSIZE>>>(d_in_array, d_out_array, N);
    square_coalesced<<<num_blocks, BLOCKSIZE>>>(d_in_array, d_out_array, N);
    cudaDeviceSynchronize(); // wait for the kernel to finish

    // Measure the time taken to execute the kernel
    // Note: you do not need to understand the time measurement code
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    square_coalesced<<<num_blocks, BLOCKSIZE>>>(d_in_array, d_out_array, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("Time taken: %f ms\n", milliseconds);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    }


    // Free device memory
    cudaFree(d_in_array);
    cudaFree(d_out_array);

    // Free host memory
    free(in_array);
    free(out_array);

    return 0;
}
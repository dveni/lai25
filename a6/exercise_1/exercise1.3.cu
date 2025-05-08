#include <stdio.h>


# define EL_PER_THREAD 4
# define BLOCKSIZE 256

__global__ void square_uncoalesced(const float* in_array, float* out_array, const int N) {
    const int threadid = threadIdx.x;
    const int blockid = blockIdx.x;
    const int blocksize = blockDim.x;
    const int globalid = (blockid * blocksize + threadid) * EL_PER_THREAD;

    if (globalid >= N) {
        return; // Out of bounds
    }

    float in[EL_PER_THREAD];

    for (int i = 0; i < EL_PER_THREAD; i++) { //assume N is a multiple of 4
        in[i] = in_array[globalid + i];
        in[i] = in[i] * in[i];
        out_array[globalid + i] = in[i];
    }
}

// TODO change the indexing on this kernel to make the loads and stores coalesced
__global__ void square_coalesced(const float* in_array, float* out_array, const int N) {
    const int threadid = threadIdx.x;
    const int blockid = blockIdx.x;
    const int blocksize = blockDim.x;
    const int globalid = (blockid * blocksize + threadid);

    if (globalid >= N) {
        return; // Out of bounds
    }

    float in[EL_PER_THREAD];

    for (int i = 0; i < EL_PER_THREAD; i++) { // TODO!
        in[i] = in_array[globalid + i*EL_PER_THREAD];
        in[i] = in[i] * in[i];
        out_array[globalid + i*EL_PER_THREAD] = in[i];
    }
}


int main() {
    const int N = 1024*1024*1024;
    float *in_array, *out_array, *out_array_ref;
    float *d_in_array, *d_out_array;

    // Allocate host memory
    in_array = (float*) malloc(N * sizeof(float));
    out_array = (float*) malloc(N * sizeof(float));
    out_array_ref = (float*) malloc(N * sizeof(float));

    // Initialize input array
    for (int i = 0; i < N; i++) {
        in_array[i] = static_cast<float>(i);
    }

    // Allocate device memory
    cudaMalloc((void**)&d_in_array, N * sizeof(float));
    cudaMalloc((void**)&d_out_array, N * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_in_array, in_array, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel with 256 threads per block and enough blocks to cover the array
    int num_blocks = (N + (BLOCKSIZE*EL_PER_THREAD)-1) / (BLOCKSIZE*EL_PER_THREAD);
    square_uncoalesced<<<num_blocks, BLOCKSIZE>>>(d_in_array, d_out_array, N);

    // copy back
    cudaMemcpy(out_array_ref, d_out_array, N * sizeof(float), cudaMemcpyDeviceToHost);

    
    square_coalesced<<<num_blocks, BLOCKSIZE>>>(d_in_array, d_out_array, N);
    square_coalesced<<<num_blocks, BLOCKSIZE>>>(d_in_array, d_out_array, N);
    square_coalesced<<<num_blocks, BLOCKSIZE>>>(d_in_array, d_out_array, N);

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


    // copy back
    cudaMemcpy(out_array, d_out_array, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_in_array);
    cudaFree(d_out_array);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    }

    // Verify results
    for (int i = 0; i < N; i++) {
        if (out_array[i] != out_array_ref[i]) {
            printf("Mismatch at index %d: %f != %f\n", i, out_array[i], out_array_ref[i]);
            free(in_array);
            free(out_array);
            break;
        }
    }

    // Free host memory
    free(in_array);
    free(out_array);

    return 0;
}

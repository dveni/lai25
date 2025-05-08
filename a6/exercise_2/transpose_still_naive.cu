#include <stdio.h>

const int TILE_DIM = 32;
const int BLOCK_ROWS = 1;

__global__ void transposeCoalesced(const float *idata, float *odata)
{  
   // Allocate shared memory for the tile that is being loaded by one block
   __shared__ float tile[TILE_DIM][TILE_DIM+1];

   // Calculate the global thread index to start loading data at
   int x = blockIdx.x * TILE_DIM + threadIdx.x;
   int y = blockIdx.y * TILE_DIM + threadIdx.y;

   // Calculate the width of the entire grid in terms of number of elements
   int width = gridDim.x * TILE_DIM;

   // each thread loads one element of the tile per iteration in contiguous (coalesced) manner.
   // We do this loading TILE_DIM/BLOCK_ROWS many times.
   for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
      tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x];

   // this instruction syncronizes all threads in the block to make sure all of the 
   // data is loaded into shared memory before we start writing it back to global memory
   __syncthreads(); 

   x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
   y = blockIdx.x * TILE_DIM + threadIdx.y;

   // Now we write the transposed data back to global memory
   // each thread writes to the same location in global memory tile that it loaded from in the data-loading loop on line 20
   // but the tile gets transposed because we switch the shared memory indices
   for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
      odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
}


int main(){
   const int N = 1024*32;
   const int M = 1024*32;

   float *in_array, *out_array;
   float *d_in_array, *d_out_array;

   // Allocate host memory
   in_array = (float*) malloc(N * M * sizeof(float));
   out_array = (float*) malloc(N * M * sizeof(float));

   // Initialize input array
   for (int i = 0; i < N * M; i++) {
       in_array[i] = static_cast<float>(i);
   }

   // Allocate device memory
   cudaMalloc((void**)&d_in_array, N * M * sizeof(float));
   cudaMalloc((void**)&d_out_array, N * M * sizeof(float));

   // Copy input data to device
   cudaMemcpy(d_in_array, in_array, N * M * sizeof(float), cudaMemcpyHostToDevice);

   // Launch kernel with 256 threads per block and enough blocks to cover the array
   dim3 dimGrid(N/TILE_DIM, M/TILE_DIM, 1);
   dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);
   transposeCoalesced<<<dimGrid, dimBlock>>>(d_in_array, d_out_array);
   transposeCoalesced<<<dimGrid, dimBlock>>>(d_in_array, d_out_array);
   transposeCoalesced<<<dimGrid, dimBlock>>>(d_in_array, d_out_array);
   cudaDeviceSynchronize();

   // time the kernel execution
   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   cudaEventRecord(start, 0);
   transposeCoalesced<<<dimGrid, dimBlock>>>(d_in_array, d_out_array);
   cudaEventRecord(stop, 0);
   cudaEventSynchronize(stop);
   float milliseconds = 0;
   cudaEventElapsedTime(&milliseconds, start, stop);
   printf("Time taken for kernel execution: %f ms\n", milliseconds);
   cudaEventDestroy(start);
   cudaEventDestroy(stop);


   // Check for errors in kernel launch
   cudaError_t err = cudaGetLastError();
   if (err != cudaSuccess) {
       fprintf(stderr, "Error launching kernel: %s\n", cudaGetErrorString(err));
       return -1;
   }
   // Synchronize device
   cudaDeviceSynchronize();

   // Copy output data back to host
   cudaMemcpy(out_array, d_out_array, N * M * sizeof(float), cudaMemcpyDeviceToHost);

   // Verify results
   for (int i = 0; i < N; i++) {
      for (int j=0; j < M; j++){
          if (out_array[i*M + j] != in_array[j*N + i]) {
              printf("Mismatch at index %d: %f != %f\n", i, out_array[i*M + j], in_array[j*N + i]);
              break;
          }
      }
    }

   // Free device memory
   cudaFree(d_in_array);
   cudaFree(d_out_array);

   // Free host memory
   free(in_array);
   free(out_array);

   return 0;
}

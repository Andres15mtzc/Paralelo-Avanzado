#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define GPUErrorAssertion(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void unrolling2(int* input, int* temp, int size) {
    int tid = threadIdx.x;
    int BLOCK_OFFSET = blockIdx.x * blockDim.x * 2;
    int index = BLOCK_OFFSET + tid;

    int* i_data = input + BLOCK_OFFSET;

    if ((index + blockDim.x) < size) {
        input[index] += input[index + blockDim.x];
    }

    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset / 2) {
        if (tid < offset) {
            i_data[tid] += i_data[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        temp[blockIdx.x] = i_data[0];
    }


}

__global__ void transpose(int* input, int* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int index_in = y * width + x;
        int index_out = x * height + y;
        output[index_out] = input[index_in];
    }
}

int main() {
    int data_size = 1 << 10;
    int byte_size = data_size * sizeof(int);
    int block_size = 32;
    int parallel_reduction = 2;

    int* h_input, * h_ref;
    h_input = (int*)malloc(byte_size);

    for (int i = 0; i < data_size; i++) h_input[i] = (double)(rand() % 10);

    dim3 block(block_size);
    dim3 grid((data_size / byte_size) / parallel_reduction);

    int temp = sizeof(int) * grid.x;
    h_ref = (int*)malloc(temp);

    int* d_input, * d_temp;

    cudaMalloc((void**)&d_input, byte_size);
    cudaMalloc((void**)&d_temp, temp);

    cudaMemset(d_temp, 0, temp);
    cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice);

    if (parallel_reduction == 2)
        unrolling2 << < grid, block >> > (d_input, d_temp, data_size);

    cudaDeviceSynchronize();
    cudaMemcpy(h_ref, d_temp, temp, cudaMemcpyDeviceToHost);

    int gpu_result = 0;
    for (int i = 0; i < grid.x; i++) {
        gpu_result += h_ref[i];
    }

    cudaFree(d_input);
    cudaFree(d_temp);
    free(h_input);
    free(h_ref);
    
    return 0;
}
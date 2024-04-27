﻿#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>

#define MATRIX_SIZE 4

__global__ void dot_prodcut(int* a, int* b, int* c, int size)
{
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;

    if (row < MATRIX_SIZE && col < MATRIX_SIZE)
    {
        int line = 0;
        for (int x = 0; x < 4; x++) {
            line += a[row * MATRIX_SIZE + x] * b[x * MATRIX_SIZE + col];
        }

        c[row * MATRIX_SIZE + col] = line;

    }
}

int main()
{
    const int matrix_size = MATRIX_SIZE * MATRIX_SIZE;
    dim3 blockSize(4, 4);
    dim3 gridSize((MATRIX_SIZE + blockSize.x - 1) / blockSize.x, (MATRIX_SIZE + blockSize.y - 1) / blockSize.y);

    int* a_cpu = (int*)malloc(matrix_size * sizeof(int));
    int* b_cpu = (int*)malloc(matrix_size * sizeof(int));
    int* c_cpu = (int*)malloc(matrix_size * sizeof(int));

    int* a_device;
    int* b_device;
    int* c_device;

    cudaMalloc((void**)&a_device, matrix_size * sizeof(int));
    cudaMalloc((void**)&b_device, matrix_size * sizeof(int));
    cudaMalloc((void**)&c_device, matrix_size * sizeof(int));

    // Inicializar matrices en el host
    for (int i = 0; i < matrix_size; ++i)
    {
        a_cpu[i] = 1;
        b_cpu[i] = 2;
    }

    // Transferir datos del host al dispositivo
    cudaMemcpy(a_device, a_cpu, matrix_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(b_device, b_cpu, matrix_size * sizeof(int), cudaMemcpyHostToDevice);

    // Lanzar el kernel
    dot_prodcut << <gridSize, blockSize >> > (a_device, b_device, c_device, matrix_size);

    // Transferir datos del dispositivo al host
    cudaMemcpy(c_cpu, c_device, matrix_size * sizeof(int), cudaMemcpyDeviceToHost);

    // Imprimir resultados
    printf("Matrix A:\n");
    for (int i = 0; i < matrix_size; ++i)
    {
        printf("%d ", a_cpu[i]);
        if ((i + 1) % 4 == 0)
            printf("\n");
    }

    printf("\nMatrix B:\n");
    for (int i = 0; i < matrix_size; ++i)
    {
        printf("%d ", b_cpu[i]);
        if ((i + 1) % 4 == 0)
            printf("\n");
    }

    printf("\nMatrix C (Result):\n");
    for (int i = 0; i < matrix_size; ++i)
    {
        printf("%d ", c_cpu[i]);
        if ((i + 1) % 4 == 0)
            printf("\n");
    }

    // Liberar memoria
    free(a_cpu);
    free(b_cpu);
    free(c_cpu);

    cudaFree(a_device);
    cudaFree(b_device);
    cudaFree(c_device);

    cudaDeviceReset();

    return 0;
}
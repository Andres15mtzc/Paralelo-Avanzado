#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

__global__ void suma_arreglos(const int *arr1, const int *arr2, const int *arr3, int *arrFinal)
{
    int globalID = (blockIdx.z * blockDim.z + threadIdx.z) * (gridDim.y * blockDim.y) * (gridDim.x * blockDim.x) + (blockIdx.y * blockDim.y + threadIdx.y) * (gridDim.x * blockDim.x) + blockIdx.x * blockDim.x + threadIdx.x;

    arrFinal[globalID] = arr1[globalID] + arr2[globalID] + arr3[globalID];
    printf("[DEVICE] GlobalId: %d   (%d,%d,%d)    Val: %d\n", globalID, arr1[globalID], arr2[globalID], arr3[globalID], arrFinal[globalID]);
}

int main()
{
    dim3 blockSize(10, 10, 5);
    dim3 gridSize(5, 2, 2);

    // Llené los arreglos con valores predeterminados
    const int arraySize = 10000;
    int arr1[arraySize] = { 0 };
    std::fill(std::begin(arr1), std::end(arr1), 1);
    int arr2[arraySize] = { 0 };
    std::fill(std::begin(arr2), std::end(arr2), 2);
    int arr3[arraySize] = { 0 };
    std::fill(std::begin(arr3), std::end(arr3), 3);
    int arrFinal[arraySize] = { 0 };

    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
    int* dev_d = 0;

    cudaMalloc((void**)&dev_d, arraySize * sizeof(int));
    cudaMalloc((void**)&dev_a, arraySize * sizeof(int));
    cudaMalloc((void**)&dev_b, arraySize * sizeof(int));
    cudaMalloc((void**)&dev_c, arraySize * sizeof(int));

    cudaMemcpy(dev_a, arr1, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, arr2, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, arr3, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    suma_arreglos << <gridSize, blockSize >> > (dev_a, dev_b, dev_c, dev_d);

    cudaMemcpy(arrFinal, dev_d, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

    cudaDeviceReset();
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    cudaFree(dev_d);

    return 0;
}

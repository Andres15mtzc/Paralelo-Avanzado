#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void print_hello_cuda()
{
    int globalID = (blockIdx.y * blockDim.y + threadIdx.y) * (gridDim.x * blockDim.x) + blockIdx.x * blockDim.x + threadIdx.x;
    printf("[DEVICE] ThreadIsx.x: %d\n", globalID);
}

int main()
{
    print_hello_cuda << <2, 8 >> > ();

    return 0;
}
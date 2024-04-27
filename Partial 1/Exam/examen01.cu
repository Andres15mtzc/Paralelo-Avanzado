
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void orBitToBit(const unsigned char* A, const unsigned char* B, unsigned char* C, int size) {
    int ID = blockIdx.x * blockDim.x + threadIdx.x;
    if (ID < size) {
        C[ID] = A[ID] | B[ID];
    }
}

int main()
{
    const int threadsNo = 1024;
    const int size_in_bytes = 1000; 
    int size = size_in_bytes / sizeof(unsigned char); 

    unsigned char* A = (unsigned char*)malloc(size_in_bytes);
    unsigned char* B = (unsigned char*)malloc(size_in_bytes);
    unsigned char* C = (unsigned char*)malloc(size_in_bytes);

    for (int i = 0; i < size; ++i) {
        A[i] = i % 127;
        B[i] = (i + 1) % 127;
    }

    unsigned char* dev_A;
    unsigned char* dev_B;
    unsigned char* dev_C;

    cudaMalloc((void**)&dev_A, size_in_bytes);
    cudaMalloc((void**)&dev_B, size_in_bytes);
    cudaMalloc((void**)&dev_C, size_in_bytes);

    cudaMemcpy(dev_A, A, size_in_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, size_in_bytes, cudaMemcpyHostToDevice);

    int Nblocks = (size + threadsNo - 1) / threadsNo;

    orBitToBit << <Nblocks, threadsNo >> > (dev_A, dev_B, dev_C, size);

    cudaMemcpy(C, dev_C, size_in_bytes, cudaMemcpyDeviceToHost);

    printf("Resultado de la operación OR bit a bit:\n");
    for (int i = 0; i < size; ++i) {
        printf("%u | %u = %u\n", A[i], B[i], C[i]);
    }

    // Calculos

    /*
    * En un kernel de CUDA se realiza una operación OR bit a bit entre dos vectores. 
    * Los arreglos A y B son los dos vectores de entrada, y C es la salida de N-Size 
    * Todas los arreglos constan de enteros sin signo de 8 bits. La longitud de los arreglos en bytes viene dada por el parámetro size_in_bytes.
    * Si el compute capability 7.5 admite un máximo de 1024 hilos por SM, un máximo de 16 bloques por SM, y un máximo de 46 SM ¿cual es total de hilos que se pueden usar, 
    * y cuantos hilos por bloque se necesitan para aprovechar al maximo los recursos? ¿Cuánto vale N-size?
    */

    int noSM = 46;
    int noBlock_SM = 16;

    int totalBlocks = noSM * noBlock_SM;

    int noThreads_SM = 1024;

    int totalThreads = noSM * noThreads_SM;

    int noThreads_Block = totalThreads / totalBlocks;

    printf("\nNumero de SM: %d\n", noSM);
    printf("Numero de bloques por SM: %d\n", noBlock_SM);
    printf("Numero de hilos por SM: %d\n", noThreads_SM);

    printf("Total de bloques (SM * bloques por SM): %d\n", totalBlocks);

    printf("\nRESULTADO\nTotal de hilos (SM * hilos por SM): %d\n", totalThreads);
    printf("Numero de hilos por bloque (total de hilos * total de bloques): %d\n", noThreads_Block);

    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);

    free(A);
    free(B);
    free(C);

    return 0;
}

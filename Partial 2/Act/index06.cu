
#include "device_launch_parameters.h"

#include <iostream>
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include <cuda_runtime_api.h>

using namespace std;

int partition(int arr[], int start, int end)
{
    int pivot = arr[end];
    int pIndex = start;

    for (int i = start; i < end; i++) {
        if (arr[i] <= pivot) {
            swap(arr[i], arr[pIndex]);
            pIndex++;
        }
    }

    swap(arr[pIndex], arr[end]);

    return pIndex;
}

void quickSort(int arr[], int start, int end) {
    
    if (start >= end) {
        return;
    }

    int pivot = partition(arr, start, end);

    quickSort(arr, start, pivot - 1);

    quickSort(arr, pivot + 1, end);

}

int main()
{
    int arr[] = { 9, -3, 5, 2, 6, 8, -6, 1, 3, 5, 7, 10, 12 };
    int n = sizeof(arr) / sizeof(arr[0]);

    quickSort(arr, 0, n - 1);

    // imprime la array ordenada
    for (int i = 0; i < n; i++) {
        cout << arr[i] << " ";
    }

    return 0;
}
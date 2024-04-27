#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iostream>
#include <cstdlib>
#include <ctime>

#define WIDTH 1024
#define HEIGHT 1024

using namespace std;

void transpose(int* in, int* out, int w, int h) {
    for (int x = 0; x < w; ++x) {
        for (int y = 0; y < h; ++y) {
            int i_in = y * w + x;
            int i_out = x * h + y;
            out[i_out] = in[i_in];
        }
    }
}

int main() {
    const int size = WIDTH * HEIGHT;

    int* M = new int[size];
    int* T = new int[size];

    // Initialize matrix with random values
    srand(time(NULL));
    for (int i = 0; i < size; ++i) {
        M[i] = rand() % 9;
    }

    cout << "Original matrix: \n";
    for (int i = 0; i < 20; ++i) {
        cout << "M[" << i << "] = " << M[i] << endl;
    }

    transpose(M, T, WIDTH, HEIGHT);

    cout << "Transposed matrix: \n";
    for (int i = 0; i < 20; ++i) {
        cout << "T[" << i << "] = " << T[i] << endl;
    }

    delete[] M;
    delete[] T;

    return 0;
}
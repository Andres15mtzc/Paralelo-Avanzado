#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cstdlib>
#include <ctime>

#define WIDTH 1024
#define HEIGHT 1024

using namespace std;

void conv2D(int* mat, int* r, int w, int h) {
    for (int row = 0; row < h; ++row) {
        for (int col = 0; col < w; ++col) {
            int suma = 0;
            for (int i = -1; i <= 1; ++i) {
                for (int j = -1; j <= 1; ++j) {
                    int curRow = row + i;
                    int curCol = col + j;
                    if (curRow >= 0 && curRow < h && curCol >= 0 && curCol < w) {
                        suma += mat[curRow * w + curCol];
                    }
                }
            }
            r[row * w + col] = suma;
        }
    }
}

int main() {
    const int size = WIDTH * HEIGHT;
    int* M_bef = new int[size];
    int* M_conv = new int[size];

    srand(time(NULL));
    for (int i = 0; i < size; ++i) {
        M_bef[i] = rand() % 9;
    }

    cout << "M_before: \n";
    for (int i = 0; i < 20; ++i) {
        cout << "M_bef[" << i << "] = " << M_bef[i] << endl;
    }

    conv2D(M_bef, M_conv, WIDTH, HEIGHT);

    cout << "M_convolution: \n";
    for (int i = 0; i < 20; ++i) {
        cout << "M_conv[" << i << "] = " << M_conv[i] << endl;
    }

    delete[] M_bef;
    delete[] M_conv;

    return 0;
}
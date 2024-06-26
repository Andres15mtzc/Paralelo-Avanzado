﻿// Author: Andrés Martínez Cabrera 

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <vector>
#include <cassert>

__global__ void es_seguro_parallel(char* tablero, int* cond1, int* cond2, int* cond3, int* fila, int* columna, char* num);

class ValidateSudoku {
private:
    std::vector<std::vector<char>> tablero;
    std::vector<char> lista_invertida;

    bool encontrar_casilla_vacia(int& fila, int& columna) {
        for (fila = 0; fila < 9; ++fila) {
            for (columna = 0; columna < 9; ++columna) {
                if (tablero[fila][columna] == '.') {
                    return true; // Encontrar la primera casilla vacía
                }
            }
        }
        return false; // No hay casillas vacías
    }

    bool en_fila(int fila, char num) {
        for (int columna = 0; columna < 9; ++columna) {
            if (tablero[fila][columna] == num) {
                return true; // El número ya está en la fila
            }
        }
        return false;
    }

    bool en_columna(int columna, char num) {
        for (int fila = 0; fila < 9; ++fila) {
            if (tablero[fila][columna] == num) {
                return true; // El número ya está en la columna
            }
        }
        return false;
    }

    bool en_subcuadro(int fila_inicio, int columna_inicio, char num) {
        for (int fila = 0; fila < 3; ++fila) {
            for (int columna = 0; columna < 3; ++columna) {
                if (tablero[fila + fila_inicio][columna + columna_inicio] == num) {
                    return true; // El número ya está en el subcuadro
                }
            }
        }
        return false;
    }

    bool es_seguro(int fila, int columna, char num) {
        return !en_fila(fila, num) && !en_columna(columna, num) &&
            !en_subcuadro(fila - fila % 3, columna - columna % 3, num);
    }

    bool es_seguro_gpu(int h_fila, int h_columna, char h_num) {
        std::vector<char> linear_board;
        for (const auto& row : tablero) {
            linear_board.insert(linear_board.end(), row.begin(), row.end());
        }
        char* d_board;
        cudaMalloc((void**)&d_board, 81 * sizeof(char));
        cudaMemcpy(d_board, linear_board.data(), 81 * sizeof(char), cudaMemcpyHostToDevice);

        int* d_solved_fila, * d_solved_col, * d_solved_sub, * d_fila, * d_columna;
        char* d_num;
        cudaMalloc((void**)&d_solved_fila, sizeof(int));
        cudaMalloc((void**)&d_solved_col, sizeof(int));
        cudaMalloc((void**)&d_solved_sub, sizeof(int));
        cudaMalloc((void**)&d_fila, sizeof(int));
        cudaMalloc((void**)&d_columna, sizeof(int));
        cudaMalloc((void**)&d_num, sizeof(char));
        int h_solved_fila = 0;
        int h_solved_col = 0;
        int h_solved_sub = 0;
        cudaMemcpy(d_solved_fila, &h_solved_fila, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_solved_col, &h_solved_col, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_solved_sub, &h_solved_sub, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_fila, &h_fila, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_columna, &h_columna, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_num, &h_num, sizeof(char), cudaMemcpyHostToDevice);

        es_seguro_parallel << <32, 1 >> > (d_board, d_solved_fila, d_solved_col, d_solved_sub, d_fila, d_columna, d_num);

        cudaMemcpy(&h_solved_fila, d_solved_fila, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_solved_col, d_solved_col, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_solved_sub, d_solved_sub, sizeof(int), cudaMemcpyDeviceToHost);

        cudaFree(d_board);
        cudaFree(d_solved_fila);
        cudaFree(d_solved_col);
        cudaFree(d_solved_sub);
        cudaFree(d_fila);
        cudaFree(d_columna);
        cudaFree(d_num);

        return (h_solved_fila == 0 && h_solved_col == 0 && h_solved_sub == 0);
    }

public:
    ValidateSudoku(std::vector<std::vector<char>> tablero) : tablero(tablero) {}

    void chequeo_general() {
        assert(tablero.size() == 9 && "El tablero ingresado no respeta el formato 9x9");

        for (const auto& fila : tablero) {
            assert(fila.size() == 9 && "El tablero ingresado no respeta el formato 9x9");
        }
    }

    void chequeo_filas(const std::vector<std::vector<char>>& lista_a_chequear = {}) {
        const auto& lista = lista_a_chequear.empty() ? tablero : lista_a_chequear;

        for (const auto& fila : lista) {
            for (char elemento : fila) {
                if (elemento != '.') {
                    assert(std::count(fila.begin(), fila.end(), elemento) == 1 && "El tablero ingresado no es válido");
                }
            }
        }
    }

    void chequeo_columnas() {
        for (int column_index = 0; column_index < 9; ++column_index) {
            lista_invertida.clear();
            for (int row_index = 0; row_index < 9; ++row_index) {
                lista_invertida.push_back(tablero[row_index][column_index]);
            }
            chequeo_filas({ lista_invertida });
        }
    }

    void chequeo_subcuadros() {
        chequeo_3_subcuadros(0, 3);
        chequeo_3_subcuadros(3, 6);
        chequeo_3_subcuadros(6, 9);
    }

    void chequeo_3_subcuadros(int rango1, int rango2) {
        lista_invertida.clear();
        for (int row_index = 0; row_index < 9; ++row_index) {
            if (row_index == 3 || row_index == 6) {
                lista_invertida.clear();
            }
            for (int column_index = rango1; column_index < rango2; ++column_index) {
                lista_invertida.push_back(tablero[column_index][row_index]);
                if (lista_invertida.size() == 9) {
                    chequeo_filas({ lista_invertida });
                }
            }
        }
    }

    bool solucionar_sudoku() {
        // Lógica de resolución de Sudoku 
        int fila, columna;
        if (!encontrar_casilla_vacia(fila, columna)) {
            return true; // Si no hay casillas vacías, el Sudoku ya está resuelto
        }

        for (char num = '1'; num <= '9'; ++num) {
            if (es_seguro(fila, columna, num)) {
                tablero[fila][columna] = num; // Asignar el número si es seguro

                if (solucionar_sudoku()) {
                    return true; // Si el número asignado lleva a una solución, retornar true
                }

                tablero[fila][columna] = '.'; // Si el número no lleva a una solución, deshacer el cambio
            }
        }
        return false; // No hay solución para este estado del Sudoku
    }

    bool solucionar_sudoku_parallel() {
        // Lógica de resolución de Sudoku 
        int fila, columna;
        if (!encontrar_casilla_vacia(fila, columna)) {
            return true; // Si no hay casillas vacías, el Sudoku ya está resuelto
        }

        for (char num = '1'; num <= '9'; ++num) {
            if (es_seguro_gpu(fila, columna, num)) {
                tablero[fila][columna] = num; // Asignar el número si es seguro

                if (solucionar_sudoku_parallel()) {
                    return true; // Si el número asignado lleva a una solución, retornar true
                }

                tablero[fila][columna] = '.'; // Si el número no lleva a una solución, deshacer el cambio
            }
        }
        return false; // No hay solución para este estado del Sudoku
    }

    bool solucionar_sudoku_parallel_montecarlo(double p) {
        // Lógica de resolución de Sudoku 
        int fila, columna;
        if (!encontrar_casilla_vacia(fila, columna)) {
            return true;
        }

        for (char num = '1'; num <= '9'; ++num) {
            if (es_seguro_gpu(fila, columna, num)) {
                tablero[fila][columna] = num;
                if (solucionar_sudoku_parallel_montecarlo(p) && rand() / (double)RAND_MAX < p) {
                    return true;
                }
                tablero[fila][columna] = '.';
            }
        }
        return false;

    }

    void imprimir_tablero() {
        for (const auto& fila : tablero) {
            for (char c : fila) {
                std::cout << c << " ";
            }
            std::cout << std::endl;
        }
    }

};

__global__ void es_seguro_parallel(char* tablero, int* cond1, int* cond2, int* cond3, int* fila, int* columna, char* num) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= 0 && idx < 9) {
        if (tablero[9 * *fila + idx] == *num) *cond1 = 1;
        return;
    }
    else if (idx >= 9 && idx < 18) {
        if (tablero[9 * (idx - 9) + *columna] == *num) *cond2 = 1;
        return;
    }
    else if (idx >= 18 && idx < 27) {
        int row = *fila - *fila % 3;
        int col = (idx - 18) % 3 + *columna - *columna % 3;
        if (tablero[9 * row + col] == *num) *cond3 = 1;
        return;
    }
    return;
}

int main()
{
    std::vector<std::vector<char>> board = {
        {'5', '3', '4', '6', '7', '8', '.', '1', '.'},
        {'6', '.', '.', '1', '9', '5', '.', '.', '.'},
        {'.', '9', '8', '.', '.', '.', '.', '6', '.'},
        {'8', '.', '.', '.', '6', '.', '.', '.', '3'},
        {'4', '.', '.', '8', '.', '3', '.', '.', '1'},
        {'7', '.', '.', '.', '2', '.', '.', '.', '6'},
        {'.', '6', '.', '.', '.', '.', '2', '8', '.'},
        {'.', '.', '.', '4', '1', '9', '.', '.', '5'},
        {'.', '.', '.', '.', '8', '.', '.', '7', '9'}
    };

    ValidateSudoku sudoku(board);
    sudoku.chequeo_general();
    sudoku.chequeo_filas();
    sudoku.chequeo_columnas();
    sudoku.chequeo_subcuadros();
    std::cout << "El tablero de Sudoku ingresado es válido" << std::endl;

    //CPU
    printf("\nCPU\n");
    clock_t cpu_start, cpu_stop;
    double cpu_time_taken;

    cpu_start = clock();
    if (sudoku.solucionar_sudoku()) {
        std::cout << "El Sudoku ha sido resuelto:" << std::endl;
        sudoku.imprimir_tablero();
    }
    else {
        std::cout << "No hay solución para el Sudoku proporcionado." << std::endl;
    }
    cpu_stop = clock();

    cpu_time_taken = ((double)(cpu_stop - cpu_start)) / CLOCKS_PER_SEC;
    printf("El tiempo de ejecucion en CPU es %lf\n", cpu_time_taken);

    //GPU
    ValidateSudoku sudoku2(board);

    printf("\nGPU\n");
    clock_t gpu_start, gpu_stop;
    double gpu_time_taken;

    gpu_start = clock();
    if (sudoku2.solucionar_sudoku_parallel()) {
        std::cout << "El Sudoku ha sido resuelto:" << std::endl;
        sudoku2.imprimir_tablero();
    }
    else {
        std::cout << "No hay solución para el Sudoku proporcionado." << std::endl;
    }
    gpu_stop = clock();

    gpu_time_taken = ((double)(gpu_stop - gpu_start)) / CLOCKS_PER_SEC;
    printf("El tiempo de ejecucion en GPU es %lf\n", gpu_time_taken);

    // Montecarlo
    ValidateSudoku sudoku3(board);

    printf("\nMonte Carlo Optimization\n");
    clock_t mc_start, mc_stop;
    double mc_time_taken;

    double p = 0.5;
    mc_start = clock();

    if (sudoku3.solucionar_sudoku_parallel_montecarlo(p)) {
        std::cout << "El Sudoku ha sido resuelto:" << std::endl;
        sudoku3.imprimir_tablero();
    }
    else {
        std::cout << "No hay solución para el Sudoku proporcionado." << std::endl;
    }

    mc_stop = clock();
    mc_time_taken = ((double)(mc_stop - mc_start)) / CLOCKS_PER_SEC;
    printf("El tiempo de ejecucion en Monte Carlo con p = %lf es %lf\n", p, mc_time_taken);

    return 0;
}
#ifndef HEADER_MATRIX
#define HEADER_MATRIX

#include <stdio.h>

//typedef double cudaPrecision;
typedef float cudaPrecision;

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.col + col)
struct Matrix {
  size_t col;
  size_t row;
  cudaPrecision* elements;

  Matrix() {
      elements = NULL;
      col = 0;
      row = 0;
  }
};

#define READ_MATRIX(filename, matrix) read_matrix(filename, matrix, #matrix)
int read_matrix(char *fileName, Matrix &matrix, const char *varName);

#define WRITE_MATRIX(filename, matrix) write_matrix(filename, matrix, #matrix)
int write_matrix(char *fileName, const Matrix &matrix, const char *varName);

Matrix init_matrix_zero(int row, int col);
Matrix init_matrix_rand(int row, int col);
Matrix init_matrix_seq(int row, int col);
void free_matrix(Matrix &matrix);
void print_matrix(const Matrix &matrix);

#endif

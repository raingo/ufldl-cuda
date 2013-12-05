/**
 * @author Yuncheng Li (raingomm[AT]gmail.com)
 * @version 2013/11/28
 */

#include <iostream>
#include <matio.h>
#include <time.h>
#include "matrix.hpp"

using namespace std;

/**
 *  read a matrix from a file into a given matrix
 **/
int read_matrix(char *fileName, Matrix &matrix, const char *varName)
{
    mat_t    *matfp;
    matvar_t *matvar;
    int res = 0;

    matfp = Mat_Open(fileName,MAT_ACC_RDONLY);
    if ( NULL == matfp ) {
        fprintf(stderr,"Error opening MAT file %s\n", fileName);
        res = -1;
        goto real_exit;
    }

    matvar = Mat_VarRead(matfp,varName);
    if ( NULL == matvar )
    {
        fprintf(stderr,"Error reading var %s\n", varName);
        res = -1;
        goto close_matfp;
    }

    if (matvar -> data_type != MAT_T_SINGLE || matvar -> class_type != MAT_C_SINGLE || matvar -> data_size != 4) // if cudaPrecision == float
    //if (matvar -> data_type != MAT_T_DOUBLE || matvar -> class_type != MAT_C_DOUBLE || matvar -> data_size != 8) // if cudaPrecision == double
    {
        fprintf(stderr,"only double data is supported %s\n", varName);
        //fprintf(stderr, "(%d, %d)\n", matvar -> data_type, matvar -> class_type);
        res = -1;
        goto free_matvar;
    }

    if (matvar -> rank != 2)
    {
        fprintf(stderr,"only 2d matrix is supported %s\n", varName);
        res = -1;
        goto free_matvar;
    }

    if (matvar -> isComplex)
    {
        fprintf(stderr,"complex matrix is not supported %s\n", varName);
        res = -1;
        goto free_matvar;
    }

    matrix.row = matvar -> dims[0];
    matrix.col = matvar -> dims[1];

    matrix.elements = new cudaPrecision[matrix.row * matrix.col];
    memcpy(matrix.elements, matvar -> data, matvar -> nbytes);
    res = 0;

free_matvar:
    Mat_VarFree(matvar);
close_matfp:
    Mat_Close(matfp);
real_exit:
    return res;
}

/**
 *  Write matrix to a given file with name:
 **/
int write_matrix(char *fileName, const Matrix &matrix, const char *varName)
{
    mat_t *matfp;
    matvar_t *matvar;
    size_t dims[2];
    int res = 0;

    matfp = Mat_CreateVer(fileName, NULL, MAT_FT_DEFAULT);
    if ( NULL == matfp ) {
        fprintf(stderr,"Error creating MAT file \"test.mat\"\n");
        res = -1;
        goto real_exit;
    }

    dims[0] = matrix.row;
    dims[1] = matrix.col;
    // matvar = Mat_VarCreate(varName, MAT_C_DOUBLE, MAT_T_DOUBLE, 2,dims, matrix.elements, 0);
    matvar = Mat_VarCreate(varName, MAT_C_SINGLE, MAT_T_SINGLE, 2,dims, matrix.elements, 0);
    if ( NULL == matvar ) {
        fprintf(stderr,"Error creating variable for %s\n", varName);
        res = -1;
        goto close_matfp;
    } else {
        Mat_VarWrite(matfp, matvar, MAT_COMPRESSION_NONE);
        Mat_VarFree(matvar);
    }

    res = 0;
close_matfp:
    Mat_Close(matfp);
real_exit:
    return res;
}

/**
 *  Generate a random float number rangine from 0 to 9
 *  for each matrix element
 **/
Matrix init_matrix_rand(int row, int col)
{
    Matrix A;
    int i, j;

    A.row = row;
    A.col = col;
    A.elements = new cudaPrecision[A.row * A.col];

    srand((unsigned)time(NULL));
    for(i=0; i<A.row; i++){
        for(j=0; j<A.col; j++){
            A.elements[i * A.col + j] = (cudaPrecision)rand()/((cudaPrecision)RAND_MAX/10);
        }
    }
    return A;
}

Matrix init_matrix_zero(int row, int col)
{
    Matrix A;

    A.row = row;
    A.col = col;

    A.elements = new cudaPrecision[A.row * A.col];
    memset(A.elements, 0, A.row * A.col * sizeof(cudaPrecision));
    return A;
}

Matrix init_matrix_seq(int row, int col)
{
    Matrix A;
    int i, j;
    int idx = 0;

    A.row = row;
    A.col = col;

    A.elements = new cudaPrecision[A.row * A.col];
    for(i=0; i<A.row; i++){
        for(j=0; j<A.col; j++){
            A.elements[i * A.col + j] = (cudaPrecision)(idx++);
        }
    }
    return A;
}

void free_matrix(Matrix &matrix)
{
    delete [] matrix.elements;
    matrix.elements = NULL;
    matrix.col = 0;
    matrix.row = 0;
}

void print_matrix(const Matrix &matrix)
{
    int i, j;
    for (i = 0; i < matrix.row; i++) {
        for (j = 0; j < matrix.col; j++) {
			printf("%.2f ", matrix.elements[i * matrix.col + j]);
        }
        printf("\n");
    }
}

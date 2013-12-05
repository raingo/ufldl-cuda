#ifndef COMMON_MDE4DSMS
#define COMMON_MDE4DSMS

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>

#include <matrix.hpp>

//Round a / b to nearest higher integer value
inline int iDivUp(int a, int b)
{
    return (a + b - 1) / b;
    //return (a % b != 0) ? (a / b + 1) : (a / b);
}

//Align a to nearest higher multiple of b
inline int iAlignUp(int a, int b)
{
    return (a % b != 0) ? (a - a % b + b) : a;
}



#define BLOCK_SIZE 16
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

#define RUN_KERNEL_4(kernel, gridDim, blockDim, param1, param2, param3, param4) \
    do { \
        /*fprintf(stderr, "launching kernel: %s\n", #kernel);*/ \
        kernel<<<gridDim, blockDim>>>(param1, param2, param3, param4); \
        cudaStatus = cudaGetLastError(); \
        if (cudaStatus != cudaSuccess) { \
            fprintf(stderr, "%s launch failed: %s\n", #kernel,  cudaGetErrorString(cudaStatus)); \
            res = -1; \
            goto error; \
        } \
        cudaStatus = cudaDeviceSynchronize(); \
        if (cudaStatus != cudaSuccess) { \
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus); \
            res = -1; \
            goto error; \
        } \
    } while(0)

#define RUN_KERNEL_3(kernel, gridDim, blockDim, param1, param2, param3) \
    do { \
        /*fprintf(stderr, "launching kernel: %s\n", #kernel);*/ \
        kernel<<<gridDim, blockDim>>>(param1, param2, param3); \
        cudaStatus = cudaGetLastError(); \
        if (cudaStatus != cudaSuccess) { \
            fprintf(stderr, "%s launch failed: %s\n", #kernel,  cudaGetErrorString(cudaStatus)); \
            res = -1; \
            goto error; \
        } \
        cudaStatus = cudaDeviceSynchronize(); \
        if (cudaStatus != cudaSuccess) { \
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus); \
            res = -1; \
            goto error; \
        } \
    } while(0)

#define RUN_KERNEL(kernel, gridDim, blockDim, param1, param2) \
    do { \
        /*fprintf(stderr, "launching kernel: %s\n", #kernel);*/ \
        kernel<<<gridDim, blockDim>>>(param1, param2); \
        cudaStatus = cudaGetLastError(); \
        if (cudaStatus != cudaSuccess) { \
            fprintf(stderr, "%s launch failed: %s\n", #kernel,  cudaGetErrorString(cudaStatus)); \
            res = -1; \
            goto error; \
        } \
        cudaStatus = cudaDeviceSynchronize(); \
        if (cudaStatus != cudaSuccess) { \
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus); \
            res = -1; \
            goto error; \
        } \
    } while(0)

#define IO_MATRIX_WRAPPER(filename, base_dir, test_name, suffix, io_func, matrix) \
do { \
    fprintf(stderr, "%s... %s\n", #io_func, #matrix); \
    strcpy(filename, base_dir); \
    strcat(filename, test_name); \
    strcat(filename, "\\"); \
    strcat(filename, #matrix); \
    strcat(filename, suffix); \
    strcat(filename, ".mat"); \
    if(io_func(filename, matrix, #matrix) == -1) \
        goto real_exit; \
} while(0)

#define CUDA_PUT_MATRIX(dst, src) \
do { \
    /*fprintf(stderr, "memcpy %s\n", #dst);*/ \
    fflush(stderr); \
    status = cublasSetVector(dst.row * dst.col, sizeof(cudaPrecision), src.elements, 1, dst.elements, 1); \
    if (status != CUBLAS_STATUS_SUCCESS) \
    { \
        fprintf(stderr, " cublasSetVector %s filed!\n", #dst); \
        goto Error; \
    } \
} while(0)

#define CUDA_FETCH_MATRIX(dst, src) \
do { \
    /*fprintf(stderr, "memcpy %s\n", #dst);*/ \
    fflush(stderr); \
    status = cublasGetVector(dst.row * dst.col, sizeof(cudaPrecision), src.elements, 1, dst.elements, 1); \
    if (status != CUBLAS_STATUS_SUCCESS) \
    { \
        fprintf(stderr, "!!!! device access error (read %s)\n", #dst); \
        goto Error; \
    } \
} while(0)

#define CUDA_CLONE_MATRIX(dst, src) \
do { \
    fprintf(stderr, "Allocating %s\n", #dst); \
    fflush(stderr); \
    dst = src; \
    cudaStatus = cudaMalloc((void **)&dst.elements, dst.row * dst.col * sizeof(cudaPrecision)); \
    if (cudaStatus != cudaSuccess) { \
        fprintf(stderr, "cudaMalloc %s failed!\n", #dst); \
        dst.elements = NULL; \
        goto Error; \
    } \
    status = cublasSetVector(dst.row * dst.col, sizeof(cudaPrecision), src.elements, 1, dst.elements, 1); \
    if (status != CUBLAS_STATUS_SUCCESS) \
    { \
        fprintf(stderr, " cublasSetVector %s filed!\n", #dst); \
        goto Error; \
    } \
}while(0)

#define CUDA_ONES(dst, n_row, n_col) \
do { \
    cudaStatus = cudaMalloc((void **)&dst.elements, n_row * n_col * sizeof(cudaPrecision)); \
    dst.row = n_row; \
    dst.col = n_col; \
    if (cudaStatus != cudaSuccess) { \
        fprintf(stderr, "cudaMalloc %s failed!\n", #dst); \
        dst.elements = NULL; \
        goto Error; \
    } \
    cudaPrecision *tmp = new cudaPrecision[n_row * n_col]; \
    for (int i = 0; i < n_row * n_col; ++i) \
        tmp[i] = 1.0; \
    status = cublasSetVector(n_row * n_col, sizeof(cudaPrecision), tmp, 1, dst.elements, 1); \
    if (status != CUBLAS_STATUS_SUCCESS) \
    { \
        fprintf(stderr, " cublasSetVector %s failed!\n", #dst); \
        goto Error; \
    } \
    delete [] tmp; \
} while(0)

#define CUDA_ZEROS(dst, n_row, n_col) \
do { \
    dst.row = n_row; \
    dst.col = n_col; \
    cudaStatus = cudaMalloc((void **)&dst.elements, n_row * n_col * sizeof(cudaPrecision)); \
    if (cudaStatus != cudaSuccess) { \
        fprintf(stderr, "cudaMalloc %s failed!\n", #dst); \
        dst.elements = NULL; \
        goto Error; \
    } \
    cudaStatus = cudaMemset(dst.elements, 0, n_row * n_col * sizeof(cudaPrecision)); \
    if (cudaStatus != cudaSuccess) { \
        fprintf(stderr, "cudaMemcpy %s failed!\n", #dst); \
        goto Error; \
    } \
} while(0)


#define CUDA_ZERO_MATRIX(dst, src) \
do { \
    fprintf(stderr, "Allocating %s\n", #dst); \
    fflush(stderr); \
    CUDA_ZEROS(dst, src.row, src.col); \
}while(0)

// Global parameters and shared variables
typedef struct _gHandler_t {
    cublasHandle_t * cublas;
    cudaPrecision betaParam;
    cudaPrecision sparsityParam;
    cudaPrecision lambda;

    Matrix nSamplesOnes; // for summantion
    Matrix dInputOnes; // for summantion
    Matrix dHiddenOnes; // for summantion
} gHandler_t;

inline gHandler_t * createGlobalHandle(int nSamples, int dInput, int dHidden)
{
    cudaError_t cudaStatus;
    cublasStatus_t status;
    gHandler_t *handle = new gHandler_t;
    handle -> cublas = new cublasHandle_t;
    cublasCreate(handle -> cublas);
    handle -> betaParam = 5;
    handle -> sparsityParam = .035;
    handle -> lambda = 3e-3;

    CUDA_ONES(handle -> nSamplesOnes, nSamples, 1);
    CUDA_ONES(handle -> dHiddenOnes, dHidden, 1);
    CUDA_ONES(handle -> dInputOnes, dInput, 1);
Error:
    return handle;
}

inline void destroyGlobalHandle(gHandler_t ** _handle)
{
    gHandler_t *handle = *_handle;

    if (handle)
    {
        cublasDestroy(*(handle -> cublas));
        cudaFree((handle -> dHiddenOnes).elements);
        cudaFree((handle -> dInputOnes).elements);
        cudaFree((handle -> nSamplesOnes).elements);
        delete handle -> cublas;
        delete handle;
    }

    *_handle = NULL;
}

// COMMON ROUTINES
__global__ void biasAndLogistic(Matrix Z, Matrix b);

int gpu_blas_dot(const Matrix &src1, const Matrix &src2,cublasHandle_t *handle, cudaPrecision *result);
int gpu_blas_nrm2(const Matrix &src, cublasHandle_t *handle, cudaPrecision *result);
int gpu_blas_axpy(Matrix &dst, const Matrix &src, cublasHandle_t *handle, cudaPrecision alpha = 1);
int gpu_blas_mcopy(Matrix &dst, const Matrix &src, cublasHandle_t *handle);
int gpu_blas_mmul(const Matrix &C, const Matrix &A, const Matrix &B,
        cublasOperation_t transa, cublasOperation_t transb,
        cublasHandle_t *handle, cudaPrecision alf = 1, cudaPrecision bet = 0);

__global__ void bias(Matrix Z, Matrix b);
__global__ void dSparsity(Matrix sparsity_der, Matrix rho, cudaPrecision beta, cudaPrecision sparsity);
__global__ void costSparsity(Matrix rho, cudaPrecision sparsity);
__global__ void annScaling(Matrix delta2, Matrix sparsity_der, Matrix a2);

// ANN ROUTINES
int gpu_twolayer_ff(const Matrix &d_W1,
        const Matrix &d_b1,
        const Matrix &d_W2,
        const Matrix &d_b2,
        const Matrix &d_input,
        Matrix &d_a2,
        Matrix &d_a3,
        gHandler_t *handle);

int gpu_twolayer_bp(const Matrix &d_input, // INPUT
        const Matrix &d_rho, // INPUT
        const Matrix &d_W1, // INPUT
        const Matrix &d_W2, // INPUT
        const Matrix &d_a2, // INPUT
        const Matrix &d_a3, // INPUT
        Matrix &d_sparsity_der, // AUX
        Matrix &d_delta2, // AUX
        Matrix &d_delta3, // INPUT
        Matrix &d_pGradW1, // OUTPUT
        Matrix &d_pGradW2, // OUTPUT
        Matrix &d_pGrad1, // OUTPUT
        Matrix &d_pGradb2, // OUTPUT
        gHandler_t *handle); // INPUT

int gpu_twolayer_cost(const Matrix &d_data, //INPUT
        const Matrix &d_a2, // INPUT
        const Matrix &d_a3, // INPUT
        const Matrix &d_W1, // INPUT
        const Matrix &d_W2, // INPUT
        Matrix &d_KL, // AUX
        Matrix &d_rho, // OUTPUT
        Matrix &d_delta, // OUTPUT
        Matrix &cost, // OUTPUT (1 * 1)
        gHandler_t *handle); // INPUT

#endif /* end of include guard: COMMON_MDE4DSMS */

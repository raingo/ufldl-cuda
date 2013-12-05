#include "local_config.h"
#include "common.cuh"
#include <string.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t twoLayerBP(Matrix &pGradW1, Matrix &pGradW2, Matrix &pGradb1, Matrix &pGradb2, const Matrix &input, const Matrix &delta3, const Matrix &W1, const Matrix &W2, const Matrix &a2, const Matrix &rho);

int main_2bp()
{
    const char *base_dir = BASE_DIR;
    const char *test_name = "test_2bp";
    char filename[256];
    char *input_suffiex = "";
    char *res_suffiex = "res";

	Matrix pGradW1, pGradW2, pGradb1, pGradb2, input, delta3, W1, W2, a2, rho;

    int res = 0;

//'pGradW1', 'pGradW2', 'pGradb1', 'pGradb2', 'input', ...
//    'delta3', 'W1', 'W2', 'a2'

    cudaError_t cudaStatus;

    IO_MATRIX_WRAPPER(filename, base_dir, test_name, input_suffiex, read_matrix, pGradW1); // just for a2's size
    IO_MATRIX_WRAPPER(filename, base_dir, test_name, input_suffiex, read_matrix, pGradW2); // just for a3's size
    IO_MATRIX_WRAPPER(filename, base_dir, test_name, input_suffiex, read_matrix, pGradb1);
    IO_MATRIX_WRAPPER(filename, base_dir, test_name, input_suffiex, read_matrix, pGradb2);
    IO_MATRIX_WRAPPER(filename, base_dir, test_name, input_suffiex, read_matrix, input);
    IO_MATRIX_WRAPPER(filename, base_dir, test_name, input_suffiex, read_matrix, delta3);
    IO_MATRIX_WRAPPER(filename, base_dir, test_name, input_suffiex, read_matrix, W1);
    IO_MATRIX_WRAPPER(filename, base_dir, test_name, input_suffiex, read_matrix, W2);
    IO_MATRIX_WRAPPER(filename, base_dir, test_name, input_suffiex, read_matrix, a2);
    IO_MATRIX_WRAPPER(filename, base_dir, test_name, input_suffiex, read_matrix, rho);

    // Add vectors in parallel.
    cudaStatus = twoLayerBP(pGradW1, pGradW2, pGradb1, pGradb2, input, delta3, W1, W2, a2, rho);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "twoLayerBP failed!\n");
        res = -1;
        goto real_exit;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!\n");
        res = -1;
        goto real_exit;
    }

    IO_MATRIX_WRAPPER(filename, base_dir, test_name, res_suffiex, write_matrix, pGradW1);
    IO_MATRIX_WRAPPER(filename, base_dir, test_name, res_suffiex, write_matrix, pGradW2);
    IO_MATRIX_WRAPPER(filename, base_dir, test_name, res_suffiex, write_matrix, pGradb1);
    IO_MATRIX_WRAPPER(filename, base_dir, test_name, res_suffiex, write_matrix, pGradb2);

real_exit:
	free_matrix(pGradW1);
    free_matrix(pGradW2);
    free_matrix(pGradb1);
    free_matrix(pGradb2);
    free_matrix(input);
    free_matrix(delta3);
    free_matrix(W1);
    free_matrix(W2);
    free_matrix(a2);
    return res;
}

/*
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
        Matrix &d_pGradb1, // OUTPUT
        Matrix &d_pGradb2, // OUTPUT
        gHandler_t *handle) // INPUT
*/

// Helper function for using CUDA to add vectors in parallel.
cudaError_t twoLayerBP(Matrix &pGradW1, Matrix &pGradW2, Matrix &pGradb1, Matrix &pGradb2, const Matrix &input, const Matrix &delta3, const Matrix &W1, const Matrix &W2, const Matrix &a2, const Matrix &rho)
{

    cudaError_t cudaStatus;
    gHandler_t * handle = NULL;
    cublasStatus_t status;

    int dInput = input.row;
    int nSamples = input.col;
    int dHidden = a2.row;
    int dOutput = dInput;

    int i, niter = 500;
	clock_t startTime, stopTime, elapsedTime;

    Matrix d_input, d_rho, d_W1, d_W2, d_a2, d_a3, d_sparsity_der, d_delta2, d_delta3, d_pGradW1, d_pGradW2, d_pGradb1, d_pGradb2;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
        goto Error;
    }

    // Allocate GPU buffers
    CUDA_ZERO_MATRIX(d_pGradW1, pGradW1);
    CUDA_ZERO_MATRIX(d_pGradW2, pGradW2);
    CUDA_ZERO_MATRIX(d_pGradb1, pGradb1);
    CUDA_ZERO_MATRIX(d_pGradb2, pGradb2);

    CUDA_CLONE_MATRIX(d_input, input);
    CUDA_CLONE_MATRIX(d_delta3, delta3);
    CUDA_CLONE_MATRIX(d_W1, W1);
    CUDA_CLONE_MATRIX(d_W2, W2);
    CUDA_CLONE_MATRIX(d_a2, a2);
    CUDA_CLONE_MATRIX(d_rho, rho);

    CUDA_ZEROS(d_a3, dOutput, nSamples);
    CUDA_ZEROS(d_sparsity_der, a2.row, 1);
    CUDA_ZEROS(d_delta2, dHidden, nSamples); // % dHidden * nSamples

    handle = createGlobalHandle(nSamples, dInput, dHidden);

    fprintf(stderr, "gpu_twolayer_bp\n");
    fflush(stderr);

    startTime = clock();

    for (i = 0; i < niter; ++i)
        if (gpu_twolayer_bp(d_input, d_rho, d_W1, d_W2, d_a2, d_a3, d_sparsity_der, d_delta2, d_delta3, d_pGradW1, d_pGradW2, d_pGradb1, d_pGradb2, handle) == -1)
        {
            cudaStatus = cudaErrorLaunchFailure;
            fprintf(stderr, "gpu_twolayer_bp error\n");
        }
        else
        {
            CUDA_FETCH_MATRIX(pGradW1, d_pGradW1);
            CUDA_FETCH_MATRIX(pGradW2, d_pGradW2);
            CUDA_FETCH_MATRIX(pGradb1, d_pGradb1);
            CUDA_FETCH_MATRIX(pGradb2, d_pGradb2);
        }

    stopTime = clock();
    elapsedTime = stopTime - startTime;
    printf("OWLQN Optimization takes: %5.2f s \n", ((float)elapsedTime/CLOCKS_PER_SEC));
    printf("Number of Evaluation: %d\n", niter);
Error:
    destroyGlobalHandle(&handle);
    cudaFree(d_rho.elements);
    cudaFree(d_sparsity_der.elements);
    cudaFree(d_delta2.elements);
    cudaFree(d_delta3.elements);
    cudaFree(d_a2.elements);
    cudaFree(d_a3.elements);
    cudaFree(d_input.elements);
    cudaFree(d_W1.elements);
    cudaFree(d_W2.elements);
    cudaFree(d_pGradW1.elements);
    cudaFree(d_pGradW2.elements);
    cudaFree(d_pGradb1.elements);
    cudaFree(d_pGradb2.elements);

    return cudaStatus;
}

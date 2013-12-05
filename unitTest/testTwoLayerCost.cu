#include "common.cuh"
#include <string.h>
#include "local_config.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t twoLayerCost(Matrix &cost, Matrix &delta, Matrix &rho, const Matrix &data, const Matrix &a2, const Matrix &a3, const Matrix &W1, const Matrix &W2);

int main_2cost()
{
    const char *base_dir = BASE_DIR;
    const char *test_name = "test_2cost";
    char filename[256];
    char *input_suffiex = "";
    char *res_suffiex = "res";

    int res = 0;


//'cost', 'delta', 'rho', 'data', 'a2', ...
//    'a3', 'W1', 'W2'

	Matrix cost, delta, rho, data, a2, a3, W1, W2;

    cudaError_t cudaStatus;

    IO_MATRIX_WRAPPER(filename, base_dir, test_name, input_suffiex, read_matrix, W1);
    IO_MATRIX_WRAPPER(filename, base_dir, test_name, input_suffiex, read_matrix, W2);
    IO_MATRIX_WRAPPER(filename, base_dir, test_name, input_suffiex, read_matrix, a2);
    IO_MATRIX_WRAPPER(filename, base_dir, test_name, input_suffiex, read_matrix, a3);
    IO_MATRIX_WRAPPER(filename, base_dir, test_name, input_suffiex, read_matrix, data);
    IO_MATRIX_WRAPPER(filename, base_dir, test_name, input_suffiex, read_matrix, rho);
    IO_MATRIX_WRAPPER(filename, base_dir, test_name, input_suffiex, read_matrix, cost);
    IO_MATRIX_WRAPPER(filename, base_dir, test_name, input_suffiex, read_matrix, delta);

    // Add vectors in parallel.
    cudaStatus = twoLayerCost(cost, delta, rho, data, a2, a3, W1, W2);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "towLayerCost failed!\n");
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

    IO_MATRIX_WRAPPER(filename, base_dir, test_name, res_suffiex, write_matrix, cost);
    IO_MATRIX_WRAPPER(filename, base_dir, test_name, res_suffiex, write_matrix, delta);
    IO_MATRIX_WRAPPER(filename, base_dir, test_name, res_suffiex, write_matrix, rho);

real_exit:
    free_matrix(cost);
    free_matrix(delta);
    free_matrix(rho);
    free_matrix(data);
    free_matrix(W1);
    free_matrix(W2);
    free_matrix(a2);
    free_matrix(a3);
    return res;
}

/*

int gpu_twolayer_cost(const Matrix &d_data, //INPUT
        const Matrix &d_a2, // INPUT
        const Matrix &d_a3, // INPUT
        const Matrix &d_W1, // INPUT
        const Matrix &d_W2, // INPUT
        Matrix &d_KL, // AUX
        Matrix &d_rho, // OUTPUT
        Matrix &d_delta, // OUTPUT
        Matrix &cost, // OUTPUT (1 * 1)
        gHandler_t *handle) // INPUT
 */

// Helper function for using CUDA to add vectors in parallel.
cudaError_t twoLayerCost(Matrix &cost, Matrix &delta, Matrix &rho, const Matrix &data, const Matrix &a2, const Matrix &a3, const Matrix &W1, const Matrix &W2)
{

    cudaError_t cudaStatus;
    gHandler_t * handle = NULL;
    cublasStatus_t status;

    int i, niter = 500;
	clock_t startTime, stopTime, elapsedTime;

    int dInput = data.row;
    int nSamples = data.col;
    int dHidden = a2.row;
    int dOutput = dInput;

    Matrix d_data, d_a2, d_a3, d_W1, d_W2, d_KL, d_rho, d_delta;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
        goto Error;
    }

    // Allocate GPU buffers
    CUDA_ZERO_MATRIX(d_delta, delta);
    CUDA_ZERO_MATRIX(d_rho, rho);
    CUDA_ZERO_MATRIX(d_KL, rho);

    CUDA_CLONE_MATRIX(d_data, data);
    CUDA_CLONE_MATRIX(d_W1, W1);
    CUDA_CLONE_MATRIX(d_W2, W2);
    CUDA_CLONE_MATRIX(d_a2, a2);
    CUDA_CLONE_MATRIX(d_a3, a3);

    handle = createGlobalHandle(nSamples, dInput, dHidden);

    fprintf(stderr, "gpu_twolayer_cost\n");
    fflush(stderr);

    startTime = clock();

    for (i = 0; i < niter; ++i)
        if (gpu_twolayer_cost(d_data, d_a2, d_a3, d_W1, d_W2, d_KL, d_rho, d_delta, cost, handle) == -1)
        {
            fprintf(stderr, "gpu_twolayer_cost error\n");
            cudaStatus = cudaErrorLaunchFailure;
        }
        else
        {
            CUDA_FETCH_MATRIX(rho, d_rho);
            CUDA_FETCH_MATRIX(delta, d_delta);
        }
    stopTime = clock();
    elapsedTime = stopTime - startTime;
    printf("OWLQN Optimization takes: %5.2f s \n", ((float)elapsedTime/CLOCKS_PER_SEC));
    printf("Number of Evaluation: %d\n", niter);
Error:
    destroyGlobalHandle(&handle);
    cudaFree(d_rho.elements);
    cudaFree(d_KL.elements);
    cudaFree(d_delta.elements);
    cudaFree(d_a2.elements);
    cudaFree(d_a3.elements);
    cudaFree(d_data.elements);
    cudaFree(d_W1.elements);
    cudaFree(d_W2.elements);

    return cudaStatus;
}

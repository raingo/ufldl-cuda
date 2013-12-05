#include "common.cuh"
#include <string.h>
#include "local_config.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t twoLayerFF(Matrix &a2, Matrix &a3, const Matrix input, const Matrix W1, const Matrix W2, const Matrix b1, const Matrix b2);

int main_2ff()
{
    const char *base_dir = BASE_DIR;
    const char *test_name = "test_2ff";
    char filename[256];
    char *input_suffiex = "";
    char *res_suffiex = "res";
    //Matrix A = init_matrix_seq(10, 5);
	Matrix a2, a3, input, W1, W2, b1, b2;
    cudaError_t cudaStatus;

    int res = 0;

    IO_MATRIX_WRAPPER(filename, base_dir, test_name, input_suffiex, read_matrix, a2); // just for a2's size
    IO_MATRIX_WRAPPER(filename, base_dir, test_name, input_suffiex, read_matrix, a3); // just for a3's size
    IO_MATRIX_WRAPPER(filename, base_dir, test_name, input_suffiex, read_matrix, input);
    IO_MATRIX_WRAPPER(filename, base_dir, test_name, input_suffiex, read_matrix, W1);
    IO_MATRIX_WRAPPER(filename, base_dir, test_name, input_suffiex, read_matrix, W2);
    IO_MATRIX_WRAPPER(filename, base_dir, test_name, input_suffiex, read_matrix, b1);
    IO_MATRIX_WRAPPER(filename, base_dir, test_name, input_suffiex, read_matrix, b2);

    // Add vectors in parallel.
    cudaStatus = twoLayerFF(a2, a3, input, W1, W2, b1, b2);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "twoLayerFF failed!\n");
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

    IO_MATRIX_WRAPPER(filename, base_dir, test_name, res_suffiex, write_matrix, a2);
    IO_MATRIX_WRAPPER(filename, base_dir, test_name, res_suffiex, write_matrix, a3);

real_exit:
    free_matrix(a2);
    free_matrix(a3);
    free_matrix(input);
    free_matrix(W1);
    free_matrix(W2);
    free_matrix(b1);
    free_matrix(b2);
    return res;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t twoLayerFF(Matrix &a2, Matrix &a3, const Matrix input, const Matrix W1, const Matrix W2, const Matrix b1, const Matrix b2)
{
    int dInput = input.row;
    int nSamples = input.col;
    int dHidden = a2.row;
    int dOutput = dInput;
	int i, niter = 500;

	clock_t startTime, stopTime, elapsedTime;

	Matrix d_a2, d_a3, d_input, d_W1, d_W2, d_b1, d_b2;
    cudaError_t cudaStatus;
    gHandler_t * handle = NULL;
    cublasStatus_t status;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    CUDA_ZERO_MATRIX(d_a2, a2);
    CUDA_ZERO_MATRIX(d_a3, a3);
    CUDA_CLONE_MATRIX(d_input, input);
    CUDA_CLONE_MATRIX(d_W1, W1);
    CUDA_CLONE_MATRIX(d_W2, W2);
    CUDA_CLONE_MATRIX(d_b1, b1);
    CUDA_CLONE_MATRIX(d_b2, b2);

    handle = createGlobalHandle(nSamples, dInput, dHidden);

    fprintf(stderr, "gpu_twolayer_ff\n");
    fflush(stderr);

	startTime = clock();

	for (i = 0; i < niter; ++i)
		if (gpu_twolayer_ff(d_W1, d_b1, d_W2, d_b2, d_input, d_a2, d_a3, handle) == -1)
		{
			cudaStatus = cudaErrorLaunchFailure;
			fprintf(stderr, "gpu_twolayer_ff error\n");
		}
		else
		{
			CUDA_FETCH_MATRIX(a2, d_a2);
			CUDA_FETCH_MATRIX(a3, d_a3);
		}

	stopTime = clock();
	elapsedTime = stopTime - startTime;
	printf("OWLQN Optimization takes: %5.2f s \n", ((float)elapsedTime/CLOCKS_PER_SEC));
	printf("Number of Evaluation: %d\n", niter);

Error:
    destroyGlobalHandle(&handle);
    cudaFree(d_a2.elements);
    cudaFree(d_a3.elements);
    cudaFree(d_input.elements);
    cudaFree(d_W1.elements);
    cudaFree(d_W2.elements);
    cudaFree(d_b1.elements);
    cudaFree(d_b2.elements);

    return cudaStatus;
}

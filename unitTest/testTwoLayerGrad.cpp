#include "common.cuh"
#include <string.h>
#include "local_config.h"

#include "sparseAutoencoderLinearCost.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

int main_grad()
{
    const char *base_dir = BASE_DIR;
    const char *test_name = "test_2grad";
    char filename[256];
    char *input_suffiex = "";
    char *res_suffiex = "res";

    int res = 0;

    cudaWrappers *cudaOps = NULL;
    sparseAutoencoderLinearCost *obj = NULL;

    int dInput, nSamples, dHidden, dOutput;

// 'cost', 'grad', 'theta', 'data'

	Matrix cost, grad, theta, data;
    DblVec v_grad, v_theta;

    IO_MATRIX_WRAPPER(filename, base_dir, test_name, input_suffiex, read_matrix, cost);
    IO_MATRIX_WRAPPER(filename, base_dir, test_name, input_suffiex, read_matrix, grad);
    IO_MATRIX_WRAPPER(filename, base_dir, test_name, input_suffiex, read_matrix, theta);
    IO_MATRIX_WRAPPER(filename, base_dir, test_name, input_suffiex, read_matrix, data);

    dInput = data.row;
    nSamples = data.col;
    dHidden = 400;
    dOutput = dInput;

    fprintf(stderr, "init cudaWrappers\n");
    cudaOps = new cudaWrappers(data, dHidden);
    obj = new sparseAutoencoderLinearCost(*cudaOps, dInput, dHidden);

    v_theta.insert(v_theta.end(), theta.elements, theta.elements + theta.row * theta.col);
    *cost.elements = obj -> Eval(v_theta, v_grad);
    std::copy(v_grad.begin(), v_grad.end(), grad.elements);

    IO_MATRIX_WRAPPER(filename, base_dir, test_name, res_suffiex, write_matrix, cost);
    IO_MATRIX_WRAPPER(filename, base_dir, test_name, res_suffiex, write_matrix, grad);

real_exit:
    free_matrix(cost);
    free_matrix(grad);
    free_matrix(data);
    free_matrix(theta);

    if (obj)
        delete obj;

    if (cudaOps)
        delete cudaOps;

    return res;
}

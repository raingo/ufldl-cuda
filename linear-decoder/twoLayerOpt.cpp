#include "common.cuh"
#include <string.h>
#include "local_config.h"

#include "sparseAutoencoderLinearCost.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

int main_opt()
{
	bool quiet = false;
	double tol = 1e-4, regweight = 0;
	int m = 10;

    const char *base_dir = BASE_DIR;
    const char *test_name = "test_2opt";
    char filename[256];
    char *input_suffiex = "";
    char *res_suffiex = "res";

    int res = 0;

    cudaWrappers *cudaOps = NULL;
    sparseAutoencoderLinearCost *obj = NULL;
	OWLQN opt(quiet);

    int dInput, nSamples, dHidden, dOutput;
	clock_t startTime, stopTime, elapsedTime;

	Matrix theta, data, ans;
    DblVec v_theta, v_ans;

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
    v_ans = DblVec(v_theta.size());

	startTime = clock();
	opt.Minimize(*obj, v_theta, v_ans, regweight, tol, m);
	stopTime = clock();
	elapsedTime = stopTime - startTime;
	printf("OWLQN Optimization takes: %5.2f s \n", ((float)elapsedTime/CLOCKS_PER_SEC));
	printf("Number of Evaluation: %d\n", obj -> get_n_eval());

    //*cost.elements = obj -> Eval(v_theta, v_grad);
    ans = init_matrix_zero(v_ans.size(), 1);
    std::copy(v_ans.begin(), v_ans.end(), ans.elements);

    IO_MATRIX_WRAPPER(filename, base_dir, test_name, res_suffiex, write_matrix, ans);

real_exit:
    free_matrix(data);
    free_matrix(theta);
    free_matrix(ans);

    if (obj)
        delete obj;

    if (cudaOps)
        delete cudaOps;

    return res;
}

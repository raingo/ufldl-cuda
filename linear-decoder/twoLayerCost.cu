#include "common.cuh"

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
{
    int res = 0;
    cudaError_t cudaStatus;
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid;

    int nSamples;
    cudaPrecision nrm_;

    //nSamples = size(data, 2);
    nSamples = d_data.col;

    //rho = sum(a2, 2) / nSamples; % dHidden * 1
    if ((res = gpu_blas_mmul(d_rho, d_a2, handle -> nSamplesOnes, CUBLAS_OP_N, CUBLAS_OP_N, handle -> cublas, 1.0 / nSamples)) == -1)
    {
        fprintf(stderr, "rho = sum(a2, 2) / nSamples ERROR \n");
        goto error;
    }

    // delta = a3 - data;
    if ((res = gpu_blas_mcopy(d_delta, d_a3, handle -> cublas)) == -1)
    {
        fprintf(stderr, "delta = a3 ERROR \n");
        goto error;
    }
    if ((res = gpu_blas_axpy(d_delta, d_data, handle -> cublas, -1.0)) == -1)
    {
        fprintf(stderr, "delta -= data ERROR \n");
        goto error;
    }


    // cost = norm(delta(:), 2) ^ 2;
    // cost = cost / nSamples;
    if ((res = gpu_blas_nrm2(d_delta, handle -> cublas, &nrm_)) == -1)
    {
        fprintf(stderr, "nrm = norm(delta(:), 2) ERROR \n");
        goto error;
    }
    *cost.elements = nrm_* nrm_;
    *cost.elements /= nSamples;

    //cost = cost + lambda * (norm(W1(:), 2) ^ 2 + norm(W2(:), 2) ^ 2);
    //cost = cost / 2;
    if ((res = gpu_blas_nrm2(d_W1, handle -> cublas, &nrm_)) == -1)
    {
        fprintf(stderr, "nrm = norm(W1(:), 2) ERROR \n");
        goto error;
    }
    *cost.elements += handle -> lambda * nrm_* nrm_;
    if ((res = gpu_blas_nrm2(d_W2, handle -> cublas, &nrm_)) == -1)
    {
        fprintf(stderr, "nrm = norm(W2(:), 2) ERROR \n");
        goto error;
    }
    *cost.elements += handle -> lambda * nrm_* nrm_;
    *cost.elements /= 2.0;

    // cost = cost + beta * sum(KL(rho, sparsityParam));
    if ((res = gpu_blas_mcopy(d_KL, d_rho, handle -> cublas)) == -1)
    {
        fprintf(stderr, "KL = rho ERROR \n");
        goto error;
    }

    dimGrid.x = iDivUp(d_KL.col, dimBlock.x);
    dimGrid.y = iDivUp(d_KL.row, dimBlock.y);
    RUN_KERNEL(costSparsity, dimGrid, dimBlock, d_KL, handle -> sparsityParam);
    if ((res = gpu_blas_dot(d_KL, handle -> dHiddenOnes, handle -> cublas, &nrm_)) == -1)
    {
        fprintf(stderr, "nrm = sum(KL) ERROR \n");
        goto error;
    }
    *cost.elements += handle -> betaParam * nrm_;

error:
    return res;
}

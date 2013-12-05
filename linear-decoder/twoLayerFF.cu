#include "common.cuh"

int gpu_twolayer_ff(const Matrix &d_W1,
        const Matrix &d_b1,
        const Matrix &d_W2,
        const Matrix &d_b2,
        const Matrix &d_input,
        Matrix &d_a2,
        Matrix &d_a3,
        gHandler_t *handle)
{
    cudaError_t cudaStatus;
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid;
    int res = 0;

    // a2 = W1 * input; % dHidden * nSamples
    if ((res = gpu_blas_mmul(d_a2, d_W1, d_input, CUBLAS_OP_N, CUBLAS_OP_N, handle -> cublas)) == -1)
        goto error;

    // a2 = bsxfun(@plus, a2, b1);
    // a2 = sigmoid(a2);
    dimGrid.x = iDivUp(d_a2.col, dimBlock.x);
    dimGrid.y = iDivUp(d_a2.row, dimBlock.y);
    RUN_KERNEL(biasAndLogistic, dimGrid, dimBlock, d_a2, d_b1);

    // a3 = W2 * a2;
    if ((res = gpu_blas_mmul(d_a3, d_W2, d_a2, CUBLAS_OP_N, CUBLAS_OP_N, handle -> cublas)) == -1)
        goto error;

    // a3 = bsxfun(@plus, a3, b2); % dOutput * nSamples
    dimGrid.x = iDivUp(d_a3.col, dimBlock.x);
    dimGrid.y = iDivUp(d_a3.row, dimBlock.y);
    RUN_KERNEL(bias, dimGrid, dimBlock, d_a3, d_b2);

error:
    return res;
}

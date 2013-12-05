#include "common.cuh"

//function [pGradW1, pGradW2, pGradb1, pGradb2] = twoLayerBP(input, delta3, W1, W2, a2, ...
//        rho, beta, sparsity, lambda)

// delta2, delta3 are auxilliary matrix
// delta2: size = a2
// delta3: size = a3
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
{
    int res = 0;
    cudaError_t cudaStatus;
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid;

    // nSamples = size(input, 2);
    int nSamples = d_input.col;

    // sparsity_der = beta * (- sparsity ./ rho + (1 - sparsity) ./ (1 - rho));
    dimGrid.x = iDivUp(d_sparsity_der.col, dimBlock.x);
    dimGrid.y = iDivUp(d_sparsity_der.row, dimBlock.y);
    RUN_KERNEL_4(dSparsity, dimGrid, dimBlock, d_sparsity_der, d_rho, handle -> betaParam, handle -> sparsityParam);

    // delta2 = W2' * delta3;
    if ((res = gpu_blas_mmul(d_delta2, d_W2, d_delta3, CUBLAS_OP_T, CUBLAS_OP_N, handle -> cublas)) == -1)
    {
        fprintf(stderr, "delta2 = W2' * delta3 ERROR\n");
        goto error;
    }

    // delta2 = (delta2 + sparsity_der) .* a2 .* (1 - a2); % dHidden * nSamples
    dimGrid.x = iDivUp(d_a2.col, dimBlock.x);
    dimGrid.y = iDivUp(d_a2.row, dimBlock.y);
    RUN_KERNEL_3(annScaling, dimGrid, dimBlock, d_delta2, d_sparsity_der, d_a2);

    // pGradW2 = delta3 * a2'; % dHidden * dInput
    // pGradW2 = pGradW2 / nSamples + lambda * W2;
    if ((res = gpu_blas_mcopy(d_pGradW2, d_W2, handle -> cublas)) == -1)
        goto error;
    if ((res = gpu_blas_mmul(d_pGradW2, d_delta3, d_a2, CUBLAS_OP_N, CUBLAS_OP_T, handle -> cublas, 1.0 / nSamples, handle -> lambda)) == -1)
    {
        fprintf(stderr, "pGradW2 = delta3 * a2' / nSamples + lambda * W2; ERROR \n");
        goto error;
    }

    // pGradW1 = delta2 * input'; % dOutput * dHidden
    // pGradW1 = pGradW1 / nSamples + lambda * W1;
    if ((res = gpu_blas_mcopy(d_pGradW1, d_W1, handle -> cublas)) == -1)
        goto error;
    if ((res = gpu_blas_mmul(d_pGradW1, d_delta2, d_input, CUBLAS_OP_N, CUBLAS_OP_T, handle -> cublas, 1.0 / nSamples, handle -> lambda)) == -1)
    {
        fprintf(stderr, "pGradW1 = pGradW1 / nSamples + lambda * W1;\n");
        goto error;
    }

    // pGradb2 = sum(delta3, 2); % dInput * 1
    // pGradb2 = pGradb2/ nSamples;
    if ((res = gpu_blas_mmul(d_pGradb2, d_delta3, handle -> nSamplesOnes, CUBLAS_OP_N, CUBLAS_OP_N, handle -> cublas, 1.0 / nSamples)) == -1)
    {
        fprintf(stderr, "pGradb2 = sum(delta3, 2);\n");
        goto error;
    }

    // pGradb1 = sum(delta2, 2); % dHidden * 1
    // pGradb1= pGradb1 / nSamples;
    if ((res = gpu_blas_mmul(d_pGradb1, d_delta2, handle -> nSamplesOnes, CUBLAS_OP_N, CUBLAS_OP_N, handle -> cublas, 1.0 / nSamples)) == -1)
    {
        fprintf(stderr, "pGradb1 = sum(delta2, 2);\n");
        goto error;
    }

error:
    return res;
}

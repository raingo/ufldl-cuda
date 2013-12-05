#include "sparseAutoencoderLinearCost.h"

using namespace std;

cudaWrappers::cudaWrappers(const Matrix &data, int dHidden):handle(NULL)
{
    int dInput = data.row;
    int nSamples = data.col;
    int dOutput = dInput;
    cudaError_t cudaStatus;
    cublasStatus_t status;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
        goto Error;
    }

    CUDA_CLONE_MATRIX(d_data, data);

    CUDA_ZEROS(d_a2, dHidden, nSamples);
    CUDA_ZEROS(d_a3, dOutput, nSamples);
    CUDA_ZEROS(d_W1, dHidden, dInput);
    CUDA_ZEROS(d_W2, dOutput, dHidden);
    CUDA_ZEROS(d_b1, dHidden, 1);
    CUDA_ZEROS(d_b2, dOutput, 1);
    CUDA_ZEROS(d_delta3, dOutput, nSamples);
    CUDA_ZEROS(d_delta2, dHidden, nSamples);

    CUDA_ZEROS(d_rho, dHidden, 1);
    CUDA_ZEROS(d_sparsity_der, dHidden, 1);
    CUDA_ZEROS(d_KL, dHidden, 1);

    CUDA_ZEROS(d_pGradW1, dHidden, dInput);
    CUDA_ZEROS(d_pGradW2, dOutput, dHidden);
    CUDA_ZEROS(d_pGradb1, dHidden, 1);
    CUDA_ZEROS(d_pGradb2, dOutput, 1);

    handle = createGlobalHandle(nSamples, dInput, dHidden);

    return;
Error:
    throw -1;
}


int cudaWrappers::gpu_eval_ann(Matrix &cost,
        Matrix &pGradW1, Matrix &pGradW2,
        Matrix &pGradb1, Matrix &pGradb2,
        const Matrix &W1, const Matrix &W2,
        const Matrix &b1, const Matrix &b2)
{
    int res = 0;
    cudaError_t cudaStatus;
    cublasStatus_t status;

    CUDA_PUT_MATRIX(d_W1, W1);
    CUDA_PUT_MATRIX(d_W2, W2);
    CUDA_PUT_MATRIX(d_b1, b1);
    CUDA_PUT_MATRIX(d_b2, b2);

    //fprintf(stderr, "gpu_twolayer_ff\n");
    //fflush(stderr);
    if (gpu_twolayer_ff(d_W1, d_b1, d_W2, d_b2, d_data, d_a2, d_a3, handle) == -1)
    {
        res = -1;
        goto Error;
    }

    //fprintf(stderr, "gpu_twolayer_cost\n");
    //fflush(stderr);
    if (gpu_twolayer_cost(d_data, d_a2, d_a3, d_W1, d_W2, d_KL, d_rho, d_delta3, cost, handle) == -1)
    {
        res = -1;
        goto Error;
    }

    //fprintf(stderr, "gpu_twolayer_bp\n");
    //fflush(stderr);
    if (gpu_twolayer_bp(d_data, d_rho, d_W1, d_W2, d_a2, d_a3, d_sparsity_der, d_delta2, d_delta3, d_pGradW1, d_pGradW2, d_pGradb1, d_pGradb2, handle) == -1)
    {
        res = -1;
        goto Error;
    }

    CUDA_FETCH_MATRIX(pGradW1, d_pGradW1);
    CUDA_FETCH_MATRIX(pGradW2, d_pGradW2);
    CUDA_FETCH_MATRIX(pGradb1, d_pGradb1);
    CUDA_FETCH_MATRIX(pGradb2, d_pGradb2);

Error:
    return res;
}

cudaWrappers::~cudaWrappers()
{
    release_matrices();
}

void cudaWrappers::release_matrices()
{
    cudaError_t cudaStatus;
    cublasStatus_t status;
    destroyGlobalHandle(&handle);

    cudaFree(d_data.elements);
    cudaFree(d_a2.elements);
    cudaFree(d_a3.elements);

    cudaFree(d_W1.elements);
    cudaFree(d_W2.elements);
    cudaFree(d_b1.elements);
    cudaFree(d_b2.elements);

    cudaFree(d_rho.elements);
    cudaFree(d_sparsity_der.elements);
    cudaFree(d_KL.elements);

    cudaFree(d_delta2.elements);
    cudaFree(d_delta3.elements);

    cudaFree(d_pGradW1.elements);
    cudaFree(d_pGradW2.elements);
    cudaFree(d_pGradb1.elements);
    cudaFree(d_pGradb2.elements);

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!\n");
        //ignore
    }
}

void sparseAutoencoderLinearCost::m2v(const std::vector<Matrix> &Ms, DblVec &V)
{
    std::vector<Matrix>::const_iterator itM;
    V.clear();
    for (itM = Ms.begin(); itM != Ms.end(); ++itM)
    {
        int len = (*itM).row * (*itM).col;
        cudaPrecision *begin = (*itM).elements;
        cudaPrecision *end= begin + len;
        V.insert(V.end(), begin, end);
    }
}

void sparseAutoencoderLinearCost::v2m(std::vector<Matrix> &Ms, const DblVec &V)
{
    std::vector<Matrix>::iterator itM;
    DblVec::const_iterator itV = V.begin();
    for (itM = Ms.begin(); itM != Ms.end(); ++itM)
    {
        int len = (*itM).row * (*itM).col;
        cudaPrecision *begin = (*itM).elements;
        std::copy(itV, itV + len, begin);
        itV = itV + len;
    }
}

double sparseAutoencoderLinearCost::Eval(const DblVec& input, DblVec& gradient)
{
    v2m(m_Ms, input);

    if (m_cudaWrappers.gpu_eval_ann(m_cost, m_W1, m_W2, m_b1, m_b2, m_W1, m_W2, m_b1, m_b2) == -1)
        throw -1;
    ++m_n_eval;

    m2v(m_Ms, gradient);

    return *m_cost.elements;
}

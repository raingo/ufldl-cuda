#include "common.cuh"
/**
 *   Cublas GPU matrix multiplication
 */

int gpu_blas_dot(const Matrix &src1, const Matrix &src2,cublasHandle_t *handle, cudaPrecision *result)
{
    cublasStatus_t cublasStatus;

    if (src1.row != src2.row || src1.col != src2.col)
    {
        fprintf(stderr, "gpu_blas_dot error: dimension not compatible\n");
        return -1;
    }

    // cublasStatus = cublasDdot(*handle, src1.row * src1.col, src1.elements, 1, src2.elements, 1, result);
    cublasStatus = cublasSdot(*handle, src1.row * src1.col, src1.elements, 1, src2.elements, 1, result);

    if (cublasStatus != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "cublasSaxpy error\n");
        return -1;
    }
    return 0;
}

int gpu_blas_nrm2(const Matrix &src, cublasHandle_t *handle, cudaPrecision *result)
{
    cublasStatus_t cublasStatus;

    // cublasStatus = cublasDnrm2(*handle, src.row * src.col, src.elements, 1, result);
    cublasStatus = cublasSnrm2(*handle, src.row * src.col, src.elements, 1, result);

    if (cublasStatus != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "cublasSaxpy error\n");
        return -1;
    }
    return 0;
}

// dst = alpha * src + dst
int gpu_blas_axpy(Matrix &dst, const Matrix &src, cublasHandle_t *handle, cudaPrecision alpha /*= 1*/)
{
    cublasStatus_t cublasStatus;

    if (dst.row != src.row || dst.col != src.col)
    {
        fprintf(stderr, "gpu_blas_minus error: dimension not compatible\n");
        return -1;
    }

    // cublasStatus = cublasDaxpy(*handle, dst.row * dst.col, &alpha, src.elements, 1, dst.elements, 1);
    cublasStatus = cublasSaxpy(*handle, dst.row * dst.col, &alpha, src.elements, 1, dst.elements, 1);

    if (cublasStatus != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "cublasSaxpy error\n");
        return -1;
    }
    return 0;
}

int gpu_blas_mcopy(Matrix &dst, const Matrix &src, cublasHandle_t *handle)
{
    cublasStatus_t cublasStatus;

    if (dst.row != src.row || dst.col != src.col)
    {
        fprintf(stderr, "gpu_blas_mcopy error: dimension not compatible\n");
        return -1;
    }

    // cublasStatus = cublasDcopy(*handle, dst.row * dst.col, src.elements, 1, dst.elements, 1);
    cublasStatus = cublasScopy(*handle, dst.row * dst.col, src.elements, 1, dst.elements, 1);

    if (cublasStatus != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "cublasScopy error\n");
        return -1;
    }
    return 0;
}

// Multiply the arrays A and B on GPU and save the result in C
// C(m,n) = A(m,k) * B(k,n)
int gpu_blas_mmul(const Matrix &C, const Matrix &A, const Matrix &B,
        cublasOperation_t transa, cublasOperation_t transb,
        cublasHandle_t *handle, cudaPrecision alf /* = 1*/, cudaPrecision bet /* = 0*/)
{
    int lda = A.row;
    int ldb = B.row;
    int ldc = C.row;

    int m, n, ka, kb;

    const cudaPrecision *alpha = &alf;
    const cudaPrecision *beta = &bet;
    cublasStatus_t cublasStatus;
    cudaError_t cudaStatus;

    if (transa == CUBLAS_OP_N)
    {
        m = A.row;
        ka = A.col;
    }
    else
    {
        m = A.col;
        ka = A.row;
    }

    if (transb == CUBLAS_OP_N)
    {
        n = B.col;
        kb = B.row;
    }
    else
    {
        n = B.row;
        kb = B.col;
    }

    if (ka != kb)
    {
        fprintf(stderr, "gpu_blas_mmul error: dimension not compatible: (%d, %d)\n", ka, kb);
        return -1;
    }

    //fprintf(stderr, "(%d, %d) = (%d, %d) * (%d %d)\n", C.row, C.col, A.row, A.col, B.row, B.col);

    // cublasStatus = cublasDgemm(*handle, transa, transb, m, n, ka, alpha, A.elements, lda, B.elements, ldb, beta, C.elements, ldc);
    cublasStatus = cublasSgemm(*handle, transa, transb, m, n, ka, alpha, A.elements, lda, B.elements, ldb, beta, C.elements, ldc);

    if (cublasStatus != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "cublasSgemm error\n");
        return -1;
    }

    /*
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        return -1;
    }*/

    return 0;
}

// Z(nHidden, nSample) = logistic(Z(nHidden, nSample) + B(nHidden))
// in place update
__global__ void biasAndLogistic(Matrix Z, Matrix b)
{
  cudaPrecision z;

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < Z.row && col < Z.col)
  {
      z = Z.elements[IDX2C(row, col, Z.row)]; // column major
      z += b.elements[row];
      //z = 1.0f / (1.0f + exp(-z)); // fast-math
      z = 1.0f / (1.0f + __expf(-z)); // fast-math

      Z.elements[IDX2C(row, col, Z.row)] = z;
  }
}

// Z(nHidden, nSample) = logistic(Z(nHidden, nSample) + B(nHidden))
// in place update
__global__ void bias(Matrix Z, Matrix b)
{
    cudaPrecision z;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < Z.row && col < Z.col)
    {
        z = Z.elements[IDX2C(row, col, Z.row)];
        z += b.elements[row];

        Z.elements[IDX2C(row, col, Z.row)] = z;
    }
}

// ann bp scaling delta2 = (delta2 + sparsity_der) .* a2 .* (1 - a2)
__global__ void annScaling(Matrix delta2, Matrix sparsity_der, Matrix a2)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < delta2.row && col < delta2.col)
    {
        int offset = IDX2C(row, col, delta2.row);

        cudaPrecision delta2_ = delta2.elements[offset];
        cudaPrecision sparsity_der_ = sparsity_der.elements[row];
        cudaPrecision a2_ = a2.elements[offset];

        delta2.elements[offset] = (delta2_ + sparsity_der_) * a2_ * (1 - a2_);
    }
}

// sparsity_der = beta * (- sparsity ./ rho + (1 - sparsity) ./ (1 - rho));
// sparsity derivative
__global__ void dSparsity(Matrix sparsity_der, Matrix rho, cudaPrecision beta, cudaPrecision sparsity)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rho.row && col < rho.col)
    {
        int offset = IDX2C(row, col, rho.row);

        cudaPrecision rho_ = rho.elements[offset];
        sparsity_der.elements[offset] = beta * (- sparsity / rho_ + (1 - sparsity) / (1 - rho_));
        // sparsity_der.elements[offset] = 0;
    }
}

// KL div, store elements into matrix
// div = sparsty .* log(sparsty ./ rho) + (1 - sparsty) .* log((1 - sparsty) ./ (1 - rho));
// save into rho
__global__ void costSparsity(Matrix rho, cudaPrecision sparsity)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rho.row && col < rho.col)
    {
        int offset = IDX2C(row, col, rho.row);

        cudaPrecision rho_ = rho.elements[offset];
        // cudaPrecision div = sparsity * log(sparsity / rho_) + (1 - sparsity) * log((1 - sparsity) / (1 - rho_)); // __logf: fast-math
        cudaPrecision div = sparsity * __logf(sparsity / rho_) + (1 - sparsity) * __logf((1 - sparsity) / (1 - rho_)); // __logf: fast-math

        rho.elements[offset] = div;
    }
}

#pragma once

#include <vector>

#include "OWLQN.h"
#include "matrix.hpp"
#include "common.cuh"

class cudaWrappers {

private:
    Matrix d_data; // Training data
	Matrix d_a2, d_a3, d_W1, d_W2, d_b1, d_b2; // Network
    Matrix d_rho, d_sparsity_der, d_delta2, d_delta3, d_KL;  // Aux
    Matrix d_pGradW1, d_pGradW2, d_pGradb1, d_pGradb2; // Output

    gHandler_t * handle;

    void release_matrices();

public:
    cudaWrappers(const Matrix &data, int dHidden);
    ~cudaWrappers();

    int gpu_eval_ann(Matrix &cost,
            Matrix &pGradW1, Matrix &pGradW2,
            Matrix &pGradb1, Matrix &pGradb2,
            const Matrix &W1, const Matrix &W2,
            const Matrix &b1, const Matrix &b2);
};

class sparseAutoencoderLinearCost: public DifferentiableFunction {

private:
	cudaWrappers &m_cudaWrappers;

    Matrix m_W1;
    Matrix m_W2;
    Matrix m_b1;
    Matrix m_b2;
    std::vector<Matrix> m_Ms;

    Matrix m_cost;

    int m_n_eval;

    // matrices to vector
    void m2v(const std::vector<Matrix> &Ms, DblVec &V);

    // vector to matrices
    void v2m(std::vector<Matrix> &Ms, const DblVec &V);

public:
	sparseAutoencoderLinearCost(cudaWrappers& wrapper, int dInput, int dHidden) : m_cudaWrappers(wrapper) {
        m_W1 =  init_matrix_zero(dHidden, dInput);
        m_W2 =  init_matrix_zero(dInput, dHidden);
        m_b1 =  init_matrix_zero(dHidden, 1);
        m_b2 =  init_matrix_zero(dInput, 1);
        m_cost =  init_matrix_zero(1, 1);

        m_Ms.push_back(m_W1);
        m_Ms.push_back(m_W2);
        m_Ms.push_back(m_b1);
        m_Ms.push_back(m_b2);

        m_n_eval = 0;
    }
    ~sparseAutoencoderLinearCost() {
        free_matrix(m_W1);
        free_matrix(m_W2);
        free_matrix(m_b1);
        free_matrix(m_b2);
        free_matrix(m_cost);
    }

    int get_n_eval() { return m_n_eval;}

	double Eval(const DblVec& input, DblVec& gradient);

};

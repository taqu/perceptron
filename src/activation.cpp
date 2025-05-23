#include "activation.h"

namespace mindnn
{
	void Sigmoid::activate(const Matrix& Z, Matrix& A)
        {
            A.array() = Scalar(1) / (Scalar(1) + (-Z.array()).exp());
        }

    void Sigmoid::apply_jacobian(const Matrix& Z, const Matrix& A, const Matrix& F, Matrix& G)
        {
            G.array() = A.array() * (Scalar(1) - A.array()) * F.array();
        }

    void ReLU::activate(const Matrix& Z, Matrix& A)
        {
            A.array() = Z.array().cwiseMax(Scalar(0));
        }

        void ReLU::apply_jacobian(const Matrix& Z, const Matrix& A, const Matrix& F, Matrix& G)
        {
            G.array() = (A.array() > Scalar(0)).select(F, Scalar(0));
        }

        void Softmax::activate(const Matrix& Z, Matrix& A)
        {
            A.array() = (Z.rowwise() - Z.colwise().maxCoeff()).array().exp();
            RowArray colsums = A.colwise().sum();
            A.array().rowwise() /= colsums;
        }

        void Softmax::apply_jacobian(const Matrix& Z, const Matrix& A, const Matrix& F, Matrix& G)
        {
            RowArray a_dot_f = A.cwiseProduct(F).colwise().sum();
            G.array() = A.array() * (F.array().rowwise() - a_dot_f);
        }

        void Mish::activate(const Matrix& Z, Matrix& A)
        {
            // h(x) = tanh(softplus(x)) = (1 + exp(x))^2 - 1
            //                            ------------------
            //                            (1 + exp(x))^2 + 1
            // Let s = exp(-abs(x)), t = 1 + s
            // If x >= 0, then h(x) = (t^2 - s^2) / (t^2 + s^2)
            // If x <= 0, then h(x) = (t^2 - 1) / (t^2 + 1)
            Matrix S = (-Z.array().abs()).exp();
            A.array() = (S.array() + Scalar(1)).square();  // t^2
            S.noalias() = (Z.array() >= Scalar(0)).select(S.cwiseAbs2(), Scalar(1));  // s^2 or 1
            A.array() = (A.array() - S.array()) / (A.array() + S.array());
            A.array() *= Z.array();
        }

        void Mish::apply_jacobian(const Matrix& Z, const Matrix& A, const Matrix& F, Matrix& G)
        {
            // Let h(x) = tanh(softplus(x))
            // Mish'(x) = h(x) + x * h'(x)
            // h'(x) = tanh'(softplus(x)) * softplus'(x)
            //       = [1 - h(x)^2] * exp(x) / (1 + exp(x))
            //       = [1 - h(x)^2] / (1 + exp(-x))
            // Mish'(x) = h(x) + [x - Mish(x) * h(x)] / (1 + exp(-x))
            // A = Mish(Z) = Z .* h(Z) => h(Z) = A ./ Z, h(0) = 0.6
            G.noalias() = (Z.array() == Scalar(0)).select(Scalar(0.6), A.cwiseQuotient(Z));
            G.array() += (Z.array() - A.array() * G.array()) / (Scalar(1) + (-Z).array().exp());
            G.array() *= F.array();
        }

        void Tanh::activate(const Matrix& Z, Matrix& A)
        {
            A.array() = Z.array().tanh();
        }

         void Tanh::apply_jacobian(const Matrix& Z, const Matrix& A, const Matrix& F, Matrix& G)
        {
            G.array() = (Scalar(1) - A.array().square()) * F.array();
        }

}

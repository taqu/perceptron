#ifndef INC_MINDNN_ACTIVATION_H_
#define INC_MINDNN_ACTIVATION_H_
#include "mindnn.h"
#include <Eigen/Core>

namespace mindnn
{
enum class ActivationType
{
    Identity,
    Sigmoid,
    ReLU,
    Softmax,
    Mish,
    Tanh,
};

class Identity
{
public:
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    inline static constexpr ActivationType Type = ActivationType::Identity;

    // a = activation(z) = z
    // Z = [z1, ..., zn], A = [a1, ..., an], n observations
    static inline void activate(const Matrix& Z, Matrix& A)
    {
        A.noalias() = Z;
    }

    // Apply the Jacobian matrix J to a vector f
    // J = d_a / d_z = I
    // g = J * f = f
    // Z = [z1, ..., zn], G = [g1, ..., gn], F = [f1, ..., fn]
    // Note: When entering this function, Z and G may point to the same matrix
    static inline void apply_jacobian(const Matrix& Z, const Matrix& A, const Matrix& F, Matrix& G)
    {
        G.noalias() = F;
    }
};

class Sigmoid
{
public:
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    inline static constexpr ActivationType Type = ActivationType::Sigmoid;

    // a = activation(z) = 1 / (1 + exp(-z))
    // Z = [z1, ..., zn], A = [a1, ..., an], n observations
    static void activate(const Matrix& Z, Matrix& A);

    // Apply the Jacobian matrix J to a vector f
    // J = d_a / d_z = diag(a .* (1 - a))
    // g = J * f = a .* (1 - a) .* f
    // Z = [z1, ..., zn], G = [g1, ..., gn], F = [f1, ..., fn]
    // Note: When entering this function, Z and G may point to the same matrix
    static void apply_jacobian(const Matrix& Z, const Matrix& A, const Matrix& F, Matrix& G);
};

class ReLU
{
public:
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    inline static constexpr ActivationType Type = ActivationType::ReLU;

    // a = activation(z) = max(z, 0)
    // Z = [z1, ..., zn], A = [a1, ..., an], n observations
    static void activate(const Matrix& Z, Matrix& A);

    // Apply the Jacobian matrix J to a vector f
    // J = d_a / d_z = diag(sign(a)) = diag(a > 0)
    // g = J * f = (a > 0) .* f
    // Z = [z1, ..., zn], G = [g1, ..., gn], F = [f1, ..., fn]
    // Note: When entering this function, Z and G may point to the same matrix
    static void apply_jacobian(const Matrix& Z, const Matrix& A, const Matrix& F, Matrix& G);
};

class Softmax
{
public:
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using RowArray = Eigen::Array<Scalar, 1, Eigen::Dynamic>;
    inline static constexpr ActivationType Type = ActivationType::Softmax;

    // a = activation(z) = softmax(z)
    // Z = [z1, ..., zn], A = [a1, ..., an], n observations
    static void activate(const Matrix& Z, Matrix& A);

    // Apply the Jacobian matrix J to a vector f
    // J = d_a / d_z = diag(a) - a * a'
    // g = J * f = a .* f - a * (a' * f) = a .* (f - a'f)
    // Z = [z1, ..., zn], G = [g1, ..., gn], F = [f1, ..., fn]
    // Note: When entering this function, Z and G may point to the same matrix
    static void apply_jacobian(const Matrix& Z, const Matrix& A, const Matrix& F, Matrix& G);
};

class Mish
{
public:
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    inline static constexpr ActivationType Type = ActivationType::Mish;

    // Mish(x) = x * tanh(softplus(x))
    // softplus(x) = log(1 + exp(x))
    // a = activation(z) = Mish(z)
    // Z = [z1, ..., zn], A = [a1, ..., an], n observations
    static void activate(const Matrix& Z, Matrix& A);

    // Apply the Jacobian matrix J to a vector f
    // J = d_a / d_z = diag(Mish'(z))
    // g = J * f = Mish'(z) .* f
    // Z = [z1, ..., zn], G = [g1, ..., gn], F = [f1, ..., fn]
    // Note: When entering this function, Z and G may point to the same matrix
    static void apply_jacobian(const Matrix& Z, const Matrix& A, const Matrix& F, Matrix& G);
};

class Tanh
{
public:
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    inline static constexpr ActivationType Type = ActivationType::Tanh;

    // a = activation(z) = tanh(z)
    // Z = [z1, ..., zn], A = [a1, ..., an], n observations
    static void activate(const Matrix& Z, Matrix& A);

    // Apply the Jacobian matrix J to a vector f
    // tanh'(x) = 1 - tanh(x)^2
    // J = d_a / d_z = diag(1 - a^2)
    // g = J * f = (1 - a^2) .* f
    // Z = [z1, ..., zn], G = [g1, ..., gn], F = [f1, ..., fn]
    // Note: When entering this function, Z and G may point to the same matrix
    static void apply_jacobian(const Matrix& Z, const Matrix& A, const Matrix& F, Matrix& G);
};

} // namespace mindnn

#endif //INC_MINDNN_ACTIVATION_H_


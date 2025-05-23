#ifndef INC_MINDNN_OPTIMIZER_H_
#define INC_MINDNN_OPTIMIZER_H_
#include "mindnn.h"
#include <Eigen/Core>
#include <unordered_map>

namespace mindnn
{
class Optimizer
{
public:
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using ConstAlignedMapVec = Vector::ConstAlignedMapType;
    using AlignedMapVec = Vector::AlignedMapType;

    virtual ~Optimizer() {}

    ///
    /// Reset the optimizer to clear all historical information
    ///
    virtual void reset() {};

    ///
    /// Update the parameter vector using its gradient
    ///
    /// It is assumed that the memory addresses of `dvec` and `vec` do not
    /// change during the training process. This is used to implement optimization
    /// algorithms that have "memories". See the AdaGrad algorithm for an example.
    ///
    /// \param dvec The gradient of the parameter. Read-only
    /// \param vec  On entering, the current parameter vector. On exit, the
    ///             updated parameters.
    ///
    virtual void update(ConstAlignedMapVec& dvec, AlignedMapVec& vec) = 0;
};

class RMSProp: public Optimizer
{
public:
    using Array = Eigen::Array<Scalar, Eigen::Dynamic, 1>;

    RMSProp(const Scalar& lrate = Scalar(0.001), const Scalar& eps = Scalar(1e-6), const Scalar& gamma = Scalar(0.9));

    virtual void reset() override;

    virtual void update(ConstAlignedMapVec& dvec, AlignedMapVec& vec) override;

private:
    std::unordered_map<const Scalar*, Array> history_;

    Scalar lrate_;
    Scalar eps_;
    Scalar gamma_;
};

class SGD: public Optimizer
{
public:
    SGD(const Scalar& lrate = Scalar(0.001), const Scalar& decay = Scalar(0));

    virtual void update(ConstAlignedMapVec& dvec, AlignedMapVec& vec) override;

private:
    Scalar lrate_;
    Scalar decay_;
};

class AdaGrad: public Optimizer
{
public:
    typedef Eigen::Array<Scalar, Eigen::Dynamic, 1> Array;

    AdaGrad(const Scalar& lrate = Scalar(0.001), const Scalar& eps = Scalar(1e-6));

    virtual void reset() override;

    virtual void update(ConstAlignedMapVec& dvec, AlignedMapVec& vec) override;

private:
    std::unordered_map<const Scalar*, Array> history_;
    Scalar lrate_;
    Scalar eps_;
};

class Adam: public Optimizer
{
public:
    typedef Eigen::Array<Scalar, Eigen::Dynamic, 1> Array;

    Adam(const Scalar& lrate = Scalar(0.001), const Scalar& eps = Scalar(1e-6),
         const Scalar& beta1 = Scalar(0.9), const Scalar& beta2 = Scalar(0.999));

    virtual void reset() override;

    // https://ruder.io/optimizing-gradient-descent/index.html
    virtual void update(ConstAlignedMapVec& dvec, AlignedMapVec& vec) override;

private:
    std::unordered_map<const Scalar*, Array> history_m_;
    std::unordered_map<const Scalar*, Array> history_v_;
    Scalar beta1t_;
    Scalar beta2t_;

    Scalar rate_;
    Scalar eps_;
    Scalar beta1_;
    Scalar beta2_;
};

} // namespace mindnn
#endif // INC_MINDNN_OPTIMIZER_H_

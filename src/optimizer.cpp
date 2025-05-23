#include "optimizer.h"

namespace mindnn
{
RMSProp::RMSProp(const Scalar& lrate, const Scalar& eps, const Scalar& gamma)
    : lrate_(lrate)
    , eps_(eps)
    , gamma_(gamma)
{
}

void RMSProp::reset()
{
    history_.clear();
}

void RMSProp::update(ConstAlignedMapVec& dvec, AlignedMapVec& vec)
{
    // Get the accumulated squared gradient associated with this gradient
    Array& grad_square = history_[dvec.data()];

    // If length is zero, initialize it
    if(grad_square.size() == 0) {
        grad_square.resize(dvec.size());
        grad_square.setZero();
    }

    // Update accumulated squared gradient
    grad_square = gamma_ * grad_square + (Scalar(1) - gamma_) * dvec.array().square();
    // Update parameters
    vec.array() -= lrate_ * dvec.array() / (grad_square + eps_).sqrt();
}

SGD::SGD(const Scalar& lrate, const Scalar& decay)
    : lrate_(lrate)
    , decay_(decay)
{
}

void SGD::update(ConstAlignedMapVec& dvec, AlignedMapVec& vec)
{
    vec.noalias() -= lrate_ * (dvec + decay_ * vec);
}

AdaGrad::AdaGrad(const Scalar& lrate, const Scalar& eps)
    : lrate_(lrate)
    , eps_(eps)
{
}

void AdaGrad::reset()
{
    history_.clear();
}

void AdaGrad::update(ConstAlignedMapVec& dvec, AlignedMapVec& vec)
{
    // Get the accumulated squared gradient associated with this gradient
    Array& grad_square = history_[dvec.data()];

    // If length is zero, initialize it
    if(grad_square.size() == 0) {
        grad_square.resize(dvec.size());
        grad_square.setZero();
    }

    // Update accumulated squared gradient
    grad_square += dvec.array().square();
    // Update parameters
    vec.array() -= lrate_ * dvec.array() / (grad_square.sqrt() + eps_);
}

Adam::Adam(const Scalar& lrate, const Scalar& eps,
           const Scalar& beta1, const Scalar& beta2)
    : beta1t_(beta1)
    , beta2t_(beta2)
    , rate_(lrate)
    , eps_(eps)
    , beta1_(beta1)
    , beta2_(beta2)
{
}

void Adam::reset()
{
    history_m_.clear();
    history_v_.clear();
    beta1t_ = beta1_;
    beta2t_ = beta2_;
}

void Adam::update(ConstAlignedMapVec& dvec, AlignedMapVec& vec)
{
    using std::sqrt;
    // Get the m and v vectors associated with this gradient
    Array& mvec = history_m_[dvec.data()];
    Array& vvec = history_v_[dvec.data()];

    // If length is zero, initialize it
    if(mvec.size() == 0) {
        mvec.resize(dvec.size());
        mvec.setZero();
    }

    if(vvec.size() == 0) {
        vvec.resize(dvec.size());
        vvec.setZero();
    }

    // Update m and v vectors
    mvec = beta1_ * mvec + (Scalar(1) - beta1_) * dvec.array();
    vvec = beta2_ * vvec + (Scalar(1) - beta2_) * dvec.array().square();
    // Correction coefficients
    const Scalar correct1 = Scalar(1) / (Scalar(1) - beta1t_);
    const Scalar correct2 = Scalar(1) / sqrt(Scalar(1) - beta2t_);
    // Update parameters
    vec.array() -= (rate_ * correct1) * mvec / (correct2 * vvec.sqrt() + eps_);
    beta1t_ *= beta1_;
    beta2t_ *= beta2_;
}

} // namespace mindnn

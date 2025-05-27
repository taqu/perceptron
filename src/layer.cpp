#include "layer.h"
#ifdef _DEBUG
#include <iostream>
#endif

namespace mindnn
{
    //--- Layer
//--------------------------------------
	Layer::Layer(int32_t in_size, int32_t out_size)
{
}

	Layer::~Layer()
    {
    }

	int32_t Layer::in_size() const
    {
        return in_size_;
    }

	int32_t Layer::out_size() const
    {
        return out_size_;
    }

    LayerNorm::LayerNorm()
        :Layer(0,0)
        ,eps_(1.0e-5)
    {
    }

    void LayerNorm::initialize(Scalar , Scalar , bool , RandomPCG32_128& )
    {
    }

    void LayerNorm::forward(const Matrix& prev_layer_output)
    {
        Eigen::Index N = prev_layer_output.rows();
        Eigen::Index D = prev_layer_output.cols();
        a_.resize(N, D);
        mean_.resize(N);
        rstd_.resize(N);

        Scalar invD = 1.0 / D;
        for(int32_t i=0; i<N; ++i){
            auto row = prev_layer_output.row(i);
            Scalar mean = row.mean();
            Scalar sum = 0;
            for(int32_t j = 0; j < D; ++j) {
                Scalar x = row(j) - mean;
                sum += x*x;
            }
            mean_(i) = mean;
            rstd_(i) = 1.0f/std::sqrt(sum / D + eps_);

            for(int32_t j = 0; j < D; ++j) {
                a_(i,j) = (row(j)-mean)*rstd_(i);
            }
        }
    }

    const LayerNorm::Matrix& LayerNorm::output() const
    {
        return a_;
    }

    void LayerNorm::backward(const Matrix& prev_layer_output, const Matrix& next_layer_input)
    {
        using VectorDynamic = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

        Eigen::Index N = prev_layer_output.rows();
        Eigen::Index D = prev_layer_output.cols();
        din_.resize(N, D);
        Matrix norm(N, D);
        for(int32_t i = 0; i < N; ++i) {
            for(int32_t j=0; j<D; ++j){
                norm(i,j) = (prev_layer_output(i,j) - mean_(i))*rstd_(i);
            }
        }
        const Matrix& dnorm = next_layer_input;
        auto mean = mindnn::mean(dnorm.array() * norm.array());
        auto dnorm_mean = mindnn::mean(dnorm);
        for(int32_t i = 0; i < norm.cols(); ++i) {
            for(int32_t j = 0; j < norm.rows(); ++j) {
                din_(j, i) = dnorm(j, i) - dnorm_mean(j) - norm(j, i) * mean(j);
                din_(j, i) *= rstd_(j);
            }
        }
        std::cout << din_ << std::endl;
    }

    const LayerNorm::Matrix& LayerNorm::backward() const
    {
        return din_;
    }

    void LayerNorm::update(Optimizer& optimizer)
    {
    }

    const std::vector<Scalar>& LayerNorm::get_weights() const
    {
        return std::vector<Scalar>();
    }

    void LayerNorm::set_weights(const std::vector<Scalar>&)
    {
    }

    LayerType LayerNorm::layer_type() const
    {
        return LayerType::LayerNorm;
    }

    ActivationType LayerNorm::activation_type() const
    {
        return ActivationType::Identity;
    }

    void LayerNorm::fill_meta_info(MetaInfo& metainfo, int32_t index)
    {
        std::string istr = std::to_string(index);
            metainfo.insert_or_assign("Layer" + istr, static_cast<int32_t>(layer_type()));
    }

    const LayerNorm::Vector& LayerNorm::mean() const
    {
        return mean_;
    }

const LayerNorm::Vector& LayerNorm::rstd() const
    {
        return rstd_;
    }
    }


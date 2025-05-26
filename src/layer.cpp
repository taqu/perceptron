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

    void LayerNorm::initialize(Scalar mean, Scalar sigma, bool bias, RandomPCG32_128& random)
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
        for(int32_t i = 0; i < N; ++i) {
            for(int32_t j=0; j<D; ++j){
                norm(i,j) = (prev_layer_output(i,j) - mean_(i))*rstd_(i);
            }
        }
        auto mean = mindnn::mean(dnorm.array() * norm.array());
        std::cout << "mean: " << mean << std::endl;
        int32_t r0 = mean.rows();
        int32_t c0 = mean.cols();
        int32_t r1 = norm.rows();
        int32_t c1 = norm.cols();
        auto nmean = mean.transpose()*norm;
        int32_t nr0 = nmean.rows();
        int32_t nc0 = nmean.cols();
        std::cout << "nmean: " << nmean << std::endl;
        //auto mmean = dnorm * norm;
        //int32_t r0 = mmean.rows();
        //int32_t c0 = mmean.cols();
        //for(int32_t i=0; i<N; ++i){
        //    auto d = dnorm.row(i);
        //    auto x0 = d*norm;
        //    auto m = mindnn::mean(d) - norm*mindnn::mean(x0);
        //    dx.row(i) = d - mindnn::mean(d) - norm*mindnn::mean(d*norm);
        //    dx.row(i) *= rstd_;
        //}
    }

    const LayerNorm::Matrix& LayerNorm::backward() const
    {
        return Matrix();
    }

    void LayerNorm::update(Optimizer& optimizer)
    {
    }

    const std::vector<Scalar>& LayerNorm::get_weights() const
    {
        return std::vector<Scalar>();
    }

    void LayerNorm::set_weights(const std::vector<Scalar>& parameters)
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


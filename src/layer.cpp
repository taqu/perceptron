#include "layer.h"

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
    {
    }

    void LayerNorm::initialize(Scalar mean, Scalar sigma, bool bias, RandomPCG32_128& random)
    {
    }

    void LayerNorm::forward(const Matrix& prev_layer_output)
    {
    }

    const LayerNorm::Matrix& LayerNorm::output() const
    {
    }

    void LayerNorm::backward(const Matrix& prev_layer_output, const Matrix& next_layer_input)
    {
    }

    const LayerNorm::Matrix& LayerNorm::backward() const
    {
    
    }

    void LayerNorm::update(Optimizer& optimizer)
    {
    }

    const std::vector<Scalar>& LayerNorm::get_weights() const
    {
    }

    void LayerNorm::set_weights(const std::vector<Scalar>& parameters)
    {
    }

    LayerType LayerNorm::layer_type() const
    {
    }

    ActivationType LayerNorm::activation_type() const
    {
    }

    void LayerNorm::fill_meta_info(MetaInfo& metainfo, int32_t index)
    {
    }
}


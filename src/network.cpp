#include "network.h"
#include <bit>

namespace mindnn
{
Network::Network()
    : output_(nullptr)
{
}

Network::~Network()
{
    for(size_t i = 0; i < layers_.size(); ++i) {
        delete layers_[i];
    }
    layers_.clear();
    delete output_;
    output_ = nullptr;
}

bool Network::validate_sizes() const
{
    return true;
}

void Network::initialize(Scalar mu, Scalar sigma, int64_t seed)
{
    random_.srand(std::bit_cast<uint64_t>(seed));
    for(size_t i=0; i<layers_.size(); ++i){
        layers_[i]->initialize(mu, sigma, false, random_);
    }
}

void Network::forward(const Matrix& input)
{
    if(layers_.size() <= 0) {
        return;
    }
    layers_[0]->forward(input);
    for(size_t i = 1; i < layers_.size(); ++i) {
        layers_[i]->forward(layers_[i - 1]->output());
    }
}

void Network::update(Optimizer& optimizer)
{
    if(layers_.size() <= 0) {
        return;
    }
    for(size_t i = 0; i < layers_.size(); ++i) {
        layers_[i]->update(optimizer);
    }
}
typename Network::MetaInfo Network::get_meta_info() const
        {
            MetaInfo map;
            map.insert(std::make_pair("Nlayers", static_cast<int32_t>(layers_.size())));

            for (size_t i = 0; i < layers_.size(); ++i)
            {
                layers_[i]->fill_meta_info(map, i);
            }
            map.insert(std::make_pair("OutputLayer", static_cast<int32_t>(output_->output_type())));
            return map;
        }

} // namespace mindnn

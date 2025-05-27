#ifndef INC_MINDNN_NETWORK_H_
#define INC_MINDNN_NETWORK_H_
#include "layer.h"
#include "mindnn.h"
#include "output.h"
#include <Eigen/Core>
#include <unordered_map>
#include <vector>

namespace mindnn
{
class Layer;
class Output;
class Optimizer;

class Network
{
public:
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using IntegerVector = Eigen::RowVectorXi;
    using MetaInfo = std::unordered_map<std::string, int32_t>;
    Network();
    ~Network();

    bool validate_sizes() const;
    void initialize(Scalar mu=Scalar(0), Scalar sigma=Scalar(0.01), int64_t seed=-1);
    void forward(const Matrix& input);
    template<class TargetType>
    void backward(const Matrix& input, const TargetType& target);
    void update(Optimizer& optimizer);

    MetaInfo get_meta_info() const;

    void add_layer(Layer* layer);
    void set_output(Output* output);
private:
    RandomPCG32_128 random_;
    std::vector<Layer*> layers_;
    Output* output_;
};

template<class TargetType>
void Network::backward<TargetType>(const Matrix& input, const TargetType& target)
{
    if(layers_.size() <= 0) {
        return;
    }
    output_->check_target_data(target);
    output_->evaluate(layers_[layers_.size() - 1]);
    if(layers_.size() <= 1) {
        layers_[0]->backward(input, output_->barkward_data());
        return;
    }
    layers_[layers_.size() - 1]->backward(layers_[layers_.size() - 2]->output(), output_->barkward_data());
    for(size_t i = layers_.size() - 2; 0 < i; --i) {
        layers_[i]->backward(layers_[i - 1]->output(), layers_[i + 1]->backward());
    }
    layers_[0]->backward(input, layers_[1]->backward());
}
} // namespace mindnn
#endif // INC_MINDNN_NETWORK_H_

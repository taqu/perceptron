#ifndef INC_MINDNN_LAYER_H_
#define INC_MINDNN_LAYER_H_
#include "Eigen/Core"
#include "mindnn.h"
#include <string>
#include <unordered_map>
#include <vector>
#include "activation.h"

namespace mindnn
{
class Optimizer;

static constexpr Scalar const_zero = 0.0f;
static constexpr Scalar const_one = 1.0f;
static constexpr Scalar const_two = 2.0f;

enum class LayerType
{
    Dense,
    Convolutional,
    MaxPooling,
    LayerNorm,
};

//--- Layer
//--------------------------------------
class Layer
{
public:
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using MetaInfo = std::unordered_map<std::string, int32_t>;

    Layer(int32_t in_size, int32_t out_size);

    int32_t in_size() const;
    int32_t out_size() const;

    virtual void initialize(Scalar mean, Scalar sigma, bool bias, RandomPCG32_128& random) = 0;
    virtual void forward(const Matrix& prev_layer_output) = 0;
    virtual const Matrix& output() const = 0;
    virtual void backward(const Matrix& prev_layer_output, const Matrix& next_layer_input) = 0;
    virtual const Matrix& backward() const = 0;
    virtual void update(Optimizer& optimizer) = 0;

    virtual const std::vector<Scalar>& get_weights() const = 0;
    virtual void set_weights(const std::vector<Scalar>& parameters) = 0;

    virtual LayerType layer_type() const = 0;
    virtual ActivationType activation_type() const = 0;
    virtual void fill_meta_info(MetaInfo& metainfo, int32_t index) = 0;

protected:
    virtual ~Layer();

    int32_t in_size_;
    int32_t out_size_;
};

//--- Dense
//--------------------------------------
template<class Activation>
class Dense: public Layer
{
public:
    using ConstAlignedMapVec = Vector::ConstAlignedMapType;
    using AlignedMapVec = Vector::AlignedMapType;

    Dense(int32_t in_size, int32_t out_size);

    virtual void initialize(bool bias);
    virtual void initialize(Scalar mean, Scalar sigma, bool bias, RandomPCG32_128& random) override;
    virtual void forward(const Matrix& prev_layer_output) override;
    virtual const Matrix& output() const override;
    virtual void backward(const Matrix& prev_layer_output, const Matrix& next_layer_input) override;
    virtual const Matrix& backward() const override;
    virtual void update(Optimizer& optimizer) override;

    virtual std::vector<Scalar> get_weights() const override;
    virtual void set_weights(const std::vector<Scalar>& parameters) override;

    virtual LayerType layer_type() const override;
    virtual ActivationType activation_type() const override;
    virtual void fill_meta_info(MetaInfo& metainfo, int32_t index) override;

private:
    Dense(const Dense&) = delete;
    Dense& operator=(const Dense&) = delete;
    Matrix weight_;
    Vector bias_;
    Matrix dw_;
    Matrix db_;
    Matrix z_;
    Matrix a_;
    Matrix din_;
};

template<class Activation>
Dense<Activation>::Dense(int32_t in_size, int32_t out_size)
    : Layer(in_size, out_size)
{
}

template<class Activation>
void Dense<Activation>::initialize(bool bias)
{
    weight_.resize(in_size_, out_size_);
    dw_.resize(in_size_, out_size_);
    if(bias) {
        bias_.resize(out_size_);
        db_.resize(out_size_);
    } else {
        bias_.resize(0);
        db_.resize(0);
    }
}

template<class Activation>
void Dense<Activation>::initialize(Scalar mean, Scalar sigma, bool bias, RandomPCG32_128& random)
{
    initialize(bias);
    normal_random(weight_.data(), weight_.size(), random, mean, sigma);
    if(0 < bias_.size()) {
        normal_random(bias_.data(), bias_.size(), random, mean, sigma);
    }
}

template<class Activation>
void Dense<Activation>::forward(const Matrix& prev_layer_output)
{
    int32_t cols = prev_layer_output.cols();
    // Linear term z = W' * in + b
    z_.resize(out_size_, cols);
    z_.noalias() = weight_.transpose() * prev_layer_output;
    if(0 < bias_.size()) {
        z_.colwise() += bias_;
    }
    // Apply activation function
    a_.resize(out_size_, cols);
    Activation::activate(z_, a_);
}

template<class Activation>
const Dense<Activation>::Matrix& Dense<Activation>::output() const
{
    return a_;
}

template<class Activation>
void Dense<Activation>::backward(const Matrix& prev_layer_output, const Matrix& next_layer_input)
{
    int32_t cols = prev_layer_output.cols();
    // After forward stage, z_ contains z = W' * in + b
    // Now we need to calculate d(L) / d(z) = [d(a) / d(z)] * [d(L) / d(a)]
    // d(L) / d(a) is computed in the next layer, contained in next_layer_data
    // The Jacobian matrix J = d(a) / d(z) is determined by the activation function
    Activation::apply_jacobian(z_, a_, next_layer_data, z_);
    // Now z_ contains d(L) / d(z)
    // Derivative for weights, d(L) / d(W) = [d(L) / d(z)] * in'
    dw_.noalias() = prev_layer_output * z_.transpose() / cols;
    // Derivative for bias, d(L) / d(b) = d(L) / d(z)
    db_.noalias() = z_.rowwise().mean();
    // Compute d(L) / din_ = W * [d(L) / d(z)]
    din_.resize(in_size_, cols);
    din_.noalias() = weight_ * z_;
}

template<class Activation>
const Dense<Activation>::Matrix& Dense<Activation>::backward() const
{
    return din_;
}

template<class Activation>
void Dense<Activation>::update(Optimizer& optimizer)
{
    ConstAlignedMapVec dw(dw_.data(), dw_.size());
    ConstAlignedMapVec db(db_.data(), db_.size());
    AlignedMapVec w(weight_.data(), weight_.size());
    optimizer.update(dw, w);
    if(0 < bias_.size()) {
        AlignedMapVec b(bias_.data(), bias_.size());
        optimizer.update(db, b);
    }
}

template<class Activation>
std::vector<Scalar> Dense<Activation>::get_weights() const
{
    if(0 < bias_.size()) {
        std::vector<Scalar> result(weight_.size() + bias_.size());
        std::copy(weight_.data(), weight_.data() + weight_.size(), result.begin());
        std::copy(bias_.data(), bias_.data() + bias_.size(), result.begin() + weight_.size());
        return result;
    } else {
        std::vector<Scalar> result(weight_.size());
        std::copy(weight_.data(), weight_.data() + weight_.size(), result.begin());
    }
}

template<class Activation>
void Dense<Activation>::set_weights(const std::vector<Scalar>& parameters)
{
    assert(static_cast<Eigen::Index>(parameters.size()) == (weight_.size() + bias_.size()));
    std::copy(parameters.begin(), parameters.begin() + weight_.size(), weight_.data());
    if(0 < bias_.size()) {
        std::copy(parameters.begin() + weight_.size(), parameters.end(), bias_.data());
    }
}

template<class Activation>
LayerType Dense<Activation>::layer_type() const
{
    return LayerType::Dense;
}

template<class Activation>
ActivationType Dense<Activation>::activation_type() const
{
    return Activation::Type;
}

template<class Activation>
void Dense<Activation>::fill_meta_info(MetaInfo& metainfo, int32_t index)
{
    std::string istr = std::to_string(index);
    metainfo.insert_or_assign("Layer" + istr, static_cast<int32_t>(layer_type()));
    metainfo.insert_or_assign("Activation" + istr, static_cast<int32_t>(activation_type()));
    metainfo.insert_or_assign("in_size" + istr, in_size());
    metainfo.insert_or_assign("out_size" + istr, out_size());
}

//--- Convolutional
//--------------------------------------
template<class Activation>
class Convolutional: public Layer
{
public:
    using ConstAlignedMapVec = Vector::ConstAlignedMapType;
    using ConstAlignedMapMat = Matrix::ConstAlignedMapType;
    using AlignedMapVec = Vector::AlignedMapType;
    using RMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using ConstMapMat = Eigen::Map<const Matrix>;

    struct Dims
    {
        Dims(int32_t in_channels_, int32_t out_channels_,
             int32_t channel_rows_, int32_t channel_cols_,
             int32_t filter_rows_, int32_t filter_cols_)
            : in_channels_(in_channels_)
            , out_channels_(out_channels_)
            , channel_rows_(channel_rows_)
            , channel_cols_(channel_cols_)
            , filter_rows_(filter_rows_)
            , filter_cols_(filter_cols_)
            , img_rows_(channel_rows_)
            , img_cols_(in_channels_ * channel_cols_)
            , conv_rows_(channel_rows_ - filter_rows_ + 1)
            , conv_cols_(channel_cols_ - filter_cols_ + 1)
        {
        }

        int32_t in_channels_;
        int32_t out_channels_;
        int32_t channel_rows_;
        int32_t channel_cols_;
        int32_t filter_rows_;
        int32_t filter_cols_;
        int32_t img_rows_;
        int32_t img_cols_;
        int32_t conv_rows_;
        int32_t conv_cols_;
    };

    static void convolve_valid(
        Scalar* dst,
        const Dims& dims,
        const Scalar* src, bool image_outer_loop, int32_t cols,
        const Scalar* filter_data);

    static void flatten_mat(
        RMatrix& flat_matrix,
        const Dims& dims, const Scalar* src,
        int32_t stride, int32_t ncols);

    static void moving_product(
        Matrix& result,
        int32_t step,
        const RMatrix& mat1,
        Eigen::Map<const Matrix>& mat2);

    static void convolve_full(
        Scalar* dst,
    const Dims& dims,
    const Scalar* src, int32_t ncols, const Scalar* filter_data);

    Convolutional(int32_t in_width, int32_t in_height,
                  int32_t in_channels_, int32_t out_channels_,
                  int32_t window_width, int32_t window_height);

    virtual void initialize(bool bias);
    virtual void initialize(Scalar mean, Scalar sigma, bool bias, RandomPCG32_128& random) override;
    virtual void forward(const Matrix& prev_layer_output) override;
    virtual const Matrix& output() const override;
    virtual void backward(const Matrix& prev_layer_output, const Matrix& next_layer_input) override;
    virtual const Matrix& backward() const override;
    virtual void update(Optimizer& optimizer) override;

    virtual std::vector<Scalar> get_weights() const override;
    virtual void set_weights(const std::vector<Scalar>& parameters) override;
    std::vector<Scalar> get_derivatives() const;

    virtual LayerType layer_type() const override;
    virtual ActivationType activation_type() const override;
    virtual void fill_meta_info(MetaInfo& metainfo, int32_t index) override;

private:
    Convolutional(const Convolutional&) = delete;
    Convolutional& operator=(const Convolutional&) = delete;

    Dims dims_;
    Vector filter_;
    Vector df_;
    Vector bias_;
    Vector db_;
    Matrix z_;
    Matrix a_;
    Matrix din_;
};

template<class Activation>
void Convolutional<Activation>::convolve_valid(
    Scalar* dst,
    const Dims& dims,
    const Scalar* src, bool image_outer_loop, int32_t cols,
    const Scalar* filter_data)
{
    // Flat matrix
    int32_t flat_rows = dims.conv_rows_ * cols;
    int32_t flat_cols = dims.filter_rows_ * dims.channel_cols_;
    int32_t channel_size = dims.channel_rows_ * dims.channel_cols_;
    // Distance between two images
    int32_t img_stride = image_outer_loop ? (dims.img_rows_ * dims.img_cols_) : channel_size;
    // Distance between two channels
    int32_t channel_stride = image_outer_loop ? channel_size : (channel_size * cols);
    RMatrix flat_mat(flat_rows, flat_cols);
    // Convolution results
    int32_t& res_rows = flat_rows;
    int32_t res_cols = dims.conv_cols_ * dims.out_channels_;
    Matrix res = Matrix::Zero(res_rows, res_cols);
    int32_t& step = dims.filter_rows_;
    int32_t filter_size = dims.filter_rows_ * dims.filter_cols_;
    int32_t filter_stride = filter_size_ * dims.out_channels_;

    for(int32_t i = 0; i < dims.in_channels_; ++i, src += channel_stride, filter_data += filter_stride) {
        // Flatten source image
        flatten_mat(flat_mat, dims, src, img_stride, cols);
        // Compute the convolution result
        ConstMapMat filter(filter_data, filter_size, dims.out_channels_);
        moving_product(step, flat_mat, filter, res);
    }

    // The layout of 'res' is very complicated
    /*
     * obs0_out0[0, 0] obs0_out1[0, 0] obs0_out2[0, 0] obs0_out0[0, 1] obs0_out1[0, 1] obs0_out2[0, 1] ...
     * obs0_out0[1, 0] obs0_out1[1, 0] obs0_out2[1, 0] obs0_out0[1, 1] obs0_out1[1, 1] obs0_out2[1, 1] ...
     * obs0_out0[2, 0] obs0_out1[2, 0] obs0_out2[2, 0] obs0_out0[2, 1] obs0_out1[2, 1] obs0_out2[2, 1] ...
     * obs1_out0[0, 0] obs1_out1[0, 0] obs1_out2[0, 0] obs1_out0[0, 1] obs1_out1[0, 1] obs1_out2[0, 1] ...
     * obs1_out0[1, 0] obs1_out1[1, 0] obs1_out2[1, 0] obs1_out0[1, 1] obs1_out1[1, 1] obs1_out2[1, 1] ...
     * obs1_out0[2, 0] obs1_out1[2, 0] obs1_out2[2, 0] obs1_out0[2, 1] obs1_out1[2, 1] obs1_out2[2, 1] ...
     * ...
     *
     */
    // obs<k>_out<l> means the convolution result of the k-th image on the l-th output channel
    // [i, j] gives the matrix indices
    // The destination has the layout
    /*
     * obs0_out0[0, 0] obs0_out0[0, 1] obs0_out0[0, 2] obs0_out1[0, 0] obs0_out1[0, 1] obs0_out1[0, 2] ...
     * obs0_out0[1, 0] obs0_out0[1, 1] obs0_out0[1, 2] obs0_out1[1, 0] obs0_out1[1, 1] obs0_out1[1, 2] ...
     * obs0_out0[2, 0] obs0_out0[2, 1] obs0_out0[2, 2] obs0_out1[2, 0] obs0_out1[2, 1] obs0_out1[2, 2] ...
     *
     */
    // which in a larger scale looks like
    // [obs0_out0 obs0_out1 obs0_out2 obs1_out0 obs1_out1 obs1_out2 obs2_out0 ...]
    // Copy data to destination
    // dst[a, b] corresponds to obs<k>_out<l>[i, j]
    // where k = b / (conv_cols_ * out_channels_),
    //       l = (b % (conv_cols_ * out_channels_)) / conv_cols_
    //       i = a,
    //       j = b % conv_cols_
    // and then obs<k>_out<l>[i, j] corresponds to res[c, d]
    // where c = k * conv_rows_ + i,
    //       d = j * out_channels_ + l
    int32_t dest_rows = dims.conv_rows_;
    int32_t dest_cols = res_cols * cols;
    const Scalar* res_data = res.data();
    std::size_t copy_bytes = sizeof(Scalar) * dest_rows;

    for(int32_t b = 0; b < dest_cols; ++b, dst += dest_rows) {
        int32_t k = b / res_cols;
        int32_t l = (b % res_cols) / dims.conv_cols_;
        int32_t j = b % dims.conv_cols_;
        int32_t d = j * dims.out_channels_ + l;
        int32_t res_col_head = d * res_rows;
        std::memcpy(dst, res_data + res_col_head + k * dims.conv_rows_, copy_bytes);
    }
}

template<class Activation>
void Convolutional<Activation>::flatten_mat(
        RMatrix& flat_matrix,
        const Dims& dims, const Scalar* src,
        int32_t stride, int32_t ncols)
{
    // Number of bytes in the segment that will be copied at one time
    int32_t segment_size = dims.filter_rows_;
    size_t copy_bytes = sizeof(Scalar) * segment_size;
    Scalar* writer = flat_mat.data();
    int32_t channel_size = dims.channel_rows_ * dims.channel_cols_;

    for (int32_t i = 0; i < ncols; ++i, src += stride)
    {
        const Scalar* reader_row = src;
        const Scalar* const reader_row_end = src + dims.conv_rows_;

        for (; reader_row < reader_row_end; ++reader_row)
        {
            const Scalar* reader = reader_row;
            const Scalar* const reader_end = reader + channel_size;

            for (; reader < reader_end; reader += dims.channel_rows_, writer += segment_size)
            {
                ::memcpy(writer, reader, copy_bytes);
            }
        }
    }

}

template<class Activation>
void Convolutional<Activation>::moving_product(
        Matrix& result,
        int32_t step,
        const RMatrix& mat1,
        Eigen::Map<const Matrix>& mat2)
{
    int32_t row1 = mat1.rows();
    int32_t col1 = mat1.cols();
    int32_t row2 = mat2.rows();
    int32_t col2 = mat2.cols();
    int32_t col_end = col1 - row2;
    int32_t res_start_col = 0;

    for (int32_t left_end = 0; left_end <= col_end; left_end += step, res_start_col += col2)
    {
        result.block(0, res_start_col, row1, col2).noalias() += mat1.block(0, left_end, row1, row2) * mat2;
    }
}

template<class Activation>
void Convolutional<Activation>::convolve_full(
        Scalar* dst,
    const Dims& dims,
    const Scalar* src, int32_t ncols, const Scalar* filter_data)
{
    // Padding sizes
    int32_t padding_top = dims.filter_rows_ - 1;
    int32_t padding_left = dims.filter_cols_ - 1;
    // Dimension of convolution result using "full" rule
    int32_t conv_rows_ = dims.channel_rows_ + padding_top;
    int32_t conv_cols_ = dims.channel_cols_ + padding_left;
    // Add (top and bottom) padding to source images
    int32_t pad_rows = dims.img_rows_ + padding_top * 2;
    int32_t pad_cols = dims.img_cols_ * ncols;
    Matrix pad_mat(pad_rows, pad_cols);
    ConstMapMat src_mat(src, dims.img_rows_, pad_cols);
    pad_mat.topRows(padding_top).setZero();
    pad_mat.bottomRows(padding_top).setZero();
    pad_mat.block(padding_top, 0, dims.img_rows_, pad_cols).noalias() = src_mat;
    src = pad_mat.data();
    ConvDims pad_dim(dims.in_channels_, dims.out_channels_, pad_rows, dims.channel_cols_,
                     dims.filter_rows_, dims.filter_cols_);
    // Flat matrix
    int32_t flat_rows = conv_rows_ * ncols;
    int32_t flat_cols = dims.filter_rows_ * dims.channel_cols_;
    int32_t img_stride = pad_rows * dims.img_cols_;
    int32_t channel_stride = pad_rows * dims.channel_cols_;
    RMatrix flat_mat(flat_rows, flat_cols);
    // The processing of filters are different from the "valid" rule in two ways:
    // 1. The layout of input channels and output channels are switched
    // 2. The filters need to be rotated, which is equivalent to reversing the vector of each filter
    // We also separate filters that belong to different input channels
    std::vector<Matrix> filters_in(dims.in_channels_);
    int32_t filter_size = dims.filter_rows_ * dims.filter_cols_;
    int32_t nfilter = dims.in_channels_ * dims.out_channels_;

    for (int32_t i = 0; i < dims.in_channels_; ++i)
    {
        filters_in[i].resize(filter_size, dims.out_channels_);
    }

    const Scalar* reader = filter_data;

    for (int32_t i = 0; i < nfilter; ++i, reader += filter_size)
    {
        Scalar* writer = filters_in[i % dims.in_channels_].data() +
                         (i / dims.in_channels_) * filter_size;
        std::reverse_copy(reader, reader + filter_size, writer);
    }

    // Convolution results
    int32_t res_rows = flat_rows;
    int32_t res_cols = conv_cols_ * dims.out_channels_;
    Matrix res = Matrix::Zero(res_rows, res_cols);
    int32_t step = dims.filter_rows_;
    int32_t filter_padding = padding_left * dims.filter_rows_;

    for (int32_t i = 0; i < dims.in_channels_; ++i, src += channel_stride)
    {
        // Flatten source image
        flatten_mat(flat_mat, pad_dim, src, img_stride, ncols);
        // Compute the convolution result
        moving_product(res, filter_padding, step, flat_mat, filters_in[i]);
    }

    // Copy results to destination
    int32_t dest_rows = conv_rows_;
    int32_t  dest_cols = res_cols * ncols;
    const Scalar* res_data = res.data();
    const std::size_t copy_bytes = sizeof(Scalar) * dest_rows;

    for (int32_t b = 0; b < dest_cols; ++b, dst += dest_rows)
    {
        int32_t k = b / res_cols;
        int32_t l = (b % res_cols) / conv_cols_;
        int32_t j = b % conv_cols_;
        int32_t d = j * dims.out_channels_ + l;
        int32_t res_col_head = d * res_rows;
        std::memcpy(dst, res_data + res_col_head + k * conv_rows_, copy_bytes);
    }
}

template<class Activation>
Convolutional<Activation>::Convolutional(int32_t in_width, int32_t in_height,
                                         int32_t in_channels_, int32_t out_channels_,
                                         int32_t window_width, int32_t window_height)
    : Layer(in_width * in_height * in_channels_, (in_width - window_width + 1) * (in_height - window_height + 1) * out_channels_)
    , dims_(in_channels_, out_channels_, in_height, in_width, window_height, window_width)
{
}

template<class Activation>
void Convolutional<Activation>::initialize(bool bias)
{
    int32_t filter_data_size = dims_.in_channels_ * dims_.out_channels_ * dims_.filter_rows_ * dims_.filter_cols_;
    // Filter parameters
    filter_.resize(filter_data_size);
    df_.resize(filter_data_size);
    // Bias term
    if(bias) {
        bias_.resize(dims_.out_channels_);
        db_.resize(dims_.out_channels_);
    } else {
        bias_.resize(0);
        db_.resize(0);
    }
}

template<class Activation>
void Convolutional<Activation>::initialize(Scalar mean, Scalar sigma, bool bias, RandomPCG32_128& random)
{
    initialize(bias);
    normal_random(filter_.data(), filter_.size(), random, mean, sigma);
    if(0 < bias_.size()) {
        normal_random(bias_.data(), bias_.size(), random, mean, sigma);
    }
}

template<class Activation>
void Convolutional<Activation>::forward(const Matrix& prev_layer_output)
{
    // Each column is an observation
    int32_t cols = prev_layer_output.cols();
    // Linear term, z = conv(in, w) + b
    z_.resize(out_size_, cols);
    // Convolution
    convolve_valid(z_.data(), dims_, prev_layer_output.data(), true, cols, filter_.data());

    // Add bias terms
    // Each column of z_ contains dims_.out_channels_ channels, and each channel has
    // dims_.conv_rows_ * dims_.conv_cols_ elements
    int32_t channel_start_row = 0;
    int32_t channel_nelem = dims_.conv_rows_ * dims_.conv_cols_;

    if(0 < bias_.size()) {
        for(int32_t i = 0; i < dims_.out_channels_; ++i, channel_start_row += channel_nelem) {
            z_.block(channel_start_row, 0, channel_nelem, cols).array() += bias_[i];
        }
    }

    // Apply activation function
    a_.resize(out_size_, cols);
    Activation::activate(z_, a_);
}

template<class Activation>
const Convolutional<Activation>::Matrix& Convolutional<Activation>::output() const
{
    return a_;
}

template<class Activation>
void Convolutional<Activation>::backward(const Matrix& prev_layer_output, const Matrix& next_layer_input)
{
    int32_t cols = prev_layer_output.cols();
    // After forward stage, z_ contains z = conv(in, w) + b
    // Now we need to calculate d(L) / d(z) = [d(a) / d(z)] * [d(L) / d(a)]
    // d(L) / d(a) is computed in the next layer, contained in next_layer_data
    // The Jacobian matrix J = d(a) / d(z) is determined by the activation function
    Matrix& dLz = z_;
    Activation::apply_jacobian(z_, a_, next_layer_data, dLz);
    // z_j = sum_i(conv(in_i, w_ij)) + b_j
    //
    // d(z_k) / d(w_ij) = 0, if k != j
    // d(L) / d(w_ij) = [d(z_j) / d(w_ij)] * [d(L) / d(z_j)] = sum_i{ [d(z_j) / d(w_ij)] * [d(L) / d(z_j)] }
    // = sum_i(conv(in_i, d(L) / d(z_j)))
    //
    // z_j is an image (matrix), b_j is a scalar
    // d(z_j) / d(b_j) = a matrix of the same size of d(z_j) filled with 1
    // d(L) / d(b_j) = (d(L) / d(z_j)).sum()
    //
    // d(z_j) / d(in_i) = conv_full_op(w_ij_rotate)
    // d(L) / d(in_i) = sum_j((d(z_j) / d(in_i)) * (d(L) / d(z_j))) = sum_j(conv_full(d(L) / d(z_j), w_ij_rotate))
    // Derivative for weights
    Dims back_conv_dim(cols, dims_.out_channels_, dims_.channel_rows_,
                                     dims_.channel_cols_,
                                     dims_.conv_rows_, dims_.conv_cols_);
    convolve_valid(df_.data(), back_conv_dim, prev_layer_output.data(), false,
                             dims_.in_channels_, dLz.data());
    df_ /= cols;
    // Derivative for bias
    // Aggregate d(L) / d(z) in each output channel
    ConstAlignedMapMat dLz_by_channel(dLz.data(), dims_.conv_rows_ * dims_.conv_cols_,
                                      dims_.out_channels_ * cols);
    Vector dLb = dLz_by_channel.colwise().sum();
    // Average over observations
    ConstAlignedMapMat dLb_by_obs(dLb.data(), dims_.out_channels_, cols);
    db_.noalias() = dLb_by_obs.rowwise().mean();
    // Compute d(L) / d_in = conv_full(d(L) / d(z), w_rotate)
    din_.resize(this->m_in_size, cols);
    Dims conv_full_dim(dims_.out_channels_, dims_.in_channels_,
                                     dims_.conv_rows_, dims_.conv_cols_, dims_.filter_rows_, dims_.filter_cols_);
    convolve_full(din_.data(), conv_full_dim, dLz.data(), cols, filter_.data());
}

template<class Activation>
const Convolutional<Activation>::Matrix& Convolutional<Activation>::backward() const
{
    return din_;
}

template<class Activation>
void Convolutional<Activation>::update(Optimizer& optimizer)
{
    ConstAlignedMapVec dw(df_.data(), df_.size());
    ConstAlignedMapVec db(db_.data(), db_.size());
    AlignedMapVec w(filter_.data(), filter_.size());
    optimizer.update(dw, w);
    if(0 < bias_.size()) {
        AlignedMapVec b(bias_.data(), bias_.size());
        optimizer.update(db, b);
    }
}

template<class Activation>
std::vector<Scalar> Convolutional<Activation>::get_weights() const
{
    std::vector<Scalar> result(filter_.size() + bias_.size());
    // Copy the data of filters and bias to a long vector
    std::copy(filter_.data(), filter_.data() + filter_.size(), result.begin());
    if(0<bias_.size()){
    std::copy(bias_.data(), bias_.data() + bias_.size(), result.begin() + filter_.size());
    }
    return result;
}

template<class Activation>
void Convolutional<Activation>::set_weights(const std::vector<Scalar>& parameters)
{
    assert(static_cast<Eigen::Index>(parameters.size()) == (filter_.size() + bias_.size()));

    std::copy(parameters.begin(), parameters.begin() + filter_.size(), filter_.data());
    if(0 < bias_.size()) {
        std::copy(parameters.begin() + filter_.size(), parameters.end(), bias_.data());
    }
}

template<class Activation>
std::vector<Scalar> Convolutional<Activation>::get_derivatives() const
{
    std::vector<Scalar> result(df_.size() + db_.size());
    // Copy the data of filters and bias to a long vector
    std::copy(df_.data(), df_.data() + df_.size(), result.begin());
    std::copy(db_.data(), db_.data() + db_.size(), result.begin() + df_.size());
    return res;
}

template<class Activation>
LayerType Convolutional<Activation>::layer_type() const
{
    return LayerType::Convolutional;
}

template<class Activation>
ActivationType Convolutional<Activation>::activation_type() const
{
    return Activation::Type;
}

template<class Activation>
void Convolutional<Activation>::fill_meta_info(MetaInfo& metainfo, int32_t index)
{
    std::string istr = std::to_string(index);
    metainfo.insert_or_assign("Layer" + istr, static_cast<int32_t>(layer_type()));
    metainfo.insert_or_assign("Activation" + istr, static_cast<int32_t>(activation_type()));
    metainfo.insert_or_assign("in_channels_" + istr, dims_.in_channels_);
    metainfo.insert_or_assign("out_channels_" + istr, dims_.out_channels_);
    metainfo.insert_or_assign("in_height" + istr, dims_.channel_rows_);
    metainfo.insert_or_assign("in_width" + istr, dims_.channel_cols_);
    metainfo.insert_or_assign("window_width" + istr, dims_.filter_cols_);
    metainfo.insert_or_assign("window_height" + istr, dims_.filter_rows_);
}


//--- MaxPooling
//------------------------------------
template<class Activation>
class MaxPooling: public Layer
{
    private:
        using IntMatrix = Eigen::MatrixXi;

    public:
        // Currently we only implement the "valid" rule
        // https://stackoverflow.com/q/37674306
        MaxPooling(int32_t in_width_, int32_t in_height_, int32_t in_channels_,
                   int32_t pooling_width_, int32_t pooling_height_) :
            Layer(in_width_ * in_height_ * in_channels_,
                  (in_width_ / pooling_width_) * (in_height_ / pooling_height_) * in_channels_),
            channel_rows_(in_height_), channel_cols_(in_width_),
            in_channels_(in_channels_),
            pool_rows_(pooling_height_), pool_cols_(pooling_width_),
            out_rows_(channel_rows_ / pool_rows_),
            out_cols_(channel_cols_ / pool_cols_)
        {}

        virtual void initialize(const Scalar& mu, const Scalar& sigma, RandomPCG32_128& random) override;

        virtual void initialize() override;

        virtual void forward(const Matrix& prev_layer_data) override;
        virtual const Matrix& output() const override;
        virtual void backward(const Matrix& prev_layer_data, const Matrix& next_layer_data) override;
        virtual const Matrix& backward() const override;
        virtual void update(Optimizer& opt) override;

        virtual std::vector<Scalar> get_parameters() const override;
        virtual void set_parameters(const std::vector<Scalar>& param) override;

        virtual std::vector<Scalar> get_derivatives() const override;
        virtual LayerType layer_type() const override;
        virtual ActivationType activation_type() const override;

        virtual void fill_meta_info(MetaInfo& metainfo, int32_t index) const override;

        private:
            int32_t channel_rows_;
        int32_t channel_cols_;
        int32_t in_channels_;
        int32_t pool_rows_;
        int32_t pool_cols_;

        int32_t out_rows_;
        int32_t out_cols_;

        IntMatrix loc_;             // Record the locations of maximums
        Matrix z_;                  // Max pooling results
        Matrix a_;                  // Output of this layer, a = act(z)
        Matrix din_;                // Derivative of the input of this layer.
                                     // Note that input of this layer is also the output of previous layer

};

template<class Activation>
MaxPooling<Activation>::MaxPooling(int32_t in_width_, int32_t in_height_, int32_t in_channels_,
                   int32_t pooling_width_, int32_t pooling_height_) :
            Layer(in_width_ * in_height_ * in_channels_,
                  (in_width_ / pooling_width_) * (in_height_ / pooling_height_) * in_channels_),
            channel_rows_(in_height_), channel_cols_(in_width_),
            in_channels_(in_channels_),
            pool_rows_(pooling_height_), pool_cols_(pooling_width_),
            out_rows_(channel_rows_ / pool_rows_),
            out_cols_(channel_cols_ / pool_cols_)
        {}

template<class Activation>
        void MaxPooling<Activation>::initialize(const Scalar& mu, const Scalar& sigma, RandomPCG32_128& random)
        {
        }

        template<class Activation>
        void MaxPooling<Activation>::initialize()
        {
        }

        template<class Activation>
        void MaxPooling<Activation>::forward(const Matrix& prev_layer_data)
        {
            // Each column is an observation
            int32_t nobs = prev_layer_data.cols();
            loc_.resize(this->m_out_size, nobs);
            z_.resize(this->m_out_size, nobs);
            // Use loc_ to store the address of each pooling block relative to the beginning of the data
            int32_t* loc_data = loc_.data();
            int32_t channel_end = prev_layer_data.size();
            int32_t channel_stride = channel_rows_ * channel_cols_;
            int32_t col_end_gap = channel_rows_ * pool_cols_ * out_cols_;
            int32_t col_stride = channel_rows_ * pool_cols_;
            int32_t row_end_gap = out_rows_ * pool_rows_;

            for (int32_t channel_start = 0; channel_start < channel_end;
                    channel_start += channel_stride)
            {
                int32_t col_end = channel_start + col_end_gap;

                for (int32_t col_start = channel_start; col_start < col_end;
                        col_start += col_stride)
                {
                    int32_t row_end = col_start + row_end_gap;

                    for (int32_t row_start = col_start; row_start < row_end;
                            row_start += pool_rows_, loc_data++)
                    {
                        *loc_data = row_start;
                    }
                }
            }

            // Find the location of the max value in each block
            loc_data = loc_.data();
            const int32_t* const loc_end = loc_data + loc_.size();
            Scalar* z_data = z_.data();
            const Scalar* src = prev_layer_data.data();

            for (; loc_data < loc_end; loc_data++, z_data++)
            {
                int32_t offset = *loc_data;
                *z_data = internal::find_block_max(src + offset, pool_rows_, pool_cols_,
                                                   channel_rows_, *loc_data);
                *loc_data += offset;
            }

            // Apply activation function
            a_.resize(this->m_out_size, nobs);
            Activation::activate(z_, a_);
        }

        template<class Activation>
        const MaxPooling<Activation>::Matrix& MaxPooling<Activation>::output() const
        {
            return a_;
        }

        template<class Activation>
        void MaxPooling<Activation>::backward(const Matrix& prev_layer_data, const Matrix& next_layer_data)
        {
            int32_t nobs = prev_layer_data.cols();
            // After forward stage, z_ contains z = max_pooling(in)
            // Now we need to calculate d(L) / d(z) = [d(a) / d(z)] * [d(L) / d(a)]
            // d(L) / d(z) is computed in the next layer, contained in next_layer_data
            // The Jacobian matrix J = d(a) / d(z) is determined by the activation function
            Matrix& dLz = z_;
            Activation::apply_jacobian(z_, a_, next_layer_data, dLz);
            // d(L) / d(in_i) = sum_j{ [d(z_j) / d(in_i)] * [d(L) / d(z_j)] }
            // d(z_j) / d(in_i) = 1 if in_i is used to compute z_j and is the maximum
            //                  = 0 otherwise
            din_.resize(this->m_in_size, nobs);
            din_.setZero();
            int32_t dLz_size = dLz.size();
            const Scalar* dLz_data = dLz.data();
            const int32_t* loc_data = loc_.data();
            Scalar* din_data = din_.data();

            for (int32_t i = 0; i < dLz_size; i++)
            {
                din_data[loc_data[i]] += dLz_data[i];
            }
        }

        template<class Activation>
        const MaxPooling<Activation>::Matrix& MaxPooling<Activation>::backward() const
        {
            return din_;
        }

        template<class Activation>
        void MaxPooling<Activation>::update(Optimizer& opt) {}

        template<class Activation>
        std::vector<Scalar> MaxPooling<Activation>::get_parameters() const
        {
            return std::vector<Scalar>();
        }

        template<class Activation>
        void MaxPooling<Activation>::set_parameters(const std::vector<Scalar>& param) {}

        template<class Activation>
        std::vector<Scalar> MaxPooling<Activation>::get_derivatives() const
        {
            return std::vector<Scalar>();
        }

        template<class Activation>
        LayerType MaxPooling<Activation>::layer_type() const
        {
            return LayerType::MaxPooling;
        }

        template<class Activation>
        ActivationType MaxPooling<Activation>::activation_type() const
        {
            return Activation::Type;
        }

        template<class Activation>
        void MaxPooling<Activation>::fill_meta_info(MetaInfo& metainfo, int32_t index) const
        {
            std::string istr = std::to_string(index);
            metainfo.insert_or_assign("Layer" + istr, static_cast<int32_t>(layer_type()));
    metainfo.insert_or_assign("Activation" + istr, static_cast<int32_t>(activation_type()));
    metainfo.insert_or_assign("in_width" + istr, channel_cols_);
    metainfo.insert_or_assign("in_height" + istr, channel_rows_);
    metainfo.insert_or_assign("in_channels" + istr, in_channels_);
    metainfo.insert_or_assign("pooling_width" + istr, pool_cols_);
metainfo.insert_or_assign("pooling_height" + istr, pool_rows_);
        }

        //--- LayerNorm
        //--------------------------
        class LayerNorm : public Layer
        {
        public:
            LayerNorm();

    virtual void initialize(Scalar mean, Scalar sigma, bool bias, RandomPCG32_128& random) override;
    virtual void forward(const Matrix& prev_layer_output) override;
    virtual const Matrix& output() const override;
    virtual void backward(const Matrix& prev_layer_output, const Matrix& next_layer_input) override;
    virtual const Matrix& backward() const override;
    virtual void update(Optimizer& optimizer) override;

    virtual const std::vector<Scalar>& get_weights() const override;
    virtual void set_weights(const std::vector<Scalar>& parameters) override;

    virtual LayerType layer_type() const override;
    virtual ActivationType activation_type() const override;
    virtual void fill_meta_info(MetaInfo& metainfo, int32_t index) override;
        };
        } // namespace mindnn
#endif // INC_MINDNN_LAYER_H_

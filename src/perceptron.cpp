#include "perceptron.h"
#include <cassert>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace perceptron
{
//--- System
//-----------------------------------------------------
System System::instance_;

System& System::getInstance()
{
    return instance_;
}

void System::initialize()
{
    std::random_device device;
    instance_.engine_.seed(device());
}

void System::terminate()
{
}

std::mt19937& System::getRand()
{
    return engine_;
}

System::System()
{
}

System::~System()
{
}

//--- Tensor
//-----------------------------------------------------
Tensor::Tensor()
    : ndims_(0)
    , dims_{}
    , data_(nullptr)
{
}

Tensor::Tensor(std::initializer_list<u32> dims)
    : ndims_(static_cast<u32>(dims.size()))
    , dims_{}
{
    assert(0 < dims.size() && dims.size() <= 4);
    u32 count = 0;
    u32 total = 1;
    for(u32 d: dims) {
        dims_[count] = d;
        total *= d;
        ++count;
    }
    data_ = (f32*)PERC_MALLOC(sizeof(f32) * total);
    ::memset(data_, 0, sizeof(f32) * total);
}

Tensor::Tensor(Tensor&& other)
    : ndims_(other.ndims_)
    , data_(other.data_)
{
    ::memcpy(dims_, other.dims_, sizeof(u32) * 4);

    other.ndims_ = 0;
    ::memset(other.dims_, 0, sizeof(u32) * 4);
    other.data_ = nullptr;
}

Tensor& Tensor::operator=(Tensor&& other)
{
    if(this != &other) {
        PERC_FREE(data_);
        ndims_ = other.ndims_;
        ::memcpy(dims_, other.dims_, sizeof(u32) * 4);
        data_ = other.data_;

        other.ndims_ = 0;
        ::memset(other.dims_, 0, sizeof(u32) * 4);
        other.data_ = nullptr;
    }
    return *this;
}

Tensor::Tensor(const Tensor& other)
    : ndims_(other.ndims_)
{
    ::memcpy(dims_, other.dims_, sizeof(u32) * 4);
    u32 size = sizeof(f32) * other.total();
    data_ = static_cast<f32*>(PERC_MALLOC(size));
    ::memcpy(data_, other.data_, size);
}

Tensor& Tensor::operator=(const Tensor& other)
{
    if(this != &other) {
        PERC_FREE(data_);
        ndims_ = other.ndims_;
        ::memcpy(dims_, other.dims_, sizeof(u32) * 4);
        u32 size = sizeof(f32) * other.total();
        data_ = static_cast<f32*>(PERC_MALLOC(size));
        ::memcpy(data_, other.data_, size);
    }
    return *this;
}

Tensor::~Tensor()
{
    PERC_FREE(data_);
    data_ = nullptr;
}

void Tensor::reshape(u32 ndims, const u32 dims[4])
{
    ndims_ = ndims;
    ::memcpy(dims_, dims, sizeof(u32) * 4);

    PERC_FREE(data_);
    data_ = nullptr;
    u32 size = total() * sizeof(f32);
    data_ = static_cast<f32*>(PERC_MALLOC(size));
}

u32 Tensor::ndims() const
{
    return ndims_;
}

u32 Tensor::dim(u32 d) const
{
    return dims_[d];
}

const u32* Tensor::dims() const
{
    return dims_;
}

u32 Tensor::total() const
{
    u32 p = dims_[0];
    for(u32 i = 1; i < ndims_; ++i) {
        p *= dims_[i];
    }
    return p;
}

f32 Tensor::operator[](u32 x0) const
{
    return data_[x0];
}

f32& Tensor::operator[](u32 x0)
{
    return data_[x0];
}

f32 Tensor::operator()(u32 x0) const
{
    assert(ndims_ == 1);
    assert(x0 < dims_[0]);
    return data_[x0];
}

f32& Tensor::operator()(u32 x0)
{
    assert(ndims_ == 1);
    assert(x0 < dims_[0]);
    return data_[x0];
}

f32 Tensor::operator()(u32 x0, u32 x1) const
{
    assert(ndims_ == 2);
    assert(x0 < dims_[0]);
    assert(x1 < dims_[1]);
    return data_[x0 * dims_[1] + x1];
}

f32& Tensor::operator()(u32 x0, u32 x1)
{
    assert(ndims_ == 2);
    assert(x0 < dims_[0]);
    assert(x1 < dims_[1]);
    return data_[x0 * dims_[1] + x1];
}

f32 Tensor::operator()(u32 x0, u32 x1, u32 x2) const
{
    assert(ndims_ == 3);
    assert(x0 < dims_[0]);
    assert(x1 < dims_[1]);
    assert(x2 < dims_[2]);
    return data_[x0 * dims_[1] * dims_[2] + x1 * dims_[2] + x2];
}

f32& Tensor::operator()(u32 x0, u32 x1, u32 x2)
{
    assert(ndims_ == 3);
    assert(x0 < dims_[0]);
    assert(x1 < dims_[1]);
    assert(x2 < dims_[2]);
    return data_[x0 * dims_[1] * dims_[2] + x1 * dims_[2] + x2];
}

f32 Tensor::operator()(u32 x0, u32 x1, u32 x2, u32 x3) const
{
    assert(ndims_ == 4);
    assert(x0 < dims_[0]);
    assert(x1 < dims_[1]);
    assert(x2 < dims_[2]);
    assert(x3 < dims_[3]);
    return data_[x0 * dims_[1] * dims_[2] * dims_[3] + x1 * dims_[2] * dims_[3] + x2 * dims_[2] + x3];
}

f32& Tensor::operator()(u32 x0, u32 x1, u32 x2, u32 x3)
{
    assert(ndims_ == 4);
    assert(x0 < dims_[0]);
    assert(x1 < dims_[1]);
    assert(x2 < dims_[2]);
    assert(x3 < dims_[3]);
    return data_[x0 * dims_[1] * dims_[2] * dims_[3] + x1 * dims_[2] * dims_[3] + x2 * dims_[2] + x3];
}

const f32* Tensor::begin1() const
{
    return data_;
}

f32* Tensor::begin1()
{
    return data_;
}

const f32* Tensor::begin2(u32 x0) const
{
    return data_ + x0 * dims_[1];
}

f32* Tensor::begin2(u32 x0)
{
    return data_ + x0 * dims_[1];
}

const f32* Tensor::begin3(u32 x0, u32 x1) const
{
    return data_ + x0 * dims_[1] * dims_[2] + x1 * dims_[2];
}

f32* Tensor::begin3(u32 x0, u32 x1)
{
    return data_ + x0 * dims_[1] * dims_[2] + x1 * dims_[2];
}

const f32* Tensor::begin4(u32 x0, u32 x1, u32 x2) const
{
    return data_ + x0 * dims_[1] * dims_[2] * dims_[3] + x1 * dims_[2] * dims_[3] + x2 * dims_[3];
}

f32* Tensor::begin4(u32 x0, u32 x1, u32 x2)
{
    return data_ + x0 * dims_[1] * dims_[2] * dims_[3] + x1 * dims_[2] * dims_[3] + x2 * dims_[3];
}

bool same_shape(const Tensor& x0, const Tensor& x1)
{
    if(x0.ndims() != x1.ndims()) {
        return false;
    }
    for(u32 i = 0; i < x0.ndims(); ++i) {
        if(x0.dim(i) != x1.dim(i)) {
            return false;
        }
    }
    return true;
}

void print1(const Tensor& x)
{
    printf("|");
    for(u32 i = 0; i < x.dim(0); ++i) {
        printf("%f, ", x(i));
    }
    printf("|\n");
}

void print2(const Tensor& x)
{
    for(u32 i = 0; i < x.dim(0); ++i) {
        printf("[%d] |", i);
        for(u32 j = 0; j < x.dim(1); ++j) {
            printf("%f, ", x(i, j));
        }
        printf("|\n");
    }
}

void print3(const Tensor& x)
{
    for(u32 i = 0; i < x.dim(0); ++i) {
        printf("[%d]\n", i);
        for(u32 j = 0; j < x.dim(1); ++j) {
            printf(" |");
            for(u32 k = 0; k < x.dim(2); ++k) {
                printf("%f, ", x(i,j,k));
            }
            printf("|\n");
        }
    }
}

Tensor mul(const Tensor& input, const Tensor& weight)
{
    assert(input.dim(1) == weight.dim(1));
    Tensor result({input.dim(0), weight.dim(0)});
    for(u32 b = 0; b < input.dim(0); ++b) {
        const f32* in = input.begin2(b);
        f32* ret = result.begin2(b);
        for(u32 i = 0; i < weight.dim(0); ++i) {
            const f32* w = weight.begin2(i);
            f32 r = 0.0f;
            for(u32 j = 0; j < input.dim(0); ++j) {
                r += in[j] * w[j];
            }
            ret[i] = r;
        }
    }
    return result;
}

Tensor mul(const Tensor& input, const Tensor& weight, const Tensor& bias)
{
    assert(input.dim(1) == weight.dim(1));
    assert(weight.dim(0) == bias.dim(0));
    Tensor result({input.dim(0), weight.dim(0)});
    for(u32 b = 0; b < input.dim(0); ++b) {
        const f32* in = input.begin2(b);
        f32* ret = result.begin2(b);
        for(u32 i = 0; i < weight.dim(0); ++i) {
            const f32* w = weight.begin2(i);
            f32 r = 0.0f;
            for(u32 j = 0; j < input.dim(0); ++j) {
                r += in[j] * w[j];
            }
            ret[i] = r + bias(i);
        }
    }
    return result;
}

void step(Tensor& x)
{
    u32 total = x.total();
    for(u32 i = 0; i < total; ++i) {
        x[i] = 0.0f < x[i] ? 1.0f : 0.0f;
    }
}

void sigmoid(Tensor& x)
{
    u32 total = x.total();
    for(u32 i = 0; i < total; ++i) {
        x[i] = 1.0f / (1.0f + std::expf(-x[i]));
    }
}

void ReLU(Tensor& x)
{
    u32 total = x.total();
    for(u32 i = 0; i < total; ++i) {
        x[i] = 0.0f < x[i] ? x[i] : 0.0f;
    }
}

void softmax(Tensor& x)
{
    assert(2 == x.ndims());
    assert(0 < x.dim(0));
    assert(0 < x.dim(1));
    for(u32 b = 0; b < x.dim(0); ++b) {
        f32 c = x(b, 0);
        for(u32 i = 1; i < x.dim(1); ++i) {
            c = std::max(x(b, i), c);
        }

        f32 sum = 0.0f;
        f32 c0 = 0.0f;
        for(u32 i = 0; i < x.dim(1); ++i) {
            f32 y = std::expf(x(b, i) - c) - c0;
            f32 t = sum + y;
            c0 = t - sum - y;
            sum = t;
        }
        f32 isum = (1.0e-7f <= std::abs(sum)) ? 1.0f / sum : 1.0f;
        for(u32 i = 0; i < x.dim(1); ++i) {
            x(b, i) = std::expf(x(b, i) - c) * isum;
        }
    }
}

f32 mean_squared_error(const Tensor& y, const Tensor& t)
{
    assert(2 == y.ndims());
    assert(2 == t.ndims());
    f32 total = 0.0f;
    f32 c0 = 0.0f;
    for(u32 i = 0; i < y.dim(0); ++i) {
        const f32* py = y.begin2(i);
        const f32* pt = t.begin2(i);
        f32 sum = 0.0f;
        f32 c = 0.0f;
        for(u32 j = 0; j < y.dim(1); ++j) {
            f32 d = py[j] - pt[j];
            f32 ty = d * d - c;
            f32 tt = sum + ty;
            c = tt - sum - ty;
            sum = tt;
        }
        {
            f32 ty = sum - c0;
            f32 tt = total + ty;
            c0 = tt - total - ty;
            total = tt;
        }
    }
    return total * 0.5f;
}

f32 cross_entropy_error(const Tensor& y, const Tensor& t)
{
    assert(2 == y.ndims());
    assert(2 == t.ndims());
    static constexpr float delta = 1.0e-7f;
    f32 total = 0.0f;
    f32 c0 = 0.0f;
    for(u32 i = 0; i < y.dim(0); ++i) {
        const f32* py = y.begin2(i);
        const f32* pt = t.begin2(i);
        f32 sum = 0.0f;
        f32 c = 0.0f;
        for(u32 j = 0; j < y.dim(1); ++j) {
            f32 ty = pt[j] * std::logf(py[j] + delta) - c;
            f32 tt = sum + ty;
            c = tt - sum - ty;
            sum = tt;
        }
        {
            f32 ty = sum - c0;
            f32 tt = total + ty;
            c0 = tt - total - ty;
            total = tt;
        }
    }
    return -total;
}

Tensor back_sigmoid(const Tensor& x, const Tensor& last)
{
    Tensor result = x;
    for(u32 i = 0; i < x.dim(0); ++i) {
        for(u32 j = 0; j < x.dim(1); ++j) {
            if(last.begin2(i)[j] < 0.0f) {
                result.begin2(i)[j] = 0.0f;
            }
        }
    }
    return result;
}

Tensor back_ReLU(const Tensor& x, const Tensor& last)
{
    Tensor result({x.dim(0), x.dim(1)});
    for(u32 i = 0; i < x.dim(0); ++i) {
        for(u32 j = 0; j < x.dim(1); ++j) {
            f32 t = last.begin2(i)[j];
            result.begin2(i)[j] = x.begin2(i)[j] * (1.0f - t) * t;
        }
    }
    return result;
}

Tensor back_softmax(const Tensor& x)
{
    return x;
}

//--- TensorT
//-----------------------------------------------------
TensorT::TensorT()
    : tensor_(nullptr)
    ,dims_{}
    ,cache_{}
{
}

TensorT::TensorT(const Tensor* tensor)
    :tensor_(tensor)
{
    assert(nullptr != tensor_);
    assert(1 < tensor_->ndims() && tensor_->ndims() < 4);
    dims_[0] = tensor_->dims_[0];
    if(2 == tensor_->ndims()) {
        dims_[1] = tensor_->dims_[1];
    } else if(3 == tensor_->ndims()) {
        dims_[2] = tensor_->dims_[2];
        dims_[1] = tensor_->dims_[1];
    }
    dims_[3] = 0;
    cache_[0] = dims_[1]*dims_[2];
    cache_[1] = dims_[1]*dims_[2]*dims_[3];
    cache_[2] = dims_[2]*dims_[3];
}

TensorT::TensorT(TensorT&& other)
    : tensor_(other.tensor_)
{
    ::memcpy(dims_, other.dims_, sizeof(u32) * 4);
    ::memcpy(cache_, other.cache_, sizeof(u32) * 3);

    other.tensor_ = nullptr;
    ::memset(other.dims_, 0, sizeof(u32) * 4);
    ::memset(other.cache_, 0, sizeof(u32) * 3);
}

TensorT& TensorT::operator=(TensorT&& other)
{
    if(this != &other) {
        tensor_ = other.tensor_;
        ::memcpy(dims_, other.dims_, sizeof(u32) * 4);
        ::memcpy(cache_, other.cache_, sizeof(u32) * 3);

        other.tensor_ = nullptr;
        ::memset(other.dims_, 0, sizeof(u32) * 4);
        ::memset(other.cache_, 0, sizeof(u32) * 3);
    }
    return *this;
}

TensorT::TensorT(const TensorT& other)
    : tensor_(other.tensor_)
{
    ::memcpy(dims_, other.dims_, sizeof(u32) * 4);
    ::memcpy(cache_, other.cache_, sizeof(u32) * 3);
}

TensorT& TensorT::operator=(const TensorT& other)
{
    if(this != &other) {
        tensor_ = other.tensor_;
        ::memcpy(dims_, other.dims_, sizeof(u32) * 4);
        ::memcpy(cache_, other.cache_, sizeof(u32) * 3);
    }
    return *this;
}

TensorT::~TensorT()
{
    tensor_ = nullptr;
}

u32 TensorT::ndims() const
{
    return tensor_->ndims_;
}

u32 TensorT::dim(u32 d) const
{
    return dims_[d];
}

u32 TensorT::total() const
{
    u32 p = dims_[0];
    for(u32 i = 1; i < tensor_->ndims_; ++i) {
        p *= dims_[i];
    }
    return p;
}

f32 TensorT::operator[](u32 x0) const
{
    return tensor_->data_[x0];
}

f32 TensorT::operator()(u32 x0) const
{
    assert(tensor_->ndims() == 1);
    assert(x0 < dims_[0]);
    return tensor_->data_[x0];
}

f32 TensorT::operator()(u32 x0, u32 x1) const
{
    assert(tensor_->ndims() == 2);
    assert(x0 < dims_[0]);
    assert(x1 < dims_[1]);
    return tensor_->data_[x0 * dims_[1] + x1];
}

f32 TensorT::operator()(u32 x0, u32 x1, u32 x2) const
{
    assert(tensor_->ndims() == 3);
    assert(x0 < dims_[0]);
    assert(x1 < dims_[1]);
    assert(x2 < dims_[2]);
    return tensor_->data_[x0 * cache_[0] + x1 * dims_[2] + x2];
}

f32 TensorT::operator()(u32 x0, u32 x1, u32 x2, u32 x3) const
{
    assert(tensor_->ndims() == 4);
    assert(x0 < dims_[0]);
    assert(x1 < dims_[1]);
    assert(x2 < dims_[2]);
    assert(x3 < dims_[3]);
    return tensor_->data_[x0 * cache_[1] + x1 * cache_[2] + x2 * dims_[2] + x3];
}

void print1(const TensorT& x)
{
    printf("|");
    for(u32 i = 0; i < x.dim(0); ++i) {
        printf("%f, ", x(i));
    }
    printf("|\n");
}

void print2(const TensorT& x)
{
    for(u32 i = 0; i < x.dim(0); ++i) {
        printf("[%d] |", i);
        for(u32 j = 0; j < x.dim(1); ++j) {
            printf("%f, ", x(i, j));
        }
        printf("|\n");
    }
}

void print3(const TensorT& x)
{
    for(u32 i = 0; i < x.dim(0); ++i) {
        printf("[%d]\n", i);
        for(u32 j = 0; j < x.dim(1); ++j) {
            printf(" |");
            for(u32 k = 0; k < x.dim(2); ++k) {
                printf("%f, ", x(i,j,k));
            }
            printf("|\n");
        }
    }
}

Tensor mul(const TensorT& x0, const TensorT& x1)
{
    assert(x0.dim(1) == x1.dim(1));
    Tensor result({x0.dim(0), x1.dim(0)});
    for(u32 b = 0; b < x0.dim(0); ++b) {
        for(u32 i = 0; i < x1.dim(0); ++i) {
            f32 r = 0.0f;
            for(u32 j = 0; j < x0.dim(0); ++j) {
                r += x0(b,j) * x1(i,j);
            }
            result(b,i) = r;
        }
    }
    return result;
}

Tensor mul(const Tensor& x0, const TensorT& x1)
{
    assert(x0.dim(1) == x1.dim(1));
    Tensor result({x0.dim(0), x1.dim(0)});
    for(u32 b = 0; b < x0.dim(0); ++b) {
        for(u32 i = 0; i < x1.dim(0); ++i) {
            f32 r = 0.0f;
            for(u32 j = 0; j < x0.dim(0); ++j) {
                r += x0(b,j) * x1(i,j);
            }
            result(b,i) = r;
        }
    }
    return result;
}

Tensor mul(const TensorT& x0, const Tensor& x1)
{
    assert(x0.dim(1) == x1.dim(1));
    Tensor result({x0.dim(0), x1.dim(0)});
    for(u32 b = 0; b < x0.dim(0); ++b) {
        for(u32 i = 0; i < x1.dim(0); ++i) {
            f32 r = 0.0f;
            for(u32 j = 0; j < x0.dim(0); ++j) {
                r += x0(b,j) * x1(i,j);
            }
            result(b,i) = r;
        }
    }
    return result;
}

//--- Relu
//-----------------------------------------------------
Relu::Relu()
{
}

Relu::~Relu()
{
}

Tensor Relu::forward(const Tensor& x)
{
    if(!same_shape(mask_, x)) {
        mask_.reshape(x.ndims(), x.dims());
    }
    u32 total = x.total();
    for(u32 i = 0; i < total; ++i) {
        mask_.begin1()[i] = x.begin1()[i] <= 0.0;
    }
    Tensor r;
    r.reshape(x.ndims(), x.dims());
    for(u32 i = 0; i < total; ++i) {
        r.begin1()[i] = mask_.begin1()[i] ? 0.0f : x.begin1()[i];
    }
    return r;
}

Tensor Relu::backward(const Tensor& x)
{
    u32 total = x.total();
    Tensor r;
    r.reshape(x.ndims(), x.dims());
    for(u32 i = 0; i < total; ++i) {
        r.begin1()[i] = mask_.begin1()[i] ? 0.0f : x.begin1()[i];
    }
    return r;
}

//--- Sigmoid
Sigmoid::Sigmoid()
{
}

Sigmoid::~Sigmoid()
{
}

Tensor Sigmoid::forward(const Tensor& x)
{
    u32 total = x.total();
    Tensor r;
    r.reshape(x.ndims(), x.dims());
    for(u32 i = 0; i < total; ++i) {
        r.begin1()[i] = 1.0f / (1.0f + std::expf(-x.begin1()[i]));
    }
    r_ = r;
    return r;
}

Tensor Sigmoid::backward(const Tensor& x)
{
    u32 total = x.total();
    Tensor r;
    r.reshape(x.ndims(), x.dims());
    for(u32 i = 0; i < total; ++i) {
        r.begin1()[i] = x.begin1()[i] * (1.0f - r_.begin1()[i]) * r_.begin1()[i];
    }
    return r;
}

//--- Affine
//-----------------------------------------------------
Affine::Affine(std::initializer_list<u32> dims)
    : weight_(dims)
{
    bias_ = std::move(Tensor({weight_.dim(0)}));
}

Affine::Affine(Affine&& other)
    : weight_(std::move(other.weight_))
{
}

Affine& Affine::operator=(Affine&& other)
{
    if(this != &other) {
        weight_ = std::move(other.weight_);
    }
    return *this;
}

Affine::~Affine()
{
}

u32 Affine::ndims() const
{
    return weight_.ndims();
}

u32 Affine::dim(u32 d) const
{
    return weight_.dim(d);
}

Tensor Affine::forward(const Tensor& x)
{
    assert(2 == x.ndims());
    assert(x.dim(1) == weight_.dim(1));
    Tensor r;
    if(bias_.ndims() <= 0) {
        r = mul(x, weight_);
    } else {
        r = mul(x, weight_, bias_);
    }
    //x_ = x;
    return r;
}

Tensor Affine::backward(const Tensor& x)
{
    assert(2 == x.ndims());
    TensorT wt(&weight_);
    assert(x.dim(1) == wt.dim(1));
    Tensor r = mul(x, wt);
    return r;
}

f32 Affine::weight(u32 x0, u32 x1) const
{
    assert(x0 < weight_.dim(0));
    assert(x1 < weight_.dim(1));
    return weight_.begin2(x0)[x1];
}

f32& Affine::weight(u32 x0, u32 x1)
{
    assert(x0 < weight_.dim(0));
    assert(x1 < weight_.dim(1));
    return weight_.begin2(x0)[x1];
}

f32 Affine::bias(u32 x0) const
{
    assert(x0 < bias_.dim(0));
    return bias_[x0];
}

f32& Affine::bias(u32 x0)
{
    assert(x0 < bias_.dim(0));
    return bias_[x0];
}

//--- Softmax
//-----------------------------------------------------
Softmax::Softmax()
{
}

Softmax::~Softmax()
{
}

Tensor Softmax::forward(const Tensor& x)
{
    y_ = x;
    softmax(y_);
    return y_;
}

Tensor Softmax::backward(const Tensor& t)
{
    u32 batch_size = t.dim(0);
    Tensor dx;
    dx.reshape(t.ndims(), t.dims());
    u32 total = t.total();
    f32 ibatch_size = 1.0f / batch_size;
    for(u32 i = 0; i < total; ++i) {
        dx.begin1()[i] = (y_.begin1()[i] - t.begin1()[i]) * ibatch_size;
    }
    return dx;
}

//--- Loss
//-----------------------------------------------------
Loss::Loss()
{
}

Loss::~Loss()
{
}

Tensor Loss::forward(const Tensor& x)
{
    y_ = x;
    softmax(y_);
    return y_;
}

Tensor Loss::backward(const Tensor& t)
{
    u32 batch_size = t.dim(0);
    Tensor dx;
    dx.reshape(t.ndims(), t.dims());
    u32 total = t.total();
    f32 ibatch_size = 1.0f / batch_size;
    for(u32 i = 0; i < total; ++i) {
        dx.begin1()[i] = (y_.begin1()[i] - t.begin1()[i]) * ibatch_size;
    }
    return dx;
}

//--- Layer
//-----------------------------------------------------
Layer::Layer(Activation activation, bool bias, bool train, std::initializer_list<u32> dims)
    : activation_(activation)
    , train_(train)
    , weight_(dims)
{
    if(bias) {
        bias_ = std::move(Tensor({weight_.dim(0)}));
    }
}

void Layer::set_random()
{
    f32 sigma = 0.05f;
    switch(activation_) {
    case Activation::Sigmoid:
        sigma = 1.0f / std::sqrtf(static_cast<f32>(weight_.dim(1)));
        break;
    case Activation::ReLU:
        sigma = std::sqrtf(2.0f / static_cast<f32>(weight_.dim(1)));
        break;
    case Activation::Softmax:
        sigma = 1.0f / std::sqrtf(static_cast<f32>(weight_.dim(1)));
        break;
    }
    random(weight_, sigma, System::getInstance().getRand());
    if(0 < bias_.ndims()) {
        random(bias_, sigma, System::getInstance().getRand());
    }
}

Layer::Layer(Layer&& other)
    : activation_(other.activation_)
    , train_(other.train_)
    , weight_(std::move(other.weight_))
{
}

Layer& Layer::operator=(Layer&& other)
{
    if(this != &other) {
        activation_ = other.activation_;
        train_ = other.train_;
        weight_ = std::move(other.weight_);
    }
    return *this;
}

Layer::~Layer()
{
}

u32 Layer::ndims() const
{
    return weight_.ndims();
}

u32 Layer::dim(u32 d) const
{
    return weight_.dim(d);
}

Tensor Layer::forward(const Tensor& input)
{
    assert(2 == input.ndims());
    assert(input.dim(1) == weight_.dim(1));
    Tensor result;
    if(bias_.ndims() <= 0) {
        result = mul(input, weight_);
    } else {
        result = mul(input, weight_, bias_);
    }
    switch(activation_) {
    case Activation::Sigmoid:
        sigmoid(result);
        if(train_) {
            last_input_ = result;
        }
        break;
    case Activation::ReLU:
        if(train_) {
            last_input_ = input;
        }
        ReLU(result);
        break;
    case Activation::Softmax:
        softmax(result);
        break;
    }
    return result;
}

f32 Layer::weight(u32 x0, u32 x1) const
{
    assert(x0 < weight_.dim(0));
    assert(x1 < weight_.dim(1));
    return weight_.begin2(x0)[x1];
}

f32& Layer::weight(u32 x0, u32 x1)
{
    assert(x0 < weight_.dim(0));
    assert(x1 < weight_.dim(1));
    return weight_.begin2(x0)[x1];
}

f32 Layer::bias(u32 x0) const
{
    assert(x0 < bias_.dim(0));
    return bias_[x0];
}

f32& Layer::bias(u32 x0)
{
    assert(x0 < bias_.dim(0));
    return bias_[x0];
}

void Layer::set_train(bool train)
{
    train_ = train;
}

void BaseLogger::write(s32 /*step*/, f32 /*loss*/, const char* format, ...)
{
    assert(nullptr != format);
    va_list ap;
    va_start(ap, format);
#ifdef _MSC_VER
    vfprintf_s(stdout, format, ap);
#else
    vfprintf(stdout, format, ap);
#endif
    va_end(ap);
}

//--- Model
//-----------------------------------------------------
Model::Model()
    : capacity_(0)
    , size_(0)
    , layers_(nullptr)
{
}

Model::~Model()
{
    for(u32 i = 0; i < size_; ++i) {
        layers_[i]->~ILayer();
        PERC_FREE(layers_[i]);
    }
    PERC_FREE(layers_);
    capacity_ = 0;
    size_ = 0;
    layers_ = nullptr;
}

u32 Model::size() const
{
    return size_;
}

u32 Model::input_size() const
{
    assert(0 < size_);
    return layers_[0]->dim(1);
}

u32 Model::output_size() const
{
    assert(0 < size_);
    return layers_[size_ - 1]->dim(0);
}

const ILayer& Model::operator[](u32 index) const
{
    assert(index < size_);
    return *layers_[index];
}

ILayer& Model::operator[](u32 index)
{
    assert(index < size_);
    return *layers_[index];
}

void Model::add(ILayer* layer)
{
    assert(nullptr != layer);
    if(capacity_ <= size_) {
        u32 capacity = capacity_ + 16;
        u8* layers = static_cast<u8*>(PERC_MALLOC(sizeof(ILayer*) * capacity));
        ::memcpy(layers, layers_, sizeof(ILayer*)*capacity_);
        PERC_FREE(layers_);
        layers_ = reinterpret_cast<ILayer**>(layers);
        capacity_ = capacity;
    }
    assert(size_<capacity_);
    layers_[size_] = layer;
    ++size_;
}

Tensor Model::predict(const Tensor& input)
{
    assert(0 < size_);
    Tensor result = layers_[0]->forward(input);
    for(u32 i = 1; i < size_; ++i) {
        result = layers_[i]->forward(result);
    }
    return result;
}

f32 Model::loss(const Tensor& input0, const Tensor& input1)
{
    Tensor t = predict(input0);
    return mean_squared_error(input1, t);
}

} // namespace perceptron

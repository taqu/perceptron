#include "perceptron.h"
#include <cassert>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <numbers>

namespace perceptron
{
namespace
{
    inline uint32_t rotr32(uint32_t x, uint32_t r)
    {
        return (x >> r) | (x << (-r & 31));
    }

    template<class T>
    f32 normal_distribution(T& engine)
    {
        f32 w,z,s;
        do{
            w = engine.frand() * 2.0f - 1.0f;
            z = engine.frand() * 2.0f - 1.0f;
            s = w*w + z*z;
        }while(1.0f<=s || 0.0f == s);
        return std::sqrtf(-2.0f*std::logf(s)/s)*w;
    }
} // namespace

//--- RandomPCG32
//-----------------------------------------------------------------
RandomPCG32::RandomPCG32()
{
    srand(1234U);
}

RandomPCG32::RandomPCG32(uint32_t x)
{
    srand(x);
}

RandomPCG32::~RandomPCG32()
{
}

void RandomPCG32::srand(uint32_t x)
{
    uint32_t* state32 = reinterpret_cast<uint32_t*>(&state_);
    state32[0] = x;
    state32[1] = 1812433253UL * (state32[0] ^ (state32[0] >> 30)) + 1;
}

void RandomPCG32::srand(uint64_t x)
{
    state_ = x;
}

uint32_t RandomPCG32::rand()
{
    uint64_t x = state_;
    state_ = state_ * Multiplier + Increment;
    uint32_t count = static_cast<uint32_t>(x >> 59);
    x ^= x >> 18;
    return rotr32(static_cast<uint32_t>(x >> 27), count);
}

uint32_t RandomPCG32::range(uint32_t maxx)
{
    return rand_impl::range(*this, maxx);
}

float RandomPCG32::frand()
{
    return rand_impl::frandom_downey_opt32(*this);
}

//--- RandomPCG32_128
//-----------------------------------------------------------------
RandomPCG32_128::RandomPCG32_128()
{
    srand(1234U, 5678U);
}

RandomPCG32_128::RandomPCG32_128(uint32_t x0, uint32_t x1)
{
    srand(x0, x1);
}

RandomPCG32_128::~RandomPCG32_128()
{
}

void RandomPCG32_128::srand(uint32_t x0, uint32_t x1)
{
    state0_.srand(x0);
    state1_.srand(x1);
    x1_ = state1_.rand();
}

void RandomPCG32_128::srand(uint64_t x0, uint64_t x1)
{
    state0_.srand(x0);
    state1_.srand(x1);
    x1_ = state1_.rand();
}

uint32_t RandomPCG32_128::rand()
{
    uint32_t x = state0_.rand();
    if(0 == x)[[unlikely]]{
        x1_ = state1_.rand();
    }
    return x + x1_;
}

uint32_t RandomPCG32_128::range(uint32_t maxx)
{
    return rand_impl::range(*this, maxx);
}

float RandomPCG32_128::frand()
{
    return rand_impl::frandom_downey_opt32(*this);
}

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
    u64 s0 = (static_cast<u64>(device())<<32) | device();
    u64 s1 = (static_cast<u64>(device())<<32) | device();
    instance_.engine_.srand(s0, s1);
}

void System::terminate()
{
}

RandomPCG32_128& System::getRand()
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

Tensor::Tensor(u32 ndims, const u32 dims[4])
{
    ndims_ = ndims;
    ::memcpy(dims_, dims, sizeof(u32) * 4);

    u32 size = total() * sizeof(f32);
    data_ = static_cast<f32*>(PERC_MALLOC(size));
    ::memset(data_, 0, size);
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
    u32 size = total() * sizeof(f32);
    data_ = static_cast<f32*>(PERC_MALLOC(size));
    ::memset(data_, 0, size);
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
                printf("%f, ", x(i, j, k));
            }
            printf("|\n");
        }
    }
}

Tensor mul(const Tensor& x0, const Tensor& x1)
{
    assert(x0.dim(1) == x1.dim(0));
    Tensor result({x0.dim(0), x1.dim(1)});
    for(u32 b = 0; b < x0.dim(0); ++b) {
        const f32* in = x0.begin2(b);
        f32* ret = result.begin2(b);
        for(u32 i = 0; i < x1.dim(1); ++i) {
            f32 r = 0.0f;
            for(u32 j = 0; j < x1.dim(0); ++j) {
                r += in[j] * x1(j, i);
            }
            ret[i] = r;
        }
    }
    return result;
}

Tensor mul(const Tensor& x0, const Tensor& x1, const Tensor& bias)
{
    assert(x0.dim(1) == x1.dim(0));
    assert(x1.dim(1) == bias.dim(0));
    Tensor result({x0.dim(0), x1.dim(1)});
    for(u32 b = 0; b < x0.dim(0); ++b) {
        const f32* in = x0.begin2(b);
        f32* ret = result.begin2(b);
        for(u32 i = 0; i < x1.dim(1); ++i) {
            f32 r = 0.0f;
            for(u32 j = 0; j < x1.dim(0); ++j) {
                r += in[j] * x1(j, i);
            }
            ret[i] = r + bias(i);
        }
    }
    return result;
}

Tensor mul_transpose(const Tensor& x0, const Tensor& x1)
{
    assert(x0.dim(1) == x1.dim(1));
    Tensor result({x0.dim(0), x1.dim(0)});
    for(u32 b = 0; b < x0.dim(0); ++b) {
        const f32* in = x0.begin2(b);
        f32* ret = result.begin2(b);
        for(u32 i = 0; i < x1.dim(0); ++i) {
            const f32* w = x1.begin2(i);
            f32 r = 0.0f;
            for(u32 j = 0; j < x1.dim(1); ++j) {
                r += in[j] * w[j];
            }
            ret[i] = r;
        }
    }
    return result;
}

Tensor mul_transpose(const Tensor& x0, const Tensor& x1, const Tensor& bias)
{
    assert(x0.dim(1) == x1.dim(1));
    assert(x1.dim(0) == bias.dim(0));
    Tensor result({x0.dim(0), x1.dim(0)});
    for(u32 b = 0; b < x0.dim(0); ++b) {
        const f32* in = x0.begin2(b);
        f32* ret = result.begin2(b);
        for(u32 i = 0; i < x1.dim(0); ++i) {
            const f32* w = x1.begin2(i);
            f32 r = 0.0f;
            for(u32 j = 0; j < x1.dim(1); ++j) {
                r += in[j] * w[j];
            }
            ret[i] = r + bias(i);
        }
    }
    return result;
}

Tensor transpose_mul(const Tensor& x0, const Tensor& x1)
{
    assert(x0.dim(0) == x1.dim(0));
    Tensor result({x0.dim(1), x1.dim(1)});
    for(u32 i = 0; i < x0.dim(1); ++i) {
        f32* ret = result.begin2(i);
        for(u32 j = 0; j < x1.dim(1); ++j) {
            f32 r = 0.0f;
            for(u32 k = 0; k < x1.dim(0); ++k) {
                r += x0(k,i) * x1(k,j);
            }
            ret[j] = r;
        }
    }
    return result;
}

Tensor batch_sum(const Tensor& tensor)
{
    assert(2 == tensor.ndims());
    Tensor result({tensor.dim(1)});
    f32* ret = result.begin1();
    for(u32 i = 0; i < tensor.dim(1); ++i) {
        f32 r = 0.0f;
        for(u32 j = 0; j < tensor.dim(0); ++j) {
            r += tensor(j,i);
        }
        ret[i] = r;
    }
    return result;
}

Tensor batch_sub(const Tensor& x0, const Tensor& x1)
{
    assert(x0.dim(0) == x1.dim(0));
    assert(x0.dim(1) == x1.dim(1));
    Tensor result({x0.dim(0), x0.dim(1)});
    u32 total = x0.total();
    for(u32 i=0; i<total; ++i){
        result[i] = x0[i] - x1[i];
    }
    return result;
}

void batch_mul(Tensor& tensor, f32 x)
{
    u32 total = tensor.total();
    for(u32 i = 0; i < total; ++i) {
        tensor[i] *= x;
    }
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
    assert(y.dim(0) == t.dim(0));
    assert(y.dim(1) == t.dim(1));
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

Tensor numerical_gradient(std::function<f32(const Tensor&,const Tensor&)> f, const Tensor&x, const Tensor& t)
{
    const f32 h = 1.0e-4f;
    Tensor x0 = x;
    Tensor grad(x.ndims(), x.dims());
    u32 total = x.total();
    for(u32 i=0; i<total; ++i){
        f32 tmp = x0[i];
        x0[i] = tmp + h;
        f32 fxh0 = f(x0, t);

        x0[i] = tmp - h;
        f32 fxh1 = f(x0, t);

        grad[i] = (fxh0-fxh1)/(2*h);
    }
    return grad;
}

//--- TensorMask
//-----------------------------------------------------
TensorMask::TensorMask()
    : ndims_(0)
    , dims_{}
    , data_(nullptr)
{
}

TensorMask::TensorMask(std::initializer_list<u32> dims)
    : ndims_(static_cast<u32>(dims.size()))
    , dims_{}
{
    u32 count = 0;
    u32 total = 1;
    for(u32 d: dims) {
        dims_[count] = d;
        total *= d;
        ++count;
    }
    u32 num = ((total + 31UL) & ~31UL) >> 5;
    data_ = (u32*)PERC_MALLOC(sizeof(u32) * num);
    ::memset(data_, 0, sizeof(u32) * num);
}

TensorMask::TensorMask(TensorMask&& other)
    : ndims_(other.ndims_)
    , data_(other.data_)
{
    ::memcpy(dims_, other.dims_, sizeof(u32) * 4);
    other.ndims_ = 0;
    ::memset(other.dims_, 0, sizeof(u32) * 4);
    other.data_ = nullptr;
}

TensorMask& TensorMask::operator=(TensorMask&& other)
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

TensorMask::TensorMask(const TensorMask& other)
    : ndims_(other.ndims_)
{
    ::memcpy(dims_, other.dims_, sizeof(u32) * 4);
    u32 num = ((other.total() + 31UL) & ~31UL) >> 5;
    u32 size = sizeof(u32) * num;
    data_ = static_cast<u32*>(PERC_MALLOC(size));
    ::memcpy(data_, other.data_, size);
}

TensorMask& TensorMask::operator=(const TensorMask& other)
{
    if(this != &other) {
        PERC_FREE(data_);
        ndims_ = other.ndims_;
        ::memcpy(dims_, other.dims_, sizeof(u32) * 4);
        u32 num = ((other.total() + 31UL) & ~31UL) >> 5;
        u32 size = sizeof(u32) * num;
        data_ = static_cast<u32*>(PERC_MALLOC(size));
        ::memcpy(data_, other.data_, size);
    }
    return *this;
}

TensorMask::~TensorMask()
{
    PERC_FREE(data_);
    data_ = nullptr;
}

void TensorMask::reshape(u32 ndims, const u32 dims[4])
{
    ndims_ = ndims;
    ::memcpy(dims_, dims, sizeof(u32) * 4);

    PERC_FREE(data_);
    data_ = nullptr;
    u32 num = ((total() + 31UL) & ~31UL) >> 5;
    u32 size = num * sizeof(u32);
    data_ = static_cast<u32*>(PERC_MALLOC(size));
    ::memset(data_, 0, size);
}

u32 TensorMask::ndims() const
{
    return ndims_;
}

u32 TensorMask::dim(u32 d) const
{
    return dims_[d];
}

const u32* TensorMask::dims() const
{
    return dims_;
}

u32 TensorMask::total() const
{
    u32 p = dims_[0];
    for(u32 i = 1; i < ndims_; ++i) {
        p *= dims_[i];
    }
    return p;
}

bool TensorMask::operator[](u32 x0) const
{
    u32 i = x0 >> 5;
    u32 b = x0 - (i << 5);
    return 0 != ((data_[i] >> b) & 0x01UL);
}

bool TensorMask::operator()(u32 x0) const
{
    assert(ndims_ == 1);
    assert(x0 < dims_[0]);
    u32 i = x0 >> 5;
    u32 b = x0 - (i << 5);
    return 0 != ((data_[i] >> b) & 0x01UL);
}

bool TensorMask::operator()(u32 x0, u32 x1) const
{
    assert(ndims_ == 2);
    assert(x0 < dims_[0]);
    assert(x1 < dims_[1]);
    u32 index = x0 * dims_[1] + x1;
    u32 i = index >> 5;
    u32 b = index - (i << 5);
    return 0 != ((data_[i] >> b) & 0x01UL);
}

bool TensorMask::operator()(u32 x0, u32 x1, u32 x2) const
{
    assert(ndims_ == 3);
    assert(x0 < dims_[0]);
    assert(x1 < dims_[1]);
    assert(x2 < dims_[2]);
    u32 index = x0 * dims_[1] * dims_[2] + x1 * dims_[2] + x2;
    u32 i = index >> 5;
    u32 b = index - (i << 5);
    return 0 != ((data_[i] >> b) & 0x01UL);
}

bool TensorMask::operator()(u32 x0, u32 x1, u32 x2, u32 x3) const
{
    assert(ndims_ == 4);
    assert(x0 < dims_[0]);
    assert(x1 < dims_[1]);
    assert(x2 < dims_[2]);
    assert(x3 < dims_[3]);
    u32 index = x0 * dims_[1] * dims_[2] * dims_[3] + x1 * dims_[2] * dims_[3] + x2 * dims_[2] + x3;
    u32 i = index >> 5;
    u32 b = index - (i << 5);
    return 0 != ((data_[i] >> b) & 0x01UL);
}

void TensorMask::set_linear(u32 x, bool b)
{
    u32 i = x >> 5;
    u32 shift = x - (i << 5);
    if(b) {
        data_[i] |= (0x01UL << shift);
    } else {
        data_[i] &= ~(0x01UL << shift);
    }
}

void TensorMask::set_linear(u32 x)
{
    u32 i = x >> 5;
    u32 b = x - (i << 5);
    data_[i] |= (0x01UL << b);
}

void TensorMask::reset_linear(u32 x)
{
    u32 i = x >> 5;
    u32 b = x - (i << 5);
    data_[i] &= ~(0x01UL << b);
}

bool same_shape(const TensorMask& x0, const TensorMask& x1)
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

bool same_shape(const TensorMask& x0, const Tensor& x1)
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

bool same_shape(const Tensor& x0, const TensorMask& x1)
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

//--- Relu
//-----------------------------------------------------
Relu* Relu::create()
{
    void* mem = PERC_MALLOC(sizeof(Relu));
    return new(mem) Relu();
}

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
        mask_.set_linear(i, x.begin1()[i] <= 0.0);
    }
    Tensor r(x.ndims(), x.dims());
    for(u32 i = 0; i < total; ++i) {
        r.begin1()[i] = mask_[i] ? 0.0f : x.begin1()[i];
    }
    return r;
}

Tensor Relu::backward(const Tensor& x)
{
    u32 total = x.total();
    Tensor r;
    r.reshape(x.ndims(), x.dims());
    for(u32 i = 0; i < total; ++i) {
        r.begin1()[i] = mask_[i] ? 0.0f : x.begin1()[i];
    }
    return r;
}

void print_mask(const Relu& x)
{
    switch(x.mask_.ndims()) {
    case 1:
        print_mask1(x.mask_);
        break;
    case 2:
        print_mask2(x.mask_);
        break;
    }
}

void print_mask1(const TensorMask& x)
{
    printf("|");
    for(u32 i = 0; i < x.dim(0); ++i) {
        printf("%d, ", x(i));
    }
    printf("|\n");
}

void print_mask2(const TensorMask& x)
{
    for(u32 i = 0; i < x.dim(0); ++i) {
        printf("[%d] |", i);
        for(u32 j = 0; j < x.dim(1); ++j) {
            printf("%d, ", x(i, j));
        }
        printf("|\n");
    }
}

//--- Sigmoid
//-----------------------------------------------------
Sigmoid* Sigmoid::create()
{
    void* mem = PERC_MALLOC(sizeof(Sigmoid));
    return new(mem) Sigmoid();
}

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

void print_out(const Sigmoid& x)
{
    switch(x.r_.ndims()) {
    case 1:
        print_out1(x.r_);
        break;
    case 2:
        print_out2(x.r_);
        break;
    }
}

void print_out1(const Tensor& x)
{
    printf("|");
    for(u32 i = 0; i < x.dim(0); ++i) {
        printf("%f, ", x(i));
    }
    printf("|\n");
}

void print_out2(const Tensor& x)
{
    for(u32 i = 0; i < x.dim(0); ++i) {
        printf("[%d] |", i);
        for(u32 j = 0; j < x.dim(1); ++j) {
            printf("%f, ", x(i, j));
        }
        printf("|\n");
    }
}

//--- Affine
//-----------------------------------------------------
Affine* Affine::create(std::initializer_list<u32> dims, bool bias, bool train)
{
    void* mem = PERC_MALLOC(sizeof(Affine));
    return new(mem) Affine(dims, bias, train);
}

Affine::Affine(std::initializer_list<u32> dims, bool bias, bool train)
    : weight_(dims)
    , train_(train)
{
    if(bias) {
        bias_ = std::move(Tensor({weight_.dim(1)}));
    }
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
    assert(x.dim(1) == weight_.dim(0));
    Tensor r;
    if(bias_.ndims() <= 0) {
        r = mul(x, weight_);
    } else {
        r = mul(x, weight_, bias_);
    }
    if(train_) {
        x_ = x;
    }
    return r;
}

Tensor Affine::backward(const Tensor& x)
{
    assert(2 == x.ndims());
    Tensor dx = mul_transpose(x, weight_);
    dw_ = transpose_mul(x_, x);
    db_ = batch_sum(x);
    return dx;
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

const Tensor& Affine::w() const
{
    return weight_;
}

const Tensor& Affine::b() const
{
    return bias_;
}

const Tensor& Affine::dw() const
{
    return dw_;
}

const Tensor& Affine::db() const
{
    return db_;
}

void Affine::numerical_gradient(std::function<f32()> f)
{
    const f32 h = 1.0e-4f;
    dw_.reshape(weight_.ndims(), weight_.dims());
    u32 total = weight_.total();
    for(u32 i=0; i<total; ++i){
        f32 tmp = weight_[i];
        weight_[i] = tmp + h;
        f32 fxh0 = f();

        weight_[i] = tmp - h;
        f32 fxh1 = f();

        dw_[i] = (fxh0-fxh1)/(2*h);
        weight_[i] = tmp;
    }

    db_.reshape(bias_.ndims(), bias_.dims());
    total = bias_.total();
    for(u32 i=0; i<total; ++i){
        f32 tmp = bias_[i];
        bias_[i] = tmp + h;
        f32 fxh0 = f();

        bias_[i] = tmp - h;
        f32 fxh1 = f();

        db_[i] = (fxh0-fxh1)/(2*h);
        bias_[i] = tmp;
    }
}

void Affine::update(f32 learning_rate)
{
    u32 total;
    total = weight_.total();
    for(u32 i = 0; i < total; ++i) {
        weight_[i] -= learning_rate * dw_[i];
    }

    total = bias_.total();
    for(u32 i = 0; i < total; ++i) {
        bias_[i] -= learning_rate * db_[i];
    }
}

void random(Affine& x, f32 weight_std)
{
    u32 total;
    total = x.weight_.total();
    auto&& engine = System::getInstance().getRand();
    for(u32 i=0; i<total; ++i){
        x.weight_[i] = (std::abs)(normal_distribution(engine) * weight_std);
    }

    total = x.bias_.total();
    if(0<total){
        ::memset(&x.bias_[0], 0, sizeof(f32)*total);
    }
}

//--- Softmax
//-----------------------------------------------------
Softmax* Softmax::create()
{
    void* mem = PERC_MALLOC(sizeof(Softmax));
    return new(mem) Softmax();
}

Softmax::Softmax()
{
}

Softmax::~Softmax()
{
}

Tensor Softmax::forward(const Tensor& x)
{
    x_ = x;
    softmax(x_);
    return x_;
}

Tensor Softmax::backward(const Tensor& x)
{
    f32 inv_batch_size = 1.0f/x.dim(1);
    Tensor r = batch_sub(x_, x);
    batch_mul(r, inv_batch_size);
    return r;
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
        ::memcpy(layers, layers_, sizeof(ILayer*) * capacity_);
        PERC_FREE(layers_);
        layers_ = reinterpret_cast<ILayer**>(layers);
        capacity_ = capacity;
    }
    assert(size_ < capacity_);
    layers_[size_] = layer;
    ++size_;
}

Tensor Model::predict(const Tensor& x)
{
    assert(0 < size_);
    Tensor r = layers_[0]->forward(x);
    u32 size = (1<size_)? size_-1 : 0;
    for(u32 i = 1; i < size; ++i) {
        r = layers_[i]->forward(r);
    }
    return r;
}

Tensor Model::forward(const Tensor& x)
{
    assert(0 < size_);
    Tensor r = layers_[0]->forward(x);
    for(u32 i = 1; i < size_; ++i) {
        r = layers_[i]->forward(r);
    }
    return r;
}

Tensor Model::backward(const Tensor& t)
{
    assert(0 < size_);
    Tensor r = layers_[size_-1]->backward(t);
    for(u32 i = 1; i < size_; ++i) {
        r = layers_[size_-i-1]->backward(r);
    }
    return r;
}

void Model::update(f32 learningRate)
{
    for(u32 i=0; i<size_; ++i){
        layers_[i]->update(learningRate);
    }
}

f32 Model::loss(const Tensor& x, const Tensor& t)
{
    Tensor r = forward(x);
    return cross_entropy_error(r, t);
}

void Model::gradient(const Tensor& x, const Tensor& t)
{
    forward(x);
    backward(t);
}

void Model::numerical_gradient(const Tensor& x, const Tensor& t)
{
    auto f = [model=this,x,t](){
        return model->loss(x,t);
    };

    for(u32 i=0; i<size_; ++i){
        layers_[i]->numerical_gradient(f);
    }
}

f32 Model::accuracy(const Tensor& x, const Tensor& t, std::function<f32(const Tensor& y,const Tensor& t)> f)
{
    Tensor y = forward(x);
    return f(y,t);
}
} // namespace perceptron

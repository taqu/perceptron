#include <cassert>
#include <cstdint>
#include <mimalloc.h>
#include <random>
#include <type_traits>

namespace perceptron
{
using s8 = int8_t;
using s16 = int16_t;
using s32 = int32_t;

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;

using f32 = float;

#ifndef PERC_MALLOC
#    define PERC_MALLOC(size) ::mi_malloc(size)
#endif

#ifndef PERC_FREE
#    define PERC_FREE(ptr) ::mi_free(ptr)
#endif

//--- System
//-----------------------------------------------------
class System
{
public:
    static System& getInstance();
    static void initialize();
    static void terminate();

    std::mt19937& getRand();

private:
    System(const System&) = delete;
    System& operator=(const System&) = delete;
    System();
    ~System();
    static System instance_;
    std::mt19937 engine_;
};

//--- Array
//-----------------------------------------------------
template<class T>
class Array
{
    static_assert(std::is_trivially_copyable<T>::value == true, "T should be trivially copyable.");

public:
    Array();
    Array(Array&& other);
    Array& operator=(Array&& other);
    ~Array();
    u32 capacity() const;
    u32 size() const;
    const T& operator[](u32 index) const;
    T& operator[](u32 index);
    void push_back(const T& x);

private:
    Array(const Array&) = delete;
    Array& operator=(const Array&) = delete;
    void expand();
    u32 capacity_;
    u32 size_;
    T* items_;
};

template<class T>
Array<T>::Array()
    : capacity_(0)
    , size_(0)
    , items_(nullptr)
{
}

template<class T>
Array<T>::Array(Array&& other)
    : capacity_(other.capacity_)
    , size_(other.size_)
    , items_(other.items_)
{
    other.capacity_ = 0;
    other.size_ = 0;
    other.items_ = nullptr;
}

template<class T>
Array<T>& Array<T>::operator=(Array&& other)
{
    if(this != &other) {
        PERC_FREE(items_);
        capacity_ = other.capacity_;
        size_ = other.size_;
        items_ = other.items_;
        other.capacity_ = 0;
        other.size_ = 0;
        other.items_ = nullptr;
    }
    return *this;
}

template<class T>
Array<T>::~Array()
{
    PERC_FREE(items_);
    capacity_ = 0;
    size_ = 0;
    items_ = nullptr;
}

template<class T>
u32 Array<T>::capacity() const
{
    return capacity_;
}

template<class T>
u32 Array<T>::size() const
{
    return size_;
}

template<class T>
const T& Array<T>::operator[](u32 index) const
{
    assert(index < size_);
    return items_[index];
}

template<class T>
T& Array<T>::operator[](u32 index)
{
    assert(index < size_);
    return items_[index];
}

template<class T>
void Array<T>::push_back(const T& x)
{
    if(capacity_ <= size_) {
        expand();
    }
    items_[size_] = x;
    ++size_;
}

template<class T>
void Array<T>::expand()
{
    u32 capacity = capacity_ + 16;
    T* items = static_cast<T*>(PERC_MALLOC(sizeof(T) * capacity));
    ::memcpy(items, items_, sizeof(T) * capacity_);
    PERC_FREE(items_);
    capacity_ = capacity;
    items_ = items;
}

//--- Tensor
//-----------------------------------------------------
class Tensor
{
public:
    Tensor();
    Tensor(std::initializer_list<u32> dims);
    Tensor(Tensor&& other);
    Tensor& operator=(Tensor&& other);
    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);
    ~Tensor();
    void reshape(u32 ndims, const u32 dims[4]);
    u32 ndims() const;
    u32 dim(u32 d) const;
    const u32* dims() const;
    u32 total() const;
    f32 operator[](u32 x0) const;
    f32& operator[](u32 x0);

    f32 operator()(u32 x0) const;
    f32& operator()(u32 x0);

    f32 operator()(u32 x0, u32 x1) const;
    f32& operator()(u32 x0, u32 x1);

    f32 operator()(u32 x0, u32 x1, u32 x2) const;
    f32& operator()(u32 x0, u32 x1, u32 x2);

    f32 operator()(u32 x0, u32 x1, u32 x2, u32 x3) const;
    f32& operator()(u32 x0, u32 x1, u32 x2, u32 x3);

    const f32* begin1() const;
    f32* begin1();

    const f32* begin2(u32 x0) const;
    f32* begin2(u32 x0);

    const f32* begin3(u32 x0, u32 x1) const;
    f32* begin3(u32 x0, u32 x1);

    const f32* begin4(u32 x0, u32 x1, u32 x2) const;
    f32* begin4(u32 x0, u32 x1, u32 x2);

private:
    friend class TensorT;
    u32 ndims_;
    u32 dims_[4];
    f32* data_;
};

void print1(const Tensor& x);
void print2(const Tensor& x);
void print3(const Tensor& x);

Tensor mul(const Tensor& input, const Tensor& weight);
Tensor mul(const Tensor& input, const Tensor& weight, const Tensor& bias);
void step(Tensor& x);
void sigmoid(Tensor& x);
void ReLU(Tensor& x);
void softmax(Tensor& x);
f32 mean_squared_error(const Tensor& y, const Tensor& t);
f32 cross_entropy_error(const Tensor& y, const Tensor& t);

Tensor back_sigmoid(const Tensor& x, const Tensor& last);
Tensor back_ReLU(const Tensor& x, const Tensor& last);
Tensor back_softmax(const Tensor& x);

template<class T>
void random(Tensor& x, f32 sigma, T& engine)
{
    std::normal_distribution<float> dist(0.0f, sigma);
    u32 total = x.total();
    for(u32 i = 0; i < total; ++i) {
        x[i] = dist(engine);
    }
}

template<class T>
class TTensor
{
public:
    TTensor();
    TTensor(std::initializer_list<u32> dims);
    TTensor(TTensor&& other);
    TTensor& operator=(TTensor&& other);
    TTensor(const TTensor& other);
    TTensor& operator=(const TTensor& other);
    ~TTensor();
    void reshape(u32 ndims, const u32 dims[4]);

    u32 ndims() const;
    u32 dim(u32 d) const;
    const u32* dims() const;
    u32 total() const;
    T operator[](u32 x0) const;
    T& operator[](u32 x0);

    T operator()(u32 x0) const;
    T& operator()(u32 x0);

    T operator()(u32 x0, u32 x1) const;
    T& operator()(u32 x0, u32 x1);

    T operator()(u32 x0, u32 x1, u32 x2) const;
    T& operator()(u32 x0, u32 x1, u32 x2);

    T operator()(u32 x0, u32 x1, u32 x2, u32 x3) const;
    T& operator()(u32 x0, u32 x1, u32 x2, u32 x3);

    const T* begin1() const;
    T* begin1();

    const T* begin2(u32 x0) const;
    T* begin2(u32 x0);

    const T* begin3(u32 x0, u32 x1) const;
    T* begin3(u32 x0, u32 x1);

    const T* begin4(u32 x0, u32 x1, u32 x2) const;
    T* begin4(u32 x0, u32 x1, u32 x2);

private:
    u32 ndims_;
    u32 dims_[4];
    T* data_;
};

template<class T>
TTensor<T>::TTensor()
    : ndims_(0)
    , dims_{}
    , data_(nullptr)
{
}

template<class T>
TTensor<T>::TTensor(std::initializer_list<u32> dims)
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
    data_ = (f32*)PERC_MALLOC(sizeof(T) * total);
    ::memset(data_, 0, sizeof(T) * total);
}

template<class T>
TTensor<T>::TTensor(TTensor&& other)
    : ndims_(other.ndims_)
    , data_(other.data_)
{
    ::memcpy(dims_, other.dims_, sizeof(u32) * 4);

    other.ndims_ = 0;
    other.ndims_ = 0;
    ::memset(other.dims_, 0, sizeof(u32) * 4);
    other.data_ = nullptr;
}

template<class T>
TTensor<T>& TTensor<T>::operator=(TTensor&& other)
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

template<class T>
TTensor<T>::TTensor(const TTensor& other)
    : ndims_(other.ndims_)
{
    ::memcpy(dims_, other.dims_, sizeof(u32) * 4);
    u32 size = sizeof(T) * other.total();
    data_ = static_cast<T*>(PERC_MALLOC(size));
    ::memcpy(data_, other.data_, size);
}

template<class T>
TTensor<T>& TTensor<T>::operator=(const TTensor& other)
{
    if(this != &other) {
        PERC_FREE(data_);
        ndims_ = other.ndims_;
        ::memcpy(dims_, other.dims_, sizeof(u32) * 4);
        u32 size = sizeof(T) * other.total();
        data_ = static_cast<T*>(PERC_MALLOC(size));
        ::memcpy(data_, other.data_, size);
    }
    return *this;
}

template<class T>
TTensor<T>::~TTensor()
{
    PERC_FREE(data_);
    data_ = nullptr;
}

template<class T>
void TTensor<T>::reshape(u32 ndims, const u32 dims[4])
{
    ndims_ = ndims;
    ::memcpy(dims_, dims, sizeof(u32) * 4);

    PERC_FREE(data_);
    data_ = nullptr;
    u32 size = total() * sizeof(T);
    data_ = static_cast<T*>(PERC_MALLOC(size));
}

template<class T>
u32 TTensor<T>::ndims() const
{
    return ndims_;
}

template<class T>
u32 TTensor<T>::dim(u32 d) const
{
    return dims_[d];
}

template<class T>
const u32* TTensor<T>::dims() const
{
    return dims_;
}

template<class T>
u32 TTensor<T>::total() const
{
    u32 p = dims_[0];
    for(u32 i = 1; i < ndims_; ++i) {
        p *= dims_[i];
    }
    return p;
}

template<class T>
T TTensor<T>::operator[](u32 x0) const
{
    return data_[x0];
}

template<class T>
T& TTensor<T>::operator[](u32 x0)
{
    return data_[x0];
}

template<class T>
T TTensor<T>::operator()(u32 x0) const
{
    assert(ndims_ == 1);
    assert(x0 < dims_[0]);
    return data_[x0];
}

template<class T>
T& TTensor<T>::operator()(u32 x0)
{
    assert(ndims_ == 1);
    assert(x0 < dims_[0]);
    return data_[x0];
}

template<class T>
T TTensor<T>::operator()(u32 x0, u32 x1) const
{
    assert(ndims_ == 2);
    assert(x0 < dims_[0]);
    assert(x1 < dims_[1]);
    return data_[x0 * dims_[1] + x1];
}

template<class T>
T& TTensor<T>::operator()(u32 x0, u32 x1)
{
    assert(ndims_ == 2);
    assert(x0 < dims_[0]);
    assert(x1 < dims_[1]);
    return data_[x0 * dims_[1] + x1];
}

template<class T>
T TTensor<T>::operator()(u32 x0, u32 x1, u32 x2) const
{
    assert(ndims_ == 3);
    assert(x0 < dims_[0]);
    assert(x1 < dims_[1]);
    assert(x2 < dims_[2]);
    return data_[x0 * dims_[1] * dims_[2] + x1 * dims_[2] + x2];
}

template<class T>
T& TTensor<T>::operator()(u32 x0, u32 x1, u32 x2)
{
    assert(ndims_ == 3);
    assert(x0 < dims_[0]);
    assert(x1 < dims_[1]);
    assert(x2 < dims_[2]);
    return data_[x0 * dims_[1] * dims_[2] + x1 * dims_[2] + x2];
}

template<class T>
T TTensor<T>::operator()(u32 x0, u32 x1, u32 x2, u32 x3) const
{
    assert(ndims_ == 4);
    assert(x0 < dims_[0]);
    assert(x1 < dims_[1]);
    assert(x2 < dims_[2]);
    assert(x3 < dims_[3]);
    return data_[x0 * dims_[1] * dims_[2] * dims_[3] + x1 * dims_[2] * dims_[3] + x2 * dims_[2] + x3];
}

template<class T>
T& TTensor<T>::operator()(u32 x0, u32 x1, u32 x2, u32 x3)
{
    assert(ndims_ == 4);
    assert(x0 < dims_[0]);
    assert(x1 < dims_[1]);
    assert(x2 < dims_[2]);
    assert(x3 < dims_[3]);
    return data_[x0 * dims_[1] * dims_[2] * dims_[3] + x1 * dims_[2] * dims_[3] + x2 * dims_[2] + x3];
}

template<class T>
const T* TTensor<T>::begin1() const
{
    return data_;
}

template<class T>
T* TTensor<T>::begin1()
{
    return data_;
}

template<class T>
const T* TTensor<T>::begin2(u32 x0) const
{
    return data_ + x0 * dims_[1];
}

template<class T>
T* TTensor<T>::begin2(u32 x0)
{
    return data_ + x0 * dims_[1];
}

template<class T>
const T* TTensor<T>::begin3(u32 x0, u32 x1) const
{
    return data_ + x0 * dims_[1] * dims_[2] + x1 * dims_[2];
}

template<class T>
T* TTensor<T>::begin3(u32 x0, u32 x1)
{
    return data_ + x0 * dims_[1] * dims_[2] + x1 * dims_[2];
}

template<class T>
const T* TTensor<T>::begin4(u32 x0, u32 x1, u32 x2) const
{
    return data_ + x0 * dims_[1] * dims_[2] * dims_[3] + x1 * dims_[2] * dims_[3] + x2 * dims_[3];
}

template<class T>
T* TTensor<T>::begin4(u32 x0, u32 x1, u32 x2)
{
    return data_ + x0 * dims_[1] * dims_[2] * dims_[3] + x1 * dims_[2] * dims_[3] + x2 * dims_[3];
}

bool same_shape(const Tensor& x0, const Tensor& x1);
template<class T>
bool same_shape(const TTensor<T>& x0, const Tensor& x1)
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

template<class T>
bool same_shape(const Tensor& x0, const TTensor<T>& x1)
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

template<class T, class U>
bool same_shape(const TTensor<T>& x0, const TTensor<U>& x1)
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

//--- TensorT
//-----------------------------------------------------
class TensorT
{
public:
    TensorT();
    explicit TensorT(const Tensor* tensor);
    TensorT(TensorT&& other);
    TensorT& operator=(TensorT&& other);
    TensorT(const TensorT& other);
    TensorT& operator=(const TensorT& other);
    ~TensorT();
    u32 ndims() const;
    u32 dim(u32 d) const;
    u32 total() const;
    f32 operator[](u32 x0) const;

    f32 operator()(u32 x0) const;

    f32 operator()(u32 x0, u32 x1) const;

    f32 operator()(u32 x0, u32 x1, u32 x2) const;

    f32 operator()(u32 x0, u32 x1, u32 x2, u32 x3) const;

private:
    const Tensor* tensor_;
    u32 dims_[4];
    u32 cache_[3];
};

void print1(const TensorT& x);
void print2(const TensorT& x);
void print3(const TensorT& x);
Tensor mul(const TensorT& x0, const TensorT& x1);
Tensor mul(const Tensor& x0, const TensorT& x1);
Tensor mul(const TensorT& x0, const Tensor& x1);

enum class Activation
{
    Sigmoid,
    ReLU,
    Softmax,
    None,
};

//--- ILayer
//-----------------------------------------------------
class ILayer
{
public:
    virtual ~ILayer() {}
    virtual u32 dim(u32 /*index*/) const { return 0;}
    virtual Tensor forward(const Tensor& x) = 0;
    virtual Tensor backward(const Tensor& x) = 0;

protected:
    ILayer() {}
};

//--- Relu
//-----------------------------------------------------
class Relu: public ILayer
{
public:
    Relu();
    ~Relu();
    virtual Tensor forward(const Tensor& x) override;
    virtual Tensor backward(const Tensor& x) override;

protected:
    TTensor<bool> mask_;
};

//--- Sigmoid
//-----------------------------------------------------
class Sigmoid: public ILayer
{
public:
    Sigmoid();
    ~Sigmoid();
    virtual Tensor forward(const Tensor& x) override;
    virtual Tensor backward(const Tensor& x) override;

protected:
    Tensor r_;
};

//--- Affine
//-----------------------------------------------------
class Affine: public ILayer
{
public:
    explicit Affine(std::initializer_list<u32> dims);
    Affine(Affine&& other);
    Affine& operator=(Affine&& other);
    ~Affine();

    u32 ndims() const;
    virtual u32 dim(u32 d) const override;
    Tensor forward(const Tensor& x);
    Tensor backward(const Tensor& x);
    f32 weight(u32 x0, u32 x1) const;
    f32& weight(u32 x0, u32 x1);
    f32 bias(u32 x0) const;
    f32& bias(u32 x0);
private:
    Affine(const Affine&) = delete;
    Affine& operator=(const Affine&) = delete;
    Tensor weight_;
    Tensor bias_;
    //Tensor x_;
};

//--- Softmax
//-----------------------------------------------------
class Softmax: public ILayer
{
public:
    Softmax();
    ~Softmax();
    virtual Tensor forward(const Tensor& x) override;
    virtual Tensor backward(const Tensor& t) override;
protected:
    Tensor y_;
};

//--- Loss
//-----------------------------------------------------
class Loss: public ILayer
{
public:
    Loss();
    ~Loss();
    virtual Tensor forward(const Tensor& y) override;
    virtual Tensor backward(const Tensor& t) override;
protected:
    Tensor y_;
};

//--- Layer
//-----------------------------------------------------
class Layer
{
public:
    explicit Layer(Activation activation, bool bias, bool train, std::initializer_list<u32> dims);
    Layer(Layer&& other);
    Layer& operator=(Layer&& other);
    ~Layer();

    u32 ndims() const;
    u32 dim(u32 d) const;
    void set_random();
    Tensor forward(const Tensor& input);
    f32 weight(u32 x0, u32 x1) const;
    f32& weight(u32 x0, u32 x1);
    f32 bias(u32 x0) const;
    f32& bias(u32 x0);

    void set_train(bool train);

private:
    Layer(const Layer&) = delete;
    Layer& operator=(const Layer&) = delete;
    Activation activation_;
    bool train_;
    Tensor weight_;
    Tensor bias_;
    Tensor last_input_;
};

//--- Layer
//-----------------------------------------------------
class IDataLoader
{
public:
    virtual bool next() = 0;
    virtual const Tensor& getInput() = 0;
    virtual const Tensor& getOutput() = 0;
};

//--- Layer
//-----------------------------------------------------
class ILogger
{
public:
    virtual void write(s32 step, f32 loss, const char* format, ...) = 0;
};

//--- Layer
//-----------------------------------------------------
class BaseLogger: public ILogger
{
public:
    virtual void write(s32 step, f32 loss, const char* format, ...) override;
};

//--- Layer
//-----------------------------------------------------
struct TrainParam
{
    u32 totalSteps_ = 1;
    u32 stepsInEpoch_ = 1;
    f32 learningRate_ = 1.0e-7f;
};

//--- Model
//-----------------------------------------------------
class Model
{
public:
    Model();
    ~Model();
    u32 size() const;
    u32 input_size() const;
    u32 output_size() const;

    const ILayer& operator[](u32 index) const;
    ILayer& operator[](u32 index);
    void add(ILayer* layer);

    Tensor predict(const Tensor& input);
    f32 loss(const Tensor& input0, const Tensor& input1);

private:
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    u32 capacity_;
    u32 size_;
    ILayer** layers_;
};
} // namespace perceptron

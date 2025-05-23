#ifndef INC_MINDNN_H_
#define INC_MINDNN_H_
#include <cassert>
#include <cmath>
#include <cstdint>

void* mindnn_malloc(size_t size, const char* file, int32_t line);
void* mindnn_aligned_alloc(size_t alignment, size_t size, const char* file, int32_t line);
void* mindnn_malloc(size_t size);
void* mindnn_aligned_alloc(size_t alignment, size_t size);

void mindnn_free(void* ptr);
void mindnn_aligned_free(void* ptr, size_t alignment);

void* operator new(size_t size);
void* operator new(size_t, std::align_val_t alignment);

void* operator new[](size_t size);
void* operator new[](size_t, std::align_val_t alignment);

void* operator new(size_t size, const char* file, int32_t line);
void* operator new(size_t size, std::align_val_t alignment, const char* file, int32_t line);

void* operator new[](size_t size, const char* file, int32_t line);
void* operator new[](size_t size, std::align_val_t alignment, const char* file, int32_t line);

void operator delete(void* ptr);
void operator delete(void* ptr, std::align_val_t alignment);
void operator delete[](void* ptr);
void operator delete[](void* ptr, std::align_val_t alignment);

#ifdef _DEBUG

#    ifndef MINDNN_MALLOC
#        define MINDNN_MALLOC(size) mindnn_malloc(size, __FILE__, __LINE__)
#    endif

#    ifndef MINDNN_ALIGNED_MALLOC
#        define MINDNN_ALIGNED_MALLOC(align, size) mindnn_aligned_alloc(align, size, __FILE__, __LINE__)
#    endif

#    ifndef MINDNN_FREE
#        define MINDNN_FREE(ptr) \
            mindnn_free(ptr); \
            (ptr) = nullptr
#    endif

#    ifndef MINDNN_ALIGNED_FREE
#        define MINDNN_ALIGNED_FREE(ptr, align) \
            mindnn_aligned_free(ptr, align); \
            (ptr) = nullptr
#    endif

#    ifndef MINDNN_NEW
#        define MINDNN_NEW new(__FILE__, __LINE__)
#    endif

#    ifndef MINDNN_DELETE
#        define MINDNN_DELETE(ptr) \
            delete ptr; \
            (ptr) = nullptr
#    endif

#    ifndef MINDNN_DELETE_ARRAY
#        define MINDNN_DELETE_ARRAY(ptr) \
            delete[] ptr; \
            (ptr) = nullptr
#    endif

#    ifndef MINDNN_NEW_ALIGNED
#        define MINDNN_NEW_ALIGNED(align) new(std::align_val_t{align}, __FILE__, __LINE__)
#    endif

#    ifndef MINDNN_DELETE_ALIGNED
#        define MINDNN_DELETE_ALIGNED(ptr, align) \
            ::operator ::delete(ptr, std::align_val_t{align}); \
            (ptr) = nullptr
#    endif

#    ifndef MINDNN_DELETE_ALIGNED_ARRAY
#        define MINDNN_DELETE_ALIGNED_ARRAY(ptr, align) \
            ::operator ::delete[](ptr, std::align_val_t{align}); \
            (ptr) = nullptr
#    endif

#else

#    ifndef MINDNN_MALLOC
#        define MINDNN_MALLOC(size) mindnn_malloc(size)
#    endif

#    ifndef MINDNN_ALIGNED_MALLOC
#        define MINDNN_ALIGNED_MALLOC(align, size) mindnn_aligned_alloc(align, size)
#    endif

#    ifndef MINDNN_FREE
#        define MINDNN_FREE(ptr) \
            mindnn_free(ptr); \
            (ptr) = nullptr
#    endif

#    ifndef MINDNN_ALIGNED_FREE
#        define MINDNN_ALIGNED_FREE(ptr, align) \
            mindnn_aligned_free(ptr, align); \
            (ptr) = nullptr
#    endif

#    ifndef MINDNN_NEW
#        define MINDNN_NEW new
#    endif

#    ifndef MINDNN_DELETE
#        define MINDNN_DELETE(ptr) \
            delete ptr; \
            (ptr) = nullptr
#    endif

#    ifndef MINDNN_DELETE_ARRAY
#        define MINDNN_DELETE_ARRAY(ptr) \
            delete[] ptr; \
            (ptr) = nullptr
#    endif

#    ifndef MINDNN_NEW_ALIGNED
#        define MINDNN_NEW_ALIGNED(align) new(align)
#    endif

#    ifndef MINDNN_DELETE_ALIGNED
#        define MINDNN_DELETE_ALIGNED(ptr, align) \
            ::operator ::delete(ptr, std::align_val_t{align}); \
            (ptr) = nullptr
#    endif

#    ifndef MINDNN_DELETE_ALIGNED_ARRAY
#        define MINDNN_DELETE_ALIGNED_ARRAY(ptr, align) \
            ::operator ::delete[](ptr, std::align_val_t{align}); \
            (ptr) = nullptr
#    endif

#endif

namespace mindnn
{
using Scalar = float;

//--- RandomPCG32
//-----------------------------------------------------------------
class RandomPCG32
{
public:
    inline static constexpr uint64_t Multiplier = 6364136223846793005ULL;
    inline static constexpr uint64_t Increment = 1442695040888963407ULL;

    RandomPCG32();
    explicit RandomPCG32(uint32_t x);
    ~RandomPCG32();

    void srand(uint32_t x);
    void srand(uint64_t x);
    uint32_t rand();
    float frand();

    uint32_t operator()()
    {
        return rand();
    }

private:
    uint64_t state_;
};

//--- RandomPCG32_128
//-----------------------------------------------------------------
class RandomPCG32_128
{
public:
    RandomPCG32_128();
    ~RandomPCG32_128();

    void srand(uint32_t x0, uint32_t x1);
    void srand(uint64_t x0, uint64_t x1);
    void srand(uint64_t x0);
    uint32_t rand();
    uint32_t range(uint32_t maxx);
    float frand();

    uint32_t operator()()
    {
        return rand();
    }

private:
    RandomPCG32 state0_;
    RandomPCG32 state1_;
    uint32_t x1_;
};

template<class T>
float normal_distribution(T& engine)
{
    float w, z, s;
    do {
        w = engine.frand() * 2.0f - 1.0f;
        z = engine.frand() * 2.0f - 1.0f;
        s = w * w + z * z;
    } while(1.0f <= s || 0.0f == s);
    return std::sqrtf(-2.0f * std::logf(s) / s) * w;
}

void normal_random(Scalar* dst, int32_t size, RandomPCG32_128& random, Scalar mean=0, Scalar sigma=1);

class System
{
public:
    static System& instance();
    RandomPCG32_128& random();

private:
    System(const System&) = delete;
    System& operator=(const System&) = delete;
    System();
    ~System();
    static System instance_;

    RandomPCG32_128 random_;
};

} // namespace mindnn
#endif // INC_MINDNN_H_

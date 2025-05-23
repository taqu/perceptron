#include "mindnn.h"
#include <mimalloc.h>
#include <numbers>

void* mindnn_malloc(size_t size, const char* file, int32_t line)
{
	return ::mi_malloc(size);
}

void* mindnn_aligned_alloc(size_t alignment, size_t size, const char* file, int32_t line)
{
	return ::mi_aligned_alloc(alignment, size);
}

void* mindnn_malloc(size_t size)
{
	return ::mi_malloc(size);
}

void* mindnn_aligned_alloc(size_t alignment, size_t size)
{
	return ::mi_aligned_alloc(alignment, size);
}

void mindnn_free(void* ptr)
{
    ::mi_free(ptr);
}

void mindnn_aligned_free(void* ptr, size_t alignment)
{
	mi_free_aligned(ptr, alignment);
}

void* operator new(std::size_t size, const char* file, int32_t line)
{
	return mi_malloc(size);
}

void* operator new(std::size_t size, std::align_val_t alignment, const char* file, int32_t line)
{
	return ::mi_aligned_alloc((std::size_t)alignment, size);
}

void* operator new[](std::size_t size, const char* file, int32_t line)
{
	return mi_malloc(size);
}

void* operator new[](std::size_t size, std::align_val_t alignment, const char* file, int32_t line)
{
	return ::mi_aligned_alloc((std::size_t)alignment, size);
}

void* operator new(std::size_t size)
{
	return mi_malloc(size);
}

void* operator new(std::size_t size, std::align_val_t alignment)
{
	return ::mi_aligned_alloc((std::size_t)alignment, size);
}

void* operator new[](std::size_t size)
{
	return mi_malloc(size);
}

void* operator new[](std::size_t size, std::align_val_t alignment)
{
	return ::mi_aligned_alloc((std::size_t)alignment, size);
}

void operator delete(void* ptr)
{
    ::mi_free(ptr);
}

void operator delete(void* ptr, std::align_val_t alignment)
{
	::mi_free_aligned(ptr, (std::size_t)alignment);
}

void operator delete[](void* ptr)
{
	::mi_free(ptr);
}

void operator delete[](void* ptr, std::align_val_t alignment)
{
	::mi_free_aligned(ptr, (std::size_t)alignment);
}

namespace mindnn
{
namespace
{
    inline uint32_t rotr32(uint32_t x, uint32_t r)
    {
        return (x >> r) | (x << (-r & 31));
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

float RandomPCG32::frand()
{
    return (rand()>>8)/16777216.0f;
}

//--- RandomPCG32_128
//-----------------------------------------------------------------
RandomPCG32_128::RandomPCG32_128()
{
    srand(1234U, 5678U);
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

void RandomPCG32_128::srand(uint64_t x0)
{
    uint64_t x1 = (x0 ^ (x0 >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x1 = (x1 ^ (x1 >> 27)) * 0x94d049bb133111ebULL;
    x1 = x1 ^ (x1 >> 31);

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

float RandomPCG32_128::frand()
{
    return (rand()>>8)/16777216.0f;
}

void normal_random(Scalar* dst, int32_t size, RandomPCG32_128& random, Scalar mean, Scalar sigma)
{
    static constexpr double two_pi = std::numbers::pi*2.0;
    for(int32_t i=0; i<size-1; i+=2){
        double t0 = sigma * std::sqrt(-2.0*std::log(random.frand()));
        double t1 = two_pi * random.frand();
        dst[i+0] = static_cast<Scalar>(t0 * std::cos(t1) + mean);
        dst[i+1] = static_cast<Scalar>(t0 * std::sin(t1) + mean);
    }
    if(size%2 == 1){
        double t0 = sigma * std::sqrt(-2.0*std::log(random.frand()));
        double t1 = two_pi * random.frand();
        dst[size-1] = static_cast<Scalar>(t0 * std::cos(t1) + mean);
    }
}

System System::instance_;
System& System::instance()
{
    return instance_;
}
        RandomPCG32_128& System::random()
{
            return random_;
}
        System::System()
{
}

        System::~System()
{
}

}


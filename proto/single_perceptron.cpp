#include <cstdint>
#include <cassert>
#include <iostream>

float dot(int32_t len, const float* v0, const float* v1)
{
    float sum = 0;
    for(int32_t i=0; i<len; ++i){
        sum += v0[i] * v1[i];
    }
    return sum;
}

float step(float x)
{
    return 0 < x ? 0 : 1;
}

float forward(int32_t len, const float* x, const float* w)
{
    float r = dot(len, x, w);
    return step(r);
}

void train(int32_t len, float* w, float t, float e)
{
    float z = forward(len, x, w);
    for(int32_t i = 0; i < len; ++i) {
        w[i] += (t - z) * x[i] * e;
    }
}

static constexpr int32_t DATA_NUMS=4;
static constexpr int32_t WEIGHT_NUMS=3;

int main(void)
{
    return 0;
}
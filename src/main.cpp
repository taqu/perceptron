#include "perceptron.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <charconv>
#include <cstdarg>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

float dot(int32_t len, const float* x0, const float* x1)
{
    float sum = 0.0f;
    for(int32_t i = 0; i < len; ++i) {
        sum += x0[i] * x1[i];
    }
    return sum;
}

float step(float x)
{
    return 0.0f< x ? 1.0f : 0.0f;
}

float forward(int32_t len, const float* x, const float* w)
{
    float u = dot(len, x, w);
    return step(u);
}

void train(int32_t len, float* w, const float* x, float t, float e)
{
    float y = forward(len, x, w);
    float d = t - y;
    for(int32_t i = 0; i < len; ++i) {
        w[i] += d * x[i] * e;
    }
}

static constexpr int32_t DATA_NUMS=4;
static constexpr int32_t WEIGHT_NUMS=3;

int main(void)
{
    float e = 0.1f;

    float x[DATA_NUMS][WEIGHT_NUMS] = {
        {1,0,0},
        {1,0,1},
        {1,0,1},
        {1,1,1},
        };
    float t[DATA_NUMS]={
        0,0,0,1,
    };
    float w[WEIGHT_NUMS] = {0,0,0};

    int32_t epoch = 10;
    for(int32_t i=0; i<epoch; ++i) {
        std::cout << "epoch: " << i << std::endl;
        for(int32_t j=0; j<DATA_NUMS; ++j) {
			train(WEIGHT_NUMS, w, x[j], t[j], e);
		}
        for(int32_t j=0; j<WEIGHT_NUMS; ++j){
            std::cout << "w[" << j << "]: " << w[j] << std::endl;
        }
        std::cout << std::endl;
	}
    std::cout << "result: " << std::endl;
    for(int32_t i=0; i<DATA_NUMS; ++i){
        std::cout << forward(WEIGHT_NUMS, x[i], w) << std::endl;
    }
    std::cout << std::endl;
    return 0;
}


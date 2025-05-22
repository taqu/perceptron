#include "catch_amalgamated.hpp"
#include <utility>
#include <iostream>
#include <array>
#include "mindnn.h"
#include "layer.h"

#define EQ_FLOAT(x0, x1) CHECK(std::abs((x0) - (x1)) < 1.0e-7f)

TEST_CASE("1Layer","[DNN]")
{
	std::cout << "1Layer" << std::endl;
    using namespace mindnn;
	static constexpr int32_t NumInputs = 3;
	static constexpr int32_t NumOutputs = 2;
	static constexpr int32_t NumIterations = 1000;
	auto print_fn = [](const float& x) -> void {std::cout << x << ' ';};

	Dense<>::Vector x = {2.0f, 0.5f, 1.0f};
	Dense<>::Vector y_true = {1.5f, 1.0f};
	Dense<> dense(NumInputs, NumOutputs, [](int32_t)->float{return 1.0f;});
	Dense<>::Vector yhat = dense.forward(x);
	float loss = MSE<>::forward(y_true, yhat);

	auto ts = std::chrono::high_resolution_clock::now();
	for(int32_t i=0; i<NumIterations; ++i){
		dense.forward(x);
	}
	auto te = std::chrono::high_resolution_clock::now();
	auto dt_us = (float)std::chrono::duration_cast<std::chrono::microseconds>(te-ts).count()/NumIterations;

	std::cout << "input x=";
	for_each(x.begin(), x.end(), print_fn);
	std::cout << '\n';

	std::cout << "output y=";
	for_each(yhat.begin(), yhat.end(), print_fn);
	std::cout << '\n';

	std::cout << "expected y=";
	for_each(y_true.begin(), y_true.end(), print_fn);
	std::cout << '\n';

	std::cout << "loss: " << loss << '\n';

	Dense<>::Vector dloss_dy = MSE<>::backward(y_true, yhat);
	dense.backward(x, dloss_dy);

	std::cout << "loss gradient: ";
	for_each(dloss_dy.begin(), dloss_dy.end(), print_fn);
	std::cout << '\n';

	std::cout << "updated dense layer weights:\n";
	std::cout << dense;

	std::cout << "time dt=" << dt_us << " usec\n";
}

TEST_CASE("2Layers","[DNN]")
{
	std::cout << "2Layers" << std::endl;
    using namespace mindnn;
	static constexpr int32_t NumInputs = 2;
	static constexpr int32_t NumOutputs = 2;
	static constexpr int32_t NumIterations = 1000;
	auto print_fn = [](const float& x) -> void {std::cout << x << ' ';};

		Dense<>::Vector x = {2.0f, 0.5f};
	Dense<>::Vector y_true = {2.0f, 1.0f};
	Dense<> dense1(NumInputs, NumOutputs, [](int32_t)->float{return 1.0f;});
	Dense<> dense2(NumInputs, NumOutputs, [](int32_t)->float{return 1.0f;});

	Dense<>::Vector y1 = dense1.forward(x);
	Dense<>::Vector y2 = dense2.forward(y1);

	float loss = MSE<>::forward(y_true, y2);

	auto ts = std::chrono::high_resolution_clock::now();
	for(int32_t i=0; i<NumIterations; ++i){
		y1 = dense1.forward(x);
		y2 = dense2.forward(y1);
	}
	auto te = std::chrono::high_resolution_clock::now();
	auto dt_us = (float)std::chrono::duration_cast<std::chrono::microseconds>(te-ts).count()/NumIterations;

	std::cout << "input x=";
	for_each(x.begin(), x.end(), print_fn);
	std::cout << '\n';

	std::cout << "output y1=";
	for_each(y1.begin(), y1.end(), print_fn);
	std::cout << '\n';

	std::cout << "output y2=";
	for_each(y2.begin(), y2.end(), print_fn);
	std::cout << '\n';

	std::cout << "expected y=";
	for_each(y_true.begin(), y_true.end(), print_fn);
	std::cout << '\n';

	std::cout << "loss: " << loss << '\n';

	Dense<>::Vector dloss_dy = MSE<>::backward(y_true, y2);

	Dense<>::Vector bw2 = dense2.backward(y1, dloss_dy);
	dense1.backward(x, bw2);

	std::cout << "loss gradient: ";
	for_each(dloss_dy.begin(), dloss_dy.end(), print_fn);
	std::cout << '\n';

	std::cout << "updated dense 1 layer weights:\n";
	std::cout << dense1;
	std::cout << "updated dense 2 layer weights:\n";
	std::cout << dense2;

	std::cout << "time dt=" << dt_us << " usec\n";
}

TEST_CASE("2Layers + bias","[DNN]")
{
	std::cout << "2Layers + bias" << std::endl;
    using namespace mindnn;
	static constexpr int32_t NumInputs = 2;
	static constexpr int32_t NumOutputs = 2;
	static constexpr int32_t NumIterations = 1000;
	auto print_fn = [](const float& x) -> void {std::cout << x << ' ';};

	Dense<>::Vector x = {2.0f, 0.5f};
	Dense<>::Vector y_true = {2.0f, 1.0f};
	Dense<> dense1(NumInputs, NumOutputs, true, const_initializer<1.0f>, const_initializer<2.0f>);
	Dense<> dense2(NumInputs, NumOutputs, true, const_initializer<1.0f>, const_initializer<2.0f>);

	Dense<>::Vector y1 = dense1.forward(x);
	Dense<>::Vector y2 = dense2.forward(y1);

	float loss = MSE<>::forward(y_true, y2);

	auto ts = std::chrono::high_resolution_clock::now();
	for(int32_t i=0; i<NumIterations; ++i){
		y1 = dense1.forward(x);
		y2 = dense2.forward(y1);
	}
	auto te = std::chrono::high_resolution_clock::now();
	auto dt_us = (float)std::chrono::duration_cast<std::chrono::microseconds>(te-ts).count()/NumIterations;

	std::cout << "input x=";
	for_each(x.begin(), x.end(), print_fn);
	std::cout << '\n';

	std::cout << "output y1=";
	for_each(y1.begin(), y1.end(), print_fn);
	std::cout << '\n';

	std::cout << "output y2=";
	for_each(y2.begin(), y2.end(), print_fn);
	std::cout << '\n';

	std::cout << "expected y=";
	for_each(y_true.begin(), y_true.end(), print_fn);
	std::cout << '\n';

	std::cout << "loss: " << loss << '\n';

	Dense<>::Vector dloss_dy = MSE<>::backward(y_true, y2);

	Dense<>::Vector bw2 = dense2.backward(y1, dloss_dy);
	dense1.backward(x, bw2);

	std::cout << "loss gradient: ";
	for_each(dloss_dy.begin(), dloss_dy.end(), print_fn);
	std::cout << '\n';

	std::cout << "updated dense 1 layer weights:\n";
	std::cout << dense1;
	std::cout << "updated dense 2 layer weights:\n";
	std::cout << dense2;

	std::cout << "time dt=" << dt_us << " usec\n";
}

TEST_CASE("Sigmoid softmax","[DNN]")
{
	std::cout << "Sigmoid softmax" << std::endl;
    using namespace mindnn;
	static constexpr int32_t NumInputs = 2;
	static constexpr int32_t NumOutputs = 2;
	static constexpr int32_t NumIterations = 1000;
	auto print_fn = [](const float& x) -> void {std::cout << x << ' ';};

	std::array<float, 2> biases_init = {1.0f, 2.0f};
	std::array<float, 4> weights_init = {1.0f, 3.0f, 2.0f, 2.0f};

	Dense<>::Vector x = {-1.0f, 0.0f};
	Dense<>::Vector y_true = {1.0f, 0.0f};
	Dense<> dense1(NumInputs, NumOutputs, true, const_initializer<1.0f>, const_initializer<2.0f>);
	Dense<> dense2(NumInputs, NumOutputs, true, [&weights_init](int32_t i){return weights_init[i];}, [&biases_init](int32_t i){return biases_init[i];});
	Sigmoid<> sigmoid;
	Softmax<> softmax;

	Dense<>::Vector y1 = dense1.forward(x);
	Dense<>::Vector y2 = sigmoid.forward(y1);
	Dense<>::Vector y3 = dense2.forward(y2);
	Dense<>::Vector y4 = softmax.forward(y3);

	float loss = MSE<>::forward(y_true, y4);

	auto ts = std::chrono::high_resolution_clock::now();
    for(int32_t i = 0; i < NumIterations; ++i) {
        y1 = dense1.forward(x);
        y2 = sigmoid.forward(y1);
        y3 = dense2.forward(y2);
        y4 = softmax.forward(y3);
    }
	auto te = std::chrono::high_resolution_clock::now();
	auto dt_us = (float)std::chrono::duration_cast<std::chrono::microseconds>(te-ts).count()/NumIterations;

	std::cout << "input x=";
	for_each(x.begin(), x.end(), print_fn);
	std::cout << '\n';

	std::cout << "output y4=";
	for_each(y4.begin(), y4.end(), print_fn);
	std::cout << '\n';

	std::cout << "expected y=";
	for_each(y_true.begin(), y_true.end(), print_fn);
	std::cout << '\n';

	std::cout << "loss: " << loss << '\n';

	Dense<>::Vector dloss_dy = MSE<>::backward(y_true, y4);
	Dense<>::Vector bw4 = softmax.backward(y3, dloss_dy);
	Dense<>::Vector bw3 = dense2.backward(y2, bw4);
	Dense<>::Vector bw2 = sigmoid.backward(y1, bw3);
	dense1.backward(x, bw2);

	std::cout << "loss gradient: ";
	for_each(dloss_dy.begin(), dloss_dy.end(), print_fn);
	std::cout << '\n';

	std::cout << "updated dense 1 layer weights:\n";
	std::cout << dense1;
	std::cout << "updated dense 2 layer weights:\n";
	std::cout << dense2;

	std::cout << "time dt=" << dt_us << " usec\n";
}

TEST_CASE("Cross entropy error","[DNN]")
{
	std::cout << "Sigmoid softmax" << std::endl;
    using namespace mindnn;
	static constexpr int32_t NumInputs = 3;
	static constexpr int32_t NumOutputs = 2;
	static constexpr int32_t NumIterations = 1000;
	auto print_fn = [](const float& x) -> void {std::cout << x << ' ';};

	std::array<float, 2> biases_init = {1.0f, 2.0f};
	std::array<float, 4> weights_init = {1.0f, 3.0f, 2.0f, 2.0f};

	Dense<>::Vector x = {-1.0f, 1.0f, 2.0f};
	Dense<>::Vector y_true = {1.0f, 0.0f};
	Dense<> dense1(NumInputs, NumOutputs, true, const_initializer<1.0f>, const_initializer<2.0f>);
	Dense<> dense2(NumOutputs, NumOutputs, true, [&weights_init](int32_t i){return weights_init[i];}, [&biases_init](int32_t i){return biases_init[i];});
	Sigmoid<> sigmoid;
	Softmax<> softmax;
	CCE<> loss_fn;

	Dense<>::Vector y1 = dense1.forward(x);
	Dense<>::Vector y2 = sigmoid.forward(y1);
	Dense<>::Vector y3 = dense2.forward(y2);
	Dense<>::Vector y4 = softmax.forward(y3);

	float loss = loss_fn.forward(y_true, y4);

	auto ts = std::chrono::high_resolution_clock::now();
    for(int32_t i = 0; i < NumIterations; ++i) {
        y1 = dense1.forward(x);
        y2 = sigmoid.forward(y1);
        y3 = dense2.forward(y2);
        y4 = softmax.forward(y3);
    }
	auto te = std::chrono::high_resolution_clock::now();
	auto dt_us = (float)std::chrono::duration_cast<std::chrono::microseconds>(te-ts).count()/NumIterations;

	std::cout << "input x=";
	for_each(x.begin(), x.end(), print_fn);
	std::cout << '\n';

	std::cout << "output y4=";
	for_each(y4.begin(), y4.end(), print_fn);
	std::cout << '\n';

	std::cout << "expected y=";
	for_each(y_true.begin(), y_true.end(), print_fn);
	std::cout << '\n';

	std::cout << "loss: " << loss << '\n';

	Dense<>::Vector dloss_dy = loss_fn.backward(y_true, y4);
	Dense<>::Vector bw4 = softmax.backward(y3, dloss_dy);
	Dense<>::Vector bw3 = dense2.backward(y2, bw4);
	Dense<>::Vector bw2 = sigmoid.backward(y1, bw3);
	dense1.backward(x, bw2);

	std::cout << "loss gradient: ";
	for_each(dloss_dy.begin(), dloss_dy.end(), print_fn);
	std::cout << '\n';

	std::cout << "updated dense 1 layer weights:\n";
	std::cout << dense1;
	std::cout << "updated dense 2 layer weights:\n";
	std::cout << dense2;

	std::cout << "time dt=" << dt_us << " usec\n";
}


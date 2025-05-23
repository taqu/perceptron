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


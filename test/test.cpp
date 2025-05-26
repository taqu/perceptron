#include "catch_amalgamated.hpp"
#include <utility>
#include <iostream>
#include <array>
#include "mindnn.h"
#include "layer.h"

#define EQ_FLOAT(x0, x1) CHECK(std::abs((x0) - (x1)) < 1.0e-7f)

TEST_CASE("sum","[DNN]")
{
    using namespace mindnn;
	Layer::Matrix a(2,3);
	a(0,0) = -1.2189;
	a(0,1) = 2.8932;
	a(0,2) = 0.0421;
    a(1,0) = -0.9734;
	a(1,1) = 2.1272;
	a(1,2) = -0.2493;

	Layer::Matrix x0 = sum(a, 0);
	std::cout  << x0 << std::endl;
	Layer::Matrix x1 = sum(a, 1);
	std::cout  << x1 << std::endl;
	EQ_FLOAT(x0(0), Scalar(-2.19229984));
	EQ_FLOAT(x0(1), Scalar(5.0204));
	EQ_FLOAT(x0(2), Scalar(-0.2072));
	EQ_FLOAT(x1(0), Scalar(1.71639991));
	EQ_FLOAT(x1(1), Scalar(0.904499888));
}

TEST_CASE("mean","[DNN]")
{
    using namespace mindnn;
	Layer::Matrix a(2,3);
	a(0,0) = -1.2189;
	a(0,1) = 2.8932;
	a(0,2) = 0.0421;
    a(1,0) = -0.9734;
	a(1,1) = 2.1272;
	a(1,2) = -0.2493;

	Layer::Matrix x0 = mean(a, 0);
	std::cout  << x0 << std::endl;
	Layer::Matrix x1 = mean(a, 1);
	std::cout  << x1 << std::endl;
	EQ_FLOAT(x0(0), Scalar(-1.09614992));
	EQ_FLOAT(x0(1), Scalar(2.5102));
	EQ_FLOAT(x0(2), Scalar(-0.1036));
	EQ_FLOAT(x1(0), Scalar(0.572133303));
	EQ_FLOAT(x1(1), Scalar(0.3015));
}

TEST_CASE("LayerNorm","[DNN]")
{
    using namespace mindnn;
	LayerNorm layerNorm;
	LayerNorm::Matrix prev_layer_output(2,3);
	prev_layer_output(0,0) = 0.3831;
	prev_layer_output(0,1) = -0.3478;
	prev_layer_output(0,2) = -1.8104;
	prev_layer_output(1,0) = 1.9287;
	prev_layer_output(1,1) = 1.5867;
	prev_layer_output(1,2) = 0.7086;
	layerNorm.forward(prev_layer_output);
	const LayerNorm::Matrix& a = layerNorm.output();
	const LayerNorm::Vector& mean = layerNorm.mean();
	std::cout << prev_layer_output << std::endl;
	std::cout << a << std::endl;
	std::cout << mean << std::endl;
	EQ_FLOAT(mean(0), Scalar(-0.5917));
	EQ_FLOAT(mean(1), Scalar(1.4080));

	LayerNorm::Matrix next_layer_input(2,3);

	layerNorm.backward(prev_layer_output, a);
#if 0
x:  tensor([[[ 0.3831, -0.3478, -1.8104],
         [ 1.9287,  1.5867,  0.7086]]], requires_grad=True)
w:  tensor([1., 1., 1.], requires_grad=True)
b:  tensor([0., 0., 0.], requires_grad=True)
mean:  tensor([[[-0.5917],
         [ 1.4080]]], grad_fn=<DivBackward0>)
rstd:  tensor([[[1.0965],
         [1.9459]]], grad_fn=<PowBackward0>)
out:  tensor([[[ 1.0689,  0.2674, -1.3364],
         [ 1.0132,  0.3477, -1.3610]]], grad_fn=<AddBackward0>)
dout:  tensor([[[-0.1477,  0.0086, -1.7503],
         [-0.2182, -1.1992, -2.0581]]])
dx:  tensor([[[-0.3245,  0.4866, -0.1622],
         [ 0.4082, -0.5671,  0.1589]]], grad_fn=<MulBackward0>)
dw:  tensor([-0.3790, -0.4147,  5.1401], grad_fn=<SumBackward1>)
db:  tensor([-0.3659, -1.1906, -3.8084])
dx error: 1.6391277313232422e-07
dw error: 0.0
db error: 0.0
#endif
}


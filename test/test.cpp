#include "catch_amalgamated.hpp"
#include <utility>
#include <perceptron.h>

#define EQ_FLOAT(x0, x1) CHECK(std::abs((x0)-(x1))<1.0e-7f)

TEST_CASE("Primitive Functions" "[Primitive]")
{
    using namespace perceptron;
    System::initialize();

	SECTION("step"){
        Tensor tensor({2,4});
        tensor(0,0) = -1.0f;
        tensor(0,1) = -0.5f;
        tensor(0,2) =  0.0f;
        tensor(0,3) =  1.0f;
        tensor(1,0) = -2.0f;
        tensor(1,1) = -0.25f;
        tensor(1,2) =  0.01f;
        tensor(1,3) =  2.0f;
        step(tensor);

        EQ_FLOAT(tensor(0,0), 0.0f);
        EQ_FLOAT(tensor(0,1), 0.0f);
        EQ_FLOAT(tensor(0,2), 0.0f);
        EQ_FLOAT(tensor(0,3), 1.0f);

        EQ_FLOAT(tensor(1,0), 0.0f);
        EQ_FLOAT(tensor(1,1), 0.0f);
        EQ_FLOAT(tensor(1,2), 1.0f);
        EQ_FLOAT(tensor(1,3), 1.0f);
    }

    SECTION("sigmoid"){
        Tensor tensor({2,3});
        tensor(0,0) = -1.0f;
        tensor(0,1) =  1.0f;
        tensor(0,2) =  2.0f;
        tensor(1,0) = -1.0f;
        tensor(1,1) =  1.0f;
        tensor(1,2) =  2.0f;
        sigmoid(tensor);

        EQ_FLOAT(tensor(0,0), 0.26894142f);
        EQ_FLOAT(tensor(0,1), 0.73105858f);
        EQ_FLOAT(tensor(0,2), 0.88079708f);

        EQ_FLOAT(tensor(1,0), 0.26894142f);
        EQ_FLOAT(tensor(1,1), 0.73105858f);
        EQ_FLOAT(tensor(1,2), 0.88079708f);
    }

    SECTION("ReLU"){
        Tensor tensor({2,1});
        tensor(0,0) = -1.0f;
        tensor(1,0) =  1.0f;
        ReLU(tensor);

        EQ_FLOAT(tensor(0,0), 0.0f);

        EQ_FLOAT(tensor(1,0), 1.0f);
    }

    SECTION("mul"){
        Tensor X({1,3});
        X(0,0) = 1.0f;
        X(0,1) = 2.0f;
        X(0,2) = 3.0f;

        Tensor W({3,2});
        W(0,0) = 1.0f;
        W(0,1) = 2.0f;
        W(1,0) = 3.0f;
        W(1,1) = 4.0f;
        W(2,0) = 5.0f;
        W(2,1) = 6.0f;

        Tensor Y = mul(X, W);
        print2(Y);
        EQ_FLOAT(Y(0,0), 22.0f);
        EQ_FLOAT(Y(0,1), 28.0f);
    }

    SECTION("mul bias"){
        Tensor X({1,3});
        X(0,0) = 1.0f;
        X(0,1) = 2.0f;
        X(0,2) = 3.0f;

        Tensor W({3,2});
        W(0,0) = 1.0f;
        W(0,1) = 2.0f;
        W(1,0) = 3.0f;
        W(1,1) = 4.0f;
        W(2,0) = 5.0f;
        W(2,1) = 6.0f;

        Tensor B({2});
        B(0) = 1.0f;
        B(1) = 2.0f;

        Tensor Y = mul(X, W, B);
        print2(Y);
        EQ_FLOAT(Y(0,0), 23.0f);
        EQ_FLOAT(Y(0,1), 30.0f);
    }

    SECTION("mul transpose"){
        Tensor X({1,3});
        X(0,0) = 1.0f;
        X(0,1) = 2.0f;
        X(0,2) = 3.0f;

        Tensor W({2,3});
        W(0,0) = 1.0f;
        W(0,1) = 3.0f;
        W(0,2) = 5.0f;
        W(1,0) = 2.0f;
        W(1,1) = 4.0f;
        W(1,2) = 6.0f;

        Tensor Y = mul_transpose(X, W);
        print2(Y);
        EQ_FLOAT(Y(0,0), 22.0f);
        EQ_FLOAT(Y(0,1), 28.0f);
    }

    SECTION("mul transpose bias"){
        Tensor X({1,3});
        X(0,0) = 1.0f;
        X(0,1) = 2.0f;
        X(0,2) = 3.0f;

        Tensor W({2,3});
        W(0,0) = 1.0f;
        W(0,1) = 3.0f;
        W(0,2) = 5.0f;
        W(1,0) = 2.0f;
        W(1,1) = 4.0f;
        W(1,2) = 6.0f;

        Tensor B({2});
        B(0) = 1.0f;
        B(1) = 2.0f;

        Tensor Y = mul_transpose(X, W, B);
        print2(Y);
        EQ_FLOAT(Y(0,0), 23.0f);
        EQ_FLOAT(Y(0,1), 30.0f);
    }

    SECTION("transpose mul"){
        Tensor X({2,3});
        X(0,0) = 1.0f;
        X(0,1) = 2.0f;
        X(0,2) = 3.0f;
        X(1,0) = 1.0f;
        X(1,1) = 2.0f;
        X(1,2) = 3.0f;

        Tensor W({2,3});
        W(0,0) = 1.0f;
        W(0,1) = 3.0f;
        W(0,2) = 5.0f;
        W(1,0) = 2.0f;
        W(1,1) = 4.0f;
        W(1,2) = 6.0f;

        Tensor Y = transpose_mul(X, W);
        print2(Y);
        EQ_FLOAT(Y(0,0), 3.0f);
        EQ_FLOAT(Y(0,1), 7.0f);
        EQ_FLOAT(Y(0,2), 11.0f);
        EQ_FLOAT(Y(1,0), 6.0f);
        EQ_FLOAT(Y(1,1), 14.0f);
        EQ_FLOAT(Y(1,2), 22.0f);
        EQ_FLOAT(Y(2,0), 9.0f);
        EQ_FLOAT(Y(2,1), 21.0f);
        EQ_FLOAT(Y(2,2), 33.0f);
    }

    SECTION("softmax"){
        Tensor tensor({1, 3});
        tensor(0,0) = 0.3f;
        tensor(0,1) = 2.9f;
        tensor(0,2) = 4.0f;
        softmax(tensor);
        print2(tensor);
        EQ_FLOAT(tensor(0,0), 0.01821127f);
        EQ_FLOAT(tensor(0,1), 0.24519181f);
        EQ_FLOAT(tensor(0,2), 0.73659691f);
    }

    SECTION("mean squared error"){
        Tensor Y({1,10});
        Y(0,0) = 0.1f;
        Y(0,1) = 0.05f;
        Y(0,2) = 0.6f;
        Y(0,3) = 0.0f;
        Y(0,4) = 0.05f;
        Y(0,5) = 0.1f;
        Y(0,6) = 0.0f;
        Y(0,7) = 0.1f;
        Y(0,8) = 0.0f;
        Y(0,9) = 0.0f;

        Tensor T({1,10});
        T(0,0) = 0.0f;
        T(0,1) = 0.0f;
        T(0,2) = 1.0f;
        T(0,3) = 0.0f;
        T(0,4) = 0.0f;
        T(0,5) = 0.0f;
        T(0,6) = 0.0f;
        T(0,7) = 0.0f;
        T(0,8) = 0.0f;
        T(0,9) = 0.0f;

        f32 error = mean_squared_error(Y,T);
        printf("%f\n", error);
        EQ_FLOAT(error, 0.0975f);
    }

    SECTION("cross entropy error"){
        Tensor Y({1,10});
        Y(0,0) = 0.1f;
        Y(0,1) = 0.05f;
        Y(0,2) = 0.6f;
        Y(0,3) = 0.0f;
        Y(0,4) = 0.05f;
        Y(0,5) = 0.1f;
        Y(0,6) = 0.0f;
        Y(0,7) = 0.1f;
        Y(0,8) = 0.0f;
        Y(0,9) = 0.0f;

        Tensor T({1,10});
        T(0,0) = 0.0f;
        T(0,1) = 0.0f;
        T(0,2) = 1.0f;
        T(0,3) = 0.0f;
        T(0,4) = 0.0f;
        T(0,5) = 0.0f;
        T(0,6) = 0.0f;
        T(0,7) = 0.0f;
        T(0,8) = 0.0f;
        T(0,9) = 0.0f;

        f32 error = cross_entropy_error(Y,T);
        printf("%f\n", error);
        EQ_FLOAT(error, 0.51082545f);
    }

    SECTION("Relu"){
        Tensor x({2,2});
        x(0,0) = 1.0f;
        x(0,1) = -0.5f;
        x(1,0) = -2.0f;
        x(1,1) = 3.0f;

        Relu relu;
        Tensor r = relu.forward(x);
        print2(r);
        print_mask(relu);
        EQ_FLOAT(r(0,0), 1.0f);
        EQ_FLOAT(r(0,1), 0.0f);
        EQ_FLOAT(r(1,0), 0.0f);
        EQ_FLOAT(r(1,1), 3.0f);

        x(0,0) = -1.0f;
        x(0,1) = 0.5f;
        x(1,0) = 2.0f;
        x(1,1) = -3.0f;
        r = relu.backward(x);
        print2(r);
        EQ_FLOAT(r(0,0), -1.0f);
        EQ_FLOAT(r(0,1), 0.0f);
        EQ_FLOAT(r(1,0), 0.0f);
        EQ_FLOAT(r(1,1), -3.0f);
    }

    SECTION("Sigmoid"){
        Tensor x({2,2});
        x(0,0) = 1.0f;
        x(0,1) = -0.5f;
        x(1,0) = -2.0f;
        x(1,1) = 3.0f;

        Sigmoid sigmoid;
        Tensor r = sigmoid.forward(x);
        print2(r);
        EQ_FLOAT(r(0,0), 0.731058598f);
        EQ_FLOAT(r(0,1), 0.377540669f);
        EQ_FLOAT(r(1,0), 0.119202922f);
        EQ_FLOAT(r(1,1), 0.952574127f);

        r = sigmoid.backward(x);
        print2(r);
        EQ_FLOAT(r(0,0), 0.19661192429f);
        EQ_FLOAT(r(0,1), -0.11750185612f);
        EQ_FLOAT(r(1,0), -0.20998717077f);
        EQ_FLOAT(r(1,1), 0.13552997871f);
    }

    SECTION("Affine"){
        Tensor x({2,2});
        x(0,0) = 1.0f;
        x(0,1) = -0.5f;
        x(1,0) = -2.0f;
        x(1,1) = 3.0f;

        Affine affine({2,3}, false, true);
        affine.weight(0,0) = 0.1f;
        affine.weight(0,1) = 0.2f;
        affine.weight(0,2) = 0.3f;
        affine.weight(1,0) = 0.4f;
        affine.weight(1,1) = 0.5f;
        affine.weight(1,2) = 0.6f;

        Tensor r = affine.forward(x);
        print2(r);
        EQ_FLOAT(r(0,0), -0.1f);
        EQ_FLOAT(r(0,1), -0.05f);
        EQ_FLOAT(r(0,2), 0.0f);
        EQ_FLOAT(r(1,0), 1.0f);
        EQ_FLOAT(r(1,1), 1.1f);
        EQ_FLOAT(r(1,2), 1.2f);

        Tensor r2 = affine.backward(r);
        print2(r2);
        EQ_FLOAT(r2(0,0), -0.02f);
        EQ_FLOAT(r2(0,1), -0.065f);
        EQ_FLOAT(r2(1,0), 0.68f);
        EQ_FLOAT(r2(1,1), 1.67000008f);

        print2(affine.dw());
        EQ_FLOAT(affine.dw()(0,0), -2.1f);
        EQ_FLOAT(affine.dw()(0,1), -2.25f);
        EQ_FLOAT(affine.dw()(0,2), -2.4f);
        EQ_FLOAT(affine.dw()(1,0), 3.05f);
        EQ_FLOAT(affine.dw()(1,1), 3.32500029f);
        EQ_FLOAT(affine.dw()(1,2), 3.60000014f);

        print1(affine.db());
        EQ_FLOAT(affine.db()(0), 0.9f);
        EQ_FLOAT(affine.db()(1), 1.05000007f);
        EQ_FLOAT(affine.db()(2), 1.2f);
    }

    SECTION("Softmax"){
        Tensor x({1,3});
        x(0,0) = 0.3f;
        x(0,1) = 2.9f;
        x(0,2) = 4.0f;
        Softmax softmax;
        Tensor r = softmax.forward(x);
        print2(r);
        EQ_FLOAT(r(0,0), 0.01821127f);
        EQ_FLOAT(r(0,1), 0.24519181f);
        EQ_FLOAT(r(0,2), 0.73659691f);

        Tensor t({1,3});
        t(0,0) = 1.0f;
        t(0,1) = 2.0f;
        t(0,2) = 3.0f;

        f32 d = cross_entropy_error(r, x);

        Tensor r2 = softmax.backward(t);
        print2(r2);
        EQ_FLOAT(r2(0,0), -0.327262938f);
        EQ_FLOAT(r2(0,1), -0.584936082f);
        EQ_FLOAT(r2(0,2), -0.754467666f);
    }

    SECTION("Gradient"){
        Tensor x({1,2});
        x(0,0) = 0.6f;
        x(0,1) = 0.9f;

        Tensor t({1,3});
        t(0,0) = 0.0f;
        t(0,1) = 0.0f;
        t(0,2) = 1.0f;

        Model model;
        {
            Affine* layer0 = Affine::create({2,3}, true, true);
            random(*layer0, 1.0f);
            //Softmax* softmax = Softmax::create();
            model.add(layer0);
            //model.add(softmax);

            print2(layer0->w());

            Tensor p = model.predict(x);
            print2(p);
            float f = model.loss(x, t);
            printf("%f\n", f);

            auto func = [model = &model](const Tensor& x, const Tensor& t){
                return model->loss(x,t);
            };
            Tensor grad = numerical_gradient(func, model.cast<Affine>(0).w(), t);
            print2(grad);

        }
    }

    #if 0
    SECTION("Backpropagation"){
        Tensor x({1,5});
        x(0,0) = 1.0f;
        x(0,1) = 3.0f;
        x(0,2) = 5.0f;
        x(0,3) = 7.0f;
        x(0,4) = 2.0f;

        Tensor t({1,4});
        t(0,0) = 0.1f;
        t(0,1) = 0.5f;
        t(0,2) = 0.2f;
        t(0,3) = 0.7f;

        Model model;
        {
            Affine* layer0 = Affine::create({5,4}, true, true);
            random(*layer0, 0.01f);
            Affine* layer1 = Affine::create({4,4}, true, true);
            random(*layer1, 0.01f);
            Softmax* softmax = Softmax::create();
            model.add(layer0);
            model.add(layer1);
            model.add(softmax);

            for(s32 i = 0; i < 10; ++i) {
                Tensor r0 = model.forward(x);
                print2(r0);
                f32 loss0 = cross_entropy_error(r0, t);
                printf("loss: %f\n", loss0);
                Tensor d0 = model.backward(t);
                model.update(0.1f);
            }
        }
    }
    #endif
    System::terminate();
}


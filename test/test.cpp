#include "catch_amalgamated.hpp"
#include <utility>
#include <perceptron.h>

#define EQ_FLOAT(x0, x1) CHECK(std::abs(x0-x1)<1.0e-7f)

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
        Tensor X({2});
        X(0) = 1.0f;
        X(1) = 2.0f;

        Tensor W({3,2});
        W(0,0) = 1.0f;
        W(0,1) = 2.0f;
        W(1,0) = 3.0f;
        W(1,1) = 4.0f;
        W(2,0) = 5.0f;
        W(2,1) = 6.0f;

        Tensor Y = mul(X, W);
        print1(Y);
        EQ_FLOAT(Y(0), 5.0f);
        EQ_FLOAT(Y(1), 11.0f);
        EQ_FLOAT(Y(2), 17.0f);
    }

    SECTION("mul bias"){
        Tensor X({2});
        X(0) = 1.0f;
        X(1) = 2.0f;

        Tensor W({3,2});
        W(0,0) = 1.0f;
        W(0,1) = 2.0f;
        W(1,0) = 3.0f;
        W(1,1) = 4.0f;
        W(2,0) = 5.0f;
        W(2,1) = 6.0f;

        Tensor B({3});
        B(0) = 1.0f;
        B(1) = 2.0f;
        B(2) = 3.0f;

        Tensor Y = mul(X, W, B);
        print1(Y);
        EQ_FLOAT(Y(0), 6.0f);
        EQ_FLOAT(Y(1), 13.0f);
        EQ_FLOAT(Y(2), 20.0f);
    }

    SECTION("softmax"){
        Tensor tensor({1, 3});
        tensor(0, 0) = 0.3f;
        tensor(0, 1) = 2.9f;
        tensor(0, 2) = 4.0f;
        softmax(tensor);
        print2(tensor);
        EQ_FLOAT(tensor(0,0), 0.01821127f);
        EQ_FLOAT(tensor(0,1), 0.24519181f);
        EQ_FLOAT(tensor(0,2), 0.73659691f);
    }

    SECTION("mean squared error"){
        Tensor Y({1, 10});
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

        Tensor T({1, 10});
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
        Tensor Y({1, 10});
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

        Tensor T({1, 10});
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

    System::terminate();
}


#ifndef INC_MINDNN_OUTPUT_H_
#define INC_MINDNN_OUTPUT_H_
#include <Eigen/Core>
#include "mindnn.h"

namespace mindnn
{
    enum class OutputType
{
        RegressionMSE,
};

	class Output
{
    public:
        using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
        using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
        using IntegerVector = Eigen::RowVectorXi;

        virtual ~Output();

        // Check the format of target data, e.g. in classification problems the
        // target data should be binary (either 0 or 1)
        virtual void check_target_data(const Matrix& target)
        {
        }

        // Another type of target data where each element is a class label
        // This version may not be sensible for regression tasks, so by default
        // we raise an exception
        virtual void check_target_data(const IntegerVector& target)
        {
        }

        // A combination of the forward stage and the back-propagation stage for the output layer
        // The computed derivative of the input should be stored in this layer, and can be retrieved by
        // the backprop_data() function
        virtual void evaluate(const Matrix& prev_layer_data, const Matrix& target) = 0;

        // Another type of target data where each element is a class label
        // This version may not be sensible for regression tasks, so by default
        // we raise an exception
        virtual void evaluate(const Matrix& prev_layer_data,
                              const IntegerVector& target)
        {
            throw std::invalid_argument("[class Output]: This output type cannot take class labels as target data");
        }

        // The derivative of the input of this layer, which is also the derivative
        // of the output of previous layer
        virtual const Matrix& barkward_data() const = 0;

        // Return the loss function value after the evaluation
        // This function can be assumed to be called after evaluate(), so that it can make use of the
        // intermediate result to save some computation
        virtual Scalar loss() const = 0;

        // Return the output layer type. It is used to export the NN model.
        virtual OutputType output_type() const = 0;

protected:
    Output();
    };

    class RegressionMSE: public Output
    {
    public:
        using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
        using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

        void evaluate(const Matrix& prev_layer_data, const Matrix& target);

        const Matrix& barkward_data() const;

        Scalar loss() const;

        OutputType output_type() const;
        private:
            Matrix din_;
    };

}
#endif //INC_MINDNN_OUTPUT_H_
#include "output.h"

namespace mindnn
{
	void RegressionMSE::evaluate(const Matrix& prev_layer_data, const Matrix& target)
{
        const int32_t cols = prev_layer_data.cols();
            const int32_t rows = prev_layer_data.rows();
            assert(cols == target.cols());
            assert(rows == target.rows());

        // Compute the derivative of the input of this layer
            // L = 0.5 * ||yhat - y||^2
            // in = yhat
            // d(L) / d(in) = yhat - y
            din_.resize(rows, cols);
            din_.noalias() = prev_layer_data - target;
}

        const typename RegressionMSE::Matrix& RegressionMSE::barkward_data() const
{
            return din_;
}

        Scalar RegressionMSE::loss() const
{
            // L = 0.5 * ||yhat - y||^2
            return din_.squaredNorm() / din_.cols() * Scalar(0.5);
}

        OutputType RegressionMSE::output_type() const
{
            return OutputType::RegressionMSE;
}
}

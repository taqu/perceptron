#ifndef INC_MINDNN_UTILS_H_
#define INC_MINDNN_UTILS_H_
#include "Eigen/Core"
#include "mindnn.h"

namespace mindnn
{
	template <int32_t N>
inline int32_t find_max(const Scalar* x)
{
    const int32_t loc = find_max < N - 1 > (x);
    return (x[N - 1] > x[loc]) ? (N - 1) : loc;
}

template <>
inline int32_t find_max<2>(const Scalar* x)
{
    return int32_t(x[1] > x[0]);
}

// n is assumed be >= 2
int32_t find_max(const Scalar* x, int32_t n);

// Find the maximum element in the block x[0:(nrow-1), 0:(ncol-1)]
// col_stride is the distance between x[0, 0] and x[0, 1]
// Special cases for small n
Scalar find_block_max(const Scalar* x, int32_t nrow, int32_t ncol, int32_t col_stride, int32_t& loc);

template<class Type=Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
Eigen::Matrix<Scalar, Eigen::Dynamic, 1> mean(const Type& matrix, int32_t dim=-1)
{
    assert(dim < 2);
    if(dim < 0) {
        dim = 1;
    }
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> result;
    switch(dim) {
    case 0: {
        result.resize(matrix.cols(), 1);
        for(int32_t i=0; i<matrix.cols(); ++i){
            result(i,0) = matrix.col(i).mean();
        }
        return result;
    }
    default: {
        result.resize(matrix.rows(), 1);
        for(int32_t i=0; i<matrix.rows(); ++i){
            result(i,0) = matrix.row(i).mean();
        }
        return result.transpose();
    }
    }
}

template<class Type=Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
Eigen::Matrix<Scalar, Eigen::Dynamic, 1> sum(const Type& matrix, int32_t dim=-1)
{
    assert(dim < 2);
    if(dim < 0) {
        dim = 1;
    }
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> result;
    switch(dim) {
    case 0: {
        result.resize(matrix.cols(), 1);
        for(int32_t i=0; i<matrix.cols(); ++i){
            result(i,0) = matrix.col(i).sum();
        }
        return result;
    }
    default: {
        result.resize(matrix.rows(), 1);
        for(int32_t i=0; i<matrix.rows(); ++i){
            result(i,0) = matrix.row(i).sum();
        }
        return result.transpose();
    }
    }
}

} // namespace mindnn
#endif //INC_MINDNN_UTILS_H_


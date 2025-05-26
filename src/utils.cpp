#include "utils.h"

namespace mindnn
{
// n is assumed be >= 2
int32_t find_max(const Scalar* x, int32_t n)
{
    switch(n) {
    case 2:
        return find_max<2>(x);

    case 3:
        return find_max<3>(x);

    case 4:
        return find_max<4>(x);

    case 5:
        return find_max<5>(x);
    }

    int32_t loc = find_max<6>(x);

    for(int32_t i = 6; i < n; ++i) {
        loc = (x[i] > x[loc]) ? i : loc;
    }
    return loc;
}

Scalar find_block_max(const Scalar* x, int32_t nrow, int32_t ncol, int32_t col_stride, int32_t& loc)
{
    // Max element in the first column
    loc = find_max(x, nrow);
    Scalar val = x[loc];
    // 2nd column
    x += col_stride;
    int32_t loc_next = find_max(x, nrow);
    Scalar val_next = x[loc_next];

    if (val_next > val)
    {
        loc = col_stride + loc_next;
        val = val_next;
    }

    if (ncol == 2)
    {
        return val;
    }

    // 3rd column
    x += col_stride;
    loc_next = find_max(x, nrow);
    val_next = x[loc_next];

    if (val_next > val)
    {
        loc = 2 * col_stride + loc_next;
        val = val_next;
    }

    if (ncol == 3)
    {
        return val;
    }

    // 4th column
    x += col_stride;
    loc_next = find_max(x, nrow);
    val_next = x[loc_next];

    if (val_next > val)
    {
        loc = 3 * col_stride + loc_next;
        val = val_next;
    }

    if (ncol == 4)
    {
        return val;
    }

    // 5th column
    x += col_stride;
    loc_next = find_max(x, nrow);
    val_next = x[loc_next];

    if (val_next > val)
    {
        loc = 4 * col_stride + loc_next;
        val = val_next;
    }

    if (ncol == 5)
    {
        return val;
    }

    // Other columns
    for (int32_t i = 5; i < ncol; ++i)
    {
        x += col_stride;
        loc_next = find_max(x, nrow);
        val_next = x[loc_next];

        if (val_next > val)
        {
            loc = i * col_stride + loc_next;
            val = val_next;
        }
    }

    return val;
}
}

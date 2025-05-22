#include "layer.h"

namespace mindnn
{
    //--- Layer
//--------------------------------------
	Layer::Layer(int32_t in_size, int32_t out_size)
{
}

	Layer::~Layer()
    {
    }

	int32_t Layer::in_size() const
    {
        return in_size_;
    }

	int32_t Layer::out_size() const
    {
        return out_size_;
    }

}


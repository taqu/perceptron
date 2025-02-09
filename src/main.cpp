#include <cstdint>
#include <cassert>
#include <cstdarg>
#include <fstream>
#include <vector>
#include <charconv>
#include <algorithm>
#include <array>
#include "perceptron.h"

struct Item
{
	inline static constexpr int32_t Size = 28;
	int32_t label_;
	uint8_t data_[Size*Size];
};

bool load(std::vector<Item>& data, const char* filepath)
{
	assert(nullptr != filepath);
	data.clear();
	std::ifstream file(filepath, std::ios::binary);
	if(!file.is_open()){
		return false;
	}
	size_t size = Item::Size*Item::Size*6+16;
	std::unique_ptr<char[]> buffer = std::make_unique<char[]>(size);
	for(;;){
        if(file.bad()){
			return false;
        }
		file.getline(buffer.get(), size);
		if(file.eof()){
			break;
		}
		Item item;
		bool first = true;
		int32_t count = 0;
		char* b = buffer.get();
		while(b[0] != '\0'){
			char* e = b;
			while(e[0] != '\0' && e[0] != ','){
				++e;
			}
			int32_t value;
            if(auto [ptr, ec] = std::from_chars(b,e,value); ec == std::errc{}) {
				if(first){
					first = false;
					item.label_ = value;
				}else{
					if((Item::Size*Item::Size)<=count){
						return false;
					}
					item.data_[count] = static_cast<uint8_t>(std::clamp(value, 0, 255));
					++count;
				}
				if(e[0]=='\0'){
                    break;
                }
                b = e+1;
            } else {
                return false;
            }
        }
		if(count != (Item::Size * Item::Size)) {
                return false;
            }
		data.push_back(std::move(item));
	}
	return true;
}

class DataLoader : public perceptron::IDataLoader
{
public:
	inline static constexpr uint32_t BatchSize = 100;
	DataLoader();
	~DataLoader();
    virtual bool next() override;
    virtual const perceptron::Tensor& getInput() override;
    virtual const perceptron::Tensor& getOutput() override;

	std::vector<Item> data_;
	std::array<uint32_t,BatchSize> samples_;
	perceptron::Tensor input_;
	perceptron::Tensor output_;
};

DataLoader::DataLoader()
    :input_({BatchSize,784})
	,output_({BatchSize,10})
{
}

DataLoader::~DataLoader()
{
}

bool DataLoader::next()
{
	using namespace perceptron;
	if(data_.size()<BatchSize){
		return false;
	}
	auto&& engine = perceptron::System::getInstance().getRand();
	for(size_t i=0; i<samples_.size(); ++i){
		samples_[i] = engine.range(data_.size());
	}
	for(uint32_t b=0; b<BatchSize; ++b){
		const Item& item = data_[samples_[b]];
		for(uint32_t i=0; i<input_.dim(1); ++i){
			uint8_t x = item.data_[i];
			input_(b,i) = static_cast<float>(x)/255.0f;
		}
		for(uint32_t i=0; i<output_.dim(1); ++i){
			if(static_cast<uint32_t>(item.label_) == i){
				output_(b,i) = 1.0f;
			}else{
				output_(b,i) = 0.0f;
			}
		}
	}
	return true;
}

const perceptron::Tensor& DataLoader::getInput()
{
	return input_;
}

const perceptron::Tensor& DataLoader::getOutput()
{
	return output_;
}

class TestDataLoader
{
public:
	TestDataLoader();
	~TestDataLoader();
	void initialize();
    const perceptron::Tensor& getInput();
    const perceptron::Tensor& getOutput();

	std::vector<Item> data_;
	perceptron::Tensor input_;
	perceptron::Tensor output_;
};

TestDataLoader::TestDataLoader()
{
}

TestDataLoader::~TestDataLoader()
{
}

void TestDataLoader::initialize()
{
	input_ = perceptron::Tensor({(uint32_t)data_.size(), 784});
	output_ = perceptron::Tensor({(uint32_t)data_.size(), 10});
}

const perceptron::Tensor& TestDataLoader::getInput()
{
	return input_;
}

const perceptron::Tensor& TestDataLoader::getOutput()
{
	return output_;
}

class Logger : public perceptron::ILogger
{
public:
	Logger();
	~Logger();
    virtual void write(perceptron::s32 step, perceptron::f32 loss, const char* format, ...) override;

	private:
		std::ofstream file_;
};

Logger::Logger()
{
	file_.open("log.txt", std::ios::binary);
}

Logger::~Logger()
{
	file_.close();
}

void Logger::write(perceptron::s32 step, perceptron::f32 loss, const char* format, ...)
{
    assert(nullptr != format);
    va_list ap;
    va_start(ap, format);
#ifdef _MSC_VER
    vfprintf_s(stdout, format, ap);
#else
    vfprintf(stdout, format, ap);
#endif
    va_end(ap);
	if(file_.is_open()){
		file_ << step << ',' << loss << std::endl;
	}
}

int main(void)
{
	using namespace perceptron;
	System::initialize();
	DataLoader dataloader;
	if(!load(dataloader.data_, "data/mnist_train.csv")){
		return -1;
	}
	TestDataLoader testloader;
	if(!load(testloader.data_, "data/mnist_test.csv")){
		return -1;
	}
	testloader.initialize();
	u32 num_iteration = 100;
	TrainParam param;
	param.totalSteps_ = 1;
	param.stepsInEpoch_ = 10;
	Logger logger;
    Model model;
    {
        Affine* layer0 = Affine::create({784, 50}, true, true);
        random(*layer0, 0.01f);
        Affine* layer1 = Affine::create({50, 10}, true, true);
        random(*layer1, 0.01f);
        Softmax* softmax = Softmax::create();
        model.add(layer0);
        model.add(layer1);
        model.add(softmax);

		std::vector<f32> acc;
        for(u32 i = 0; i < num_iteration; ++i) {
			if(!dataloader.next()){
				break;
			}
			const Tensor& x = dataloader.getInput();
			const Tensor& t = dataloader.getOutput();
			model.numerical_gradient(x,t);
			model.update(0.1f);
			Tensor y = model.forward(x);
            f32 loss0 = cross_entropy_error(y,t);
            printf("loss: %f\n", loss0);
			for(u32 i=0; i<y.dim(1); ++i){
				printf("%f,", y[i]);
			}
			printf("\n");
			if(0 == (i%param.stepsInEpoch_)){
				auto func = [](const Tensor& y,const Tensor& t){
					assert(y.dim(0) == t.dim(0));
					assert(y.dim(1) == t.dim(1));
					Tensor max_y({y.dim(0)});
					Tensor max_t({t.dim(0)});
					for(u32 i=0; i<y.dim(0); ++i){
						f32 maxy = y(i,0);
						u32 maxi = 0;
						for(u32 j=1; j<y.dim(1); ++j){
							if(maxy<y(i,j)){
								maxy = y(i,j);
								maxi = j;
							}
						}
						max_y[i] = maxi;

						f32 maxt = t(i,0);
						maxi = 0;
						for(u32 j=1; j<t.dim(1); ++j){
							if(maxt<t(i,j)){
								maxt = t(i,j);
								maxi = j;
							}
						}
						max_t[i] = maxi;
					}
					u32 count = 0;
					for(u32 i=0; i<max_y.dim(0); ++i){
						if(isEqual(max_y[i], max_t[i])){
							++count;
						}
					}
					return static_cast<f32>(count)/y.dim(0);
				};
				f32 test_acc = model.accuracy(testloader.getInput(), testloader.getOutput(), func);
				printf("accuracy: %f\n", test_acc);
			}
        }
    }
    System::terminate();
	return 0;
}

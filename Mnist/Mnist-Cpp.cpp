#pragma cling add_library_path("/home/abhishek/Downloads/libtorch-cxx11-abi-shared-with-deps-1.8.1+cpu/libtorch/lib/")
#pragma cling add_include_path("/home/abhishek/Downloads/libtorch-cxx11-abi-shared-with-deps-1.8.1+cpu/libtorch/include")
#pragma cling add_include_path("/home/abhishek/Downloads/libtorch-cxx11-abi-shared-with-deps-1.8.1+cpu/libtorch/include/torch/csrc/api/include/")

#pragma cling load("/home/abhishek/Downloads/libtorch-cxx11-abi-shared-with-deps-1.8.1+cpu/libtorch/lib/libc10.so")
#pragma cling load("/home/abhishek/Downloads/libtorch-cxx11-abi-shared-with-deps-1.8.1+cpu/libtorch/lib/libgomp-75eea7e8.so.1")
#pragma cling load("/home/abhishek/Downloads/libtorch-cxx11-abi-shared-with-deps-1.8.1+cpu/libtorch/lib/libtorch.so")

#include <torch/torch.h>
#include <iostream>
#include <typeinfo>

torch::Tensor Flatten(torch::Tensor x){
    return x.view({x.sizes()[0], -1});
}

struct Net: torch::nn::Module {
    Net()
        : conv1(torch::nn::Conv2dOptions(1, 6, 3)), 
    conv2(torch::nn::Conv2dOptions(6, 16, 3)), 
    fc1(576, 120), 
    fc2(120, 84), fc3(84, 10){
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("fc1", fc1);
        register_module("fc2", fc2);
        register_module("fc3", fc3);
    }
    
    torch::Tensor forward(torch::Tensor x){
        x = torch::max_pool2d(torch::relu(conv1->forward(x)), 2);
        x = torch::max_pool2d(torch::relu(conv2->forward(x)), 2);
        x = Flatten(x);
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        x = fc3->forward(x);
        return x;
    }
    
    torch::nn::Conv2d conv1;
    torch::nn::Conv2d conv2;
    torch::nn::Linear fc1;
    torch::nn::Linear fc2;
    torch::nn::Linear fc3;
}

Net model;

torch::Tensor ip = torch::randn({1, 1, 32, 32});
auto output = model.forward(ip);

std::cout << output;

template<typename DataLoader>
void trainer(DataLoader& loader, Net& net){
    for(auto &batch: loader){
        auto images = batch.data, targets = batch.target;
        std::cout << images.sizes() << " " << targets.sizes(); 
    }
}

void test(){
    
}

auto train_dataset = torch::data::datasets::MNIST("./data");

const size_t train_dataset_size = train_dataset.size().value();

std::cout << train_dataset_size; 

auto item = train_dataset.get(0)

std::cout << item.data.sizes();

auto train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
    std::move(train_dataset), 4);

auto& batch = *train_loader;

typeid(batch).name()



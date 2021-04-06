#include <iostream>
#include <torch/torch.h>

torch::Tensor Flatten(torch::Tensor x){
    return x.view({x.sizes()[0], -1});
}

struct AlexNet: torch::nn::Module {
    AlexNet()
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

AlexNet model;

torch::Tensor ip = torch::randn({1, 1, 32, 32});
auto output = model.forward(ip);

std::cout << output;

void train(){
    
}

void test(){
    
}

#pragma once

#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>

// Training function
template <typename Dataloader>
void train(
    torch::jit::script::Module net,
    torch::nn::Linear lin, Dataloader &train_dl,
    torch::optim::Optimizer &optimizer,
    size_t dataset_size,
    torch::Device device);

// Testing function
template <typename Dataloader>
void test(
    torch::jit::script::Module net,
    torch::nn::Linear lin, Dataloader &test_dl,
    size_t dataset_size);

struct Main
{
    const char *data_path = "/home/abhishek/Desktop/libtorch-example-codes/Mnist/Mnist-PyTorch/data/MNIST/raw";
    const char *jit_model_path = "/home/abhishek/Desktop/libtorch-example-codes/Mnist/resnet18.pt";
    size_t epochs = 10;
    size_t train_batch_size = 128;
    size_t test_batch_size = 1000;
    const int log_interval = 10;
};

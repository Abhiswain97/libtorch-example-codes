#include <torch/torch.h>
#include <typeinfo>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

torch::Tensor Flatten(torch::Tensor &x)
{
    return x.view({x.sizes()[0], -1});
}

struct Net : torch::nn::Module
{
    Net()
        : conv1(torch::nn::Conv2dOptions(1, 6, 3)),
          conv2(torch::nn::Conv2dOptions(6, 16, 3)),
          fc1(400, 120),
          fc2(120, 84), fc3(84, 10)
    {
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("fc1", fc1);
        register_module("fc2", fc2);
        register_module("fc3", fc3);
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = torch::max_pool2d(torch::relu(conv1->forward(x)), 2);

        x = torch::max_pool2d(torch::relu(conv2->forward(x)), 2);

        x = x.view({-1, 400});

        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));

        x = fc3->forward(x);

        return torch::log_softmax(x, /*dim=*/1);
    }

    torch::nn::Conv2d conv1;
    torch::nn::Conv2d conv2;
    torch::nn::Linear fc1;
    torch::nn::Linear fc2;
    torch::nn::Linear fc3;
};

Net model;

template <typename DataLoader>
void train(
    Net &model,
    DataLoader &data_loader,
    torch::optim::Optimizer &optimizer,
    torch::Device &device)
{
    model.train();
    size_t batch_idx = 0;
    for (auto &batch : data_loader)
    {
        auto data = batch.data.to(device);
        auto targets = batch.target.to(device);

        optimizer.zero_grad();
        auto output = model.forward(data);

        auto loss = torch::nll_loss(output, targets);

        AT_ASSERT(!std::isnan(loss.template item<float>()));

        loss.backward();
        optimizer.step();

        if (++batch_idx % 100 == 0)
        {
            std::printf(
                "\rBatch %ld | Loss: %.4f",
                batch_idx,
                loss.template item<float>());
        }
    }
}

auto main() -> int
{
    Net model;

    torch::DeviceType device_type;

    if (torch::cuda::is_available())
    {
        device_type = torch::kCUDA;
    }
    else
    {
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);

    model.to(device);

    auto train_dataset = torch::data::datasets::MNIST("/home/abhishek/Desktop/libtorch-example-codes/Mnist/data").map(torch::data::transforms::Stack<>());

    auto train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(train_dataset), 4);

    torch::optim::SGD optimizer(
        model.parameters(), torch::optim::SGDOptions(0.01).momentum(0.5));

    for (size_t i = 0; i < 10; i++)
    {
        train(model, *train_loader, optimizer, device);
    }
}

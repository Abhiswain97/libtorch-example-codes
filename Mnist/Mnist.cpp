#include <torch/torch.h>
#include <torch/script.h> // One-stop header.

struct Options
{
    size_t epochs = 5;
    int64_t batch_size = 16;
    const char *data_path = "/home/abhishek/Desktop/libtorch-example-codes/Mnist/data";
};

int main()
{
    Options options;
    torch::DeviceType device_type = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

    torch::Device device(device_type);

    /*
    Making the dataset and Data Loader
    ----------------------------------------------------------------
    */
    auto train_dataset = torch::data::datasets::MNIST(options.data_path).map(torch::data::transforms::Stack<>());
    auto train_loader = torch::data::make_data_loader(
        std::move(train_dataset),
        torch::data::DataLoaderOptions().batch_size(options.batch_size).workers(2));

    // Load the resnet50 model
    torch::jit::script::Module model = torch::jit::load("resnet50.pt");

    model.to(device);

    auto optimizer = torch::optim::SGD(model.parameters(), torch::optim::SGDOptions(1e-2).momentum(0.5));

    for (size_t i = 0; i < options.epochs; ++i)
    {
        for (auto &batch : *train_loader)
        {
            auto data = batch.data, targets = batch.target;
        }
    }
}
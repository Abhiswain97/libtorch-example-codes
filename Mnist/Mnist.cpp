#include <torch/torch.h>
#include <torch/script.h>

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

    // Load the resne18 model
    torch::jit::script::Module model = torch::jit::load("/home/abhishek/Desktop/libtorch-example-codes/Mnist/resnet18.pt");

    model.to(device);

    torch::nn::Linear fc(512, 10);

    fc->to(device);

    auto optimizer = torch::optim::SGD(fc->parameters(), torch::optim::SGDOptions(1e-2).momentum(0.5));

    const size_t train_dataset_size = train_dataset.size().value();

    // std::cout << train_dataset_size << std::endl;

    for (size_t i = 0; i < options.epochs; ++i)
    {
        int64_t batch_index = 0;

        for (auto &batch : *train_loader)
        {
            auto data = batch.data.to(device);
            auto target = batch.target.to(device);

            std::vector<torch::jit::IValue> input;
            input.push_back(data);
            optimizer.zero_grad();

            auto output = model.forward(input).toTensor();
            // For transfer learning
            output = output.view({output.size(0), -1});
            output = fc->forward(output);

            auto loss = torch::nll_loss(torch::log_softmax(output, 1), target);

            if (batch_index++ % 10 == 0)
            {
                std::printf(
                    "\rTrain Epoch: %ld [%5ld/%d] Loss: %.4f",
                    i + 1,
                    batch_index * batch.data.size(0),
                    60000,
                    loss.template item<float>());
            }

            loss.backward();
            optimizer.step();
        }
    }
}
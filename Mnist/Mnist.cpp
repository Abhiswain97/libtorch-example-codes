#include <torch/torch.h>
#include <torch/script.h>
#include <map>
#include <chrono>

using namespace torch::data;
using namespace std;

struct Options
{
    size_t epochs = 5;
    int64_t batch_size = 128;
    const char *data_path = "/home/abhishek/Desktop/libtorch-example-codes/Mnist/data";
    datasets::MNIST::Mode train = datasets::MNIST::Mode::kTrain;
    datasets::MNIST::Mode test = datasets::MNIST::Mode::kTest;
};

int main()
{
    Options options;
    torch::DeviceType device_type;

    if (torch::cuda::is_available())
    {
        std::printf("Cuda is available!\n\n");
        device_type = torch::kCUDA;
    }
    else
    {
        std::printf("Cuda is not available!\n\n");
        device_type = torch::kCPU;
    }

    torch::Device device(device_type);

    /*
    Making the dataset and Data Loader
    ----------------------------------------------------------------
    */

    auto train_dataset = datasets::MNIST(
                             options.data_path, options.train)
                             .map(transforms::Normalize<>(0.1307, 0.3081))
                             .map(transforms::Stack<>());
    auto train_loader = make_data_loader(
        move(train_dataset),
        DataLoaderOptions().batch_size(options.batch_size).workers(2));

    auto test_dataset = datasets::MNIST(
                            options.data_path, options.test)
                            .map(transforms::Normalize<>(0.1307, 0.3081))
                            .map(transforms::Stack<>());
    auto test_loader = make_data_loader(
        move(test_dataset),
        DataLoaderOptions().batch_size(options.batch_size).workers(2));

    vector<string> modes = {"Training", "Testing"};

    /*
    Load the jit resne18 model
    ----------------------------------------------------------------

    Our model is of type: torch::jit::script::Module
    Currently libtorch optimizers can't take a jit module parameters. They expect torch::Tensor
    */
    torch::jit::script::Module model = torch::jit::load("/home/abhishek/Desktop/libtorch-example-codes/Mnist/resnet18.pt");

    model.to(device);

    torch::nn::Linear fc(512, 10);

    fc->to(device);

    auto optimizer = torch::optim::SGD(fc->parameters(), torch::optim::SGDOptions(1e-2).momentum(0.5));

    auto criterion = torch::nn::CrossEntropyLoss();

    map<string, int> sizes;
    sizes["Training"] = 60000;
    sizes["Testing"] = 10000;

    double best_loss = 0.0;

    auto start_training = chrono::high_resolution_clock::now();

    for (size_t i = 0; i < options.epochs; ++i)
    {
        for (string &mode : modes)
        {
            auto &data_loader = mode == string("Training") ? *train_loader : *test_loader;

            if (mode == string("Training"))
                fc->train();
            else
            {
                torch::NoGradGuard no_grad_guard;
                fc->eval();
            };

            int64_t batch_index = 0;
            double avg_loss;

            for (auto &batch : data_loader)
            {
                auto data = batch.data.to(device);
                auto target = batch.target.to(device);

                vector<torch::jit::IValue> input;
                input.push_back(data);

                if (mode == string("Training"))
                    optimizer.zero_grad();

                auto output = model.forward(input).toTensor();

                output = output.view({output.size(0), -1});
                output = fc->forward(output);

                auto loss = criterion(output, target);

                if (mode == string("Training"))
                {
                    loss.backward();
                    optimizer.step();
                }
                else
                    avg_loss += loss.template item<float>();

                if (batch_index++ % 10 == 0)
                {
                    std::printf(
                        "\r%s Epoch: %ld [%5ld/%d] Loss: %.4f",
                        mode.c_str(),
                        i + 1,
                        batch_index * batch.data.size(0),
                        sizes[mode],
                        loss.template item<float>());
                }
            }

            if (mode == string("Training"))
                std::printf("\nAverage test loss for epoch %zu: %.4f\n", i, avg_loss / (double)sizes["Testing"]);

            if (avg_loss < best_loss)
                torch::save(fc, "../best_model.pt");
        }
    }

    auto end_training = chrono::high_resolution_clock::now();

    chrono::duration<float> test_duration = end_training - start_training;

    std::cout << "\nTraining completed in: " << test_duration.count() << "s"
              << "\n";
}

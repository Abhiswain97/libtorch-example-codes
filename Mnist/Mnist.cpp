#include <torch/torch.h>
#include <torch/script.h>
#include <chrono>

using namespace torch::data;
using namespace std;

struct Options
{
    size_t epochs = 10;
    int64_t batch_size = 64;
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
        printf("Cuda is available!");
        device_type = torch::kCUDA;
    }
    else
    {
        printf("Cuda is not available!");
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

    const size_t train_dataset_size = train_dataset.size().value();

    /*
    Training the model
    ---------------------------------------------------------------- 
    */

    auto start_training = chrono::high_resolution_clock::now();

    fc->train();

    for (size_t i = 0; i < options.epochs; ++i)
    {
        int64_t batch_index = 0;
        double avg_loss;

        for (auto &batch : *train_loader)
        {
            auto data = batch.data.to(device);
            auto target = batch.target.to(device);

            vector<torch::jit::IValue> input;
            input.push_back(data);
            optimizer.zero_grad();

            auto output = model.forward(input).toTensor();
            // For transfer learning
            output = output.view({output.size(0), -1});
            output = fc->forward(output);

            auto loss = torch::nll_loss(torch::log_softmax(output, 1), target);

            avg_loss += loss.template item<float>();

            if (batch_index++ % 10 == 0)
            {
                printf(
                    "\rTrain Epoch: %ld [%5ld/%d] Loss: %.4f",
                    i + 1,
                    batch_index * batch.data.size(0),
                    60000,
                    loss.template item<float>());
            }

            loss.backward();
            optimizer.step();
        }

        printf("\nAverage training loss for epoch %zu: %.4f\n", i, avg_loss / 60000);
    }

    auto end_training = chrono::high_resolution_clock::now();

    chrono::duration<float> train_duration = end_training - start_training;

    cout << "\nTraining completed in: " << train_duration.count() << "s"
         << "\n";

    /*
    Testing the model
    ---------------------------------------------------------------- 
    */

    auto start_testing = chrono::high_resolution_clock::now();

    torch::NoGradGuard no_grad_guard;
    fc->eval();

    for (size_t i = 0; i < options.epochs; ++i)
    {
        int64_t batch_index = 0;
        double avg_loss;

        for (auto &batch : *test_loader)
        {
            auto data = batch.data.to(device);
            auto target = batch.target.to(device);

            vector<torch::jit::IValue> input;
            input.push_back(data);

            auto output = model.forward(input).toTensor();

            output = output.view({output.size(0), -1});
            output = fc->forward(output);

            auto loss = torch::nll_loss(torch::log_softmax(output, 1), target);
            avg_loss += loss.template item<float>();

            if (batch_index++ % 10 == 0)
            {
                printf(
                    "\rTest Epoch: %ld [%5ld/%d] Loss: %.4f",
                    i + 1,
                    batch_index * batch.data.size(0),
                    10000,
                    loss.template item<float>());
            }
        }
        printf("\nAverage training loss for epoch %zu: %.4f\n", i, avg_loss / 10000);

        cout << "\n Saving model after epoch: " << i << "\n";
        torch::save(fc, "../best_model.pt");
    }

    auto end_testing = chrono::high_resolution_clock::now();

    chrono::duration<float> test_duration = end_testing - start_testing;

    cout << "\nTesting completed in: " << test_duration.count() << "s"
         << "\n";
}
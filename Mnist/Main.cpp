#include "Main.h"

template <typename Dataloader>
void train(
    torch::jit::script::Module net,
    torch::nn::Linear lin,
    Dataloader &train_dl,
    torch::optim::Optimizer &optimizer,
    size_t dataset_size,
    torch::Device device)
{
    // To calculate loss and accuracy per epoch
    float avg_loss = 0.0f;
    int64_t avg_accuracy = 0.0f;

    // Put model and lin to (device)
    net.to(device);
    lin->to(device);

    int64_t batch_index = 0;

    for (auto &batch : train_dl)
    {
        // Zero the gradients of optimizer
        optimizer.zero_grad();

        // Fetch features and labels for current batch
        auto data = batch.data.to(device);
        auto targets = batch.target.to(device);

        /* 
        Make a vector of jit Ivalues 
        and push `data` into it,
        to be able to pass it through the jit model 
        */
        std::vector<torch::jit::IValue> input;
        input.push_back(data);

        // Get the output from the jit model and cast into tensor
        auto output = net.forward(input).toTensor();

        // Resize output tesnor
        output = output.view({output.size(0), -1});

        // pass through linear layer for final output
        output = lin(output);

        // Calculate the loss
        auto loss = torch::nn::CrossEntropyLoss()(output, targets);

        if (++batch_index % Main().log_interval == 0)
            std::printf(
                "\r Batch: %ld Loss: %.4f",
                ++batch_index,
                loss.template item<float>());
        /* 
        Usual PyTorch steps:
            backprop on the nll_loss
            step the optimizer 
        */
        loss.backward();
        optimizer.zero_grad();

        //Calculate number of corrects
        auto corrects = output.argmax(1).eq(targets).sum();

        avg_loss += loss.template item<float>();
        avg_accuracy += corrects.template item<int64_t>();
    }

    std::cout << "\n Average Training loss: " << avg_loss / float(dataset_size);
    std::cout << "\n Average Training accuracy: " << (avg_accuracy / float(dataset_size)) * 100 << "%";
}

template <typename Dataloader>
void test(
    torch::jit::script::Module net,
    torch::nn::Linear lin,
    Dataloader &test_dl,
    size_t dataset_size,
    float best_loss)
{
    // To calculate loss and accuracy per epoch
    float avg_loss = 0.0f;
    int64_t avg_accuracy = 0;

    int64_t batch_index = 0;

    net.to(torch::kCPU);
    lin->to(torch::kCPU);

    for (auto &batch : test_dl)
    {
        // Fetch features and labels for current batch
        auto data = batch.data;
        auto targets = batch.target;

        /* 
        Make a vector of jit Ivalues 
        and push `data` into it,
        to be able to pass it through the jit model 
        */
        std::vector<torch::jit::IValue> input;
        input.push_back(data);

        // Get the output from the jit model and cast into tensor
        auto output = net.forward(input).toTensor();

        // Resize output tesnor
        output = output.view({output.size(0), -1});

        // pass through linear layer for final output
        output = lin(output);

        // Calculate the loss
        auto loss = torch::nn::CrossEntropyLoss()(output, targets);

        if (++batch_index % Main().log_interval == 0)
            std::printf(
                "\r Batch: %ld Loss: %.4f",
                ++batch_index,
                loss.template item<float>());

        //Calculate number of corrects
        auto corrects = output.argmax(1).eq(targets).sum();

        avg_loss += loss.template item<float>();
        avg_accuracy += corrects.template item<int64_t>();
    }

    float final_loss = avg_loss / float(dataset_size);

    std::cout << "\n Average Testing loss: " << final_loss;
    std::cout << "\n Average Testing accuracy: " << (avg_accuracy / float(dataset_size)) * 100 << "%";

    if(final_loss < best_loss ){
        std::cout << "\n Saving model..." << std::endl;
        torch::save(lin, "best_model.pt");
    }
}

auto main() -> int
{
    Main options;
    torch::DeviceType device_type = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

    static float best_loss = 0.0f;

    if (device_type == torch::kCUDA)
        std::cout << "Cuda is available! \n Training on GPU" << std::endl;
    else
        std::cout << "Cuda is not available! \n Training on CPU" << std::endl;

    torch::Device device(device_type);

    // Make datasets and Data Loaders

    // Training dataset and data loader

    auto train_ds = torch::data::datasets::MNIST(
                        options.data_path,
                        torch::data::datasets::MNIST::Mode::kTrain)
                        .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                        .map(torch::data::transforms::Stack<>());

    const size_t train_dataset_size = train_ds.size().value();

    auto train_dl = torch::data::make_data_loader(
        std::move(train_ds),
        torch::data::DataLoaderOptions()
            .batch_size(options.train_batch_size)
            .workers(2));

    // Testing dataset and data loader

    auto test_ds = torch::data::datasets::MNIST(
                       options.data_path,
                       torch::data::datasets::MNIST::Mode::kTest)
                       .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                       .map(torch::data::transforms::Stack<>());

    const size_t test_dataset_size = test_ds.size().value();

    auto test_dl = torch::data::make_data_loader(
        std::move(test_ds),
        torch::data::DataLoaderOptions()
            .batch_size(options.test_batch_size)
            .workers(2));

    // Loading the jit model
    torch::jit::script::Module model = torch::jit::load(options.jit_model_path);

    // Create the last Fully connected layer of resnet18
    torch::nn::Linear fc{512, 10};

    // Create the optimizer
    torch::optim::Adam optimizer(
        fc->parameters(),
        torch::optim::AdamOptions().lr(1e-3));

    // Iterate over the number of epochs
    for (int i = 0; i < options.epochs; i++)
    {
        std::cout << "Epoch: " << i + 1 << std::endl;

        std::cout << "Training: " << std::endl;
        train(model, fc, *train_dl, optimizer, train_dataset_size, device);

        std::cout << "\nTesting: " << std::endl;
        test(model, fc, *test_dl, test_dataset_size, best_loss);

        std::cout << "\n\n";
    }
}

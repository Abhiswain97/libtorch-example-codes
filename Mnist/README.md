# Mnist Transfer learning using Resnet18

Transfer learning for digit recognition using resnet18.

## How to run

### libtorch

- Refer to my [LibtorchDemo](https://github.com/Abhiswain97/LibtorchDemo) repo to download libtorch and test it.

- Clone this repo: `git clone https://github.com/Abhiswain97/libtorch-example-codes.git`

- Download the data from [this](http://yann.lecun.com/exdb/mnist/) link. Remember where you download it. Now in `Mnist.cpp`, in `struct Options` set, 
  ```
  data_path = <absolute path to downloaded data folder>
  ``` 

- Then to run,

  ```
  cd libtorch-example-codes/Mnist
  sh run.sh <absolute path to your libtorch download>
  ```
  
### PyTorch

- I have used a library called [MyVision](https://github.com/Abhiswain97/MyVision) to simplfy things for the demo.
 
- Jupyter notebook containing code -> [Here](https://github.com/Abhiswain97/libtorch-example-codes/blob/main/Mnist/Mnist-PyTorch/mnist-using-myvision.ipynb)

## Reference

- [PyTorch official transfer learning tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

- [Offcial examples](https://github.com/pytorch/examples/tree/master/cpp)

- [Pytorch C++ frontend design and philosophy](https://pytorch.org/tutorials/advanced/cpp_frontend.html#running-the-network-in-forward-mode)

- [libtorch all-in-one docs](https://www.ccoderun.ca/programming/doxygen/pytorch/index.html)

- [libtorch version of Pytorch tutorials](https://github.com/prabhuomkar/pytorch-cpp)

- [Transfer-Learning-Dogs-Cats-Libtorch](https://github.com/BuffetCodes/Transfer-Learning-Dogs-Cats-Libtorch)

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

- If all goes well, you should see this:

  ```
  (base) ➜  Mnist git:(main) ✗ sh run.sh ~/libtorch
  Creating build folder and building the project
  -- The C compiler identification is GNU 9.3.0
  -- The CXX compiler identification is GNU 9.3.0
  -- Detecting C compiler ABI info
  -- Detecting C compiler ABI info - done
  -- Check for working C compiler: /usr/bin/cc - skipped
  -- Detecting C compile features
  -- Detecting C compile features - done
  -- Detecting CXX compiler ABI info
  -- Detecting CXX compiler ABI info - done
  -- Check for working CXX compiler: /usr/bin/c++ - skipped
  -- Detecting CXX compile features
  -- Detecting CXX compile features - done
  -- Looking for pthread.h
  -- Looking for pthread.h - found
  -- Performing Test CMAKE_HAVE_LIBC_PTHREAD
  -- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Failed
  -- Looking for pthread_create in pthreads
  -- Looking for pthread_create in pthreads - not found
  -- Looking for pthread_create in pthread
  -- Looking for pthread_create in pthread - found
  -- Found Threads: TRUE
  -- Found CUDA: /usr/local/cuda (found version "11.2")
  -- Caffe2: CUDA detected: 11.2
  -- Caffe2: CUDA nvcc is: /usr/local/cuda/bin/nvcc
  -- Caffe2: CUDA toolkit directory: /usr/local/cuda
  -- Caffe2: Header version is: 11.2
  -- Found CUDNN: /usr/lib/x86_64-linux-gnu/libcudnn.so
  -- Found cuDNN: v8.1.1  (include: /usr/include, library: /usr/lib/x86_64-linux-gnu/libcudnn.so)
  -- /usr/local/cuda/lib64/libnvrtc.so shorthash is 369df368
  -- Autodetected CUDA architecture(s):  6.1
  -- Added CUDA NVCC flags for: -gencode;arch=compute_61,code=sm_61
  -- Found Torch: /home/abhishek/libtorch/lib/libtorch.so
  -- Configuring done
  -- Generating done
  -- Build files have been written to: /home/abhishek/Desktop/libtorch-example-codes/Mnist/build
  [ 33%] Building CXX object CMakeFiles/Mnist.dir/Mnist.cpp.o
  [ 66%] Building CXX object CMakeFiles/Mnist.dir/MnistSimple.cpp.o
  [100%] Linking CXX executable Mnist
  [100%] Built target Mnist
  Cuda is available!
  Training Epoch: 1 [ 6528/60000] Loss: 1.2269
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

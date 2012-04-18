# CUDA Workshop Projects

Here is an organized folder to develop CUDA applications which should work unchanged under Mac OS X, Linux or Windows.

## Configuration

Install [CMake](http://cmake.org/cmake/resources/software.html) via the installer or your package manger, e.g. on OS X you can do:

```
$ brew install cmake
```

Install CUDA 4.x from NVIDIA's [CUDA Zone](http://developer.nvidia.com/cuda-downloads).

Finally you will also need to make sure that [GLEW](http://glew.sourceforge.net) is installed.

## Adding a new project

In the `src/` create a project folder e.g. `matrixMul/` with its source files and create a new `CMakeLists.txt` listing the CUDA and C/C++ files and the libraries to link against e.g.:

```cmake
INCLUDE_DIRECTORIES(
  .
)

CUDA_ADD_EXECUTABLE (matrixMul
  matrixMul.cu
  matrixMul_gold.cpp
)
  
TARGET_LINK_LIBRARIES (matrixMul
  cutil
  shrutil
  ${CUDA_CUBLAS_LIBRARIES}
)
```

Finally add the new project folder, for example `matrixMul` to the `src/CMakeLists.txt` file:

```cmake
add_subdirectory (matrixMul)
```

## Compiling

Open a Visual Studio command shell or a terminal in Mac OS X or Linux and go to the top directory and do:

```
$ cmake CMakeLists.txt
```

Under Windows this should create a Visual Studio project file which you can open, and under Mac OS X or Linux do:

```
$ make
```

and hopefully the CUDA executable will be compiled in the `bin/` folder.

## License

Please refer to the NVIDIA end user license agreement (EULA) associated with this source code for terms and conditions that govern your use of this software.
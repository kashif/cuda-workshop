# CUDA Workshop Projects

Here is an organized folder to develop CUDA applications which should work unchanged under Mac OS X, Linux or Windows.

## Configuration

Install CMake from:
http://cmake.org/cmake/resources/software.html

Install CUDA 4.0 from:
http://developer.nvidia.com/cuda-downloads

Install GLEW from:
http://glew.sourceforge.net

## Adding a new project

In the `src/` create a project folder e.g. `matrixMul/` with its source files and create a new `CMakeLists.txt` listing the CUDA and C/C++ files and the libraries to link against e.g.:

    CUDA_ADD_EXECUTABLE(matrixMul
      matrixMul.cu
      matrixMul_gold.cpp
    )
  
    TARGET_LINK_LIBRARIES(matrixMul
      cutil
      shrutil
    )

Finally add the new project folder, for example `matrixMul` to the `src/CMakeLists.txt` file:

    add_subdirectory (matrixMul)

## Compiling

Open a Visual Studio command shell or a terminal in Mac OS X or Linux and go to the top directory and do:

    $ cmake CMakeLists.txt

Under Windows this should create a Visual Studio project file which you can open, and under Mac OS X or Linux do:

    $ make

and hopefully the CUDA executable will be compiled in the `bin/` folder.
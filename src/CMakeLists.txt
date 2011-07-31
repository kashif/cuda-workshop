include_directories (
  ../cutil/inc
  ../shrutil/inc
)

set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)

add_subdirectory (matrixMul)
add_subdirectory (quadFloat)
add_subdirectory (Mandelbrot)

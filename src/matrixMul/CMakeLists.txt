CUDA_ADD_EXECUTABLE (matrixMul
  matrixMul.cu
  matrixMul_gold.cpp
)
  
TARGET_LINK_LIBRARIES (matrixMul
  cutil
  shrutil
  ${CUDA_CUBLAS_LIBRARIES}
)
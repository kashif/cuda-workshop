CUDA_ADD_EXECUTABLE (quadFloat
  quadFloat.cu
)
  
TARGET_LINK_LIBRARIES (quadFloat
  cutil
  shrutil
)

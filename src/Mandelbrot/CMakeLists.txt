find_package (OpenGL)
find_package (GLUT)

include_directories (
  ${GLUT_INCLUDE_DIR}
  ${OPENGL_INCLUDE_DIR}
  )

CUDA_ADD_EXECUTABLE (Mandelbrot
  Mandelbrot.cpp
  Mandelbrot_gold.cpp
  Mandelbrot_sm11.cu
  Mandelbrot_sm13.cu
  OPTIONS "-arch=compute_20"
  )
  
TARGET_LINK_LIBRARIES (Mandelbrot
  cutil
  rendercheckgl
  ${GLUT_LIBRARIES}
  ${OPENGL_LIBRARIES}
  )

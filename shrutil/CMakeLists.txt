find_package (GLUT REQUIRED)
find_package (GLEW REQUIRED)

set (BUILD_SHARED_LIBS OFF)

include_directories (inc/
  ${CUDA_INCLUDE_DIRS}
  ${GLUT_INCLUDE_DIR}
  ${GLEW_INCLUDE_PATH}
)

ADD_LIBRARY (shrutil
  src/shrUtils.cpp
  src/rendercheckGL.cpp
  src/cmd_arg_reader.cpp
)

if (WIN32)
  SET_TARGET_PROPERTIES(shrutil PROPERTIES DEFINE_SYMBOL BUILD_DLL)
endif (WIN32)

TARGET_LINK_LIBRARIES(shrutil
  ${GLUT_LIBRARIES}
  ${GLEW_LIBRARY}
)

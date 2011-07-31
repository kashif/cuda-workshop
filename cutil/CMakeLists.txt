find_package (GLUT REQUIRED)
find_package (GLEW REQUIRED)

set (BUILD_SHARED_LIBS OFF)

include_directories (
  inc/ 
  ${CUDA_INCLUDE_DIRS}
  ${GLUT_INCLUDE_DIR}
  ${GLEW_INCLUDE_PATH}
)
  
if (WIN32)
  set ( STOPWATCH_ARCH_CPP "src/stopwatch_win.cpp")
else (WIN32)
  set ( STOPWATCH_ARCH_CPP "src/stopwatch_linux.cpp")
endif (WIN32)

ADD_LIBRARY (cutil
  src/bank_checker.cpp
  src/cmd_arg_reader.cpp
  src/cutil.cpp
  src/stopwatch.cpp
  ${STOPWATCH_ARCH_CPP}
  src/multithreading.cpp
)

ADD_LIBRARY (paramgl
  src/paramgl.cpp
  src/param.cpp
)

ADD_LIBRARY (rendercheckgl
  src/rendercheck_gl.cpp
)

if (WIN32)
  SET_TARGET_PROPERTIES(cutil PROPERTIES DEFINE_SYMBOL BUILD_DLL)
endif (WIN32)

TARGET_LINK_LIBRARIES(paramgl
  ${GLUT_LIBRARIES}
  ${GLEW_LIBRARY}
)

TARGET_LINK_LIBRARIES(rendercheckgl
  cutil
  ${GLUT_LIBRARIES}
  ${GLEW_LIBRARY}
)
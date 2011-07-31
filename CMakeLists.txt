cmake_minimum_required (VERSION 2.8.5)

set (CMAKE_MODULE_PATH 
  "${CMAKE_SOURCE_DIR}/cmake" 
  ${CMAKE_MODULE_PATH}
)
  
find_package (CUDA 4.0 REQUIRED)

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

if(UNIX)
  add_definitions(-DUNIX)
endif(UNIX)

add_subdirectory(cutil)
add_subdirectory(shrutil)
add_subdirectory(src)

CUDA_BUILD_CLEAN_TARGET()

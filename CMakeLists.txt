cmake_minimum_required(VERSION 3.16)
project(irft LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc videoio)

file(GLOB_RECURSE IR_CORE_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)
list(REMOVE_ITEM IR_CORE_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/cli/main.cpp"
)

add_library(ir_core STATIC ${IR_CORE_SOURCES})
target_include_directories(ir_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(ir_core PUBLIC ${OpenCV_LIBS})

add_executable(irft src/cli/main.cpp)
target_link_libraries(irft PRIVATE ir_core)

if(MSVC)
    target_compile_options(ir_core PRIVATE /W4)
    target_compile_options(irft PRIVATE /W4)
else()
    target_compile_options(ir_core PRIVATE -Wall -Wextra)
    target_compile_options(irft PRIVATE -Wall -Wextra)
endif()

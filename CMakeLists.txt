cmake_minimum_required(VERSION 3.25)

set(CMAKE_CONFIGURATION_TYPES "Debug" "Release")

set(PROJECT_NAME mindnn)
project(${PROJECT_NAME})

set(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(AFTER ${PROJECT_ROOT}/include)
set(HEADER_DIR ${PROJECT_ROOT}/include)
set(SOURCE_DIR ${PROJECT_ROOT}/src)
set(THIRDPARTY_DIR ${PROJECT_ROOT}/thirdparty)

# build thirdparties
configure_file(${PROJECT_ROOT}/CMakeLists_Thirdparty.txt.in ${THIRDPARTY_DIR}/CMakeLists.txt)

execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${THIRDPARTY_DIR})
if(result)
    message(FATAL_ERROR "Adding external projects failed: ${result}")
endif()

execute_process(COMMAND ${CMAKE_COMMAND} --build . --config Debug --target ALL_BUILD
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${THIRDPARTY_DIR})
if(result)
    message(FATAL_ERROR "Building debug external projects failed: ${result}")
endif()

execute_process(COMMAND ${CMAKE_COMMAND} --build . --config Release --target ALL_BUILD
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${THIRDPARTY_DIR})
if(result)
    message(FATAL_ERROR "Building release external projects failed: ${result}")
endif()

# mimalloc
set(MIMALLOC_ROOT ${THIRDPARTY_DIR}/mimalloc)
set(MIMALLOC_INCLUDE ${MIMALLOC_ROOT}/include/mimalloc-2.1)
set(MIMALLOC_LIB ${MIMALLOC_ROOT}/lib/mimalloc-2.1)

include_directories(AFTER ${MIMALLOC_INCLUDE})
link_directories(AFTER ${MIMALLOC_LIB})

add_library(MIMALLOC MODULE IMPORTED)
set_target_properties(MIMALLOC PROPERTIES IMPORTED_LOCATION_DEBUG "mimalloc-staticd.lib" IMPORTED_LOCATION_RELEASE "mimalloc-static.lib")

# eigen
set(EIGEN_ROOT ${THIRDPARTY_DIR}/eigen)
set(EIGEN_INCLUDE ${EIGEN_ROOT})
include_directories(AFTER ${EIGEN_INCLUDE})

########################################################################
# Sources
set(HEADERS
    "${HEADER_DIR}/mindnn.h"
    "${HEADER_DIR}/gguf.h"
    "${HEADER_DIR}/layer.h"
    "${HEADER_DIR}/activation.h"
    "${HEADER_DIR}/optimizer.h")

set(SOURCES
    "${SOURCE_DIR}/mindnn.cpp"
    "${SOURCE_DIR}/gguf.cpp"
    "${SOURCE_DIR}/layer.cpp"
    "${SOURCE_DIR}/activation.cpp"
    "${SOURCE_DIR}/optimizer.cpp")

source_group("include" FILES ${HEADERS})
source_group("src" FILES ${SOURCES} "${SOURCE_DIR}/main.cpp")

set(FILES ${HEADERS} ${SOURCES} "${SOURCE_DIR}/main.cpp")

set(OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_DIRECTORY}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${OUTPUT_DIRECTORY}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${OUTPUT_DIRECTORY}")

#set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

add_executable(${PROJECT_NAME} ${FILES})

if(MSVC)
    set(DEFAULT_CXX_FLAGS "/DWIN32 /D_WINDOWS /D_MBCS /W4 /WX- /nologo /fp:precise /arch:AVX /Zc:wchar_t /TP /Gd /std:c++20 /std:c11 /DEIGEN_MPL2_ONLY")
    set(CMAKE_CXX_FLAGS "${DEFAULT_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS_DEBUG "/D_DEBUG /MTd /Zi /Ob0 /Od /RTC1 /Gy /GR /GS /Gm- /EHsc")
    set(CMAKE_CXX_FLAGS_RELEASE "/MT /O2 /Oi /GL /GR /DNDEBUG /EHsc-")
    target_link_libraries(${PROJECT_NAME} MIMALLOC)

elseif(UNIX)
    set(DEFAULT_CXX_FLAGS "-Wall -O2 -std=c++20 -std=gnu++20 -march=x86-64-v3 -fno-exceptions -DEIGEN_MPL2_ONLY")
    set(CMAKE_CXX_FLAGS "${DEFAULT_CXX_FLAGS}")
    target_link_libraries(${PROJECT_NAME} MIMALLOC)
elseif(APPLE)
elseif(APPLE)
endif()

set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})
set_target_properties(${PROJECT_NAME}
    PROPERTIES
        OUTPUT_NAME_DEBUG "${PROJECT_NAME}" OUTPUT_NAME_RELEASE "${PROJECT_NAME}"
        VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

add_subdirectory(test)
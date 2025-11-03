cmake_minimum_required(VERSION 3.12)
project(dna2_energy_calculator)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find PyTorch
find_package(Torch REQUIRED)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${TORCH_INCLUDE_DIRS})

# Add definitions
add_definitions(${TORCH_DEFINITIONS})

# Compiler flags
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
endif()

# CUDA support
if(TORCH_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDA REQUIRED)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    
    # CUDA flags
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --expt-relaxed-constexpr")
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -O0")
    set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 -DNDEBUG")
endif()

# Source files
set(SOURCES
    src/dna2_data_structures.cpp
    src/dna2_mathematical_functions.cpp
    src/dna2_mesh_interpolation.cpp
    src/dna2_interactions.cpp
    src/dna2_energy_pipeline.cpp
    src/dna2_energy_calculator.cpp
)

# Header files
set(HEADERS
    include/dna2_energy_calculator.h
    include/dna2_data_structures.h
    include/dna2_mathematical_functions.h
    include/dna2_mesh_interpolation.h
    include/dna2_interactions.h
    include/dna2_energy_pipeline.h
)

# Create the main library
add_library(dna2_energy STATIC ${SOURCES} ${HEADERS})

# Link libraries
target_link_libraries(dna2_energy "${TORCH_LIBRARIES}")

# Set properties
set_target_properties(dna2_energy PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# If CUDA is available, set CUDA properties
if(TORCH_ENABLE_CUDA)
    set_target_properties(dna2_energy PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

# Create example executable
add_executable(dna2_example examples/example_usage.cpp)
target_link_libraries(dna2_example dna2_energy "${TORCH_LIBRARIES}")

# Create test executable
add_executable(dna2_test tests/test_basic.cpp)
target_link_libraries(dna2_test dna2_energy "${TORCH_LIBRARIES}")

# Installation
install(TARGETS dna2_energy
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES ${HEADERS} DESTINATION include/dna2_energy)

# Print configuration information
message(STATUS "PyTorch version: ${TORCH_VERSION}")
message(STATUS "PyTorch include dirs: ${TORCH_INCLUDE_DIRS}")
message(STATUS "PyTorch libraries: ${TORCH_LIBRARIES}")
message(STATUS "CUDA available: ${TORCH_ENABLE_CUDA}")

if(TORCH_ENABLE_CUDA)
    message(STATUS "CUDA version: ${CUDA_VERSION}")
    message(STATUS "CUDA include dirs: ${CUDA_INCLUDE_DIRS}")
    message(STATUS "CUDA libraries: ${CUDA_LIBRARIES}")
endif()

# Add custom targets for testing
add_custom_target(run_tests
    COMMAND dna2_test
    DEPENDS dna2_test
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running DNA2 energy calculator tests"
)

add_custom_target(run_example
    COMMAND dna2_example
    DEPENDS dna2_example
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running DNA2 energy calculator example"
)

# Documentation (optional)
find_package(Doxygen QUIET)
if(DOXYGEN_FOUND)
    set(DOXYGEN_INPUT_DIR ${CMAKE_SOURCE_DIR}/include)
    set(DOXYGEN_OUTPUT_DIR ${CMAKE_BINARY_DIR}/docs)
    
    configure_file(${CMAKE_SOURCE_DIR}/docs/Doxyfile.in 
                   ${CMAKE_BINARY_DIR}/Doxyfile @ONLY)
    
    add_custom_target(docs
        ${DOXYGEN_EXECUTABLE} ${CMAKE_BINARY_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM
    )
endif()

# Packaging
set(CPACK_PACKAGE_NAME "DNA2EnergyCalculator")
set(CPACK_PACKAGE_VERSION "1.0.0")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "PyTorch C++ DNA2 Energy Calculator")
set(CPACK_PACKAGE_VENDOR "oxDNA Project")
set(CPACK_PACKAGE_CONTACT "oxdna-developers@googlegroups.com")

if(WIN32)
    set(CPACK_GENERATOR "ZIP;NSIS")
elseif(APPLE)
    set(CPACK_GENERATOR "ZIP;DragNDrop")
else()
    set(CPACK_GENERATOR "ZIP;TGZ")
endif()

include(CPack)
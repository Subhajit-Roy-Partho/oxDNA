# CMakeLists.txt for Metal GPU backend
# Requires macOS and Metal-capable GPU (Apple Silicon or compatible)

IF(APPLE)
    MESSAGE(STATUS "Building Metal GPU support for Apple Silicon")

    # Find Metal framework
    FIND_LIBRARY(METAL_FRAMEWORK Metal REQUIRED)
    FIND_LIBRARY(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    FIND_LIBRARY(COREGRAPHICS_FRAMEWORK CoreGraphics REQUIRED)

    IF(NOT METAL_FRAMEWORK)
        MESSAGE(FATAL_ERROR "Metal framework not found. Metal requires macOS 10.11 or later with Metal-capable GPU.")
    ENDIF()

    MESSAGE(STATUS "Metal framework: ${METAL_FRAMEWORK}")
    MESSAGE(STATUS "Foundation framework: ${FOUNDATION_FRAMEWORK}")

    # Set Objective-C++ compiler flags
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fobjc-arc")

    # Metal shader compilation
    SET(METAL_SHADERS
        Shaders/common.metal
        Shaders/md_kernels.metal
    )

    # Compile Metal shaders to a library
    SET(METAL_LIBRARY ${CMAKE_CURRENT_BINARY_DIR}/default.metallib)

    # Custom command to compile Metal shaders
    ADD_CUSTOM_COMMAND(
        OUTPUT ${METAL_LIBRARY}
        COMMAND xcrun -sdk macosx metal -c ${CMAKE_CURRENT_SOURCE_DIR}/Shaders/common.metal -o ${CMAKE_CURRENT_BINARY_DIR}/common.air
        COMMAND xcrun -sdk macosx metal -c ${CMAKE_CURRENT_SOURCE_DIR}/Shaders/md_kernels.metal -o ${CMAKE_CURRENT_BINARY_DIR}/md_kernels.air
        COMMAND xcrun -sdk macosx metallib ${CMAKE_CURRENT_BINARY_DIR}/common.air ${CMAKE_CURRENT_BINARY_DIR}/md_kernels.air -o ${METAL_LIBRARY}
        DEPENDS ${METAL_SHADERS}
        COMMENT "Compiling Metal shaders"
        VERBATIM
    )

    ADD_CUSTOM_TARGET(metal_shaders ALL DEPENDS ${METAL_LIBRARY})

    # Copy the Metal library to the binary directory so it can be found by the executable
    ADD_CUSTOM_COMMAND(
        TARGET metal_shaders POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${METAL_LIBRARY}
                ${CMAKE_BINARY_DIR}/bin
    )

    # Metal backend source files
    SET(METAL_SOURCES
        MetalUtils.mm
        Backends/MetalBaseBackend.mm
        Backends/MD_MetalBackend.mm
        # Add more backend implementations here as needed
    )

    # Add Metal sources to the common library
    # Note: These will be added to the main project's common library
    SET(metal_common_SOURCES
        ${METAL_SOURCES}
        PARENT_SCOPE
    )

    # Metal compilation flags
    IF(METAL_DOUBLE_PRECISION)
        ADD_DEFINITIONS(-DMETAL_DOUBLE_PRECISION)
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMETAL_DOUBLE_PRECISION")
    ENDIF()

    # Include directories
    INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

    # Framework linking will be handled in the main CMakeLists.txt
    SET(METAL_FRAMEWORKS
        ${METAL_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
        ${COREGRAPHICS_FRAMEWORK}
        PARENT_SCOPE
    )

    MESSAGE(STATUS "Metal backend configured successfully")

    # Install Metal shader library
    INSTALL(FILES ${METAL_LIBRARY} DESTINATION bin)

ELSE()
    MESSAGE(STATUS "Metal backend is only available on macOS - skipping")
ENDIF(APPLE)

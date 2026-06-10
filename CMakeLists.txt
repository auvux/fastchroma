cmake_minimum_required(VERSION 3.18)
project(fastchroma LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(pybind11 CONFIG REQUIRED)
find_package(Threads REQUIRED)

# Backend: vDSP on Apple, PFFFT elsewhere. Override with -DFASTCHROMA_FFT=pffft.
set(FASTCHROMA_FFT "auto" CACHE STRING "FFT backend: auto | vdsp | pffft")
if(FASTCHROMA_FFT STREQUAL "auto")
  if(APPLE)
    set(FASTCHROMA_FFT "vdsp")
  else()
    set(FASTCHROMA_FFT "pffft")
  endif()
endif()
message(STATUS "fastchroma FFT backend: ${FASTCHROMA_FFT}")

# GPU backend: Metal on Apple, CUDA where a toolkit is found, stub otherwise.
# Override with -DFASTCHROMA_GPU=metal|cuda|none; pick a CUDA toolkit with the
# CUDACXX environment variable.
set(FASTCHROMA_GPU "auto" CACHE STRING "GPU backend: auto | metal | cuda | none")
if(FASTCHROMA_GPU STREQUAL "auto")
  if(APPLE)
    set(FASTCHROMA_GPU "metal")
  else()
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
      set(FASTCHROMA_GPU "cuda")
    else()
      set(FASTCHROMA_GPU "none")
    endif()
  endif()
endif()
message(STATUS "fastchroma GPU backend: ${FASTCHROMA_GPU}")

pybind11_add_module(_fastchroma src/fastchroma.cpp src/bindings.cpp)
target_include_directories(_fastchroma PRIVATE src src/third_party)
target_link_libraries(_fastchroma PRIVATE Threads::Threads)

if(FASTCHROMA_GPU STREQUAL "metal")
  # Metal kernels compile at runtime from source, so this only needs the
  # frameworks — no metallib step, no extra dependencies.
  if(NOT APPLE)
    message(FATAL_ERROR "FASTCHROMA_GPU=metal requires macOS")
  endif()
  enable_language(OBJCXX)
  set(CMAKE_OBJCXX_STANDARD 17)
  set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
  target_sources(_fastchroma PRIVATE src/metal_backend.mm)
  set_source_files_properties(src/metal_backend.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
  target_compile_definitions(_fastchroma PRIVATE FASTCHROMA_METAL=1)
  find_library(METAL_FRAMEWORK Metal REQUIRED)
  find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
  target_link_libraries(_fastchroma PRIVATE ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
elseif(FASTCHROMA_GPU STREQUAL "cuda")
  # Detect a user-supplied arch list before enable_language fills in the
  # toolkit default (sm_52) and masks the distinction.
  if(DEFINED CMAKE_CUDA_ARCHITECTURES AND CMAKE_CUDA_ARCHITECTURES)
    set(fastchroma_user_archs TRUE)
  else()
    set(fastchroma_user_archs FALSE)
  endif()
  enable_language(CUDA)
  target_sources(_fastchroma PRIVATE src/cuda_backend.cu)
  target_compile_definitions(_fastchroma PRIVATE FASTCHROMA_CUDA=1)
  if(NOT fastchroma_user_archs)
    # Distributable fatbin: SASS for every supported generation, plus PTX
    # (the plain "90" entry embeds both SASS and PTX) so future
    # architectures JIT. SASS is forward-compatible within a major version,
    # so 60-real also covers 61/62, 80-real covers 86 dies, etc. — the
    # explicit minors are there to get native code, not for correctness.
    set(fastchroma_archs 75-real 80-real 86-real 89-real 90)
    set(FASTCHROMA_MIN_CC 75)
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13)
      list(PREPEND fastchroma_archs 60-real 70-real)  # CUDA 13 dropped pre-Turing
      set(FASTCHROMA_MIN_CC 60)
    endif()
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
      list(APPEND fastchroma_archs 100-real 120-real)  # Blackwell
    endif()
    set(CMAKE_CUDA_ARCHITECTURES ${fastchroma_archs})
    # Tell the backend the oldest SASS shipped, so devices below it get a
    # clean CPU fallback instead of a failed kernel launch.
    target_compile_definitions(_fastchroma PRIVATE FASTCHROMA_MIN_CC=${FASTCHROMA_MIN_CC})
  endif()
  set_property(TARGET _fastchroma PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
  # Static cudart: the module needs only the NVIDIA driver at runtime, and
  # degrades to gpu_available() == False on machines without one.
  set_property(TARGET _fastchroma PROPERTY CUDA_RUNTIME_LIBRARY Static)
elseif(NOT FASTCHROMA_GPU STREQUAL "none")
  message(FATAL_ERROR "Unknown FASTCHROMA_GPU='${FASTCHROMA_GPU}' (use metal, cuda, or none)")
endif()

if(FASTCHROMA_FFT STREQUAL "vdsp")
  find_library(ACCELERATE Accelerate REQUIRED)
  target_link_libraries(_fastchroma PRIVATE ${ACCELERATE})
elseif(FASTCHROMA_FFT STREQUAL "pffft")
  target_sources(_fastchroma PRIVATE src/third_party/pffft.c)
  target_compile_definitions(_fastchroma PRIVATE FASTCHROMA_NO_ACCELERATE)
else()
  message(FATAL_ERROR "Unknown FASTCHROMA_FFT='${FASTCHROMA_FFT}' (use vdsp or pffft)")
endif()

if(MSVC)
  target_compile_options(_fastchroma PRIVATE
    $<$<COMPILE_LANGUAGE:C,CXX>:/O2;/fp:fast>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=/fp:fast>)
else()
  target_compile_options(_fastchroma PRIVATE
    $<$<COMPILE_LANGUAGE:C,CXX>:-O3;-ffast-math;-funroll-loops>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
endif()

install(TARGETS _fastchroma DESTINATION fastchroma)

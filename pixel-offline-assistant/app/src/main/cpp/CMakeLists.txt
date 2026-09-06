cmake_minimum_required(VERSION 3.22.1)
project(offline_assistant_native)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Pull in the two vendored engines as subdirectories — their own CMakeLists.txt
# build ggml/llama/whisper as static libs we link against below.
add_subdirectory(llama.cpp)
add_subdirectory(whisper.cpp)

add_library(llama_jni SHARED llama_jni.cpp)
target_link_libraries(llama_jni llama log android)

add_library(whisper_jni SHARED whisper_jni.cpp)
target_link_libraries(whisper_jni whisper log android)

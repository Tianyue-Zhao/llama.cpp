add_library(cogagent OBJECT
            cogagent_util.h
            vision_encoder.cpp
            vision-encoder.h
            cross_vision.cpp
            cross_vision.h)

target_link_libraries(cogagent PRIVATE ggml llama ${CMAKE_THREAD_LIBS_INIT})

target_include_directories(cogagent PUBLIC .)
target_include_directories(cogagent PUBLIC ../..)
target_include_directories(cogagent PUBLIC ../../common)

target_compile_features(cogagent PRIVATE cxx_std_11)

add_library(cogagent_static STATIC $<TARGET_OBJECTS:cogagent>)
if (BUILD_SHARED_LIBS)
    set_target_properties(cogagent PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(cogagent PRIVATE LLAMA_SHARED LLAMA_BUILD)
    add_library(cogagent_shared SHARED $<TARGET_OBJECTS:cogagent>)
    target_link_libraries(cogagent_shared PRIVATE ggml llama ${CMAKE_THREAD_LIBS_INIT})
    install(TARGETS cogagent_shared LIBRARY)
endif()

if (NOT MSVC)
    target_compile_options(cogagent PRIVATE -Wno-cast-qual) # stb_image.h
endif()

if(TARGET BUILD_INFO)
    add_dependencies(cogagent BUILD_INFO)
endif()

set(TARGET llama-cogagent-cli)
add_executable(${TARGET} cogagent-cli.cpp)
set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME llama-cogagent-cli)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common cogagent ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)
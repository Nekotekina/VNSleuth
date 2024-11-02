cmake_minimum_required(VERSION 3.14)
project(vnsleuth C CXX)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2 -g")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Include the llama.cpp submodule
add_subdirectory(llama.cpp)

add_executable(vnsleuth
	main.cpp
	parser.cpp
	translator.cpp
	tools/parser_base.cpp
	tools/format_arc_dsc.cpp
)

target_include_directories(vnsleuth
	PUBLIC
	${PROJECT_SOURCE_DIR}/llama.cpp/include
	${PROJECT_SOURCE_DIR}/llama.cpp/common
	${PROJECT_SOURCE_DIR}/llama.cpp/ggml/include
)

target_link_libraries(vnsleuth
	PRIVATE
	common
)

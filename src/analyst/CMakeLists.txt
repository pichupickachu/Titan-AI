cmake_minimum_required(VERSION 3.20)
project(titan_math)

set(CMAKE_CXX_STANDARD 17)
#your way tp python
set(Python_ROOT_DIR "C:/Users/potze/AppData/Local/Programs/Python/Python312")

#  python.exe
set(Python_EXECUTABLE "${Python_ROOT_DIR}/python.exe")

# finding python
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# finding pybind11
execute_process(
        COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE _pybind11_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${_pybind11_dir}")
find_package(pybind11 REQUIRED)

pybind11_add_module(titan_math titan_math.cpp)

# (from DLL errors)
if(MSVC)
    target_compile_options(titan_math PRIVATE /MT)
endif()

set_target_properties(titan_math PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
set_target_properties(titan_math PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

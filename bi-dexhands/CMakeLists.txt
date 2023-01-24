cmake_minimum_required(VERSION 3.2.4)
project(DexterousHands)
# config python path to conda env
set(PYTHON_LIBRARY /home/robot-learning/Software/miniconda3/envs/dexgrasp/lib/libpython3.8.so)
set(PYTHON_INCLUDE_DIR /home/robot-learning/Software/miniconda3/envs/dexgrasp/include/python3.8)
set(PYTHON_EXECUTABLE /home/robot-learning/Software/miniconda3/envs/dexgrasp/bin/python3.8)

find_package(YAML-CPP REQUIRED)

add_subdirectory(pybind11)
find_package(pybind11 CONFIG)

# pybind11 method:
pybind11_add_module(finger_map utils/teleop/finger_map.cpp)
target_link_libraries(finger_map PRIVATE ${YAML_CPP_LIBRARIES})
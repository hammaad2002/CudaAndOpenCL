format:
	find . -iname "CMakeLists.txt" -exec cmake-format -i "{}" \; && find . -iname "*.cmake" -exec cmake-format -i "{}" \; && clear

build:
	rm -rf build && mkdir build && cd build && cmake .. && make -j $(nproc) && mv OpenCLSimpleStructure ../ && cd .. && rm -rf build

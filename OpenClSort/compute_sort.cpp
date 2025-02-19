#include <iostream>
#include <boost/compute/core.hpp>
#include <random>
#include <vector>
#include <boost/compute/algorithm/sort.hpp>

int main(){

    // find a device
    auto device = boost::compute::system::default_device();
    std::cout << "Device: " << device.name() << std::endl;

    // meta data about the FUCKING DEVICEEEEEEEEE (HIGH ON 'FREE BIRD BY MOONLIGHT')
    std::cout << "Device's name:          " << device.name() << std::endl;
    std::cout << "Device's global memory: " << (float)device.global_memory_size() / (float)(1024*1024*1024) << " GB" << std::endl;
    std::cout << "Device's local memory:  " << (float)device.local_memory_size() / (float)(1024) << "KB" << std::endl;
    std::cout << "Device's compute_units: " << device.compute_units() << std::endl;
    std::cout << "Device's device type:   " << device.type() << std::endl;

    // create a new OpenCL context for the device
    auto ctx = boost::compute::context(device);

    // create command queue for the device
    auto queue = boost::compute::command_queue(ctx, device);

    // Do sorting of some random vector using compute's sort
    // But first lets first create a fucking useless random VECTORRRRRRRRRRRR
    std::vector<int> vectorContainer;
    size_t sizeOfContainer = 10;
    vectorContainer.resize(sizeOfContainer);
    for (int i=0; i < sizeOfContainer; i++){
        vectorContainer[i] = i;
    }

    // randomly shuffle the vector
    std::random_shuffle(vectorContainer.begin(), vectorContainer.end());

    for (auto elem: vectorContainer){
        std::cout << elem << std::endl;
    }

    std::cout << "-------------------" << std::endl;
    
    // SORT THAT SHIT BABYYYYYYY
    boost::compute::sort(vectorContainer.begin(), vectorContainer.end(), queue);

    // WITHNESS THE GREAT FUCKING POWER OF PARALLEL SORTING ALGORITHM THAT COULDN'T BE DONE ON A PEASANT, USELESS PILE OF DOGSHI*... SILICON CALLED A 'CPU'
    for (auto elem: vectorContainer){
        std::cout << elem << std::endl;
    }
    
    return 0;
}
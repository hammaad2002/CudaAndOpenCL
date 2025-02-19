#include <iostream>
#include <boost/compute/core.hpp>
#include <random>
#include <vector>
#include <boost/compute/algorithm/sort.hpp>
#include <boost/compute/closure.hpp>

struct Box3D
{
  int label;
  float score;
  float x;
  float y;
  float z;
  float length;
  float width;
  float height;
  float yaw;
  float vel_x;
  float vel_y;

  // variance
  float x_variance;
  float y_variance;
  float z_variance;
  float length_variance;
  float width_variance;
  float height_variance;
  float yaw_variance;
  float vel_x_variance;
  float vel_y_variance;
};

// witnessed the following so called 'AdApT StRuCt' macro in the world's best documentation given below.
// https://www.boost.org/doc/libs/1_81_0/boost/compute/types/struct.hpp

BOOST_COMPUTE_ADAPT_STRUCT(Box3D, Box3D, (
    label,
    score,
    x,
    y,
    z,
    length,
    width,
    height,
    yaw,
    vel_x,
    vel_y,
    x_variance,
    y_variance,
    z_variance,
    length_variance,
    width_variance,
    height_variance,
    yaw_variance,
    vel_x_variance,
    vel_y_variance
));

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

    // HARDCORE SORTING
    // Creating useless variables because I can't come up with a better logic and its 12:51 A fucking M

    Box3D varOne, varTwo, varThree;
    varOne.score = 3.f;
    varTwo.score = 5.f;
    varThree.score = 1.f;

    std::vector<Box3D> boxContainerLol;
    boxContainerLol.push_back(varOne);
    boxContainerLol.push_back(varTwo);
    boxContainerLol.push_back(varThree);

    BOOST_COMPUTE_CLOSURE(bool, score_greater, (Box3D lb, Box3D rb), (varOne),
    {
        return lb.score > rb.score;
    });

    // So the sorting should be varTwo, varOne, varThree 
    boost::compute::sort(boxContainerLol.begin(), boxContainerLol.end(), score_greater, queue);

    for (auto elem: boxContainerLol){
        printf("Score: %f\n", elem.score);
    }

    return 0;
}
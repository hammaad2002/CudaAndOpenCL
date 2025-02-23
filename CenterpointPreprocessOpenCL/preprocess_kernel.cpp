#include "preprocess_kernel.hpp"
#include "boost/compute/core.hpp"

// std::string readKernelFile(const std::string& filename)
// {
// 	// Loading kernel
// 	std::ifstream kernel_file(filename);
// 	if (!kernel_file){
// 		throw std::runtime_error("Failed to open elementwise_multiply.cl");
// 	}

// 	std::string kernel_code((std::istreambuf_iterator<char>(kernel_file)), std::istreambuf_iterator<char>());
//     return kernel_code;
// }

void generateSweepPoints_launch(
    const compute::vector<float>& input_points,
    const uint& points_size,
    const int& input_point_step,
    const float& time_lag,
    const compute::vector<float>& transform_array, 
    const int& num_features,
    const compute::vector<float>& output_points,
    compute::command_queue& queue)
{
    compute::context context = queue.get_context();

    // std::string kernel_code = readKernelFile();
    compute::program program = compute::program::build_with_source_file("preprocess_kernel.cl", context);

    compute::kernel cl_kernel = program.create_kernel("generateSweepPoints_kernel");

    cl_kernel.set_arg(0, input_points.get_buffer());
    cl_kernel.set_arg(1, points_size);
    cl_kernel.set_arg(2, input_point_step);
    cl_kernel.set_arg(3, time_lag);
    cl_kernel.set_arg(4, transform_array.get_buffer());
    cl_kernel.set_arg(5, num_features);
    cl_kernel.set_arg(6, output_points.get_buffer());

    // Calculate execution parameters
	const size_t global_size = (points_size + 255) / 256 * 256;
	const size_t local_size = 256;

    assert(num_features == 4);

    // Execute the kernel
    queue.enqueue_nd_range_kernel(
        cl_kernel,
        (size_t)1, // work_dim
        nullptr, // global_work_offset
        &global_size, // global_work_size
        &local_size); // local_work_size
}
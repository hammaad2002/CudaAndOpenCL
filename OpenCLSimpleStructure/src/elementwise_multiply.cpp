#include "elementwise_multiply.hpp"

void elementwiseMultiplyLaunch(
	const compute::vector<float>& a,
	const compute::vector<float>& b,
	const compute::vector<float>& c,
	compute::command_queue& queue)
{
	if (a.size() != b.size() || a.size() != c.size()){
		throw std::invalid_argument("Input vectors must have equal sizes");
	}

	compute::context context = queue.get_context();

	// Loading kernel
	std::ifstream kernel_file("kernel/elementwise_multiply.cl");
	if (!kernel_file){
		throw std::runtime_error("Failed to open elementwise_multiply.cl");
	}

	std::string kernel_code((std::istreambuf_iterator<char>(kernel_file)), std::istreambuf_iterator<char>());

	// Building OpenCL program
	compute::program program = compute::program::build_with_source(kernel_code, context);

	// Create kernel
	compute::kernel elementwise_multiply(program, "elementwiseMultiply");

	// Set kernel arguments
	elementwise_multiply.set_arg(0, a.get_buffer());
	elementwise_multiply.set_arg(1, b.get_buffer());
	elementwise_multiply.set_arg(2, c.get_buffer());
	elementwise_multiply.set_arg(3, static_cast<cl_uint>(a.size()));

	// Calculate execution parameters
	const size_t global_size = (a.size() + 255) / 256 * 256;
	const size_t local_size = 256;

	// Execute kernel
	queue.enqueue_nd_range_kernel(
		elementwise_multiply, // Kernel
		1,                    // Dimensions
		nullptr,              // Global offset
		&global_size,         // Global size
		&local_size);         // Local size
}


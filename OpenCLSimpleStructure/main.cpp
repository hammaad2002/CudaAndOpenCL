#include "elementwise_multiply.hpp"
#include <boost/compute/utility/source.hpp>
#include <fstream>
#include <stdexcept>
#include <iostream>

int main(){
	// Get the device
	compute::device gpu = compute::system::default_device();

	// Create context on device
	compute::context ctx(gpu);

	// Create queue on device
	compute::command_queue queue(ctx, gpu);

	// Create text data
	std::vector<float> host_a = {1, 2, 3, 4, 5};
	std::vector<float> host_b = {1, 2, 3, 4, 5};
	std::vector<float> result(static_cast<float>(host_a.size()));

	// Create device buffer
	compute::vector<float> a(host_a.size(), ctx);
	compute::vector<float> b(host_b.size(), ctx);
	compute::vector<float> c(host_a.size(), ctx);

	// Copy data to buffer
	compute::copy(host_a.begin(), host_a.end(), a.begin(), queue);
	compute::copy(host_b.begin(), host_b.end(), b.begin(), queue);

	// Execute kernel
	elementwiseMultiplyLaunch(a, b, c, queue);

	// Copy results back to host device
	compute::copy(c.begin(), c.end(), result.begin(), queue);
	queue.finish();

	// Print the results
	for (auto i: result){
		std::cout << "Answer is: " << i << "\n";
	}
}

#pragma once
#define CL_TARGET_OPENCL_VERSION 300
#include <boost/compute/container/vector.hpp>
#include <boost/compute/command_queue.hpp>

namespace compute = boost::compute;

void elementwiseMultiplyLaunch(
	const compute::vector<float>& a,
	const compute::vector<float>& b,
	const compute::vector<float>& c,
	compute::command_queue& queue);

// void elementwiseAdditionLaunch(
// 	const compute::vector<float>& a,
// 	const compute::vector<float>& b,
// 	const compute::vector<float>& c,
// 	compute::command_queue& queue);
#pragma once
#include <boost/compute/container/vector.hpp>

namespace compute = boost::compute;

void generateSweepPoints_launch(
    const compute::vector<float>& input_points,
    const uint& points_size,
    const int& input_point_step,
    const float& time_lag,
    const compute::vector<float>& transform_array, 
    const int& num_features,
    const compute::vector<float>& output_points,
    const compute::command_queue& queue);
__kernel void generateSweepPoints_kernel(
    __global const float * input_points, 
    const uint points_size, 
    const int input_point_step, 
    const float time_lag,
    __global const float * transform_array, 
    const int num_features, 
    __global float * output_points)
{
    const uint point_idx = get_global_id(0);
    if (point_idx >= points_size) return;

    const float input_x = input_points[point_idx * input_point_step + 0];
    const float input_y = input_points[point_idx * input_point_step + 1];
    const float input_z = input_points[point_idx * input_point_step + 2];

    // transform_array is expected to be column-major
    output_points[point_idx * num_features + 0] = transform_array[0] * input_x +
                                                    transform_array[4] * input_y +
                                                    transform_array[8] * input_z + transform_array[12];
    output_points[point_idx * num_features + 1] = transform_array[1] * input_x +
                                                    transform_array[5] * input_y +
                                                    transform_array[9] * input_z + transform_array[13];
    output_points[point_idx * num_features + 2] = transform_array[2] * input_x +
                                                    transform_array[6] * input_y +
                                                    transform_array[10] * input_z + transform_array[14];
    output_points[point_idx * num_features + 3] = time_lag;
}
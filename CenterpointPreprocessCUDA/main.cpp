#include <stdlib.h>
#include <cstdio>
#include "cuda_runtime_api.h"
#include "new_header.hpp"

// int main() {

//     // kernel fake data
//     const int N = 20228;
//     const float *input_points = (float*)malloc(N * sizeof(float));
//     size_t points_size = N;
//     int input_point_step = 4 /sizeof(float);
//     float time_lag = 0.1;
//     const float *transform_array = (float*)malloc(N * sizeof(float));
//     int num_features = 4;
//     float *output_points = (float*)malloc(N * sizeof(float));

//     // float *h_a = (float*)malloc(N * sizeof(testfloat));
//     // float *h_b = (float*)malloc(N * sizeof(float));
//     // float *h_c = (float*)malloc(N * sizeof(float));
    
//     // // Initialize input vectors
//     // for (int i = 0; i < N; i++) {
//     //     h_a[i] = i;
//     //     h_b[i] = i * 2;
//     // }
    
//     cudaStream_t stream{nullptr};
//     cudaStreamCreate(&stream);

//     autoware::lidar_centerpoint::generateSweepPoints_launch(
//         input_points, points_size, input_point_step, time_lag,
//         transform_array, num_features, output_points, stream);

//     for (int i = 0; i < 10; i++) {
//         printf("Value %i is: %f\n", i, output_points[i]);
//     }

//     // Free the memory in the end
//     if (stream != nullptr) {
//         cudaStreamSynchronize(stream);
//         cudaStreamDestroy(stream);

//     return 0;
//     };
// }

int main() {
    // kernel fake data
    const int N = 20228;
    
    // Host memory allocation
    float *h_input_points = (float*)malloc(N * sizeof(float));
    float *h_transform_array = (float*)malloc(16 * sizeof(float));  // Only need 16 elements for 4x4 transform
    float *h_output_points = (float*)malloc(N * 4 * sizeof(float)); // 4 features per point
    
    // Initialize input data
    for (int i = 0; i < N; i++) {
        h_input_points[i] = i % 3;  // Just example values
    }
    for (int i = 0; i < 16; i++) {
        h_transform_array[i] = 1.0f;  // Initialize transform matrix
    }
    
    // Device memory allocation
    float *d_input_points, *d_output_points;
    cudaMalloc((void**)&d_input_points, N * sizeof(float));
    cudaMalloc((void**)&d_output_points, N * 4 * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_input_points, h_input_points, N * sizeof(float), cudaMemcpyHostToDevice);
    
    size_t points_size = N;
    int input_point_step = 4;  // Assuming 4 values per point
    float time_lag = 0.1;
    int num_features = 4;
    
    cudaStream_t stream{nullptr};
    cudaStreamCreate(&stream);

    autoware::lidar_centerpoint::generateSweepPoints_launch(
        d_input_points, points_size, input_point_step, time_lag,
        h_transform_array, num_features, d_output_points, stream);
        
    // Copy results back to host
    cudaMemcpy(h_output_points, d_output_points, N * 4 * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print first 10 results
    for (int i = 0; i < 10; i++) {
        printf("Value %i is: %f\n", i, h_output_points[i]);
    }

    // Cleanup
    cudaFree(d_input_points);
    cudaFree(d_output_points);
    free(h_input_points);
    free(h_transform_array);
    free(h_output_points);
    
    if (stream != nullptr) {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }

    return 0;
}
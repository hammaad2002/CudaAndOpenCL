#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_TARGET_OPENCL_VERSION 300

#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

#define MAX_SOURCE_SIZE (0x100000)

int main() {
    const int N = 1000;
    float *h_a = (float*)malloc(N * sizeof(float));
    float *h_b = (float*)malloc(N * sizeof(float));
    float *h_c = (float*)malloc(N * sizeof(float));
    
    // Initialize input vectors
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // Load kernel source code
    FILE *fp = fopen("vector_add.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel\n");
        exit(1);
    }
    char *source_str = (char*)malloc(MAX_SOURCE_SIZE);
    size_t source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);

    // Get platform and device information
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);

    // Create OpenCL context
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

    // Create command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

    // Create memory buffers on the device
    cl_mem d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, N * sizeof(float), NULL, &ret);
    cl_mem d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, N * sizeof(float), NULL, &ret);
    cl_mem d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, N * sizeof(float), NULL, &ret);

    // Copy data to device
    ret = clEnqueueWriteBuffer(command_queue, d_a, CL_TRUE, 0, N * sizeof(float), h_a, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, d_b, CL_TRUE, 0, N * sizeof(float), h_b, 0, NULL, NULL);

    // Create and build program
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source_str, (const size_t*)&source_size, &ret);
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    // Create kernel
    cl_kernel kernel = clCreateKernel(program, "addVectors", &ret);

    // Set kernel arguments
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&d_a);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&d_b);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&d_c);
    ret = clSetKernelArg(kernel, 3, sizeof(int), (void*)&N);

    // Execute kernel
    size_t global_work_size = ((N + 255) / 256) * 256;
    size_t local_work_size = 256;
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);

    // Read result back to host
    ret = clEnqueueReadBuffer(command_queue, d_c, CL_TRUE, 0, N * sizeof(float), h_c, 0, NULL, NULL);

    // Print first 10 results
    printf("First 10 results:\n");
    for (int i = 0; i < 10; i++) {
        printf("%f + %f = %f\n", h_a[i], h_b[i], h_c[i]);
    }

    // Cleanup
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(d_a);
    clReleaseMemObject(d_b);
    clReleaseMemObject(d_c);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);
    free(source_str);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}

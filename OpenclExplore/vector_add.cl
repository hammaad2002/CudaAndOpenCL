__kernel void addVectors(__global float *a, 
                        __global float *b, 
                        __global float *c, 
                        const int n) {
    int index = get_global_id(0);
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

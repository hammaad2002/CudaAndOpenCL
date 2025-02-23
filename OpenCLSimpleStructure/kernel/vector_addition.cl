__kernel void vectorAdd(
    __global const float* a,
    __global const float* b,
    __global float* c,
    const uint size)
{
    uint idx = get_global_id(0);
    if (idx > size){
        return;
    } else {
        c[i] = a[i] + b[i];
    }
}
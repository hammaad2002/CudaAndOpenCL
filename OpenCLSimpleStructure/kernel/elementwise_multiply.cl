__kernel void elementwiseMultiply(
	__global const float* a, 
	__global const float* b,
	__global float* c,
	const uint size)
{
	const uint idx = get_global_id(0);
	if (idx >= size){
		return;
	} else {
	c[idx] = a[idx] * b[idx];
	}
}
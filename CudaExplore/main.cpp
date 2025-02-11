#include <stdio.h>
#include <stdlib.h>

// Declare the wrapper function
extern "C" void launchAddVectors(float *a, float *b, float *c, int n);

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
    
    // Call the CUDA wrapper function
    launchAddVectors(h_a, h_b, h_c, N);
    
    // Print first 10 results
    printf("First 10 results:\n");
    for (int i = 0; i < 10; i++) {
        printf("%f + %f = %f\n", h_a[i], h_b[i], h_c[i]);
    }
    
    // Cleanup
    free(h_a);
    free(h_b);
    free(h_c);
    
    return 0;
}

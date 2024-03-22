#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

__global__ void gpuVecAdd(float *a, float *b, float *c, int len) {
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if (id < len) {    // bounds check
        c[id] = a[id] + b[id];
    }
}

void cpuVecAdd(float *a, float *b, float *c, int len) {
    for (int i = 0; i < len; i++) {
        c[i] = a[i] + b[i];
    }
}

int main(int argc, char const *argv[]) {
    const int len = 1<<28;
    const size_t size = len*sizeof(float);
    printf("Vector len: %d, size (bytes) %ld\n", len, (long)size);
    clock_t start, end;
    clock_t cpuTotal = 0;
    clock_t gpuTotal = 0;

    // vectors for host and GPU
    float *h_a, *h_b, *h_c, *g_a, *g_b, *g_c, *r_c;

    // allocate memory for host vectors
    start = clock();
    h_a = (float *)malloc(size);
    h_b = (float *)malloc(size);
    h_c = (float *)malloc(size);
    end = clock();
    printf("malloc: %f\n", (double)(end - start)/CLOCKS_PER_SEC);
    cpuTotal += end - start;
    r_c = (float *)malloc(size);    // result from GPU, don't time this

    char *warmup;
    cudaMalloc(&warmup, 0);    // empty alloc to warm up GPU

    // allocate memory for GPU vectors
    start = clock();
    cudaMalloc(&g_a, size);
    cudaMalloc(&g_b, size);
    cudaMalloc(&g_c, size);
    end = clock();
    printf("cudaMalloc: %f\n", (double)(end - start)/CLOCKS_PER_SEC);
    gpuTotal += end - start;

    // setup vectors
    for (int i = 0; i < len; i++) {
        h_a[i] = i;
        h_b[i] = 2*i;
    }

    // CPU version
    start = clock();
    cpuVecAdd(h_a, h_b, h_c, len);
    end = clock();
    printf("CPU VecAdd: %f\n", (double)(end - start)/CLOCKS_PER_SEC);
    cpuTotal += end - start;

    // copy vectors to GPU
    start = clock();
    cudaMemcpy(g_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(g_b, h_b, size, cudaMemcpyHostToDevice);
    end = clock();
    printf("copy host -> gpu: %f\n", (double)(end - start)/CLOCKS_PER_SEC);
    gpuTotal += end - start;

    int blockSize = 1024;
    int gridSize = (int)ceil((float)len/blockSize);
    
    start = clock();
    gpuVecAdd<<<gridSize, blockSize>>>(g_a, g_b, g_c, len);
    printf("Error Code %d\n", (int)cudaDeviceSynchronize());   // janky as hell
    end = clock();
    printf("GPU VecAdd: %f\n", (double)(end - start)/CLOCKS_PER_SEC);
    gpuTotal += end - start;

    // copy result back to host
    start = clock();
    cudaMemcpy(r_c, g_c, size, cudaMemcpyDeviceToHost);
    end = clock();
    printf("copy gpu -> host: %f\n", (double)(end - start)/CLOCKS_PER_SEC);
    gpuTotal += end - start;

    // print first 10 elements
    // for (int i = 0; i < 10; i++) {
    //     printf("CPU: %f, GPU: %f\n", h_c[i], r_c[i]);
    // }

    // free memory
    free(h_a);
    free(h_b);
    free(h_c);
    free(r_c);
    cudaFree(g_a);
    cudaFree(g_b);
    cudaFree(g_c);

    printf("CPU Total: %f\n", (double)cpuTotal/CLOCKS_PER_SEC);
    printf("GPU Total: %f\n", (double)gpuTotal/CLOCKS_PER_SEC);
    
    return 0;
}

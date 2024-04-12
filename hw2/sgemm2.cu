#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <windows.h>
#include <profileapi.h>

const int BLOCK_SIZE = 512;
enum VERBOSITY_LEVEL {LOW, MED, HIGH};
const VERBOSITY_LEVEL VERBOSITY = LOW;

#define CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
}

static inline long myCpuTimer() {
    LARGE_INTEGER ticks;
    if (!QueryPerformanceCounter(&ticks)) {
        printf("QueryPerformanceCounter failed\n");
    
    }
    return ticks.QuadPart;
}

static inline long perfFrequency() {
    LARGE_INTEGER freq;
    if (!QueryPerformanceFrequency(&freq)) {
        printf("QueryPerformanceFrequency failed\n");
    }
    return freq.QuadPart;
}

void printTimeDelta(const char *msg, long start, long end) {
    printf("%s: %f\n", msg, (double)(end - start)/perfFrequency());
}

void generateRandMatrix(int m, int n, float *A) {
    for (int i = 0; i < m*n; i++) {
        A[i] = (rand()%200 - 100)/100.0;
    }
}

void printMatrix(int m, int n, float *A) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f ", A[i*n + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int matEq(int m, int n, float *A, float *B) {
    int rv = 1;
    for (int i = 0; i < m*n; i++) {
        if (abs((A[i] / B[i]) - 1.0) > 0.01 && abs((A[i] - B[i]) > 0.0001)) {
            printf("A[%d] = %f, B[%d] = %f\n", i, A[i], i, B[i]);
            // Print integer representation of floats bits for debugging
            int* xi = (int*)((void*)&A[i]);
            int* yi = (int*)((void*)&B[i]);
            printf("A[%d] = %d, B[%d] = %d\n", i, *xi, i, *yi);
            rv = 0;
        }
    }
    return rv;
}

void warmupGPU() {
    // empty alloc to warm up GPU
    char *warmup;
    CHECK(cudaMalloc(&warmup, 0));    
    CHECK(cudaFree(warmup));
}

void allocDeviceMatrices(int m, int k, int n, float **A_d, float **B_d, float **C_d) {
    long t0 = myCpuTimer();
    size_t sizeA = m*k*sizeof(float);
    size_t sizeB = k*n*sizeof(float);
    size_t sizeC = m*n*sizeof(float);
    CHECK(cudaMalloc(A_d, sizeA));
    CHECK(cudaMalloc(B_d, sizeB));
    CHECK(cudaMalloc(C_d, sizeC));
    printTimeDelta("\tCuda Malloc", t0, myCpuTimer());
}

void freeDeviceMatrices(float *A_d, float *B_d, float *C_d) {
    long t0 = myCpuTimer();
    CHECK(cudaFree(A_d));
    CHECK(cudaFree(B_d));
    CHECK(cudaFree(C_d));
    printTimeDelta("\tCuda Free", t0, myCpuTimer());
}

void copyToDevice(int m, int k, int n, const float *A_h, const float *B_h, float *A_d, float *B_d) {
    long t0 = myCpuTimer();
    size_t sizeA = m*k*sizeof(float);
    size_t sizeB = k*n*sizeof(float);
    CHECK(cudaMemcpy(A_d, A_h, sizeA, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(B_d, B_h, sizeB, cudaMemcpyHostToDevice));
    printTimeDelta("\tCuda Memcpy HostToDevice", t0, myCpuTimer());
}

void copyToHost(int m, int n, const float *C_d, float *C_h) {
    long t0 = myCpuTimer();
    size_t sizeC = m*n*sizeof(float);
    CHECK(cudaMemcpy(C_h, C_d, sizeC, cudaMemcpyDeviceToHost));
    printTimeDelta("\tCuda Memcpy DeviceToHost", t0, myCpuTimer());
}

void basicSgemm_h(int m, int k, int n, const float *A_h, const float *B_h, float* C_h){
    printf("Basic (Host) Matrix Multiplication \n");
    long t0 = myCpuTimer();
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C_h[i*n + j] = 0;
            for (int l = 0; l < k; l++) {
                C_h[i*n + j] += A_h[i*k + l] * B_h[l*n + j];
            }
        }
    }
    printTimeDelta("\tHost Execution", t0, myCpuTimer());
}

__global__ void matrixMulKernel_1thread1element(int m, int k, int n, const float *A_d, const float *B_d, float* C_d) {
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    int row = floor((double)id/n);  // cast to double to call device function
    int col = id%n;
    if (row < m && col < n) {
        C_d[row*n + col] = 0;
        for (int i = 0; i < k; i++) {
            C_d[row*n + col] += A_d[row*k + i] * B_d[i*n + col];
        }
    }
}


void basicSgemm_d_1thread1element(int m, int k, int n, const float *A_h, const float *B_h, float* C_h) {
    printf("Device Matrix Multiplication: 1 Thread-1 Element\n");
    float *A_d, *B_d, *C_d;
    allocDeviceMatrices(m, k, n, &A_d, &B_d, &C_d);
    copyToDevice(m, k, n, A_h, B_h, A_d, B_d);

    long t0 = myCpuTimer();
    int gridSize = (int)ceil((float)(m*n)/BLOCK_SIZE);
    matrixMulKernel_1thread1element<<<gridSize, BLOCK_SIZE>>>(m, k, n, A_d, B_d, C_d);
    CHECK(cudaDeviceSynchronize());
    printTimeDelta("\tKernel Execution", t0, myCpuTimer());

    copyToHost(m, n, C_d, C_h);
    freeDeviceMatrices(A_d, B_d, C_d);
}

int main(int argc, char const *argv[]) {
    // arg parsing
    if (argc != 4) {
        printf("Usage: %s <m> <k> <n>\n", argv[0]);
        return 1;
    }
    int m = atoi(argv[1]);
    int k = atoi(argv[2]);
    int n = atoi(argv[3]);
    
    // create A and B
    float *A_h = (float*)malloc(m*k*sizeof(float));
    float *B_h = (float*)malloc(k*n*sizeof(float));
    generateRandMatrix(m, k, A_h);
    generateRandMatrix(k, n, B_h);

    // create output (C) matrices
    float *C_h = (float*)malloc(m*n*sizeof(float));
    float *C_h_1t1e = (float*)malloc(m*n*sizeof(float));

    warmupGPU();

    // do matrix multiplication
    basicSgemm_h(m, k, n, A_h, B_h, C_h);
    basicSgemm_d_1thread1element(m, k, n, A_h, B_h, C_h_1t1e);

    if (VERBOSITY >= HIGH) {
        printMatrix(m, k, A_h);
        printMatrix(k, n, B_h);
        printMatrix(m, n, C_h);
        printMatrix(m, n, C_h_1t1e);
    }

    if(matEq(m, n, C_h, C_h_1t1e)){
        printf("1 Thread 1 Element: Correct\n");
    } else {
        printf("1 Thread 1 Element: Incorrect\n");
    }

    free(A_h);
    free(B_h);
    free(C_h);
    free(C_h_1t1e);

    return 0;
}

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>


#define CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
}

static inline long myCpuTimer() {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == -1) {
        printf("clock_gettime failed\n");
        return -1;
    }
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

void printTimeDelta(const char *msg, long start, long end) {
    printf("%s: %f\n", msg, (double)(end - start)/1e9);
}


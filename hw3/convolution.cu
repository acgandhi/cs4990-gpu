#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <opencv2/opencv.hpp>

#define FILTER_RADIUS 3
#define FILTER_SIZE 5
#define BLOCK_SIZE 32

#define CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
}

const float F_h[FILTER_SIZE][FILTER_SIZE] = {
    {1.0f/25, 1.0f/25, 1.0f/25, 1.0f/25, 1.0f/25},
    {1.0f/25, 1.0f/25, 1.0f/25, 1.0f/25, 1.0f/25},
    {1.0f/25, 1.0f/25, 1.0f/25, 1.0f/25, 1.0f/25},
    {1.0f/25, 1.0f/25, 1.0f/25, 1.0f/25, 1.0f/25},
    {1.0f/25, 1.0f/25, 1.0f/25, 1.0f/25, 1.0f/25}
};

__constant__ float F_d[FILTER_SIZE][FILTER_SIZE];

static inline long myCpuTimer() {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == -1) {
        printf("clock_gettime failed\n");
        return -1;
    }
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

static inline void printTimeDelta(const char *msg, long start, long end) {
    printf("%s: %f\n", msg, (double)(end - start)/1e9);
}

void blurImage_h(cv::Mat& Pout_Mat_h, const cv::Mat& Pin_Mat_h, unsigned int nRows, unsigned int nCols) {
    for (unsigned int i = 0; i < nRows; ++i) {
        for (unsigned int j = 0; j < nCols; ++j) {
            float sum = 0.0f;
            for (unsigned int k = 0; k < FILTER_SIZE; ++k) {
                for (unsigned int l = 0; l < FILTER_SIZE; ++l) {
                    // index of the pixel to be convolved
                    // (FILTER_RADIUS - 1) is really floor(FILTER_SIZE/2)
                    int ix = i + k - (FILTER_RADIUS - 1);
                    int iy = j + l - (FILTER_RADIUS - 1);
                    if (ix >= 0 && ix < nRows && iy >= 0 && iy < nCols) {
                        // in bounds
                        sum += Pin_Mat_h.at<uchar>(ix, iy) * F_h[k][l];
                    } else {
                        // out of bounds: zero padding
                        sum += 0.0f;
                    }
                }
            }
            Pout_Mat_h.at<uchar>(i, j) = sum;
        }
    }
}

__global__ void blurImage_tiled_Kernel(unsigned char *Pout, unsigned char *Pin, unsigned int width, unsigned int height) {
    __shared__ unsigned char tile[BLOCK_SIZE + FILTER_SIZE - 1][BLOCK_SIZE + FILTER_SIZE - 1];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row_o = blockIdx.y * blockDim.y + ty;
    int col_o = blockIdx.x * blockDim.x + tx;
    int row_i = row_o - FILTER_RADIUS;
    int col_i = col_o - FILTER_RADIUS;

    if (row_i >= 0 && row_i < height && col_i >= 0 && col_i < width) {
        tile[ty][tx] = Pin[row_i * width + col_i];
    } else {
        tile[ty][tx] = 0;
    }

    __syncthreads();

    if (row_o < height && col_o < width) {
        float sum = 0.0f;
        for (int i = 0; i < FILTER_SIZE; ++i) {
            for (int j = 0; j < FILTER_SIZE; ++j) {
                sum += tile[ty + i][tx + j] * F_d[i][j];
            }
        }
        Pout[row_o * width + col_o] = sum;
    }
}


__global__ void blurImage_Kernel(unsigned char *Pout, unsigned char *Pin, unsigned int width, unsigned int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < height && col < width) {
        float sum = 0.0f;
        for (int i = 0; i < FILTER_SIZE; ++i) {
            for (int j = 0; j < FILTER_SIZE; ++j) {
                int row_i = row - FILTER_RADIUS + i;
                int col_i = col - FILTER_RADIUS + j;
                if (row_i >= 0 && row_i < height && col_i >= 0 && col_i < width) {
                    sum += Pin[row_i * width + col_i] * F_d[i][j];
                } else {
                    sum += 0.0f;
                }
            }
        }
        Pout[row * width + col] = sum;
    }

}

void blurImage_d(cv::Mat Pout_Mat_h, cv::Mat Pin_Mat_h, unsigned int nRows, unsigned int nCols) {
    unsigned char *Pout_d, *Pin_d;
    long start = myCpuTimer();
    CHECK(cudaMalloc(&Pout_d, nRows * nCols * sizeof(unsigned char)));
    CHECK(cudaMalloc(&Pin_d, nRows * nCols * sizeof(unsigned char)));
    long end = myCpuTimer();
    printTimeDelta("\tCUDA malloc", start, end);

    CHECK(cudaMemcpy(Pin_d, Pin_Mat_h.data, nRows * nCols * sizeof(unsigned char), cudaMemcpyHostToDevice));

    dim3 dimBlock(32, 32);
    dim3 dimGrid((nCols + dimBlock.x - 1) / dimBlock.x, (nRows + dimBlock.y - 1) / dimBlock.y);

    long start = myCpuTimer();
    blurImage_Kernel<<<dimGrid, dimBlock>>>(Pout_d, Pin_d, nCols, nRows);
    cudaDeviceSynchronize();
    long end = myCpuTimer();
    printTimeDelta("GPU Time", start, end);

    CHECK(cudaMemcpy(Pout_Mat_h.data, Pout_d, nRows * nCols * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    CHECK(cudaFree(Pout_d));
    CHECK(cudaFree(Pin_d));
}

bool verify(cv::Mat answer1, cv::Mat answer2, unsigned int nRows, unsigned int nCols) {
    const float relativeTolerance = 1e-2;
    for (unsigned int i = 0; i < nRows; ++i) {
        for (unsigned int j = 0; j < nCols; ++j) {
            float relativeError = abs((float) answer1.at<uchar>(i, j) - (float) answer2.at<uchar>(i, j)) / 255.0;
            if (relativeError > relativeTolerance) {
                printf("Mismatch at (%d, %d): %d vs %d error: %f\n", i, j, answer1.at<uchar>(i, j), answer2.at<uchar>(i, j), relativeError);
                return false;
            }
        }
    }
    return true;
}

int main(int argc, char **argv) {
    // user args and image read
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <image>\n", argv[0]);
        exit(1);
    }

    cv::Mat grayImg = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (grayImg.empty()) {
        fprintf(stderr, "Could not open or find the image\n");
        return 1;
    }

    // Mat allocation
    unsigned int nRows = grayImg.rows;
    unsigned int nCols = grayImg.cols;

    cv::Mat blurred_opencv(nRows, nCols, CV_8UC1, cv::Scalar(0));
    cv::Mat blurred_cpu(nRows, nCols, CV_8UC1, cv::Scalar(0));
    cv::Mat blurred_gpu(nRows, nCols, CV_8UC1, cv::Scalar(0));
    cv::Mat blurred_tiled_gpu(nRows, nCols, CV_8UC1, cv::Scalar(0));

    // copy filter to device constant memory
    CHECK(cudaMemcpyToSymbol(F_d, F_h, FILTER_SIZE * FILTER_SIZE * sizeof(float)));

    // run kernels
    long start = myCpuTimer();
    cv::blur(grayImg, blurred_opencv, cv::Size(FILTER_SIZE, FILTER_SIZE), cv::Point(-1, -1), cv::BORDER_CONSTANT);
    long end = myCpuTimer();
    printTimeDelta("OpenCV Time", start, end);

    start = myCpuTimer();
    blurImage_h(blurred_cpu, grayImg, nRows, nCols);
    end = myCpuTimer();
    printTimeDelta("CPU Time", start, end);

    start = myCpuTimer();
    blurImage_d(blurred_gpu, grayImg, nRows, nCols);
    end = myCpuTimer();
    printTimeDelta("GPU Time", start, end);    

    // write outputs
    bool check = cv::imwrite("bluredImg_opencv.jpg", blurred_opencv);
    if (!check) {
        fprintf(stderr, "Could not write image\n");
        return 1;
    }
    check = cv::imwrite("bluredImg_cpu.jpg", blurred_cpu);
    if (!check) {
        fprintf(stderr, "Could not write image\n");
        return 1;
    }
    check = cv::imwrite("bluredImg_gpu.jpg", blurred_gpu);
    if (!check) {
        fprintf(stderr, "Could not write image\n");
        return 1;
    }
    check = cv::imwrite("bluredImg_tiled_gpu.jpg", blurred_tiled_gpu);
    if (!check) {
        fprintf(stderr, "Could not write image\n");
        return 1;
    }

    // verify results
    // sanity check: compare opencv to itself
    bool result = verify(blurred_opencv, blurred_opencv, nRows, nCols);
    printf("OpenCV vs OpenCV: %s\t\n", result ? "PASS" : "FAIL");
    result = verify(blurred_opencv, blurred_cpu, nRows, nCols);
    printf("OpenCV vs CPU: %s\t\n", result ? "PASS" : "FAIL");
    result = verify(blurred_opencv, blurred_gpu, nRows, nCols);
    printf("OpenCV vs GPU: %s\t\n", result ? "PASS" : "FAIL");
    result = verify(blurred_opencv, blurred_tiled_gpu, nRows, nCols);
    printf("OpenCV vs Tiled GPU: %s\t\n", result ? "PASS" : "FAIL");


    return 0;
}


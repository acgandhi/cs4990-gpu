#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <opencv2/opencv.hpp>

#define FILTER_RADIUS 2
#define FILTER_SIZE 5
#define BLOCK_SIZE 32
#define IN_TILE_DIM (BLOCK_SIZE)
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2*(FILTER_RADIUS))

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
                    // FILTER_RADIUS is really floor(FILTER_SIZE/2)
                    int ix = i + k - (FILTER_RADIUS);
                    int iy = j + l - (FILTER_RADIUS);
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



__global__ void blurImage_Kernel(unsigned char *Pout, unsigned char *Pin, unsigned int width, unsigned int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < height && col < width) {
        float sum = 0.0f;
        for (int i = 0; i < FILTER_SIZE; ++i) {
            for (int j = 0; j < FILTER_SIZE; ++j) {
                int row_i = row + i - (FILTER_RADIUS);
                int col_i = col + j - (FILTER_RADIUS);
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

__global__ void blurImage_tiled_Kernel(unsigned char *Pout, unsigned char *Pin, unsigned int width, unsigned int height) {
    int row = blockIdx.y * IN_TILE_DIM + threadIdx.y;
    int col = blockIdx.x * IN_TILE_DIM + threadIdx.x;   
    
    // load input tile
    __shared__ unsigned char N_s[IN_TILE_DIM][IN_TILE_DIM];
    if(row < height && col < width) {
        N_s[threadIdx.y][threadIdx.x] = Pin[row*width + col];
    } else {
        N_s[threadIdx.y][threadIdx.x] = 0;
    }
    __syncthreads();

    
    // don't compute for out of bounds threads
    if (col < width && row < height) {
        float sum = 0.0f;
        for (int i = -FILTER_RADIUS; i <= FILTER_RADIUS; ++i) {
            for (int j = -FILTER_RADIUS; j <= FILTER_RADIUS; ++j) {
                // inside the tile
                if ((int) threadIdx.y + i >= 0 && (int) threadIdx.y + i < IN_TILE_DIM && (int) threadIdx.x + j >= 0 && (int) threadIdx.x + j < IN_TILE_DIM)
                    sum += N_s[(int) threadIdx.y + i][(int) threadIdx.x + j] * F_d[i + FILTER_RADIUS][j + FILTER_RADIUS];
                // inside image, outside tile
                else if (row + i >= 0 && row + i < height && col + j >= 0 && col + j < width){
                    sum += Pin[(row + i) * width + col + j] * F_d[i + FILTER_RADIUS][j + FILTER_RADIUS];    // 0.0 * F_d[i + FILTER_RADIUS][j + FILTER_RADIUS]
                } else {
                    sum += 0.0f;
                }
            }
        }
        Pout[row*width + col] = sum;
    }
}


void blurImage_d(cv::Mat Pout_Mat_h, cv::Mat Pin_Mat_h, unsigned int nRows, unsigned int nCols) {
    unsigned char *Pout_d, *Pin_d;
    long start = myCpuTimer();
    CHECK(cudaMalloc(&Pout_d, nRows * nCols * sizeof(unsigned char)));
    CHECK(cudaMalloc(&Pin_d, nRows * nCols * sizeof(unsigned char)));
    long end = myCpuTimer();
    printTimeDelta("\tCUDA malloc", start, end);

    start = myCpuTimer();
    CHECK(cudaMemcpy(Pin_d, Pin_Mat_h.data, nRows * nCols * sizeof(unsigned char), cudaMemcpyHostToDevice));
    end = myCpuTimer();
    printTimeDelta("\tCUDA memcpy HtoD", start, end);

    dim3 dimBlock(32, 32);
    dim3 dimGrid((nCols + dimBlock.x - 1) / dimBlock.x, (nRows + dimBlock.y - 1) / dimBlock.y);

    start = myCpuTimer();
    blurImage_Kernel<<<dimGrid, dimBlock>>>(Pout_d, Pin_d, nCols, nRows);
    CHECK(cudaDeviceSynchronize());
    end = myCpuTimer();
    printTimeDelta("\tGPU Compute Time", start, end);

    start = myCpuTimer();
    CHECK(cudaMemcpy(Pout_Mat_h.data, Pout_d, nRows * nCols * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    end = myCpuTimer();
    printTimeDelta("\tCUDA memcpy DtoH", start, end);

    CHECK(cudaFree(Pout_d));
    CHECK(cudaFree(Pin_d));
}

void blurImage_tiled_d(cv::Mat Pout_Mat_h, cv::Mat Pin_Mat_h, unsigned int nRows, unsigned int nCols) {
    unsigned char *Pout_d, *Pin_d;
    long start = myCpuTimer();
    CHECK(cudaMalloc(&Pout_d, nRows * nCols * sizeof(unsigned char)));
    CHECK(cudaMalloc(&Pin_d, nRows * nCols * sizeof(unsigned char)));
    long end = myCpuTimer();
    printTimeDelta("\tCUDA malloc", start, end);

    start = myCpuTimer();
    CHECK(cudaMemcpy(Pin_d, Pin_Mat_h.data, nRows * nCols * sizeof(unsigned char), cudaMemcpyHostToDevice));
    end = myCpuTimer();
    printTimeDelta("\tCUDA memcpy HtoD", start, end);

    dim3 dimBlock(IN_TILE_DIM, IN_TILE_DIM);
    dim3 dimGrid((nCols + dimBlock.x - 1) / dimBlock.x, (nRows + dimBlock.y - 1) / dimBlock.y);

    start = myCpuTimer();
    size_t shMemSize = (IN_TILE_DIM * sizeof(unsigned char)) * (IN_TILE_DIM * sizeof(unsigned char));
    blurImage_tiled_Kernel<<<dimGrid, dimBlock, shMemSize>>>(Pout_d, Pin_d, nCols, nRows);
    CHECK(cudaDeviceSynchronize());
    end = myCpuTimer();
    printTimeDelta("\tGPU Compute Time", start, end);

    start = myCpuTimer();
    CHECK(cudaMemcpy(Pout_Mat_h.data, Pout_d, nRows * nCols * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    end = myCpuTimer();
    printTimeDelta("\tCUDA memcpy DtoH", start, end);

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
    printf("OpenCV\n");
    long start = myCpuTimer();
    cv::blur(grayImg, blurred_opencv, cv::Size(FILTER_SIZE, FILTER_SIZE), cv::Point(-1, -1), cv::BORDER_CONSTANT);
    long end = myCpuTimer();
    printTimeDelta("\tTotal Time", start, end);

    printf("CPU\n");
    start = myCpuTimer();
    blurImage_h(blurred_cpu, grayImg, nRows, nCols);
    end = myCpuTimer();
    printTimeDelta("\tTotal Time", start, end);

    printf("GPU\n");
    start = myCpuTimer();
    blurImage_d(blurred_gpu, grayImg, nRows, nCols);
    end = myCpuTimer();
    printTimeDelta("\tTotal Time", start, end);

    printf("Tiled GPU\n");
    start = myCpuTimer();
    blurImage_tiled_d(blurred_tiled_gpu, grayImg, nRows, nCols);
    end = myCpuTimer();
    printTimeDelta("\tTotal Time", start, end);

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


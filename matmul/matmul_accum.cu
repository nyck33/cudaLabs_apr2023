#include <stdio.h>
#include <cuda.h>

#define TILE_WIDTH 4 // Using a 2x2 tile for simplicity
#define WIDTH 16     // Matrix width (for a 4x4 matrix)

__global__ void matrixMulKernel(float* A, float* B, float* C, float* sumArray, int width) {
    __shared__ float tile_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float sum = 0.0;
    int sumArrayIndex = (row * width + col) * (width / TILE_WIDTH);

    for (int m = 0; m < ceil(width / (float)TILE_WIDTH); ++m) {
        if (m * TILE_WIDTH + tx < width && row < width)
            tile_A[ty][tx] = A[row * width + m * TILE_WIDTH + tx];
        else
            tile_A[ty][tx] = 0.0;

        if (m * TILE_WIDTH + ty < width && col < width)
            tile_B[ty][tx] = B[(m * TILE_WIDTH + ty) * width + col];
        else
            tile_B[ty][tx] = 0.0;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
            sum += tile_A[ty][k] * tile_B[k][tx];

        // Store cumulative sum after each tile
        sumArray[sumArrayIndex + m] = sum;

        __syncthreads();
    }

    if (row < width && col < width)
        C[row * width + col] = sum;
}

// Host code to set up and run the kernel
int main() {

    int sharedMemPerBlock;
    cudaDeviceGetAttribute(&sharedMemPerBlock, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Shared Memory per block: %d bytes\n", sharedMemPerBlock);

    float A[WIDTH * WIDTH]; // Input matrix A
    float B[WIDTH * WIDTH]; // Input matrix B
    float C[WIDTH * WIDTH]; // Output matrix C

    // Initialize matrices A and B with some values
    for (int i = 0; i < WIDTH * WIDTH; i++) {
        A[i] = 1.0;
        B[i] = 1.0;
    }

    float *dev_A, *dev_B, *dev_C, *dev_sumArray;
    cudaMalloc((void **)&dev_A, WIDTH * WIDTH * sizeof(float));
    cudaMalloc((void **)&dev_B, WIDTH * WIDTH * sizeof(float));
    cudaMalloc((void **)&dev_C, WIDTH * WIDTH * sizeof(float));

    // Allocate memory for storing cumulative sums
    cudaMalloc((void **)&dev_sumArray, WIDTH * WIDTH * (WIDTH / TILE_WIDTH) * sizeof(float));
    
    // Copy matrices A and B to device memory
    cudaMemcpy(dev_A, A, WIDTH * WIDTH * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, WIDTH * WIDTH * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimGrid(WIDTH / TILE_WIDTH, WIDTH / TILE_WIDTH, 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    matrixMulKernel<<<dimGrid, dimBlock>>>(dev_A, dev_B, dev_C, dev_sumArray, WIDTH);

    // Copy the result matrix C back to the host
    cudaMemcpy(C, dev_C, WIDTH * WIDTH * sizeof(float), cudaMemcpyDeviceToHost);

    float sumArray[WIDTH * WIDTH * (WIDTH / TILE_WIDTH)];
    cudaMemcpy(sumArray, dev_sumArray, WIDTH * WIDTH * (WIDTH / TILE_WIDTH) * sizeof(float), cudaMemcpyDeviceToHost);

    // Print cumulative sums
    for (int i = 0; i < WIDTH; i++) {
        for (int j = 0; j < WIDTH; j++) {
            printf("Cumulative sums for C[%d][%d]: ", i, j);
            for (int m = 0; m < (WIDTH / TILE_WIDTH); m++) {
                printf("%f ", sumArray[(i * WIDTH + j) * (WIDTH / TILE_WIDTH) + m]);
            }
            printf("\n");
        }
    }

    // Free device memory
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);
    cudaFree(dev_sumArray);

    return 0;
}

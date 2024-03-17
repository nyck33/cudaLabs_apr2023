#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#define TILE_WIDTH 2
#define WIDTH 8  // Example for larger WIDTH, but the approach works for any size
#define BLOCKS_PER_GRID_DIM 2  // Fixed grid dimension
#define THREADS_PER_BLOCK_DIM 2  // Fixed block dimension

__global__ void matmul(float *Md, float *Nd, float *Pd, int width, int startX, int startY) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x + startX;
    int by = blockIdx.y + startY;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    float Pvalue = 0;

    for (int m = 0; m < width/TILE_WIDTH; ++m) {
        Mds[ty][tx] = Md[Row*width + m*TILE_WIDTH + tx];
        Nds[ty][tx] = Nd[(m*TILE_WIDTH + ty)*width + Col];
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }
    if (Row < width && Col < width) {
        Pd[Row*width + Col] += Pvalue;
    }
}

void callKernel(float *Md, float *Nd, float *Pd, int width) {
    dim3 dimBlock(THREADS_PER_BLOCK_DIM, THREADS_PER_BLOCK_DIM);
    dim3 dimGrid(BLOCKS_PER_GRID_DIM, BLOCKS_PER_GRID_DIM);
    int gridWidth = BLOCKS_PER_GRID_DIM * TILE_WIDTH;

    for (int startY = 0; startY < width; startY += gridWidth) {
        for (int startX = 0; startX < width; startX += gridWidth) {
            matmul<<<dimGrid, dimBlock>>>(Md, Nd, Pd, width, startX / TILE_WIDTH, startY / TILE_WIDTH);
            cudaDeviceSynchronize();  // Ensure the kernel completes before the next launch
        }
    }
}

// The rest of the code (matmulOnDevice, main, etc.) remains the same.
//write a function to allocate memory on the device and call the kernel
void matmulOnDevice(float *M, float *N, float *P, int width) {
    float *Md, *Nd, *Pd;
    int size = width * width * sizeof(float);
    cudaMalloc((void**)&Md, size);
    cudaMalloc((void**)&Nd, size);
    cudaMalloc((void**)&Pd, size);
    cudaMemcpy(Md, M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(Nd, N, size, cudaMemcpyHostToDevice);
    callKernel(Md, Nd, Pd, width);
    cudaMemcpy(P, Pd, size, cudaMemcpyDeviceToHost);
    cudaFree(Md);
    cudaFree(Nd);
    cudaFree(Pd);
}

int main() {
    float M[WIDTH][WIDTH], N[WIDTH][WIDTH], P[WIDTH][WIDTH];
    for (int i = 0; i < WIDTH; ++i) {
        for (int j = 0; j < WIDTH; ++j) {
            M[i][j] = i;
            N[i][j] = j;
        }
    }
    //print the input matrices
    printf("M:\n");
    for (int i = 0; i < WIDTH; ++i) {
        for (int j = 0; j < WIDTH; ++j) {
            printf("%f ", M[i][j]);
        }
        printf("\n");
    }
    printf("N:\n");
    for (int i = 0; i < WIDTH; ++i) {
        for (int j = 0; j < WIDTH; ++j) {
            printf("%f ", N[i][j]);
        }
        printf("\n");
    }


    matmulOnDevice((float *)M, (float *)N, (float *)P, WIDTH);
    for (int i = 0; i < WIDTH; ++i) {
        for (int j = 0; j < WIDTH; ++j) {
            printf("%f ", P[i][j]);
        }
        printf("\n");
    }
    return 0;
}
//output


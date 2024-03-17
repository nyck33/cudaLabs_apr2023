//import cuda
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
//#include <device_functions.h>
#include <stdio.h>

#define TILE_WIDTH 2 
# define WIDTH 4

//write cuda kernel that loads the input matrix into shared memory, use Md, Nd, and Pd to represent the input matrices, and Mds, Nds to represent the shared memory
__global__ void matmul_(float *Md, float *Nd, float *Pd, int width) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x;
    int by = blockIdx.y;
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
    Pd[Row*width + Col] = Pvalue;
}

__global__ void matmul(float *Md, float *Nd, float *Pd, int width) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    float Pvalue = 0;

    
    for (int m = 0; m < width/TILE_WIDTH; ++m) {
        Mds[ty][tx] = Md[Row*width + (m*TILE_WIDTH + tx)];
        Nds[ty][tx] = Nd[(m*TILE_WIDTH + ty)*width + Col];
        __syncthreads(); // Synchronize again before proceeding to computation

        //within a phase you have two multiplications that need to be cumulated
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads(); // Wait for all threads in block to compute their partial products
    }
    
    //no grid stride loop
    /*
    Mds[ty][tx] = Md[Row*width + (0*TILE_WIDTH + tx)];
    Nds[ty][tx] = Nd[(0*TILE_WIDTH + ty)*width + Col];
    __syncthreads(); // Synchronize again before proceeding to computation

    //within a phase you have two multiplications that need to be cumulated
    for (int k = 0; k < TILE_WIDTH; ++k) {
        Pvalue += Mds[ty][k] * Nds[k][tx];
    }
    __syncthreads(); // Wait for all threads in block to compute their partial products
    */

    Pd[Row*width + Col] = Pvalue;

    
}

//write a function to call the kernel
void callKernel(float *Md, float *Nd, float *Pd, int width) {
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(width/TILE_WIDTH, width/TILE_WIDTH);
    //print the grid and block dimensions with endlines before and after the output
    printf("Grid dimensions: (%d, %d, %d)\n", dimGrid.x, dimGrid.y, dimGrid.z);
    printf("Block dimensions: (%d, %d, %d)\n", dimBlock.x, dimBlock.y, dimBlock.z);
    matmul<<<dimGrid, dimBlock>>>(Md, Nd, Pd, width);
}

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


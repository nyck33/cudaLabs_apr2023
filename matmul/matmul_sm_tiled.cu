//import cuda
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#define TILE_WIDTH 4
# define WIDTH 12

//write cuda kernel that loads the input matrix into shared memory, use Md, Nd, and Pd to represent the input matrices, and Mds, Nds to represent the shared memory
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

//write a function to call the kernel
void callKernel(float *Md, float *Nd, float *Pd, int width) {
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(width/TILE_WIDTH, width/TILE_WIDTH);
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

